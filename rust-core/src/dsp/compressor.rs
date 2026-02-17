//! Downward compressor with IIR envelope detection
//!
//! Reduces dynamic range by attenuating signals above the threshold.
//! Uses the same IIR envelope follower pattern as the noise gate for
//! consistent, low-latency level detection.

/// Downward compressor with soft-knee gain reduction
pub struct Compressor {
    /// Threshold in dB - compression starts above this level
    threshold_db: f64,

    /// Compression ratio (e.g., 4.0 = 4:1 ratio)
    ratio: f64,

    /// Attack time constant (exponential smoothing coefficient)
    attack_coeff: f64,

    /// Release time constant (exponential smoothing coefficient)
    release_coeff: f64,

    /// Makeup gain in dB to compensate for gain reduction
    makeup_gain_db: f64,

    /// Makeup gain as linear multiplier (cached)
    makeup_gain_linear: f64,

    /// Knee width in dB for soft-knee transition
    knee_db: f64,

    /// Current envelope level in dB
    envelope_db: f64,

    /// IIR envelope squared (for RMS approximation)
    envelope_squared: f64,

    /// RMS smoothing coefficient (single-pole IIR)
    rms_coeff: f64,

    /// Current gain reduction in dB (for metering)
    current_gain_reduction_db: f64,

    /// Sample rate
    sample_rate: f64,

    /// Whether compressor is enabled
    enabled: bool,

    /// Whether adaptive release is enabled
    adaptive_release: bool,

    /// Base release time in milliseconds (user-controlled)
    base_release_ms: f64,

    /// Current release time in milliseconds (adaptive value)
    current_release_ms: f64,

    /// Overage timer (samples above threshold)
    overage_timer: f64,

    /// Target release time (for smoothing)
    target_release_ms: f64,

    /// Release smoothing coefficient (100ms hysteresis)
    release_smoothing_coeff: f64,

    /// Loudness meter for auto makeup gain
    loudness_meter: Option<crate::dsp::loudness::LoudnessMeter>,

    /// Auto makeup gain enabled
    auto_makeup_enabled: bool,

    /// Target LUFS for auto makeup gain
    target_lufs: f64,

    /// Smoothed makeup gain (for transitions)
    smoothed_makeup_gain: f64,

    /// Makeup gain smoothing coefficient (200ms time constant)
    makeup_smoothing_coeff: f64,

    /// Current measured loudness (for metering)
    current_lufs: f64,
}

impl Compressor {
    /// Create a new compressor
    ///
    /// # Arguments
    /// * `threshold_db` - Threshold in dB (e.g., -20.0)
    /// * `ratio` - Compression ratio (e.g., 4.0 for 4:1)
    /// * `attack_ms` - Attack time in milliseconds
    /// * `release_ms` - Release time in milliseconds
    /// * `makeup_gain_db` - Makeup gain in dB
    /// * `knee_db` - Soft knee width in dB (0 = hard knee)
    /// * `sample_rate` - Sample rate in Hz
    pub fn new(
        threshold_db: f64,
        ratio: f64,
        attack_ms: f64,
        release_ms: f64,
        makeup_gain_db: f64,
        knee_db: f64,
        sample_rate: f64,
    ) -> Self {
        let attack_coeff = Self::time_constant_to_coeff(attack_ms, sample_rate);
        let release_coeff = Self::time_constant_to_coeff(release_ms, sample_rate);
        // RMS smoothing: 10ms time constant for fast response
        let rms_coeff = Self::time_constant_to_coeff(10.0, sample_rate);
        // Release smoothing: 100ms time constant for hysteresis
        let release_smoothing_coeff = Self::time_constant_to_coeff(100.0, sample_rate);
        // Makeup smoothing: 200ms time constant for smooth transitions
        let makeup_smoothing_coeff = Self::time_constant_to_coeff(200.0, sample_rate);

        // Create loudness meter (will be None if ebur128 feature not enabled)
        let loudness_meter = match crate::dsp::loudness::LoudnessMeter::new(sample_rate as u32) {
            Ok(meter) => Some(meter),
            Err(e) => {
                eprintln!("Failed to initialize loudness meter: {}", e);
                None
            }
        };

        Self {
            threshold_db,
            ratio: ratio.max(1.0), // Ratio must be >= 1
            attack_coeff,
            release_coeff,
            makeup_gain_db,
            makeup_gain_linear: Self::db_to_linear(makeup_gain_db),
            knee_db: knee_db.max(0.0),
            envelope_db: -120.0,
            envelope_squared: 0.0,
            rms_coeff,
            current_gain_reduction_db: 0.0,
            sample_rate,
            enabled: true,
            adaptive_release: false,
            base_release_ms: release_ms,
            current_release_ms: release_ms,
            overage_timer: 0.0,
            target_release_ms: release_ms,
            release_smoothing_coeff,
            loudness_meter,
            auto_makeup_enabled: false,
            target_lufs: -18.0,  // Podcast/streaming standard
            smoothed_makeup_gain: makeup_gain_db,  // Start with manual value
            makeup_smoothing_coeff,
            current_lufs: -100.0,
        }
    }

    /// Create with default parameters suitable for voice
    pub fn default_voice(sample_rate: f64) -> Self {
        Self::new(
            -20.0, // threshold_db
            4.0,   // ratio (4:1)
            10.0,  // attack_ms
            200.0, // release_ms
            0.0,   // makeup_gain_db
            6.0,   // knee_db (soft knee)
            sample_rate,
        )
    }

    /// Convert time constant in ms to exponential smoothing coefficient
    fn time_constant_to_coeff(time_ms: f64, sample_rate: f64) -> f64 {
        let tau = time_ms / 1000.0;
        (-1.0 / (tau * sample_rate)).exp()
    }

    /// Convert dB to linear amplitude
    #[inline]
    fn db_to_linear(db: f64) -> f64 {
        10.0_f64.powf(db / 20.0)
    }

    /// Convert linear amplitude to dB (with floor)
    #[inline]
    fn linear_to_db(linear: f64) -> f64 {
        20.0 * (linear.abs() + 1e-10).log10()
    }

    /// Set threshold in dB
    pub fn set_threshold(&mut self, threshold_db: f64) {
        self.threshold_db = threshold_db;
        self.reset_overage_timer();
    }

    /// Get current threshold in dB
    pub fn threshold_db(&self) -> f64 {
        self.threshold_db
    }

    /// Set compression ratio
    pub fn set_ratio(&mut self, ratio: f64) {
        self.ratio = ratio.max(1.0);
    }

    /// Get current ratio
    pub fn ratio(&self) -> f64 {
        self.ratio
    }

    /// Set attack time in ms
    pub fn set_attack_time(&mut self, attack_ms: f64) {
        self.attack_coeff = Self::time_constant_to_coeff(attack_ms, self.sample_rate);
    }

    /// Set release time in ms
    pub fn set_release_time(&mut self, release_ms: f64) {
        self.base_release_ms = release_ms;
        if !self.adaptive_release {
            self.current_release_ms = release_ms;
        }
        self.release_coeff = Self::time_constant_to_coeff(release_ms, self.sample_rate);
    }

    /// Enable or disable adaptive release
    pub fn set_adaptive_release(&mut self, enabled: bool) {
        self.adaptive_release = enabled;
        if !enabled {
            // Reset to base release when disabled
            self.current_release_ms = self.base_release_ms;
            self.target_release_ms = self.base_release_ms;
            self.overage_timer = 0.0;
        }
    }

    /// Check if adaptive release is enabled
    pub fn adaptive_release(&self) -> bool {
        self.adaptive_release
    }

    /// Set base release time
    pub fn set_base_release_time(&mut self, release_ms: f64) {
        self.base_release_ms = release_ms;
        if !self.adaptive_release {
            self.current_release_ms = release_ms;
        }
    }

    /// Get current release time (adaptive or base)
    pub fn current_release_time(&self) -> f64 {
        self.current_release_ms
    }

    /// Get base release time in milliseconds
    pub fn base_release_ms(&self) -> f64 {
        self.base_release_ms
    }

    /// Reset overage timer (call when threshold changes)
    pub fn reset_overage_timer(&mut self) {
        self.overage_timer = 0.0;
    }

    /// Set makeup gain in dB
    pub fn set_makeup_gain(&mut self, makeup_gain_db: f64) {
        self.makeup_gain_db = makeup_gain_db;
        self.makeup_gain_linear = Self::db_to_linear(makeup_gain_db);
        if !self.auto_makeup_enabled {
            self.smoothed_makeup_gain = makeup_gain_db;
        }
    }

    /// Get makeup gain in dB
    pub fn makeup_gain_db(&self) -> f64 {
        self.makeup_gain_db
    }

    /// Set knee width in dB
    pub fn set_knee(&mut self, knee_db: f64) {
        self.knee_db = knee_db.max(0.0);
    }

    /// Enable or disable the compressor
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if compressor is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get current gain reduction in dB (for metering)
    pub fn current_gain_reduction(&self) -> f64 {
        self.current_gain_reduction_db
    }

    /// Enable or disable auto makeup gain
    pub fn set_auto_makeup_enabled(&mut self, enabled: bool) {
        self.auto_makeup_enabled = enabled && self.loudness_meter.is_some();
        if !enabled {
            // Reset to manual makeup when disabled
            self.smoothed_makeup_gain = self.makeup_gain_db;
        }
    }

    /// Check if auto makeup is enabled
    pub fn auto_makeup_enabled(&self) -> bool {
        self.auto_makeup_enabled
    }

    /// Set target LUFS for auto makeup gain
    pub fn set_target_lufs(&mut self, target: f64) {
        self.target_lufs = target.clamp(-24.0, -12.0);
    }

    /// Get target LUFS
    pub fn target_lufs(&self) -> f64 {
        self.target_lufs
    }

    /// Get current measured loudness (for metering)
    pub fn current_lufs(&self) -> f64 {
        self.current_lufs
    }

    /// Get current applied makeup gain (for metering)
    pub fn current_makeup_gain(&self) -> f64 {
        self.smoothed_makeup_gain
    }

    /// Calculate adaptive release time based on overage duration
    fn calculate_adaptive_release(&mut self, sample_rate: f64) {
        if !self.adaptive_release {
            self.target_release_ms = self.base_release_ms;
            return;
        }

        // Scale release from 50ms to 400ms based on overage duration
        // Linear scaling over 2 seconds of sustained overage
        let max_overage_duration = 2.0; // seconds
        let overage_duration_sec = self.overage_timer / sample_rate;
        let scaling_factor = (overage_duration_sec / max_overage_duration).min(1.0);

        // Release scales from base to 8x base (max 400ms)
        let min_release = 50.0;
        let max_release = 400.0;
        let adaptive_range = max_release - min_release;

        self.target_release_ms = min_release + adaptive_range * scaling_factor;
    }

    /// Calculate and apply auto makeup gain based on loudness
    fn update_auto_makeup_gain(&mut self) {
        if !self.auto_makeup_enabled {
            // When disabled, smooth back to manual makeup gain
            let target = self.makeup_gain_db;
            let diff = target - self.smoothed_makeup_gain;
            if diff.abs() > 0.1 {
                self.smoothed_makeup_gain = self.makeup_smoothing_coeff * self.smoothed_makeup_gain
                    + (1.0 - self.makeup_smoothing_coeff) * target;
            } else {
                self.smoothed_makeup_gain = target;
            }
            return;
        }

        // Get current loudness
        if let Some(meter) = &self.loudness_meter {
            self.current_lufs = meter.loudness_momentary() as f64;

            // Calculate required gain to reach target
            // Formula: gain = target_lufs - current_lufs
            let required_gain = self.target_lufs - self.current_lufs;

            // Clamp to 0-12 dB range (prevent excessive gain)
            let clamped_gain = required_gain.clamp(0.0, 12.0);

            // Smooth gain changes with 200ms time constant
            let diff = clamped_gain - self.smoothed_makeup_gain;
            if diff.abs() > 0.1 {  // Only smooth if difference > 0.1dB
                self.smoothed_makeup_gain = self.makeup_smoothing_coeff * self.smoothed_makeup_gain
                    + (1.0 - self.makeup_smoothing_coeff) * clamped_gain;
            } else {
                self.smoothed_makeup_gain = clamped_gain;
            }
        }
    }

    /// Calculate gain reduction in dB for a given input level
    /// Implements soft-knee compression
    #[inline]
    fn compute_gain_reduction(&self, input_db: f64) -> f64 {
        let knee_half = self.knee_db / 2.0;
        let knee_start = self.threshold_db - knee_half;
        let knee_end = self.threshold_db + knee_half;

        if input_db <= knee_start {
            // Below knee: no compression
            0.0
        } else if input_db >= knee_end || self.knee_db <= 0.0 {
            // Above knee (or hard knee): full compression
            (input_db - self.threshold_db) * (1.0 - 1.0 / self.ratio)
        } else {
            // In the knee: smooth transition
            // Quadratic interpolation for soft knee
            let knee_factor = (input_db - knee_start) / self.knee_db;
            let compression_amount = (1.0 - 1.0 / self.ratio) * knee_factor * knee_factor;
            (input_db - knee_start) * compression_amount / 2.0
        }
    }

    /// Process a single sample
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        if !self.enabled {
            self.current_gain_reduction_db = 0.0;
            return input;
        }

        let input_f64 = input as f64;

        // IIR envelope follower (RMS approximation)
        let input_squared = input_f64 * input_f64;
        self.envelope_squared =
            self.rms_coeff * self.envelope_squared + (1.0 - self.rms_coeff) * input_squared;

        // Calculate RMS level in dB
        let rms = self.envelope_squared.sqrt();
        let input_db = Self::linear_to_db(rms);

        // Smooth envelope in dB domain with attack/release
        let coeff = if input_db > self.envelope_db {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope_db = coeff * self.envelope_db + (1.0 - coeff) * input_db;

        // Track overage duration
        let input_above_threshold = self.envelope_db > self.threshold_db;
        if input_above_threshold {
            // Increment overage timer (per sample)
            self.overage_timer += 1.0;
        } else {
            // Decay overage timer (quick release when below threshold)
            self.overage_timer = (self.overage_timer - 10.0).max(0.0);
        }

        // Calculate adaptive release target
        self.calculate_adaptive_release(self.sample_rate);

        // Smooth release time changes (100ms hysteresis)
        let release_diff = self.target_release_ms - self.current_release_ms;
        if release_diff.abs() > 1.0 {
            // Only smooth if difference > 1ms
            self.current_release_ms = self.release_smoothing_coeff * self.current_release_ms
                + (1.0 - self.release_smoothing_coeff) * self.target_release_ms;
        } else {
            self.current_release_ms = self.target_release_ms;
        }

        // Update release coefficient based on current adaptive release
        self.release_coeff = Self::time_constant_to_coeff(self.current_release_ms, self.sample_rate);

        // Calculate gain reduction
        let gain_reduction_db = self.compute_gain_reduction(self.envelope_db);
        self.current_gain_reduction_db = gain_reduction_db;

        // Update auto makeup gain
        self.update_auto_makeup_gain();

        // Apply gain reduction using smoothed makeup gain
        let output_gain = Self::db_to_linear(-gain_reduction_db)
            * Self::db_to_linear(self.smoothed_makeup_gain);

        (input_f64 * output_gain) as f32
    }

    /// Process a block of samples in-place
    pub fn process_block_inplace(&mut self, buffer: &mut [f32]) {
        if !self.enabled {
            self.current_gain_reduction_db = 0.0;
            return;
        }

        // Update loudness meter for entire block (more efficient)
        if let Some(meter) = &mut self.loudness_meter {
            meter.process(buffer);
        }

        // Update auto makeup gain once per block
        self.update_auto_makeup_gain();

        // Process samples using efficient block processing
        for sample in buffer.iter_mut() {
            *sample = self.process_sample_inner(*sample);
        }
    }

    /// Inner sample processing without auto makeup gain update
    /// (Called by process_block_inplace after updating makeup gain once)
    #[inline]
    fn process_sample_inner(&mut self, input: f32) -> f32 {
        if !self.enabled {
            self.current_gain_reduction_db = 0.0;
            return input;
        }

        let input_f64 = input as f64;

        // IIR envelope follower (RMS approximation)
        let input_squared = input_f64 * input_f64;
        self.envelope_squared =
            self.rms_coeff * self.envelope_squared + (1.0 - self.rms_coeff) * input_squared;

        // Calculate RMS level in dB
        let rms = self.envelope_squared.sqrt();
        let input_db = Self::linear_to_db(rms);

        // Smooth envelope in dB domain with attack/release
        let coeff = if input_db > self.envelope_db {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope_db = coeff * self.envelope_db + (1.0 - coeff) * input_db;

        // Track overage duration
        let input_above_threshold = self.envelope_db > self.threshold_db;
        if input_above_threshold {
            // Increment overage timer (per sample)
            self.overage_timer += 1.0;
        } else {
            // Decay overage timer (quick release when below threshold)
            self.overage_timer = (self.overage_timer - 10.0).max(0.0);
        }

        // Calculate adaptive release target
        self.calculate_adaptive_release(self.sample_rate);

        // Smooth release time changes (100ms hysteresis)
        let release_diff = self.target_release_ms - self.current_release_ms;
        if release_diff.abs() > 1.0 {
            // Only smooth if difference > 1ms
            self.current_release_ms = self.release_smoothing_coeff * self.current_release_ms
                + (1.0 - self.release_smoothing_coeff) * self.target_release_ms;
        } else {
            self.current_release_ms = self.target_release_ms;
        }

        // Update release coefficient based on current adaptive release
        self.release_coeff = Self::time_constant_to_coeff(self.current_release_ms, self.sample_rate);

        // Calculate gain reduction
        let gain_reduction_db = self.compute_gain_reduction(self.envelope_db);
        self.current_gain_reduction_db = gain_reduction_db;

        // Apply gain reduction using smoothed makeup gain
        let output_gain = Self::db_to_linear(-gain_reduction_db)
            * Self::db_to_linear(self.smoothed_makeup_gain);

        (input_f64 * output_gain) as f32
    }

    /// Reset compressor state
    pub fn reset(&mut self) {
        self.envelope_db = -120.0;
        self.envelope_squared = 0.0;
        self.current_gain_reduction_db = 0.0;
        self.overage_timer = 0.0;
        self.current_release_ms = self.base_release_ms;
        self.target_release_ms = self.base_release_ms;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressor_no_compression_below_threshold() {
        let mut comp = Compressor::new(-20.0, 4.0, 10.0, 200.0, 0.0, 0.0, 48000.0);

        // Very quiet signal (well below threshold)
        let input = 0.001f32; // About -60 dB
        let output = comp.process_sample(input);

        // Output should be nearly equal to input (no compression)
        assert!((output - input).abs() < 0.0001);
    }

    #[test]
    fn test_compressor_reduces_gain_above_threshold() {
        let mut comp = Compressor::new(-20.0, 4.0, 0.1, 200.0, 0.0, 0.0, 48000.0);

        // Feed loud signal to build up envelope
        let loud_signal = vec![0.3f32; 5000]; // About -10 dB
        for sample in &loud_signal {
            comp.process_sample(*sample);
        }

        // Should have some gain reduction
        assert!(comp.current_gain_reduction() > 0.0);
    }

    #[test]
    fn test_compressor_makeup_gain() {
        let mut comp = Compressor::new(-20.0, 4.0, 10.0, 200.0, 6.0, 0.0, 48000.0);

        // Quiet signal (below threshold)
        let input = 0.001f32;

        // Let envelope settle
        for _ in 0..1000 {
            comp.process_sample(input);
        }

        let output = comp.process_sample(input);

        // Output should be louder due to makeup gain (6 dB = ~2x)
        assert!(output > input * 1.5);
    }

    #[test]
    fn test_compressor_disabled() {
        let mut comp = Compressor::new(-20.0, 4.0, 10.0, 200.0, 6.0, 0.0, 48000.0);
        comp.set_enabled(false);

        let input = 0.5f32;
        let output = comp.process_sample(input);

        // When disabled, output should equal input exactly
        assert_eq!(output, input);
    }

    #[test]
    fn test_soft_knee() {
        let comp_hard = Compressor::new(-20.0, 4.0, 10.0, 200.0, 0.0, 0.0, 48000.0);
        let comp_soft = Compressor::new(-20.0, 4.0, 10.0, 200.0, 0.0, 12.0, 48000.0);

        // Test at -18 dB (within the 12 dB soft knee region: -26 to -14 dB)
        // Hard knee: this is 2 dB above threshold, so full compression applies
        // Soft knee: this is within the knee transition zone
        let at_minus_18_hard = comp_hard.compute_gain_reduction(-18.0);
        let at_minus_18_soft = comp_soft.compute_gain_reduction(-18.0);

        // Hard knee at -18 dB should have gain reduction (2 dB above threshold * 0.75 = 1.5 dB)
        assert!(
            at_minus_18_hard > 0.0,
            "Hard knee should compress at -18 dB"
        );

        // Soft knee should have less compression than hard knee within the knee region
        assert!(
            at_minus_18_soft < at_minus_18_hard,
            "Soft knee ({:.2}) should have less compression than hard knee ({:.2}) at -18 dB",
            at_minus_18_soft,
            at_minus_18_hard
        );

        // Well above threshold (outside knee), both should produce similar compression
        let well_above_hard = comp_hard.compute_gain_reduction(-5.0);
        let well_above_soft = comp_soft.compute_gain_reduction(-5.0);
        // At -5 dB (15 dB above threshold, well outside the soft knee),
        // both should be very close
        assert!((well_above_hard - well_above_soft).abs() < 0.5);
    }

    #[test]
    fn test_adaptive_release_disabled() {
        let mut comp = Compressor::new(-20.0, 4.0, 10.0, 50.0, 0.0, 0.0, 48000.0);
        comp.set_adaptive_release(false);

        // Verify current_release equals base when disabled
        assert_eq!(comp.current_release_time(), 50.0);
        assert!(!comp.adaptive_release());
    }

    #[test]
    fn test_adaptive_release_enables() {
        let mut comp = Compressor::new(-20.0, 4.0, 10.0, 50.0, 0.0, 0.0, 48000.0);
        comp.set_adaptive_release(true);

        assert!(comp.adaptive_release());

        // Feed signal above threshold to build overage
        let loud_signal = vec![0.3f32; 96000]; // 2 seconds at 48kHz
        for sample in &loud_signal {
            comp.process_sample(*sample);
        }

        // After 2 seconds overage, release should be near max (400ms)
        let current_release = comp.current_release_time();
        assert!(
            current_release > 300.0,
            "Release should scale up with overage, got {}",
            current_release
        );
    }

    #[test]
    fn test_adaptive_release_scales_linearly() {
        let mut comp = Compressor::new(-20.0, 4.0, 10.0, 50.0, 0.0, 0.0, 48000.0);
        comp.set_adaptive_release(true);

        // Test at 0.5 seconds overage (should be ~25% of max scaling)
        let quarter_signal = vec![0.3f32; (24000.0) as usize]; // 0.5 seconds
        for sample in &quarter_signal {
            comp.process_sample(*sample);
        }
        let release_quarter = comp.current_release_time();

        // Test at 1.0 seconds overage (should be ~50% of max scaling)
        comp.reset_overage_timer();
        let half_signal = vec![0.3f32; (48000.0) as usize]; // 1.0 second
        for sample in &half_signal {
            comp.process_sample(*sample);
        }
        let release_half = comp.current_release_time();

        // Half duration should have less release than quarter (accumulated)
        // Actually quarter was accumulated on zero, half starts from zero
        // So release_half should be roughly double release_quarter (minus min)
        assert!(release_half > release_quarter);
    }

    #[test]
    fn test_adaptive_release_resets_when_threshold_changes() {
        let mut comp = Compressor::new(-20.0, 4.0, 10.0, 50.0, 0.0, 0.0, 48000.0);
        comp.set_adaptive_release(true);

        // Build up overage
        let loud_signal = vec![0.3f32; 48000];
        for sample in &loud_signal {
            comp.process_sample(*sample);
        }

        // Record release before threshold change
        let release_before = comp.current_release_time();

        // Change threshold (should reset overage)
        comp.set_threshold(-30.0);

        // Feed quiet signal to let smoothing decay
        let quiet_signal = vec![0.001f32; 10000];
        for sample in &quiet_signal {
            comp.process_sample(*sample);
        }

        // After quiet signal, release should have decayed from previous high
        let release_after = comp.current_release_time();
        assert!(
            release_after < release_before,
            "Release should decay after threshold change and quiet signal, before: {}, after: {}",
            release_before,
            release_after
        );
    }

    #[test]
    fn test_adaptive_release_smooths_transitions() {
        let mut comp = Compressor::new(-20.0, 4.0, 10.0, 50.0, 0.0, 0.0, 48000.0);
        comp.set_adaptive_release(true);

        // Get base release
        let base_release = comp.current_release_time();

        // Build up some overage
        let signal = vec![0.3f32; 24000]; // 0.5 seconds
        for sample in &signal {
            comp.process_sample(*sample);
        }

        // After processing, release should have increased smoothly
        let increased_release = comp.current_release_time();
        assert!(increased_release > base_release);
        assert!(increased_release < 400.0); // Not at max yet
    }
}
