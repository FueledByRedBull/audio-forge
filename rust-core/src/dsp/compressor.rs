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

    /// Reset overage timer (call when threshold changes)
    pub fn reset_overage_timer(&mut self) {
        self.overage_timer = 0.0;
    }

    /// Set makeup gain in dB
    pub fn set_makeup_gain(&mut self, makeup_gain_db: f64) {
        self.makeup_gain_db = makeup_gain_db;
        self.makeup_gain_linear = Self::db_to_linear(makeup_gain_db);
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

        // Calculate gain reduction
        let gain_reduction_db = self.compute_gain_reduction(self.envelope_db);
        self.current_gain_reduction_db = gain_reduction_db;

        // Apply gain reduction and makeup gain
        let output_gain = Self::db_to_linear(-gain_reduction_db) * self.makeup_gain_linear;

        (input_f64 * output_gain) as f32
    }

    /// Process a block of samples in-place
    pub fn process_block_inplace(&mut self, buffer: &mut [f32]) {
        if !self.enabled {
            self.current_gain_reduction_db = 0.0;
            return;
        }

        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Reset compressor state
    pub fn reset(&mut self) {
        self.envelope_db = -120.0;
        self.envelope_squared = 0.0;
        self.current_gain_reduction_db = 0.0;
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
}
