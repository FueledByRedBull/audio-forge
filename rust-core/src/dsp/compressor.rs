//! Downward compressor with blended peak/RMS detection.
//!
//! Reduces dynamic range by attenuating signals above the threshold.

use crate::dsp::util;

const DETECTOR_PEAK_WEIGHT: f64 = 0.6;
const DETECTOR_RMS_WEIGHT: f64 = 0.4;

/// Downward compressor with soft-knee gain reduction
pub struct Compressor {
    /// Threshold in dB - compression starts above this level
    threshold_db: f64,
    /// Compression ratio (e.g., 4.0 = 4:1 ratio)
    ratio: f64,
    /// Attack time constant (exponential smoothing coefficient)
    attack_coeff: f64,
    /// Release time constant for gain-reduction smoothing
    release_coeff: f64,
    /// Release time constant for the peak detector envelope
    detector_release_coeff: f64,
    /// Makeup gain in dB to compensate for gain reduction
    makeup_gain_db: f64,
    /// Makeup gain as linear multiplier (cached)
    makeup_gain_linear: f64,
    /// Knee width in dB for soft-knee transition
    knee_db: f64,
    /// AR-smoothed log-domain peak detector (dBFS)
    peak_envelope_db: f64,
    /// Fixed-time RMS detector state (squared amplitude)
    rms_envelope_sq: f64,
    /// RMS smoothing coefficient (single-pole IIR, fixed 20ms)
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
    pub fn new(
        threshold_db: f64,
        ratio: f64,
        attack_ms: f64,
        release_ms: f64,
        makeup_gain_db: f64,
        knee_db: f64,
        sample_rate: f64,
    ) -> Self {
        let attack_coeff = util::time_constant_to_coeff(attack_ms, sample_rate);
        let release_coeff = util::time_constant_to_coeff(release_ms, sample_rate);
        let rms_coeff = util::time_constant_to_coeff(20.0, sample_rate);
        let release_smoothing_coeff = util::time_constant_to_coeff(100.0, sample_rate);
        let makeup_smoothing_coeff = util::time_constant_to_coeff(200.0, sample_rate);

        let loudness_meter = match crate::dsp::loudness::LoudnessMeter::new(sample_rate as u32) {
            Ok(meter) => Some(meter),
            Err(e) => {
                eprintln!("Failed to initialize loudness meter: {}", e);
                None
            }
        };

        Self {
            threshold_db,
            ratio: ratio.max(1.0),
            attack_coeff,
            release_coeff,
            detector_release_coeff: release_coeff,
            makeup_gain_db,
            makeup_gain_linear: util::db_to_linear(makeup_gain_db),
            knee_db: knee_db.max(0.0),
            peak_envelope_db: -120.0,
            rms_envelope_sq: 0.0,
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
            target_lufs: -18.0,
            smoothed_makeup_gain: makeup_gain_db,
            makeup_smoothing_coeff,
            current_lufs: -100.0,
        }
    }

    /// Create with default parameters suitable for voice
    pub fn default_voice(sample_rate: f64) -> Self {
        Self::new(-20.0, 4.0, 10.0, 200.0, 0.0, 6.0, sample_rate)
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
        self.attack_coeff = util::time_constant_to_coeff(attack_ms, self.sample_rate);
    }

    /// Set release time in ms
    pub fn set_release_time(&mut self, release_ms: f64) {
        self.base_release_ms = release_ms;
        if !self.adaptive_release {
            self.current_release_ms = release_ms;
            self.target_release_ms = release_ms;
            self.release_coeff = util::time_constant_to_coeff(release_ms, self.sample_rate);
        }
        self.detector_release_coeff = util::time_constant_to_coeff(release_ms, self.sample_rate);
    }

    /// Enable or disable adaptive release
    pub fn set_adaptive_release(&mut self, enabled: bool) {
        self.adaptive_release = enabled;
        if !enabled {
            self.current_release_ms = self.base_release_ms;
            self.target_release_ms = self.base_release_ms;
            self.overage_timer = 0.0;
            self.release_coeff =
                util::time_constant_to_coeff(self.current_release_ms, self.sample_rate);
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
            self.target_release_ms = release_ms;
            self.release_coeff = util::time_constant_to_coeff(release_ms, self.sample_rate);
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
        self.makeup_gain_linear = util::db_to_linear(makeup_gain_db);
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

    fn calculate_adaptive_release(&mut self, sample_rate: f64) {
        if !self.adaptive_release {
            self.target_release_ms = self.base_release_ms;
            return;
        }

        let max_overage_duration = 2.0;
        let overage_duration_sec = self.overage_timer / sample_rate;
        let scaling_factor = (overage_duration_sec / max_overage_duration).min(1.0);

        let min_release = 50.0;
        let max_release = 400.0;
        let adaptive_range = max_release - min_release;

        self.target_release_ms = min_release + adaptive_range * scaling_factor;
    }

    fn update_auto_makeup_gain(&mut self) {
        if !self.auto_makeup_enabled {
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

        if let Some(meter) = &self.loudness_meter {
            self.current_lufs = meter.loudness_momentary() as f64;
            let required_gain = self.target_lufs - self.current_lufs;
            let clamped_gain = required_gain.clamp(0.0, 12.0);

            let diff = clamped_gain - self.smoothed_makeup_gain;
            if diff.abs() > 0.1 {
                self.smoothed_makeup_gain = self.makeup_smoothing_coeff * self.smoothed_makeup_gain
                    + (1.0 - self.makeup_smoothing_coeff) * clamped_gain;
            } else {
                self.smoothed_makeup_gain = clamped_gain;
            }
        }
    }

    /// Calculate gain reduction in dB for a given detector level.
    #[inline]
    fn compute_gain_reduction(&self, detector_db: f64) -> f64 {
        let comp_factor = 1.0 - 1.0 / self.ratio;
        if self.knee_db <= 0.0 {
            if detector_db <= self.threshold_db {
                return 0.0;
            }
            return (detector_db - self.threshold_db) * comp_factor;
        }

        let knee_half = self.knee_db / 2.0;
        let knee_start = self.threshold_db - knee_half;
        let knee_end = self.threshold_db + knee_half;

        if detector_db <= knee_start {
            0.0
        } else if detector_db >= knee_end {
            (detector_db - self.threshold_db) * comp_factor
        } else {
            let x = detector_db - knee_start;
            comp_factor * x * x / (2.0 * self.knee_db)
        }
    }

    /// Process a single sample
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        self.process_sample_impl(input, true)
    }

    /// Process a block of samples in-place
    pub fn process_block_inplace(&mut self, buffer: &mut [f32]) {
        if !self.enabled {
            self.current_gain_reduction_db = 0.0;
            return;
        }

        if let Some(meter) = &mut self.loudness_meter {
            meter.process(buffer);
        }

        self.update_auto_makeup_gain();
        for sample in buffer.iter_mut() {
            *sample = self.process_sample_impl(*sample, false);
        }
    }

    #[inline]
    fn process_sample_impl(&mut self, input: f32, update_makeup_gain: bool) -> f32 {
        if !self.enabled {
            self.current_gain_reduction_db = 0.0;
            return input;
        }

        let input_f64 = input as f64;
        let input_abs = input_f64.abs();
        let inst_peak_db = util::linear_to_db(input_abs, 1e-10);
        let peak_coeff = if inst_peak_db > self.peak_envelope_db {
            self.attack_coeff
        } else {
            self.detector_release_coeff
        };
        self.peak_envelope_db =
            peak_coeff * self.peak_envelope_db + (1.0 - peak_coeff) * inst_peak_db;

        let input_squared = input_f64 * input_f64;
        self.rms_envelope_sq =
            self.rms_coeff * self.rms_envelope_sq + (1.0 - self.rms_coeff) * input_squared;
        let rms_db = util::linear_to_db(self.rms_envelope_sq.sqrt(), 1e-10);

        let detector_db =
            DETECTOR_PEAK_WEIGHT * self.peak_envelope_db + DETECTOR_RMS_WEIGHT * rms_db;

        let input_above_threshold = detector_db > self.threshold_db;
        if input_above_threshold {
            self.overage_timer += 1.0;
        } else {
            self.overage_timer = (self.overage_timer - 10.0).max(0.0);
        }

        self.calculate_adaptive_release(self.sample_rate);
        let release_diff = self.target_release_ms - self.current_release_ms;
        if release_diff.abs() > 1.0 {
            self.current_release_ms = self.release_smoothing_coeff * self.current_release_ms
                + (1.0 - self.release_smoothing_coeff) * self.target_release_ms;
        } else {
            self.current_release_ms = self.target_release_ms;
        }
        self.release_coeff =
            util::time_constant_to_coeff(self.current_release_ms, self.sample_rate);

        let target_gain_reduction_db = self.compute_gain_reduction(detector_db);
        let gr_coeff = if target_gain_reduction_db > self.current_gain_reduction_db {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.current_gain_reduction_db =
            gr_coeff * self.current_gain_reduction_db + (1.0 - gr_coeff) * target_gain_reduction_db;

        if update_makeup_gain {
            self.update_auto_makeup_gain();
        }

        let output_gain = util::db_to_linear(-self.current_gain_reduction_db)
            * util::db_to_linear(self.smoothed_makeup_gain);
        (input_f64 * output_gain) as f32
    }

    /// Reset compressor state
    pub fn reset(&mut self) {
        self.peak_envelope_db = -120.0;
        self.rms_envelope_sq = 0.0;
        self.current_gain_reduction_db = 0.0;
        self.overage_timer = 0.0;
        self.current_release_ms = self.base_release_ms;
        self.target_release_ms = self.base_release_ms;
        self.release_coeff =
            util::time_constant_to_coeff(self.current_release_ms, self.sample_rate);
        if let Some(meter) = &mut self.loudness_meter {
            if let Err(err) = meter.reset() {
                eprintln!("Failed to reset loudness meter: {}", err);
            } else {
                self.current_lufs = -100.0;
            }
        } else {
            self.current_lufs = -100.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressor_no_compression_below_threshold() {
        let mut comp = Compressor::new(-20.0, 4.0, 10.0, 200.0, 0.0, 0.0, 48_000.0);
        let input = 0.001f32;
        let output = comp.process_sample(input);
        assert!((output - input).abs() < 0.0001);
    }

    #[test]
    fn test_compressor_reduces_gain_above_threshold() {
        let mut comp = Compressor::new(-20.0, 4.0, 0.1, 200.0, 0.0, 0.0, 48_000.0);
        let loud_signal = vec![0.3f32; 5_000];
        for sample in &loud_signal {
            comp.process_sample(*sample);
        }
        assert!(comp.current_gain_reduction() > 0.0);
    }

    #[test]
    fn test_compressor_makeup_gain() {
        let mut comp = Compressor::new(-20.0, 4.0, 10.0, 200.0, 6.0, 0.0, 48_000.0);
        let input = 0.001f32;
        for _ in 0..1000 {
            comp.process_sample(input);
        }
        let output = comp.process_sample(input);
        assert!(output > input * 1.5);
    }

    #[test]
    fn test_compressor_disabled() {
        let mut comp = Compressor::new(-20.0, 4.0, 10.0, 200.0, 6.0, 0.0, 48_000.0);
        comp.set_enabled(false);
        let input = 0.5f32;
        let output = comp.process_sample(input);
        assert_eq!(output, input);
    }

    #[test]
    fn test_soft_knee() {
        let comp_hard = Compressor::new(-20.0, 4.0, 10.0, 200.0, 0.0, 0.0, 48_000.0);
        let comp_soft = Compressor::new(-20.0, 4.0, 10.0, 200.0, 0.0, 12.0, 48_000.0);

        // Inside the knee but below threshold, soft knee should start compressing
        // while hard knee still applies no gain reduction.
        let at_minus_22_hard = comp_hard.compute_gain_reduction(-22.0);
        let at_minus_22_soft = comp_soft.compute_gain_reduction(-22.0);
        assert!((at_minus_22_hard - 0.0).abs() < 1e-12);
        assert!(at_minus_22_soft > 0.0);

        let well_above_hard = comp_hard.compute_gain_reduction(-5.0);
        let well_above_soft = comp_soft.compute_gain_reduction(-5.0);
        assert!((well_above_hard - well_above_soft).abs() < 0.5);
    }

    #[test]
    fn test_soft_knee_exact_boundaries() {
        let comp = Compressor::new(-20.0, 4.0, 10.0, 200.0, 0.0, 12.0, 48_000.0);
        let w = 12.0;
        let t = -20.0;
        let comp_factor = 1.0 - 1.0 / 4.0;

        let at_knee_start = comp.compute_gain_reduction(t - w / 2.0);
        let at_knee_end = comp.compute_gain_reduction(t + w / 2.0);

        assert!((at_knee_start - 0.0).abs() < 1e-12);
        assert!((at_knee_end - ((w / 2.0) * comp_factor)).abs() < 1e-12);
    }

    #[test]
    fn test_adaptive_release_enables() {
        let mut comp = Compressor::new(-20.0, 4.0, 10.0, 50.0, 0.0, 0.0, 48_000.0);
        comp.set_adaptive_release(true);
        assert!(comp.adaptive_release());

        let loud_signal = vec![0.3f32; 96_000];
        for sample in &loud_signal {
            comp.process_sample(*sample);
        }
        let current_release = comp.current_release_time();
        assert!(current_release > 300.0);
    }

    #[test]
    fn test_release_time_changes_recovery_speed() {
        let mut fast = Compressor::new(-20.0, 4.0, 0.1, 20.0, 0.0, 0.0, 48_000.0);
        let mut slow = Compressor::new(-20.0, 4.0, 0.1, 200.0, 0.0, 0.0, 48_000.0);

        for _ in 0..4_000 {
            fast.process_sample(0.5);
            slow.process_sample(0.5);
        }

        for _ in 0..1_440 {
            fast.process_sample(0.001);
            slow.process_sample(0.001);
        }

        assert!(fast.current_gain_reduction() < slow.current_gain_reduction());
    }

    #[test]
    fn test_adaptive_release_does_not_change_detector_release_decay() {
        let mut fixed = Compressor::new(-20.0, 4.0, 1.0, 80.0, 0.0, 0.0, 48_000.0);
        let mut adaptive = Compressor::new(-20.0, 4.0, 1.0, 80.0, 0.0, 0.0, 48_000.0);
        adaptive.set_adaptive_release(true);

        for _ in 0..96_000 {
            fixed.process_sample(0.4);
            adaptive.process_sample(0.4);
        }

        assert!(adaptive.current_release_time() > fixed.current_release_time());

        let fixed_peak_before = fixed.peak_envelope_db;
        let adaptive_peak_before = adaptive.peak_envelope_db;
        for _ in 0..2_400 {
            fixed.process_sample(0.001);
            adaptive.process_sample(0.001);
        }

        let fixed_drop = fixed_peak_before - fixed.peak_envelope_db;
        let adaptive_drop = adaptive_peak_before - adaptive.peak_envelope_db;
        assert!((fixed_drop - adaptive_drop).abs() < 1e-9);
    }

    #[test]
    fn test_reset_clears_reported_loudness() {
        let mut comp = Compressor::default_voice(48_000.0);
        comp.current_lufs = -18.0;

        comp.reset();

        assert_eq!(comp.current_lufs(), -100.0);
    }
}
