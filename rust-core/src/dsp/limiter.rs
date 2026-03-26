//! Lookahead hard limiter with gentle post safety clipping.
//!
//! Uses a fixed lookahead window (~2ms) computed from runtime sample rate.

use crate::dsp::util;

/// Hard limiter with lookahead ceiling control
pub struct Limiter {
    /// Ceiling in dB (e.g., -0.5 dB)
    ceiling_db: f64,
    /// Ceiling as linear amplitude (cached)
    ceiling_linear: f64,
    /// Release time constant (exponential smoothing coefficient)
    release_coeff: f64,
    /// Current gain reduction (linear, 0.0 to 1.0)
    gain_reduction: f64,
    /// Peak hold for metering
    peak_gain_reduction_db: f64,
    /// Sample rate
    sample_rate: f64,
    /// Lookahead delay in samples (computed from sample rate).
    lookahead_samples: usize,
    /// Shared future window and delay line storage.
    delay_buffer: Vec<f32>,
    /// Ring write index.
    write_idx: usize,
    /// Whether limiter is enabled
    enabled: bool,
}

impl Limiter {
    /// Create a new limiter
    pub fn new(ceiling_db: f64, release_ms: f64, sample_rate: f64) -> Self {
        let release_coeff = util::time_constant_to_coeff(release_ms, sample_rate);
        let lookahead_samples = ((0.002 * sample_rate).round() as usize).max(1);

        Self {
            ceiling_db,
            ceiling_linear: util::db_to_linear(ceiling_db),
            release_coeff,
            gain_reduction: 1.0,
            peak_gain_reduction_db: 0.0,
            sample_rate,
            lookahead_samples,
            delay_buffer: vec![0.0; lookahead_samples],
            write_idx: 0,
            enabled: true,
        }
    }

    /// Create with default parameters (-0.5 dB ceiling, 50ms release)
    pub fn default_settings(sample_rate: f64) -> Self {
        Self::new(-0.5, 50.0, sample_rate)
    }

    /// Set ceiling in dB
    pub fn set_ceiling(&mut self, ceiling_db: f64) {
        self.ceiling_db = ceiling_db.min(0.0);
        self.ceiling_linear = util::db_to_linear(self.ceiling_db);
    }

    /// Get current ceiling in dB
    pub fn ceiling_db(&self) -> f64 {
        self.ceiling_db
    }

    /// Set release time in ms
    pub fn set_release_time(&mut self, release_ms: f64) {
        self.release_coeff = util::time_constant_to_coeff(release_ms, self.sample_rate);
    }

    /// Get configured lookahead in samples.
    pub fn lookahead_samples(&self) -> usize {
        self.lookahead_samples
    }

    /// Get lookahead in milliseconds.
    pub fn lookahead_ms(&self) -> f64 {
        self.lookahead_samples as f64 / self.sample_rate * 1000.0
    }

    /// Enable or disable the limiter
    pub fn set_enabled(&mut self, enabled: bool) {
        if self.enabled != enabled {
            self.reset();
        }
        self.enabled = enabled;
    }

    /// Check if limiter is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get current gain reduction in dB (for metering)
    pub fn current_gain_reduction(&self) -> f64 {
        if self.gain_reduction >= 1.0 {
            0.0
        } else {
            util::linear_to_db(self.gain_reduction, 1e-10)
        }
    }

    /// Get peak gain reduction since last query (for metering)
    pub fn peak_gain_reduction_and_reset(&mut self) -> f64 {
        let peak = self.peak_gain_reduction_db;
        self.peak_gain_reduction_db = 0.0;
        peak
    }

    #[inline]
    fn lookahead_peak_abs(&self) -> f64 {
        let mut peak = 0.0_f64;
        for &sample in &self.delay_buffer {
            let s = (sample as f64).abs();
            if s > peak {
                peak = s;
            }
        }
        peak
    }

    #[inline]
    fn apply_soft_clip(&self, sample: f64) -> f64 {
        if self.ceiling_linear <= 0.0 {
            return 0.0;
        }
        if sample.abs() <= self.ceiling_linear {
            return sample;
        }
        let normalized = sample / self.ceiling_linear;
        normalized.tanh() * self.ceiling_linear
    }

    /// Process a single sample
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        if !self.enabled {
            return input;
        }

        // Compute planning peak before overwriting the slot so the delayed
        // sample being output now remains part of the decision window.
        let delayed = self.delay_buffer[self.write_idx] as f64;
        let peak = self.lookahead_peak_abs().max((input as f64).abs());

        // Current sample is then written into the future side of the ring.
        self.delay_buffer[self.write_idx] = input;
        self.write_idx = (self.write_idx + 1) % self.lookahead_samples;
        let target_gain = if peak > self.ceiling_linear {
            self.ceiling_linear / peak
        } else {
            1.0
        };

        if target_gain < self.gain_reduction {
            self.gain_reduction = target_gain;
        } else {
            self.gain_reduction =
                self.release_coeff * self.gain_reduction + (1.0 - self.release_coeff) * target_gain;
        }

        let reduction_db = if self.gain_reduction < 1.0 {
            -util::linear_to_db(self.gain_reduction, 1e-10)
        } else {
            0.0
        };
        if reduction_db > self.peak_gain_reduction_db {
            self.peak_gain_reduction_db = reduction_db;
        }

        let limited = delayed * self.gain_reduction;
        self.apply_soft_clip(limited) as f32
    }

    /// Process a block of samples in-place
    pub fn process_block_inplace(&mut self, buffer: &mut [f32]) {
        if !self.enabled {
            return;
        }

        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Reset limiter state
    pub fn reset(&mut self) {
        self.gain_reduction = 1.0;
        self.peak_gain_reduction_db = 0.0;
        self.write_idx = 0;
        self.delay_buffer.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookahead_scales_with_sample_rate() {
        let lim_44 = Limiter::new(-0.5, 50.0, 44_100.0);
        let lim_48 = Limiter::new(-0.5, 50.0, 48_000.0);
        let lim_96 = Limiter::new(-0.5, 50.0, 96_000.0);

        assert_eq!(lim_44.lookahead_samples(), 88);
        assert_eq!(lim_48.lookahead_samples(), 96);
        assert_eq!(lim_96.lookahead_samples(), 192);
    }

    #[test]
    fn test_limiter_no_reduction_below_ceiling() {
        let mut lim = Limiter::new(-0.5, 50.0, 48_000.0);
        let input = 0.5f32;
        let mut output = 0.0f32;
        for _ in 0..(lim.lookahead_samples() + 8) {
            output = lim.process_sample(input);
        }
        assert!((output - input).abs() < 0.02);
    }

    #[test]
    fn test_limiter_reduces_above_ceiling() {
        let mut lim = Limiter::new(-6.0, 50.0, 48_000.0);
        let mut peak = 0.0f32;
        for _ in 0..(lim.lookahead_samples() + 64) {
            let output = lim.process_sample(0.9);
            peak = peak.max(output.abs());
        }
        assert!(peak <= 0.51);
    }

    #[test]
    fn test_limiter_disabled() {
        let mut lim = Limiter::new(-6.0, 50.0, 48_000.0);
        lim.set_enabled(false);

        let input = 0.9f32;
        let output = lim.process_sample(input);
        assert_eq!(output, input);
    }

    #[test]
    fn test_limiter_release() {
        let mut lim = Limiter::new(-6.0, 10.0, 48_000.0);
        for _ in 0..(lim.lookahead_samples() + 100) {
            lim.process_sample(0.9);
        }

        let initial_reduction = lim.gain_reduction;
        for _ in 0..1000 {
            lim.process_sample(0.1);
        }
        assert!(lim.gain_reduction > initial_reduction);
    }

    #[test]
    fn test_limiter_process_block_inplace_respects_ceiling() {
        let mut lim = Limiter::new(-3.0, 50.0, 48_000.0);
        let ceiling_linear = 10.0_f64.powf(-3.0 / 20.0) as f32;
        let mut block = vec![1.2f32; 256];
        lim.process_block_inplace(&mut block);

        for sample in block.iter().skip(lim.lookahead_samples()) {
            assert!(sample.abs() <= ceiling_linear + 0.002);
        }
    }

    #[test]
    fn test_limiter_includes_delayed_sample_in_peak_planning() {
        let mut lim = Limiter::new(-6.0, 50.0, 48_000.0);
        lim.process_sample(0.9);
        let mut delayed_output = 0.0f32;
        for _ in 0..lim.lookahead_samples() {
            delayed_output = lim.process_sample(0.0);
        }

        assert!(lim.gain_reduction < 0.95);
        assert!(delayed_output.abs() <= 0.51);
    }

    #[test]
    fn test_peak_gain_reduction_and_reset_reports_and_clears_peak() {
        let mut lim = Limiter::new(-6.0, 50.0, 48_000.0);
        for _ in 0..(lim.lookahead_samples() + 8) {
            lim.process_sample(0.95);
        }
        let peak = lim.peak_gain_reduction_and_reset();
        assert!(peak > 0.0);
        assert_eq!(lim.peak_gain_reduction_and_reset(), 0.0);
    }

    #[test]
    fn test_set_ceiling_clamps_to_zero_db_max() {
        let mut lim = Limiter::new(-6.0, 50.0, 48_000.0);
        lim.set_ceiling(3.0);
        assert_eq!(lim.ceiling_db(), 0.0);
    }

    #[test]
    fn test_limiter_disable_transition_resets_stale_state() {
        let mut lim = Limiter::new(-6.0, 50.0, 48_000.0);

        let lookahead = lim.lookahead_samples();
        lim.process_sample(0.95);
        for _ in 0..lookahead {
            lim.process_sample(0.0);
        }

        assert!(lim.current_gain_reduction() < 0.95);

        lim.set_enabled(false);
        assert_eq!(lim.current_gain_reduction(), 0.0);
        assert_eq!(lim.peak_gain_reduction_and_reset(), 0.0);

        lim.set_enabled(true);
        let output = lim.process_sample(0.0);
        assert_eq!(output, 0.0);
        assert_eq!(lim.current_gain_reduction(), 0.0);
    }

    #[test]
    fn test_limiter_enable_transition_clears_previous_delay_line() {
        let mut lim = Limiter::new(-6.0, 50.0, 48_000.0);

        let lookahead = lim.lookahead_samples();
        lim.process_sample(0.9);
        for _ in 0..(lookahead / 2).max(1) {
            lim.process_sample(0.0);
        }

        lim.set_enabled(false);
        lim.set_enabled(true);

        let mut outputs = Vec::with_capacity(lookahead + 1);
        for _ in 0..=lookahead {
            outputs.push(lim.process_sample(0.0));
        }

        assert!(outputs.iter().all(|sample| sample.abs() < 1e-6));
        assert_eq!(lim.current_gain_reduction(), 0.0);
    }
}
