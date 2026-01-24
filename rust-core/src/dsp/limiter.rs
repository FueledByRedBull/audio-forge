//! Hard limiter (brick-wall ceiling)
//!
//! Prevents signal from exceeding a ceiling level. Uses very fast attack
//! (~0.1ms) and configurable release to catch transients without lookahead.
//! This keeps latency minimal at the cost of slightly less transparent limiting.

/// Hard limiter with brick-wall ceiling
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

    /// Whether limiter is enabled
    enabled: bool,
}

impl Limiter {
    /// Create a new limiter
    ///
    /// # Arguments
    /// * `ceiling_db` - Ceiling in dB (e.g., -0.5)
    /// * `release_ms` - Release time in milliseconds
    /// * `sample_rate` - Sample rate in Hz
    pub fn new(ceiling_db: f64, release_ms: f64, sample_rate: f64) -> Self {
        let release_coeff = Self::time_constant_to_coeff(release_ms, sample_rate);

        Self {
            ceiling_db,
            ceiling_linear: Self::db_to_linear(ceiling_db),
            release_coeff,
            gain_reduction: 1.0, // No reduction initially
            peak_gain_reduction_db: 0.0,
            sample_rate,
            enabled: true,
        }
    }

    /// Create with default parameters (-0.5 dB ceiling, 50ms release)
    pub fn default_settings(sample_rate: f64) -> Self {
        Self::new(-0.5, 50.0, sample_rate)
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

    /// Convert linear amplitude to dB
    #[inline]
    fn linear_to_db(linear: f64) -> f64 {
        20.0 * (linear + 1e-10).log10()
    }

    /// Set ceiling in dB
    pub fn set_ceiling(&mut self, ceiling_db: f64) {
        self.ceiling_db = ceiling_db.min(0.0); // Can't go above 0 dB
        self.ceiling_linear = Self::db_to_linear(self.ceiling_db);
    }

    /// Get current ceiling in dB
    pub fn ceiling_db(&self) -> f64 {
        self.ceiling_db
    }

    /// Set release time in ms
    pub fn set_release_time(&mut self, release_ms: f64) {
        self.release_coeff = Self::time_constant_to_coeff(release_ms, self.sample_rate);
    }

    /// Enable or disable the limiter
    pub fn set_enabled(&mut self, enabled: bool) {
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
            Self::linear_to_db(self.gain_reduction)
        }
    }

    /// Get peak gain reduction since last query (for metering)
    pub fn peak_gain_reduction_and_reset(&mut self) -> f64 {
        let peak = self.peak_gain_reduction_db;
        self.peak_gain_reduction_db = 0.0;
        peak
    }

    /// Process a single sample
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        if !self.enabled {
            return input;
        }

        let input_f64 = input as f64;
        let input_abs = input_f64.abs();

        // Calculate required gain to stay under ceiling
        let target_gain = if input_abs > self.ceiling_linear {
            self.ceiling_linear / input_abs
        } else {
            1.0
        };

        // Attack is instant (no attack coefficient needed for brick-wall)
        // Release is smooth
        if target_gain < self.gain_reduction {
            // Need more reduction: instant attack
            self.gain_reduction = target_gain;
        } else {
            // Releasing: smooth recovery
            self.gain_reduction =
                self.release_coeff * self.gain_reduction + (1.0 - self.release_coeff) * target_gain;
        }

        // Track peak reduction for metering
        let reduction_db = if self.gain_reduction < 1.0 {
            -Self::linear_to_db(self.gain_reduction)
        } else {
            0.0
        };
        if reduction_db > self.peak_gain_reduction_db {
            self.peak_gain_reduction_db = reduction_db;
        }

        // Apply gain reduction
        (input_f64 * self.gain_reduction) as f32
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_limiter_no_reduction_below_ceiling() {
        let mut lim = Limiter::new(-0.5, 50.0, 48000.0);

        // Signal well below ceiling
        let input = 0.5f32; // About -6 dB, well below -0.5 dB ceiling
        let output = lim.process_sample(input);

        // Should pass through unchanged (or nearly so)
        assert!((output - input).abs() < 0.001);
    }

    #[test]
    fn test_limiter_reduces_above_ceiling() {
        let mut lim = Limiter::new(-6.0, 50.0, 48000.0); // -6 dB ceiling = 0.5 linear

        // Signal above ceiling
        let input = 0.9f32; // About -0.9 dB, above -6 dB ceiling
        let output = lim.process_sample(input);

        // Output should be limited to ceiling
        assert!(output.abs() <= 0.51); // -6 dB with small tolerance
    }

    #[test]
    fn test_limiter_brick_wall() {
        let mut lim = Limiter::new(-0.5, 50.0, 48000.0);
        let ceiling_linear = 10.0_f64.powf(-0.5 / 20.0) as f32;

        // Even very loud signals should be limited
        let loud_signal = vec![1.5f32; 100];
        for sample in loud_signal {
            let output = lim.process_sample(sample);
            assert!(
                output.abs() <= ceiling_linear + 0.001,
                "Output {} exceeded ceiling {}",
                output,
                ceiling_linear
            );
        }
    }

    #[test]
    fn test_limiter_disabled() {
        let mut lim = Limiter::new(-6.0, 50.0, 48000.0);
        lim.set_enabled(false);

        let input = 0.9f32; // Above ceiling
        let output = lim.process_sample(input);

        // When disabled, signal passes through unchanged
        assert_eq!(output, input);
    }

    #[test]
    fn test_limiter_release() {
        let mut lim = Limiter::new(-6.0, 10.0, 48000.0); // Fast 10ms release

        // Trigger limiting
        for _ in 0..10 {
            lim.process_sample(0.9);
        }

        // Now process quiet signal and check release
        let initial_reduction = lim.gain_reduction;

        for _ in 0..1000 {
            lim.process_sample(0.1);
        }

        // Gain reduction should have recovered (closer to 1.0)
        assert!(lim.gain_reduction > initial_reduction);
    }
}
