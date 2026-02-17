//! IIR Biquad filter implementation using Direct Form II Transposed
//!
//! Uses f64 internally for coefficient precision, f32 for audio samples.

use std::f64::consts::PI;

/// Biquad filter types
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BiquadType {
    LowShelf,
    HighShelf,
    Peaking,
    HighPass,
    LowPass,
}

/// IIR Biquad filter using Direct Form II Transposed
///
/// Transfer function: H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
///
/// Uses f64 internally for coefficient precision to avoid numerical issues,
/// but processes f32 audio samples for compatibility with audio drivers.
#[derive(Clone, Debug)]
pub struct Biquad {
    // Coefficients (normalized so a0 = 1)
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,

    // State variables for Direct Form II Transposed
    z1: f64,
    z2: f64,

    // Parameters for recalculation
    filter_type: BiquadType,
    frequency: f64,
    gain_db: f64,
    q: f64,
    sample_rate: f64,
    enabled: bool,
}

impl Biquad {
    /// Create a new biquad filter
    pub fn new(
        filter_type: BiquadType,
        frequency: f64,
        gain_db: f64,
        q: f64,
        sample_rate: f64,
    ) -> Self {
        let mut filter = Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1: 0.0,
            z2: 0.0,
            filter_type,
            frequency,
            gain_db,
            q,
            sample_rate,
            enabled: true,
        };
        filter.calculate_coefficients();
        filter
    }

    /// Calculate filter coefficients based on current parameters
    /// Uses Robert Bristow-Johnson's Audio EQ Cookbook formulas
    fn calculate_coefficients(&mut self) {
        let omega = 2.0 * PI * self.frequency / self.sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * self.q);
        let a = 10.0_f64.powf(self.gain_db / 40.0); // sqrt(10^(dB/20))

        let (b0, b1, b2, a0, a1, a2) = match self.filter_type {
            BiquadType::Peaking => {
                let b0 = 1.0 + alpha * a;
                let b1 = -2.0 * cos_omega;
                let b2 = 1.0 - alpha * a;
                let a0 = 1.0 + alpha / a;
                let a1 = -2.0 * cos_omega;
                let a2 = 1.0 - alpha / a;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::LowShelf => {
                let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
                let b0 = a * ((a + 1.0) - (a - 1.0) * cos_omega + two_sqrt_a_alpha);
                let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_omega);
                let b2 = a * ((a + 1.0) - (a - 1.0) * cos_omega - two_sqrt_a_alpha);
                let a0 = (a + 1.0) + (a - 1.0) * cos_omega + two_sqrt_a_alpha;
                let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_omega);
                let a2 = (a + 1.0) + (a - 1.0) * cos_omega - two_sqrt_a_alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::HighShelf => {
                let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
                let b0 = a * ((a + 1.0) + (a - 1.0) * cos_omega + two_sqrt_a_alpha);
                let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_omega);
                let b2 = a * ((a + 1.0) + (a - 1.0) * cos_omega - two_sqrt_a_alpha);
                let a0 = (a + 1.0) - (a - 1.0) * cos_omega + two_sqrt_a_alpha;
                let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_omega);
                let a2 = (a + 1.0) - (a - 1.0) * cos_omega - two_sqrt_a_alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::HighPass => {
                // High-pass filter - cuts frequencies below the cutoff
                let b0 = (1.0 + cos_omega) / 2.0;
                let b1 = -(1.0 + cos_omega);
                let b2 = (1.0 + cos_omega) / 2.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_omega;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::LowPass => {
                // Low-pass filter - cuts frequencies above the cutoff
                let b0 = (1.0 - cos_omega) / 2.0;
                let b1 = 1.0 - cos_omega;
                let b2 = (1.0 - cos_omega) / 2.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_omega;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
        };

        // Normalize coefficients
        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;
    }

    /// Process a single sample using Direct Form II Transposed
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        if !self.enabled {
            return input;
        }

        let input_f64 = input as f64;
        let output = self.b0 * input_f64 + self.z1;
        self.z1 = self.b1 * input_f64 - self.a1 * output + self.z2;
        self.z2 = self.b2 * input_f64 - self.a2 * output;
        output as f32
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

    /// Reset filter state
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// Set filter frequency and recalculate coefficients
    pub fn set_frequency(&mut self, frequency: f64) {
        self.frequency = frequency;
        self.calculate_coefficients();
    }

    /// Set filter gain and recalculate coefficients
    pub fn set_gain_db(&mut self, gain_db: f64) {
        self.gain_db = gain_db;
        self.calculate_coefficients();
    }

    /// Set filter Q factor and recalculate coefficients
    pub fn set_q(&mut self, q: f64) {
        self.q = q;
        self.calculate_coefficients();
    }

    /// Enable or disable the filter
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if filter is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get current frequency
    pub fn frequency(&self) -> f64 {
        self.frequency
    }

    /// Get current gain in dB
    pub fn gain_db(&self) -> f64 {
        self.gain_db
    }

    /// Get current Q factor
    pub fn q(&self) -> f64 {
        self.q
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biquad_passthrough_at_zero_gain() {
        let mut filter = Biquad::new(BiquadType::Peaking, 1000.0, 0.0, 1.0, 48000.0);

        // With 0 dB gain, a peaking filter should pass through
        let input = 0.5f32;
        let output = filter.process_sample(input);

        // Allow for small numerical error
        assert!((output - input).abs() < 0.01);
    }

    #[test]
    fn test_biquad_disabled() {
        let mut filter = Biquad::new(BiquadType::Peaking, 1000.0, 12.0, 1.0, 48000.0);
        filter.set_enabled(false);

        let input = 0.5f32;
        let output = filter.process_sample(input);

        assert_eq!(output, input);
    }

    #[test]
    fn test_biquad_boost() {
        let mut filter = Biquad::new(BiquadType::Peaking, 1000.0, 12.0, 1.0, 48000.0);

        // Feed a 1kHz sine wave and check for gain
        let sample_rate = 48000.0;
        let freq = 1000.0;
        let mut max_output = 0.0f32;

        for i in 0..4800 {
            let t = i as f64 / sample_rate;
            let input = (2.0 * PI * freq * t).sin() as f32 * 0.1;
            let output = filter.process_sample(input);
            max_output = max_output.max(output.abs());
        }

        // With 12 dB boost, output should be significantly higher than 0.1
        assert!(
            max_output > 0.2,
            "Expected boost, got max_output = {}",
            max_output
        );
    }
}
