//! 10-Band Parametric Equalizer
//!
//! Band configuration:
//! - Band 0: Low shelf (80 Hz)
//! - Bands 1-8: Peaking (160 Hz - 12 kHz)
//! - Band 9: High shelf (16 kHz)

use super::biquad::{Biquad, BiquadType};

/// Default EQ band frequencies (Hz)
pub const DEFAULT_FREQUENCIES: [f64; 10] = [
    80.0,    // Band 0: Low shelf
    160.0,   // Band 1: Peaking
    320.0,   // Band 2: Peaking
    640.0,   // Band 3: Peaking
    1280.0,  // Band 4: Peaking
    2500.0,  // Band 5: Peaking
    5000.0,  // Band 6: Peaking
    8000.0,  // Band 7: Peaking
    12000.0, // Band 8: Peaking
    16000.0, // Band 9: High shelf
];

/// Default Q factor for peaking bands (~1 octave bandwidth)
pub const DEFAULT_Q: f64 = 1.41;

/// Number of EQ bands
pub const NUM_BANDS: usize = 10;

/// 10-Band Parametric Equalizer
///
/// Provides professional-quality equalization with configurable
/// frequency, gain, and Q for each band.
pub struct ParametricEQ {
    bands: [Biquad; NUM_BANDS],
    enabled: bool,
    sample_rate: f64,
}

impl ParametricEQ {
    /// Create a new 10-band parametric EQ
    pub fn new(sample_rate: f64) -> Self {
        let bands = std::array::from_fn(|i| {
            let filter_type = match i {
                0 => BiquadType::LowShelf,
                9 => BiquadType::HighShelf,
                _ => BiquadType::Peaking,
            };
            Biquad::new(filter_type, DEFAULT_FREQUENCIES[i], 0.0, DEFAULT_Q, sample_rate)
        });

        Self {
            bands,
            enabled: true,
            sample_rate,
        }
    }

    /// Process a block of samples in-place
    pub fn process_block_inplace(&mut self, buffer: &mut [f32]) {
        if !self.enabled {
            return;
        }

        for band in &mut self.bands {
            band.process_block_inplace(buffer);
        }
    }

    /// Process a single sample through all bands
    #[inline]
    pub fn process_sample(&mut self, mut sample: f32) -> f32 {
        if !self.enabled {
            return sample;
        }

        for band in &mut self.bands {
            sample = band.process_sample(sample);
        }
        sample
    }

    /// Reset all filter states
    pub fn reset(&mut self) {
        for band in &mut self.bands {
            band.reset();
        }
    }

    /// Set gain for a specific band (0-9)
    ///
    /// # Arguments
    /// * `band_index` - Band index (0-9)
    /// * `gain_db` - Gain in dB (typically -12 to +12)
    pub fn set_band_gain(&mut self, band_index: usize, gain_db: f64) {
        if band_index < NUM_BANDS {
            self.bands[band_index].set_gain_db(gain_db);
        }
    }

    /// Set frequency for a specific band
    ///
    /// # Arguments
    /// * `band_index` - Band index (0-9)
    /// * `frequency` - Center frequency in Hz
    pub fn set_band_frequency(&mut self, band_index: usize, frequency: f64) {
        if band_index < NUM_BANDS {
            self.bands[band_index].set_frequency(frequency);
        }
    }

    /// Set Q factor for a specific band
    ///
    /// # Arguments
    /// * `band_index` - Band index (0-9)
    /// * `q` - Q factor (higher = narrower bandwidth)
    pub fn set_band_q(&mut self, band_index: usize, q: f64) {
        if band_index < NUM_BANDS {
            self.bands[band_index].set_q(q);
        }
    }

    /// Enable or disable a specific band
    pub fn set_band_enabled(&mut self, band_index: usize, enabled: bool) {
        if band_index < NUM_BANDS {
            self.bands[band_index].set_enabled(enabled);
        }
    }

    /// Check if a specific band is enabled
    pub fn is_band_enabled(&self, band_index: usize) -> bool {
        if band_index < NUM_BANDS {
            self.bands[band_index].is_enabled()
        } else {
            false
        }
    }

    /// Enable or disable the entire EQ
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if EQ is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get band parameters (frequency, gain_db, q)
    pub fn get_band_params(&self, band_index: usize) -> Option<(f64, f64, f64)> {
        if band_index < NUM_BANDS {
            let band = &self.bands[band_index];
            Some((band.frequency(), band.gain_db(), band.q()))
        } else {
            None
        }
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Get the number of bands
    pub fn num_bands(&self) -> usize {
        NUM_BANDS
    }

    /// Get default frequency for a band
    pub fn default_frequency(band_index: usize) -> Option<f64> {
        if band_index < NUM_BANDS {
            Some(DEFAULT_FREQUENCIES[band_index])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eq_flat_response() {
        let mut eq = ParametricEQ::new(48000.0);

        // With all gains at 0 dB, EQ should be nearly transparent
        let input = 0.5f32;
        let output = eq.process_sample(input);

        // Allow for small numerical accumulation from 10 filters
        assert!((output - input).abs() < 0.05);
    }

    #[test]
    fn test_eq_disabled() {
        let mut eq = ParametricEQ::new(48000.0);
        eq.set_band_gain(0, 12.0); // Boost low shelf
        eq.set_enabled(false);

        let input = 0.5f32;
        let output = eq.process_sample(input);

        assert_eq!(output, input);
    }

    #[test]
    fn test_eq_band_params() {
        let mut eq = ParametricEQ::new(48000.0);

        // Set custom parameters
        eq.set_band_frequency(5, 3000.0);
        eq.set_band_gain(5, 6.0);
        eq.set_band_q(5, 2.0);

        let params = eq.get_band_params(5).unwrap();
        assert!((params.0 - 3000.0).abs() < 0.001);
        assert!((params.1 - 6.0).abs() < 0.001);
        assert!((params.2 - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_eq_default_frequencies() {
        assert_eq!(ParametricEQ::default_frequency(0), Some(80.0));
        assert_eq!(ParametricEQ::default_frequency(9), Some(16000.0));
        assert_eq!(ParametricEQ::default_frequency(10), None);
    }
}
