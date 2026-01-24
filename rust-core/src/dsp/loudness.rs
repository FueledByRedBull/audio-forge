//! EBU R128 loudness measurement
//!
//! Implements EBU R128 standard for loudness measurement.
//! Uses momentary integration (400ms) for real-time control.
//!
//! # Usage
//!
//! ```rust
//! use mic_eq_core::dsp::loudness::LoudnessMeter;
//!
//! let mut meter = LoudnessMeter::new(48000);
//! meter.process(&audio_samples);
//! let lufs = meter.loudness_momentary();
//! ```

use ebur128::{EbuR128, Mode};
use thiserror::Error;

/// Errors related to loudness measurement
#[derive(Debug, Error)]
pub enum LoudnessError {
    #[error("Failed to initialize loudness meter: {0}")]
    InitError(String),

    #[error("Invalid sample rate: {0}")]
    InvalidSampleRate(u32),
}

/// EBU R128 loudness meter
///
/// Measures loudness according to EBU R128 standard.
/// Uses momentary integration (400ms) for real-time control.
pub struct LoudnessMeter {
    /// EBU R128 instance
    meter: EbuR128,
    /// Sample rate
    sample_rate: u32,
    /// Current momentary loudness (LUFS)
    current_lufs: f32,
}

impl LoudnessMeter {
    /// Create a new loudness meter
    ///
    /// # Arguments
    /// * `sample_rate` - Audio sample rate in Hz
    ///
    /// # Returns
    /// * `Ok(LoudnessMeter)` - Loudness meter ready for use
    /// * `Err(LoudnessError)` - If initialization fails
    pub fn new(sample_rate: u32) -> Result<Self, LoudnessError> {
        // Validate sample rate (EBU R128 supports common rates)
        if ![8000, 16000, 32000, 44100, 48000, 88200, 96000].contains(&sample_rate) {
            return Err(LoudnessError::InvalidSampleRate(sample_rate));
        }

        // Create EBU R128 meter with momentary mode
        // Use mode M (momentary) for real-time control
        let meter = EbuR128::new(1, Mode::M | Mode::HISTOGRAM, sample_rate)
            .map_err(|e| LoudnessError::InitError(e.to_string()))?;

        Ok(Self {
            meter,
            sample_rate,
            current_lufs: -100.0, // Start with very low loudness
        })
    }

    /// Process audio samples and update loudness measurement
    ///
    /// # Arguments
    /// * `samples` - Audio samples (interleaved mono)
    pub fn process(&mut self, samples: &[f32]) {
        // Process samples through EBU R128 meter
        if let Err(e) = self.meter.process(&[samples]) {
            eprintln!("Loudness meter error: {}", e);
            return;
        }

        // Update momentary loudness (400ms integration)
        match self.meter.loudness_momentary() {
            Ok(lufs) => {
                self.current_lufs = lufs;
            }
            Err(_) => {
                // Not enough data yet, keep previous value
            }
        }
    }

    /// Get current momentary loudness in LUFS
    ///
    /// Returns the loudness measured over the last 400ms.
    pub fn loudness_momentary(&self) -> f32 {
        self.current_lufs
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Reset loudness meter state
    pub fn reset(&mut self) {
        // Reset the EBU R128 meter
        // Note: ebur128 crate doesn't have explicit reset, so we recreate
        if let Ok(meter) = EbuR128::new(1, Mode::M | Mode::HISTOGRAM, self.sample_rate) {
            self.meter = meter;
        }
        self.current_lufs = -100.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loudness_meter_creation() {
        let meter = LoudnessMeter::new(48000);
        assert!(meter.is_ok());
    }

    #[test]
    fn test_loudness_meter_invalid_sample_rate() {
        let meter = LoudnessMeter::new(12345);
        assert!(meter.is_err());
    }

    #[test]
    fn test_loudness_meter_silence() {
        let mut meter = LoudnessMeter::new(48000).unwrap();
        let silence = vec![0.0f32; 48000]; // 1 second of silence
        meter.process(&silence);

        // Silence should have very low loudness
        let lufs = meter.loudness_momentary();
        assert!(lufs < -50.0, "Silence loudness too high: {}", lufs);
    }

    #[test]
    fn test_loudness_meter_tone() {
        let mut meter = LoudnessMeter::new(48000).unwrap();

        // Generate 1kHz tone at -20 dBFS
        let tone: Vec<f32> = (0..48000)
            .map(|i| {
                let phase = 2.0 * std::f32::consts::PI * 1000.0 * (i as f32 / 48000.0);
                0.1 * phase.sin() // -20 dBFS
            })
            .collect();

        meter.process(&tone);

        // -20 dBFS tone should be around -20 to -25 LUFS
        let lufs = meter.loudness_momentary();
        assert!(lufs > -30.0 && lufs < -10.0, "Tone loudness out of range: {}", lufs);
    }
}
