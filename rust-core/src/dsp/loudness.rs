//! EBU R128 loudness measurement
//!
//! Implements EBU R128 standard for loudness measurement.
//! Uses momentary integration (400ms) for real-time control.
//!
//! # Usage
//!
//! ```
//! use mic_eq_core::dsp::loudness::LoudnessMeter;
//!
//! let mut meter = LoudnessMeter::new(48000).unwrap();
//! let audio_samples = vec![0.0f32; 480]; // 10ms at 48kHz
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

    #[error("Invalid audio: {0}")]
    InvalidAudio(String),

    #[error("Loudness measurement failed: {0}")]
    MeasurementError(String),
}

fn validate_sample_rate(sample_rate: u32) -> Result<(), LoudnessError> {
    if ![8000, 16000, 32000, 44100, 48000, 88200, 96000].contains(&sample_rate) {
        return Err(LoudnessError::InvalidSampleRate(sample_rate));
    }
    Ok(())
}

/// Measure gated mono integrated loudness according to ITU-R BS.1770/EBU R128.
///
/// This is an offline helper. The library's absolute and relative gates omit
/// inactive material naturally, making it suitable for loudness-matching
/// speech renders without changing their internal dynamics.
pub fn integrated_loudness_lufs(samples: &[f32], sample_rate: u32) -> Result<f64, LoudnessError> {
    validate_sample_rate(sample_rate)?;
    if samples.is_empty() {
        return Err(LoudnessError::InvalidAudio(
            "at least one sample is required".to_string(),
        ));
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(LoudnessError::InvalidAudio(
            "samples must be finite".to_string(),
        ));
    }

    let mut meter = EbuR128::new(1, sample_rate, Mode::I | Mode::HISTOGRAM)
        .map_err(|error| LoudnessError::InitError(error.to_string()))?;
    meter
        .add_frames_f32(samples)
        .map_err(|error| LoudnessError::MeasurementError(error.to_string()))?;
    let loudness = meter
        .loudness_global()
        .map_err(|error| LoudnessError::MeasurementError(error.to_string()))?;
    if !loudness.is_finite() {
        return Err(LoudnessError::MeasurementError(
            "audio did not produce a finite gated loudness".to_string(),
        ));
    }
    Ok(loudness)
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
        validate_sample_rate(sample_rate)?;

        // Create EBU R128 meter with momentary mode
        // Use mode M (momentary) for real-time control
        let meter = EbuR128::new(1, sample_rate, Mode::M | Mode::HISTOGRAM)
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
        if let Err(e) = self.meter.add_frames_f32(samples) {
            eprintln!("Loudness meter error: {}", e);
            return;
        }

        // Update momentary loudness (400ms integration)
        match self.meter.loudness_momentary() {
            Ok(lufs) => {
                self.current_lufs = lufs as f32;
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
    pub fn reset(&mut self) -> Result<(), LoudnessError> {
        // Reset the EBU R128 meter
        // Note: ebur128 crate doesn't have explicit reset, so we recreate
        let meter = EbuR128::new(1, self.sample_rate, Mode::M | Mode::HISTOGRAM)
            .map_err(|e| LoudnessError::InitError(e.to_string()))?;
        self.meter = meter;
        self.current_lufs = -100.0;
        Ok(())
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
        assert!(
            lufs > -30.0 && lufs < -10.0,
            "Tone loudness out of range: {}",
            lufs
        );
    }

    #[test]
    fn test_loudness_meter_reset_restores_idle_state() {
        let mut meter = LoudnessMeter::new(48_000).unwrap();
        let tone = vec![0.1_f32; 48_000];
        meter.process(&tone);

        meter.reset().unwrap();

        assert_eq!(meter.loudness_momentary(), -100.0);
    }

    #[test]
    fn integrated_loudness_is_level_consistent_and_silence_gated() {
        let sample_rate = 48_000;
        // Use a programme-length fixture: the 400 ms BS.1770 blocks that
        // straddle a short clip's silence boundaries are intentionally part of
        // the measurement and can bias a two-second fixture by about 0.6 dB.
        let tone = (0..sample_rate * 8)
            .map(|index| {
                let phase =
                    2.0 * std::f32::consts::PI * 1_000.0 * index as f32 / sample_rate as f32;
                0.1 * phase.sin()
            })
            .collect::<Vec<_>>();
        let mut padded = vec![0.0_f32; sample_rate as usize];
        padded.extend_from_slice(&tone);
        padded.extend(std::iter::repeat_n(0.0_f32, sample_rate as usize));

        let tone_loudness = integrated_loudness_lufs(&tone, sample_rate).unwrap();
        let padded_loudness = integrated_loudness_lufs(&padded, sample_rate).unwrap();

        let difference = (tone_loudness - padded_loudness).abs();
        assert!(
            difference < 0.2,
            "silence-gated loudness changed by {difference:.3} dB "
        );
    }

    #[test]
    fn integrated_loudness_rejects_invalid_audio() {
        assert!(integrated_loudness_lufs(&[], 48_000).is_err());
        assert!(integrated_loudness_lufs(&[f32::NAN], 48_000).is_err());
        assert!(integrated_loudness_lufs(&[0.1], 12_345).is_err());
    }
}
