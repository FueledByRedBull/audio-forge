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
}

fn validate_sample_rate(sample_rate: u32) -> Result<(), LoudnessError> {
    if ![8000, 16000, 32000, 44100, 48000, 88200, 96000].contains(&sample_rate) {
        return Err(LoudnessError::InvalidSampleRate(sample_rate));
    }
    Ok(())
}

/// Apply the fixed 48 kHz analysis K-weighting filters to an offline mono capture.
pub(crate) fn k_weighted_48k_inplace(samples: &mut [f64]) {
    // These normalized coefficients are shared by every analysis caller. Keep
    // the zero-state, DF-II-transposed arithmetic order of the SciPy reference.
    const STAGES: [([f64; 3], [f64; 3]); 2] = [
        (
            [1.53512485958697, -2.69169618940638, 1.19839281085285],
            [1.0, -1.69065929318241, 0.73248077421585],
        ),
        ([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621]),
    ];
    for (b, a) in STAGES {
        let mut delay = [0.0_f64; 2];
        for sample in samples.iter_mut() {
            let input = *sample;
            let output = delay[0] + b[0] * input;
            delay[0] = delay[1] + input * b[1] - output * a[1];
            delay[1] = input * b[2] - output * a[2];
            *sample = output;
        }
    }
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
    /// Blocks the underlying meter rejected; reported instead of printed
    /// because `process` runs inside the realtime chain.
    processing_failures: u64,
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
            processing_failures: 0,
        })
    }

    /// Process audio samples and update loudness measurement
    ///
    /// # Arguments
    /// * `samples` - Audio samples (interleaved mono)
    pub fn process(&mut self, samples: &[f32]) {
        // Process samples through EBU R128 meter
        if self.meter.add_frames_f32(samples).is_err() {
            self.processing_failures = self.processing_failures.saturating_add(1);
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

    /// Number of audio blocks the meter rejected since creation.
    pub fn processing_failures(&self) -> u64 {
        self.processing_failures
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Reset loudness meter state in place; the DSP loop calls this on a
    /// processing-path change, so it must not allocate.
    pub fn reset(&mut self) {
        self.meter.reset();
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

        meter.reset();

        assert_eq!(meter.loudness_momentary(), -100.0);
    }

    #[test]
    fn reset_clears_the_measurement_state_like_a_new_meter() {
        let loud = vec![0.5_f32; 48_000];
        let tone: Vec<f32> = (0..9_600)
            .map(|i| 0.1 * (2.0 * std::f32::consts::PI * 1_000.0 * i as f32 / 48_000.0).sin())
            .collect();
        let mut reused = LoudnessMeter::new(48_000).unwrap();
        reused.process(&loud);
        reused.reset();
        let mut fresh = LoudnessMeter::new(48_000).unwrap();
        for block in tone.chunks(480) {
            reused.process(block);
            fresh.process(block);
            assert_eq!(reused.loudness_momentary(), fresh.loudness_momentary());
        }
        assert!(fresh.loudness_momentary() > -30.0);
    }
}
