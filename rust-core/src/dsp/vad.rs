//! Voice Activity Detection using Silero VAD
//!
//! Silero VAD is a lightweight neural network model for speech detection.
//! Model: https://github.com/snakers4/silero-vad
//!
//! # Usage
//!
//! ```rust
//! use mic_eq_core::dsp::vad::SileroVAD;
//!
//! let mut vad = SileroVAD::new(48000, 0.5)?;
//! let speech_probability = vad.process(&audio_samples)?;
//! if speech_probability > 0.5 {
//!     // Speech detected
//! }
//! ```

#![cfg(feature = "vad")]

use ndarray::{Array1, Array3, Axis, IxDyn};
use ort::{Environment, ExecutionProvider, Session, Value};
use std::env;
use std::path::PathBuf;
use thiserror::Error;

/// Gate operating modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateMode {
    /// Traditional gate using only level threshold
    ThresholdOnly,
    /// Hybrid: gate opens when level exceeded OR speech detected
    VadAssisted,
    /// VAD-only: gate opens solely based on speech probability
    VadOnly,
}

/// Silero VAD sample rate (must resample to this)
const SILERO_SAMPLE_RATE: u32 = 16000;
/// Silero VAD window size (samples per inference)
const SILERO_WINDOW_SIZE: usize = 512;

/// Errors related to VAD processing
#[derive(Debug, Error)]
pub enum VadError {
    #[error("Failed to load VAD model: {0}")]
    ModelLoadError(String),

    #[error("VAD model not found at {0}")]
    ModelNotFound(String),

    #[error("Invalid input size: expected {expected} samples, got {actual}")]
    InvalidInputSize { expected: usize, actual: usize },

    #[error("ONNX Runtime error: {0}")]
    OnnxError(String),

    #[error("VAD feature not enabled (build with --features vad)")]
    NotEnabled,
}

/// Silero VAD for voice activity detection
///
/// # Example
///
/// ```rust
/// let mut vad = SileroVAD::new(48000, 0.5)?;
/// let is_speech = vad.is_speech(&audio_samples)?;
/// ```
pub struct SileroVAD {
    /// ONNX Runtime session
    session: Session,
    /// Target sample rate
    sample_rate: u32,
    /// Speech probability threshold (0.0-1.0)
    threshold: f32,
    /// Resampling ratio (target / silero_sr)
    resample_ratio: f32,
    /// Internal buffer for accumulating samples
    buffer: Vec<f32>,
    /// Moving average of speech probability (for smoothing)
    smoothed_prob: f32,
    /// Smoothing factor for probability (0-1)
    smoothing: f32,
}

impl SileroVAD {
    /// Find Silero VAD model file
    ///
    /// Search order:
    /// 1. Environment variable `VAD_MODEL_PATH`
    /// 2. `./models/silero_vad.onnx`
    /// 3. `../models/silero_vad.onnx`
    fn find_model_path() -> Result<PathBuf, VadError> {
        // 1. Check environment variable
        if let Ok(path) = env::var("VAD_MODEL_PATH") {
            let path_buf = PathBuf::from(&path);
            if path_buf.exists() {
                return Ok(path_buf);
            }
        }

        // 2. Check ./models/ directory
        let local_model = PathBuf::from("models/silero_vad.onnx");
        if local_model.exists() {
            return Ok(local_model);
        }

        // 3. Check ../models/ directory (for dev builds)
        let parent_model = PathBuf::from("../models/silero_vad.onnx");
        if parent_model.exists() {
            return Ok(parent_model);
        }

        // 4. Check user data directory (Linux/macOS)
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            if let Ok(home) = env::var("HOME") {
                let user_model =
                    PathBuf::from(format!("{}/.local/share/audioforge/silero_vad.onnx", home));
                if user_model.exists() {
                    return Ok(user_model);
                }
            }
        }

        // 5. Check AppData (Windows)
        #[cfg(target_os = "windows")]
        {
            if let Ok(appdata) = env::var("LOCALAPPDATA") {
                let user_model =
                    PathBuf::from(format!("{}/audioforge/models/silero_vad.onnx", appdata));
                if user_model.exists() {
                    return Ok(user_model);
                }
            }
        }

        Err(VadError::ModelNotFound(
            "Silero VAD model not found. Download from https://github.com/snakers4/silero-vad \
             and place in ./models/silero_vad.onnx or set VAD_MODEL_PATH"
                .to_string(),
        ))
    }

    /// Create a new Silero VAD instance
    ///
    /// # Arguments
    /// * `sample_rate` - Audio sample rate (typically 48000)
    /// * `threshold` - Speech probability threshold (0.0-1.0), default 0.5
    ///
    /// # Returns
    /// * `Ok(SileroVAD)` - VAD instance ready for use
    /// * `Err(VadError)` - If model cannot be loaded
    pub fn new(sample_rate: u32, threshold: f32) -> Result<Self, VadError> {
        // Find model file
        let model_path = Self::find_model_path()?;

        // Create ONNX Runtime environment
        let environment = Environment::builder()
            .with_execution_providers([ExecutionProvider::cpu()])
            .build()
            .map_err(|e| VadError::ModelLoadError(e.to_string()))?;

        // Load model
        let session = environment
            .new_session(&model_path)
            .map_err(|e| VadError::ModelLoadError(e.to_string()))?;

        // Calculate resampling ratio
        let resample_ratio = SILERO_SAMPLE_RATE as f32 / sample_rate as f32;

        Ok(Self {
            session,
            sample_rate,
            threshold: threshold.clamp(0.0, 1.0),
            resample_ratio,
            buffer: Vec::with_capacity(SILERO_WINDOW_SIZE * 2),
            smoothed_prob: 0.0,
            smoothing: 0.1, // Smooth probability changes
        })
    }

    /// Get required window size for this sample rate
    ///
    /// Returns the number of input samples needed for one VAD inference.
    /// This accounts for resampling to 16kHz.
    pub fn window_size(&self) -> usize {
        // We need SILERO_WINDOW_SIZE samples at 16kHz
        // At our sample rate, we need: window_size * (sample_rate / 16000)
        ((SILERO_WINDOW_SIZE as f32 * self.sample_rate as f32) / SILERO_SAMPLE_RATE as f32).ceil()
            as usize
    }

    /// Process audio samples and return speech probability
    ///
    /// This method accumulates samples until we have enough for one inference,
    /// then returns the speech probability (0.0 = silence, 1.0 = speech).
    ///
    /// # Arguments
    /// * `samples` - Audio samples at the configured sample rate
    ///
    /// # Returns
    /// * `Ok(f32)` - Speech probability (0.0-1.0)
    /// * `Err(VadError)` - If processing fails
    pub fn process(&mut self, samples: &[f32]) -> Result<f32, VadError> {
        self.buffer.extend_from_iter(samples.iter());

        // Check if we have enough samples for inference
        if self.buffer.len() < self.window_size() {
            // Not enough data yet, return current smoothed probability
            return Ok(self.smoothed_prob);
        }

        // Extract exactly what we need
        let input_samples: Vec<f32> = self.buffer.drain(..self.window_size()).collect();

        // Resample to 16kHz if needed (simple linear interpolation)
        let resampled = if self.sample_rate != SILERO_SAMPLE_RATE {
            self.resample(&input_samples)
        } else {
            input_samples
        };

        // Ensure we have exactly 512 samples
        let resampled = if resampled.len() > SILERO_WINDOW_SIZE {
            &resampled[..SILERO_WINDOW_SIZE]
        } else {
            // Pad with zeros if needed
            let mut padded = resampled.clone();
            padded.resize(SILERO_WINDOW_SIZE, 0.0);
            &padded
        };

        // Run inference
        let prob = self.run_inference(resampled)?;

        // Smooth the probability
        self.smoothed_prob = self.smoothing * prob + (1.0 - self.smoothing) * self.smoothed_prob;

        Ok(self.smoothed_prob)
    }

    /// Quick check if speech is detected
    ///
    /// # Arguments
    /// * `samples` - Audio samples
    ///
    /// # Returns
    /// * `Ok(bool)` - True if speech detected, false otherwise
    pub fn is_speech(&mut self, samples: &[f32]) -> Result<bool, VadError> {
        let prob = self.process(samples)?;
        Ok(prob > self.threshold)
    }

    /// Get current speech probability (smoothed)
    pub fn probability(&self) -> f32 {
        self.smoothed_prob
    }

    /// Set speech detection threshold
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }

    /// Get current threshold
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Reset internal state
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.smoothed_prob = 0.0;
    }

    /// Run ONNX inference on resampled audio
    fn run_inference(&self, audio: &[f32]) -> Result<f32, VadError> {
        // Silero VAD expects input shape [1, 512]
        let input_array = Array3::from_shape_vec((1, 1, audio.len()), audio.to_vec())
            .map_err(|e| VadError::OnnxError(e.to_string()))?;

        // Create ONNX value
        let input_value = Value::from_array(
            self.session.allocator(),
            IxDyn(&[1, 1, audio.len()]),
            input_array.view(),
        )
        .map_err(|e| VadError::OnnxError(e.to_string()))?;

        // Run inference
        let outputs = self
            .session
            .run(vec![input_value])
            .map_err(|e| VadError::OnnxError(e.to_string()))?;

        // Get output probability
        let output = outputs
            .first()
            .ok_or_else(|| VadError::OnnxError("No output from VAD model".to_string()))?;

        // Extract scalar value
        let prob: f32 = output
            .try_extract()
            .map_err(|e| VadError::OnnxError(e.to_string()))?;

        Ok(prob)
    }

    /// Simple linear resampling
    fn resample(&self, input: &[f32]) -> Vec<f32> {
        if self.resample_ratio == 1.0 {
            return input.to_vec();
        }

        let output_len = (input.len() as f32 * self.resample_ratio).ceil() as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_pos = (i as f32 / self.resample_ratio) as usize;
            if src_pos < input.len() {
                output.push(input[src_pos]);
            } else {
                output.push(0.0);
            }
        }

        output
    }
}

/// VAD-based auto-gate controller
///
/// Automatically adjusts noise gate threshold based on detected noise floor.
/// Supports three gate modes: ThresholdOnly, VadAssisted, and VadOnly.
pub struct VadAutoGate {
    /// Silero VAD instance
    vad: Option<SileroVAD>,
    /// Current noise floor estimate (dB)
    noise_floor: f32,
    /// Margin above noise floor for gate threshold (dB)
    margin: f32,
    /// Adaptation rate for noise floor (0-1)
    adaptation_rate: f32,
    /// Minimum gate threshold (dB)
    min_threshold: f32,
    /// Maximum gate threshold (dB)
    max_threshold: f32,
    /// Enabled state
    enabled: bool,
    /// Gate operating mode
    gate_mode: GateMode,
    /// Speech probability threshold (0.0-1.0) for VAD-based gate
    vad_threshold: f32,
    /// Gate hold time in milliseconds
    hold_time_ms: f32,
    /// Remaining hold time in samples
    hold_timer: f32,
    /// Last gate open/closed state
    last_gate_state: bool,
    /// Current VAD probability for metering
    current_probability: f32,
}

impl VadAutoGate {
    /// Create a new VAD auto-gate controller
    pub fn new(sample_rate: u32, vad_threshold: f32) -> Self {
        // Try to load VAD, fall back to disabled if not available
        let vad = match SileroVAD::new(sample_rate, vad_threshold) {
            Ok(vad) => {
                eprintln!("VAD auto-gate enabled (Silero VAD loaded)");
                Some(vad)
            }
            Err(e) => {
                eprintln!("VAD auto-gate disabled: {}", e);
                eprintln!("  Download model from: https://github.com/snakers4/silero-vad");
                eprintln!("  Place in: ./models/silero_vad.onnx");
                None
            }
        };

        Self {
            vad,
            noise_floor: -60.0,  // Start with low noise floor
            margin: 6.0,         // 6dB above noise floor
            adaptation_rate: 0.01,  // Slow adaptation
            min_threshold: -50.0,
            max_threshold: -10.0,
            enabled: vad.is_some(),
            gate_mode: GateMode::ThresholdOnly,
            vad_threshold: 0.5,
            hold_time_ms: 200.0,
            hold_timer: 0.0,
            last_gate_state: false,
            current_probability: 0.0,
        }
    }

    /// Process audio and update gate threshold
    ///
    /// # Returns
    /// * `(gate_open, probability)` - Gate open state and speech probability
    pub fn process(&mut self, samples: &[f32]) -> (bool, f32) {
        if !self.enabled || self.vad.is_none() {
            return (true, 0.0);  // Always open if disabled
        }

        let vad = self.vad.as_mut().unwrap();

        // Get speech probability
        let prob = match vad.process(samples) {
            Ok(p) => p,
            Err(_) => return (true, 0.0),
        };

        // Store for metering
        self.current_probability = prob;

        // Determine gate state based on mode
        let vad_speech_detected = prob > self.vad_threshold;
        let level_above_threshold = self.level_above_threshold(samples);

        let gate_open = match self.gate_mode {
            GateMode::ThresholdOnly => level_above_threshold,
            GateMode::VadAssisted => level_above_threshold || vad_speech_detected,
            GateMode::VadOnly => vad_speech_detected,
        };

        // Apply hold time logic
        let smoothed_gate_open = self.apply_hold_time(gate_open);

        (smoothed_gate_open, prob)
    }

    /// Check if level is above threshold (for ThresholdOnly mode)
    fn level_above_threshold(&self, samples: &[f32]) -> bool {
        // Calculate threshold from noise floor
        let threshold = (self.noise_floor + self.margin).clamp(self.min_threshold, self.max_threshold);

        // Check if RMS level exceeds threshold
        let rms_db = compute_rms_db(samples);
        rms_db >= threshold
    }

    /// Apply hold time to prevent gate chatter
    fn apply_hold_time(&mut self, gate_open: bool) -> bool {
        // 10ms smoothing - using IIR filter coefficient
        const SMOOTHING_COEFF: f32 = 0.95; // ~10ms time constant at 48kHz

        if gate_open {
            // Gate wants to open
            self.hold_timer = self.hold_time_ms / 1000.0 * 48000.0; // Convert ms to samples
            self.last_gate_state = true;
            true
        } else {
            // Gate wants to close
            if self.last_gate_state && self.hold_timer > 0.0 {
                // In hold period, keep gate open
                self.hold_timer -= 1.0;
                true
            } else {
                // Hold period expired, allow close
                self.last_gate_state = false;
                false
            }
        }
    }

    /// Process audio and update gate threshold (legacy interface for noise floor tracking)
    ///
    /// # Returns
    /// * `(is_speech, suggested_threshold_db)` - Speech detection and threshold
    pub fn process_with_noise_floor(&mut self, samples: &[f32]) -> (bool, f32) {
        let (gate_open, prob) = self.process(samples);

        // Update noise floor estimate when silence detected
        let vad = self.vad.as_ref().unwrap();
        let is_speech = prob > vad.threshold();

        if !is_speech {
            // Silence detected - update noise floor estimate
            let rms = compute_rms_db(samples);

            // Adapt noise floor slowly toward current level
            if rms < self.noise_floor + 10.0 {
                self.noise_floor = self.adaptation_rate * rms + (1.0 - self.adaptation_rate) * self.noise_floor;
            }
        }

        // Calculate threshold
        let threshold = (self.noise_floor + self.margin).clamp(self.min_threshold, self.max_threshold);

        (gate_open, threshold)
    }

    /// Check if VAD is available
    pub fn is_available(&self) -> bool {
        self.vad.is_some()
    }

    /// Enable or disable auto-gate
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled && self.vad.is_some();
    }

    /// Check if auto-gate is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Set margin above noise floor (dB)
    pub fn set_margin(&mut self, margin: f32) {
        self.margin = margin.clamp(0.0, 20.0);
    }

    /// Get current margin
    pub fn margin(&self) -> f32 {
        self.margin
    }

    /// Get current noise floor estimate (dB)
    pub fn noise_floor(&self) -> f32 {
        self.noise_floor
    }

    /// Reset noise floor estimate
    pub fn reset(&mut self) {
        self.noise_floor = -60.0;
        self.hold_timer = 0.0;
        self.last_gate_state = false;
        self.current_probability = 0.0;
        if let Some(vad) = &mut self.vad {
            vad.reset();
        }
    }

    /// Set gate mode
    pub fn set_gate_mode(&mut self, mode: GateMode) {
        self.gate_mode = mode;
    }

    /// Get current gate mode
    pub fn gate_mode(&self) -> GateMode {
        self.gate_mode
    }

    /// Set VAD probability threshold (0.0-1.0)
    pub fn set_vad_threshold(&mut self, threshold: f32) {
        self.vad_threshold = threshold.clamp(0.0, 1.0);
        if let Some(vad) = &mut self.vad {
            vad.set_threshold(threshold);
        }
    }

    /// Get current VAD probability threshold
    pub fn vad_threshold(&self) -> f32 {
        self.vad_threshold
    }

    /// Set gate hold time in milliseconds
    pub fn set_hold_time(&mut self, hold_ms: f32) {
        self.hold_time_ms = hold_ms.clamp(0.0, 500.0);
    }

    /// Get current hold time in milliseconds
    pub fn hold_time(&self) -> f32 {
        self.hold_time_ms
    }

    /// Get current VAD probability (for metering)
    pub fn probability(&self) -> f32 {
        self.current_probability
    }
}

/// Compute RMS level in dB
fn compute_rms_db(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return -120.0;
    }

    let rms_sq: f32 = samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32;
    let rms = rms_sq.sqrt();

    if rms < 1e-6 {
        -120.0
    } else {
        20.0 * rms.log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "vad")]
    fn test_vad_window_size() {
        let vad = SileroVAD::new(48000, 0.5);
        assert!(vad.is_ok());
        let vad = vad.unwrap();
        // At 48kHz, we need (512 * 48000 / 16000) = 1536 samples
        assert_eq!(vad.window_size(), 1536);
    }

    #[test]
    fn test_rms_computation() {
        let silence = vec![0.0; 1000];
        assert!(compute_rms_db(&silence) < -100.0);

        let signal = vec![1.0; 1000];
        assert!((compute_rms_db(&signal) - 0.0).abs() < 0.1);
    }
}
