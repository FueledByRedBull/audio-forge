//! Voice Activity Detection using Silero VAD
//!
//! Silero VAD is a lightweight neural network model for speech detection.
//! Model: https://github.com/snakers4/silero-vad
//!
//! CRITICAL: Silero VAD is a STATEFUL model requiring hidden state management.
//! Each inference call requires passing h/c states from the previous call.

#![cfg(feature = "vad")]

use ndarray::{Array1, Array2, Array3};
use ort::{
    session::builder::GraphOptimizationLevel,
    session::Session,
    value::TensorRef,
};
use std::env;
use std::path::PathBuf;
use thiserror::Error;

/// Enable debug output for VAD gate operations
#[cfg(debug_assertions)]
const GATE_DEBUG: bool = true;

#[cfg(not(debug_assertions))]
const GATE_DEBUG: bool = false;

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

/// Silero VAD sample rate (model trained at 16kHz)
const SILERO_SAMPLE_RATE: u32 = 16000;
/// Silero VAD window size (512 samples at 16kHz = 32ms)
const SILERO_WINDOW_SIZE: usize = 512;
/// LSTM hidden dimension
const LSTM_HIDDEN_DIM: usize = 64;
/// Number of LSTM layers
const LSTM_NUM_LAYERS: usize = 2;
/// Combined state dimension (h + c concatenated)
const LSTM_STATE_DIM: usize = LSTM_HIDDEN_DIM * 2; // 128

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
/// IMPORTANT: This is a stateful model. The h/c LSTM states must persist
/// between inference calls for accurate detection.
pub struct SileroVAD {
    /// ONNX Runtime session
    session: Session,
    /// Target sample rate (input audio rate, will be resampled to 16kHz)
    sample_rate: u32,
    /// Speech probability threshold (0.0-1.0)
    threshold: f32,
    /// Resampling ratio (silero_sr / target_sr)
    resample_ratio: f32,
    /// Internal buffer for accumulating samples
    buffer: Vec<f32>,
    /// Combined LSTM state (h and c concatenated) - shape [2, 1, 128]
    /// Silero VAD uses a single combined state instead of separate h/c
    state: Array3<f32>,
    /// Moving average of speech probability (for smoothing)
    smoothed_prob: f32,
    /// Smoothing factor for probability (0-1)
    smoothing: f32,
    /// Pre-gain applied to audio before VAD processing (boosts weak signals)
    pre_gain: f32,
}

impl SileroVAD {
    /// Find Silero VAD model file
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

        // 4. Check user data directories
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
            "Silero VAD model not found. Download silero_vad.onnx from \
             https://github.com/snakers4/silero-vad/tree/master/files \
             and place in ./models/silero_vad.onnx or set VAD_MODEL_PATH"
                .to_string(),
        ))
    }

    /// Create a new Silero VAD instance
    ///
    /// # Arguments
    /// * `sample_rate` - Audio sample rate (typically 48000)
    /// * `threshold` - Speech probability threshold (0.0-1.0), default 0.5
    pub fn new(sample_rate: u32, threshold: f32) -> Result<Self, VadError> {
        let model_path = Self::find_model_path()?;

        // Create ONNX Runtime session with ort 2.0 API
        let session = Session::builder()
            .map_err(|e| VadError::ModelLoadError(format!("Failed to create session builder: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| VadError::ModelLoadError(format!("Failed to set optimization level: {}", e)))?
            .commit_from_file(&model_path)
            .map_err(|e| VadError::ModelLoadError(format!("Failed to load model from {:?}: {}", model_path, e)))?;

        // Calculate resampling ratio
        let resample_ratio = SILERO_SAMPLE_RATE as f32 / sample_rate as f32;

        // Initialize combined LSTM state to zeros - shape [2, 1, 128]
        // Silero VAD uses a single combined state (h and c concatenated)
        let state = Array3::<f32>::zeros((LSTM_NUM_LAYERS, 1, LSTM_STATE_DIM));

        Ok(Self {
            session,
            sample_rate,
            threshold: threshold.clamp(0.0, 1.0),
            resample_ratio,
            buffer: Vec::with_capacity(SILERO_WINDOW_SIZE * 4),
            state,
            smoothed_prob: 0.0,
            smoothing: 0.5, // Faster smoothing (less lag)
            pre_gain: 1.0,  // Default: no gain boost
        })
    }

    /// Get required window size for this sample rate
    pub fn window_size(&self) -> usize {
        // We need SILERO_WINDOW_SIZE samples at 16kHz
        // At our sample rate: window_size * (sample_rate / 16000)
        ((SILERO_WINDOW_SIZE as f32 * self.sample_rate as f32) / SILERO_SAMPLE_RATE as f32).ceil()
            as usize
    }

    /// Process audio samples and return speech probability
    pub fn process(&mut self, samples: &[f32]) -> Result<f32, VadError> {
        self.buffer.extend(samples.iter());

        // Check if we have enough samples for inference
        if self.buffer.len() < self.window_size() {
            return Ok(self.smoothed_prob);
        }

        // Extract exactly what we need
        let input_samples: Vec<f32> = self.buffer.drain(..self.window_size()).collect();
        let original_len = input_samples.len();

        // Resample to 16kHz if needed
        let resampled = if self.sample_rate != SILERO_SAMPLE_RATE {
            self.resample(&input_samples)
        } else {
            input_samples
        };

        // Ensure we have exactly 512 samples
        let mut audio_512 = vec![0.0f32; SILERO_WINDOW_SIZE];
        let copy_len = resampled.len().min(SILERO_WINDOW_SIZE);
        audio_512[..copy_len].copy_from_slice(&resampled[..copy_len]);

        // Run inference
        let prob = self.run_inference(&audio_512)?;

        // Smooth the probability
        self.smoothed_prob = self.smoothing * prob + (1.0 - self.smoothing) * self.smoothed_prob;

        Ok(self.smoothed_prob)
    }

    /// Quick check if speech is detected
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

    /// Set pre-gain for VAD input (boosts weak signals)
    /// Default is 1.0 (no gain). Values > 1.0 boost the signal.
    pub fn set_pre_gain(&mut self, gain: f32) {
        self.pre_gain = gain.max(0.1); // Minimum gain to prevent division by zero
    }

    /// Get current pre-gain
    pub fn pre_gain(&self) -> f32 {
        self.pre_gain
    }

    /// Reset internal state (including LSTM states)
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.smoothed_prob = 0.0;
        // Reset combined LSTM state to zeros
        self.state = Array3::<f32>::zeros((LSTM_NUM_LAYERS, 1, LSTM_STATE_DIM));
    }

    /// Run ONNX inference with proper Silero VAD inputs
    fn run_inference(&mut self, audio: &[f32]) -> Result<f32, VadError> {
        // Silero VAD v4/v5 expects these inputs:
        // - "input": audio tensor, shape [1, 512] or [batch, samples]
        // - "sr": sample rate tensor, scalar (16000)
        // - "state": combined LSTM state, shape [2, 1, 128] (h and c concatenated)

        // Apply pre-gain to boost weak signals (helps with quiet microphones)
        let gain_applied = self.pre_gain != 1.0;
        let gained_audio: Vec<f32> = if gain_applied {
            audio.iter().map(|&x| x * self.pre_gain).collect()
        } else {
            audio.to_vec()
        };

        // DEBUG: Log audio statistics to diagnose VAD issue
        static DEBUG_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let count = DEBUG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if GATE_DEBUG && count < 5 {  // Only log first 5 times
            let max_val = gained_audio.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            let mean_val: f32 = gained_audio.iter().map(|&x| x.abs()).sum::<f32>() / gained_audio.len() as f32;
            eprintln!("[VAD-INPUT] samples={}, gain={:.2}, max={:.6}, mean={:.6}, first_10={:?}",
                gained_audio.len(), self.pre_gain, max_val, mean_val, &gained_audio[..10.min(gained_audio.len())]);
        }

        // Create audio input array - shape [1, 512]
        let audio_array = Array2::from_shape_vec((1, SILERO_WINDOW_SIZE), gained_audio)
            .map_err(|e| VadError::OnnxError(format!("Failed to create audio array: {}", e)))?;

        // Create sample rate array - scalar
        let sr_array = Array1::from_vec(vec![SILERO_SAMPLE_RATE as i64]);

        // Create TensorRefs using ort 2.0 API
        let audio_ref = TensorRef::from_array_view(&audio_array)
            .map_err(|e| VadError::OnnxError(format!("Failed to create audio tensor ref: {}", e)))?;

        let sr_ref = TensorRef::from_array_view(&sr_array)
            .map_err(|e| VadError::OnnxError(format!("Failed to create sr tensor ref: {}", e)))?;

        let state_ref = TensorRef::from_array_view(&self.state)
            .map_err(|e| VadError::OnnxError(format!("Failed to create state tensor ref: {}", e)))?;

        // Run inference with ort 2.0 API
        // Note: Silero VAD uses "state" not "h"/"c"
        let outputs = self.session
            .run(ort::inputs! {
                "input" => audio_ref,
                "sr" => sr_ref,
                "state" => state_ref
            })
            .map_err(|e| VadError::OnnxError(format!("Inference failed: {}", e)))?;

        // Extract probability output
        // Silero VAD returns: "output" (probability), "state_n" (new combined state)
        let prob_output = outputs
            .get("output")
            .ok_or_else(|| VadError::OnnxError("Missing 'output' in model outputs".to_string()))?;

        let (_shape, prob_data) = prob_output
            .try_extract_tensor::<f32>()
            .map_err(|e| VadError::OnnxError(format!("Failed to extract output tensor: {}", e)))?;

        let prob = prob_data
            .first()
            .copied()
            .ok_or_else(|| VadError::OnnxError("Output tensor is empty".to_string()))?;

        // Update combined LSTM state from state_n output
        if let Some(state_n_output) = outputs.get("state_n") {
            let (_shape, state_n_data) = state_n_output
                .try_extract_tensor::<f32>()
                .map_err(|e| VadError::OnnxError(format!("Failed to extract state_n tensor: {}", e)))?;

            // Copy data to self.state - shape [2, 1, 128]
            let total_elements = LSTM_NUM_LAYERS * 1 * LSTM_STATE_DIM;
            if state_n_data.len() >= total_elements {
                for i in 0..LSTM_NUM_LAYERS {
                    for j in 0..1 {
                        for k in 0..LSTM_STATE_DIM {
                            let flat_idx = i * 1 * LSTM_STATE_DIM + j * LSTM_STATE_DIM + k;
                            if flat_idx < state_n_data.len() {
                                self.state[[i, j, k]] = state_n_data[flat_idx];
                            }
                        }
                    }
                }
            }
        }

        Ok(prob)
    }

    /// Simple linear resampling from sample_rate to 16kHz
    fn resample(&self, input: &[f32]) -> Vec<f32> {
        if self.resample_ratio == 1.0 {
            return input.to_vec();
        }

        let output_len = (input.len() as f32 * self.resample_ratio).ceil() as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_pos = i as f32 / self.resample_ratio;
            let src_idx = src_pos as usize;
            let frac = src_pos - src_idx as f32;

            if src_idx + 1 < input.len() {
                // Linear interpolation
                let sample = input[src_idx] * (1.0 - frac) + input[src_idx + 1] * frac;
                output.push(sample);
            } else if src_idx < input.len() {
                output.push(input[src_idx]);
            } else {
                output.push(0.0);
            }
        }

        output
    }
}

/// VAD-based auto-gate controller
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
    /// Manual gate threshold used when auto-threshold is disabled (dB)
    manual_threshold_db: f32,
    /// Auto-threshold mode enabled
    auto_threshold_enabled: bool,
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
    /// Whether hold timer is currently active (prevents restarts)
    timer_running: bool,
    /// Previous raw gate state (for transition detection)
    prev_gate_open: bool,
    /// Counter for how long gate has been closed (for debounce)
    closed_counter_samples: f32,
    /// Minimum closed time before allowing timer restart (debounce, in ms)
    debounce_time_ms: f32,
    /// Sample rate for hold time calculation
    sample_rate: u32,
    /// Current VAD probability for metering
    current_probability: f32,
}

impl VadAutoGate {
    /// Create a new VAD auto-gate controller
    pub fn new(sample_rate: u32, vad_threshold: f32) -> Self {
        let vad = match SileroVAD::new(sample_rate, vad_threshold) {
            Ok(vad) => {
                eprintln!("VAD auto-gate enabled (Silero VAD loaded)");
                Some(vad)
            }
            Err(e) => {
                eprintln!("VAD auto-gate disabled: {}", e);
                eprintln!("  Download model from: https://github.com/snakers4/silero-vad/tree/master/files");
                eprintln!("  Place in: ./models/silero_vad.onnx");
                None
            }
        };

        let enabled = vad.is_some();

        Self {
            vad,
            noise_floor: -60.0,
            margin: 10.0,  // Increased from 6.0 to 10.0 dB for better noise rejection
            adaptation_rate: 0.001,  // ~15 second time constant at 48kHz/512 samples
            min_threshold: -80.0,  // Lowered from -50.0 to allow proper adaptation in quiet rooms
            max_threshold: -10.0,
            manual_threshold_db: -40.0,
            auto_threshold_enabled: false,  // Default to manual mode
            enabled,
            gate_mode: GateMode::ThresholdOnly,
            vad_threshold,
            hold_time_ms: 200.0,
            hold_timer: 0.0,
            timer_running: false,
            prev_gate_open: false,
            closed_counter_samples: 0.0,
            debounce_time_ms: 50.0, // 50ms debounce to prevent oscillation
            sample_rate,
            current_probability: 0.0,
        }
    }

    /// Process audio and return gate state
    pub fn process(&mut self, samples: &[f32]) -> (bool, f32) {
        if !self.enabled || self.vad.is_none() {
            // When VAD disabled, gate should be CLOSED (not open!)
            return (false, 0.0);
        }

        let vad = self.vad.as_mut().unwrap();

        let prob = match vad.process(samples) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("VAD processing error: {}", e);
                return (false, 0.0);
            }
        };

        self.current_probability = prob;

        let vad_speech_detected = prob > self.vad_threshold;

        // Update noise floor estimate during LOW-CONFIDENCE periods (pauses, breaths, background noise)
        // Uses probability threshold instead of binary speech detection to catch more update opportunities
        // Uses asymmetric rates: fast attack when noise increases, slow release when decreases

        // DEBUG: Log why adaptation is NOT happening
        static ADAPT_DEBUG_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let mut adapt_debug_count = 0usize;
        if GATE_DEBUG {
            adapt_debug_count = ADAPT_DEBUG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        if GATE_DEBUG && adapt_debug_count < 30 {
            if !self.auto_threshold_enabled {
                eprintln!("[ADAPT-DEBUG] Auto-threshold DISABLED - skipping noise floor update");
            } else if prob >= 0.4 {
                eprintln!("[ADAPT-DEBUG] prob {:.2} >= 0.4 - skipping (speech detected)", prob);
            }
        }

        if self.auto_threshold_enabled && prob < 0.4 {  // Low confidence: prob < 0.4 catches pauses, breaths
            let current_rms = compute_rms_db(samples);

            if GATE_DEBUG && adapt_debug_count < 30 {
                eprintln!("[ADAPT-DEBUG] Checking RMS: {:.1} dB (need > -100.0)", current_rms);
            }

            // Only adapt when there's actual audio activity (not dead silence)
            // Lower threshold (-100 dB) accommodates quiet rooms and sensitive mics
            if current_rms > -100.0 {
                let old_floor = self.noise_floor;

                // Asymmetric rates: fast response to noise increases, slow recovery from decreases
                // This prevents "pumping" when noise fluctuates around the threshold
                let rate = if current_rms > self.noise_floor {
                    0.02    // ATTACK: Fast when room gets louder (~0.5 seconds)
                } else {
                    0.005   // RELEASE: 4x slower when room gets quieter (~2 seconds)
                };
                // Exponential smoothing: new_val = rate * sample + (1 - rate) * old_val
                self.noise_floor = rate * current_rms + (1.0 - rate) * self.noise_floor;
                // Clamp to valid range
                self.noise_floor = self.noise_floor.clamp(-80.0, -20.0);

                // Log ALL updates when debugging (not just significant ones)
                if GATE_DEBUG && adapt_debug_count < 30 {
                    let auto_threshold = (self.noise_floor + self.margin).clamp(self.min_threshold, self.max_threshold);
                    eprintln!("[AUTO-THRESHOLD] Noise floor: {:.1} -> {:.1} dB (RMS: {:.1} dB, prob={:.2}, threshold={:.1} dB)",
                        old_floor, self.noise_floor, current_rms, prob, auto_threshold);
                } else if GATE_DEBUG && (self.noise_floor - old_floor).abs() > 0.1 {
                    // After debug period, only log significant changes
                    let auto_threshold = (self.noise_floor + self.margin).clamp(self.min_threshold, self.max_threshold);
                    eprintln!("[AUTO-THRESHOLD] Noise floor: {:.1} -> {:.1} dB (RMS: {:.1} dB, prob={:.2}, threshold={:.1} dB)",
                        old_floor, self.noise_floor, current_rms, prob, auto_threshold);
                }
            }
        }

        // Debug: Log VAD decision
        if GATE_DEBUG && (prob > 0.0 || prob < 0.01) {
            // Only log when probability is very low (to see oscillation)
            static DEBUG_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let count = DEBUG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count < 50 {  // Only log first 50 times to avoid spam
                eprintln!("[VAD-DEBUG] prob={:.6}, threshold={:.2}, prob > threshold = {}, vad_speech_detected={}",
                    prob, self.vad_threshold, prob > self.vad_threshold, vad_speech_detected);
            }
        }

        let level_above_threshold = self.level_above_threshold(samples);

        let gate_open = match self.gate_mode {
            GateMode::ThresholdOnly => level_above_threshold,
            GateMode::VadAssisted => {
                // In VadAssisted mode, explain WHY gate opened
                if GATE_DEBUG && (level_above_threshold || vad_speech_detected) {
                    let level_db = compute_rms_db(samples);
                    let threshold = (self.noise_floor + self.margin).clamp(self.min_threshold, self.max_threshold);
                    if level_above_threshold && vad_speech_detected {
                        eprintln!("[GATE] VAD-Assisted OPEN: BOTH level={:.1}dB>={:.1}dB AND VAD prob={:.2}>={:.2}",
                            level_db, threshold, prob, self.vad_threshold);
                    } else if level_above_threshold {
                        eprintln!("[GATE] VAD-Assisted OPEN: level={:.1}dB>={:.1}dB (VAD prob={:.2}<{:.2} - ignored)",
                            level_db, threshold, prob, self.vad_threshold);
                    } else if vad_speech_detected {
                        eprintln!("[GATE] VAD-Assisted OPEN: VAD prob={:.2}>={:.2} (level={:.1}dB<{:.1}dB - ignored)",
                            prob, self.vad_threshold, level_db, threshold);
                    }
                }
                level_above_threshold || vad_speech_detected
            },
            GateMode::VadOnly => {
                if GATE_DEBUG && vad_speech_detected {
                    eprintln!("[GATE] VAD-Only OPEN: VAD prob={:.2}>={:.2}",
                        prob, self.vad_threshold);
                }
                vad_speech_detected
            },
        };

        let smoothed_gate_open = self.apply_hold_time(gate_open, samples.len());

        // Debug: Log gate decision vs smoothed gate
        if GATE_DEBUG && gate_open != smoothed_gate_open {
            static DEBUG_COUNT2: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let count = DEBUG_COUNT2.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count < 20 {
                eprintln!("[VAD-HOLD] gate_open={}, smoothed_gate_open={}, prob={:.4}, timer_running={}",
                    gate_open, smoothed_gate_open, prob, self.timer_running);
            }
        }

        // Debug: show final gate state after hold time
        if GATE_DEBUG && smoothed_gate_open != gate_open {
            eprintln!("[GATE] HoldTime ACTIVE: raw_gate={}, held_gate={}",
                if gate_open { "OPEN" } else { "CLOSED" },
                if smoothed_gate_open { "OPEN" } else { "CLOSED" });
        }

        (smoothed_gate_open, prob)
    }

    fn level_above_threshold(&self, samples: &[f32]) -> bool {
        let threshold = if self.auto_threshold_enabled {
            // Auto mode: noise_floor + margin
            (self.noise_floor + self.margin).clamp(self.min_threshold, self.max_threshold)
        } else {
            // Manual mode: honor the user-configured gate threshold.
            self.manual_threshold_db
                .clamp(self.min_threshold, self.max_threshold)
        };
        let rms_db = compute_rms_db(samples);
        rms_db >= threshold
    }

    fn apply_hold_time(&mut self, gate_open: bool, num_samples: usize) -> bool {
        // STANDARD GATE BEHAVIOR:
        // If the raw gate is open, we reset the hold timer to max.
        // This keeps the gate open for 'hold_time' AFTER the signal drops.

        if gate_open {
            // Signal is valid (Open) -> Reset timer and keep running
            self.hold_timer = self.hold_time_ms / 1000.0 * self.sample_rate as f32;
            self.timer_running = true;
            self.closed_counter_samples = 0.0;
        } else {
            // Signal is invalid (Closed) -> Increment closed counter
            self.closed_counter_samples += num_samples as f32;
        }

        // Decrement timer if it is running
        if self.timer_running {
            self.hold_timer -= num_samples as f32;

            // If timer expires, stop holding
            if self.hold_timer <= 0.0 {
                self.hold_timer = 0.0;
                self.timer_running = false;
                if GATE_DEBUG && self.prev_gate_open {
                     eprintln!("[GATE] HoldTimer EXPIRED (Closed)");
                }
            }
        }

        self.prev_gate_open = gate_open;

        // Gate is open if raw signal is open OR timer is still running
        let final_gate_open = gate_open || self.timer_running;

        final_gate_open
    }

    pub fn is_available(&self) -> bool {
        self.vad.is_some()
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled && self.vad.is_some();
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn set_margin(&mut self, margin: f32) {
        self.margin = margin.clamp(0.0, 20.0);
    }

    pub fn margin(&self) -> f32 {
        self.margin
    }

    pub fn noise_floor(&self) -> f32 {
        self.noise_floor
    }

    /// Set manual threshold used when auto-threshold is disabled
    pub fn set_manual_threshold(&mut self, threshold_db: f32) {
        self.manual_threshold_db = threshold_db.clamp(self.min_threshold, self.max_threshold);
    }

    /// Get manual threshold used when auto-threshold is disabled
    pub fn manual_threshold(&self) -> f32 {
        self.manual_threshold_db
    }

    /// Enable/disable auto-threshold mode
    pub fn set_auto_threshold(&mut self, enabled: bool) {
        self.auto_threshold_enabled = enabled;
        if enabled {
            // Initialize noise floor from current RMS if needed
            if self.noise_floor <= -100.0 {
                self.noise_floor = -60.0;  // Reset to sensible default
            }
            let auto_threshold = (self.noise_floor + self.margin).clamp(self.min_threshold, self.max_threshold);
            eprintln!("[AUTO-THRESHOLD] ENABLED: Noise floor={:.1} dB, margin={:.1} dB, threshold={:.1} dB",
                self.noise_floor, self.margin, auto_threshold);
        } else {
            eprintln!("[AUTO-THRESHOLD] DISABLED: Returning to manual threshold mode");
        }
    }

    /// Check if auto-threshold is enabled
    pub fn auto_threshold_enabled(&self) -> bool {
        self.auto_threshold_enabled
    }

    pub fn reset(&mut self) {
        self.noise_floor = -60.0;
        self.hold_timer = 0.0;
        self.timer_running = false;
        self.prev_gate_open = false;
        self.closed_counter_samples = 0.0;
        self.current_probability = 0.0;
        if let Some(vad) = &mut self.vad {
            vad.reset();
        }
    }

    pub fn set_gate_mode(&mut self, mode: GateMode) {
        self.gate_mode = mode;
        // Don't reset hold timer state - let it expire naturally
        // This prevents the timer from being cleared when switching modes
    }

    pub fn gate_mode(&self) -> GateMode {
        self.gate_mode
    }

    pub fn set_vad_threshold(&mut self, threshold: f32) {
        self.vad_threshold = threshold.clamp(0.0, 1.0);
        if let Some(vad) = &mut self.vad {
            vad.set_threshold(threshold);
        }
    }

    pub fn vad_threshold(&self) -> f32 {
        self.vad_threshold
    }

    pub fn set_hold_time(&mut self, hold_ms: f32) {
        self.hold_time_ms = hold_ms.clamp(0.0, 500.0);
    }

    pub fn hold_time(&self) -> f32 {
        self.hold_time_ms
    }

    pub fn probability(&self) -> f32 {
        self.current_probability
    }

    /// Set pre-gain for VAD input (boosts weak signals)
    /// Default is 1.0 (no gain). Values > 1.0 boost the signal.
    /// This helps with quiet microphones where VAD can't detect speech.
    pub fn set_pre_gain(&mut self, gain: f32) {
        if let Some(vad) = &mut self.vad {
            vad.set_pre_gain(gain);
        }
    }

    /// Get current pre-gain
    pub fn pre_gain(&self) -> f32 {
        if let Some(vad) = &self.vad {
            vad.pre_gain()
        } else {
            1.0
        }
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
    fn test_rms_computation() {
        let silence = vec![0.0; 1000];
        assert!(compute_rms_db(&silence) < -100.0);

        let signal = vec![1.0; 1000];
        assert!((compute_rms_db(&signal) - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_gate_mode_enum() {
        assert_ne!(GateMode::ThresholdOnly, GateMode::VadAssisted);
        assert_ne!(GateMode::VadAssisted, GateMode::VadOnly);
    }
}
