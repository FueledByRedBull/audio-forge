//! Noise suppression trait and model selection
//!
//! This module provides a common interface for noise suppression models,
//! allowing runtime switching between RNNoise and DeepFilterNet.

use std::sync::atomic::AtomicU32;
use std::sync::Arc;

#[cfg(feature = "deepfilter")]
pub(crate) fn deepfilter_experimental_enabled() -> bool {
    std::env::var("AUDIOFORGE_ENABLE_DEEPFILTER")
        .map(|v| {
            let normalized = v.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true" || normalized == "yes"
        })
        .unwrap_or(false)
}

/// Noise suppression model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NoiseModel {
    /// RNNoise: Low latency (~10ms), good quality
    RNNoise = 0,
    /// DeepFilterNet Low Latency: Better quality than RNNoise, ~10ms latency (no lookahead)
    #[cfg(feature = "deepfilter")]
    DeepFilterNetLL = 1,
    /// DeepFilterNet Standard: stronger cleanup, ~30ms latency (2-frame lookahead)
    #[cfg(feature = "deepfilter")]
    DeepFilterNet = 2,
}

impl NoiseModel {
    /// Get display name for UI
    pub fn display_name(&self) -> &'static str {
        match self {
            NoiseModel::RNNoise => "RNNoise (Low Latency)",
            #[cfg(feature = "deepfilter")]
            NoiseModel::DeepFilterNetLL => "DeepFilterNet LL (Fast)",
            #[cfg(feature = "deepfilter")]
            NoiseModel::DeepFilterNet => "DeepFilterNet (Best Quality)",
        }
    }

    /// Get short identifier for presets/config
    pub fn id(&self) -> &'static str {
        match self {
            NoiseModel::RNNoise => "rnnoise",
            #[cfg(feature = "deepfilter")]
            NoiseModel::DeepFilterNetLL => "deepfilter-ll",
            #[cfg(feature = "deepfilter")]
            NoiseModel::DeepFilterNet => "deepfilter",
        }
    }

    /// Parse model from string identifier
    pub fn from_id(id: &str) -> Option<Self> {
        match id.to_lowercase().as_str() {
            "rnnoise" => Some(NoiseModel::RNNoise),
            #[cfg(feature = "deepfilter")]
            "deepfilter-ll" | "deepfilterll" => Some(NoiseModel::DeepFilterNetLL),
            #[cfg(feature = "deepfilter")]
            "deepfilter" | "deepfilternet" => Some(NoiseModel::DeepFilterNet),
            _ => None,
        }
    }

    /// Get all available models
    pub fn available() -> Vec<NoiseModel> {
        #[cfg_attr(not(feature = "deepfilter"), allow(unused_mut))]
        let mut models = vec![NoiseModel::RNNoise];
        #[cfg(feature = "deepfilter")]
        {
            // DeepFilter uses upstream C FFI and can hard-crash on some systems.
            // Keep it opt-in so RNNoise remains the safe default.
            if deepfilter_experimental_enabled() {
                models.push(NoiseModel::DeepFilterNetLL);
                models.push(NoiseModel::DeepFilterNet);
            }
        }
        models
    }
}

/// Common interface for noise suppression models
///
/// Both RNNoise and DeepFilterNet implement this trait, allowing runtime model
/// selection through a boxed trait object.
pub trait NoiseSuppressor: Send {
    /// Push input samples into the processor's input buffer.
    ///
    /// Returns the number of samples accepted by the fixed input buffer.
    fn push_samples(&mut self, samples: &[f32]) -> usize;

    /// Process accumulated frames
    ///
    /// Call this after pushing samples. It will process as many
    /// complete frames as possible (480 samples per frame at 48kHz).
    fn process_frames(&mut self);

    /// Get available output samples count
    fn available_samples(&self) -> usize;

    /// Pop processed samples from output buffer.
    ///
    /// Non-RT convenience API: this allocates the returned `Vec`. CPAL
    /// callbacks and the DSP processing loop must use `pop_samples_into`.
    fn pop_samples(&mut self, count: usize) -> Vec<f32>;

    /// Pop processed samples into caller-provided buffer.
    ///
    /// Returns the number of samples written into `buffer`.
    fn pop_samples_into(&mut self, buffer: &mut [f32]) -> usize;

    /// Pop all available samples from output buffer.
    ///
    /// Non-RT convenience API: this allocates the returned `Vec`. CPAL
    /// callbacks and the DSP processing loop must use caller-provided buffers.
    fn pop_all_samples(&mut self) -> Vec<f32>;

    /// Set wet/dry mix strength (0.0 = dry/original, 1.0 = wet/processed)
    fn set_strength(&self, value: f32);

    /// Get current wet/dry mix strength
    fn get_strength(&self) -> f32;

    /// Enable or disable processing (disabled = passthrough)
    fn set_enabled(&mut self, enabled: bool);

    /// Check if processing is enabled
    fn is_enabled(&self) -> bool;

    /// Soft reset: clear buffers without resetting model state
    ///
    /// Preferred over hard reset as it preserves learned noise profile.
    fn soft_reset(&mut self);

    /// Get pending input samples count (waiting for frame completion)
    fn pending_input(&self) -> usize;

    /// Drain and return pending input samples without processing.
    ///
    /// This is used when bypassing the suppressor to output raw audio.
    /// Returns pending samples that haven't been processed yet.
    ///
    /// Non-RT convenience API: this allocates the returned `Vec`.
    fn drain_pending_input(&mut self) -> Vec<f32>;

    /// Get the model type
    fn model_type(&self) -> NoiseModel;

    /// Get expected latency in samples
    fn latency_samples(&self) -> usize;

    /// Whether the underlying backend is operational.
    fn backend_available(&self) -> bool;

    /// Backend load/runtime error, when one is available.
    fn backend_error(&self) -> Option<&str>;

    /// Whether the backend permanently failed and is in passthrough fallback.
    fn backend_failed(&self) -> bool;
}

/// Runtime-selected noise suppressor. The box is created off the RT path and
/// moved through the existing command/retirement queues without dropping it in
/// the audio callback.
pub type NoiseSuppressionEngine = Box<dyn NoiseSuppressor>;

/// Create a runtime-selected noise suppressor.
pub fn new_noise_suppression_engine(
    model: NoiseModel,
    strength: Arc<AtomicU32>,
) -> NoiseSuppressionEngine {
    match model {
        NoiseModel::RNNoise => Box::new(super::RNNoiseProcessor::new(strength)),
        #[cfg(feature = "deepfilter")]
        NoiseModel::DeepFilterNetLL => {
            use super::deepfilter_ffi::DeepFilterModel;
            Box::new(super::DeepFilterProcessor::new(
                strength,
                DeepFilterModel::LowLatency,
            ))
        }
        #[cfg(feature = "deepfilter")]
        NoiseModel::DeepFilterNet => {
            use super::deepfilter_ffi::DeepFilterModel;
            Box::new(super::DeepFilterProcessor::new(
                strength,
                DeepFilterModel::Standard,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_model_display_names() {
        assert_eq!(NoiseModel::RNNoise.display_name(), "RNNoise (Low Latency)");
        assert_eq!(NoiseModel::RNNoise.id(), "rnnoise");
    }

    #[test]
    fn test_noise_model_from_id() {
        assert_eq!(NoiseModel::from_id("rnnoise"), Some(NoiseModel::RNNoise));
        assert_eq!(NoiseModel::from_id("RNNOISE"), Some(NoiseModel::RNNoise));
        assert_eq!(NoiseModel::from_id("invalid"), None);
    }

    #[test]
    fn test_available_models() {
        let models = NoiseModel::available();
        assert!(models.contains(&NoiseModel::RNNoise));
    }
}
