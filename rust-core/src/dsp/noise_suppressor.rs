//! Noise suppression trait and model selection
//!
//! This module provides a common interface for noise suppression models,
//! allowing runtime switching between RNNoise and DeepFilterNet.

use std::sync::atomic::AtomicU32;
use std::sync::Arc;

#[cfg(feature = "deepfilter")]
fn deepfilter_experimental_enabled() -> bool {
    std::env::var("AUDIOFORGE_ENABLE_DEEPFILTER")
        .map(|v| {
            let normalized = v.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true" || normalized == "yes"
        })
        .unwrap_or(false)
}

/// Noise suppression model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseModel {
    /// RNNoise: Low latency (~10ms), good quality
    RNNoise,
    /// DeepFilterNet Low Latency: Better quality than RNNoise, ~10ms latency (no lookahead)
    #[cfg(feature = "deepfilter")]
    DeepFilterNetLL,
    /// DeepFilterNet Standard: Best quality, ~40ms latency (2-frame lookahead)
    #[cfg(feature = "deepfilter")]
    DeepFilterNet,
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

    /// Get latency description for UI
    pub fn latency_description(&self) -> &'static str {
        match self {
            NoiseModel::RNNoise => "~10ms",
            #[cfg(feature = "deepfilter")]
            NoiseModel::DeepFilterNetLL => "~10ms",
            #[cfg(feature = "deepfilter")]
            NoiseModel::DeepFilterNet => "~40ms",
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
        #[allow(unused_mut)]
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
/// Both RNNoise and DeepFilterNet implement this trait, allowing
/// runtime model selection through the `NoiseSuppressionEngine` enum.
pub trait NoiseSuppressor: Send {
    /// Push input samples into the processor's input buffer
    fn push_samples(&mut self, samples: &[f32]);

    /// Process accumulated frames
    ///
    /// Call this after pushing samples. It will process as many
    /// complete frames as possible (480 samples per frame at 48kHz).
    fn process_frames(&mut self);

    /// Get available output samples count
    fn available_samples(&self) -> usize;

    /// Pop processed samples from output buffer
    fn pop_samples(&mut self, count: usize) -> Vec<f32>;

    /// Pop all available samples from output buffer
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

    /// Drain and return pending input samples without processing
    ///
    /// This is used when bypassing the suppressor to output raw audio.
    /// Returns pending samples that haven't been processed yet.
    fn drain_pending_input(&mut self) -> Vec<f32>;

    /// Get the model type
    fn model_type(&self) -> NoiseModel;

    /// Get expected latency in samples
    fn latency_samples(&self) -> usize;
}

/// Enum wrapper for runtime model selection
///
/// This allows switching between noise suppression models at runtime
/// while maintaining a common interface.
pub enum NoiseSuppressionEngine {
    RNNoise(super::RNNoiseProcessor),
    #[cfg(feature = "deepfilter")]
    DeepFilterLL(super::DeepFilterProcessor),
    #[cfg(feature = "deepfilter")]
    DeepFilter(super::DeepFilterProcessor),
}

impl NoiseSuppressionEngine {
    /// Create a new noise suppression engine with the specified model
    pub fn new(model: NoiseModel, strength: Arc<AtomicU32>) -> Self {
        match model {
            NoiseModel::RNNoise => {
                NoiseSuppressionEngine::RNNoise(super::RNNoiseProcessor::new(strength))
            }
            #[cfg(feature = "deepfilter")]
            NoiseModel::DeepFilterNetLL => {
                use super::deepfilter_ffi::DeepFilterModel;
                NoiseSuppressionEngine::DeepFilterLL(super::DeepFilterProcessor::new(
                    strength,
                    DeepFilterModel::LowLatency,
                ))
            }
            #[cfg(feature = "deepfilter")]
            NoiseModel::DeepFilterNet => {
                use super::deepfilter_ffi::DeepFilterModel;
                NoiseSuppressionEngine::DeepFilter(super::DeepFilterProcessor::new(
                    strength,
                    DeepFilterModel::Standard,
                ))
            }
        }
    }

    /// Get the current model type
    pub fn model_type(&self) -> NoiseModel {
        match self {
            NoiseSuppressionEngine::RNNoise(_) => NoiseModel::RNNoise,
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilterLL(_) => NoiseModel::DeepFilterNetLL,
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilter(_) => NoiseModel::DeepFilterNet,
        }
    }

    /// Whether the underlying backend is actually operational.
    ///
    /// For DeepFilter, construction may fall back to passthrough if df.dll/model
    /// are unavailable; this returns false in that case.
    pub fn backend_available(&self) -> bool {
        match self {
            NoiseSuppressionEngine::RNNoise(_) => true,
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilterLL(d) => d.is_ffi_available(),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilter(d) => d.is_ffi_available(),
        }
    }
}

// Implement NoiseSuppressor for the enum by delegating to inner type
impl NoiseSuppressor for NoiseSuppressionEngine {
    fn push_samples(&mut self, samples: &[f32]) {
        match self {
            NoiseSuppressionEngine::RNNoise(r) => r.push_samples(samples),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilterLL(d) => d.push_samples(samples),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilter(d) => d.push_samples(samples),
        }
    }

    fn process_frames(&mut self) {
        match self {
            NoiseSuppressionEngine::RNNoise(r) => r.process_frames(),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilterLL(d) => d.process_frames(),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilter(d) => d.process_frames(),
        }
    }

    fn available_samples(&self) -> usize {
        match self {
            NoiseSuppressionEngine::RNNoise(r) => r.available_samples(),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilterLL(d) => d.available_samples(),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilter(d) => d.available_samples(),
        }
    }

    fn pop_samples(&mut self, count: usize) -> Vec<f32> {
        match self {
            NoiseSuppressionEngine::RNNoise(r) => r.pop_samples(count),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilterLL(d) => d.pop_samples(count),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilter(d) => d.pop_samples(count),
        }
    }

    fn pop_all_samples(&mut self) -> Vec<f32> {
        match self {
            NoiseSuppressionEngine::RNNoise(r) => r.pop_all_samples(),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilterLL(d) => d.pop_all_samples(),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilter(d) => d.pop_all_samples(),
        }
    }

    fn set_strength(&self, value: f32) {
        match self {
            NoiseSuppressionEngine::RNNoise(r) => r.set_strength(value),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilterLL(d) => d.set_strength(value),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilter(d) => d.set_strength(value),
        }
    }

    fn get_strength(&self) -> f32 {
        match self {
            NoiseSuppressionEngine::RNNoise(r) => r.get_strength(),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilterLL(d) => d.get_strength(),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilter(d) => d.get_strength(),
        }
    }

    fn set_enabled(&mut self, enabled: bool) {
        match self {
            NoiseSuppressionEngine::RNNoise(r) => r.set_enabled(enabled),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilterLL(d) => d.set_enabled(enabled),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilter(d) => d.set_enabled(enabled),
        }
    }

    fn is_enabled(&self) -> bool {
        match self {
            NoiseSuppressionEngine::RNNoise(r) => r.is_enabled(),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilterLL(d) => d.is_enabled(),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilter(d) => d.is_enabled(),
        }
    }

    fn soft_reset(&mut self) {
        match self {
            NoiseSuppressionEngine::RNNoise(r) => r.soft_reset(),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilterLL(d) => d.soft_reset(),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilter(d) => d.soft_reset(),
        }
    }

    fn pending_input(&self) -> usize {
        match self {
            NoiseSuppressionEngine::RNNoise(r) => r.pending_input(),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilterLL(d) => d.pending_input(),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilter(d) => d.pending_input(),
        }
    }

    fn drain_pending_input(&mut self) -> Vec<f32> {
        match self {
            NoiseSuppressionEngine::RNNoise(r) => r.drain_pending_input(),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilterLL(d) => d.drain_pending_input(),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilter(d) => d.drain_pending_input(),
        }
    }

    fn model_type(&self) -> NoiseModel {
        match self {
            NoiseSuppressionEngine::RNNoise(_) => NoiseModel::RNNoise,
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilterLL(_) => NoiseModel::DeepFilterNetLL,
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilter(_) => NoiseModel::DeepFilterNet,
        }
    }

    fn latency_samples(&self) -> usize {
        match self {
            NoiseSuppressionEngine::RNNoise(r) => r.latency_samples(),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilterLL(d) => d.latency_samples(),
            #[cfg(feature = "deepfilter")]
            NoiseSuppressionEngine::DeepFilter(d) => d.latency_samples(),
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
