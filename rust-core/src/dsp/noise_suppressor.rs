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
}

/// Per-sample gate state, captured after the original gate has applied its gain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GateControl {
    pub gain: f32,
    pub open: bool,
}

impl GateControl {
    pub const BYPASS: Self = Self {
        gain: 1.0,
        open: false,
    };
}

impl Default for GateControl {
    fn default() -> Self {
        Self::BYPASS
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct ControlledSample {
    pub sample: f32,
    pub gate: GateControl,
}

/// Uses only the model's existing delay; it never queues or delays audio.
pub(crate) struct GateCompensation {
    history: [GateControl; 1440],
    delay: usize,
    position: usize,
    filled: usize,
    permission_samples: usize,
}

impl GateCompensation {
    pub fn new(delay: usize) -> Self {
        assert!(delay <= 1440);
        Self {
            history: [GateControl::BYPASS; 1440],
            delay,
            position: 0,
            filled: 0,
            permission_samples: 0,
        }
    }

    pub fn reset(&mut self) {
        self.history.fill(GateControl::BYPASS);
        self.position = 0;
        self.filled = 0;
        self.permission_samples = 0;
    }

    /// The current input control is the future control of the emitted wet sample.
    pub fn next_ratio(&mut self, future: GateControl) -> f32 {
        if self.delay == 0 {
            return 1.0;
        }
        let current = self.history[self.position];
        self.history[self.position] = future;
        self.position = (self.position + 1) % self.delay;
        if self.filled < self.delay {
            self.filled += 1;
            return 1.0;
        }

        // The floor is the gate's existing 36 dB attenuation range. Emit the
        // weight before advancing it, so neither permission edge steps the ratio.
        const GAIN_FLOOR: f32 = 0.015_848_933;
        let raw_ratio = (current.gain.max(future.gain) / current.gain.max(GAIN_FLOOR)).max(1.0);
        let weight = self.permission_samples as f32 / self.delay as f32;
        let ratio = 1.0 + weight * (raw_ratio - 1.0);
        if future.open {
            self.permission_samples = (self.permission_samples + 1).min(self.delay);
        } else {
            self.permission_samples = self.permission_samples.saturating_sub(1);
        }
        ratio
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

    /// Pop processed samples into caller-provided buffer.
    ///
    /// Returns the number of samples written into `buffer`.
    fn pop_samples_into(&mut self, buffer: &mut [f32]) -> usize;

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

    /// Number of frames successfully processed by an external inference backend.
    fn successful_inference_frames(&self) -> u64 {
        0
    }
}

/// Runtime-selected noise suppressor. The box is created off the RT path and
/// moved through the existing command/retirement queues without dropping it in
/// the audio callback.
pub type NoiseSuppressionEngine = Box<dyn NoiseSuppressor>;

pub(crate) trait ControlledNoiseSuppressor: NoiseSuppressor {
    fn push_controlled_samples(&mut self, samples: &[f32], controls: &[GateControl]) -> usize;
}

pub(crate) type ControlledNoiseSuppressionEngine = Box<dyn ControlledNoiseSuppressor>;

/// Create a runtime-selected noise suppressor.
pub fn new_noise_suppression_engine(
    model: NoiseModel,
    strength: Arc<AtomicU32>,
) -> NoiseSuppressionEngine {
    new_controlled_noise_suppression_engine(model, strength)
}

pub(crate) fn new_controlled_noise_suppression_engine(
    model: NoiseModel,
    strength: Arc<AtomicU32>,
) -> ControlledNoiseSuppressionEngine {
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
mod gate_compensation_tests {
    use super::{GateCompensation, GateControl};

    fn controls(length: usize) -> Vec<GateControl> {
        (0..length)
            .map(|index| GateControl {
                gain: 0.015_848_933 + 0.95 * (0.5 + 0.5 * (index as f32 * 0.007).sin()),
                open: (index / 137) % 3 != 0,
            })
            .collect()
    }

    fn reference(input: &[GateControl], delay: usize) -> Vec<f32> {
        let mut counter = 0usize;
        let mut ratios = vec![1.0; input.len()];
        if delay == 0 {
            return ratios;
        }
        for index in delay..input.len() {
            let current = input[index - delay];
            let future = input[index];
            let raw = (current.gain.max(future.gain) / current.gain.max(0.015_848_933)).max(1.0);
            ratios[index] = 1.0 + (counter as f32 / delay as f32) * (raw - 1.0);
            counter = if future.open {
                (counter + 1).min(delay)
            } else {
                counter.saturating_sub(1)
            };
        }
        ratios
    }

    #[test]
    fn matches_aligned_v3_equation_for_arbitrary_input_partitions() {
        let input = controls(8003);
        for delay in [0, 1, 480, 1440] {
            let expected = reference(&input, delay);
            for partition in [1, 127, 480, 997] {
                let mut state = GateCompensation::new(delay);
                let actual: Vec<_> = input
                    .chunks(partition)
                    .flat_map(|chunk| {
                        chunk
                            .iter()
                            .map(|&control| state.next_ratio(control))
                            .collect::<Vec<_>>()
                    })
                    .collect();
                for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        actual, expected,
                        "delay={delay}, partition={partition}, sample={index}"
                    );
                }
            }
        }
    }

    #[test]
    fn permission_edges_emit_the_previous_weight() {
        let delay = 480;
        let mut state = GateCompensation::new(delay);
        for _ in 0..delay {
            assert_eq!(
                state.next_ratio(GateControl {
                    gain: 0.02,
                    open: false
                }),
                1.0
            );
        }
        assert_eq!(
            state.next_ratio(GateControl {
                gain: 0.8,
                open: true
            }),
            1.0
        );
        let second = state.next_ratio(GateControl {
            gain: 0.8,
            open: true,
        });
        assert_eq!(second, 1.0 + 39.0 / 480.0);
        let closing = state.next_ratio(GateControl {
            gain: 0.8,
            open: false,
        });
        assert_eq!(closing, 1.0 + 2.0 / 480.0 * 39.0);
        let closed = state.next_ratio(GateControl {
            gain: 0.8,
            open: false,
        });
        assert_eq!(closed, second);
        assert_eq!(
            state.next_ratio(GateControl {
                gain: 0.8,
                open: false
            }),
            1.0
        );
    }

    #[test]
    fn reset_does_not_advance_permission_during_invalid_prefix() {
        let input = controls(1900);
        let mut used = GateCompensation::new(480);
        for control in &input {
            used.next_ratio(*control);
        }
        used.reset();
        let mut fresh = GateCompensation::new(480);
        for control in &input {
            assert_eq!(used.next_ratio(*control), fresh.next_ratio(*control));
        }
    }

    #[test]
    fn bypass_zero_delay_and_nonopening_controls_remain_identity() {
        for delay in [0, 480, 1440] {
            let mut bypass = GateCompensation::new(delay);
            let mut closed = GateCompensation::new(delay);
            for control in controls(5000) {
                assert_eq!(bypass.next_ratio(GateControl::BYPASS), 1.0);
                assert_eq!(
                    closed.next_ratio(GateControl {
                        open: false,
                        ..control
                    }),
                    1.0
                );
            }
        }
        let mut no_delay = GateCompensation::new(0);
        for control in controls(5000) {
            assert_eq!(no_delay.next_ratio(control), 1.0);
        }
    }

    #[test]
    fn ratio_has_the_existing_attenuation_bound() {
        let mut state = GateCompensation::new(480);
        for index in 0..3000 {
            let gain = if index % 997 < 600 { 0.0 } else { 1.0 };
            let ratio = state.next_ratio(GateControl { gain, open: true });
            assert!(ratio.is_finite() && (1.0..=1.0 / 0.015_848_933).contains(&ratio));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models() -> Vec<NoiseModel> {
        vec![
            NoiseModel::RNNoise,
            #[cfg(feature = "deepfilter")]
            NoiseModel::DeepFilterNetLL,
            #[cfg(feature = "deepfilter")]
            NoiseModel::DeepFilterNet,
        ]
    }

    #[test]
    fn controlled_factory_upcasts_and_preserves_public_calls() {
        for model in models() {
            let mut public: NoiseSuppressionEngine = new_controlled_noise_suppression_engine(
                model,
                Arc::new(AtomicU32::new(0.37_f32.to_bits())),
            );
            let mut controlled = new_controlled_noise_suppression_engine(
                model,
                Arc::new(AtomicU32::new(0.37_f32.to_bits())),
            );
            assert_eq!(public.backend_available(), controlled.backend_available());
            for frame in 0..12 {
                if frame == 4 {
                    public.soft_reset();
                    controlled.soft_reset();
                }
                public.set_enabled(frame != 7);
                controlled.set_enabled(frame != 7);
                let raw = std::array::from_fn::<_, 480, _>(|index| {
                    ((index + frame * 480) as f32 * 0.03).sin() * 0.1
                });
                let controls = [GateControl::BYPASS; 480];
                for start in (0..480).step_by(127) {
                    let end = (start + 127).min(480);
                    assert_eq!(public.push_samples(&raw[start..end]), end - start);
                    assert_eq!(
                        controlled.push_controlled_samples(&raw[start..end], &controls[start..end]),
                        end - start
                    );
                    public.process_frames();
                    controlled.process_frames();
                }
                let mut a = [0.0; 480];
                let mut b = [0.0; 480];
                assert_eq!(
                    public.pop_samples_into(&mut a),
                    controlled.pop_samples_into(&mut b)
                );
                assert_eq!(a, b, "{model:?}, frame {frame}");
                assert_eq!(public.pending_input(), controlled.pending_input());
                assert_eq!(public.latency_samples(), controlled.latency_samples());
            }
        }
    }

    #[test]
    fn controlled_wet_mix_keeps_aligned_dry_and_frame_strength_arithmetic() {
        for model in models() {
            let mut controlled = new_controlled_noise_suppression_engine(
                model,
                Arc::new(AtomicU32::new(0.0_f32.to_bits())),
            );
            let mut full_wet =
                new_noise_suppression_engine(model, Arc::new(AtomicU32::new(1.0_f32.to_bits())));
            if !controlled.backend_available() {
                continue;
            }
            let standard = model.id() == "deepfilter";
            let mut compensation = GateCompensation::new(if standard { 0 } else { 480 });
            let delay = controlled.latency_samples();
            let mut dry_history = vec![0.0; delay];
            let mut strength = 0.0_f32;
            let alpha = 1.0 - (-10.0_f32 / 15.0).exp();
            for (frame, target) in [0.0, 0.0, 1.0, 1.0, 0.37, 0.0, 0.0, 1.0]
                .into_iter()
                .enumerate()
            {
                let gain = if frame < 3 { 0.02_f32 } else { 0.8 };
                let controls = [GateControl {
                    gain,
                    open: frame >= 3,
                }; 480];
                let raw = std::array::from_fn::<_, 480, _>(|index| {
                    ((index + frame * 480) as f32 * 0.017).sin() * 0.2 * gain
                });
                controlled.set_strength(target);
                for start in (0..480).step_by(127) {
                    let end = (start + 127).min(480);
                    controlled.push_controlled_samples(&raw[start..end], &controls[start..end]);
                    controlled.process_frames();
                }
                full_wet.push_samples(&raw);
                full_wet.process_frames();
                let mut wet = [0.0; 480];
                let mut actual = [0.0; 480];
                assert_eq!(full_wet.pop_samples_into(&mut wet), 480);
                assert_eq!(controlled.pop_samples_into(&mut actual), 480);
                if model == NoiseModel::RNNoise {
                    strength = target * alpha + strength * (1.0 - alpha);
                } else {
                    strength += alpha * (target - strength);
                }
                for index in 0..480 {
                    let wet = wet[index] * compensation.next_ratio(controls[index]);
                    let dry = dry_history[index];
                    let expected = strength * wet + (1.0 - strength) * dry;
                    assert_eq!(
                        actual[index], expected,
                        "{model:?}, frame={frame}, sample={index}"
                    );
                }
                dry_history.drain(..480);
                dry_history.extend_from_slice(&raw);
            }
        }
    }

    #[test]
    fn controlled_partitioned_queue_and_partial_reset_keep_sample_alignment() {
        for model in models() {
            for partition in [1, 127, 997] {
                for strength in [0.0_f32, 1.0] {
                    let mut ordinary = new_noise_suppression_engine(
                        model,
                        Arc::new(AtomicU32::new(strength.to_bits())),
                    );
                    let mut controlled = new_controlled_noise_suppression_engine(
                        model,
                        Arc::new(AtomicU32::new(strength.to_bits())),
                    );
                    let delay = if controlled.backend_available() && model.id() != "deepfilter" {
                        480
                    } else {
                        0
                    };
                    for epoch in 0..2 {
                        let raw: Vec<_> = (0..5117)
                            .map(|index| ((index + epoch * 5100) as f32 * 0.023).sin() * 0.1)
                            .collect();
                        let controls: Vec<_> = (0..raw.len())
                            .map(|index| GateControl {
                                gain: if (index / 719) % 2 == 0 { 0.02 } else { 0.8 },
                                open: (index / 301) % 3 != 0,
                            })
                            .collect();
                        let mut output_index = 0usize;
                        let mut ratio = GateCompensation::new(delay);
                        for start in (0..raw.len()).step_by(partition) {
                            let end = (start + partition).min(raw.len());
                            assert_eq!(ordinary.push_samples(&raw[start..end]), end - start);
                            assert_eq!(
                                controlled.push_controlled_samples(
                                    &raw[start..end],
                                    &controls[start..end]
                                ),
                                end - start
                            );
                            ordinary.process_frames();
                            controlled.process_frames();
                            assert_eq!(
                                ordinary.available_samples(),
                                controlled.available_samples()
                            );
                            while ordinary.available_samples() > 0 {
                                let mut expected = [0.0; 73];
                                let mut actual = [0.0; 73];
                                let count = ordinary.pop_samples_into(&mut expected);
                                assert_eq!(controlled.pop_samples_into(&mut actual), count);
                                for index in 0..count {
                                    let compensation = ratio.next_ratio(controls[output_index]);
                                    if strength == 1.0 {
                                        expected[index] *= compensation;
                                    }
                                    assert_eq!(actual[index], expected[index], "{model:?}, partition={partition}, epoch={epoch}, sample={output_index}");
                                    output_index += 1;
                                }
                            }
                        }
                        assert_eq!(ordinary.pending_input(), 317);
                        assert_eq!(controlled.pending_input(), 317);
                        ordinary.soft_reset();
                        controlled.soft_reset();
                        assert_eq!(controlled.pending_input(), 0);
                        assert_eq!(controlled.available_samples(), 0);
                    }
                }
            }
        }
    }

    #[test]
    fn controlled_rnnoise_push_process_pop_reset_does_not_allocate() {
        let mut engine = new_controlled_noise_suppression_engine(
            NoiseModel::RNNoise,
            Arc::new(AtomicU32::new(1.0_f32.to_bits())),
        );
        let raw = [0.05; 480];
        let controls = [GateControl {
            gain: 0.3,
            open: true,
        }; 480];
        let mut output = [0.0; 480];
        engine.push_controlled_samples(&raw, &controls);
        engine.process_frames();
        engine.pop_samples_into(&mut output);
        crate::test_alloc::assert_no_allocations("controlled RNNoise/reset", || {
            for _ in 0..10 {
                assert_eq!(engine.push_controlled_samples(&raw, &controls), 480);
                engine.process_frames();
                assert_eq!(engine.pop_samples_into(&mut output), 480);
            }
            engine.soft_reset();
        });
    }

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
}
