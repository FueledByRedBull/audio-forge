//! Digital Signal Processing components

pub mod biquad;
pub mod compressor;
pub mod eq;
pub mod gate;
pub mod limiter;
pub mod loudness;
pub mod noise_suppressor;
pub mod rnnoise;

#[cfg(feature = "vad")]
pub mod vad;

#[cfg(feature = "deepfilter")]
pub mod deepfilter_ffi;

pub use biquad::{Biquad, BiquadType};
pub use compressor::Compressor;
pub use eq::{ParametricEQ, DEFAULT_FREQUENCIES, DEFAULT_Q, NUM_BANDS};
pub use gate::NoiseGate;
pub use limiter::Limiter;
pub use loudness::{LoudnessError, LoudnessMeter};
pub use noise_suppressor::{NoiseModel, NoiseSuppressionEngine, NoiseSuppressor};
pub use rnnoise::{RNNoiseProcessor, RNNOISE_FRAME_SIZE};

#[cfg(feature = "vad")]
pub use vad::{GateMode, VadAutoGate};

#[cfg(feature = "deepfilter")]
pub use deepfilter_ffi::{DeepFilterProcessor, DEEPFILTER_FRAME_SIZE};
