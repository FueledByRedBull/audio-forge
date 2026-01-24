//! Digital Signal Processing components

pub mod biquad;
pub mod compressor;
pub mod eq;
pub mod gate;
pub mod limiter;
pub mod noise_suppressor;
pub mod rnnoise;

#[cfg(feature = "deepfilter")]
pub mod deepfilter_ffi;

pub use biquad::{Biquad, BiquadType};
pub use compressor::Compressor;
pub use eq::{ParametricEQ, DEFAULT_FREQUENCIES, DEFAULT_Q, NUM_BANDS};
pub use gate::NoiseGate;
pub use limiter::Limiter;
pub use noise_suppressor::{NoiseModel, NoiseSuppressor, NoiseSuppressionEngine};
pub use rnnoise::{RNNoiseProcessor, RNNOISE_FRAME_SIZE};

#[cfg(feature = "deepfilter")]
pub use deepfilter_ffi::{DeepFilterProcessor, DEEPFILTER_FRAME_SIZE};
