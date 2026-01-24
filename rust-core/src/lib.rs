//! MicEq Core - High-performance DSP engine for real-time audio processing
//!
//! Processing chain: Mic Input → Noise Gate → RNNoise → 10-Band IIR EQ → Output

use pyo3::prelude::*;

pub mod audio;
pub mod dsp;

// Re-export main types
pub use audio::{AudioProcessor, PyAudioProcessor};
pub use dsp::{Biquad, Compressor, Limiter, NoiseGate, ParametricEQ, RNNoiseProcessor};

/// Python module initialization
#[pymodule]
fn mic_eq_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Main audio processor
    m.add_class::<audio::PyAudioProcessor>()?;

    // VAD Gate Mode enum (VAD feature only)
    #[cfg(feature = "vad")]
    m.add_class::<audio::PyGateMode>()?;

    // Device enumeration
    m.add_class::<audio::DeviceInfo>()?;
    m.add_function(wrap_pyfunction!(audio::list_input_devices, m)?)?;
    m.add_function(wrap_pyfunction!(audio::list_output_devices, m)?)?;

    Ok(())
}
