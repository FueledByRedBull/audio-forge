//! Audio input/output and processing module

pub mod buffer;
pub mod clock;
pub mod device;
pub mod input;
pub mod output;
pub mod processor;

pub use buffer::{AudioConsumer, AudioProducer, AudioRingBuffer};
pub use device::{list_input_devices, list_output_devices, DeviceInfo};
pub use input::{AudioDeviceInfo, AudioError, AudioInput, TARGET_SAMPLE_RATE};
pub use output::AudioOutput;
pub use processor::{AudioProcessor, PyAudioProcessor};

#[cfg(feature = "vad")]
pub use processor::PyGateMode;
