//! Audio input/output and processing module

pub mod buffer;
pub mod clock;
pub mod device;
pub mod input;
pub mod output;
pub mod processor;
pub mod rt;

use std::sync::{Mutex, MutexGuard};

static DEVICE_ENUMERATION_LOCK: Mutex<()> = Mutex::new(());

/// Serialize host-device enumeration across input and output callers.
///
/// CPAL's WASAPI enumeration can cross native process-global state. Keeping
/// these infrequent control-plane queries mutually exclusive avoids native
/// races when callers refresh input and output device lists concurrently.
pub(crate) fn lock_device_enumeration() -> MutexGuard<'static, ()> {
    DEVICE_ENUMERATION_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub use buffer::{AudioConsumer, AudioProducer, AudioRingBuffer};
pub use device::{list_input_devices, list_output_devices, DeviceInfo};
pub use input::{AudioDeviceInfo, AudioError, AudioInput, TARGET_SAMPLE_RATE};
pub use output::AudioOutput;
pub use processor::{
    AudioBlockProcessor, AudioProcessor, OfflineDspBlockProcessor, PyAudioProcessor,
};
pub use rt::{FixedAudioBuffer, RtCommandQueue, RtErrorCode};

#[cfg(feature = "vad")]
pub use processor::PyGateMode;
