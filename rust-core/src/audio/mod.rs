//! Audio input/output and processing module

use cpal::{SupportedBufferSize, SupportedStreamConfig, SupportedStreamConfigRange};

const MIN_FIXED_BUFFER_FRAMES: u32 = 16;
const MAX_FIXED_BUFFER_FRAMES: u32 = 8192;

fn parse_fixed_buffer_frames(value: Option<&str>) -> Option<u32> {
    let frames = value?.trim().parse::<u32>().ok()?;
    (MIN_FIXED_BUFFER_FRAMES..=MAX_FIXED_BUFFER_FRAMES)
        .contains(&frames)
        .then_some(frames)
}

fn supported_fixed_buffer_frames(frames: u32, supported: &SupportedBufferSize) -> bool {
    match supported {
        SupportedBufferSize::Range { min, max } => (*min..=*max).contains(&frames),
        SupportedBufferSize::Unknown => true,
    }
}

fn preferred_sample_rate_from_ranges(
    default_rate: u32,
    ranges: &[(u32, u32)],
    target_rate: u32,
) -> u32 {
    if ranges
        .iter()
        .any(|(min_rate, max_rate)| *min_rate <= target_rate && target_rate <= *max_rate)
    {
        target_rate
    } else {
        default_rate
    }
}

fn find_48khz_config(
    configs: impl Iterator<Item = SupportedStreamConfigRange>,
    target_rate: u32,
) -> Option<SupportedStreamConfig> {
    for config in configs {
        let min_rate = config.min_sample_rate().0;
        let max_rate = config.max_sample_rate().0;
        if preferred_sample_rate_from_ranges(0, &[(min_rate, max_rate)], target_rate) == target_rate
        {
            return Some(config.with_sample_rate(cpal::SampleRate(target_rate)));
        }
    }
    None
}

pub mod buffer;
pub mod clock;
pub mod device;
pub mod input;
pub mod output;
pub mod processor;
pub mod rt;

pub use buffer::{AudioConsumer, AudioProducer, AudioRingBuffer};
pub use device::{list_input_devices, list_output_devices, DeviceInfo};
pub use input::{AudioDeviceInfo, AudioError, AudioInput, TARGET_SAMPLE_RATE};
pub use output::AudioOutput;
pub use processor::{AudioProcessor, OfflineDspBlockProcessor, PyAudioProcessor};
pub use rt::{FixedAudioBuffer, RtCommandQueue, RtErrorCode};

#[cfg(feature = "vad")]
pub use processor::PyGateMode;
