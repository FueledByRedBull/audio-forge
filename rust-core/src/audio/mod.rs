//! Audio input/output and processing module

use cpal::{SupportedBufferSize, SupportedStreamConfig, SupportedStreamConfigRange};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use self::rt::store_rt_error;

/// Evaluate `$run` with `$t` bound to the Rust sample type of `$format`, or
/// `$other` for a format no stream builder handles. The input and output
/// dispatch tables and their preflight share this one list.
macro_rules! dispatch_sample_format {
    ($format:expr, $t:ident => $run:expr, _ => $other:expr) => {
        match $format {
            cpal::SampleFormat::I8 => {
                type $t = i8;
                $run
            }
            cpal::SampleFormat::I16 => {
                type $t = i16;
                $run
            }
            cpal::SampleFormat::I24 => {
                type $t = cpal::I24;
                $run
            }
            cpal::SampleFormat::I32 => {
                type $t = i32;
                $run
            }
            cpal::SampleFormat::I64 => {
                type $t = i64;
                $run
            }
            cpal::SampleFormat::U8 => {
                type $t = u8;
                $run
            }
            cpal::SampleFormat::U16 => {
                type $t = u16;
                $run
            }
            cpal::SampleFormat::U24 => {
                type $t = cpal::U24;
                $run
            }
            cpal::SampleFormat::U32 => {
                type $t = u32;
                $run
            }
            cpal::SampleFormat::U64 => {
                type $t = u64;
                $run
            }
            cpal::SampleFormat::F32 => {
                type $t = f32;
                $run
            }
            cpal::SampleFormat::F64 => {
                type $t = f64;
                $run
            }
            _ => $other,
        }
    };
}

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

/// Reject a format the stream callbacks cannot serve: they divide by the channel
/// count, and the capture ring and backlog thresholds scale with the rate.
fn validate_stream_format(channels: u16, sample_rate: u32) -> Result<(), input::AudioError> {
    if channels == 0 || sample_rate == 0 {
        return Err(input::AudioError::InvalidDeviceFormat(format!(
            "{channels} channels at {sample_rate} Hz"
        )));
    }
    Ok(())
}

// Windows is pinned to cpal 0.17.3 and the other platforms use 0.18 (see
// Cargo.toml); the few places the two APIs differ are the cfg pairs below.

/// The error type of a stream's error callback.
#[cfg(windows)]
type StreamError = cpal::StreamError;
#[cfg(not(windows))]
type StreamError = cpal::Error;

/// Whether the stream behind `error` can no longer deliver audio.
#[cfg(windows)]
fn stream_is_gone(error: &StreamError) -> bool {
    matches!(
        error,
        cpal::StreamError::DeviceNotAvailable | cpal::StreamError::StreamInvalidated
    )
}
#[cfg(not(windows))]
fn stream_is_gone(error: &StreamError) -> bool {
    matches!(
        error.kind(),
        cpal::ErrorKind::DeviceNotAvailable | cpal::ErrorKind::StreamInvalidated
    )
}

/// The stream config argument of `build_input_stream` and `build_output_stream`:
/// 0.17 borrows it, 0.18 takes it by value.
#[cfg(windows)]
fn stream_config_arg(config: &cpal::StreamConfig) -> &cpal::StreamConfig {
    config
}
#[cfg(not(windows))]
fn stream_config_arg(config: &cpal::StreamConfig) -> cpal::StreamConfig {
    *config
}

/// Shared stream error callback: count the error, publish its code and, when
/// the stream can no longer deliver audio, ask the supervisor for a restart
/// instead of waiting for the callback-stall watchdog.
fn handle_stream_error(
    error: &StreamError,
    code: RtErrorCode,
    error_count: &AtomicU64,
    rt_error_code: &AtomicU32,
    restart_requested: &AtomicBool,
) {
    error_count.fetch_add(1, Ordering::Relaxed);
    store_rt_error(rt_error_code, code);
    if stream_is_gone(error) {
        restart_requested.store(true, Ordering::Release);
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
        let min_rate = config.min_sample_rate();
        let max_rate = config.max_sample_rate();
        if preferred_sample_rate_from_ranges(0, &[(min_rate, max_rate)], target_rate) == target_rate
        {
            return Some(config.with_sample_rate(target_rate));
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
#[cfg(windows)]
pub(crate) mod scheduling;
#[cfg(target_os = "linux")]
pub mod virtual_mic;

pub use buffer::{AudioConsumer, AudioProducer, AudioRingBuffer};
pub use device::{list_input_devices, list_output_devices, DeviceInfo};
pub use input::{AudioDeviceInfo, AudioError, AudioInput, TARGET_SAMPLE_RATE};
pub use output::AudioOutput;
pub use processor::{AudioProcessor, OfflineDspBlockProcessor, PyAudioProcessor};
pub use rt::{FixedAudioBuffer, RtCommandQueue, RtErrorCode};

#[cfg(feature = "vad")]
pub use processor::PyGateMode;

#[cfg(test)]
mod tests {
    use super::*;
    use cpal::{SampleFormat, SizedSample};

    #[test]
    fn sample_format_dispatch_binds_every_pcm_format_cpal_reports() {
        for format in [
            SampleFormat::I8,
            SampleFormat::I16,
            SampleFormat::I24,
            SampleFormat::I32,
            SampleFormat::I64,
            SampleFormat::U8,
            SampleFormat::U16,
            SampleFormat::U24,
            SampleFormat::U32,
            SampleFormat::U64,
            SampleFormat::F32,
            SampleFormat::F64,
        ] {
            let bound =
                dispatch_sample_format!(format, S => Some(<S as SizedSample>::FORMAT), _ => None);
            assert_eq!(bound, Some(format), "{format} has no stream builder");
        }
        let unsupported = dispatch_sample_format!(
            SampleFormat::DsdU8,
            S => Some(<S as SizedSample>::FORMAT),
            _ => None
        );
        assert_eq!(unsupported, None);
    }

    #[test]
    fn stream_format_without_channels_or_rate_is_rejected() {
        assert!(validate_stream_format(1, 48_000).is_ok());
        assert!(validate_stream_format(2, 384_000).is_ok());
        assert!(validate_stream_format(0, 48_000).is_err());
        assert!(validate_stream_format(2, 0).is_err());
    }

    #[cfg(windows)]
    fn stream_error_cases() -> Vec<(StreamError, bool)> {
        vec![
            (cpal::StreamError::DeviceNotAvailable, true),
            (cpal::StreamError::StreamInvalidated, true),
            (cpal::StreamError::BufferUnderrun, false),
        ]
    }

    #[cfg(not(windows))]
    fn stream_error_cases() -> Vec<(StreamError, bool)> {
        vec![
            (cpal::Error::new(cpal::ErrorKind::DeviceNotAvailable), true),
            (cpal::Error::new(cpal::ErrorKind::StreamInvalidated), true),
            (cpal::Error::new(cpal::ErrorKind::Xrun), false),
            // The backend already rerouted the stream; it keeps delivering audio.
            (cpal::Error::new(cpal::ErrorKind::DeviceChanged), false),
        ]
    }

    #[test]
    fn stream_errors_request_a_restart_only_when_the_stream_is_gone() {
        for (error, restarts) in stream_error_cases() {
            let count = AtomicU64::new(0);
            let code = AtomicU32::new(RtErrorCode::None as u32);
            let restart = AtomicBool::new(false);
            handle_stream_error(
                &error,
                RtErrorCode::InputStreamError,
                &count,
                &code,
                &restart,
            );
            assert_eq!(count.load(Ordering::Relaxed), 1);
            assert_eq!(
                RtErrorCode::from_u32(code.load(Ordering::Relaxed)),
                RtErrorCode::InputStreamError
            );
            assert_eq!(restart.load(Ordering::Acquire), restarts, "{error:?}");
        }
    }
}
