//! Audio output playback using cpal
//!
//! Real-time audio playback to VB Audio Cable or other output devices.
//! Requests 48kHz sample rate from device.
//!
//! Adapted from Spectral Workbench project.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{
    Device, FromSample, SampleFormat, SizedSample, Stream, StreamConfig, SupportedStreamConfig,
    SupportedStreamConfigRange,
};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use super::buffer::AudioConsumer;
use super::clock::now_micros;
use super::input::{AudioDeviceInfo, AudioError, TARGET_SAMPLE_RATE};

/// Audio output stream
pub struct AudioOutput {
    stream: Stream,
    device_info: AudioDeviceInfo,
}

pub(crate) struct OutputStreamSetup {
    pub(crate) device: Device,
    pub(crate) supported_config: SupportedStreamConfig,
    pub(crate) device_info: AudioDeviceInfo,
}

impl AudioOutput {
    fn select_device(device: Device) -> Result<OutputStreamSetup, AudioError> {
        let name = device
            .name()
            .map_err(|e| AudioError::DeviceName(e.to_string()))?;

        let supported_configs: Vec<_> = device
            .supported_output_configs()
            .map_err(|e| AudioError::DefaultConfig(e.to_string()))?
            .collect();
        let default_config = device.default_output_config().ok();

        let supported_config = default_config
            .as_ref()
            .and_then(|default| {
                find_48khz_config(
                    supported_configs
                        .iter()
                        .filter(|config| {
                            config.channels() == default.channels()
                                && config.sample_format() == default.sample_format()
                        })
                        .cloned(),
                )
            })
            .or_else(|| find_48khz_config(supported_configs.iter().cloned()))
            .or(default_config)
            .ok_or_else(|| {
                AudioError::DefaultConfig("No suitable output config found".to_string())
            })?;

        let device_info = AudioDeviceInfo {
            name,
            sample_rate: supported_config.sample_rate().0,
            channels: supported_config.channels(),
        };

        Ok(OutputStreamSetup {
            device,
            supported_config,
            device_info,
        })
    }

    pub(crate) fn from_default_device_setup() -> Result<OutputStreamSetup, AudioError> {
        let host = cpal::default_host();
        let device = host.default_output_device().ok_or(AudioError::NoDevice)?;
        Self::select_device(device)
    }

    pub(crate) fn from_named_device_setup(name: &str) -> Result<OutputStreamSetup, AudioError> {
        let host = cpal::default_host();
        let device = host
            .output_devices()
            .map_err(|e| AudioError::DeviceName(e.to_string()))?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
            .ok_or_else(|| AudioError::DeviceNotFound(name.to_string()))?;
        Self::select_device(device)
    }

    fn convert_output_sample<T>(sample: f32) -> T
    where
        T: SizedSample + FromSample<f32>,
    {
        T::from_sample(sample.clamp(-1.0, 1.0))
    }

    fn fill_underrun_tail<T>(
        data: &mut [T],
        copied_frames: usize,
        num_channels: usize,
        last_sample: f32,
    ) where
        T: SizedSample + FromSample<f32>,
    {
        let total_frames = if num_channels == 1 {
            data.len()
        } else {
            data.len() / num_channels
        };
        let remaining_frames = total_frames.saturating_sub(copied_frames);
        if remaining_frames == 0 {
            return;
        }

        let fade_frames = remaining_frames.min(64);
        if num_channels == 1 {
            for i in 0..remaining_frames {
                let value = if i < fade_frames {
                    let t = (i + 1) as f32 / fade_frames as f32;
                    let gain = ((1.0 - t) * std::f32::consts::FRAC_PI_2).sin();
                    last_sample * gain
                } else {
                    0.0
                };
                data[copied_frames + i] = Self::convert_output_sample::<T>(value);
            }
        } else {
            for i in 0..remaining_frames {
                let value = if i < fade_frames {
                    let t = (i + 1) as f32 / fade_frames as f32;
                    let gain = ((1.0 - t) * std::f32::consts::FRAC_PI_2).sin();
                    last_sample * gain
                } else {
                    0.0
                };
                let converted = Self::convert_output_sample::<T>(value);
                let frame_idx = copied_frames + i;
                for channel in 0..num_channels {
                    data[frame_idx * num_channels + channel] = converted;
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_stream<T>(
        device: Device,
        stream_config: StreamConfig,
        device_info: AudioDeviceInfo,
        consumer: AudioConsumer,
        recording_active: Arc<AtomicBool>,
        output_muted: Arc<AtomicBool>,
        last_callback_time_us: Arc<AtomicU64>,
        underrun_streak: Arc<AtomicU32>,
        total_underruns: Arc<AtomicU64>,
    ) -> Result<Self, AudioError>
    where
        T: SizedSample + FromSample<f32>,
    {
        let mut consumer = consumer;
        let num_channels = device_info.channels as usize;

        const OUTPUT_SCRATCH_CAPACITY: usize = 8192;
        let mut mono_scratch: Vec<f32> = vec![0.0; OUTPUT_SCRATCH_CAPACITY];

        let recording_active_clone = Arc::clone(&recording_active);
        let output_muted_clone = Arc::clone(&output_muted);

        let stream = device
            .build_output_stream(
                &stream_config,
                move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                    last_callback_time_us.store(now_micros(), Ordering::Relaxed);

                    if recording_active_clone.load(Ordering::Relaxed)
                        || output_muted_clone.load(Ordering::Relaxed)
                    {
                        underrun_streak.store(0, Ordering::Relaxed);
                        let silence = Self::convert_output_sample::<T>(0.0);
                        for sample in data.iter_mut() {
                            *sample = silence;
                        }
                        consumer.set_last_sample(0.0);
                        return;
                    }

                    let available = consumer.len();

                    if num_channels == 1 {
                        let needed = data.len();
                        if available < needed {
                            underrun_streak.fetch_add(1, Ordering::Relaxed);
                            total_underruns.fetch_add(1, Ordering::Relaxed);
                        } else {
                            underrun_streak.store(0, Ordering::Relaxed);
                        }

                        let to_read = available.min(data.len());
                        let mut copied = 0usize;
                        let mut last_written_sample = None;
                        while copied < to_read {
                            let batch = (to_read - copied).min(OUTPUT_SCRATCH_CAPACITY);
                            let count = consumer.read(&mut mono_scratch[..batch]);
                            if count == 0 {
                                break;
                            }

                            for (dst, &sample) in data[copied..copied + count]
                                .iter_mut()
                                .zip(mono_scratch[..count].iter())
                            {
                                *dst = Self::convert_output_sample::<T>(sample);
                                last_written_sample = Some(sample);
                            }

                            copied += count;
                            if count < batch {
                                break;
                            }
                        }

                        if copied < data.len() {
                            let last =
                                last_written_sample.unwrap_or_else(|| consumer.last_sample());
                            Self::fill_underrun_tail(data, copied, 1, last);
                            consumer.set_last_sample(0.0);
                        }
                    } else {
                        let mono_samples = data.len() / num_channels;
                        if available < mono_samples {
                            underrun_streak.fetch_add(1, Ordering::Relaxed);
                            total_underruns.fetch_add(1, Ordering::Relaxed);
                        } else {
                            underrun_streak.store(0, Ordering::Relaxed);
                        }

                        let to_read = available.min(mono_samples);
                        let mut copied_frames = 0usize;
                        let mut last_written_sample = None;
                        while copied_frames < to_read {
                            let batch = (to_read - copied_frames).min(OUTPUT_SCRATCH_CAPACITY);
                            let count = consumer.read(&mut mono_scratch[..batch]);
                            if count == 0 {
                                break;
                            }

                            for (i, &sample) in mono_scratch[..count].iter().enumerate() {
                                let frame_idx = copied_frames + i;
                                let converted = Self::convert_output_sample::<T>(sample);
                                for channel in 0..num_channels {
                                    data[frame_idx * num_channels + channel] = converted;
                                }
                                last_written_sample = Some(sample);
                            }

                            copied_frames += count;
                            if count < batch {
                                break;
                            }
                        }

                        if copied_frames < mono_samples {
                            let last =
                                last_written_sample.unwrap_or_else(|| consumer.last_sample());
                            Self::fill_underrun_tail(data, copied_frames, num_channels, last);
                            consumer.set_last_sample(0.0);
                        }
                    }
                },
                move |err| {
                    eprintln!("[OUTPUT] Audio output error: {}", err);
                },
                None,
            )
            .map_err(|e| AudioError::BuildStream(e.to_string()))?;

        Ok(Self {
            stream,
            device_info,
        })
    }

    /// Create audio output from default device
    pub fn from_default_device(
        consumer: AudioConsumer,
        recording_active: Arc<AtomicBool>,
        output_muted: Arc<AtomicBool>,
        last_callback_time_us: Arc<AtomicU64>,
        underrun_streak: Arc<AtomicU32>,
        total_underruns: Arc<AtomicU64>,
    ) -> Result<Self, AudioError> {
        let setup = Self::from_default_device_setup()?;
        Self::from_setup(
            setup,
            consumer,
            recording_active,
            output_muted,
            last_callback_time_us,
            underrun_streak,
            total_underruns,
        )
    }

    /// Create audio output from device by name
    pub fn from_device_name(
        name: &str,
        consumer: AudioConsumer,
        recording_active: Arc<AtomicBool>,
        output_muted: Arc<AtomicBool>,
        last_callback_time_us: Arc<AtomicU64>,
        underrun_streak: Arc<AtomicU32>,
        total_underruns: Arc<AtomicU64>,
    ) -> Result<Self, AudioError> {
        let setup = Self::from_named_device_setup(name)?;
        Self::from_setup(
            setup,
            consumer,
            recording_active,
            output_muted,
            last_callback_time_us,
            underrun_streak,
            total_underruns,
        )
    }

    pub(crate) fn from_setup(
        setup: OutputStreamSetup,
        consumer: AudioConsumer,
        recording_active: Arc<AtomicBool>,
        output_muted: Arc<AtomicBool>,
        last_callback_time_us: Arc<AtomicU64>,
        underrun_streak: Arc<AtomicU32>,
        total_underruns: Arc<AtomicU64>,
    ) -> Result<Self, AudioError> {
        let sample_format = setup.supported_config.sample_format();
        let stream_config = setup.supported_config.config();

        match sample_format {
            SampleFormat::I8 => Self::build_stream::<i8>(
                setup.device,
                stream_config,
                setup.device_info,
                consumer,
                recording_active,
                output_muted,
                last_callback_time_us,
                underrun_streak,
                total_underruns,
            ),
            SampleFormat::F32 => Self::build_stream::<f32>(
                setup.device,
                stream_config,
                setup.device_info,
                consumer,
                recording_active,
                output_muted,
                last_callback_time_us,
                underrun_streak,
                total_underruns,
            ),
            SampleFormat::F64 => Self::build_stream::<f64>(
                setup.device,
                stream_config,
                setup.device_info,
                consumer,
                recording_active,
                output_muted,
                last_callback_time_us,
                underrun_streak,
                total_underruns,
            ),
            SampleFormat::I16 => Self::build_stream::<i16>(
                setup.device,
                stream_config,
                setup.device_info,
                consumer,
                recording_active,
                output_muted,
                last_callback_time_us,
                underrun_streak,
                total_underruns,
            ),
            SampleFormat::I32 => Self::build_stream::<i32>(
                setup.device,
                stream_config,
                setup.device_info,
                consumer,
                recording_active,
                output_muted,
                last_callback_time_us,
                underrun_streak,
                total_underruns,
            ),
            SampleFormat::I64 => Self::build_stream::<i64>(
                setup.device,
                stream_config,
                setup.device_info,
                consumer,
                recording_active,
                output_muted,
                last_callback_time_us,
                underrun_streak,
                total_underruns,
            ),
            SampleFormat::U8 => Self::build_stream::<u8>(
                setup.device,
                stream_config,
                setup.device_info,
                consumer,
                recording_active,
                output_muted,
                last_callback_time_us,
                underrun_streak,
                total_underruns,
            ),
            SampleFormat::U16 => Self::build_stream::<u16>(
                setup.device,
                stream_config,
                setup.device_info,
                consumer,
                recording_active,
                output_muted,
                last_callback_time_us,
                underrun_streak,
                total_underruns,
            ),
            SampleFormat::U32 => Self::build_stream::<u32>(
                setup.device,
                stream_config,
                setup.device_info,
                consumer,
                recording_active,
                output_muted,
                last_callback_time_us,
                underrun_streak,
                total_underruns,
            ),
            SampleFormat::U64 => Self::build_stream::<u64>(
                setup.device,
                stream_config,
                setup.device_info,
                consumer,
                recording_active,
                output_muted,
                last_callback_time_us,
                underrun_streak,
                total_underruns,
            ),
            other => Err(AudioError::UnsupportedSampleFormat(other.to_string())),
        }
    }

    /// Start playing audio
    pub fn start(&self) -> Result<(), AudioError> {
        self.stream
            .play()
            .map_err(|e| AudioError::PlayStream(e.to_string()))
    }

    /// Pause audio playback
    pub fn pause(&self) -> Result<(), AudioError> {
        self.stream
            .pause()
            .map_err(|e| AudioError::PlayStream(e.to_string()))
    }

    /// Get device information
    pub fn device_info(&self) -> &AudioDeviceInfo {
        &self.device_info
    }
}

/// Find a config that supports 48kHz
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
) -> Option<cpal::SupportedStreamConfig> {
    for config in configs {
        let min_rate = config.min_sample_rate().0;
        let max_rate = config.max_sample_rate().0;
        if preferred_sample_rate_from_ranges(0, &[(min_rate, max_rate)], TARGET_SAMPLE_RATE)
            == TARGET_SAMPLE_RATE
        {
            return Some(config.with_sample_rate(cpal::SampleRate(TARGET_SAMPLE_RATE)));
        }
    }
    None
}

/// List available audio output devices
pub fn list_output_devices() -> Result<Vec<AudioDeviceInfo>, AudioError> {
    let host = cpal::default_host();
    let mut devices = Vec::new();

    let device_iter = host
        .output_devices()
        .map_err(|e| AudioError::DeviceName(e.to_string()))?;

    for device in device_iter {
        if let Ok(name) = device.name() {
            if let Ok(config) = device.default_output_config() {
                devices.push(AudioDeviceInfo {
                    name,
                    sample_rate: config.sample_rate().0,
                    channels: config.channels(),
                });
            }
        }
    }

    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_devices() {
        let _ = list_output_devices();
    }

    #[test]
    fn test_convert_output_i16_sample() {
        let sample = AudioOutput::convert_output_sample::<i16>(1.0);
        assert_eq!(sample, i16::MAX);
    }

    #[test]
    fn test_convert_output_u16_sample() {
        let sample = AudioOutput::convert_output_sample::<u16>(0.0);
        assert_eq!(sample, u16::MAX / 2 + 1);
    }

    #[test]
    fn test_fill_underrun_tail_fades_then_silences_mono() {
        let mut data = vec![0_i16; 80];
        AudioOutput::fill_underrun_tail(&mut data, 0, 1, 1.0);
        assert_ne!(data[0], 0);
        assert_eq!(data[63], 0);
        assert_eq!(data[79], 0);
    }

    #[test]
    fn test_fill_underrun_tail_writes_all_channels() {
        let mut data = vec![0_i16; 12];
        AudioOutput::fill_underrun_tail(&mut data, 1, 2, 0.5);
        assert_ne!(data[2], 0);
        assert_eq!(data[2], data[3]);
        assert_eq!(data[10], 0);
        assert_eq!(data[11], 0);
    }

    #[test]
    fn test_preferred_sample_rate_uses_target_when_supported() {
        let chosen = preferred_sample_rate_from_ranges(44_100, &[(44_100, 48_000)], 48_000);
        assert_eq!(chosen, 48_000);
    }

    #[test]
    fn test_preferred_sample_rate_falls_back_to_default_when_target_missing() {
        let chosen = preferred_sample_rate_from_ranges(44_100, &[(44_100, 44_100)], 48_000);
        assert_eq!(chosen, 44_100);
    }
}
