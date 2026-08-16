//! Audio output playback using cpal
//!
//! Real-time audio playback to VB Audio Cable or other output devices.
//! Requests 48kHz sample rate from device.
//!
//! Adapted from Spectral Workbench project.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{
    BufferSize, Device, FromSample, SampleFormat, SizedSample, Stream, StreamConfig,
    SupportedStreamConfig,
};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use super::buffer::AudioConsumer;
use super::clock::now_micros;
use super::input::{AudioDeviceInfo, AudioError, TARGET_SAMPLE_RATE};
use super::rt::{store_rt_error, RtErrorCode};
use super::{find_48khz_config, parse_fixed_buffer_frames, supported_fixed_buffer_frames};

pub(crate) struct OutputProbeControl {
    pub(crate) consumer: AudioConsumer,
    pub(crate) active: Arc<AtomicBool>,
    pub(crate) complete: Arc<AtomicBool>,
    pub(crate) cancel_requested: Arc<AtomicBool>,
}

/// Audio output stream
pub struct AudioOutput {
    stream: Stream,
    device_info: AudioDeviceInfo,
    fixed_buffer_frames: Option<u32>,
}

pub(crate) struct OutputStreamSetup {
    pub(crate) device: Device,
    pub(crate) supported_config: SupportedStreamConfig,
    pub(crate) device_info: AudioDeviceInfo,
}

impl AudioOutput {
    fn requested_fixed_buffer_frames() -> Option<u32> {
        let value = std::env::var("AUDIOFORGE_FIXED_OUTPUT_BUFFER_FRAMES").ok();
        parse_fixed_buffer_frames(value.as_deref())
    }

    fn preflight_config<T>(device: &Device, config: &StreamConfig) -> bool
    where
        T: SizedSample + FromSample<f32>,
    {
        device
            .build_output_stream(
                config,
                |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                    let silence = T::from_sample(0.0_f32);
                    for sample in data {
                        *sample = silence;
                    }
                },
                |_error| {},
                None,
            )
            .is_ok()
    }

    fn preflight_sample_format(
        device: &Device,
        sample_format: SampleFormat,
        config: &StreamConfig,
    ) -> bool {
        match sample_format {
            SampleFormat::I8 => Self::preflight_config::<i8>(device, config),
            SampleFormat::F32 => Self::preflight_config::<f32>(device, config),
            SampleFormat::F64 => Self::preflight_config::<f64>(device, config),
            SampleFormat::I16 => Self::preflight_config::<i16>(device, config),
            SampleFormat::I32 => Self::preflight_config::<i32>(device, config),
            SampleFormat::I64 => Self::preflight_config::<i64>(device, config),
            SampleFormat::U8 => Self::preflight_config::<u8>(device, config),
            SampleFormat::U16 => Self::preflight_config::<u16>(device, config),
            SampleFormat::U32 => Self::preflight_config::<u32>(device, config),
            SampleFormat::U64 => Self::preflight_config::<u64>(device, config),
            _ => false,
        }
    }

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
                    TARGET_SAMPLE_RATE,
                )
            })
            .or_else(|| find_48khz_config(supported_configs.iter().cloned(), TARGET_SAMPLE_RATE))
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

    pub(crate) fn from_named_device_ordinal_setup(
        name: &str,
        name_ordinal: u32,
    ) -> Result<OutputStreamSetup, AudioError> {
        let host = cpal::default_host();
        let device = host
            .output_devices()
            .map_err(|error| AudioError::DeviceName(error.to_string()))?
            .filter(|device| device.name().map(|value| value == name).unwrap_or(false))
            .nth(name_ordinal as usize)
            .ok_or_else(|| {
                AudioError::DeviceNotFound(format!("{name} (occurrence {name_ordinal})"))
            })?;
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

    fn drain_muted_consumer(
        consumer: &mut AudioConsumer,
        frames_needed: usize,
        scratch: &mut [f32],
    ) -> usize {
        let mut drained = 0usize;
        while drained < frames_needed {
            let batch = (frames_needed - drained).min(scratch.len());
            if batch == 0 {
                break;
            }
            let count = consumer.read(&mut scratch[..batch]);
            if count == 0 {
                break;
            }
            drained += count;
            if count < batch {
                break;
            }
        }
        consumer.set_last_sample(0.0);
        drained
    }

    fn render_probe<T>(
        data: &mut [T],
        num_channels: usize,
        probe: &mut OutputProbeControl,
        scratch: &mut [f32],
    ) -> bool
    where
        T: SizedSample + FromSample<f32>,
    {
        let frame_count = if num_channels == 1 {
            data.len()
        } else {
            data.len() / num_channels
        };
        let silence = Self::convert_output_sample::<T>(0.0);
        for sample in data.iter_mut() {
            *sample = silence;
        }

        let mut rendered = 0usize;
        while rendered < frame_count {
            let batch = (frame_count - rendered).min(scratch.len());
            if batch == 0 {
                break;
            }
            let count = probe.consumer.read(&mut scratch[..batch]);
            if count == 0 {
                break;
            }
            for (offset, &sample) in scratch[..count].iter().enumerate() {
                let converted = Self::convert_output_sample::<T>(sample);
                let frame = rendered + offset;
                if num_channels == 1 {
                    data[frame] = converted;
                } else {
                    for channel in 0..num_channels {
                        data[frame * num_channels + channel] = converted;
                    }
                }
            }
            rendered += count;
            if count < batch {
                break;
            }
        }

        probe.consumer.is_empty()
    }

    fn discard_probe(probe: &mut OutputProbeControl, scratch: &mut [f32]) {
        while !probe.consumer.is_empty() {
            let count = probe.consumer.read(scratch);
            if count == 0 {
                break;
            }
        }
        probe.consumer.set_last_sample(0.0);
        probe.cancel_requested.store(false, Ordering::Release);
        probe.active.store(false, Ordering::Release);
        probe.complete.store(true, Ordering::Release);
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
        error_count: Arc<AtomicU64>,
        rt_error_code: Arc<AtomicU32>,
        output_probe: Option<OutputProbeControl>,
        fixed_buffer_frames: Option<u32>,
    ) -> Result<Self, AudioError>
    where
        T: SizedSample + FromSample<f32>,
    {
        let mut consumer = consumer;
        let num_channels = device_info.channels as usize;

        const OUTPUT_SCRATCH_CAPACITY: usize = 8192;
        let mut mono_scratch: Vec<f32> = vec![0.0; OUTPUT_SCRATCH_CAPACITY];
        let mut output_probe = output_probe;

        let recording_active_clone = Arc::clone(&recording_active);
        let output_muted_clone = Arc::clone(&output_muted);

        let stream = device
            .build_output_stream(
                &stream_config,
                move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                    // RT_REGION_START: cpal_output_callback
                    last_callback_time_us.store(now_micros(), Ordering::Relaxed);

                    if let Some(probe) = output_probe.as_mut() {
                        if probe.cancel_requested.load(Ordering::Acquire) {
                            Self::discard_probe(probe, &mut mono_scratch);
                        } else if probe.active.load(Ordering::Acquire) {
                            let frames_needed = if num_channels == 1 {
                                data.len()
                            } else {
                                data.len() / num_channels
                            };
                            Self::drain_muted_consumer(
                                &mut consumer,
                                frames_needed,
                                &mut mono_scratch,
                            );
                            underrun_streak.store(0, Ordering::Relaxed);
                            let complete =
                                Self::render_probe(data, num_channels, probe, &mut mono_scratch);
                            if complete {
                                probe.active.store(false, Ordering::Release);
                                probe.complete.store(true, Ordering::Release);
                            }
                            return;
                        }
                    }

                    if recording_active_clone.load(Ordering::Relaxed)
                        || output_muted_clone.load(Ordering::Relaxed)
                    {
                        let frames_needed = if num_channels == 1 {
                            data.len()
                        } else {
                            data.len() / num_channels
                        };
                        Self::drain_muted_consumer(&mut consumer, frames_needed, &mut mono_scratch);
                        underrun_streak.store(0, Ordering::Relaxed);
                        let silence = Self::convert_output_sample::<T>(0.0);
                        for sample in data.iter_mut() {
                            *sample = silence;
                        }
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
                    // RT_REGION_END: cpal_output_callback
                },
                move |err| {
                    let _ = err;
                    error_count.fetch_add(1, Ordering::Relaxed);
                    store_rt_error(rt_error_code.as_ref(), RtErrorCode::OutputStreamError);
                },
                None,
            )
            .map_err(|e| AudioError::BuildStream(e.to_string()))?;

        Ok(Self {
            stream,
            device_info,
            fixed_buffer_frames,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_setup(
        setup: OutputStreamSetup,
        consumer: AudioConsumer,
        recording_active: Arc<AtomicBool>,
        output_muted: Arc<AtomicBool>,
        last_callback_time_us: Arc<AtomicU64>,
        underrun_streak: Arc<AtomicU32>,
        total_underruns: Arc<AtomicU64>,
        error_count: Arc<AtomicU64>,
        rt_error_code: Arc<AtomicU32>,
        output_probe: Option<OutputProbeControl>,
    ) -> Result<Self, AudioError> {
        let sample_format = setup.supported_config.sample_format();
        let mut stream_config = setup.supported_config.config();
        let mut fixed_buffer_frames = None;
        if let Some(frames) = Self::requested_fixed_buffer_frames() {
            if supported_fixed_buffer_frames(frames, setup.supported_config.buffer_size()) {
                let mut candidate = stream_config.clone();
                candidate.buffer_size = BufferSize::Fixed(frames);
                if Self::preflight_sample_format(&setup.device, sample_format, &candidate) {
                    stream_config = candidate;
                    fixed_buffer_frames = Some(frames);
                }
            }
        }

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
                error_count,
                rt_error_code,
                output_probe,
                fixed_buffer_frames,
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
                error_count,
                rt_error_code,
                output_probe,
                fixed_buffer_frames,
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
                error_count,
                rt_error_code,
                output_probe,
                fixed_buffer_frames,
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
                error_count,
                rt_error_code,
                output_probe,
                fixed_buffer_frames,
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
                error_count,
                rt_error_code,
                output_probe,
                fixed_buffer_frames,
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
                error_count,
                rt_error_code,
                output_probe,
                fixed_buffer_frames,
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
                error_count,
                rt_error_code,
                output_probe,
                fixed_buffer_frames,
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
                error_count,
                rt_error_code,
                output_probe,
                fixed_buffer_frames,
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
                error_count,
                rt_error_code,
                output_probe,
                fixed_buffer_frames,
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
                error_count,
                rt_error_code,
                output_probe,
                fixed_buffer_frames,
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

    pub fn fixed_buffer_frames(&self) -> Option<u32> {
        self.fixed_buffer_frames
    }
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
    #[ignore = "requires a Windows audio endpoint"]
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
    fn test_muted_output_drain_discards_stale_samples() {
        let rb = crate::audio::AudioRingBuffer::new(16);
        let (mut producer, mut consumer) = rb.split();
        assert_eq!(producer.write(&[0.1, 0.2, 0.3, 0.4, 0.5]), 5);

        let mut scratch = vec![0.0_f32; 4];
        let drained = AudioOutput::drain_muted_consumer(&mut consumer, 3, &mut scratch);

        assert_eq!(drained, 3);
        assert_eq!(consumer.len(), 2);
        assert_eq!(consumer.last_sample(), 0.0);

        let mut remaining = vec![0.0_f32; 2];
        assert_eq!(consumer.read(&mut remaining), 2);
        assert_eq!(remaining, vec![0.4, 0.5]);
    }

    #[test]
    fn test_probe_render_duplicates_mono_across_output_channels() {
        let rb = crate::audio::AudioRingBuffer::new(8);
        let (mut producer, consumer) = rb.split();
        assert_eq!(producer.write(&[0.25, -0.5, 0.75]), 3);
        let mut probe = OutputProbeControl {
            consumer,
            active: Arc::new(AtomicBool::new(true)),
            complete: Arc::new(AtomicBool::new(false)),
            cancel_requested: Arc::new(AtomicBool::new(false)),
        };
        let mut output = [9.0_f32; 8];
        let mut scratch = [0.0_f32; 8];

        let complete = AudioOutput::render_probe(&mut output, 2, &mut probe, &mut scratch);

        assert!(complete);
        assert_eq!(output, [0.25, 0.25, -0.5, -0.5, 0.75, 0.75, 0.0, 0.0]);
    }

    #[test]
    fn test_probe_render_continues_across_callbacks_without_padding_source() {
        let rb = crate::audio::AudioRingBuffer::new(8);
        let (mut producer, consumer) = rb.split();
        assert_eq!(producer.write(&[0.1, 0.2, 0.3, 0.4]), 4);
        let mut probe = OutputProbeControl {
            consumer,
            active: Arc::new(AtomicBool::new(true)),
            complete: Arc::new(AtomicBool::new(false)),
            cancel_requested: Arc::new(AtomicBool::new(false)),
        };
        let mut first = [0.0_f32; 2];
        let mut second = [0.0_f32; 3];
        let mut scratch = [0.0_f32; 8];

        assert!(!AudioOutput::render_probe(
            &mut first,
            1,
            &mut probe,
            &mut scratch
        ));
        assert!(AudioOutput::render_probe(
            &mut second,
            1,
            &mut probe,
            &mut scratch
        ));
        assert_eq!(first, [0.1, 0.2]);
        assert_eq!(second, [0.3, 0.4, 0.0]);
    }

    #[test]
    fn test_discard_probe_clears_queue_and_publishes_completion() {
        let rb = crate::audio::AudioRingBuffer::new(8);
        let (mut producer, consumer) = rb.split();
        assert_eq!(producer.write(&[0.1, 0.2, 0.3]), 3);
        let active = Arc::new(AtomicBool::new(true));
        let complete = Arc::new(AtomicBool::new(false));
        let cancel_requested = Arc::new(AtomicBool::new(true));
        let mut probe = OutputProbeControl {
            consumer,
            active: Arc::clone(&active),
            complete: Arc::clone(&complete),
            cancel_requested: Arc::clone(&cancel_requested),
        };
        let mut scratch = [0.0_f32; 2];

        AudioOutput::discard_probe(&mut probe, &mut scratch);

        assert!(probe.consumer.is_empty());
        assert!(!active.load(Ordering::Acquire));
        assert!(complete.load(Ordering::Acquire));
        assert!(!cancel_requested.load(Ordering::Acquire));
    }
}
