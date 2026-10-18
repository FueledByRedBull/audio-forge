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
use super::device::{device_for_endpoint_id, device_name, persisted_endpoint_id};
use super::input::{AudioDeviceInfo, AudioError, TARGET_SAMPLE_RATE};
use super::rt::RtErrorCode;
use super::{
    find_48khz_config, handle_stream_error, parse_fixed_buffer_frames,
    supported_fixed_buffer_frames, validate_stream_format,
};

pub(crate) struct OutputProbeControl {
    pub(crate) consumer: AudioConsumer,
    pub(crate) active: Arc<AtomicBool>,
    pub(crate) complete: Arc<AtomicBool>,
    pub(crate) cancel_requested: Arc<AtomicBool>,
}

/// How long an unprimed stream keeps hiding underruns as start-up priming,
/// measured from the first underrun the output callback sees after real audio
/// was queued (the writer restamps its marker on every write, so the marker
/// cannot be the origin). The observed start-up gap is a few 10 ms callbacks;
/// DeepFilter and resampler warm-up add well under 500 ms. 1 s clears that with
/// margin and still surfaces a stream that never catches up (an oversized fixed
/// buffer, a DSP slower than real time).
pub(crate) const OUTPUT_PRIMING_MAX_US: u64 = 1_000_000;

/// Frames over which an underrun fades out to silence and the next audio fades back in.
const UNDERRUN_FADE_FRAMES: usize = 64;

/// Output-callback underrun accounting shared with the processor.
pub(crate) struct OutputUnderruns {
    pub(crate) streak: Arc<AtomicU32>,
    pub(crate) total: Arc<AtomicU64>,
    /// Empty-queue callbacks before the stream is primed (see `primed`).
    pub(crate) priming_total: Arc<AtomicU64>,
    /// Time of the DSP loop's first queued write (0 until then); a start resets it.
    pub(crate) audio_queued_at_us: Arc<AtomicU64>,
    /// Callback time of the first underrun seen after real audio was queued
    /// (0 until then); a start resets it. Bounds priming, see `OUTPUT_PRIMING_MAX_US`.
    pub(crate) priming_started_at_us: Arc<AtomicU64>,
    /// Set by the callback once it fully serves a request after real audio was
    /// queued, i.e. the queue has a cushion; a start resets it.
    pub(crate) primed: Arc<AtomicBool>,
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
        dispatch_sample_format!(
            sample_format,
            S => Self::preflight_config::<S>(device, config),
            _ => false
        )
    }

    fn select_device(device: Device) -> Result<OutputStreamSetup, AudioError> {
        let name = device_name(&device).map_err(AudioError::DeviceName)?;

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
        validate_stream_format(supported_config.channels(), supported_config.sample_rate())?;

        let device_info = AudioDeviceInfo {
            name,
            endpoint_id: device
                .id()
                .ok()
                .map(|device_id| persisted_endpoint_id(&device_id)),
            sample_rate: supported_config.sample_rate(),
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

    pub(crate) fn from_named_device_identity_setup(
        name: &str,
        name_ordinal: u32,
        endpoint_id: Option<&str>,
    ) -> Result<OutputStreamSetup, AudioError> {
        let host = cpal::default_host();
        let device = if let Some(endpoint_id) = endpoint_id {
            device_for_endpoint_id(endpoint_id, false).map_err(AudioError::DeviceNotFound)?
        } else {
            host.output_devices()
                .map_err(|error| AudioError::DeviceName(error.to_string()))?
                .filter(|device| device_name(device).is_ok_and(|value| value == name))
                .nth(name_ordinal as usize)
                .ok_or_else(|| {
                    AudioError::DeviceNotFound(format!("{name} (occurrence {name_ordinal})"))
                })?
        };
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

        let fade_frames = remaining_frames.min(UNDERRUN_FADE_FRAMES);
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

    // RT_REGION_START: cpal_output_renderer
    /// Ramp up the first audio after an underrun, which ended in silence.
    fn fade_in_after_underrun(samples: &mut [f32], consumer: &mut AudioConsumer) {
        let remaining = consumer.resume_fade_remaining();
        let count = remaining.min(samples.len());
        if count == 0 {
            return;
        }
        let elapsed = UNDERRUN_FADE_FRAMES - remaining;
        for (i, sample) in samples[..count].iter_mut().enumerate() {
            let t = (elapsed + i + 1) as f32 / UNDERRUN_FADE_FRAMES as f32;
            *sample *= (t * std::f32::consts::FRAC_PI_2).sin();
        }
        consumer.set_resume_fade_remaining(remaining - count);
        // A tail starting inside the fade must fade from what was played.
        consumer.set_last_sample(samples[samples.len() - 1]);
    }

    fn render_output<T>(
        data: &mut [T],
        num_channels: usize,
        consumer: &mut AudioConsumer,
        scratch: &mut [f32],
        now_us: u64,
        underruns: &OutputUnderruns,
    ) -> usize
    where
        T: SizedSample + FromSample<f32>,
    {
        let frames_needed = data.len() / num_channels;
        let mut copied_frames = 0usize;
        // Sampled before reading, so a request served by the silent prime alone
        // cannot count as served real audio.
        let queued_at_us = underruns.audio_queued_at_us.load(Ordering::Relaxed);

        while copied_frames < frames_needed {
            let batch = (frames_needed - copied_frames).min(scratch.len());
            if batch == 0 {
                break;
            }
            let count = consumer.read(&mut scratch[..batch]);
            if count == 0 {
                break;
            }
            Self::fade_in_after_underrun(&mut scratch[..count], consumer);

            for (i, &sample) in scratch[..count].iter().enumerate() {
                let frame_idx = copied_frames + i;
                let converted = Self::convert_output_sample::<T>(sample);
                if num_channels == 1 {
                    data[frame_idx] = converted;
                } else {
                    for channel in 0..num_channels {
                        data[frame_idx * num_channels + channel] = converted;
                    }
                }
            }

            copied_frames += count;
            if count < batch {
                break;
            }
        }

        if copied_frames < frames_needed {
            // Until a callback is fully served after real audio was queued, the
            // queue has no cushion: running dry is start-up priming, not a dropout.
            // That ends OUTPUT_PRIMING_MAX_US after the first underrun that follows
            // real audio, so a stream that never catches up still reports its
            // dropouts. The writer restamps the marker on every write, so the
            // bound runs from its own start stamp. Before the first write (marker
            // 0) the supervisor's write-age check applies.
            let priming = !underruns.primed.load(Ordering::Relaxed)
                && (queued_at_us == 0 || {
                    let started_us = match underruns.priming_started_at_us.compare_exchange(
                        0,
                        now_us,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => now_us,
                        Err(earlier_us) => earlier_us,
                    };
                    now_us.saturating_sub(started_us) < OUTPUT_PRIMING_MAX_US
                });
            if priming {
                underruns.priming_total.fetch_add(1, Ordering::Relaxed);
            } else {
                underruns.streak.fetch_add(1, Ordering::Relaxed);
                underruns.total.fetch_add(1, Ordering::Relaxed);
            }
            Self::fill_underrun_tail(data, copied_frames, num_channels, consumer.last_sample());
            consumer.set_last_sample(0.0);
            consumer.set_resume_fade_remaining(UNDERRUN_FADE_FRAMES);
        } else {
            underruns.streak.store(0, Ordering::Relaxed);
            if queued_at_us != 0 {
                underruns.primed.store(true, Ordering::Relaxed);
            }
        }

        copied_frames
    }
    // RT_REGION_END: cpal_output_renderer

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
        underruns: OutputUnderruns,
        error_count: Arc<AtomicU64>,
        rt_error_code: Arc<AtomicU32>,
        restart_requested: Arc<AtomicBool>,
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
                    let now_us = now_micros();
                    last_callback_time_us.store(now_us, Ordering::Relaxed);

                    if let Some(probe) = output_probe.as_mut() {
                        if probe.cancel_requested.load(Ordering::Acquire)
                            || output_muted_clone.load(Ordering::Acquire)
                        {
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
                            underruns.streak.store(0, Ordering::Relaxed);
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
                        underruns.streak.store(0, Ordering::Relaxed);
                        let silence = Self::convert_output_sample::<T>(0.0);
                        for sample in data.iter_mut() {
                            *sample = silence;
                        }
                        return;
                    }

                    Self::render_output(
                        data,
                        num_channels,
                        &mut consumer,
                        &mut mono_scratch,
                        now_us,
                        &underruns,
                    );
                    // RT_REGION_END: cpal_output_callback
                },
                move |err| {
                    handle_stream_error(
                        &err,
                        RtErrorCode::OutputStreamError,
                        &error_count,
                        &rt_error_code,
                        &restart_requested,
                    );
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
        underruns: OutputUnderruns,
        error_count: Arc<AtomicU64>,
        rt_error_code: Arc<AtomicU32>,
        restart_requested: Arc<AtomicBool>,
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

        dispatch_sample_format!(
            sample_format,
            S => Self::build_stream::<S>(
                setup.device,
                stream_config,
                setup.device_info,
                consumer,
                recording_active,
                output_muted,
                last_callback_time_us,
                underruns,
                error_count,
                rt_error_code,
                restart_requested,
                output_probe,
                fixed_buffer_frames,
            ),
            _ => Err(AudioError::UnsupportedSampleFormat(sample_format.to_string()))
        )
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
        if let Ok(name) = device_name(&device) {
            if let Ok(config) = device.default_output_config() {
                devices.push(AudioDeviceInfo {
                    name,
                    endpoint_id: device
                        .id()
                        .ok()
                        .map(|device_id| persisted_endpoint_id(&device_id)),
                    sample_rate: config.sample_rate(),
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

    fn queued_underruns(streak: u32, total: u64) -> OutputUnderruns {
        OutputUnderruns {
            streak: Arc::new(AtomicU32::new(streak)),
            total: Arc::new(AtomicU64::new(total)),
            priming_total: Arc::new(AtomicU64::new(0)),
            audio_queued_at_us: Arc::new(AtomicU64::new(1)),
            priming_started_at_us: Arc::new(AtomicU64::new(0)),
            primed: Arc::new(AtomicBool::new(true)),
        }
    }

    /// State right after a start: nothing but the silent prime has been queued.
    fn priming_underruns() -> OutputUnderruns {
        let underruns = queued_underruns(0, 0);
        underruns.audio_queued_at_us.store(0, Ordering::Relaxed);
        underruns.primed.store(false, Ordering::Relaxed);
        underruns
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
    fn test_render_output_reads_full_request_in_scratch_batches() {
        let rb = crate::audio::AudioRingBuffer::new(16);
        let (mut producer, mut consumer) = rb.split();
        let source = [0.1_f32, -0.2, 0.3, -0.4, 0.5];
        assert_eq!(producer.write(&source), source.len());

        let mut output = [0.0_f32; 5];
        let mut scratch = [0.0_f32; 2];
        let underruns = queued_underruns(3, 7);

        let mut copied = 0;
        crate::test_alloc::assert_no_allocations("output renderer full read", || {
            copied = AudioOutput::render_output(
                &mut output,
                1,
                &mut consumer,
                &mut scratch,
                0,
                &underruns,
            );
        });

        assert_eq!(copied, source.len());
        assert_eq!(output, source);
        assert_eq!(underruns.streak.load(Ordering::Relaxed), 0);
        assert_eq!(underruns.total.load(Ordering::Relaxed), 7);
    }

    #[test]
    fn test_render_output_counts_short_read_and_fills_tail() {
        let rb = crate::audio::AudioRingBuffer::new(16);
        let (mut producer, mut consumer) = rb.split();
        let source = [0.5_f32, -0.25];
        assert_eq!(producer.write(&source), source.len());

        let mut output = [9.0_f32; 5];
        let mut scratch = [0.0_f32; 3];
        let underruns = queued_underruns(2, 4);

        let mut copied = 0;
        crate::test_alloc::assert_no_allocations("output renderer short mono read", || {
            copied = AudioOutput::render_output(
                &mut output,
                1,
                &mut consumer,
                &mut scratch,
                0,
                &underruns,
            );
        });

        assert_eq!(copied, source.len());
        assert_eq!(&output[..source.len()], &source);
        assert!(output[2] < 0.0);
        assert_eq!(output[4], 0.0);
        assert_eq!(consumer.last_sample(), 0.0);
        assert_eq!(underruns.streak.load(Ordering::Relaxed), 3);
        assert_eq!(underruns.total.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_render_output_counts_short_stereo_and_empty_reads() {
        let rb = crate::audio::AudioRingBuffer::new(16);
        let (mut producer, mut consumer) = rb.split();
        assert_eq!(producer.write(&[0.5_f32]), 1);

        let mut output = [9.0_f32; 6];
        let mut scratch = [0.0_f32; 2];
        let underruns = queued_underruns(4, 8);

        let mut copied = 0;
        crate::test_alloc::assert_no_allocations("output renderer short stereo read", || {
            copied = AudioOutput::render_output(
                &mut output,
                2,
                &mut consumer,
                &mut scratch,
                0,
                &underruns,
            );
        });

        assert_eq!(copied, 1);
        assert_eq!(&output[..2], &[0.5, 0.5]);
        assert!(output[2] > 0.0);
        assert_eq!(output[4], 0.0);
        assert_eq!(output[5], 0.0);
        assert_eq!(consumer.last_sample(), 0.0);
        assert_eq!(underruns.streak.load(Ordering::Relaxed), 5);
        assert_eq!(underruns.total.load(Ordering::Relaxed), 9);

        let rb = crate::audio::AudioRingBuffer::new(16);
        let (_producer, mut consumer) = rb.split();
        let mut output = [9.0_f32; 4];
        let mut scratch = [0.0_f32; 2];
        let underruns = queued_underruns(0, 0);

        let mut copied = 0;
        crate::test_alloc::assert_no_allocations("output renderer empty read", || {
            copied = AudioOutput::render_output(
                &mut output,
                2,
                &mut consumer,
                &mut scratch,
                0,
                &underruns,
            );
        });

        assert_eq!(copied, 0);
        assert_eq!(output, [0.0; 4]);
        assert_eq!(underruns.streak.load(Ordering::Relaxed), 1);
        assert_eq!(underruns.total.load(Ordering::Relaxed), 1);
    }

    /// Serve 480-frame mono callbacks of a 997 Hz tone at 0.5, with callback
    /// `dropout` finding the queue empty when `dropout` is set.
    fn render_dropout(dropout: Option<usize>) -> Vec<f32> {
        use crate::test_signal::sine;
        let rb = crate::audio::AudioRingBuffer::new(4_096);
        let (mut producer, mut consumer) = rb.split();
        let underruns = queued_underruns(0, 0);
        let mut scratch = [0.0_f32; 256];
        let mut next = 0;
        let mut rendered = Vec::new();
        for callback in 0..20 {
            if Some(callback) != dropout {
                let block: Vec<f32> = (next..next + 480)
                    .map(|n| sine(997.0, 0.5, 48_000.0, n))
                    .collect();
                assert_eq!(producer.write(&block), 480);
                next += 480;
            }
            let mut output = [0.0_f32; 480];
            AudioOutput::render_output(&mut output, 1, &mut consumer, &mut scratch, 0, &underruns);
            rendered.extend_from_slice(&output);
        }
        rendered
    }

    #[test]
    fn test_audio_after_an_underrun_fades_back_in() {
        use crate::test_signal::max_residual;
        let output = render_dropout(Some(10));
        let clean = render_dropout(None);
        let max_step = |range: std::ops::Range<usize>| {
            range
                .map(|n| (output[n] - output[n - 1]).abs())
                .fold(0.0_f32, f32::max)
        };
        // Resuming at full level stepped 0.476 out of silence (residual 0.476);
        // now 0.055, under the tone's own largest step of 0.065, and a
        // residual of 0.012, under the existing fade-out's 0.024.
        let resume_step = max_step(5_270..5_340);
        let steady_step = max_step(1..4_800);
        assert!(
            resume_step <= steady_step,
            "resume step {resume_step}, tone {steady_step}"
        );
        let resume = max_residual(&output, &[997.0], 48_000.0, 5_270, 5_400);
        let fade_out = max_residual(&output, &[997.0], 48_000.0, 4_790, 4_900);
        assert!(
            resume < fade_out,
            "resume residual {resume}, fade-out {fade_out}"
        );

        // Steady state: bit-identical before the dropout and from 64 frames
        // after the resume, where the tone continues one callback late.
        assert_eq!(output[..4_800], clean[..4_800]);
        assert!((5_344..9_600).all(|n| output[n] == clean[n - 480]));
    }

    fn render_empty_queue(underruns: &OutputUnderruns) {
        render_empty_queue_at(underruns, 0);
    }

    fn render_empty_queue_at(underruns: &OutputUnderruns, now_us: u64) {
        let rb = crate::audio::AudioRingBuffer::new(16);
        let (_producer, mut consumer) = rb.split();
        let mut output = [9.0_f32; 4];
        let mut scratch = [0.0_f32; 2];
        crate::test_alloc::assert_no_allocations("output renderer empty queue", || {
            AudioOutput::render_output(
                &mut output,
                2,
                &mut consumer,
                &mut scratch,
                now_us,
                underruns,
            );
        });
        assert_eq!(output, [0.0; 4]);
    }

    fn render_samples(underruns: &OutputUnderruns, queued: usize) -> usize {
        let rb = crate::audio::AudioRingBuffer::new(16);
        let (mut producer, mut consumer) = rb.split();
        producer.write(&[0.25_f32; 8][..queued]);
        let mut output = [9.0_f32; 4];
        let mut scratch = [0.0_f32; 2];
        AudioOutput::render_output(&mut output, 2, &mut consumer, &mut scratch, 0, underruns)
    }

    fn counts(underruns: &OutputUnderruns) -> (u64, u64, u32) {
        (
            underruns.priming_total.load(Ordering::Relaxed),
            underruns.total.load(Ordering::Relaxed),
            underruns.streak.load(Ordering::Relaxed),
        )
    }

    #[test]
    fn test_render_output_empty_queue_before_first_real_audio_is_priming_not_dropout() {
        let underruns = priming_underruns();

        render_empty_queue(&underruns);
        render_empty_queue(&underruns);
        render_empty_queue(&underruns);

        assert_eq!(counts(&underruns), (3, 0, 0));
        assert!(!underruns.primed.load(Ordering::Relaxed));
    }

    #[test]
    fn test_prime_silence_alone_never_ends_priming() {
        // The silent prime is written straight to the queue, so the marker stays 0.
        let underruns = priming_underruns();

        assert_eq!(render_samples(&underruns, 2), 2);
        assert!(!underruns.primed.load(Ordering::Relaxed));
        render_empty_queue(&underruns);

        assert_eq!(counts(&underruns), (1, 0, 0));
    }

    #[test]
    fn test_dry_callbacks_after_first_real_write_are_still_priming() {
        let underruns = priming_underruns();
        // The DSP loop has queued its first block, but the prime silence is gone
        // and no callback has been fully served yet.
        underruns.audio_queued_at_us.store(42, Ordering::Relaxed);

        render_empty_queue(&underruns);
        render_empty_queue(&underruns);

        assert_eq!(counts(&underruns), (2, 0, 0));
        assert!(!underruns.primed.load(Ordering::Relaxed));
    }

    #[test]
    fn test_priming_is_bounded_when_no_callback_is_ever_fully_served() {
        let underruns = priming_underruns();
        let first_write_us = 5_000_000;
        // The real writer restamps the marker on every successful write, so the
        // bound must run from the first write of this start, not from the marker.
        let callback = |now_us: u64| {
            underruns
                .audio_queued_at_us
                .store(now_us, Ordering::Relaxed);
            render_empty_queue_at(&underruns, now_us);
        };

        let mut now_us = first_write_us;
        while now_us < first_write_us + OUTPUT_PRIMING_MAX_US {
            callback(now_us);
            now_us += 10_000;
        }
        let priming_callbacks = counts(&underruns).0;
        assert_eq!(priming_callbacks, OUTPUT_PRIMING_MAX_US / 10_000);
        assert_eq!(counts(&underruns), (priming_callbacks, 0, 0));

        // A stream that keeps writing but never catches up shows its dropouts again.
        callback(first_write_us + OUTPUT_PRIMING_MAX_US);
        callback(first_write_us + OUTPUT_PRIMING_MAX_US + 10_000);
        assert_eq!(counts(&underruns), (priming_callbacks, 2, 2));
        assert!(!underruns.primed.load(Ordering::Relaxed));
    }

    #[test]
    fn test_underrun_after_a_fully_served_real_callback_is_a_dropout() {
        let underruns = priming_underruns();
        underruns.audio_queued_at_us.store(42, Ordering::Relaxed);
        render_empty_queue(&underruns);

        assert_eq!(render_samples(&underruns, 2), 2);
        assert!(underruns.primed.load(Ordering::Relaxed));
        render_empty_queue(&underruns);
        render_empty_queue(&underruns);

        assert_eq!(counts(&underruns), (1, 2, 2));
    }

    #[test]
    fn test_restart_returns_to_priming() {
        let underruns = queued_underruns(0, 0);
        render_empty_queue(&underruns);
        assert_eq!(counts(&underruns), (0, 1, 1));

        // What a start does before it builds the stream.
        underruns.audio_queued_at_us.store(0, Ordering::Relaxed);
        underruns.priming_started_at_us.store(0, Ordering::Relaxed);
        underruns.primed.store(false, Ordering::Relaxed);
        render_empty_queue(&underruns);

        assert_eq!(counts(&underruns), (1, 1, 1));
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

    #[test]
    fn test_24_bit_output_formats_render_from_a_synthetic_queue() {
        use crate::audio::buffer::AudioRingBuffer;
        use cpal::{I24, U24};

        let (mut producer, mut consumer) = AudioRingBuffer::new(8).split();
        assert_eq!(producer.write(&[0.5, -0.5, 1.0, -1.0]), 4);
        let mut scratch = [0.0_f32; 4];
        let underruns = queued_underruns(0, 0);
        let mut signed = [I24::new(0).unwrap(); 4];
        let copied = AudioOutput::render_output::<I24>(
            &mut signed,
            1,
            &mut consumer,
            &mut scratch,
            0,
            &underruns,
        );
        assert_eq!(copied, 4);
        let signed: Vec<i32> = signed.iter().map(|sample| sample.inner()).collect();
        assert!((signed[0] - 4_194_304).abs() <= 1, "{signed:?}");
        assert!((signed[1] + 4_194_304).abs() <= 1, "{signed:?}");
        assert!(
            signed[2] >= 8_388_606 && signed[3] <= -8_388_607,
            "{signed:?}"
        );

        assert_eq!(producer.write(&[0.0, 1.0, -1.0, 0.5]), 4);
        let mut unsigned = [U24::new(0).unwrap(); 4];
        let copied = AudioOutput::render_output::<U24>(
            &mut unsigned,
            1,
            &mut consumer,
            &mut scratch,
            0,
            &underruns,
        );
        assert_eq!(copied, 4);
        let unsigned: Vec<i32> = unsigned.iter().map(|sample| sample.inner()).collect();
        assert!((unsigned[0] - 8_388_608).abs() <= 1, "{unsigned:?}");
        assert!(
            unsigned[1] >= 16_777_214 && unsigned[2] <= 1,
            "{unsigned:?}"
        );
        assert!((unsigned[3] - 12_582_912).abs() <= 1, "{unsigned:?}");
    }
}
