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
#[cfg(windows)]
use super::rt::store_rt_error;
use super::rt::RtErrorCode;
#[cfg(windows)]
use super::wasapi_render::{
    probe_standard_buffer, short_period_is_verified, LowPeriodStream, Opened,
};
use super::{
    find_48khz_config, handle_stream_error, requested_fixed_buffer_frames, stream_config_arg,
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

/// What the queue must hold beyond one request before a start's first real audio
/// plays. Blocks arrive every 10 ms, so a stream that starts on less runs dry
/// before the next one on a short callback, and the first blocks come unevenly.
const OUTPUT_START_RESERVE_MS: u32 = 15;

/// Frames over which an underrun fades out to silence and the next audio fades back in.
const UNDERRUN_FADE_FRAMES: usize = 64;

/// Output-callback underrun accounting shared with the processor.
pub(crate) struct OutputUnderruns {
    pub(crate) streak: Arc<AtomicU32>,
    pub(crate) total: Arc<AtomicU64>,
    /// Callbacks with nothing or too little to play before the stream is primed
    /// (see `primed`); a shorter callback counts more of them per second.
    pub(crate) priming_total: Arc<AtomicU64>,
    /// Time of the DSP loop's first queued write (0 until then); a start resets it.
    pub(crate) audio_queued_at_us: Arc<AtomicU64>,
    /// Callback time of the first underrun seen after real audio was queued
    /// (0 until then); a start resets it. Bounds priming, see `OUTPUT_PRIMING_MAX_US`.
    pub(crate) priming_started_at_us: Arc<AtomicU64>,
    /// Set by the callback once it fully serves a request after real audio was
    /// queued, i.e. the queue has a cushion; a start resets it.
    pub(crate) primed: Arc<AtomicBool>,
    /// Largest request since the stream was primed; the first ones can ask for
    /// the whole device buffer. Diagnostic only; a start resets it.
    pub(crate) largest_request_frames: Arc<AtomicU32>,
    /// Largest request that the last few callbacks all reached. One late
    /// callback asks for the whole device buffer, so it cannot move this. A
    /// start resets it.
    pub(crate) sustained_request_frames: Arc<AtomicU32>,
}

const REQUEST_HISTORY_LEN: usize = 4;

/// The last few output request sizes of one stream.
#[derive(Default)]
struct RequestHistory {
    recent: [u32; REQUEST_HISTORY_LEN],
    filled: usize,
    next: usize,
}

impl RequestHistory {
    /// The smallest of the last `REQUEST_HISTORY_LEN` requests, once that many
    /// have been recorded.
    fn record(&mut self, frames: u32) -> Option<u32> {
        self.recent[self.next] = frames;
        self.next = (self.next + 1) % REQUEST_HISTORY_LEN;
        self.filled = (self.filled + 1).min(REQUEST_HISTORY_LEN);
        if self.filled == REQUEST_HISTORY_LEN {
            self.recent.iter().copied().min()
        } else {
            None
        }
    }
}

impl OutputUnderruns {
    fn record_request(&self, history: &mut RequestHistory, frames: usize) {
        let frames = frames as u32;
        self.largest_request_frames
            .fetch_max(frames, Ordering::Relaxed);
        if let Some(sustained) = history.record(frames) {
            self.sustained_request_frames
                .fetch_max(sustained, Ordering::Relaxed);
        }
    }
}

enum OutputBackend {
    Cpal(Stream),
    #[cfg(windows)]
    LowPeriod(LowPeriodStream),
}

/// What the output device's own buffering holds after the queue, read once when
/// the stream opens. Frames are at the stream's sample rate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OutputDeviceBuffer {
    /// Frames the device buffers ahead of the speaker; 0 when unknown.
    pub frames: u32,
    /// The callback period in frames.
    pub period_frames: u32,
    /// HRESULT of the failed read; 0 when it was read or not attempted.
    pub failure_code: u32,
}

/// Audio output stream
pub struct AudioOutput {
    stream: OutputBackend,
    device_info: AudioDeviceInfo,
    fixed_buffer_frames: Option<u32>,
    device_buffer: OutputDeviceBuffer,
}

/// Where a running output stream reports that it has died.
#[derive(Clone)]
pub(crate) struct OutputFailure {
    pub(crate) error_count: Arc<AtomicU64>,
    pub(crate) rt_error_code: Arc<AtomicU32>,
    pub(crate) restart_requested: Arc<AtomicBool>,
    /// HRESULT that kept the stream off a short period or ended it there; 0 if none.
    #[cfg(windows)]
    pub(crate) low_period_failure_code: Arc<AtomicU32>,
}

#[cfg(windows)]
impl OutputFailure {
    /// The same as a cpal stream that reports its device gone.
    pub(crate) fn report(&self, hresult: u32) {
        self.low_period_failure_code
            .store(hresult, Ordering::Relaxed);
        self.error_count.fetch_add(1, Ordering::Relaxed);
        store_rt_error(&self.rt_error_code, RtErrorCode::OutputStreamError);
        self.restart_requested.store(true, Ordering::Release);
    }
}

/// What the output callback owns, on either backend.
pub(crate) struct OutputRenderer {
    consumer: AudioConsumer,
    num_channels: usize,
    mono_scratch: Vec<f32>,
    output_probe: Option<OutputProbeControl>,
    request_history: RequestHistory,
    recording_active: Arc<AtomicBool>,
    output_muted: Arc<AtomicBool>,
    last_callback_time_us: Arc<AtomicU64>,
    underruns: OutputUnderruns,
    start_reserve_frames: usize,
}

impl OutputRenderer {
    pub(crate) fn render<T>(&mut self, data: &mut [T])
    where
        T: SizedSample + FromSample<f32>,
    {
        // RT_REGION_START: cpal_output_callback
        let num_channels = self.num_channels;
        let now_us = now_micros();
        self.last_callback_time_us.store(now_us, Ordering::Relaxed);

        if let Some(probe) = self.output_probe.as_mut() {
            if probe.cancel_requested.load(Ordering::Acquire)
                || self.output_muted.load(Ordering::Acquire)
            {
                AudioOutput::discard_probe(probe, &mut self.mono_scratch);
            } else if probe.active.load(Ordering::Acquire) {
                let frames_needed = data.len() / num_channels;
                AudioOutput::drain_muted_consumer(
                    &mut self.consumer,
                    frames_needed,
                    &mut self.mono_scratch,
                );
                self.underruns.streak.store(0, Ordering::Relaxed);
                let complete =
                    AudioOutput::render_probe(data, num_channels, probe, &mut self.mono_scratch);
                if complete {
                    probe.active.store(false, Ordering::Release);
                    probe.complete.store(true, Ordering::Release);
                }
                return;
            }
        }

        if self.recording_active.load(Ordering::Relaxed)
            || self.output_muted.load(Ordering::Relaxed)
        {
            AudioOutput::render_muted(
                data,
                num_channels,
                &mut self.consumer,
                &mut self.mono_scratch,
            );
            self.underruns.streak.store(0, Ordering::Relaxed);
            return;
        }

        if AudioOutput::hold_for_start_reserve(
            data,
            num_channels,
            &mut self.consumer,
            now_us,
            &self.underruns,
            self.start_reserve_frames,
        ) {
            return;
        }

        if self.underruns.primed.load(Ordering::Relaxed) {
            self.underruns
                .record_request(&mut self.request_history, data.len() / num_channels);
        }
        AudioOutput::render_output(
            data,
            num_channels,
            &mut self.consumer,
            &mut self.mono_scratch,
            now_us,
            &self.underruns,
        );
        // RT_REGION_END: cpal_output_callback
    }
}

pub(crate) struct OutputStreamSetup {
    pub(crate) device: Device,
    pub(crate) supported_config: SupportedStreamConfig,
    pub(crate) device_info: AudioDeviceInfo,
}

impl AudioOutput {
    fn requested_fixed_buffer_frames() -> Option<u32> {
        let value = std::env::var("AUDIOFORGE_FIXED_OUTPUT_BUFFER_FRAMES").ok();
        requested_fixed_buffer_frames(value.as_deref())
    }

    fn preflight_config<T>(device: &Device, config: &StreamConfig) -> bool
    where
        T: SizedSample + FromSample<f32>,
    {
        device
            .build_output_stream(
                stream_config_arg(config),
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
        let total_frames = data.len() / num_channels;
        let remaining_frames = total_frames.saturating_sub(copied_frames);
        if remaining_frames == 0 {
            return;
        }

        let fade_frames = remaining_frames.min(UNDERRUN_FADE_FRAMES);
        let tail = |i: usize| -> f32 {
            if i >= fade_frames {
                0.0
            } else if last_sample == 0.0 {
                // A zero of either sign times the non-negative gain is itself.
                last_sample
            } else {
                let t = (i + 1) as f32 / fade_frames as f32;
                let gain = ((1.0 - t) * std::f32::consts::FRAC_PI_2).sin();
                last_sample * gain
            }
        };
        if num_channels == 1 {
            for i in 0..remaining_frames {
                data[copied_frames + i] = Self::convert_output_sample::<T>(tail(i));
            }
        } else {
            for i in 0..remaining_frames {
                let converted = Self::convert_output_sample::<T>(tail(i));
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

    /// Stamps the start of priming on first use; true until the bound has passed.
    fn priming_bound_open(underruns: &OutputUnderruns, now_us: u64) -> bool {
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
    }

    /// Plays silence, as priming, while a start's first real audio is too little
    /// to last until the next block. Bounded like priming, so a stream that never
    /// builds the reserve plays what it has and shows its dropouts.
    fn hold_for_start_reserve<T>(
        data: &mut [T],
        num_channels: usize,
        consumer: &mut AudioConsumer,
        now_us: u64,
        underruns: &OutputUnderruns,
        reserve_frames: usize,
    ) -> bool
    where
        T: SizedSample + FromSample<f32>,
    {
        if underruns.primed.load(Ordering::Relaxed)
            || underruns.audio_queued_at_us.load(Ordering::Relaxed) == 0
            || consumer.len() >= data.len() / num_channels + reserve_frames
            || !Self::priming_bound_open(underruns, now_us)
        {
            return false;
        }
        underruns.priming_total.fetch_add(1, Ordering::Relaxed);
        Self::fill_underrun_tail(data, 0, num_channels, 0.0);
        consumer.set_last_sample(0.0);
        consumer.set_resume_fade_remaining(UNDERRUN_FADE_FRAMES);
        true
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
                && (queued_at_us == 0 || Self::priming_bound_open(underruns, now_us));
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
        // Unmute, the end of a recording and the end of a probe resume out of silence.
        consumer.set_resume_fade_remaining(UNDERRUN_FADE_FRAMES);
        drained
    }

    /// Drain the queue and play silence while muted or recording; the cut
    /// fades from the last played sample like an underrun tail.
    fn render_muted<T>(
        data: &mut [T],
        num_channels: usize,
        consumer: &mut AudioConsumer,
        scratch: &mut [f32],
    ) where
        T: SizedSample + FromSample<f32>,
    {
        let frames_needed = data.len() / num_channels;
        let last_played = consumer.last_sample();
        Self::drain_muted_consumer(consumer, frames_needed, scratch);
        Self::fill_underrun_tail(data, 0, num_channels, last_played);
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
        let frame_count = data.len() / num_channels;
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

    fn build_stream<T>(
        device: Device,
        stream_config: StreamConfig,
        mut renderer: OutputRenderer,
        failure: OutputFailure,
    ) -> Result<Stream, AudioError>
    where
        T: SizedSample + FromSample<f32>,
    {
        device
            .build_output_stream(
                stream_config_arg(&stream_config),
                move |data: &mut [T], _: &cpal::OutputCallbackInfo| renderer.render(data),
                move |err| {
                    handle_stream_error(
                        &err,
                        RtErrorCode::OutputStreamError,
                        &failure.error_count,
                        &failure.rt_error_code,
                        &failure.restart_requested,
                    );
                },
                None,
            )
            .map_err(|e| AudioError::BuildStream(e.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_setup(
        setup: OutputStreamSetup,
        consumer: AudioConsumer,
        recording_active: Arc<AtomicBool>,
        output_muted: Arc<AtomicBool>,
        last_callback_time_us: Arc<AtomicU64>,
        underruns: OutputUnderruns,
        failure: OutputFailure,
        output_probe: Option<OutputProbeControl>,
    ) -> Result<Self, AudioError> {
        const OUTPUT_SCRATCH_CAPACITY: usize = 8192;
        let renderer = OutputRenderer {
            consumer,
            num_channels: setup.device_info.channels as usize,
            mono_scratch: vec![0.0; OUTPUT_SCRATCH_CAPACITY],
            output_probe,
            request_history: RequestHistory::default(),
            recording_active,
            output_muted,
            last_callback_time_us,
            underruns,
            start_reserve_frames: (setup.device_info.sample_rate * OUTPUT_START_RESERVE_MS / 1000)
                as usize,
        };

        #[cfg(windows)]
        let renderer = {
            let mut renderer = renderer;
            // The variable asks for a cpal buffer, and 0 keeps cpal's default.
            let opted_out = std::env::var_os("AUDIOFORGE_FIXED_OUTPUT_BUFFER_FRAMES").is_some();
            let endpoint_id = setup
                .device_info
                .endpoint_id
                .clone()
                .filter(|_| !opted_out && short_period_is_verified(&setup.device_info.name));
            if let Some(endpoint_id) = endpoint_id {
                match LowPeriodStream::open(
                    endpoint_id,
                    setup.device_info.sample_rate,
                    setup.device_info.channels,
                    renderer,
                    failure.clone(),
                ) {
                    Opened::Running(stream) => {
                        return Ok(Self {
                            device_buffer: stream.device_buffer(),
                            stream: OutputBackend::LowPeriod(stream),
                            device_info: setup.device_info,
                            fixed_buffer_frames: None,
                        });
                    }
                    Opened::Declined {
                        renderer: returned,
                        failure_code: code,
                    } => {
                        renderer = *returned;
                        failure
                            .low_period_failure_code
                            .store(code, Ordering::Relaxed);
                    }
                }
            }
            renderer
        };

        let sample_format = setup.supported_config.sample_format();
        let mut stream_config = setup.supported_config.config();
        let mut fixed_buffer_frames = None;
        if let Some(frames) = Self::requested_fixed_buffer_frames() {
            if supported_fixed_buffer_frames(frames, setup.supported_config.buffer_size()) {
                let candidate = StreamConfig {
                    buffer_size: BufferSize::Fixed(frames),
                    ..stream_config
                };
                if Self::preflight_sample_format(&setup.device, sample_format, &candidate) {
                    stream_config = candidate;
                    fixed_buffer_frames = Some(frames);
                }
            }
        }

        #[cfg(windows)]
        let device_buffer = probe_standard_buffer(
            setup.device_info.endpoint_id.as_deref(),
            setup.device_info.sample_rate,
            setup.device_info.channels,
            sample_format,
            fixed_buffer_frames,
        );
        #[cfg(not(windows))]
        let device_buffer = OutputDeviceBuffer::default();

        let stream = dispatch_sample_format!(
            sample_format,
            S => Self::build_stream::<S>(setup.device, stream_config, renderer, failure),
            _ => Err(AudioError::UnsupportedSampleFormat(sample_format.to_string()))
        )?;
        Ok(Self {
            stream: OutputBackend::Cpal(stream),
            device_info: setup.device_info,
            fixed_buffer_frames,
            device_buffer,
        })
    }

    /// Start playing audio
    pub fn start(&self) -> Result<(), AudioError> {
        match &self.stream {
            OutputBackend::Cpal(stream) => stream
                .play()
                .map_err(|e| AudioError::PlayStream(e.to_string())),
            #[cfg(windows)]
            OutputBackend::LowPeriod(stream) => {
                stream.start();
                Ok(())
            }
        }
    }

    /// Pause audio playback
    pub fn pause(&self) -> Result<(), AudioError> {
        match &self.stream {
            OutputBackend::Cpal(stream) => stream
                .pause()
                .map_err(|e| AudioError::PlayStream(e.to_string())),
            #[cfg(windows)]
            OutputBackend::LowPeriod(stream) => {
                stream.pause();
                Ok(())
            }
        }
    }

    /// Get device information
    pub fn device_info(&self) -> &AudioDeviceInfo {
        &self.device_info
    }

    pub fn fixed_buffer_frames(&self) -> Option<u32> {
        self.fixed_buffer_frames
    }

    /// The device buffering read when the stream opened.
    pub fn device_buffer(&self) -> OutputDeviceBuffer {
        self.device_buffer
    }

    /// The short engine period this stream runs on; 0 on the cpal path.
    pub fn low_period_frames(&self) -> u32 {
        match &self.stream {
            OutputBackend::Cpal(_) => 0,
            #[cfg(windows)]
            OutputBackend::LowPeriod(stream) => stream.period_frames(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queued_underruns(streak: u32, total: u64) -> OutputUnderruns {
        OutputUnderruns {
            streak: Arc::new(AtomicU32::new(streak)),
            total: Arc::new(AtomicU64::new(total)),
            priming_total: Arc::new(AtomicU64::new(0)),
            audio_queued_at_us: Arc::new(AtomicU64::new(1)),
            priming_started_at_us: Arc::new(AtomicU64::new(0)),
            primed: Arc::new(AtomicBool::new(true)),
            largest_request_frames: Arc::new(AtomicU32::new(0)),
            sustained_request_frames: Arc::new(AtomicU32::new(0)),
        }
    }

    #[test]
    fn test_request_history_reports_after_four_requests() {
        let mut history = RequestHistory::default();
        assert_eq!(history.record(480), None);
        assert_eq!(history.record(480), None);
        assert_eq!(history.record(960), None);
        assert_eq!(history.record(480), Some(480));
    }

    #[test]
    fn test_request_history_ignores_a_single_late_callback() {
        let mut history = RequestHistory::default();
        let mut sustained = 0;
        for frames in [480, 480, 960, 480, 480, 480, 480, 960, 480, 480] {
            sustained = sustained.max(history.record(frames).unwrap_or(0));
        }
        assert_eq!(sustained, 480);
    }

    #[test]
    fn test_request_history_follows_a_real_longer_period() {
        let mut history = RequestHistory::default();
        for _ in 0..4 {
            history.record(960);
        }
        assert_eq!(history.record(960), Some(960));

        let mut history = RequestHistory::default();
        for _ in 0..4 {
            history.record(480);
        }
        let mut last = None;
        for _ in 0..4 {
            last = history.record(960);
        }
        assert_eq!(last, Some(960));
    }

    #[test]
    fn test_record_request_keeps_largest_and_sustained_apart() {
        let underruns = queued_underruns(0, 0);
        let mut history = RequestHistory::default();
        for frames in [480, 480, 960, 480, 480] {
            underruns.record_request(&mut history, frames);
        }
        assert_eq!(
            underruns.largest_request_frames.load(Ordering::Relaxed),
            960
        );
        assert_eq!(
            underruns.sustained_request_frames.load(Ordering::Relaxed),
            480
        );
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

    /// Like `render_dropout`, but callback 10 is muted while the DSP keeps
    /// queueing, so the tone stays on its timeline.
    fn render_mute(mute: Option<usize>) -> Vec<f32> {
        use crate::test_signal::sine;
        let rb = crate::audio::AudioRingBuffer::new(4_096);
        let (mut producer, mut consumer) = rb.split();
        let underruns = queued_underruns(0, 0);
        let mut scratch = [0.0_f32; 256];
        let mut rendered = Vec::new();
        for callback in 0..20 {
            let block: Vec<f32> = (callback * 480..callback * 480 + 480)
                .map(|n| sine(997.0, 0.5, 48_000.0, n))
                .collect();
            assert_eq!(producer.write(&block), 480);
            let mut output = [0.0_f32; 480];
            if Some(callback) == mute {
                AudioOutput::render_muted(&mut output, 1, &mut consumer, &mut scratch);
            } else {
                AudioOutput::render_output(
                    &mut output,
                    1,
                    &mut consumer,
                    &mut scratch,
                    0,
                    &underruns,
                );
            }
            rendered.extend_from_slice(&output);
        }
        rendered
    }

    #[test]
    fn test_mute_fades_out_and_unmute_fades_back_in() {
        let output = render_mute(Some(10));
        let clean = render_mute(None);
        let max_step = |range: std::ops::Range<usize>| {
            range
                .map(|n| (output[n] - output[n - 1]).abs())
                .fold(0.0_f32, f32::max)
        };
        // The hard cut stepped 0.451 into silence and unmute 0.438 out of it;
        // now 0.011 and 0.056, under the tone's own largest step of 0.065.
        let steady = max_step(1..4_800);
        let mute = max_step(4_800..4_900);
        let unmute = max_step(5_270..5_340);
        assert!(mute <= steady, "mute step {mute}, tone {steady}");
        assert!(unmute <= steady, "unmute step {unmute}, tone {steady}");

        // Steady state: bit-identical before the mute and from 64 frames
        // after the unmute; the muted callback is silent after its fade.
        assert_eq!(output[..4_800], clean[..4_800]);
        assert!(output[4_864..5_280].iter().all(|&sample| sample == 0.0));
        assert_eq!(output[5_344..], clean[5_344..]);
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

    const TEST_RESERVE: usize = 720;

    fn hold(underruns: &OutputUnderruns, queued: usize, now_us: u64) -> (bool, [f32; 8], usize) {
        let rb = crate::audio::AudioRingBuffer::new(1024);
        let (mut producer, mut consumer) = rb.split();
        producer.write(&vec![0.25_f32; queued]);
        let mut output = [9.0_f32; 8];
        let mut held = false;
        crate::test_alloc::assert_no_allocations("output start hold", || {
            held = AudioOutput::hold_for_start_reserve(
                &mut output,
                2,
                &mut consumer,
                now_us,
                underruns,
                TEST_RESERVE,
            );
        });
        (held, output, consumer.len())
    }

    #[test]
    fn test_first_real_audio_waits_for_one_request_plus_the_reserve() {
        let underruns = priming_underruns();
        underruns.audio_queued_at_us.store(42, Ordering::Relaxed);
        let enough = 4 + TEST_RESERVE;

        let (held, output, left) = hold(&underruns, enough - 1, 0);
        assert!(held);
        assert_eq!(output, [0.0; 8]);
        assert_eq!(left, enough - 1, "a held callback reads nothing");
        assert_eq!(counts(&underruns), (1, 0, 0));
        assert!(!underruns.primed.load(Ordering::Relaxed));

        let (held, output, left) = hold(&underruns, enough, 10);
        assert!(!held);
        assert_eq!(output, [9.0; 8], "the normal path renders this one");
        assert_eq!(left, enough);
    }

    #[test]
    fn test_held_start_fades_in_its_first_audio() {
        let underruns = priming_underruns();
        underruns.audio_queued_at_us.store(42, Ordering::Relaxed);
        let rb = crate::audio::AudioRingBuffer::new(16);
        let (_producer, mut consumer) = rb.split();
        let mut output = [9.0_f32; 8];
        assert!(AudioOutput::hold_for_start_reserve(
            &mut output,
            2,
            &mut consumer,
            0,
            &underruns,
            TEST_RESERVE,
        ));
        assert_eq!(consumer.resume_fade_remaining(), UNDERRUN_FADE_FRAMES);
    }

    #[test]
    fn test_hold_needs_real_audio_and_an_unprimed_stream() {
        // Before the first write the supervisor's write-age check applies.
        let underruns = priming_underruns();
        assert!(!hold(&underruns, 0, 0).0);
        // Later a short queue is a dropout, not a start.
        let underruns = queued_underruns(0, 0);
        assert!(!hold(&underruns, 0, 0).0);
    }

    #[test]
    fn test_hold_ends_at_the_priming_bound() {
        let underruns = priming_underruns();
        underruns.audio_queued_at_us.store(42, Ordering::Relaxed);
        assert!(hold(&underruns, 1, 5_000_000).0);
        assert!(hold(&underruns, 1, 5_000_000 + OUTPUT_PRIMING_MAX_US - 1).0);
        assert!(!hold(&underruns, 1, 5_000_000 + OUTPUT_PRIMING_MAX_US).0);
    }

    /// Real dropouts over a start on 128-frame callbacks, where the first 10 ms
    /// blocks come unevenly: 480 samples at 0, 30, 40, 50 ... ms.
    fn dropouts_over_an_uneven_start(reserve_frames: usize) -> u64 {
        const CALLBACK_US: u64 = 2_667;
        let underruns = priming_underruns();
        let rb = crate::audio::AudioRingBuffer::new(4096);
        let (mut producer, mut consumer) = rb.split();
        let mut output = [0.0_f32; 128];
        let mut scratch = [0.0_f32; 256];
        let mut next_write_us = 0;
        let mut now_us = 0;
        while now_us < 200_000 {
            if now_us >= next_write_us {
                producer.write(&[0.25_f32; 480]);
                underruns
                    .audio_queued_at_us
                    .store(now_us + 1, Ordering::Relaxed);
                next_write_us += if next_write_us == 0 { 30_000 } else { 10_000 };
            }
            if !AudioOutput::hold_for_start_reserve(
                &mut output,
                1,
                &mut consumer,
                now_us,
                &underruns,
                reserve_frames,
            ) {
                AudioOutput::render_output(
                    &mut output,
                    1,
                    &mut consumer,
                    &mut scratch,
                    now_us,
                    &underruns,
                );
            }
            now_us += CALLBACK_US;
        }
        underruns.total.load(Ordering::Relaxed)
    }

    #[test]
    fn test_start_reserve_removes_the_dropouts_of_an_uneven_start() {
        assert!(dropouts_over_an_uneven_start(0) > 0);
        assert_eq!(dropouts_over_an_uneven_start(TEST_RESERVE), 0);
    }

    fn renderer_for_test(consumer: AudioConsumer, underruns: OutputUnderruns) -> OutputRenderer {
        OutputRenderer {
            consumer,
            num_channels: 2,
            mono_scratch: vec![0.0; 256],
            output_probe: None,
            request_history: RequestHistory::default(),
            recording_active: Arc::new(AtomicBool::new(false)),
            output_muted: Arc::new(AtomicBool::new(false)),
            last_callback_time_us: Arc::new(AtomicU64::new(0)),
            underruns,
            start_reserve_frames: TEST_RESERVE,
        }
    }

    #[test]
    fn test_muted_callback_never_holds() {
        let underruns = priming_underruns();
        underruns.audio_queued_at_us.store(42, Ordering::Relaxed);
        let rb = crate::audio::AudioRingBuffer::new(16);
        let (_producer, consumer) = rb.split();
        let mut renderer = renderer_for_test(consumer, underruns);
        renderer.output_muted.store(true, Ordering::Relaxed);

        renderer.render(&mut [9.0_f32; 8]);

        assert_eq!(renderer.underruns.priming_total.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_held_callback_records_no_request_and_a_primed_one_does() {
        let underruns = priming_underruns();
        underruns.audio_queued_at_us.store(42, Ordering::Relaxed);
        let rb = crate::audio::AudioRingBuffer::new(16);
        let (_producer, consumer) = rb.split();
        let mut renderer = renderer_for_test(consumer, underruns);

        renderer.render(&mut [9.0_f32; 8]);
        assert_eq!(renderer.underruns.priming_total.load(Ordering::Relaxed), 1);
        assert_eq!(
            renderer
                .underruns
                .largest_request_frames
                .load(Ordering::Relaxed),
            0
        );

        renderer.underruns.primed.store(true, Ordering::Relaxed);
        renderer.render(&mut [9.0_f32; 8]);
        assert_eq!(
            renderer
                .underruns
                .largest_request_frames
                .load(Ordering::Relaxed),
            4
        );
    }

    #[cfg(windows)]
    #[test]
    fn test_stream_failure_reports_its_code_and_asks_for_a_restart() {
        let failure = OutputFailure {
            error_count: Arc::new(AtomicU64::new(0)),
            rt_error_code: Arc::new(AtomicU32::new(0)),
            restart_requested: Arc::new(AtomicBool::new(false)),
            low_period_failure_code: Arc::new(AtomicU32::new(0)),
        };

        failure.report(0x8000_4005);

        assert_eq!(failure.error_count.load(Ordering::Relaxed), 1);
        assert_ne!(failure.rt_error_code.load(Ordering::Relaxed), 0);
        assert!(failure.restart_requested.load(Ordering::Acquire));
        assert_eq!(
            failure.low_period_failure_code.load(Ordering::Relaxed),
            0x8000_4005
        );
    }

    #[cfg(windows)]
    #[test]
    fn test_unknown_endpoint_hands_the_renderer_back_with_its_error() {
        let underruns = priming_underruns();
        let rb = crate::audio::AudioRingBuffer::new(16);
        let (_producer, consumer) = rb.split();
        let failure = OutputFailure {
            error_count: Arc::new(AtomicU64::new(0)),
            rt_error_code: Arc::new(AtomicU32::new(0)),
            restart_requested: Arc::new(AtomicBool::new(false)),
            low_period_failure_code: Arc::new(AtomicU32::new(0)),
        };

        let opened = LowPeriodStream::open(
            "no-such-endpoint".to_string(),
            48_000,
            2,
            renderer_for_test(consumer, underruns),
            failure,
        );

        match opened {
            Opened::Declined { failure_code, .. } => assert_ne!(failure_code, 0),
            Opened::Running(_) => panic!("an unknown endpoint cannot run"),
        }
    }

    #[cfg(windows)]
    fn failure_for_test() -> OutputFailure {
        OutputFailure {
            error_count: Arc::new(AtomicU64::new(0)),
            rt_error_code: Arc::new(AtomicU32::new(0)),
            restart_requested: Arc::new(AtomicBool::new(false)),
            low_period_failure_code: Arc::new(AtomicU32::new(0)),
        }
    }

    #[cfg(windows)]
    #[test]
    fn test_a_setup_that_panics_hands_the_renderer_back() {
        let rb = crate::audio::AudioRingBuffer::new(16);
        let (_producer, consumer) = rb.split();

        let opened = LowPeriodStream::open_with(
            "panicking-endpoint".to_string(),
            Box::new(|| panic!("driver fault during setup")),
            renderer_for_test(consumer, priming_underruns()),
            failure_for_test(),
        );

        match opened {
            Opened::Declined {
                mut renderer,
                failure_code,
            } => {
                assert_ne!(failure_code, 0);
                let mut data = [1.0_f32; 8];
                renderer.render(&mut data);
            }
            Opened::Running(_) => panic!("a setup that panicked cannot run"),
        }
    }

    #[cfg(windows)]
    #[test]
    fn test_a_device_with_no_shorter_period_declines_with_code_zero() {
        let rb = crate::audio::AudioRingBuffer::new(16);
        let (_producer, consumer) = rb.split();

        let opened = LowPeriodStream::open_with(
            "plain-endpoint".to_string(),
            Box::new(|| Ok(None)),
            renderer_for_test(consumer, priming_underruns()),
            failure_for_test(),
        );

        match opened {
            Opened::Declined { failure_code, .. } => assert_eq!(failure_code, 0),
            Opened::Running(_) => panic!("no client cannot run"),
        }
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
