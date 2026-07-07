//! Audio input capture using cpal with automatic resampling
//!
//! Real-time audio capture from microphone or line-in.
//! Supports any device sample rate with automatic resampling to 48 kHz.
//!
//! Adapted from Spectral Workbench project.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{
    Device, FromSample, Sample, SampleFormat, SizedSample, Stream, StreamConfig,
    SupportedStreamConfig, SupportedStreamConfigRange,
};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use thiserror::Error;

use super::buffer::AudioProducer;
use super::clock::now_micros;
use super::rt::{store_rt_error, RtErrorCode};

/// Target sample rate for internal processing
pub const TARGET_SAMPLE_RATE: u32 = 48000;
pub const INPUT_PHASE_WARNING_CORRELATION: f32 = -0.75;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputChannelMode {
    Average = 0,
    Left = 1,
    Right = 2,
    MaxRms = 3,
    PhaseSafeMono = 4,
}

impl InputChannelMode {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Average),
            1 => Some(Self::Left),
            2 => Some(Self::Right),
            3 => Some(Self::MaxRms),
            4 => Some(Self::PhaseSafeMono),
            _ => None,
        }
    }

    pub fn from_id(value: &str) -> Option<Self> {
        match value {
            "average" => Some(Self::Average),
            "left" => Some(Self::Left),
            "right" => Some(Self::Right),
            "max_rms" => Some(Self::MaxRms),
            "phase_safe_mono" => Some(Self::PhaseSafeMono),
            _ => None,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::Average => "average",
            Self::Left => "left",
            Self::Right => "right",
            Self::MaxRms => "max_rms",
            Self::PhaseSafeMono => "phase_safe_mono",
        }
    }
}

pub struct InputStreamOptions {
    pub channel_mode: Arc<AtomicU8>,
    pub stereo_correlation: Arc<AtomicU32>,
    pub phase_warning_count: Arc<AtomicU64>,
}

impl InputStreamOptions {
    pub fn new(
        channel_mode: Arc<AtomicU8>,
        stereo_correlation: Arc<AtomicU32>,
        phase_warning_count: Arc<AtomicU64>,
    ) -> Self {
        Self {
            channel_mode,
            stereo_correlation,
            phase_warning_count,
        }
    }
}

impl Default for InputStreamOptions {
    fn default() -> Self {
        Self {
            channel_mode: Arc::new(AtomicU8::new(InputChannelMode::Average as u8)),
            stereo_correlation: Arc::new(AtomicU32::new(1.0_f32.to_bits())),
            phase_warning_count: Arc::new(AtomicU64::new(0)),
        }
    }
}

struct InputCallbackMetrics {
    last_callback_time_us: Arc<AtomicU64>,
    error_count: Arc<AtomicU64>,
    rt_error_code: Arc<AtomicU32>,
}

impl InputCallbackMetrics {
    fn new(
        last_callback_time_us: Arc<AtomicU64>,
        error_count: Arc<AtomicU64>,
        rt_error_code: Arc<AtomicU32>,
    ) -> Self {
        Self {
            last_callback_time_us,
            error_count,
            rt_error_code,
        }
    }
}

#[derive(Error, Debug)]
pub enum AudioError {
    #[error("No audio device found")]
    NoDevice,

    #[error("Failed to get device name: {0}")]
    DeviceName(String),

    #[error("Failed to get default config: {0}")]
    DefaultConfig(String),

    #[error("Failed to build stream: {0}")]
    BuildStream(String),

    #[error("Failed to play stream: {0}")]
    PlayStream(String),

    #[error("Device not found: {0}")]
    DeviceNotFound(String),

    #[error("Unsupported audio sample format: {0}")]
    UnsupportedSampleFormat(String),
}

/// Audio device information
#[derive(Debug, Clone)]
pub struct AudioDeviceInfo {
    pub name: String,
    pub sample_rate: u32,
    pub channels: u16,
}

/// Audio input stream with optional resampling
pub struct AudioInput {
    stream: Stream,
    device_info: AudioDeviceInfo,
}

impl AudioInput {
    fn select_device(
        device: Device,
    ) -> Result<(Device, SupportedStreamConfig, AudioDeviceInfo), AudioError> {
        let name = device
            .name()
            .map_err(|e| AudioError::DeviceName(e.to_string()))?;

        let supported_configs: Vec<_> = device
            .supported_input_configs()
            .map_err(|e| AudioError::DefaultConfig(e.to_string()))?
            .collect();
        let default_config = device.default_input_config().ok();

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
                AudioError::DefaultConfig("No suitable input config found".to_string())
            })?;

        let device_info = AudioDeviceInfo {
            name,
            sample_rate: supported_config.sample_rate().0,
            channels: supported_config.channels(),
        };

        Ok((device, supported_config, device_info))
    }

    fn normalize_input_sample<T>(sample: T) -> f32
    where
        T: Sample,
        f32: FromSample<T>,
    {
        f32::from_sample(sample)
    }

    fn strongest_channel_index<T>(
        interleaved: &[T],
        num_channels: usize,
        frame_count: usize,
    ) -> usize
    where
        T: Sample + Copy,
        f32: FromSample<T>,
    {
        let mut best_channel = 0usize;
        let mut best_energy = f32::NEG_INFINITY;
        for channel in 0..num_channels {
            let mut energy = 0.0_f32;
            for frame_idx in 0..frame_count {
                let sample =
                    Self::normalize_input_sample(interleaved[frame_idx * num_channels + channel]);
                energy += sample * sample;
            }
            if energy > best_energy {
                best_energy = energy;
                best_channel = channel;
            }
        }
        best_channel
    }

    fn stereo_correlation<T>(interleaved: &[T], frame_count: usize) -> Option<f32>
    where
        T: Sample + Copy,
        f32: FromSample<T>,
    {
        if frame_count == 0 {
            return None;
        }

        let mut sum_lr = 0.0_f32;
        let mut sum_l2 = 0.0_f32;
        let mut sum_r2 = 0.0_f32;
        for frame in interleaved.chunks_exact(2).take(frame_count) {
            let left = Self::normalize_input_sample(frame[0]);
            let right = Self::normalize_input_sample(frame[1]);
            sum_lr += left * right;
            sum_l2 += left * left;
            sum_r2 += right * right;
        }

        let denom = (sum_l2 * sum_r2).sqrt();
        if denom <= f32::EPSILON {
            None
        } else {
            Some((sum_lr / denom).clamp(-1.0, 1.0))
        }
    }

    fn mix_interleaved_to_mono_with_mode<T>(
        interleaved: &[T],
        num_channels: usize,
        mode: InputChannelMode,
        mono: &mut [f32],
    ) -> (usize, Option<f32>)
    where
        T: Sample + Copy,
        f32: FromSample<T>,
    {
        if num_channels == 0 || mono.is_empty() {
            return (0, None);
        }

        let frame_count = (interleaved.len() / num_channels).min(mono.len());
        let stereo_correlation = (num_channels == 2)
            .then(|| Self::stereo_correlation(interleaved, frame_count))
            .flatten();

        let selected_mode = if mode == InputChannelMode::PhaseSafeMono
            && num_channels == 2
            && stereo_correlation.unwrap_or(1.0) < INPUT_PHASE_WARNING_CORRELATION
        {
            InputChannelMode::MaxRms
        } else {
            mode
        };

        match selected_mode {
            InputChannelMode::Left => {
                for (frame_idx, chunk) in interleaved
                    .chunks_exact(num_channels)
                    .take(frame_count)
                    .enumerate()
                {
                    mono[frame_idx] = Self::normalize_input_sample(chunk[0]);
                }
            }
            InputChannelMode::Right => {
                let channel = if num_channels > 1 { 1 } else { 0 };
                for (frame_idx, chunk) in interleaved
                    .chunks_exact(num_channels)
                    .take(frame_count)
                    .enumerate()
                {
                    mono[frame_idx] = Self::normalize_input_sample(chunk[channel]);
                }
            }
            InputChannelMode::MaxRms => {
                let channel = Self::strongest_channel_index(interleaved, num_channels, frame_count);
                for (frame_idx, chunk) in interleaved
                    .chunks_exact(num_channels)
                    .take(frame_count)
                    .enumerate()
                {
                    mono[frame_idx] = Self::normalize_input_sample(chunk[channel]);
                }
            }
            InputChannelMode::Average | InputChannelMode::PhaseSafeMono => {
                let inv_channel_count = 1.0 / num_channels as f32;
                for (frame_idx, chunk) in interleaved
                    .chunks_exact(num_channels)
                    .take(frame_count)
                    .enumerate()
                {
                    let mut sum = 0.0_f32;
                    for sample in chunk.iter().copied() {
                        sum += Self::normalize_input_sample(sample);
                    }
                    mono[frame_idx] = sum * inv_channel_count;
                }
            }
        }

        (frame_count, stereo_correlation)
    }

    #[cfg(test)]
    fn mix_interleaved_to_mono<T>(interleaved: &[T], num_channels: usize, mono: &mut [f32]) -> usize
    where
        T: Sample + Copy,
        f32: FromSample<T>,
    {
        Self::mix_interleaved_to_mono_with_mode(
            interleaved,
            num_channels,
            InputChannelMode::Average,
            mono,
        )
        .0
    }

    fn build_stream<T>(
        device: Device,
        stream_config: StreamConfig,
        producer: AudioProducer,
        metrics: InputCallbackMetrics,
        device_info: AudioDeviceInfo,
        options: InputStreamOptions,
    ) -> Result<Self, AudioError>
    where
        T: SizedSample,
        f32: FromSample<T>,
    {
        let mut producer = producer;
        let num_channels = device_info.channels as usize;
        let channel_mode = options.channel_mode;
        let stereo_correlation = options.stereo_correlation;
        let phase_warning_count = options.phase_warning_count;
        let last_callback_time_us = metrics.last_callback_time_us;
        let error_count = metrics.error_count;
        let rt_error_code = metrics.rt_error_code;

        const INPUT_SCRATCH_CAPACITY: usize = 8192;
        let mut mono_scratch: Vec<f32> = vec![0.0; INPUT_SCRATCH_CAPACITY];

        let stream = device
            .build_input_stream(
                &stream_config,
                move |data: &[T], _: &cpal::InputCallbackInfo| {
                    // RT_REGION_START: cpal_input_callback
                    last_callback_time_us.store(now_micros(), Ordering::Relaxed);

                    if num_channels == 1 {
                        let mut sample_idx = 0usize;
                        while sample_idx < data.len() {
                            let chunk_len = (data.len() - sample_idx).min(INPUT_SCRATCH_CAPACITY);
                            for (dst, src) in mono_scratch[..chunk_len]
                                .iter_mut()
                                .zip(data[sample_idx..sample_idx + chunk_len].iter().copied())
                            {
                                *dst = Self::normalize_input_sample(src);
                            }
                            producer.write(&mono_scratch[..chunk_len]);
                            sample_idx += chunk_len;
                        }
                    } else {
                        let frames = data.len() / num_channels;
                        let mut frame_idx = 0usize;
                        while frame_idx < frames {
                            let chunk_frames = (frames - frame_idx).min(INPUT_SCRATCH_CAPACITY);
                            let start = frame_idx * num_channels;
                            let end = start + chunk_frames * num_channels;
                            let interleaved = &data[start..end];
                            let mode =
                                InputChannelMode::from_u8(channel_mode.load(Ordering::Relaxed))
                                    .unwrap_or(InputChannelMode::Average);
                            let (written_frames, correlation) =
                                Self::mix_interleaved_to_mono_with_mode(
                                    interleaved,
                                    num_channels,
                                    mode,
                                    &mut mono_scratch,
                                );
                            if let Some(correlation) = correlation {
                                stereo_correlation.store(correlation.to_bits(), Ordering::Relaxed);
                                if correlation < INPUT_PHASE_WARNING_CORRELATION {
                                    phase_warning_count.fetch_add(1, Ordering::Relaxed);
                                }
                            }

                            producer.write(&mono_scratch[..written_frames]);
                            frame_idx += chunk_frames;
                        }
                    }
                    // RT_REGION_END: cpal_input_callback
                },
                move |err| {
                    let _ = err;
                    error_count.fetch_add(1, Ordering::Relaxed);
                    store_rt_error(rt_error_code.as_ref(), RtErrorCode::InputStreamError);
                },
                None,
            )
            .map_err(|e| AudioError::BuildStream(e.to_string()))?;

        Ok(Self {
            stream,
            device_info,
        })
    }

    /// Create audio input from default device
    pub fn from_default_device(
        producer: AudioProducer,
        last_callback_time_us: Arc<AtomicU64>,
        error_count: Arc<AtomicU64>,
        rt_error_code: Arc<AtomicU32>,
    ) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host.default_input_device().ok_or(AudioError::NoDevice)?;

        Self::from_device(
            device,
            producer,
            last_callback_time_us,
            error_count,
            rt_error_code,
        )
    }

    pub fn from_default_device_with_options(
        producer: AudioProducer,
        last_callback_time_us: Arc<AtomicU64>,
        error_count: Arc<AtomicU64>,
        rt_error_code: Arc<AtomicU32>,
        options: InputStreamOptions,
    ) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host.default_input_device().ok_or(AudioError::NoDevice)?;

        Self::from_device_with_options(
            device,
            producer,
            last_callback_time_us,
            error_count,
            rt_error_code,
            options,
        )
    }

    /// Create audio input from device by name
    pub fn from_device_name(
        name: &str,
        producer: AudioProducer,
        last_callback_time_us: Arc<AtomicU64>,
        error_count: Arc<AtomicU64>,
        rt_error_code: Arc<AtomicU32>,
    ) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host
            .input_devices()
            .map_err(|e| AudioError::DeviceName(e.to_string()))?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
            .ok_or_else(|| AudioError::DeviceNotFound(name.to_string()))?;

        Self::from_device(
            device,
            producer,
            last_callback_time_us,
            error_count,
            rt_error_code,
        )
    }

    pub fn from_device_name_with_options(
        name: &str,
        producer: AudioProducer,
        last_callback_time_us: Arc<AtomicU64>,
        error_count: Arc<AtomicU64>,
        rt_error_code: Arc<AtomicU32>,
        options: InputStreamOptions,
    ) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host
            .input_devices()
            .map_err(|e| AudioError::DeviceName(e.to_string()))?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
            .ok_or_else(|| AudioError::DeviceNotFound(name.to_string()))?;

        Self::from_device_with_options(
            device,
            producer,
            last_callback_time_us,
            error_count,
            rt_error_code,
            options,
        )
    }

    /// Create audio input from specific device
    pub fn from_device(
        device: Device,
        producer: AudioProducer,
        last_callback_time_us: Arc<AtomicU64>,
        error_count: Arc<AtomicU64>,
        rt_error_code: Arc<AtomicU32>,
    ) -> Result<Self, AudioError> {
        Self::from_device_with_options(
            device,
            producer,
            last_callback_time_us,
            error_count,
            rt_error_code,
            InputStreamOptions::default(),
        )
    }

    pub fn from_device_with_options(
        device: Device,
        producer: AudioProducer,
        last_callback_time_us: Arc<AtomicU64>,
        error_count: Arc<AtomicU64>,
        rt_error_code: Arc<AtomicU32>,
        options: InputStreamOptions,
    ) -> Result<Self, AudioError> {
        let (device, supported_config, device_info) = Self::select_device(device)?;
        let device_sample_rate = supported_config.sample_rate().0;
        let sample_format = supported_config.sample_format();
        let stream_config = supported_config.config();
        let callback_metrics =
            InputCallbackMetrics::new(last_callback_time_us, error_count, rt_error_code);

        if device_sample_rate != TARGET_SAMPLE_RATE {
            println!(
                "Device sample rate {} Hz differs from target {} Hz - resampling in DSP thread",
                device_sample_rate, TARGET_SAMPLE_RATE
            );
        }

        match sample_format {
            SampleFormat::I8 => Self::build_stream::<i8>(
                device,
                stream_config,
                producer,
                callback_metrics,
                device_info,
                options,
            ),
            SampleFormat::F32 => Self::build_stream::<f32>(
                device,
                stream_config,
                producer,
                callback_metrics,
                device_info,
                options,
            ),
            SampleFormat::F64 => Self::build_stream::<f64>(
                device,
                stream_config,
                producer,
                callback_metrics,
                device_info,
                options,
            ),
            SampleFormat::I16 => Self::build_stream::<i16>(
                device,
                stream_config,
                producer,
                callback_metrics,
                device_info,
                options,
            ),
            SampleFormat::I32 => Self::build_stream::<i32>(
                device,
                stream_config,
                producer,
                callback_metrics,
                device_info,
                options,
            ),
            SampleFormat::I64 => Self::build_stream::<i64>(
                device,
                stream_config,
                producer,
                callback_metrics,
                device_info,
                options,
            ),
            SampleFormat::U8 => Self::build_stream::<u8>(
                device,
                stream_config,
                producer,
                callback_metrics,
                device_info,
                options,
            ),
            SampleFormat::U16 => Self::build_stream::<u16>(
                device,
                stream_config,
                producer,
                callback_metrics,
                device_info,
                options,
            ),
            SampleFormat::U32 => Self::build_stream::<u32>(
                device,
                stream_config,
                producer,
                callback_metrics,
                device_info,
                options,
            ),
            SampleFormat::U64 => Self::build_stream::<u64>(
                device,
                stream_config,
                producer,
                callback_metrics,
                device_info,
                options,
            ),
            other => Err(AudioError::UnsupportedSampleFormat(other.to_string())),
        }
    }

    /// Start capturing audio
    pub fn start(&self) -> Result<(), AudioError> {
        self.stream
            .play()
            .map_err(|e| AudioError::PlayStream(e.to_string()))
    }

    /// Pause audio capture
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

/// List available audio input devices
pub fn list_input_devices() -> Result<Vec<AudioDeviceInfo>, AudioError> {
    let host = cpal::default_host();
    let mut devices = Vec::new();

    let device_iter = host
        .input_devices()
        .map_err(|e| AudioError::DeviceName(e.to_string()))?;

    for device in device_iter {
        if let Ok(name) = device.name() {
            if let Ok(config) = device.default_input_config() {
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
) -> Option<SupportedStreamConfig> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_devices() {
        let _ = list_input_devices();
    }

    #[test]
    fn test_normalize_i16_input_sample() {
        let normalized = AudioInput::normalize_input_sample(i16::MAX);
        assert!(normalized > 0.99);
    }

    #[test]
    fn test_normalize_u16_input_sample() {
        let normalized = AudioInput::normalize_input_sample(0_u16);
        assert!(normalized <= -0.99);
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

    #[test]
    fn test_mix_interleaved_stereo_input_to_mono_average() {
        let interleaved = [1.0_f32, 1.0, 0.25, 0.75];
        let mut mono = [0.0_f32; 4];

        let written = AudioInput::mix_interleaved_to_mono(&interleaved, 2, &mut mono);

        assert_eq!(written, 2);
        assert!((mono[0] - 1.0).abs() < 1e-6);
        assert!((mono[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_mix_interleaved_stereo_input_preserves_phase_cancellation() {
        let interleaved = [1.0_f32, -1.0, 0.5, -0.5];
        let mut mono = [1.0_f32; 4];

        let written = AudioInput::mix_interleaved_to_mono(&interleaved, 2, &mut mono);

        assert_eq!(written, 2);
        assert!(mono[..written].iter().all(|sample| sample.abs() < 1e-6));
    }

    #[test]
    fn test_mix_interleaved_multichannel_input_does_not_switch_with_alternating_loudness() {
        let interleaved = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let mut mono = [0.0_f32; 4];

        let written = AudioInput::mix_interleaved_to_mono(&interleaved, 2, &mut mono);

        assert_eq!(written, 4);
        assert!(mono[..written]
            .iter()
            .all(|sample| (*sample - 0.5).abs() < 1e-6));
    }

    #[test]
    fn test_mix_interleaved_multichannel_input_is_stable_and_bounded() {
        let interleaved = [1.0_f32, 0.5, -0.5, -1.0, -0.5, 0.5];
        let mut mono = [0.0_f32; 4];

        let written = AudioInput::mix_interleaved_to_mono(&interleaved, 3, &mut mono);

        assert_eq!(written, 2);
        assert!((mono[0] - (1.0 / 3.0)).abs() < 1e-6);
        assert!((mono[1] + (1.0 / 3.0)).abs() < 1e-6);
        assert!(mono[..written].iter().all(|sample| sample.abs() <= 1.0));
    }

    #[test]
    fn test_mix_interleaved_left_and_right_modes_select_channels() {
        let interleaved = [0.75_f32, -0.25, 0.5, -0.5];
        let mut mono = [0.0_f32; 2];

        let (written, _) = AudioInput::mix_interleaved_to_mono_with_mode(
            &interleaved,
            2,
            InputChannelMode::Left,
            &mut mono,
        );
        assert_eq!(written, 2);
        assert_eq!(mono, [0.75, 0.5]);

        let (written, _) = AudioInput::mix_interleaved_to_mono_with_mode(
            &interleaved,
            2,
            InputChannelMode::Right,
            &mut mono,
        );
        assert_eq!(written, 2);
        assert_eq!(mono, [-0.25, -0.5]);
    }

    #[test]
    fn test_mix_interleaved_max_rms_selects_one_channel_for_whole_block() {
        let interleaved = [0.1_f32, 0.9, 0.8, 0.2, 0.1, -0.9, 0.8, -0.2];
        let mut mono = [0.0_f32; 4];

        let (written, _) = AudioInput::mix_interleaved_to_mono_with_mode(
            &interleaved,
            2,
            InputChannelMode::MaxRms,
            &mut mono,
        );

        assert_eq!(written, 4);
        assert_eq!(mono, [0.9, 0.2, -0.9, -0.2]);
    }

    #[test]
    fn test_mix_interleaved_reports_negative_stereo_correlation() {
        let interleaved = [1.0_f32, -1.0, 0.5, -0.5, -0.25, 0.25];
        let mut mono = [0.0_f32; 3];

        let (written, correlation) = AudioInput::mix_interleaved_to_mono_with_mode(
            &interleaved,
            2,
            InputChannelMode::Average,
            &mut mono,
        );

        assert_eq!(written, 3);
        assert!(correlation.unwrap() < INPUT_PHASE_WARNING_CORRELATION);
    }

    #[test]
    fn test_phase_safe_mono_uses_stronger_channel_when_stereo_is_anti_phase() {
        let interleaved = [0.5_f32, -1.0, 0.25, -0.5, -0.5, 1.0];
        let mut mono = [0.0_f32; 3];

        let (written, correlation) = AudioInput::mix_interleaved_to_mono_with_mode(
            &interleaved,
            2,
            InputChannelMode::PhaseSafeMono,
            &mut mono,
        );

        assert_eq!(written, 3);
        assert!(correlation.unwrap() < INPUT_PHASE_WARNING_CORRELATION);
        assert_eq!(mono, [-1.0, -0.5, 1.0]);
    }
}
