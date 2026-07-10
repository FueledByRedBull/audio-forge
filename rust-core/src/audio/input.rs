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
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use thiserror::Error;

use super::buffer::AudioProducer;
use super::clock::now_micros;
use super::rt::{store_rt_error, RtErrorCode};

/// Target sample rate for internal processing
pub const TARGET_SAMPLE_RATE: u32 = 48000;
pub const INPUT_PHASE_WARNING_CORRELATION: f32 = -0.75;
const PHASE_SAFE_MAX_DELAY_SAMPLES: i32 = 8;
const PHASE_SAFE_MIN_CORRELATION: f32 = 0.35;
const PHASE_SAFE_MIN_IMPROVEMENT: f32 = 0.04;
const PHASE_SAFE_HISTORY_SAMPLES: usize = 16;
const PHASE_SAFE_INTERPOLATION_LATENCY: f32 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseRescueStrategy {
    None = 0,
    PolarityFlip = 1,
    FractionalDelay = 2,
    MaxRmsFallback = 3,
}

impl PhaseRescueStrategy {
    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::PolarityFlip,
            2 => Self::FractionalDelay,
            3 => Self::MaxRmsFallback,
            _ => Self::None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PolarityFlip => "polarity_flip",
            Self::FractionalDelay => "fractional_delay",
            Self::MaxRmsFallback => "max_rms_fallback",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PhaseSafeMixDiagnostics {
    strategy: PhaseRescueStrategy,
    estimated_delay_samples: f32,
    polarity_flipped: bool,
}

impl Default for PhaseSafeMixDiagnostics {
    fn default() -> Self {
        Self {
            strategy: PhaseRescueStrategy::None,
            estimated_delay_samples: 0.0,
            polarity_flipped: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PhaseAlignmentCandidate {
    strategy: PhaseRescueStrategy,
    delay_samples: f32,
    polarity: f32,
    correlation: f32,
}

/// Callback-persistent history for cubic Lagrange fractional-delay alignment.
///
/// Both channels receive the same two-sample causal base latency; only the
/// leading channel receives the additional estimated fractional delay.
#[derive(Debug, Clone)]
struct PhaseSafeMonoState {
    left_history: [f32; PHASE_SAFE_HISTORY_SAMPLES],
    right_history: [f32; PHASE_SAFE_HISTORY_SAMPLES],
    filled: usize,
    last_candidate: Option<PhaseAlignmentCandidate>,
}

impl Default for PhaseSafeMonoState {
    fn default() -> Self {
        Self {
            left_history: [0.0; PHASE_SAFE_HISTORY_SAMPLES],
            right_history: [0.0; PHASE_SAFE_HISTORY_SAMPLES],
            filled: 0,
            last_candidate: None,
        }
    }
}

impl PhaseSafeMonoState {
    #[inline]
    fn push(&mut self, left: f32, right: f32) {
        self.left_history
            .copy_within(..PHASE_SAFE_HISTORY_SAMPLES - 1, 1);
        self.right_history
            .copy_within(..PHASE_SAFE_HISTORY_SAMPLES - 1, 1);
        self.left_history[0] = left;
        self.right_history[0] = right;
        self.filled = (self.filled + 1).min(PHASE_SAFE_HISTORY_SAMPLES);
    }

    #[inline]
    /// Evaluate the four-point Lagrange/Farrow polynomial at a fractional delay.
    fn lagrange_sample(history: &[f32; PHASE_SAFE_HISTORY_SAMPLES], delay: f32) -> f32 {
        let delay = delay.clamp(2.0, (PHASE_SAFE_HISTORY_SAMPLES - 3) as f32);
        let upper_delay = delay.ceil() as usize;
        let t = upper_delay as f32 - delay;
        let x0 = history[upper_delay + 1];
        let x1 = history[upper_delay];
        let x2 = history[upper_delay - 1];
        let x3 = history[upper_delay - 2];
        let l0 = -t * (t - 1.0) * (t - 2.0) / 6.0;
        let l1 = (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0;
        let l2 = -(t + 1.0) * t * (t - 2.0) / 2.0;
        let l3 = (t + 1.0) * t * (t - 1.0) / 6.0;
        x0 * l0 + x1 * l1 + x2 * l2 + x3 * l3
    }
}

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
    pub phase_rescue_strategy: Arc<AtomicU8>,
    pub phase_estimated_delay_samples: Arc<AtomicU32>,
    pub phase_polarity_flipped: Arc<AtomicBool>,
}

impl InputStreamOptions {
    pub fn new(
        channel_mode: Arc<AtomicU8>,
        stereo_correlation: Arc<AtomicU32>,
        phase_warning_count: Arc<AtomicU64>,
        phase_rescue_strategy: Arc<AtomicU8>,
        phase_estimated_delay_samples: Arc<AtomicU32>,
        phase_polarity_flipped: Arc<AtomicBool>,
    ) -> Self {
        Self {
            channel_mode,
            stereo_correlation,
            phase_warning_count,
            phase_rescue_strategy,
            phase_estimated_delay_samples,
            phase_polarity_flipped,
        }
    }
}

impl Default for InputStreamOptions {
    fn default() -> Self {
        Self {
            channel_mode: Arc::new(AtomicU8::new(InputChannelMode::Average as u8)),
            stereo_correlation: Arc::new(AtomicU32::new(1.0_f32.to_bits())),
            phase_warning_count: Arc::new(AtomicU64::new(0)),
            phase_rescue_strategy: Arc::new(AtomicU8::new(PhaseRescueStrategy::None as u8)),
            phase_estimated_delay_samples: Arc::new(AtomicU32::new(0.0_f32.to_bits())),
            phase_polarity_flipped: Arc::new(AtomicBool::new(false)),
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

    fn delayed_correlation<T>(
        interleaved: &[T],
        frame_count: usize,
        delay: i32,
        polarity: f32,
    ) -> Option<f32>
    where
        T: Sample + Copy,
        f32: FromSample<T>,
    {
        let start = if delay < 0 { (-delay) as usize } else { 0 };
        let end = if delay > 0 {
            frame_count.saturating_sub(delay as usize)
        } else {
            frame_count
        };
        if end.saturating_sub(start) < 3 {
            return None;
        }

        let mut sum_lr = 0.0_f32;
        let mut sum_l2 = 0.0_f32;
        let mut sum_r2 = 0.0_f32;
        for left_idx in start..end {
            let right_idx = (left_idx as i32 + delay) as usize;
            let left = Self::normalize_input_sample(interleaved[left_idx * 2]);
            let right = Self::normalize_input_sample(interleaved[right_idx * 2 + 1]) * polarity;
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

    fn best_phase_alignment<T>(
        interleaved: &[T],
        frame_count: usize,
        current_correlation: f32,
    ) -> Option<PhaseAlignmentCandidate>
    where
        T: Sample + Copy,
        f32: FromSample<T>,
    {
        let mut best_delay = 0_i32;
        let mut best_polarity = 1.0_f32;
        let mut best_corr = f32::NEG_INFINITY;

        for polarity in [1.0_f32, -1.0_f32] {
            for delay in -PHASE_SAFE_MAX_DELAY_SAMPLES..=PHASE_SAFE_MAX_DELAY_SAMPLES {
                if let Some(corr) =
                    Self::delayed_correlation(interleaved, frame_count, delay, polarity)
                {
                    if corr > best_corr {
                        best_corr = corr;
                        best_delay = delay;
                        best_polarity = polarity;
                    }
                }
            }
        }

        if best_corr < PHASE_SAFE_MIN_CORRELATION
            || best_corr - current_correlation < PHASE_SAFE_MIN_IMPROVEMENT
        {
            return None;
        }

        let mut refined_delay = best_delay as f32;
        if best_delay > -PHASE_SAFE_MAX_DELAY_SAMPLES && best_delay < PHASE_SAFE_MAX_DELAY_SAMPLES {
            if let (Some(prev), Some(center), Some(next)) = (
                Self::delayed_correlation(interleaved, frame_count, best_delay - 1, best_polarity),
                Self::delayed_correlation(interleaved, frame_count, best_delay, best_polarity),
                Self::delayed_correlation(interleaved, frame_count, best_delay + 1, best_polarity),
            ) {
                let denom = prev - 2.0 * center + next;
                if denom.abs() > 1e-6 {
                    let offset = (0.5 * (prev - next) / denom).clamp(-0.5, 0.5);
                    refined_delay += offset;
                }
            }
        }

        let strategy = if best_polarity < 0.0 && refined_delay.abs() < 0.25 {
            PhaseRescueStrategy::PolarityFlip
        } else {
            PhaseRescueStrategy::FractionalDelay
        };

        Some(PhaseAlignmentCandidate {
            strategy,
            delay_samples: refined_delay,
            polarity: best_polarity,
            correlation: best_corr,
        })
    }

    fn mix_phase_safe_stereo<T>(
        interleaved: &[T],
        frame_count: usize,
        mono: &mut [f32],
        stereo_correlation: Option<f32>,
        state: &mut PhaseSafeMonoState,
    ) -> PhaseSafeMixDiagnostics
    where
        T: Sample + Copy,
        f32: FromSample<T>,
    {
        let current_correlation = stereo_correlation.unwrap_or(1.0);
        let detected_candidate =
            Self::best_phase_alignment(interleaved, frame_count, current_correlation);
        if let Some(candidate) = detected_candidate {
            state.last_candidate = Some(candidate);
        } else if current_correlation >= INPUT_PHASE_WARNING_CORRELATION {
            state.last_candidate = None;
        }
        let candidate = detected_candidate.or(state.last_candidate);
        let Some(candidate) = candidate else {
            if current_correlation < INPUT_PHASE_WARNING_CORRELATION {
                let channel = Self::strongest_channel_index(interleaved, 2, frame_count);
                for (frame_idx, chunk) in interleaved.chunks_exact(2).take(frame_count).enumerate()
                {
                    mono[frame_idx] = Self::normalize_input_sample(chunk[channel]);
                }
                return PhaseSafeMixDiagnostics {
                    strategy: PhaseRescueStrategy::MaxRmsFallback,
                    estimated_delay_samples: 0.0,
                    polarity_flipped: false,
                };
            }

            for (frame_idx, chunk) in interleaved.chunks_exact(2).take(frame_count).enumerate() {
                let left = Self::normalize_input_sample(chunk[0]);
                let right = Self::normalize_input_sample(chunk[1]);
                mono[frame_idx] = 0.5 * (left + right);
            }
            return PhaseSafeMixDiagnostics::default();
        };

        let mix_gain = (1.0 / (2.0 * (0.5 + 0.5 * candidate.correlation.max(0.0)).sqrt()))
            .clamp(0.5, std::f32::consts::FRAC_1_SQRT_2);
        for frame_idx in 0..frame_count {
            let left = Self::normalize_input_sample(interleaved[frame_idx * 2]);
            let right = Self::normalize_input_sample(interleaved[frame_idx * 2 + 1]);
            state.push(left, right);

            if candidate.strategy == PhaseRescueStrategy::PolarityFlip {
                mono[frame_idx] = (left + right * candidate.polarity) * mix_gain;
                continue;
            }

            let required_history =
                (PHASE_SAFE_INTERPOLATION_LATENCY + candidate.delay_samples.abs()).ceil() as usize
                    + 2;
            if state.filled <= required_history {
                mono[frame_idx] = if left.abs() >= right.abs() {
                    left
                } else {
                    right
                };
                continue;
            }

            let (aligned_left, aligned_right) = if candidate.delay_samples >= 0.0 {
                (
                    PhaseSafeMonoState::lagrange_sample(
                        &state.left_history,
                        PHASE_SAFE_INTERPOLATION_LATENCY + candidate.delay_samples,
                    ),
                    PhaseSafeMonoState::lagrange_sample(
                        &state.right_history,
                        PHASE_SAFE_INTERPOLATION_LATENCY,
                    ),
                )
            } else {
                (
                    PhaseSafeMonoState::lagrange_sample(
                        &state.left_history,
                        PHASE_SAFE_INTERPOLATION_LATENCY,
                    ),
                    PhaseSafeMonoState::lagrange_sample(
                        &state.right_history,
                        PHASE_SAFE_INTERPOLATION_LATENCY - candidate.delay_samples,
                    ),
                )
            };
            mono[frame_idx] = (aligned_left + aligned_right * candidate.polarity) * mix_gain;
        }

        PhaseSafeMixDiagnostics {
            strategy: candidate.strategy,
            estimated_delay_samples: candidate.delay_samples,
            polarity_flipped: candidate.polarity < 0.0,
        }
    }

    #[cfg(test)]
    fn mix_interleaved_to_mono_with_mode<T>(
        interleaved: &[T],
        num_channels: usize,
        mode: InputChannelMode,
        mono: &mut [f32],
    ) -> (usize, Option<f32>, PhaseSafeMixDiagnostics)
    where
        T: Sample + Copy,
        f32: FromSample<T>,
    {
        let mut phase_state = PhaseSafeMonoState::default();
        Self::mix_interleaved_to_mono_with_mode_and_state(
            interleaved,
            num_channels,
            mode,
            mono,
            &mut phase_state,
        )
    }

    fn mix_interleaved_to_mono_with_mode_and_state<T>(
        interleaved: &[T],
        num_channels: usize,
        mode: InputChannelMode,
        mono: &mut [f32],
        phase_state: &mut PhaseSafeMonoState,
    ) -> (usize, Option<f32>, PhaseSafeMixDiagnostics)
    where
        T: Sample + Copy,
        f32: FromSample<T>,
    {
        if num_channels == 0 || mono.is_empty() {
            return (0, None, PhaseSafeMixDiagnostics::default());
        }

        let frame_count = (interleaved.len() / num_channels).min(mono.len());
        let stereo_correlation = (num_channels == 2)
            .then(|| Self::stereo_correlation(interleaved, frame_count))
            .flatten();

        let mut diagnostics = PhaseSafeMixDiagnostics::default();
        match mode {
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
            InputChannelMode::PhaseSafeMono if num_channels == 2 => {
                diagnostics = Self::mix_phase_safe_stereo(
                    interleaved,
                    frame_count,
                    mono,
                    stereo_correlation,
                    phase_state,
                );
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

        (frame_count, stereo_correlation, diagnostics)
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
        let phase_rescue_strategy = options.phase_rescue_strategy;
        let phase_estimated_delay_samples = options.phase_estimated_delay_samples;
        let phase_polarity_flipped = options.phase_polarity_flipped;
        let last_callback_time_us = metrics.last_callback_time_us;
        let error_count = metrics.error_count;
        let rt_error_code = metrics.rt_error_code;

        const INPUT_SCRATCH_CAPACITY: usize = 8192;
        let mut mono_scratch: Vec<f32> = vec![0.0; INPUT_SCRATCH_CAPACITY];
        let mut phase_safe_state = PhaseSafeMonoState::default();

        let stream = device
            .build_input_stream(
                &stream_config,
                move |data: &[T], _: &cpal::InputCallbackInfo| {
                    // RT_REGION_START: cpal_input_callback
                    last_callback_time_us.store(now_micros(), Ordering::Relaxed);

                    if num_channels == 1 {
                        phase_rescue_strategy
                            .store(PhaseRescueStrategy::None as u8, Ordering::Relaxed);
                        phase_estimated_delay_samples.store(0.0_f32.to_bits(), Ordering::Relaxed);
                        phase_polarity_flipped.store(false, Ordering::Relaxed);
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
                            let (written_frames, correlation, phase_diagnostics) =
                                Self::mix_interleaved_to_mono_with_mode_and_state(
                                    interleaved,
                                    num_channels,
                                    mode,
                                    &mut mono_scratch,
                                    &mut phase_safe_state,
                                );
                            phase_rescue_strategy
                                .store(phase_diagnostics.strategy as u8, Ordering::Relaxed);
                            phase_estimated_delay_samples.store(
                                phase_diagnostics.estimated_delay_samples.to_bits(),
                                Ordering::Relaxed,
                            );
                            phase_polarity_flipped
                                .store(phase_diagnostics.polarity_flipped, Ordering::Relaxed);
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
    #[ignore = "requires a Windows audio endpoint"]
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

        let (written, _, _) = AudioInput::mix_interleaved_to_mono_with_mode(
            &interleaved,
            2,
            InputChannelMode::Left,
            &mut mono,
        );
        assert_eq!(written, 2);
        assert_eq!(mono, [0.75, 0.5]);

        let (written, _, _) = AudioInput::mix_interleaved_to_mono_with_mode(
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

        let (written, _, _) = AudioInput::mix_interleaved_to_mono_with_mode(
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

        let (written, correlation, _) = AudioInput::mix_interleaved_to_mono_with_mode(
            &interleaved,
            2,
            InputChannelMode::Average,
            &mut mono,
        );

        assert_eq!(written, 3);
        assert!(correlation.unwrap() < INPUT_PHASE_WARNING_CORRELATION);
    }

    #[test]
    fn test_phase_safe_mono_recovers_anti_phase_with_polarity_flip() {
        let interleaved = [0.5_f32, -0.5, 0.25, -0.25, -0.5, 0.5];
        let mut mono = [0.0_f32; 3];

        let (written, correlation, diagnostics) = AudioInput::mix_interleaved_to_mono_with_mode(
            &interleaved,
            2,
            InputChannelMode::PhaseSafeMono,
            &mut mono,
        );

        assert_eq!(written, 3);
        assert!(correlation.unwrap() < INPUT_PHASE_WARNING_CORRELATION);
        assert_eq!(diagnostics.strategy, PhaseRescueStrategy::PolarityFlip);
        assert!(diagnostics.polarity_flipped);
        assert!((diagnostics.estimated_delay_samples).abs() < 0.25);
        assert!((mono[0] - 0.5).abs() < 1e-6);
        assert!((mono[1] - 0.25).abs() < 1e-6);
        assert!((mono[2] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_phase_safe_mono_aligns_delayed_stereo() {
        let frames = 256usize;
        let delay = 3usize;
        let sr = 48_000.0_f32;
        let freq = 1_000.0_f32;
        let mut left = vec![0.0_f32; frames];
        for (idx, sample) in left.iter_mut().enumerate() {
            let t = idx as f32 / sr;
            *sample = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5;
        }

        let mut interleaved = vec![0.0_f32; frames * 2];
        for idx in 0..frames {
            interleaved[idx * 2] = left[idx];
            interleaved[idx * 2 + 1] = if idx >= delay { left[idx - delay] } else { 0.0 };
        }

        let mut mono = vec![0.0_f32; frames];
        let (written, _correlation, diagnostics) = AudioInput::mix_interleaved_to_mono_with_mode(
            &interleaved,
            2,
            InputChannelMode::PhaseSafeMono,
            &mut mono,
        );

        let mut average_error = 0.0_f32;
        let mut aligned_error = 0.0_f32;
        let output_delay = delay + PHASE_SAFE_INTERPOLATION_LATENCY as usize;
        for idx in (output_delay + PHASE_SAFE_HISTORY_SAMPLES)..written - delay {
            let reference = left[idx - output_delay];
            let average = 0.5 * (interleaved[idx * 2] + interleaved[idx * 2 + 1]);
            average_error += (average - reference).powi(2);
            aligned_error += (mono[idx] - reference).powi(2);
        }
        average_error = average_error.sqrt();
        aligned_error = aligned_error.sqrt();

        assert_eq!(diagnostics.strategy, PhaseRescueStrategy::FractionalDelay);
        assert!(
            (diagnostics.estimated_delay_samples - delay as f32).abs() < 0.6,
            "estimated delay {}",
            diagnostics.estimated_delay_samples
        );
        assert!(
            aligned_error < average_error * 0.6,
            "aligned_error={aligned_error} average_error={average_error}"
        );
    }

    #[test]
    fn test_phase_safe_mono_leaves_normal_stereo_average_unchanged() {
        let interleaved = [0.5_f32, 0.5, -0.25, -0.25, 0.1, 0.1];
        let mut mono = [0.0_f32; 3];

        let (written, correlation, diagnostics) = AudioInput::mix_interleaved_to_mono_with_mode(
            &interleaved,
            2,
            InputChannelMode::PhaseSafeMono,
            &mut mono,
        );

        assert_eq!(written, 3);
        assert_eq!(diagnostics.strategy, PhaseRescueStrategy::None);
        assert!(correlation.unwrap() > 0.99);
        assert_eq!(mono, [0.5, -0.25, 0.1]);
    }

    #[test]
    fn test_lagrange_fractional_delay_reduces_sweep_null_error_vs_linear() {
        let sample_rate = 48_000.0_f64;
        let duration_samples = 48_000usize;
        let delay = 5.35_f32;
        let mut history = [0.0_f32; PHASE_SAFE_HISTORY_SAMPLES];
        let mut lagrange_error = 0.0_f64;
        let mut linear_error = 0.0_f64;
        let mut measured = 0usize;

        let sweep_sample = |position: f64| -> f32 {
            let time = position / sample_rate;
            let sweep_rate = (18_000.0 - 500.0) / (duration_samples as f64 / sample_rate);
            let phase =
                2.0 * std::f64::consts::PI * (500.0 * time + 0.5 * sweep_rate * time * time);
            (0.5 * phase.sin()) as f32
        };

        for index in 0..duration_samples {
            history.copy_within(..PHASE_SAFE_HISTORY_SAMPLES - 1, 1);
            history[0] = sweep_sample(index as f64);
            if index < PHASE_SAFE_HISTORY_SAMPLES {
                continue;
            }

            let reference = sweep_sample(index as f64 - delay as f64);
            let lagrange = PhaseSafeMonoState::lagrange_sample(&history, delay);
            let lower = delay.floor() as usize;
            let fraction = delay - lower as f32;
            let linear = history[lower] * (1.0 - fraction) + history[lower + 1] * fraction;
            lagrange_error += (lagrange - reference).powi(2) as f64;
            linear_error += (linear - reference).powi(2) as f64;
            measured += 1;
        }

        let lagrange_rms = (lagrange_error / measured as f64).sqrt();
        let linear_rms = (linear_error / measured as f64).sqrt();
        assert!(
            lagrange_rms < linear_rms * 0.60,
            "lagrange_rms={lagrange_rms} linear_rms={linear_rms}"
        );
    }

    #[test]
    fn test_phase_safe_fractional_delay_state_survives_callback_boundaries() {
        let sample_rate = 48_000.0_f64;
        let frames = 2048usize;
        let delay = 3.4_f64;
        let source = |position: f64| -> f32 {
            let time = position / sample_rate;
            (0.22 * (2.0 * std::f64::consts::PI * 3100.0 * time).sin()
                + 0.16 * (2.0 * std::f64::consts::PI * 9100.0 * time).sin()
                + 0.09 * (2.0 * std::f64::consts::PI * 15_000.0 * time).sin()) as f32
        };
        let mut interleaved = vec![0.0_f32; frames * 2];
        for index in 0..frames {
            interleaved[index * 2] = source(index as f64);
            interleaved[index * 2 + 1] = source(index as f64 - delay);
        }

        let mut state = PhaseSafeMonoState::default();
        let mut output = vec![0.0_f32; frames];
        let callback_frames = 128usize;
        for start in (0..frames).step_by(callback_frames) {
            let end = (start + callback_frames).min(frames);
            let (_, _, diagnostics) = AudioInput::mix_interleaved_to_mono_with_mode_and_state(
                &interleaved[start * 2..end * 2],
                2,
                InputChannelMode::PhaseSafeMono,
                &mut output[start..end],
                &mut state,
            );
            assert_eq!(diagnostics.strategy, PhaseRescueStrategy::FractionalDelay);
        }

        let expected_delay = delay + PHASE_SAFE_INTERPOLATION_LATENCY as f64;
        let mut square_error = 0.0_f64;
        let mut square_reference = 0.0_f64;
        let mut max_boundary_error = 0.0_f32;
        let start = PHASE_SAFE_HISTORY_SAMPLES * 2;
        for (index, &aligned_sample) in output.iter().enumerate().take(frames).skip(start) {
            let reference = source(index as f64 - expected_delay);
            let error = aligned_sample - reference;
            square_error += (error * error) as f64;
            square_reference += (reference * reference) as f64;
            if index % callback_frames == 0 {
                max_boundary_error = max_boundary_error.max(error.abs());
            }
        }
        let relative_null = (square_error / square_reference.max(1e-12)).sqrt();
        assert!(relative_null < 0.18, "relative_null={relative_null}");
        assert!(
            max_boundary_error < 0.12,
            "max_boundary_error={max_boundary_error}"
        );
    }

    #[test]
    #[ignore = "release-mode callback cost measurement"]
    fn benchmark_phase_safe_mono_callback_cost() {
        const FRAMES: usize = 128;
        const CALLBACKS: usize = 20_000;
        let mut interleaved = [0.0_f32; FRAMES * 2];
        for index in 0..FRAMES {
            let phase =
                2.0 * std::f32::consts::PI * 3_700.0 * index as f32 / TARGET_SAMPLE_RATE as f32;
            let delayed_phase =
                phase - 2.0 * std::f32::consts::PI * 3_700.0 * 3.4 / TARGET_SAMPLE_RATE as f32;
            interleaved[index * 2] = 0.25 * phase.sin();
            interleaved[index * 2 + 1] = 0.25 * delayed_phase.sin();
        }

        let measure = |mode| {
            let mut state = PhaseSafeMonoState::default();
            let mut output = [0.0_f32; FRAMES];
            for _ in 0..100 {
                let result = AudioInput::mix_interleaved_to_mono_with_mode_and_state(
                    std::hint::black_box(&interleaved),
                    2,
                    mode,
                    std::hint::black_box(&mut output),
                    &mut state,
                );
                std::hint::black_box(result);
            }
            let started = std::time::Instant::now();
            for _ in 0..CALLBACKS {
                let result = AudioInput::mix_interleaved_to_mono_with_mode_and_state(
                    std::hint::black_box(&interleaved),
                    2,
                    mode,
                    std::hint::black_box(&mut output),
                    &mut state,
                );
                std::hint::black_box(result);
            }
            started.elapsed()
        };

        let average = measure(InputChannelMode::Average);
        let phase_safe = measure(InputChannelMode::PhaseSafeMono);
        let frames = (FRAMES * CALLBACKS) as f64;
        println!(
            "phase-safe mono average={:.2} ns/frame phase-safe={:.2} ns/frame ratio={:.2}",
            average.as_nanos() as f64 / frames,
            phase_safe.as_nanos() as f64 / frames,
            phase_safe.as_secs_f64() / average.as_secs_f64(),
        );
    }
}
