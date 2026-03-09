//! Audio input capture using cpal with automatic resampling
//!
//! Real-time audio capture from microphone or line-in.
//! Supports any device sample rate with automatic resampling to 48 kHz.
//!
//! Adapted from Spectral Workbench project.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, FromSample, Sample, SampleFormat, SizedSample, Stream, StreamConfig};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use thiserror::Error;

use super::buffer::AudioProducer;
use super::clock::now_micros;

/// Target sample rate for internal processing
pub const TARGET_SAMPLE_RATE: u32 = 48000;

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
    fn normalize_input_sample<T>(sample: T) -> f32
    where
        T: Sample,
        f32: FromSample<T>,
    {
        f32::from_sample(sample)
    }

    fn build_stream<T>(
        device: Device,
        stream_config: StreamConfig,
        producer: AudioProducer,
        last_callback_time_us: Arc<AtomicU64>,
        device_info: AudioDeviceInfo,
    ) -> Result<Self, AudioError>
    where
        T: SizedSample,
        f32: FromSample<T>,
    {
        let mut producer = producer;
        let num_channels = device_info.channels as usize;

        const INPUT_SCRATCH_CAPACITY: usize = 8192;
        let mut mono_scratch: Vec<f32> = vec![0.0; INPUT_SCRATCH_CAPACITY];

        let stream = device
            .build_input_stream(
                &stream_config,
                move |data: &[T], _: &cpal::InputCallbackInfo| {
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

                            let mut written_frames = 0usize;
                            for chunk in interleaved.chunks_exact(num_channels) {
                                let sum: f32 = chunk
                                    .iter()
                                    .copied()
                                    .map(Self::normalize_input_sample)
                                    .sum();
                                mono_scratch[written_frames] = sum / num_channels as f32;
                                written_frames += 1;
                            }

                            producer.write(&mono_scratch[..written_frames]);
                            frame_idx += chunk_frames;
                        }
                    }
                },
                move |err| {
                    eprintln!("Audio input error: {}", err);
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
    ) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host.default_input_device().ok_or(AudioError::NoDevice)?;

        Self::from_device(device, producer, last_callback_time_us)
    }

    /// Create audio input from device by name
    pub fn from_device_name(
        name: &str,
        producer: AudioProducer,
        last_callback_time_us: Arc<AtomicU64>,
    ) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host
            .input_devices()
            .map_err(|e| AudioError::DeviceName(e.to_string()))?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
            .ok_or_else(|| AudioError::DeviceNotFound(name.to_string()))?;

        Self::from_device(device, producer, last_callback_time_us)
    }

    /// Create audio input from specific device
    pub fn from_device(
        device: Device,
        producer: AudioProducer,
        last_callback_time_us: Arc<AtomicU64>,
    ) -> Result<Self, AudioError> {
        let name = device
            .name()
            .map_err(|e| AudioError::DeviceName(e.to_string()))?;

        let supported_config = device
            .default_input_config()
            .map_err(|e| AudioError::DefaultConfig(e.to_string()))?;

        let device_sample_rate = supported_config.sample_rate().0;
        let channels = supported_config.channels();
        let sample_format = supported_config.sample_format();

        // Report actual device sample rate; conversion (if needed) happens in DSP thread.
        let device_info = AudioDeviceInfo {
            name: name.clone(),
            sample_rate: device_sample_rate,
            channels,
        };

        let stream_config = supported_config.config();

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
                last_callback_time_us,
                device_info,
            ),
            SampleFormat::F32 => Self::build_stream::<f32>(
                device,
                stream_config,
                producer,
                last_callback_time_us,
                device_info,
            ),
            SampleFormat::F64 => Self::build_stream::<f64>(
                device,
                stream_config,
                producer,
                last_callback_time_us,
                device_info,
            ),
            SampleFormat::I16 => Self::build_stream::<i16>(
                device,
                stream_config,
                producer,
                last_callback_time_us,
                device_info,
            ),
            SampleFormat::I32 => Self::build_stream::<i32>(
                device,
                stream_config,
                producer,
                last_callback_time_us,
                device_info,
            ),
            SampleFormat::I64 => Self::build_stream::<i64>(
                device,
                stream_config,
                producer,
                last_callback_time_us,
                device_info,
            ),
            SampleFormat::U8 => Self::build_stream::<u8>(
                device,
                stream_config,
                producer,
                last_callback_time_us,
                device_info,
            ),
            SampleFormat::U16 => Self::build_stream::<u16>(
                device,
                stream_config,
                producer,
                last_callback_time_us,
                device_info,
            ),
            SampleFormat::U32 => Self::build_stream::<u32>(
                device,
                stream_config,
                producer,
                last_callback_time_us,
                device_info,
            ),
            SampleFormat::U64 => Self::build_stream::<u64>(
                device,
                stream_config,
                producer,
                last_callback_time_us,
                device_info,
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
}
