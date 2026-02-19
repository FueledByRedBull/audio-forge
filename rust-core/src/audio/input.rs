//! Audio input capture using cpal with automatic resampling
//!
//! Real-time audio capture from microphone or line-in.
//! Supports any device sample rate with automatic resampling to 48 kHz.
//!
//! Adapted from Spectral Workbench project.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};
use rubato::{FftFixedIn, Resampler};
use std::sync::{Arc, Mutex};
use thiserror::Error;

use super::buffer::AudioProducer;

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

    #[error("Failed to create resampler: {0}")]
    ResamplerError(String),

    #[error("Device not found: {0}")]
    DeviceNotFound(String),
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
    /// Create audio input from default device
    pub fn from_default_device(producer: AudioProducer) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host.default_input_device().ok_or(AudioError::NoDevice)?;

        Self::from_device(device, producer)
    }

    /// Create audio input from device by name
    pub fn from_device_name(name: &str, producer: AudioProducer) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host
            .input_devices()
            .map_err(|e| AudioError::DeviceName(e.to_string()))?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
            .ok_or_else(|| AudioError::DeviceNotFound(name.to_string()))?;

        Self::from_device(device, producer)
    }

    /// Create audio input from specific device
    pub fn from_device(device: Device, producer: AudioProducer) -> Result<Self, AudioError> {
        let name = device
            .name()
            .map_err(|e| AudioError::DeviceName(e.to_string()))?;

        let config = device
            .default_input_config()
            .map_err(|e| AudioError::DefaultConfig(e.to_string()))?;

        let device_sample_rate = config.sample_rate().0;
        let channels = config.channels();

        // Report target sample rate (we resample to 48kHz internally)
        let device_info = AudioDeviceInfo {
            name: name.clone(),
            sample_rate: TARGET_SAMPLE_RATE,
            channels,
        };

        let stream_config: StreamConfig = config.into();

        // Wrap producer in Arc<Mutex> for thread-safe access
        let producer = Arc::new(Mutex::new(producer));

        // Create resampler if device rate differs from target
        let needs_resampling = device_sample_rate != TARGET_SAMPLE_RATE;

        if needs_resampling {
            println!(
                "Device sample rate {} Hz differs from target {} Hz - enabling resampling",
                device_sample_rate, TARGET_SAMPLE_RATE
            );

            // Create FFT-based resampler (high quality, handles any ratio)
            let chunk_size = 1024;
            let resampler = FftFixedIn::<f64>::new(
                device_sample_rate as usize,
                TARGET_SAMPLE_RATE as usize,
                chunk_size,
                2, // sub_chunks for interpolation quality
                1, // mono
            )
            .map_err(|e| AudioError::ResamplerError(e.to_string()))?;

            let resampler = Arc::new(Mutex::new(resampler));
            let input_buffer: Arc<Mutex<Vec<f64>>> =
                Arc::new(Mutex::new(Vec::with_capacity(chunk_size * 2)));

            let producer_clone = Arc::clone(&producer);
            let resampler_clone = Arc::clone(&resampler);
            let input_buffer_clone = Arc::clone(&input_buffer);
            let num_channels = channels as usize;

            let stream = device
                .build_input_stream(
                    &stream_config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        // Convert to mono if stereo, then accumulate
                        if let Ok(mut buffer) = input_buffer_clone.try_lock() {
                            if num_channels == 1 {
                                buffer.extend(data.iter().map(|&s| s as f64));
                            } else {
                                // Convert stereo to mono by averaging
                                for chunk in data.chunks(num_channels) {
                                    let mono: f64 = chunk.iter().map(|&s| s as f64).sum::<f64>()
                                        / num_channels as f64;
                                    buffer.push(mono);
                                }
                            }

                            // Process when we have enough samples
                            if let Ok(mut resampler) = resampler_clone.try_lock() {
                                let input_frames_needed = resampler.input_frames_next();

                                while buffer.len() >= input_frames_needed {
                                    let input_chunk: Vec<f64> =
                                        buffer.drain(..input_frames_needed).collect();

                                    if let Ok(output) = resampler.process(&[input_chunk], None) {
                                        if !output.is_empty() && !output[0].is_empty() {
                                            if let Ok(mut prod) = producer_clone.try_lock() {
                                                // Convert f64 back to f32
                                                let samples: Vec<f32> =
                                                    output[0].iter().map(|&s| s as f32).collect();
                                                prod.write(&samples);
                                            }
                                        }
                                    }
                                }
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
        } else {
            // No resampling needed - direct passthrough
            let producer_clone = Arc::clone(&producer);
            let num_channels = channels as usize;
            let mut mono_scratch: Vec<f32> = Vec::new();

            let stream = device
                .build_input_stream(
                    &stream_config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        if let Ok(mut prod) = producer_clone.try_lock() {
                            if num_channels == 1 {
                                prod.write(data);
                            } else {
                                // Convert interleaved multi-channel input to mono without
                                // allocating in the callback.
                                let frames = data.len() / num_channels;
                                if mono_scratch.len() < frames {
                                    mono_scratch.resize(frames, 0.0);
                                }

                                let mut written_frames = 0usize;
                                for chunk in data.chunks_exact(num_channels) {
                                    let sum: f32 = chunk.iter().copied().sum();
                                    mono_scratch[written_frames] = sum / num_channels as f32;
                                    written_frames += 1;
                                }

                                prod.write(&mono_scratch[..written_frames]);
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
}
