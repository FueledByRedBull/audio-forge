//! Audio output playback using cpal
//!
//! Real-time audio playback to VB Audio Cable or other output devices.
//! Requests 48kHz sample rate from device.
//!
//! Adapted from Spectral Workbench project.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig, SupportedStreamConfigRange};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::buffer::AudioConsumer;
use super::input::{AudioDeviceInfo, AudioError, TARGET_SAMPLE_RATE};

/// Audio output stream
pub struct AudioOutput {
    stream: Stream,
    device_info: AudioDeviceInfo,
}

impl AudioOutput {
    /// Create audio output from default device
    pub fn from_default_device(
        consumer: AudioConsumer,
        recording_active: Arc<AtomicBool>,
    ) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host.default_output_device().ok_or(AudioError::NoDevice)?;

        Self::from_device(device, consumer, recording_active)
    }

    /// Create audio output from device by name
    pub fn from_device_name(
        name: &str,
        consumer: AudioConsumer,
        recording_active: Arc<AtomicBool>,
    ) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host
            .output_devices()
            .map_err(|e| AudioError::DeviceName(e.to_string()))?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
            .ok_or_else(|| AudioError::DeviceNotFound(name.to_string()))?;

        Self::from_device(device, consumer, recording_active)
    }

    /// Create audio output from specific device
    pub fn from_device(
        device: Device,
        consumer: AudioConsumer,
        recording_active: Arc<AtomicBool>,
    ) -> Result<Self, AudioError> {
        let name = device
            .name()
            .map_err(|e| AudioError::DeviceName(e.to_string()))?;

        // Try to find a config that supports 48kHz
        let supported_configs = device
            .supported_output_configs()
            .map_err(|e| AudioError::DefaultConfig(e.to_string()))?;

        let config = find_48khz_config(supported_configs)
            .or_else(|| {
                // Fall back to default config if 48kHz not explicitly supported
                device.default_output_config().ok()
            })
            .ok_or_else(|| {
                AudioError::DefaultConfig("No suitable output config found".to_string())
            })?;

        let sample_rate = config.sample_rate().0;
        let channels = config.channels();

        let device_info = AudioDeviceInfo {
            name: name.clone(),
            sample_rate,
            channels,
        };

        let stream_config: StreamConfig = config.into();

        // Wrap consumer in Arc<Mutex> for thread-safe access
        let consumer = Arc::new(Mutex::new(consumer));
        let num_channels = channels as usize;
        let mut stereo_scratch: Vec<f32> = Vec::new();
        let mut last_output_sample: f32 = 0.0;

        // Build audio output stream
        let consumer_clone = Arc::clone(&consumer);
        let recording_active_clone = Arc::clone(&recording_active);

        let stream = device
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    // Check if recording is active - if so, output silence to prevent
                    // user from hearing themselves while recording
                    if recording_active_clone.load(Ordering::Relaxed) {
                        for sample in data.iter_mut() {
                            *sample = 0.0;
                        }
                        return;
                    }

                    match consumer_clone.try_lock() {
                        Ok(mut cons) => {
                            let available = cons.len();

                            if num_channels == 1 {
                                // Mono output - read whatever is available, then smooth-fill remainder.
                                let to_read = available.min(data.len());
                                let read = if to_read > 0 {
                                    cons.read(&mut data[..to_read])
                                } else {
                                    0
                                };

                                if read < data.len() {
                                    let last = if read > 0 {
                                        data[read - 1]
                                    } else {
                                        cons.last_sample()
                                    };
                                    let remain = data.len() - read;
                                    for (i, sample) in data[read..].iter_mut().enumerate() {
                                        // Linear fade from last sample to 0.0 over missing region.
                                        let t = (i + 1) as f32 / remain as f32;
                                        *sample = last * (1.0 - t);
                                    }
                                }

                                if let Some(last) = data.last().copied() {
                                    last_output_sample = last;
                                }
                            } else {
                                // Stereo output - duplicate mono to both channels
                                let mono_samples = data.len() / num_channels;

                                if stereo_scratch.len() < mono_samples {
                                    stereo_scratch.resize(mono_samples, 0.0);
                                }

                                let to_read = available.min(mono_samples);
                                let count = if to_read > 0 {
                                    cons.read(&mut stereo_scratch[..to_read])
                                } else {
                                    0
                                };

                                for (i, &sample) in stereo_scratch[..count].iter().enumerate() {
                                    for ch in 0..num_channels {
                                        data[i * num_channels + ch] = sample;
                                    }
                                }

                                let remaining_frames = mono_samples.saturating_sub(count);
                                if remaining_frames > 0 {
                                    let last = if count > 0 {
                                        stereo_scratch[count - 1]
                                    } else {
                                        cons.last_sample()
                                    };
                                    for (i, frame) in data.chunks_mut(num_channels).skip(count).enumerate() {
                                        let t = (i + 1) as f32 / remaining_frames as f32;
                                        let value = last * (1.0 - t);
                                        for ch in frame.iter_mut() {
                                            *ch = value;
                                        }
                                    }
                                }

                                if let Some(last) = data.chunks(num_channels).last() {
                                    if let Some(&sample) = last.first() {
                                        last_output_sample = sample;
                                    }
                                }
                            }
                        }
                        Err(_) => {
                            // Lock contention - avoid hard silence pop by fading from last sample.
                            let len = data.len().max(1);
                            for (i, sample) in data.iter_mut().enumerate() {
                                let t = (i + 1) as f32 / len as f32;
                                *sample = last_output_sample * (1.0 - t);
                            }
                            last_output_sample = 0.0;
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
fn find_48khz_config(
    configs: impl Iterator<Item = SupportedStreamConfigRange>,
) -> Option<cpal::SupportedStreamConfig> {
    for config in configs {
        let min_rate = config.min_sample_rate().0;
        let max_rate = config.max_sample_rate().0;

        if min_rate <= TARGET_SAMPLE_RATE && TARGET_SAMPLE_RATE <= max_rate {
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
}
