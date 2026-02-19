//! Audio output playback using cpal
//!
//! Real-time audio playback to VB Audio Cable or other output devices.
//! Requests 48kHz sample rate from device.
//!
//! Adapted from Spectral Workbench project.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig, SupportedStreamConfigRange};
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

impl AudioOutput {
    /// Create audio output from default device
    pub fn from_default_device(
        consumer: AudioConsumer,
        recording_active: Arc<AtomicBool>,
        last_callback_time_us: Arc<AtomicU64>,
        underrun_streak: Arc<AtomicU32>,
        total_underruns: Arc<AtomicU64>,
    ) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host.default_output_device().ok_or(AudioError::NoDevice)?;

        Self::from_device(
            device,
            consumer,
            recording_active,
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
        last_callback_time_us: Arc<AtomicU64>,
        underrun_streak: Arc<AtomicU32>,
        total_underruns: Arc<AtomicU64>,
    ) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host
            .output_devices()
            .map_err(|e| AudioError::DeviceName(e.to_string()))?
            .find(|d| d.name().map(|n| n == name).unwrap_or(false))
            .ok_or_else(|| AudioError::DeviceNotFound(name.to_string()))?;

        Self::from_device(
            device,
            consumer,
            recording_active,
            last_callback_time_us,
            underrun_streak,
            total_underruns,
        )
    }

    /// Create audio output from specific device
    pub fn from_device(
        device: Device,
        consumer: AudioConsumer,
        recording_active: Arc<AtomicBool>,
        last_callback_time_us: Arc<AtomicU64>,
        underrun_streak: Arc<AtomicU32>,
        total_underruns: Arc<AtomicU64>,
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

        let mut consumer = consumer;
        let num_channels = channels as usize;
        // Fixed callback scratch to avoid heap growth in real-time thread.
        const OUTPUT_SCRATCH_CAPACITY: usize = 8192;
        let mut stereo_scratch: Vec<f32> = vec![0.0; OUTPUT_SCRATCH_CAPACITY];

        // Build audio output stream
        let recording_active_clone = Arc::clone(&recording_active);

        let stream = device
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    last_callback_time_us.store(now_micros(), Ordering::Relaxed);

                    // Check if recording is active - if so, output silence to prevent
                    // user from hearing themselves while recording
                    if recording_active_clone.load(Ordering::Relaxed) {
                        underrun_streak.store(0, Ordering::Relaxed);
                        for sample in data.iter_mut() {
                            *sample = 0.0;
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

                        // Mono output - read whatever is available, then smooth-fill remainder.
                        let to_read = available.min(data.len());
                        let read = if to_read > 0 {
                            consumer.read(&mut data[..to_read])
                        } else {
                            0
                        };

                        if read < data.len() {
                            let last = if read > 0 {
                                data[read - 1]
                            } else {
                                consumer.last_sample()
                            };
                            let remain = data.len() - read;
                            for (i, sample) in data[read..].iter_mut().enumerate() {
                                // Linear fade from last sample to 0.0 over missing region.
                                let t = (i + 1) as f32 / remain as f32;
                                *sample = last * (1.0 - t);
                            }
                        }

                    } else {
                        // Stereo output - duplicate mono to all channels
                        let mono_samples = data.len() / num_channels;
                        if available < mono_samples {
                            underrun_streak.fetch_add(1, Ordering::Relaxed);
                            total_underruns.fetch_add(1, Ordering::Relaxed);
                        } else {
                            underrun_streak.store(0, Ordering::Relaxed);
                        }

                        let mut copied_frames = 0usize;
                        let to_read = available.min(mono_samples);
                        while copied_frames < to_read {
                            let batch = (to_read - copied_frames).min(OUTPUT_SCRATCH_CAPACITY);
                            let count = consumer.read(&mut stereo_scratch[..batch]);
                            if count == 0 {
                                break;
                            }

                            for (i, &sample) in stereo_scratch[..count].iter().enumerate() {
                                let frame_idx = copied_frames + i;
                                for ch in 0..num_channels {
                                    data[frame_idx * num_channels + ch] = sample;
                                }
                            }
                            copied_frames += count;
                            if count < batch {
                                break;
                            }
                        }

                        let remaining_frames = mono_samples.saturating_sub(copied_frames);
                        if remaining_frames > 0 {
                            let last = if copied_frames > 0 {
                                data[(copied_frames - 1) * num_channels]
                            } else {
                                consumer.last_sample()
                            };
                            for (i, frame) in data
                                .chunks_mut(num_channels)
                                .skip(copied_frames)
                                .enumerate()
                            {
                                let t = (i + 1) as f32 / remaining_frames as f32;
                                let value = last * (1.0 - t);
                                for ch in frame.iter_mut() {
                                    *ch = value;
                                }
                            }
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
