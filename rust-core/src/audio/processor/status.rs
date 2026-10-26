use super::routing::InputCleanupMode;
use super::AudioProcessor;
use crate::audio::input::InputChannelMode;
use std::sync::atomic::Ordering;
#[cfg(not(windows))]
use thread_priority::{set_current_thread_priority, ThreadPriority};

impl AudioProcessor {
    /// Check if processing is running
    pub fn is_running(&self) -> bool {
        self.shared.stream.running.load(Ordering::SeqCst)
    }

    /// Get active input device name for the running stream.
    pub fn active_input_device_name(&self) -> Option<String> {
        if self.is_running() {
            self.lock_devices().input_name.clone()
        } else {
            None
        }
    }

    /// Get the stable endpoint ID for the running input stream, when present.
    pub fn active_input_device_endpoint_id(&self) -> Option<String> {
        if self.is_running() {
            self.lock_devices()
                .input_stream
                .as_ref()
                .and_then(|stream| stream.endpoint_id.clone())
        } else {
            None
        }
    }

    /// Get the selected input friendly-name ordinal for the running stream.
    pub fn active_input_device_name_ordinal(&self) -> Option<u32> {
        if self.is_running() {
            Some(self.lock_devices().input_name_ordinal)
        } else {
            None
        }
    }

    /// Get the sample rate negotiated for the running input stream.
    pub fn active_input_sample_rate(&self) -> Option<u32> {
        if self.is_running() {
            self.lock_devices()
                .input_stream
                .as_ref()
                .map(|stream| stream.sample_rate)
        } else {
            None
        }
    }

    /// Get active output device name for the running stream.
    pub fn active_output_device_name(&self) -> Option<String> {
        if self.is_running() {
            self.lock_devices().output_name.clone()
        } else {
            None
        }
    }

    /// Get the stable endpoint ID for the running output stream, when present.
    pub fn active_output_device_endpoint_id(&self) -> Option<String> {
        if self.is_running() {
            self.lock_devices()
                .output_stream
                .as_ref()
                .and_then(|stream| stream.endpoint_id.clone())
        } else {
            None
        }
    }

    /// Get the selected output friendly-name ordinal for the running stream.
    pub fn active_output_device_name_ordinal(&self) -> Option<u32> {
        if self.is_running() {
            Some(self.lock_devices().output_name_ordinal)
        } else {
            None
        }
    }

    /// Set master bypass
    pub fn set_bypass(&self, bypass: bool) {
        self.shared.mode.bypass.store(bypass, Ordering::SeqCst);
    }

    /// Get bypass state
    pub fn is_bypass(&self) -> bool {
        self.shared.mode.bypass.load(Ordering::SeqCst)
    }

    /// Enable/disable true raw monitor path.
    pub fn set_raw_monitor_enabled(&self, enabled: bool) {
        self.shared
            .mode
            .raw_monitor
            .store(enabled, Ordering::Release);
    }

    /// Get true raw monitor path state.
    pub fn is_raw_monitor_enabled(&self) -> bool {
        self.shared.mode.raw_monitor.load(Ordering::Acquire)
    }

    /// Set input channel mixdown mode.
    pub fn set_input_channel_mode(&self, mode: InputChannelMode) {
        self.shared
            .input
            .channel_mode
            .store(mode as u8, Ordering::Release);
    }

    /// Get input channel mixdown mode.
    pub fn input_channel_mode(&self) -> InputChannelMode {
        InputChannelMode::from_u8(self.shared.input.channel_mode.load(Ordering::Acquire))
            .unwrap_or(InputChannelMode::PhaseSafeMono)
    }

    /// Set adaptive input cleanup mode.
    pub fn set_input_cleanup_mode(&self, mode: InputCleanupMode) {
        self.shared
            .input
            .cleanup_mode
            .store(mode as u8, Ordering::Release);
    }

    /// Get adaptive input cleanup mode.
    pub fn input_cleanup_mode(&self) -> InputCleanupMode {
        InputCleanupMode::from_u8(self.shared.input.cleanup_mode.load(Ordering::Acquire))
            .unwrap_or_default()
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
