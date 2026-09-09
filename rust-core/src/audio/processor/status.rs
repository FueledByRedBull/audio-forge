impl AudioProcessor {
    /// Check if processing is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get active input device name for the running stream.
    pub fn active_input_device_name(&self) -> Option<String> {
        if self.is_running() {
            self.input_device_name.clone()
        } else {
            None
        }
    }

    /// Get the stable endpoint ID for the running input stream, when present.
    pub fn active_input_device_endpoint_id(&self) -> Option<String> {
        if self.is_running() {
            self.audio_input
                .as_ref()
                .and_then(|input| input.device_info().endpoint_id.clone())
        } else {
            None
        }
    }

    /// Get the selected input friendly-name ordinal for the running stream.
    pub fn active_input_device_name_ordinal(&self) -> Option<u32> {
        if self.is_running() {
            Some(self.input_device_name_ordinal)
        } else {
            None
        }
    }

    /// Get the sample rate negotiated for the running input stream.
    pub fn active_input_sample_rate(&self) -> Option<u32> {
        if self.is_running() {
            self.audio_input
                .as_ref()
                .map(|input| input.device_info().sample_rate)
        } else {
            None
        }
    }

    /// Get active output device name for the running stream.
    pub fn active_output_device_name(&self) -> Option<String> {
        if self.is_running() {
            self.output_device_name.clone()
        } else {
            None
        }
    }

    /// Get the stable endpoint ID for the running output stream, when present.
    pub fn active_output_device_endpoint_id(&self) -> Option<String> {
        if self.is_running() {
            self.audio_output
                .as_ref()
                .and_then(|output| output.device_info().endpoint_id.clone())
        } else {
            None
        }
    }

    /// Get the selected output friendly-name ordinal for the running stream.
    pub fn active_output_device_name_ordinal(&self) -> Option<u32> {
        if self.is_running() {
            Some(self.output_device_name_ordinal)
        } else {
            None
        }
    }

    /// Set master bypass
    pub fn set_bypass(&self, bypass: bool) {
        self.bypass.store(bypass, Ordering::SeqCst);
    }

    /// Get bypass state
    pub fn is_bypass(&self) -> bool {
        self.bypass.load(Ordering::SeqCst)
    }

    /// Enable/disable true raw monitor path.
    pub fn set_raw_monitor_enabled(&self, enabled: bool) {
        self.raw_monitor_enabled.store(enabled, Ordering::Release);
    }

    /// Get true raw monitor path state.
    pub fn is_raw_monitor_enabled(&self) -> bool {
        self.raw_monitor_enabled.load(Ordering::Acquire)
    }

    /// Set input channel mixdown mode.
    pub fn set_input_channel_mode(&self, mode: InputChannelMode) {
        self.input_channel_mode.store(mode as u8, Ordering::Release);
    }

    /// Get input channel mixdown mode.
    pub fn input_channel_mode(&self) -> InputChannelMode {
        InputChannelMode::from_u8(self.input_channel_mode.load(Ordering::Acquire))
            .unwrap_or(InputChannelMode::PhaseSafeMono)
    }

    /// Set adaptive input cleanup mode.
    pub fn set_input_cleanup_mode(&self, mode: InputCleanupMode) {
        self.input_cleanup_mode.store(mode as u8, Ordering::Release);
    }

    /// Get adaptive input cleanup mode.
    pub fn input_cleanup_mode(&self) -> InputCleanupMode {
        InputCleanupMode::from_u8(self.input_cleanup_mode.load(Ordering::Acquire))
            .unwrap_or_default()
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
