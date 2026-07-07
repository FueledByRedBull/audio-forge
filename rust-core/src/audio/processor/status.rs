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

    /// Get active output device name for the running stream.
    pub fn active_output_device_name(&self) -> Option<String> {
        if self.is_running() {
            self.output_device_name.clone()
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

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }


}
