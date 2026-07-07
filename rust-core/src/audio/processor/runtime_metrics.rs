impl AudioProcessor {
    // === Metering ===

    /// Get input peak level in dB
    pub fn get_input_peak_db(&self) -> f32 {
        f32::from_bits(self.input_peak.load(Ordering::Relaxed))
    }

    /// Get input RMS level in dB
    pub fn get_input_rms_db(&self) -> f32 {
        f32::from_bits(self.input_rms.load(Ordering::Relaxed))
    }

    /// Get output peak level in dB
    pub fn get_output_peak_db(&self) -> f32 {
        f32::from_bits(self.output_peak.load(Ordering::Relaxed))
    }

    /// Get output RMS level in dB
    pub fn get_output_rms_db(&self) -> f32 {
        f32::from_bits(self.output_rms.load(Ordering::Relaxed))
    }

    /// Get compressor gain reduction in dB
    pub fn get_compressor_gain_reduction_db(&self) -> f32 {
        f32::from_bits(self.compressor_gain_reduction.load(Ordering::Relaxed))
    }

    /// Get current processing latency in milliseconds
    pub fn get_latency_ms(&self) -> f32 {
        let latency_us = self.latency_us.load(Ordering::Relaxed);
        latency_us as f32 / 1000.0
    }

    /// Set user latency compensation in milliseconds (added to reported latency).
    pub fn set_latency_compensation_ms(&self, compensation_ms: f32) {
        let clamped_ms = compensation_ms.clamp(0.0, 5000.0);
        let compensation_us = (clamped_ms * 1000.0) as u64;
        self.latency_compensation_us
            .store(compensation_us, Ordering::Relaxed);
    }

    /// Get configured latency compensation in milliseconds.
    pub fn get_latency_compensation_ms(&self) -> f32 {
        let compensation_us = self.latency_compensation_us.load(Ordering::Relaxed);
        compensation_us as f32 / 1000.0
    }

    // === DSP Performance Metrics ===

    /// Get DSP processing time in milliseconds
    pub fn get_dsp_time_ms(&self) -> f32 {
        let us = self.dsp_time_us.load(Ordering::Relaxed);
        us as f32 / 1000.0
    }

    /// Get input buffer fill level in samples
    pub fn get_input_buffer_samples(&self) -> u32 {
        self.input_buffer_len.load(Ordering::Relaxed)
    }

    /// Get smoothed input buffer fill level in samples
    pub fn get_input_buffer_smoothed_samples(&self) -> u32 {
        self.smoothed_input_buffer_len.load(Ordering::Relaxed)
    }

    /// Get output buffer fill level in samples
    pub fn get_output_buffer_samples(&self) -> u32 {
        self.output_buffer_len.load(Ordering::Relaxed)
    }

    /// Get active output device sample rate in Hz.
    pub fn output_sample_rate(&self) -> u32 {
        self.output_sample_rate.load(Ordering::Relaxed)
    }

    /// Get noise suppressor buffer fill level in samples
    pub fn get_rnnoise_buffer_samples(&self) -> u32 {
        self.suppressor_buffer_len.load(Ordering::Relaxed)
    }

    // === Dropped Sample Tracking ===

    /// Get dropped sample count (samples lost due to buffer overflow)
    pub fn get_dropped_samples(&self) -> u64 {
        self.input_dropped.load(Ordering::Relaxed)
    }

    /// Reset dropped sample counter
    pub fn reset_dropped_samples(&self) {
        self.input_dropped.store(0, Ordering::Relaxed);
    }

    /// Get total real-time lock contention events.
    pub fn get_lock_contention_count(&self) -> u64 {
        self.lock_contention_count.load(Ordering::Relaxed)
    }

    /// Reset lock contention counter.
    pub fn reset_lock_contention_count(&self) {
        self.lock_contention_count.store(0, Ordering::Relaxed);
    }

    pub fn get_suppressor_non_finite_count(&self) -> u64 {
        self.suppressor_non_finite_count.load(Ordering::Relaxed)
    }

    pub fn get_rt_error_code(&self) -> u32 {
        self.rt_error_code.load(Ordering::Relaxed)
    }

    pub fn get_rt_error_name(&self) -> &'static str {
        RtErrorCode::from_u32(self.get_rt_error_code()).as_str()
    }

    pub fn get_input_callback_error_count(&self) -> u64 {
        self.input_callback_error_count.load(Ordering::Relaxed)
    }

    pub fn get_output_callback_error_count(&self) -> u64 {
        self.output_callback_error_count.load(Ordering::Relaxed)
    }

    pub fn get_rt_buffer_overflow_count(&self) -> u64 {
        self.rt_buffer_overflow_count.load(Ordering::Relaxed)
    }

    /// Age of last input callback heartbeat in milliseconds.
    pub fn get_input_callback_age_ms(&self) -> u64 {
        let last = self.last_input_callback_time_us.load(Ordering::Relaxed);
        if last == 0 {
            return u64::MAX;
        }
        let now = now_micros();
        now.saturating_sub(last) / 1000
    }

    /// Age of last output callback heartbeat in milliseconds.
    pub fn get_output_callback_age_ms(&self) -> u64 {
        let last = self.last_output_callback_time_us.load(Ordering::Relaxed);
        if last == 0 {
            return u64::MAX;
        }
        let now = now_micros();
        now.saturating_sub(last) / 1000
    }

    /// Current consecutive output underrun streak.
    pub fn get_output_underrun_streak(&self) -> u32 {
        self.output_underrun_streak.load(Ordering::Relaxed)
    }

    /// Total output underrun callbacks since start.
    pub fn get_output_underrun_total(&self) -> u64 {
        self.output_underrun_total.load(Ordering::Relaxed)
    }

    /// Samples dropped by DSP-side jitter control.
    pub fn get_jitter_dropped_samples(&self) -> u64 {
        self.jitter_dropped_samples.load(Ordering::Relaxed)
    }

    /// Number of normal drift-retime adjustments.
    pub fn get_output_retime_adjustment_count(&self) -> u64 {
        self.output_retime_adjustment_count.load(Ordering::Relaxed)
    }

    /// Number of true output recovery events.
    pub fn get_output_recovery_event_count(&self) -> u64 {
        self.output_recovery_event_count.load(Ordering::Relaxed)
    }

    /// Backward-compatible alias for true output recovery events.
    pub fn get_output_recovery_count(&self) -> u64 {
        self.get_output_recovery_event_count()
    }

    pub fn get_dsp_idle_wakeup_count(&self) -> u64 {
        self.dsp_idle_wakeup_count.load(Ordering::Relaxed)
    }

    pub fn get_dsp_idle_sleep_us(&self) -> u64 {
        self.dsp_idle_sleep_us.load(Ordering::Relaxed)
    }

    /// Whether the currently selected suppressor backend is operational.
    pub fn is_noise_backend_available(&self) -> bool {
        self.noise_backend_available.load(Ordering::Relaxed)
    }

    /// Whether the suppressor backend has failed after startup.
    pub fn noise_backend_failed(&self) -> bool {
        self.noise_backend_failed.load(Ordering::Relaxed)
    }

    /// Last suppressor backend error, if any.
    pub fn noise_backend_error(&self) -> Option<String> {
        let stored = self
            .noise_backend_error
            .lock()
            .ok()
            .and_then(|error| error.clone());
        stored.or_else(|| {
            let code = RtErrorCode::from_u32(self.rt_error_code.load(Ordering::Relaxed));
            (code == RtErrorCode::SuppressorBackendFailed).then(|| code.as_str().to_string())
        })
    }

    /// Suppress watchdog-driven stream recovery during intrusive UI workflows.
    pub fn set_recovery_suppressed(&self, suppressed: bool) {
        self.recovery_suppressed
            .store(suppressed, Ordering::Release);
    }

    /// Whether watchdog-driven stream recovery is currently suppressed.
    pub fn is_recovery_suppressed(&self) -> bool {
        self.recovery_suppressed.load(Ordering::Acquire)
    }
}
