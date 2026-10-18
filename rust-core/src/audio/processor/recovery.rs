/// The restart cause an audio callback left only as an error code or counter.
/// Resampler failures always request a restart; a stream error does only when
/// the device is gone, so it is checked last.
fn restart_cause(rt_error: RtErrorCode, output_resampler_failed: bool) -> Option<&'static str> {
    match rt_error {
        RtErrorCode::InputResamplerFailed => Some(rt_error.as_str()),
        _ if output_resampler_failed => Some("output resampler failed"),
        RtErrorCode::InputStreamError | RtErrorCode::OutputStreamError => Some(rt_error.as_str()),
        _ => None,
    }
}

impl AudioProcessor {
    /// Service pending stream recovery requests.
    ///
    /// Returns:
    /// - None: no recovery attempt was made
    /// - Some(true): recovery succeeded
    /// - Some(false): recovery attempt failed
    pub fn service_recovery(&mut self) -> Option<bool> {
        if self.recovery_suppressed.load(Ordering::Acquire) {
            return None;
        }

        if !self.restart_requested.load(Ordering::Acquire) {
            return None;
        }

        let rt_error = RtErrorCode::from_u32(self.rt_error_code.load(Ordering::Acquire));
        let output_resampler_failed = self.output_resampler_error_count.load(Ordering::Relaxed) > 0;
        let reason_pending = self
            .recovery_state
            .lock()
            .map(|state| state.reason_pending)
            .unwrap_or(false);
        // A suppressor code stays set after a single non-finite block that did
        // not request a restart, so it only explains this request when nothing
        // else (a panic, a stall, an output-resampler failure) asked for it.
        if matches!(
            rt_error,
            RtErrorCode::SuppressorBackendFailed | RtErrorCode::SuppressorNonFinite
        ) && !reason_pending
            && !output_resampler_failed
            && self.rebuild_noise_suppressor()
        {
            self.restart_requested.store(false, Ordering::Release);
            // Handled, unless a newer error was stored during the rebuild.
            let _ = self.rt_error_code.compare_exchange(
                rt_error as u32,
                RtErrorCode::None as u32,
                Ordering::AcqRel,
                Ordering::Relaxed,
            );
            if let Ok(mut state) = self.recovery_state.lock() {
                state.last_error = None;
                state.last_reason =
                    Some(format!("{}; suppressor engine rebuilt", rt_error.as_str()));
            }
            return Some(true);
        }

        // Stream errors and resampler failures request a restart from the audio
        // callbacks, which cannot record why; name the cause here unless a panic
        // or stall already recorded its own reason for this request.
        if let Ok(mut state) = self.recovery_state.lock() {
            if !state.reason_pending {
                if let Some(cause) = restart_cause(rt_error, output_resampler_failed) {
                    state.last_reason = Some(cause.to_string());
                    state.reason_pending = true;
                }
            }
        }

        let now = now_micros();
        let last_attempt = self.last_restart_attempt_us.load(Ordering::Acquire);
        let backoff_idx = self.restart_backoff_index.load(Ordering::Acquire);
        let backoff_ms = match backoff_idx {
            0 => 0,
            1 => 2000,
            2 => 5000,
            _ => 10000,
        };

        if last_attempt > 0 && now.saturating_sub(last_attempt) < backoff_ms * 1000 {
            return None;
        }

        self.restart_requested.store(false, Ordering::Release);
        if let Ok(mut state) = self.recovery_state.lock() {
            state.reason_pending = false;
        }
        self.recovering.store(true, Ordering::Release);
        self.last_restart_attempt_us.store(now, Ordering::Release);

        let input_name = self.input_device_name.clone();
        let output_name = self.output_device_name.clone();
        let input_name_ordinal = self.input_device_name_ordinal;
        let output_name_ordinal = self.output_device_name_ordinal;
        let input_endpoint_id = self.input_device_endpoint_id.clone();
        let output_endpoint_id = self.output_device_endpoint_id.clone();

        self.stop();

        let (success, last_error) = match self.start_with_device_ordinals_and_ids(
            input_name.as_deref(),
            input_name_ordinal,
            input_endpoint_id.as_deref(),
            output_name.as_deref(),
            output_name_ordinal,
            output_endpoint_id.as_deref(),
        ) {
            Ok(_) => (true, None),
            Err(err) => (
                false,
                Some(format!("Restart failed for selected devices: {err}")),
            ),
        };

        if let Ok(mut state) = self.recovery_state.lock() {
            if success {
                state.last_error = None;
                state.restart_count = state.restart_count.saturating_add(1);
            } else {
                state.last_error = last_error.clone();
                // The retry below is still for the recorded cause.
                state.reason_pending = true;
            }
        }

        if success {
            self.restart_backoff_index.store(0, Ordering::Release);
        } else {
            let next = (backoff_idx + 1).min(3);
            self.restart_backoff_index.store(next, Ordering::Release);
            self.restart_requested.store(true, Ordering::Release);
        }

        self.recovering.store(false, Ordering::Release);
        Some(success)
    }

    /// Whether a restart has been requested by the supervisor.
    pub fn is_recovery_requested(&self) -> bool {
        self.restart_requested.load(Ordering::Acquire)
    }

    /// Whether a recovery attempt is in progress.
    pub fn is_recovering(&self) -> bool {
        self.recovering.load(Ordering::Acquire)
    }

    /// Number of successful stream restarts.
    pub fn get_stream_restart_count(&self) -> u64 {
        self.recovery_state
            .lock()
            .map(|s| s.restart_count)
            .unwrap_or(0)
    }

    /// Last recovery error string, if any.
    pub fn get_last_stream_error(&self) -> Option<String> {
        self.recovery_state
            .lock()
            .ok()
            .and_then(|s| s.last_error.clone())
    }

    /// Last recovery trigger reason string, if any.
    pub fn get_last_restart_reason(&self) -> Option<String> {
        self.recovery_state
            .lock()
            .ok()
            .and_then(|s| s.last_reason.clone())
    }
}
