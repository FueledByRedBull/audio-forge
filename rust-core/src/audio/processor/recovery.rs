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

/// Wait before the next restart attempt, by the number of attempts since the
/// streams last ran healthily. The last entry is the cap.
const RESTART_BACKOFF_MS: [u64; 6] = [0, 2_000, 5_000, 10_000, 20_000, 30_000];
/// Input and output must still be delivering this long after a restart before
/// it counts as recovered and the back-off starts over.
const HEALTHY_STREAM_MS: u64 = 10_000;
/// A callback older than this no longer shows a running stream (the
/// supervisor's stall limit).
const FRESH_CALLBACK_MS: u64 = 2_500;
/// A suppressor rebuild that follows another one sooner than this waits.
const SUPPRESSOR_REBUILD_MIN_INTERVAL_MS: u64 = 10_000;

impl AudioProcessor {
    /// Service pending stream recovery requests.
    ///
    /// Returns:
    /// - None: no recovery attempt was made
    /// - Some(true): recovery succeeded
    /// - Some(false): recovery attempt failed
    pub fn service_recovery(&mut self) -> Option<bool> {
        self.service_recovery_at(now_micros(), |processor| {
            let input_name = processor.input_device_name.clone();
            let output_name = processor.output_device_name.clone();
            let input_name_ordinal = processor.input_device_name_ordinal;
            let output_name_ordinal = processor.output_device_name_ordinal;
            let input_endpoint_id = processor.input_device_endpoint_id.clone();
            let output_endpoint_id = processor.output_device_endpoint_id.clone();
            processor.start_with_device_ordinals_and_ids(
                input_name.as_deref(),
                input_name_ordinal,
                input_endpoint_id.as_deref(),
                output_name.as_deref(),
                output_name_ordinal,
                output_endpoint_id.as_deref(),
            )
        })
    }

    /// A restart that opened the streams only counts as successful once both
    /// keep delivering: a device that opens but stays silent must keep growing
    /// the back-off, which the start result alone cannot show.
    fn reset_backoff_after_healthy_streams(&self, now: u64) {
        let attempt = self.last_restart_attempt_us.load(Ordering::Acquire);
        let delivering = |last_us: &AtomicU64| {
            let last_us = last_us.load(Ordering::Relaxed);
            last_us > attempt && now.saturating_sub(last_us) < FRESH_CALLBACK_MS * 1000
        };
        if attempt > 0
            && now.saturating_sub(attempt) >= HEALTHY_STREAM_MS * 1000
            && delivering(&self.last_input_callback_time_us)
            && delivering(&self.last_output_write_time)
        {
            self.restart_backoff_index.store(0, Ordering::Release);
        }
    }

    /// `service_recovery` at a given time, with the stream start supplied by
    /// the caller so tests need neither a clock nor an audio device.
    fn service_recovery_at(
        &mut self,
        now: u64,
        start: impl FnOnce(&mut Self) -> Result<String, String>,
    ) -> Option<bool> {
        if self.recovery_suppressed.load(Ordering::Acquire) {
            return None;
        }

        self.reset_backoff_after_healthy_streams(now);

        if !self.restart_requested.load(Ordering::Acquire) {
            return None;
        }

        let rt_error = RtErrorCode::from_u32(self.rt_error_code.load(Ordering::Acquire));
        let output_resampler_failed = self.output_resampler_error_count.load(Ordering::Relaxed) > 0;
        let (reason_pending, last_rebuild_us) = self
            .recovery_state
            .lock()
            .map(|state| (state.reason_pending, state.last_suppressor_rebuild_us))
            .unwrap_or((false, 0));
        // A suppressor code stays set after a single non-finite block that did
        // not request a restart, so it only explains this request when nothing
        // else (a panic, a stall, an output-resampler failure) asked for it.
        let suppressor_fault = matches!(
            rt_error,
            RtErrorCode::SuppressorBackendFailed | RtErrorCode::SuppressorNonFinite
        ) && !reason_pending
            && !output_resampler_failed;
        // A suppressor that keeps failing is rebuilt at most once per interval;
        // the request stays pending until then.
        if suppressor_fault
            && last_rebuild_us > 0
            && now.saturating_sub(last_rebuild_us) < SUPPRESSOR_REBUILD_MIN_INTERVAL_MS * 1000
        {
            return None;
        }
        if suppressor_fault && self.rebuild_noise_suppressor() {
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
                state.last_suppressor_rebuild_us = now;
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

        let last_attempt = self.last_restart_attempt_us.load(Ordering::Acquire);
        let backoff_idx = self.restart_backoff_index.load(Ordering::Acquire) as usize;
        let backoff_ms = RESTART_BACKOFF_MS[backoff_idx.min(RESTART_BACKOFF_MS.len() - 1)];

        if last_attempt > 0 && now.saturating_sub(last_attempt) < backoff_ms * 1000 {
            return None;
        }

        self.restart_requested.store(false, Ordering::Release);
        if let Ok(mut state) = self.recovery_state.lock() {
            state.reason_pending = false;
        }
        self.recovering.store(true, Ordering::Release);
        self.last_restart_attempt_us.store(now, Ordering::Release);

        self.stop();

        let (success, last_error) = match start(self) {
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

        // Every attempt grows the back-off; only healthy streams reset it.
        let next = (backoff_idx + 1).min(RESTART_BACKOFF_MS.len() - 1);
        self.restart_backoff_index
            .store(next as u32, Ordering::Release);
        if !success {
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
