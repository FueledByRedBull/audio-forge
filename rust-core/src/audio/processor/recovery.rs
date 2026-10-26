use super::lifecycle::Streams;
use super::supervisor::{output_silent_under_signal, SILENT_OUTPUT_REASON};
use super::AudioProcessor;
use crate::audio::clock::now_micros;
use crate::audio::rt::RtErrorCode;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(windows))]
use thread_priority::{set_current_thread_priority, ThreadPriority};

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
pub(super) const RESTART_BACKOFF_MS: [u64; 6] = [0, 2_000, 5_000, 10_000, 20_000, 30_000];
/// Input and output must still be delivering this long after a restart before
/// it counts as recovered and the back-off starts over.
const HEALTHY_STREAM_MS: u64 = 10_000;
/// A callback older than this no longer shows a running stream (the
/// supervisor's stall limit).
const FRESH_CALLBACK_MS: u64 = 2_500;
/// A suppressor rebuild that follows another one sooner than this is skipped.
const SUPPRESSOR_REBUILD_MIN_INTERVAL_MS: u64 = 10_000;

impl AudioProcessor {
    /// Service pending stream recovery requests.
    ///
    /// Returns:
    /// - None: no recovery attempt was made
    /// - Some(true): recovery succeeded
    /// - Some(false): recovery attempt failed
    pub fn service_recovery(&self) -> Option<bool> {
        self.service_recovery_at(now_micros(), |processor, streams| {
            let devices = processor.lock_devices();
            let input_name = devices.input_name.clone();
            let output_name = devices.output_name.clone();
            let input_name_ordinal = devices.input_name_ordinal;
            let output_name_ordinal = devices.output_name_ordinal;
            let input_endpoint_id = devices.input_endpoint_id.clone();
            let output_endpoint_id = devices.output_endpoint_id.clone();
            drop(devices);
            processor.start_locked(
                streams,
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
    /// keep delivering for `HEALTHY_STREAM_MS` with no silent output seen under
    /// a live microphone in that time: a device that opens but stays silent must
    /// keep growing the back-off, which the start result alone cannot show, and
    /// one pause in speech must not clear it. A request still pending means the
    /// last restart did not cure it, whatever the streams deliver.
    fn reset_backoff_after_healthy_streams(&self, now: u64) {
        let attempt = self
            .shared
            .stream
            .last_restart_attempt_us
            .load(Ordering::Acquire);
        let silent_seen = self
            .shared
            .stream
            .silent_output_seen_us
            .load(Ordering::Relaxed);
        let delivering = |last_us: &AtomicU64| {
            let last_us = last_us.load(Ordering::Relaxed);
            last_us > attempt && now.saturating_sub(last_us) < FRESH_CALLBACK_MS * 1000
        };
        if attempt > 0
            && now.saturating_sub(attempt.max(silent_seen)) >= HEALTHY_STREAM_MS * 1000
            && delivering(&self.shared.input.callback_time_us)
            && delivering(&self.shared.output.last_write_time_us)
            && !self.shared.stream.restart_requested.load(Ordering::Acquire)
        {
            self.shared
                .stream
                .restart_backoff_index
                .store(0, Ordering::Release);
        }
    }

    /// Drop a pending restart that was asked for because the output went silent
    /// under a live microphone once that has ended by itself. Another cause that
    /// asked too keeps the request, and so does a stopped stream: that is the
    /// retry of a restart that failed.
    fn withdraw_stale_silent_output_request(&self) -> bool {
        let stream = &self.shared.stream;
        let silent_output_pending = stream.recovery_state.lock().is_ok_and(|state| {
            state.reason_pending && state.last_reason.as_deref() == Some(SILENT_OUTPUT_REASON)
        });
        if !silent_output_pending
            || !stream.running.load(Ordering::Acquire)
            || self.shared.dsp.rt_error_code.load(Ordering::Acquire) != RtErrorCode::None as u32
            || self
                .shared
                .output
                .resampler_error_count
                .load(Ordering::Relaxed)
                > 0
            || output_silent_under_signal(&self.shared)
        {
            return false;
        }
        if let Ok(mut state) = stream.recovery_state.lock() {
            state.reason_pending = false;
        }
        // A cause that arrived meanwhile puts its request back.
        self.finish_suppressor_request(RtErrorCode::None, false);
        true
    }

    /// Clear the restart request once a suppressor fault has been dealt with
    /// (`clear_code`: it was rebuilt, so its error code is done too). Another
    /// cause may have asked for a restart since the request was read: a stream
    /// error stores its own code first, a stall marks its reason pending first,
    /// an output-resampler failure counts first. That request is put back, so
    /// it is never lost to this one.
    pub(super) fn finish_suppressor_request(&self, handled: RtErrorCode, clear_code: bool) {
        self.shared
            .stream
            .restart_requested
            .store(false, Ordering::Release);
        let code_unchanged = if clear_code {
            self.shared
                .dsp
                .rt_error_code
                .compare_exchange(
                    handled as u32,
                    RtErrorCode::None as u32,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                )
                .is_ok()
        } else {
            self.shared.dsp.rt_error_code.load(Ordering::Acquire) == handled as u32
        };
        let other_reason = self
            .shared
            .stream
            .recovery_state
            .lock()
            .map(|state| state.reason_pending)
            .unwrap_or(false)
            || self
                .shared
                .output
                .resampler_error_count
                .load(Ordering::Relaxed)
                > 0;
        if !code_unchanged || other_reason {
            self.shared
                .stream
                .restart_requested
                .store(true, Ordering::Release);
        }
    }

    /// `service_recovery` at a given time, with the stream start supplied by
    /// the caller so tests need neither a clock nor an audio device.
    pub(super) fn service_recovery_at(
        &self,
        now: u64,
        start: impl FnOnce(&Self, &mut Streams) -> Result<String, String>,
    ) -> Option<bool> {
        let stop_epoch = self.shared.stream.stop_epoch.load(Ordering::Acquire);
        if self
            .shared
            .stream
            .recovery_suppressed
            .load(Ordering::Acquire)
        {
            return None;
        }

        self.reset_backoff_after_healthy_streams(now);

        if !self.shared.stream.restart_requested.load(Ordering::Acquire)
            || self.withdraw_stale_silent_output_request()
        {
            return None;
        }

        let rt_error = RtErrorCode::from_u32(self.shared.dsp.rt_error_code.load(Ordering::Acquire));
        let output_resampler_failed = self
            .shared
            .output
            .resampler_error_count
            .load(Ordering::Relaxed)
            > 0;
        let (reason_pending, last_rebuild_us) = self
            .shared
            .stream
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
        // A suppressor that keeps failing is rebuilt at most once per interval.
        // The request is dropped rather than held: the DSP thread asks again
        // while the fault lasts, and a pending request would keep the supervisor
        // from noticing a stalled stream, which must still restart it.
        if suppressor_fault
            && last_rebuild_us > 0
            && now.saturating_sub(last_rebuild_us) < SUPPRESSOR_REBUILD_MIN_INTERVAL_MS * 1000
        {
            self.finish_suppressor_request(rt_error, false);
            return None;
        }
        if suppressor_fault && self.rebuild_noise_suppressor() {
            self.finish_suppressor_request(rt_error, true);
            if let Ok(mut state) = self.shared.stream.recovery_state.lock() {
                state.last_error = None;
                if !state.reason_pending {
                    state.last_reason =
                        Some(format!("{}; suppressor engine rebuilt", rt_error.as_str()));
                }
                state.last_suppressor_rebuild_us = now;
            }
            return Some(true);
        }

        // Stream errors and resampler failures request a restart from the audio
        // callbacks, which cannot record why; name the cause here unless a panic
        // or stall already recorded its own reason for this request.
        if let Ok(mut state) = self.shared.stream.recovery_state.lock() {
            if !state.reason_pending {
                if let Some(cause) = restart_cause(rt_error, output_resampler_failed) {
                    state.last_reason = Some(cause.to_string());
                    state.reason_pending = true;
                }
            }
        }

        let last_attempt = self
            .shared
            .stream
            .last_restart_attempt_us
            .load(Ordering::Acquire);
        let backoff_idx = self
            .shared
            .stream
            .restart_backoff_index
            .load(Ordering::Acquire) as usize;
        let backoff_ms = RESTART_BACKOFF_MS[backoff_idx.min(RESTART_BACKOFF_MS.len() - 1)];

        if last_attempt > 0 && now.saturating_sub(last_attempt) < backoff_ms * 1000 {
            return None;
        }

        self.shared.stream.recovering.store(true, Ordering::Release);
        // One lock hold for the stop and the start, so no other caller can start or
        // stop in between.
        let mut streams = self.lock_streams();
        // A stop or a suppression that came first while this waited for the lock
        // stands: the restart would reopen devices the caller just let go.
        if self.shared.stream.stop_epoch.load(Ordering::Acquire) != stop_epoch
            || self
                .shared
                .stream
                .recovery_suppressed
                .load(Ordering::Acquire)
        {
            self.shared
                .stream
                .recovering
                .store(false, Ordering::Release);
            return None;
        }
        // The wait for the lock can outlast the condition that asked.
        if self.withdraw_stale_silent_output_request() {
            self.shared
                .stream
                .recovering
                .store(false, Ordering::Release);
            return None;
        }
        self.shared
            .stream
            .restart_requested
            .store(false, Ordering::Release);
        if let Ok(mut state) = self.shared.stream.recovery_state.lock() {
            state.reason_pending = false;
        }
        self.shared
            .stream
            .last_restart_attempt_us
            .store(now, Ordering::Release);
        self.stop_locked(&mut streams);

        let (success, last_error) = match start(self, &mut streams) {
            Ok(_) => (true, None),
            Err(err) => (
                false,
                Some(format!("Restart failed for selected devices: {err}")),
            ),
        };
        drop(streams);

        if let Ok(mut state) = self.shared.stream.recovery_state.lock() {
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
        self.shared
            .stream
            .restart_backoff_index
            .store(next as u32, Ordering::Release);
        if !success {
            self.shared
                .stream
                .restart_requested
                .store(true, Ordering::Release);
        }

        self.shared
            .stream
            .recovering
            .store(false, Ordering::Release);
        Some(success)
    }

    /// Whether a restart has been requested by the supervisor.
    pub fn is_recovery_requested(&self) -> bool {
        self.shared.stream.restart_requested.load(Ordering::Acquire)
    }

    /// Whether a recovery attempt is in progress.
    pub fn is_recovering(&self) -> bool {
        self.shared.stream.recovering.load(Ordering::Acquire)
    }

    /// Number of successful stream restarts.
    pub fn get_stream_restart_count(&self) -> u64 {
        self.shared
            .stream
            .recovery_state
            .lock()
            .map(|s| s.restart_count)
            .unwrap_or(0)
    }

    /// Last recovery error string, if any.
    pub fn get_last_stream_error(&self) -> Option<String> {
        self.shared
            .stream
            .recovery_state
            .lock()
            .ok()
            .and_then(|s| s.last_error.clone())
    }

    /// Last recovery trigger reason string, if any.
    pub fn get_last_restart_reason(&self) -> Option<String> {
        self.shared
            .stream
            .recovery_state
            .lock()
            .ok()
            .and_then(|s| s.last_reason.clone())
    }
}
