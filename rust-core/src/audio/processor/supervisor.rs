use super::diagnostics::StreamRecoveryState;
use super::lifecycle::Streams;
use super::shared::Shared;
use super::AudioProcessor;
use crate::audio::clock::now_micros;
#[cfg(feature = "vad")]
use crate::dsp::vad::GateMode;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub(super) const CHECK_INTERVAL_MS: u64 = 250;
pub(super) const CALLBACK_STALL_MS: u64 = 2500;
const WRITE_STALL_MS: u64 = 3000;
pub(super) const STARTUP_GRACE_MS: u64 = 5000;
pub(super) const CONSECUTIVE_STALL_CHECKS: u32 = 3;
/// The microphone has signal above this input level.
const SIGNAL_INPUT_DB: f32 = -35.0;
/// The output is silent below this level.
const SILENT_OUTPUT_DB: f32 = -85.0;
/// How long the output must stay silent under a live microphone to count as a stall.
pub(super) const SILENT_OUTPUT_MS: u64 = 1500;
/// A VAD-only gate with a fused score below this is closed on purpose.
#[cfg(feature = "vad")]
const GATE_CLOSED_SCORE: f32 = 0.12;

/// Ask for a restart because the streams stalled or went silent. The cause is
/// recorded as the pending reason first, so an error code left over from
/// earlier (a suppressor fault, say) cannot send the request into a suppressor
/// rebuild.
pub(super) fn request_stall_restart(
    recovery_state: &Mutex<StreamRecoveryState>,
    restart_requested: &AtomicBool,
    reason: String,
) {
    if let Ok(mut state) = recovery_state.lock() {
        state.last_reason = Some(reason);
        state.reason_pending = true;
    }
    restart_requested.store(true, Ordering::Release);
}

/// A VAD-only gate that scores the voice low has closed on purpose.
#[cfg(feature = "vad")]
fn gate_expected_closed(shared: &Shared) -> bool {
    shared.gate.rt_control.gate_mode.load(Ordering::Relaxed) == GateMode::VadOnly as u8
        && f32::from_bits(shared.gate.fused_score.load(Ordering::Relaxed)) < GATE_CLOSED_SCORE
}

#[cfg(not(feature = "vad"))]
fn gate_expected_closed(_: &Shared) -> bool {
    false
}

/// Whether the output is silent although the microphone has signal and nothing
/// explains it: not an output mute, not a VAD-only gate that is closed. Raw
/// monitor and bypass send the input on, so they never look silent here.
pub(super) fn output_silent_under_signal(shared: &Shared) -> bool {
    let level = |bits: &AtomicU32| f32::from_bits(bits.load(Ordering::Relaxed));
    level(&shared.input.rms) > SIGNAL_INPUT_DB
        && level(&shared.output.rms) < SILENT_OUTPUT_DB
        && !shared.output.muted.load(Ordering::Acquire)
        && !gate_expected_closed(shared)
}

/// Watchdog over the streams: asks for a restart once a callback or the output
/// writer has been silent for `CONSECUTIVE_STALL_CHECKS` checks, or the output
/// has stayed silent under a live microphone for `SILENT_OUTPUT_MS`.
pub(super) struct Supervisor {
    shared: Arc<Shared>,
    consecutive_stalls: u32,
    /// When the output was first seen silent under a live microphone, if it still is.
    silent_since_us: Option<u64>,
}

impl Supervisor {
    pub(super) fn new(shared: Arc<Shared>) -> Self {
        Self {
            shared,
            consecutive_stalls: 0,
            silent_since_us: None,
        }
    }

    /// Check every `CHECK_INTERVAL_MS` until `supervisor_running` is cleared.
    fn run(mut self) {
        while self
            .shared
            .stream
            .supervisor_running
            .load(Ordering::Acquire)
        {
            std::thread::sleep(Duration::from_millis(CHECK_INTERVAL_MS));
            self.check(now_micros());
        }
    }

    /// One watchdog pass at `now` (`now_micros` time). Returns whether it
    /// requested a restart.
    pub(super) fn check(&mut self, now: u64) -> bool {
        let stream = &self.shared.stream;
        if !stream.running.load(Ordering::Acquire) {
            return false;
        }

        if stream.recovering.load(Ordering::Acquire)
            || stream.restart_requested.load(Ordering::Acquire)
        {
            self.consecutive_stalls = 0;
            self.silent_since_us = None;
            return false;
        }

        if stream.recovery_suppressed.load(Ordering::Acquire)
            || self.shared.recording.active.load(Ordering::Acquire)
        {
            self.consecutive_stalls = 0;
            self.silent_since_us = None;
            return false;
        }

        let last_start = stream.last_start_time_us.load(Ordering::Relaxed);
        if last_start > 0 && now.saturating_sub(last_start) < STARTUP_GRACE_MS * 1000 {
            self.consecutive_stalls = 0;
            self.silent_since_us = None;
            return false;
        }

        let age_ms = |last: u64| {
            if last > 0 {
                now.saturating_sub(last) / 1000
            } else {
                u64::MAX
            }
        };
        let input_age_ms = age_ms(self.shared.input.callback_time_us.load(Ordering::Relaxed));
        let output_age_ms = age_ms(self.shared.output.callback_time_us.load(Ordering::Relaxed));
        let write_age_ms = age_ms(
            self.shared
                .output
                .last_write_time_us
                .load(Ordering::Relaxed),
        );

        if input_age_ms > CALLBACK_STALL_MS
            || output_age_ms > CALLBACK_STALL_MS
            || write_age_ms > WRITE_STALL_MS
        {
            self.consecutive_stalls = self.consecutive_stalls.saturating_add(1);
            if self.consecutive_stalls < CONSECUTIVE_STALL_CHECKS {
                return false;
            }
            let reason = format!(
                "input_cb_age_ms={}, output_cb_age_ms={}, output_write_age_ms={}",
                input_age_ms, output_age_ms, write_age_ms
            );
            request_stall_restart(&stream.recovery_state, &stream.restart_requested, reason);
            self.consecutive_stalls = 0;
            return true;
        }
        self.consecutive_stalls = 0;

        // The callbacks run, so judge what the loop makes of the signal.
        if !output_silent_under_signal(&self.shared) {
            self.silent_since_us = None;
            return false;
        }
        let silent_since = *self.silent_since_us.get_or_insert(now);
        if now.saturating_sub(silent_since) < SILENT_OUTPUT_MS * 1000 {
            return false;
        }
        request_stall_restart(
            &stream.recovery_state,
            &stream.restart_requested,
            "output went silent while the microphone had signal".to_string(),
        );
        self.silent_since_us = None;
        true
    }
}

impl AudioProcessor {
    pub(super) fn ensure_supervisor(&self, streams: &mut Streams) {
        if streams.supervisor_thread.is_some() {
            return;
        }

        self.shared
            .stream
            .supervisor_running
            .store(true, Ordering::Release);

        let supervisor = Supervisor::new(Arc::clone(&self.shared));
        streams.supervisor_thread = Some(std::thread::spawn(move || supervisor.run()));
    }
}
