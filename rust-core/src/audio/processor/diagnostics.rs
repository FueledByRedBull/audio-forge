use crate::dsp::noise_suppressor::ControlledNoiseSuppressionEngine as NoiseSuppressionEngine;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;
#[cfg(not(windows))]
use thread_priority::{set_current_thread_priority, ThreadPriority};

pub(super) struct NoiseBackendDiagnostics {
    pub(super) available: bool,
    failed: bool,
    error: Option<String>,
}

pub(super) fn noise_backend_diagnostics(
    suppressor: &NoiseSuppressionEngine,
) -> NoiseBackendDiagnostics {
    NoiseBackendDiagnostics {
        available: suppressor.backend_available(),
        failed: suppressor.backend_failed(),
        error: suppressor.backend_error().map(str::to_string),
    }
}

pub(super) fn store_backend_diagnostics(
    available: &AtomicBool,
    failed: &AtomicBool,
    error: &Mutex<Option<String>>,
    diagnostics: NoiseBackendDiagnostics,
) {
    available.store(diagnostics.available, Ordering::Relaxed);
    failed.store(diagnostics.failed, Ordering::Relaxed);
    if let Ok(mut guard) = error.lock() {
        *guard = diagnostics.error;
    }
}

pub(super) fn update_backend_diagnostics(
    available: &AtomicBool,
    failed: &AtomicBool,
    error: &Mutex<Option<String>>,
    suppressor: &NoiseSuppressionEngine,
) {
    store_backend_diagnostics(
        available,
        failed,
        error,
        noise_backend_diagnostics(suppressor),
    );
}

pub(super) fn update_backend_status_rt(
    available: &AtomicBool,
    failed: &AtomicBool,
    rt_error_code: &AtomicU32,
    suppressor: &NoiseSuppressionEngine,
) {
    available.store(suppressor.backend_available(), Ordering::Relaxed);
    failed.store(suppressor.backend_failed(), Ordering::Relaxed);
    if suppressor.backend_failed() || suppressor.backend_error().is_some() {
        crate::audio::rt::store_rt_error(
            rt_error_code,
            crate::audio::rt::RtErrorCode::SuppressorBackendFailed,
        );
    }
}

#[derive(Default)]
pub(super) struct StreamRecoveryState {
    pub(super) last_error: Option<String>,
    pub(super) last_reason: Option<String>,
    /// `last_reason` describes the pending restart request, so recovery must
    /// not replace it with a callback's error code.
    pub(super) reason_pending: bool,
    pub(super) restart_count: u64,
    /// When the suppressor was last rebuilt (`now_micros`; 0 = never).
    pub(super) last_suppressor_rebuild_us: u64,
}
