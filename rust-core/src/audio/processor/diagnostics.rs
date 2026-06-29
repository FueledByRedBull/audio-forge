struct NoiseBackendDiagnostics {
    available: bool,
    failed: bool,
    error: Option<String>,
}

fn noise_backend_diagnostics(suppressor: &NoiseSuppressionEngine) -> NoiseBackendDiagnostics {
    NoiseBackendDiagnostics {
        available: suppressor.backend_available(),
        failed: suppressor.backend_failed(),
        error: suppressor.backend_error().map(str::to_string),
    }
}

fn store_backend_diagnostics(
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

fn update_backend_diagnostics(
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

fn update_backend_status_rt(
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
struct StreamRecoveryState {
    last_error: Option<String>,
    last_reason: Option<String>,
    restart_count: u64,
}
