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
    self.recovering.store(true, Ordering::Release);
    self.last_restart_attempt_us.store(now, Ordering::Release);

    let input_name = self.input_device_name.clone();
    let output_name = self.output_device_name.clone();

    self.stop();

    let mut success = false;
    let mut last_error: Option<String> = None;

    match self.start(input_name.as_deref(), output_name.as_deref()) {
        Ok(_) => {
            success = true;
        }
        Err(err) => {
            last_error = Some(format!("Restart failed for selected devices: {}", err));
            match self.start(None, None) {
                Ok(_) => {
                    success = true;
                }
                Err(fallback_err) => {
                    last_error = Some(format!(
                        "Restart failed for selected + default devices: {}",
                        fallback_err
                    ));
                }
            }
        }
    }

    if let Ok(mut state) = self.recovery_state.lock() {
        if success {
            state.last_error = None;
            state.restart_count = state.restart_count.saturating_add(1);
        } else {
            state.last_error = last_error.clone();
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
