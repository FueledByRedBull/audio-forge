impl AudioProcessor {
fn ensure_supervisor(&mut self) {
    if self.supervisor_thread.is_some() {
        return;
    }

    self.supervisor_running.store(true, Ordering::Release);

    let running = Arc::clone(&self.running);
    let supervisor_running = Arc::clone(&self.supervisor_running);
    let restart_requested = Arc::clone(&self.restart_requested);
    let recovering = Arc::clone(&self.recovering);
    let last_input_callback_time_us = Arc::clone(&self.last_input_callback_time_us);
    let last_output_callback_time_us = Arc::clone(&self.last_output_callback_time_us);
    let last_output_write_time = Arc::clone(&self.last_output_write_time);
    let last_start_time_us = Arc::clone(&self.last_start_time_us);
    let recovery_state = Arc::clone(&self.recovery_state);
    let recording_active = Arc::clone(&self.recording_active);
    let output_muted = Arc::clone(&self.output_muted);
    let recovery_suppressed = Arc::clone(&self.recovery_suppressed);

    self.supervisor_thread = Some(std::thread::spawn(move || {
        const CHECK_INTERVAL_MS: u64 = 250;
        const CALLBACK_STALL_MS: u64 = 2500;
        const WRITE_STALL_MS: u64 = 3000;
        const STARTUP_GRACE_MS: u64 = 5000;
        const CONSECUTIVE_STALL_CHECKS: u32 = 3;
        let mut consecutive_stalls = 0u32;

        while supervisor_running.load(Ordering::Acquire) {
            std::thread::sleep(std::time::Duration::from_millis(CHECK_INTERVAL_MS));

            if !running.load(Ordering::Acquire) {
                continue;
            }

            if recovering.load(Ordering::Acquire) || restart_requested.load(Ordering::Acquire) {
                consecutive_stalls = 0;
                continue;
            }

            if recovery_suppressed.load(Ordering::Acquire)
                || recording_active.load(Ordering::Acquire)
                || output_muted.load(Ordering::Acquire)
            {
                consecutive_stalls = 0;
                continue;
            }

            let now = now_micros();
            let last_start = last_start_time_us.load(Ordering::Relaxed);
            if last_start > 0 && now.saturating_sub(last_start) < STARTUP_GRACE_MS * 1000 {
                consecutive_stalls = 0;
                continue;
            }

            let last_in = last_input_callback_time_us.load(Ordering::Relaxed);
            let last_out = last_output_callback_time_us.load(Ordering::Relaxed);
            let last_write = last_output_write_time.load(Ordering::Relaxed);

            let input_age_ms = if last_in > 0 {
                now.saturating_sub(last_in) / 1000
            } else {
                u64::MAX
            };
            let output_age_ms = if last_out > 0 {
                now.saturating_sub(last_out) / 1000
            } else {
                u64::MAX
            };
            let write_age_ms = if last_write > 0 {
                now.saturating_sub(last_write) / 1000
            } else {
                u64::MAX
            };

            if input_age_ms > CALLBACK_STALL_MS
                || output_age_ms > CALLBACK_STALL_MS
                || write_age_ms > WRITE_STALL_MS
            {
                consecutive_stalls = consecutive_stalls.saturating_add(1);
                if consecutive_stalls < CONSECUTIVE_STALL_CHECKS {
                    continue;
                }
                let reason = format!(
                    "input_cb_age_ms={}, output_cb_age_ms={}, output_write_age_ms={}",
                    input_age_ms, output_age_ms, write_age_ms
                );
                if let Ok(mut state) = recovery_state.lock() {
                    state.last_reason = Some(reason);
                }
                restart_requested.store(true, Ordering::Release);
                consecutive_stalls = 0;
            } else {
                consecutive_stalls = 0;
            }
        }
    }));
}
}
