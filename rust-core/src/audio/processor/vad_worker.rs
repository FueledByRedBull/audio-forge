#[cfg(all(test, feature = "vad"))]
static VAD_WORKER_FORCE_INFERENCE_ERROR: AtomicBool = AtomicBool::new(false);

impl AudioProcessor {
#[cfg(feature = "vad")]
fn ensure_vad_worker(&mut self, vad_consumer: super::buffer::AudioConsumer) {
    if let Some(handle) = self.vad_worker_thread.take() {
        if !handle.is_finished() {
            self.vad_worker_thread = Some(handle);
            return;
        }
        let _ = handle.join();
    }

    self.vad_worker_running.store(true, Ordering::Release);
    let running = Arc::clone(&self.vad_worker_running);
    let probability = Arc::clone(&self.vad_raw_probability);
    let available = Arc::clone(&self.vad_backend_available);
    let last_update_us = Arc::clone(&self.vad_last_update_us);
    let sample_rate = self.sample_rate;
    let threshold = self
        .gate_rt_control
        .snapshot()
        .unwrap_or_else(GateControlState::new)
        .vad_threshold;

    self.vad_worker_thread = Some(std::thread::spawn(move || {
        let mut worker_consumer = vad_consumer;
        let mut vad = None;
        let mut local = Vec::with_capacity(VAD_WORKER_MAX_BUFFER_SAMPLES);
        while running.load(Ordering::Acquire) {
            if vad.is_none() {
                match SileroVAD::new(sample_rate, threshold) {
                    Ok(candidate) => {
                        available.store(true, Ordering::Release);
                        vad = Some(candidate);
                    }
                    Err(_) => {
                        available.store(false, Ordering::Release);
                        std::thread::sleep(std::time::Duration::from_millis(50));
                        continue;
                    }
                }
            }

            local.clear();
            let available_samples = worker_consumer.len();
            if available_samples > 0 {
                let to_read = available_samples.min(VAD_WORKER_MAX_BUFFER_SAMPLES);
                local.resize(to_read, 0.0);
                let read = worker_consumer.read(&mut local);
                local.truncate(read);
            }

            if !local.is_empty() {
                #[cfg(test)]
                if VAD_WORKER_FORCE_INFERENCE_ERROR.swap(false, Ordering::AcqRel) {
                    available.store(false, Ordering::Release);
                    vad = None;
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    continue;
                }
                let inference = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    vad.as_mut().expect("VAD backend initialized").process_latest(&local)
                }));
                match inference {
                    Ok(Ok(Some(prob))) => {
                        probability.store(prob.clamp(0.0, 1.0).to_bits(), Ordering::Release);
                        last_update_us.store(now_micros(), Ordering::Release);
                        available.store(true, Ordering::Release);
                    }
                    Ok(Ok(None)) => {}
                    Ok(Err(_)) | Err(_) => {
                        available.store(false, Ordering::Release);
                        vad = None;
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                }
                local.clear();
            } else {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
    }));
}

#[cfg(feature = "vad")]
fn stop_vad_worker(&mut self) {
    self.vad_worker_running.store(false, Ordering::Release);
    if let Some(handle) = self.vad_worker_thread.take() {
        let _ = handle.join();
    }
    self.vad_available.store(false, Ordering::Release);
    self.vad_backend_available
        .store(false, Ordering::Release);
    self.vad_last_update_us.store(0, Ordering::Release);
}
}
