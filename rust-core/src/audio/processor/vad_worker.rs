impl AudioProcessor {
#[cfg(feature = "vad")]
fn ensure_vad_worker(&mut self, vad_consumer: super::buffer::AudioConsumer) {
    if self.vad_worker_thread.is_some() {
        return;
    }

    self.vad_worker_running.store(true, Ordering::Release);
    let running = Arc::clone(&self.vad_worker_running);
    let probability = Arc::clone(&self.vad_probability);
    let available = Arc::clone(&self.vad_available);
    let last_update_us = Arc::clone(&self.vad_last_update_us);
    let sample_rate = self.sample_rate;
    let threshold = self
        .gate_rt_control
        .snapshot()
        .unwrap_or_else(GateControlState::new)
        .vad_threshold;

    self.vad_worker_thread = Some(std::thread::spawn(move || {
        let mut worker_consumer = vad_consumer;
        let mut vad = match SileroVAD::new(sample_rate, threshold) {
            Ok(vad) => {
                available.store(true, Ordering::Release);
                vad
            }
            Err(_) => {
                available.store(false, Ordering::Release);
                while running.load(Ordering::Acquire) {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                return;
            }
        };

        let mut local = Vec::with_capacity(VAD_WORKER_MAX_BUFFER_SAMPLES);
        while running.load(Ordering::Acquire) {
            local.clear();
            let available_samples = worker_consumer.len();
            if available_samples > 0 {
                let to_read = available_samples.min(VAD_WORKER_MAX_BUFFER_SAMPLES);
                local.resize(to_read, 0.0);
                let read = worker_consumer.read(&mut local);
                local.truncate(read);
            }

            if !local.is_empty() {
                match vad.process(&local) {
                    Ok(prob) => {
                        probability.store(prob.clamp(0.0, 1.0).to_bits(), Ordering::Release);
                        last_update_us.store(now_micros(), Ordering::Release);
                        available.store(true, Ordering::Release);
                    }
                    Err(_) => {
                        available.store(false, Ordering::Release);
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
    self.vad_last_update_us.store(0, Ordering::Release);
}
}
