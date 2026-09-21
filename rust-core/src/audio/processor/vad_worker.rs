#[cfg(all(test, feature = "vad"))]
static VAD_WORKER_FORCE_INFERENCE_ERROR: AtomicBool = AtomicBool::new(false);
#[cfg(all(test, feature = "vad"))]
static VAD_WORKER_LAST_PRE_GAIN_BITS: AtomicU32 = AtomicU32::new(1.0_f32.to_bits());
#[cfg(all(test, feature = "vad"))]
static VAD_WORKER_DISCONTINUITY_RESETS: AtomicU64 = AtomicU64::new(0);
#[cfg(all(test, feature = "vad"))]
static VAD_WORKER_DISCONTINUITY_FLUSHED_SAMPLES: AtomicU64 = AtomicU64::new(0);
#[cfg(all(test, feature = "vad"))]
static VAD_WORKER_MODEL_INITS: AtomicU64 = AtomicU64::new(0);
#[cfg(all(test, feature = "vad"))]
thread_local! {
    static VAD_WORKER_FORCE_DISCONTINUITY_AFTER_FINAL_CHECK: Cell<bool> = const { Cell::new(false) };
}

#[cfg(feature = "vad")]
const VAD_RESULT_SNAPSHOT_MAX_RETRIES: usize = 8;

#[cfg(feature = "vad")]
#[derive(Clone, Copy, Debug)]
struct VadResultSnapshot {
    probability: f32,
    backend_available: bool,
    last_update_us: u64,
    source_sample_end: u64,
    generation: u64,
}

#[cfg(feature = "vad")]
impl VadResultSnapshot {
    #[inline]
    fn is_fresh(&self, source_sample_clock: u64, now_us: u64, sample_rate: u32) -> bool {
        if self.last_update_us == 0 {
            return false;
        }
        let wall_age_us = now_us.saturating_sub(self.last_update_us);
        let source_age_us = source_sample_clock
            .saturating_sub(self.source_sample_end)
            .saturating_mul(1_000_000)
            / u64::from(sample_rate.max(1));
        wall_age_us.max(source_age_us) <= VAD_PROBABILITY_STALE_US
    }
}

#[cfg(feature = "vad")]
struct VadResultAtomics<'a> {
    sequence: &'a AtomicU64,
    result_generation: &'a AtomicU64,
    probability: &'a AtomicU32,
    backend_available: &'a AtomicBool,
    last_update_us: &'a AtomicU64,
    source_sample_end: &'a AtomicU64,
    source_discontinuity: &'a AtomicU64,
}

#[cfg(feature = "vad")]
impl VadResultAtomics<'_> {
    #[inline]
    fn invalidate(&self, backend_available: bool) {
        let generation = self.source_discontinuity.load(Ordering::Acquire);
        self.sequence.fetch_add(1, Ordering::AcqRel);
        self.probability.store(0.0_f32.to_bits(), Ordering::Relaxed);
        self.last_update_us.store(0, Ordering::Relaxed);
        self.source_sample_end.store(0, Ordering::Relaxed);
        self.backend_available
            .store(backend_available, Ordering::Relaxed);
        self.result_generation.store(generation, Ordering::Relaxed);
        self.sequence.fetch_add(1, Ordering::Release);
    }

    #[inline]
    fn publish_vad_inference_result(
        &self,
        expected_generation: u64,
        probability: f32,
        source_sample_end: u64,
        last_update_us: u64,
        backend_available: bool,
    ) -> bool {
        if self.source_discontinuity.load(Ordering::Acquire) != expected_generation {
            return false;
        }

        self.sequence.fetch_add(1, Ordering::AcqRel);
        if self.source_discontinuity.load(Ordering::Acquire) != expected_generation {
            self.sequence.fetch_add(1, Ordering::Release);
            return false;
        }

        // This is the worker's final generation check. The test hook inserts
        // the discontinuity at the exact boundary that used to leave four
        // unrelated stores carrying stale evidence.
        #[cfg(test)]
        if VAD_WORKER_FORCE_DISCONTINUITY_AFTER_FINAL_CHECK.with(|flag| flag.replace(false)) {
            self.source_discontinuity.fetch_add(1, Ordering::Release);
        }
        self.probability
            .store(probability.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        self.source_sample_end
            .store(source_sample_end, Ordering::Relaxed);
        self.last_update_us.store(last_update_us, Ordering::Relaxed);
        self.backend_available
            .store(backend_available, Ordering::Relaxed);
        self.result_generation
            .store(expected_generation, Ordering::Relaxed);
        self.sequence.fetch_add(1, Ordering::Release);
        true
    }

    #[inline]
    fn snapshot(&self) -> Option<VadResultSnapshot> {
        for _ in 0..VAD_RESULT_SNAPSHOT_MAX_RETRIES {
            let sequence_before = self.sequence.load(Ordering::Acquire);
            if (sequence_before & 1) != 0 {
                std::hint::spin_loop();
                continue;
            }

            let generation_before = self.source_discontinuity.load(Ordering::Acquire);
            let snapshot = VadResultSnapshot {
                probability: f32::from_bits(self.probability.load(Ordering::Acquire)),
                backend_available: self.backend_available.load(Ordering::Acquire),
                last_update_us: self.last_update_us.load(Ordering::Acquire),
                source_sample_end: self.source_sample_end.load(Ordering::Acquire),
                generation: self.result_generation.load(Ordering::Acquire),
            };
            let generation_after = self.source_discontinuity.load(Ordering::Acquire);
            std::sync::atomic::fence(Ordering::Acquire);
            let sequence_after = self.sequence.load(Ordering::Acquire);
            let generation_final = self.source_discontinuity.load(Ordering::Acquire);
            if sequence_before == sequence_after
                && (sequence_after & 1) == 0
                && generation_before == generation_after
                && generation_after == generation_final
                && snapshot.generation == generation_final
            {
                return Some(snapshot);
            }
            std::hint::spin_loop();
        }
        None
    }
}

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
    let source_sample_end = Arc::clone(&self.vad_source_sample_end);
    let source_discontinuity = Arc::clone(&self.vad_source_discontinuity);
    let result_sequence = Arc::clone(&self.vad_result_sequence);
    let result_generation = Arc::clone(&self.vad_result_generation);
    let gate_rt_control = Arc::clone(&self.gate_rt_control);
    let sample_rate = self.sample_rate;
    let threshold = self
        .gate_rt_control
        .snapshot()
        .unwrap_or_else(GateControlState::new)
        .vad_threshold;

    self.vad_worker_thread = Some(std::thread::spawn(move || {
        let result = VadResultAtomics {
            sequence: result_sequence.as_ref(),
            result_generation: result_generation.as_ref(),
            probability: probability.as_ref(),
            backend_available: available.as_ref(),
            last_update_us: last_update_us.as_ref(),
            source_sample_end: source_sample_end.as_ref(),
            source_discontinuity: source_discontinuity.as_ref(),
        };
        let mut worker_consumer = vad_consumer;
        let mut vad: Option<SileroVAD> = None;
        let mut source_samples_read = 0_u64;
        let mut vad_source_base = 0_u64;
        let mut observed_discontinuity = source_discontinuity.load(Ordering::Acquire);
        let mut local = Vec::with_capacity(VAD_WORKER_MAX_BUFFER_SAMPLES);
        while running.load(Ordering::Acquire) {
            let current_discontinuity = source_discontinuity.load(Ordering::Acquire);
            if current_discontinuity != observed_discontinuity {
                observed_discontinuity = current_discontinuity;
                // A dropped ring block means the next samples are not
                // contiguous with the model's recurrent state. The marker is
                // published before the drop; discard all queued samples
                // before resetting so none can bridge the gap.
                local.clear();
                let mut flushed_samples = 0_u64;
                // Bound the drain so a producer that is still overrunning the
                // queue cannot starve the worker forever. Any samples left in
                // the queue are consumed by the freshly reset model.
                for _ in 0..8 {
                    let available_samples = worker_consumer.len();
                    if available_samples == 0 {
                        break;
                    }
                    let to_read = available_samples.min(VAD_WORKER_MAX_BUFFER_SAMPLES);
                    local.resize(to_read, 0.0);
                    let read = worker_consumer.read(&mut local);
                    local.clear();
                    if read == 0 {
                        break;
                    }
                    flushed_samples = flushed_samples.saturating_add(read as u64);
                    source_samples_read = source_samples_read.saturating_add(read as u64);
                }
                if let Some(worker_vad) = vad.as_mut() {
                    worker_vad.reset();
                }
                vad_source_base = source_samples_read;
                result.invalidate(vad.is_some());
                #[cfg(test)]
                VAD_WORKER_DISCONTINUITY_RESETS.fetch_add(1, Ordering::Release);
                #[cfg(test)]
                VAD_WORKER_DISCONTINUITY_FLUSHED_SAMPLES
                    .fetch_add(flushed_samples, Ordering::Release);
            }
            if vad.is_none() {
                match SileroVAD::new(sample_rate, threshold) {
                    Ok(candidate) => {
                        #[cfg(test)]
                        VAD_WORKER_MODEL_INITS.fetch_add(1, Ordering::Release);
                        vad_source_base = source_samples_read;
                        vad = Some(candidate);
                        result.invalidate(true);
                    }
                    Err(_) => {
                        result.invalidate(false);
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
                source_samples_read = source_samples_read.saturating_add(read as u64);
            }

            // A marker may have been published while the consumer read the
            // queue. Do not run inference on that read; the next iteration
            // drains the queue and resets the model before accepting data.
            if source_discontinuity.load(Ordering::Acquire) != observed_discontinuity {
                local.clear();
                continue;
            }

            if !local.is_empty() {
                // Keep the model's input gain in sync with the lock-free gate
                // controls. This runs on the worker, away from the RT path.
                if let Some(control) = gate_rt_control.snapshot() {
                    if control.pre_gain.is_finite() {
                        let worker_vad = vad.as_mut().expect("VAD backend initialized");
                        worker_vad.set_pre_gain(control.pre_gain);
                        #[cfg(test)]
                        VAD_WORKER_LAST_PRE_GAIN_BITS
                            .store(worker_vad.pre_gain().to_bits(), Ordering::Release);
                    }
                }
                #[cfg(test)]
                if VAD_WORKER_FORCE_INFERENCE_ERROR.swap(false, Ordering::AcqRel) {
                    result.invalidate(false);
                    vad = None;
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    continue;
                }
                let inference = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    vad.as_mut().expect("VAD backend initialized").process_latest(&local)
                }));
                match inference {
                    Ok(Ok(Some(prob))) => {
                        let processed_samples = vad
                            .as_ref()
                        .expect("VAD backend initialized")
                            .processed_input_samples();
                        if !result.publish_vad_inference_result(
                            observed_discontinuity,
                            prob,
                            vad_source_base.saturating_add(processed_samples),
                            now_micros(),
                            true,
                        ) {
                            result.invalidate(vad.is_some());
                            local.clear();
                            continue;
                        }
                    }
                    Ok(Ok(None)) => {}
                    Ok(Err(_)) | Err(_) => {
                        result.invalidate(false);
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
    let result = VadResultAtomics {
        sequence: self.vad_result_sequence.as_ref(),
        result_generation: self.vad_result_generation.as_ref(),
        probability: self.vad_raw_probability.as_ref(),
        backend_available: self.vad_backend_available.as_ref(),
        last_update_us: self.vad_last_update_us.as_ref(),
        source_sample_end: self.vad_source_sample_end.as_ref(),
        source_discontinuity: self.vad_source_discontinuity.as_ref(),
    };
    result.invalidate(false);
}
}
