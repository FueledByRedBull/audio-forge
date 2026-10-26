#[cfg(feature = "vad")]
use super::control::GateControlState;
#[cfg(feature = "vad")]
use super::lifecycle::Streams;
#[cfg(feature = "vad")]
use super::shared::VadShared;
use super::AudioProcessor;
#[cfg(feature = "vad")]
use super::{VAD_PROBABILITY_STALE_US, VAD_WORKER_MAX_BUFFER_SAMPLES};
#[cfg(feature = "vad")]
use crate::audio::buffer::AudioConsumer;
#[cfg(feature = "vad")]
use crate::audio::clock::now_micros;
#[cfg(feature = "vad")]
use crate::dsp::vad::SileroVAD;
#[cfg(feature = "vad")]
use std::sync::atomic::AtomicBool;
#[cfg(feature = "vad")]
use std::sync::atomic::AtomicU32;
#[cfg(feature = "vad")]
use std::sync::atomic::AtomicU64;
#[cfg(feature = "vad")]
use std::sync::atomic::Ordering;
#[cfg(feature = "vad")]
use std::sync::Arc;
#[cfg(feature = "vad")]
use std::sync::Mutex;
#[cfg(not(windows))]
use thread_priority::{set_current_thread_priority, ThreadPriority};

#[cfg(all(test, feature = "vad"))]
pub(super) static VAD_WORKER_FORCE_INFERENCE_ERROR: AtomicBool = AtomicBool::new(false);
#[cfg(all(test, feature = "vad"))]
pub(super) static VAD_WORKER_LAST_PRE_GAIN_BITS: AtomicU32 = AtomicU32::new(1.0_f32.to_bits());
#[cfg(all(test, feature = "vad"))]
pub(super) static VAD_WORKER_DISCONTINUITY_RESETS: AtomicU64 = AtomicU64::new(0);
#[cfg(all(test, feature = "vad"))]
pub(super) static VAD_WORKER_DISCONTINUITY_FLUSHED_SAMPLES: AtomicU64 = AtomicU64::new(0);
#[cfg(all(test, feature = "vad"))]
pub(super) static VAD_WORKER_MODEL_INITS: AtomicU64 = AtomicU64::new(0);
#[cfg(all(test, feature = "vad"))]
thread_local! {
    pub(super) static VAD_WORKER_FORCE_DISCONTINUITY_AFTER_FINAL_CHECK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(feature = "vad")]
const VAD_RESULT_SNAPSHOT_MAX_RETRIES: usize = 8;

#[cfg(feature = "vad")]
#[derive(Clone, Copy, Debug)]
pub(super) struct VadResultSnapshot {
    pub(super) probability: f32,
    pub(super) backend_available: bool,
    pub(super) last_update_us: u64,
    pub(super) source_sample_end: u64,
    generation: u64,
}

#[cfg(feature = "vad")]
impl VadResultSnapshot {
    #[inline]
    pub(super) fn is_fresh(&self, source_sample_clock: u64, now_us: u64, sample_rate: u32) -> bool {
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
pub(super) struct VadResultAtomics<'a> {
    pub(super) sequence: &'a AtomicU64,
    pub(super) result_generation: &'a AtomicU64,
    pub(super) probability: &'a AtomicU32,
    pub(super) backend_available: &'a AtomicBool,
    pub(super) last_update_us: &'a AtomicU64,
    pub(super) source_sample_end: &'a AtomicU64,
    pub(super) source_discontinuity: &'a AtomicU64,
}

#[cfg(feature = "vad")]
impl VadResultAtomics<'_> {
    // Odd sequence first, then a release fence so no result store can move above it.
    #[inline]
    fn begin_write(&self) {
        self.sequence.fetch_add(1, Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::Release);
    }

    #[inline]
    pub(super) fn invalidate(&self, backend_available: bool) {
        let generation = self.source_discontinuity.load(Ordering::Acquire);
        self.begin_write();
        self.probability.store(0.0_f32.to_bits(), Ordering::Relaxed);
        self.last_update_us.store(0, Ordering::Relaxed);
        self.source_sample_end.store(0, Ordering::Relaxed);
        self.backend_available
            .store(backend_available, Ordering::Relaxed);
        self.result_generation.store(generation, Ordering::Relaxed);
        self.sequence.fetch_add(1, Ordering::Release);
    }

    #[inline]
    pub(super) fn publish_vad_inference_result(
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

        self.begin_write();
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
    pub(super) fn snapshot(&self) -> Option<VadResultSnapshot> {
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

/// Retry delay after consecutive VAD load or inference failures. A corrupt
/// model or an unsupported rate does not fix itself, so rebuilding the ONNX
/// session 20 times a second only burns CPU next to the realtime loop.
#[cfg(feature = "vad")]
#[derive(Default)]
pub(super) struct VadRetryBackoff {
    failures: u32,
}

#[cfg(feature = "vad")]
impl VadRetryBackoff {
    const FIRST_DELAY_MS: u64 = 50;
    const MAX_DELAY_MS: u64 = 5_000;

    pub(super) fn next_delay(&mut self) -> std::time::Duration {
        let delay_ms = Self::FIRST_DELAY_MS
            .saturating_mul(1_u64 << self.failures.min(10))
            .min(Self::MAX_DELAY_MS);
        self.failures = self.failures.saturating_add(1);
        std::time::Duration::from_millis(delay_ms)
    }

    pub(super) fn reset(&mut self) {
        self.failures = 0;
    }
}

/// Sleep for up to `delay`, returning as soon as `running` is cleared, so
/// `stop_vad_worker`'s join never waits out a retry back-off.
#[cfg(feature = "vad")]
pub(super) fn sleep_while_running(running: &AtomicBool, delay: std::time::Duration) {
    const SLICE: std::time::Duration = std::time::Duration::from_millis(20);
    let deadline = std::time::Instant::now() + delay;
    while running.load(Ordering::Acquire) {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        std::thread::sleep(remaining.min(SLICE));
    }
}

/// Publish the last VAD worker failure for diagnostics (`None` clears it).
#[cfg(feature = "vad")]
pub(super) fn publish_vad_worker_error(slot: &Mutex<Option<String>>, error: Option<String>) {
    if let Ok(mut slot) = slot.lock() {
        *slot = error;
    }
}

#[cfg(feature = "vad")]
impl VadShared {
    /// Borrowed view of the fields the VAD worker publishes its results through.
    pub(super) fn result(&self) -> VadResultAtomics<'_> {
        VadResultAtomics {
            sequence: &self.result_sequence,
            result_generation: &self.result_generation,
            probability: &self.raw_probability,
            backend_available: &self.backend_available,
            last_update_us: &self.last_update_us,
            source_sample_end: &self.source_sample_end,
            source_discontinuity: &self.source_discontinuity,
        }
    }
}

impl AudioProcessor {
    #[cfg(feature = "vad")]
    pub(super) fn ensure_vad_worker(&self, streams: &mut Streams, vad_consumer: AudioConsumer) {
        if let Some(handle) = streams.vad_worker_thread.take() {
            if !handle.is_finished() {
                streams.vad_worker_thread = Some(handle);
                return;
            }
            let _ = handle.join();
        }

        self.shared
            .vad
            .worker_running
            .store(true, Ordering::Release);
        let shared = Arc::clone(&self.shared);
        publish_vad_worker_error(&shared.vad.worker_error, None);
        let sample_rate = self.sample_rate;
        let threshold = self
            .shared
            .gate
            .rt_control
            .snapshot()
            .unwrap_or_else(GateControlState::new)
            .vad_threshold;

        streams.vad_worker_thread = Some(std::thread::spawn(move || {
            let result = shared.vad.result();
            let mut worker_consumer = vad_consumer;
            let mut vad: Option<SileroVAD> = None;
            let mut source_samples_read = 0_u64;
            let mut vad_source_base = 0_u64;
            let mut observed_discontinuity =
                shared.vad.source_discontinuity.load(Ordering::Acquire);
            let mut retry = VadRetryBackoff::default();
            let mut local = Vec::with_capacity(VAD_WORKER_MAX_BUFFER_SAMPLES);
            while shared.vad.worker_running.load(Ordering::Acquire) {
                let current_discontinuity = shared.vad.source_discontinuity.load(Ordering::Acquire);
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
                            publish_vad_worker_error(&shared.vad.worker_error, None);
                            result.invalidate(true);
                        }
                        Err(error) => {
                            publish_vad_worker_error(
                                &shared.vad.worker_error,
                                Some(error.to_string()),
                            );
                            result.invalidate(false);
                            sleep_while_running(&shared.vad.worker_running, retry.next_delay());
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
                if shared.vad.source_discontinuity.load(Ordering::Acquire) != observed_discontinuity
                {
                    local.clear();
                    continue;
                }

                if !local.is_empty() {
                    // Keep the model's input gain in sync with the lock-free gate
                    // controls. This runs on the worker, away from the RT path.
                    if let Some(control) = shared.gate.rt_control.snapshot() {
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
                        publish_vad_worker_error(
                            &shared.vad.worker_error,
                            Some("forced test failure".to_string()),
                        );
                        result.invalidate(false);
                        vad = None;
                        sleep_while_running(&shared.vad.worker_running, retry.next_delay());
                        continue;
                    }
                    let inference = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        vad.as_mut()
                            .expect("VAD backend initialized")
                            .process_latest(&local)
                    }));
                    match inference {
                        Ok(Ok(Some(prob))) => {
                            retry.reset();
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
                        failure => {
                            let reason = match failure {
                                Ok(Err(error)) => error.to_string(),
                                _ => "VAD inference panicked".to_string(),
                            };
                            publish_vad_worker_error(&shared.vad.worker_error, Some(reason));
                            result.invalidate(false);
                            vad = None;
                            sleep_while_running(&shared.vad.worker_running, retry.next_delay());
                        }
                    }
                    local.clear();
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            }
        }));
    }

    /// Why the VAD backend is unavailable (missing or corrupt model, unsupported
    /// rate, failed inference), or `None` while it loads and runs.
    #[cfg(feature = "vad")]
    pub fn get_vad_worker_error(&self) -> Option<String> {
        self.shared
            .vad
            .worker_error
            .lock()
            .ok()
            .and_then(|error| error.clone())
    }

    #[cfg(feature = "vad")]
    pub(super) fn stop_vad_worker(&self, streams: &mut Streams) {
        self.shared
            .vad
            .worker_running
            .store(false, Ordering::Release);
        if let Some(handle) = streams.vad_worker_thread.take() {
            let _ = handle.join();
        }
        self.shared.vad.available.store(false, Ordering::Release);
        let result = self.shared.vad.result();
        result.invalidate(false);
    }
}
