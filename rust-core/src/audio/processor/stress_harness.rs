//! The production DSP loop on in-memory queues instead of audio devices, and the
//! seeded contention stress test built on it.

use super::control::{CompressorControlState, DeesserControlState, GateControlState};
use super::dsp_loop::{DspLoop, DspLoopSetup, Pass};
use super::shared::Shared;
#[cfg(feature = "vad")]
use super::VAD_WORKER_MAX_BUFFER_SAMPLES;
use super::{AudioProcessor, RT_PROCESS_BUFFER_CAPACITY};
#[cfg(feature = "vad")]
use crate::audio::buffer::AudioRingBuffer;
use crate::audio::buffer::{AudioConsumer, AudioProducer};
use crate::audio::input::TARGET_SAMPLE_RATE;
use crate::dsp::eq::{EqBandConfig, NUM_BANDS};
use crate::dsp::noise_suppressor::{
    ControlledNoiseSuppressionEngine as NoiseSuppressionEngine, NoiseModel,
};
use crate::dsp::rnnoise::RNNOISE_FRAME_SIZE;
use ringbuf::traits::Observer;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How an `OfflineLoop` is wired.
pub(super) struct OfflineLoopConfig {
    pub(super) input_rate: u32,
    pub(super) output_rate: u32,
    /// Replaces the first suppressor engine (a fake, say).
    pub(super) suppressor: Option<NoiseSuppressionEngine>,
}

impl Default for OfflineLoopConfig {
    fn default() -> Self {
        Self {
            input_rate: TARGET_SAMPLE_RATE,
            output_rate: TARGET_SAMPLE_RATE,
            suppressor: None,
        }
    }
}

/// A `DspLoop` running on queues the caller feeds and drains pass by pass. It is
/// set up like `start` sets up the live loop (reset counters, recording tap,
/// engine hand-off queues, output priming), so control calls on the owning
/// `AudioProcessor` reach it through the same shared state as in the app.
pub(super) struct OfflineLoop {
    dsp: DspLoop,
    input: AudioProducer,
    output: AudioConsumer,
    /// The VAD queue; `pass` empties it like a worker that keeps up.
    #[cfg(feature = "vad")]
    vad: AudioConsumer,
}

impl OfflineLoop {
    pub(super) fn new(processor: &AudioProcessor, config: OfflineLoopConfig) -> Self {
        Self::new_with_start_window(processor, config, processor.engine_builder(), || {})
    }

    /// `new`, running `window` where `start` still has its stream unmarked (the
    /// first engine is built and its model published, `running` is not set yet),
    /// and settling a selection that arrived by then with engines from `build`.
    pub(super) fn new_with_start_window(
        processor: &AudioProcessor,
        config: OfflineLoopConfig,
        build: impl Fn(NoiseModel, bool) -> NoiseSuppressionEngine,
        window: impl FnOnce(),
    ) -> Self {
        // The same steps, in the same order, as `start_streams_unguarded`.
        let shared: &Arc<Shared> = &processor.shared;
        let engine_rate = processor.sample_rate;
        let recording_producer = processor.reset_recording_tap();
        shared.reset_for_start();
        let (input_producer, input_consumer, input_dropped) =
            processor.input_ring(config.input_rate);
        let (output_producer, output_consumer) = processor.output_ring(config.output_rate);
        let (input_resampler, output_resampler) = processor
            .build_resamplers(config.input_rate, config.output_rate)
            .expect("the offline loop's rates are supported");

        #[cfg(feature = "vad")]
        let (vad_queue, vad_consumer) = AudioRingBuffer::new(VAD_WORKER_MAX_BUFFER_SAMPLES).split();

        let (suppressor_commands, suppressor_retire) = shared.suppressor.open_queues();
        let (suppressor_control, initial_engine) = processor.initial_suppressor();
        let suppressor = config.suppressor.unwrap_or(initial_engine);

        window();
        processor.mark_running();
        let dsp = DspLoop::new(DspLoopSetup {
            shared: Arc::clone(shared),
            sample_rate: engine_rate,
            input_sample_rate: config.input_rate,
            output_sample_rate: config.output_rate,
            input: input_consumer,
            input_dropped,
            output: output_producer,
            recording: recording_producer,
            #[cfg(feature = "vad")]
            vad_queue,
            input_resampler,
            output_resampler,
            suppressor,
            suppressor_control,
            suppressor_commands,
            suppressor_retire,
        });
        processor.settle_selection_after_start(build);
        Self {
            dsp,
            input: input_producer,
            output: output_consumer,
            #[cfg(feature = "vad")]
            vad: vad_consumer,
        }
    }

    /// Queue device-rate samples; returns how many fit (the rest count as
    /// dropped capture samples, like a full capture queue).
    pub(super) fn push_input(&mut self, samples: &[f32]) -> usize {
        self.input.write(samples)
    }

    #[cfg(all(test, feature = "deepfilter"))]
    pub(super) fn drift_error_ema(&self) -> f32 {
        self.dsp.drift_error_ema()
    }

    pub(super) fn pass(&mut self) -> Pass {
        let pass = self.dsp.pass();
        #[cfg(feature = "vad")]
        {
            let mut discard = [0.0_f32; 4096];
            while self.vad.read(&mut discard) > 0 {}
        }
        pass
    }

    /// Run passes until the loop reports idle.
    pub(super) fn run_until_idle(&mut self) {
        for _ in 0..1000 {
            if matches!(self.pass(), Pass::Idle { .. }) {
                return;
            }
        }
        panic!("the loop never went idle");
    }

    /// Everything queued for the output device since the last call.
    pub(super) fn drain_output(&mut self) -> Vec<f32> {
        let mut out = Vec::new();
        let mut chunk = [0.0_f32; 4096];
        loop {
            let read = self.output.read(&mut chunk);
            if read == 0 {
                return out;
            }
            out.extend_from_slice(&chunk[..read]);
        }
    }
}

/// Results from a deterministic control-thread/DSP-thread contention run.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct ControlDspStressReport {
    pub control_updates: usize,
    pub processed_blocks: usize,
    pub snapshot_rearms: usize,
    pub model_switches: usize,
    pub suppressor_resets: usize,
    pub max_output_abs: f32,
}

#[derive(Clone, Copy)]
struct StressRng(u64);

impl StressRng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }

    fn range_f64(&mut self, minimum: f64, maximum: f64) -> f64 {
        minimum + (maximum - minimum) * self.unit_f64()
    }

    fn boolean(&mut self) -> bool {
        // The LCG low bit only alternates and correlates with the draw cadence.
        (self.next_u64() >> 63) != 0
    }
}

/// One random control call through the `AudioProcessor` methods the app uses.
fn random_control_update(processor: &AudioProcessor, rng: &mut StressRng) -> Result<(), String> {
    match rng.next_u64() % 6 {
        0 => {
            let mut gate = GateControlState::new();
            gate.enabled = rng.boolean();
            gate.threshold_db = rng.range_f64(-70.0, -15.0);
            gate.attack_ms = rng.range_f64(0.1, 80.0);
            gate.release_ms = rng.range_f64(10.0, 800.0);
            processor.apply_gate_settings(&gate);
        }
        1 => {
            let mut bands: Vec<EqBandConfig> = (0..NUM_BANDS)
                .map(EqBandConfig::default_for_index)
                .collect();
            let band = (rng.next_u64() as usize) % NUM_BANDS;
            bands[band].frequency_hz = rng.range_f64(30.0, 20_000.0);
            bands[band].gain_db = rng.range_f64(-6.0, 6.0);
            bands[band].q = rng.range_f64(0.2, 6.0);
            processor.set_eq_enabled(rng.boolean());
            processor
                .apply_eq_settings_v2(bands)
                .map_err(|error| format!("EQ update rejected: {error}"))?;
        }
        2 => {
            let mut compressor = CompressorControlState::new();
            compressor.enabled = rng.boolean();
            compressor.threshold_db = rng.range_f64(-50.0, -6.0);
            compressor.ratio = rng.range_f64(1.0, 12.0);
            compressor.attack_ms = rng.range_f64(0.1, 80.0);
            compressor.base_release_ms = rng.range_f64(20.0, 500.0);
            compressor.makeup_gain_db = rng.range_f64(0.0, 6.0);
            processor.apply_compressor_settings(&compressor);
        }
        3 => {
            let mut deesser = DeesserControlState::new();
            deesser.enabled = rng.boolean();
            deesser.low_cut_hz = rng.range_f64(2_000.0, 10_000.0);
            deesser.high_cut_hz = rng.range_f64(deesser.low_cut_hz + 200.0, 16_000.0);
            deesser.threshold_db = rng.range_f64(-50.0, -8.0);
            deesser.ratio = rng.range_f64(1.0, 12.0);
            processor.apply_deesser_settings(&deesser);
        }
        4 => {
            processor.set_limiter_enabled(rng.boolean());
            processor.set_limiter_ceiling(rng.range_f64(-9.0, -0.1));
            processor.set_limiter_release(rng.range_f64(10.0, 300.0));
        }
        _ => processor.set_rnnoise_enabled(rng.boolean()),
    }
    Ok(())
}

/// Exercise the live control handoff concurrently with the production DSP loop.
///
/// A control thread drives random `AudioProcessor` setters (gate, EQ,
/// compressor, de-esser, limiter, suppressor on/off, engine rebuilds) while this
/// thread feeds `DspLoop` blocks through in-memory queues, so the dirty flags,
/// seqlock snapshots, engine hand-off queues and resets run exactly as in the app.
#[doc(hidden)]
pub fn run_seeded_control_dsp_stress(
    seed: u64,
    iterations: usize,
) -> Result<ControlDspStressReport, String> {
    if iterations == 0 {
        return Err("iterations must be greater than zero".to_string());
    }

    const BLOCK_SIZE: usize = RNNOISE_FRAME_SIZE;
    const MAX_OUTPUT_ABS: f32 = 16.0;
    const _: () = assert!(BLOCK_SIZE <= RT_PROCESS_BUFFER_CAPACITY);

    let processor = AudioProcessor::new();
    let mut offline = OfflineLoop::new(&processor, OfflineLoopConfig::default());
    let shared = Arc::clone(&processor.shared);
    // Force one unstable read so dirty-flag rearming is covered deterministically:
    // a change is pending while a writer is mid-update (odd sequence).
    shared.eq.control.seq.store(1, Ordering::Release);
    shared.eq.dirty.store(true, Ordering::Release);
    let control_done = AtomicBool::new(false);
    let resets_consumed = AtomicUsize::new(0);

    std::thread::scope(|scope| {
        let control = scope.spawn(|| -> Result<(usize, usize), String> {
            let deadline = Instant::now() + Duration::from_secs(2);
            while shared.dsp.lock_contention_count.load(Ordering::Acquire) == 0 {
                if Instant::now() >= deadline {
                    control_done.store(true, Ordering::Release);
                    return Err("DSP thread did not rearm an unstable snapshot".to_string());
                }
                std::thread::yield_now();
            }
            shared.eq.control.seq.fetch_add(1, Ordering::Release);

            let mut rng = StressRng::new(seed);
            let mut queued_engines = 0;
            let result = (|| {
                for index in 0..iterations {
                    random_control_update(&processor, &mut rng)?;
                    if index % 97 == 0 {
                        #[cfg(feature = "deepfilter")]
                        {
                            let model = if queued_engines % 2 == 0 {
                                crate::dsp::noise_suppressor::NoiseModel::DeepFilterNetLL
                            } else {
                                crate::dsp::noise_suppressor::NoiseModel::RNNoise
                            };
                            processor.set_noise_model(model);
                        }
                        // The recovery path's own engine swap; it works without DeepFilter.
                        if processor.rebuild_noise_suppressor() {
                            queued_engines += 1;
                        }
                    }
                    if index % 43 == 0 {
                        // A disable raises a suppressor reset; wait for the loop to take it.
                        processor.set_rnnoise_enabled(false);
                        let deadline = Instant::now() + Duration::from_secs(2);
                        while shared.suppressor.reset_requested.load(Ordering::Acquire) {
                            if Instant::now() >= deadline {
                                return Err(
                                    "a suppressor reset never reached the DSP thread".to_string()
                                );
                            }
                            std::thread::yield_now();
                        }
                        resets_consumed.fetch_add(1, Ordering::Relaxed);
                    }
                    processor.drain_retired_suppressors();
                    std::thread::yield_now();
                }
                Ok(())
            })();
            control_done.store(true, Ordering::Release);
            result.map(|()| (iterations, queued_engines))
        });

        let mut rng = StressRng::new(seed ^ 0xa076_1d64_78bd_642f);
        let mut processed_blocks = 0;
        let mut max_output_abs = 0.0_f32;
        let mut block = [0.0_f32; BLOCK_SIZE];
        let mut sample_index = 0_usize;
        let outcome = (|| -> Result<(), String> {
            while processed_blocks < iterations || !control_done.load(Ordering::Acquire) {
                for sample in block.iter_mut() {
                    let phase = sample_index as f32
                        * (2.0 * std::f32::consts::PI * 173.0 / TARGET_SAMPLE_RATE as f32);
                    let noise = (rng.unit_f64() as f32 - 0.5) * 0.02;
                    *sample = phase.sin() * 0.22 + noise;
                    sample_index += 1;
                }
                if offline.push_input(&block) != block.len() {
                    return Err("the capture queue was full".to_string());
                }
                if offline.pass() == Pass::Busy {
                    processed_blocks += 1;
                }
                for sample in offline.drain_output() {
                    if !sample.is_finite() {
                        return Err("non-finite DSP output".to_string());
                    }
                    max_output_abs = max_output_abs.max(sample.abs());
                    if max_output_abs > MAX_OUTPUT_ABS {
                        return Err(format!("DSP output exceeded bound: {max_output_abs}"));
                    }
                }
                std::thread::yield_now();
            }
            Ok(())
        })();
        // Whatever the DSP side concluded, let the control thread finish first.
        control_done.store(true, Ordering::Release);
        let control_outcome = control
            .join()
            .map_err(|_| "control stress thread panicked".to_string())?;
        // Let the loop take any engine still queued.
        offline.run_until_idle();
        outcome?;
        let (control_updates, queued_engines) = control_outcome?;

        let report = ControlDspStressReport {
            control_updates,
            processed_blocks,
            snapshot_rearms: shared.dsp.lock_contention_count.load(Ordering::Relaxed) as usize,
            model_switches: queued_engines,
            suppressor_resets: resets_consumed.load(Ordering::Relaxed),
            max_output_abs,
        };
        if report.snapshot_rearms == 0 {
            return Err("snapshot dirty flag was never rearmed".to_string());
        }
        let engines_left = shared
            .suppressor
            .pending_tx
            .lock()
            .ok()
            .and_then(|tx| tx.as_ref().map(|tx| !tx.is_empty()))
            .unwrap_or(false);
        if queued_engines > 0 && engines_left {
            return Err("no queued suppressor engine reached the DSP thread".to_string());
        }
        if report.suppressor_resets == 0 {
            return Err("no suppressor reset reached the DSP thread".to_string());
        }
        Ok(report)
    })
}
