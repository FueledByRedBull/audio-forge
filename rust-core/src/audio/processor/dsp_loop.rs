//! The DSP thread: pulls capture audio from the input ring, runs it through
//! gate -> suppressor -> de-esser -> EQ -> compressor -> limiter and queues the
//! result for the output stream. Setup (`DspLoop::new`) allocates everything;
//! the per-block code between the `RT_REGION` markers never does.

use super::control::{
    apply_compressor_control, apply_deesser_control, apply_eq_control, apply_gate_control,
    apply_limiter_control, apply_suppressor_control, apply_true_peak_limiter_control,
    release_ms_to_tenth_ms, AtomicLimiterControlState, CompressorControlState, DeesserControlState,
    EqControlSnapshot, GateControlState, LimiterControlState, SuppressorControlState,
};
use super::diagnostics::{update_backend_status_rt, StreamRecoveryState};
use super::output_writer::{
    OutputWriteContext, OutputWriteCounters, OutputWriteLimits, SuppressorRecoveryFade,
};
use super::raw_recording::record_raw_recording_block;
use super::resampling::{
    clear_resampler_output_samples, duration_samples, has_resampler_output_capacity,
    next_process_idle_sleep_us, signal_input_resampler_failure, smoothing_coeff_for_block,
    smoothing_coeff_for_time_constant, DriftRetimer, LatencyComponents,
};
use super::routing::{
    apply_input_pre_filter, new_input_pre_filter, publish_input_cleanup_bypassed,
    publish_input_cleanup_diagnostics, publish_reported_latency, record_level_extremes,
    sanitize_and_clamp_input_inplace, sanitize_non_finite_inplace, select_processing_path,
    update_decaying_peak_db, update_meter_block_stats, uses_clean_write_path,
    AdaptiveInputCleanupState, InputCleanupMode, InputPreFilterState, MeterState, ProcessingPath,
};
use super::shared::Shared;
#[cfg(feature = "vad")]
use super::VAD_PROBABILITY_STALE_US;
use super::{
    COMPRESSOR_DEFAULT_RELEASE_TENTH_MS, OUTPUT_HARD_BACKLOG_MS, OUTPUT_PRIME_MS,
    OUTPUT_TARGET_HIGH_MS, RESAMPLER_CHUNK_SIZE, RT_INPUT_CHUNK_CAPACITY,
    RT_OUTPUT_SCRATCH_CAPACITY, RT_PROCESS_BUFFER_CAPACITY, RT_RESAMPLE_QUEUE_CAPACITY,
    RT_SUPPRESSOR_OUTPUT_CAPACITY,
};
use crate::audio::buffer::{AudioConsumer, AudioProducer};
use crate::audio::clock::now_micros;
use crate::audio::rt::{store_rt_error, FixedAudioBuffer, FixedAudioRing, RtErrorCode};
use crate::dsp::biquad::Biquad;
use crate::dsp::noise_suppressor::{
    ControlledNoiseSuppressionEngine as NoiseSuppressionEngine, GateControl,
};
#[cfg(feature = "vad")]
use crate::dsp::vad::VadAutoGate;
#[cfg(feature = "vad")]
use crate::dsp::AutoMakeupActivityInput;
use crate::dsp::{
    Compressor, DeEsser, Limiter, NoiseGate, ParametricEQ, TruePeakDetector, TruePeakLimiter,
};
use ringbuf::traits::{Consumer, Observer, Producer};
use ringbuf::{HeapCons, HeapProd};
use rubato::{Resampler, SincFixedIn};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
#[cfg(not(windows))]
use thread_priority::{set_current_thread_priority, ThreadPriority};

#[cfg(feature = "vad")]
#[inline]
pub(super) fn publish_vad_processing_path_discontinuity(
    previous_path: ProcessingPath,
    current_path: ProcessingPath,
    source_discontinuity: &AtomicU64,
) {
    if (previous_path == ProcessingPath::Full) != (current_path == ProcessingPath::Full) {
        source_discontinuity.fetch_add(1, Ordering::Release);
    }
}

/// Read a control snapshot before the loop starts. An unstable snapshot falls
/// back to defaults but re-arms `dirty` (and counts the contention), so the
/// first pass applies the real state instead of leaving the defaults in place.
pub(super) fn startup_control<T>(
    snapshot: Option<T>,
    fallback: fn() -> T,
    dirty: &AtomicBool,
    contention: &AtomicU64,
) -> T {
    rearm_unless_stable(snapshot, dirty, contention).unwrap_or_else(fallback)
}

/// Pass a stable control snapshot through; for an unstable one (`None`)
/// re-arm `dirty` and count the contention, so the next pass retries.
pub(super) fn rearm_unless_stable<T>(
    snapshot: Option<T>,
    dirty: &AtomicBool,
    contention: &AtomicU64,
) -> Option<T> {
    if snapshot.is_none() {
        dirty.store(true, Ordering::Release);
        contention.fetch_add(1, Ordering::Relaxed);
    }
    snapshot
}

/// Consume a pending control change: swap `dirty` off and return a stable
/// snapshot, or re-arm it (see `rearm_unless_stable`). `None` when nothing
/// was pending or the snapshot was unstable.
pub(super) fn take_dirty<T>(
    dirty: &AtomicBool,
    snapshot: impl FnOnce() -> Option<T>,
    contention: &AtomicU64,
) -> Option<T> {
    if dirty.swap(false, Ordering::AcqRel) {
        rearm_unless_stable(snapshot(), dirty, contention)
    } else {
        None
    }
}

/// Consume a pending limiter change once per block, whatever the processing
/// path. Bypass and Raw monitor never run the lookahead limiter, but they still
/// write through the output stage, which reads the ceiling and release
/// published here. An unstable snapshot is re-armed and counted in `contention`.
pub(super) fn apply_pending_limiter_control(
    limiter_dirty: &AtomicBool,
    limiter_rt_control: &AtomicLimiterControlState,
    limiter_rt: &mut Limiter,
    output_limiter_release_ms: &Cell<f32>,
    output_ceiling_linear: &Cell<f32>,
    limiter_lookahead_samples: &AtomicU64,
    contention: &AtomicU64,
) {
    if let Some(control) = take_dirty(limiter_dirty, || limiter_rt_control.snapshot(), contention) {
        apply_limiter_control(limiter_rt, &control);
        output_limiter_release_ms.set(control.release_ms as f32);
        output_ceiling_linear.set(10.0_f32.powf(limiter_rt.ceiling_db() as f32 / 20.0));
        limiter_lookahead_samples.store(limiter_rt.lookahead_samples() as u64, Ordering::Relaxed);
    }
}

#[cfg(test)]
thread_local! {
    static INPUT_DROP_AFTER_READ: Cell<bool> = const { Cell::new(false) };
    static INPUT_DROP_DURING_DRAIN: Cell<bool> = const { Cell::new(false) };
}

struct ContiguousInputRead {
    samples: usize,
    discarded: usize,
    discontinuity: bool,
}

fn read_contiguous_input(
    consumer: &mut AudioConsumer,
    capture_dropped: &AtomicU64,
    observed_dropped: &mut u64,
    buffer: &mut [f32],
    backlog_limits: (usize, usize),
) -> ContiguousInputRead {
    let dropped_before = capture_dropped.load(Ordering::Acquire);
    let capture_gap = dropped_before != *observed_dropped;
    let queued = consumer.len();
    // An overflow has no sample position in the capture queue. Discard only
    // this observed length, never chase a producer that continues writing.
    let mut remaining = if capture_gap {
        queued
    } else if queued > backlog_limits.0 {
        queued.saturating_sub(backlog_limits.1)
    } else {
        0
    };
    let mut discarded = 0;
    while remaining > 0 {
        let count = remaining.min(buffer.len());
        let read = consumer.read(&mut buffer[..count]);
        if read == 0 {
            break;
        }
        remaining -= read;
        discarded += read;
        #[cfg(test)]
        if INPUT_DROP_DURING_DRAIN.with(|flag| flag.replace(false)) {
            capture_dropped.fetch_add(1, Ordering::Release);
        }
    }
    // A newer overflow during the drain remains pending for the next call.
    *observed_dropped = dropped_before;
    if capture_dropped.load(Ordering::Acquire) != dropped_before {
        return ContiguousInputRead {
            samples: 0,
            discarded,
            discontinuity: true,
        };
    }
    let samples = consumer.read(buffer);
    #[cfg(test)]
    if INPUT_DROP_AFTER_READ.with(|flag| flag.replace(false)) {
        capture_dropped.fetch_add(1, Ordering::Release);
    }
    if capture_dropped.load(Ordering::Acquire) != dropped_before {
        return ContiguousInputRead {
            samples: 0,
            discarded: discarded + samples,
            discontinuity: true,
        };
    }
    ContiguousInputRead {
        samples,
        discarded,
        discontinuity: capture_gap || discarded > 0,
    }
}

/// Zero non-finite samples left by the downstream chain. Returns whether any
/// were found, so the caller can count the event and reset the stateful stages
/// (a NaN in a biquad or envelope state would otherwise repeat forever).
pub(super) fn contain_non_finite(buffer: &mut [f32]) -> bool {
    let mut found = false;
    for sample in buffer {
        if !sample.is_finite() {
            *sample = 0.0;
            found = true;
        }
    }
    found
}

/// Notices a suppressor that is fed but produces nothing: it counts the
/// samples handed to it since its last output (or since a reset) and calls it
/// starved once that is 0.4 s of audio. Counting audio instead of wall time
/// makes it independent of the block cadence, and a quiet or disabled input
/// feeds nothing, so it cannot add up to a false alarm.
pub(super) struct SuppressorStarvation {
    fed_samples: usize,
    limit_samples: usize,
}

impl SuppressorStarvation {
    pub(super) fn new(sample_rate: u32) -> Self {
        Self {
            fed_samples: 0,
            limit_samples: duration_samples(sample_rate, 400),
        }
    }

    /// Output arrived, or the suppressor was reset or is not in use.
    pub(super) fn restart(&mut self) {
        self.fed_samples = 0;
    }

    /// The suppressor was handed `samples`.
    pub(super) fn fed(&mut self, samples: usize) {
        self.fed_samples = self.fed_samples.saturating_add(samples);
    }

    pub(super) fn starved(&self) -> bool {
        self.fed_samples >= self.limit_samples
    }
}

/// Advance an exponential moving average by the wall time since its last update.
pub(super) fn advance_time_ema(
    state: &mut f32,
    sample: f32,
    elapsed_ms: f32,
    time_constant_ms: f32,
) {
    let alpha = 1.0 - (-elapsed_ms / time_constant_ms).exp();
    *state += alpha * (sample - *state);
}

/// Keep why the DSP thread died where `last_restart_reason` reports it; the
/// panic message otherwise goes to a stderr nobody reads in the windowed app.
/// `requests_restart` marks the reason as the cause of the pending restart.
pub(super) fn record_dsp_thread_panic(
    payload: &(dyn std::any::Any + Send),
    recovery_state: &Mutex<StreamRecoveryState>,
    requests_restart: bool,
) {
    let message = crate::panic_message(payload);
    if let Ok(mut state) = recovery_state.lock() {
        state.last_reason = Some(format!("DSP thread panicked: {message}"));
        state.reason_pending |= requests_restart;
    }
}

/// Run the DSP loop body. A panic is recorded and answered with a restart
/// request; otherwise the thread would die behind `is_running() == true` until
/// the write-stall watchdog noticed several seconds later.
pub(super) fn run_guarded_dsp_loop(
    body: impl FnOnce(),
    recovery_state: &Mutex<StreamRecoveryState>,
    restart_requested: &AtomicBool,
) {
    if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        record_dsp_thread_panic(payload.as_ref(), recovery_state, true);
        restart_requested.store(true, Ordering::Release);
    }
}

/// Real-time scheduling is an optimisation: without it the DSP thread runs at
/// normal priority, which still works. Report the fallback through diagnostics.
#[cfg_attr(not(windows), allow(dead_code))]
pub(super) fn tolerate_scheduling_failure<G, E>(
    registration: Result<G, E>,
    degraded: &AtomicBool,
) -> Option<G> {
    degraded.store(registration.is_err(), Ordering::Relaxed);
    registration.ok()
}

/// The sinc resampler writes `output_frames_max` frames per call; a smaller
/// scratch would drop every chunk and leave the stream silent.
pub(super) fn ensure_resampler_scratch_capacity(
    resampler: &SincFixedIn<f64>,
    scratch_capacity: usize,
    direction: &str,
) -> Result<(), String> {
    let needed = resampler.output_frames_max();
    if needed > scratch_capacity {
        return Err(format!(
            "The {direction} sample-rate conversion needs {needed} frames per chunk, above the {scratch_capacity} available"
        ));
    }
    Ok(())
}

/// Everything the DSP thread starts with. `start` builds it on the control
/// thread; `DspLoop::new` then allocates the working buffers on the DSP thread.
pub(super) struct DspLoopSetup {
    pub(super) shared: Arc<Shared>,
    /// Engine rate the whole chain runs at.
    pub(super) sample_rate: u32,
    pub(super) input_sample_rate: u32,
    pub(super) output_sample_rate: u32,
    pub(super) input: AudioConsumer,
    /// Dropped-sample counter of the capture ring.
    pub(super) input_dropped: Arc<AtomicU64>,
    pub(super) output: AudioProducer,
    /// Raw recording tap (calibration).
    pub(super) recording: AudioProducer,
    /// Queue feeding the VAD worker.
    #[cfg(feature = "vad")]
    pub(super) vad_queue: AudioProducer,
    pub(super) input_resampler: Option<SincFixedIn<f64>>,
    pub(super) output_resampler: Option<SincFixedIn<f64>>,
    /// First suppressor engine; built on the control thread because a DLL or
    /// model load must not run on the DSP thread.
    pub(super) suppressor: NoiseSuppressionEngine,
    pub(super) suppressor_control: SuppressorControlState,
    /// Engines queued by `set_noise_model` / rebuild.
    pub(super) suppressor_commands: HeapCons<NoiseSuppressionEngine>,
    /// Retired engines go back here so they are dropped off the DSP thread.
    pub(super) suppressor_retire: HeapProd<NoiseSuppressionEngine>,
}

/// What one pass over the input did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Pass {
    /// A block was processed or an input gap was handled; go straight on.
    Busy,
    /// No usable input; sleep this long before the next pass.
    Idle { sleep_us: u64 },
}

/// The DSP thread's state: input stage, processing pipeline and pacing. `new`
/// does every allocation; `run` (or `pass`, in tests) never allocates.
pub(super) struct DspLoop {
    shared: Arc<Shared>,
    input: InputStage,
    pacing: Pacing,
    pipeline: Pipeline,
}

/// Time constant of the block-rate and fill-level metrics.
const TAU_MS: f32 = 200.0;
const LATENCY_UPDATE_INTERVAL: Duration = Duration::from_millis(100);
const NON_FINITE_REBUILD_THRESHOLD: u32 = 3;
const NON_FINITE_REBUILD_WINDOW_MS: u64 = 2000;
const OUTPUT_MAX_CATCHUP_RATIO: f32 = 1.03;
const OUTPUT_MAX_EMERGENCY_CATCHUP_RATIO: f32 = 1.06;

/// Capture side: ring consumer, gap detection and the optional resampler.
struct InputStage {
    consumer: AudioConsumer,
    /// Dropped-sample counter of the capture ring and the value last acted on.
    capture_dropped: Arc<AtomicU64>,
    observed_capture_drops: u64,
    /// Backlog thresholds in input-rate samples: drain above `.0` down to `.1`.
    backlog_limits: (usize, usize),
    /// Raw device-rate read buffer.
    raw: FixedAudioBuffer<f32, RT_INPUT_CHUNK_CAPACITY>,
    /// The engine-rate block handed to the pipeline.
    block: FixedAudioBuffer<f32, RT_PROCESS_BUFFER_CAPACITY>,
    resampler: Option<SincFixedIn<f64>>,
    resampler_out: Option<Vec<Vec<f64>>>,
    resample_queue: FixedAudioRing<f64, RT_RESAMPLE_QUEUE_CAPACITY>,
    resampler_frame: [f64; RESAMPLER_CHUNK_SIZE],
}

/// Fill-level smoothing, latency publication and idle pacing.
struct Pacing {
    last_latency_update: Instant,
    last_input_smoothing: Instant,
    smoothed_input_len: f32,
    consecutive_idle_wakeups: u32,
    input_resampler_delay_samples: u64,
    output_resampler_delay_samples: u64,
    output_sample_rate: u32,
    true_peak_lookahead_samples: u64,
    sample_rate: u32,
}

/// Level meters of the input and output taps.
struct LevelMeters {
    input: MeterState,
    output: MeterState,
    output_short_term_power: f32,
    coeff: f32,
    sample_rate: u32,
}

/// De-esser, both EQ layers, compressor and the sample limiter.
struct DownstreamChain {
    deesser: DeEsser,
    correction_eq: ParametricEQ,
    tone_eq: ParametricEQ,
    compressor: Compressor,
    limiter: Limiter,
    /// Converts the VAD result's source clock to microseconds.
    #[cfg(feature = "vad")]
    sample_rate: u32,
}

/// The suppressor engine with its hand-off queues and health tracking.
struct SuppressorStage {
    engine: NoiseSuppressionEngine,
    commands: HeapCons<NoiseSuppressionEngine>,
    retired: HeapProd<NoiseSuppressionEngine>,
    /// Engine that could not be handed back yet because the retire queue was full.
    deferred_retire: Option<NoiseSuppressionEngine>,
    starvation: SuppressorStarvation,
    recovery_fade: SuppressorRecoveryFade,
    /// Samples popped from the engine for the current block.
    output: FixedAudioBuffer<f32, RT_SUPPRESSOR_OUTPUT_CAPACITY>,
    non_finite_window_started_at: Option<Instant>,
    non_finite_window_count: u32,
}

/// Output side: jitter control, true-peak limiting, resampling and the queue write.
struct OutputStage {
    producer: AudioProducer,
    queue_control_scratch: FixedAudioBuffer<f32, RT_OUTPUT_SCRATCH_CAPACITY>,
    discontinuity_fade_scratch: FixedAudioBuffer<f32, RT_OUTPUT_SCRATCH_CAPACITY>,
    safety_scratch: FixedAudioBuffer<f32, RT_OUTPUT_SCRATCH_CAPACITY>,
    true_peak_detector: TruePeakDetector,
    true_peak_limiter: TruePeakLimiter,
    drift_error_ema: f32,
    drift_retimer: DriftRetimer,
    /// Fade-in still owed after a discontinuity; shared by the pipeline and the writer.
    discontinuity_fade_remaining: Cell<usize>,
    discontinuity_fade_samples: usize,
    /// Fade-in owed after a short write dropped samples.
    output_fade_remaining: usize,
    ceiling_linear: Cell<f32>,
    limiter_release_ms: Cell<f32>,
    applied_limiter_release_ms: f32,
    limits: OutputWriteLimits,
    resampler: Option<SincFixedIn<f64>>,
    resampler_out: Option<Vec<Vec<f64>>>,
    resample_queue: FixedAudioRing<f64, RT_RESAMPLE_QUEUE_CAPACITY>,
    resampler_frame: [f64; RESAMPLER_CHUNK_SIZE],
    resampled_scratch: FixedAudioBuffer<f32, RT_OUTPUT_SCRATCH_CAPACITY>,
}

/// Everything that processes one block, whichever path it takes.
struct Pipeline {
    sample_rate: u32,
    /// Alpha of the per-block metric EMAs (10 ms nominal block, `TAU_MS`).
    metric_alpha: f32,
    pre_filter_state: InputPreFilterState,
    pre_filter: Biquad,
    adaptive_cleanup: AdaptiveInputCleanupState,
    recording: AudioProducer,
    gate: NoiseGate,
    gate_controls: FixedAudioBuffer<GateControl, RT_PROCESS_BUFFER_CAPACITY>,
    suppressor: SuppressorStage,
    chain: DownstreamChain,
    meters: LevelMeters,
    output: OutputStage,
    limiter_lookahead_samples: AtomicU64,
    previous_path: ProcessingPath,
    #[cfg(feature = "vad")]
    vad_queue: AudioProducer,
    #[cfg(feature = "vad")]
    vad_source_sample_clock: u64,
}

impl DspLoop {
    /// Allocate every buffer and bring every stage to its configured state.
    /// This is the setup half; nothing after it allocates.
    pub(super) fn new(setup: DspLoopSetup) -> Self {
        let DspLoopSetup {
            shared,
            sample_rate,
            input_sample_rate,
            output_sample_rate,
            input,
            input_dropped,
            output,
            recording,
            #[cfg(feature = "vad")]
            vad_queue,
            input_resampler,
            output_resampler,
            suppressor,
            suppressor_control,
            suppressor_commands,
            suppressor_retire,
        } = setup;
        let sample_rate_hz = f64::from(sample_rate);
        let input_resampler_delay_samples = input_resampler
            .as_ref()
            .map(|resampler| resampler.output_delay() as u64)
            .unwrap_or(0);
        let output_resampler_delay_samples = output_resampler
            .as_ref()
            .map(|resampler| resampler.output_delay() as u64)
            .unwrap_or(0);

        let mut gate = NoiseGate::new(-40.0, 10.0, 100.0, sample_rate_hz);
        let gate_snapshot = startup_control(
            shared.gate.rt_control.snapshot(),
            GateControlState::new,
            &shared.gate.dirty,
            &shared.dsp.lock_contention_count,
        );
        #[cfg(feature = "vad")]
        {
            let vad_auto_gate =
                VadAutoGate::without_backend(sample_rate, gate_snapshot.vad_threshold);
            gate.set_vad_auto_gate(Some(vad_auto_gate));
        }
        let mut correction_eq = ParametricEQ::new(sample_rate_hz);
        let mut tone_eq = ParametricEQ::new(sample_rate_hz);
        let mut compressor = Compressor::default_voice(sample_rate_hz);
        let mut deesser = DeEsser::new(sample_rate_hz);
        let mut limiter = Limiter::default_settings(sample_rate_hz);
        let mut suppressor_engine = suppressor;
        let output_ceiling_linear = Cell::new(10.0_f32.powf(limiter.ceiling_db() as f32 / 20.0));
        let limiter_lookahead_samples = AtomicU64::new(limiter.lookahead_samples() as u64);
        apply_gate_control(&mut gate, &gate_snapshot);
        #[cfg(feature = "vad")]
        {
            shared
                .gate
                .noise_floor_db
                .store(gate.noise_floor().to_bits(), Ordering::Relaxed);
            shared.vad.available.store(false, Ordering::Relaxed);
        }
        apply_suppressor_control(&mut suppressor_engine, &suppressor_control);
        update_backend_status_rt(
            &shared.suppressor.backend_available,
            &shared.suppressor.backend_failed,
            shared.dsp.rt_error_code.as_ref(),
            &suppressor_engine,
        );
        let eq_snapshot = startup_control(
            shared.eq.control.snapshot(),
            EqControlSnapshot::new,
            &shared.eq.dirty,
            &shared.dsp.lock_contention_count,
        );
        apply_eq_control(&mut correction_eq, &mut tone_eq, &eq_snapshot);
        let compressor_snapshot = startup_control(
            shared.compressor.rt_control.snapshot(),
            CompressorControlState::new,
            &shared.compressor.dirty,
            &shared.dsp.lock_contention_count,
        );
        apply_compressor_control(&mut compressor, &compressor_snapshot);
        // Start in the configured state; enable ramps are for live changes.
        compressor.finish_enable_transition();
        correction_eq.reset();
        tone_eq.reset();
        let deesser_snapshot = startup_control(
            shared.deesser.rt_control.snapshot(),
            DeesserControlState::new,
            &shared.deesser.dirty,
            &shared.dsp.lock_contention_count,
        );
        apply_deesser_control(&mut deesser, &deesser_snapshot);
        deesser.finish_enable_transition();
        let limiter_snapshot = startup_control(
            shared.limiter.rt_control.snapshot(),
            LimiterControlState::new,
            &shared.limiter.dirty,
            &shared.dsp.lock_contention_count,
        );
        apply_limiter_control(&mut limiter, &limiter_snapshot);
        output_ceiling_linear.set(10.0_f32.powf(limiter.ceiling_db() as f32 / 20.0));

        // Keep the existing priority policy outside Windows.
        #[cfg(not(windows))]
        if let Err(_e) = set_current_thread_priority(ThreadPriority::Max) {
            processor_debug_log!("Warning: Could not set audio thread priority: {:?}", _e);
        }

        // Jitter-buffer write control: keep output queue in a healthy range.
        let output_target_low_samples = duration_samples(output_sample_rate, OUTPUT_PRIME_MS);
        let output_target_high_samples =
            duration_samples(output_sample_rate, OUTPUT_TARGET_HIGH_MS);
        let output_target_center_samples =
            (output_target_low_samples + output_target_high_samples).div_ceil(2);
        let output_hard_backlog_samples =
            duration_samples(output_sample_rate, OUTPUT_HARD_BACKLOG_MS);
        let discontinuity_fade_samples = duration_samples(output_sample_rate, 6).max(1);
        let mut true_peak_limiter = TruePeakLimiter::default_settings(output_sample_rate as f32);
        apply_true_peak_limiter_control(&mut true_peak_limiter, &limiter_snapshot);
        let true_peak_lookahead_samples = true_peak_limiter.lookahead_samples() as u64;

        // Time-based EMA smoothing for metrics (200ms time constant). The
        // per-block metrics assume one block per 10 ms (480 samples @ 48kHz).
        let metric_alpha = 1.0 - (-10.0_f32 / TAU_MS).exp();

        let previous_path = select_processing_path(
            shared.mode.raw_monitor.load(Ordering::Acquire),
            shared.mode.bypass.load(Ordering::SeqCst),
        );
        // Raw monitor and Bypass leave the suppressor latency at 0; the Full
        // path would only store it after its first block, which the first
        // latency tick can precede.
        if previous_path == ProcessingPath::Full
            && shared.suppressor.enabled.load(Ordering::Acquire)
        {
            shared.suppressor.latency_samples.store(
                suppressor_engine.latency_samples() as u32,
                Ordering::Relaxed,
            );
        }

        let now = Instant::now();
        let input_stage = InputStage {
            consumer: input,
            capture_dropped: input_dropped,
            observed_capture_drops: 0,
            backlog_limits: (
                duration_samples(input_sample_rate, 250),
                duration_samples(input_sample_rate, 100),
            ),
            raw: FixedAudioBuffer::new(),
            block: FixedAudioBuffer::new(),
            resampler_out: input_resampler
                .as_ref()
                .map(|r| r.output_buffer_allocate(true)),
            resampler: input_resampler,
            resample_queue: FixedAudioRing::new(),
            resampler_frame: [0.0; RESAMPLER_CHUNK_SIZE],
        };
        let output_stage = OutputStage {
            producer: output,
            queue_control_scratch: FixedAudioBuffer::new(),
            discontinuity_fade_scratch: FixedAudioBuffer::new(),
            safety_scratch: FixedAudioBuffer::new(),
            true_peak_detector: TruePeakDetector::new(),
            true_peak_limiter,
            drift_error_ema: 0.0,
            drift_retimer: DriftRetimer::default(),
            discontinuity_fade_remaining: Cell::new(0),
            discontinuity_fade_samples,
            output_fade_remaining: 0,
            ceiling_linear: output_ceiling_linear,
            limiter_release_ms: Cell::new(limiter_snapshot.release_ms as f32),
            applied_limiter_release_ms: limiter_snapshot.release_ms as f32,
            limits: OutputWriteLimits {
                output_target_center_samples,
                output_hard_backlog_samples,
                discontinuity_fade_samples,
                max_catchup_ratio: OUTPUT_MAX_CATCHUP_RATIO,
                max_emergency_catchup_ratio: OUTPUT_MAX_EMERGENCY_CATCHUP_RATIO,
            },
            resampler_out: output_resampler
                .as_ref()
                .map(|r| r.output_buffer_allocate(true)),
            resampler: output_resampler,
            resample_queue: FixedAudioRing::new(),
            resampler_frame: [0.0; RESAMPLER_CHUNK_SIZE],
            resampled_scratch: FixedAudioBuffer::new(),
        };
        let pipeline = Pipeline {
            sample_rate,
            metric_alpha,
            pre_filter_state: InputPreFilterState::default(),
            // Pre-filter at 80Hz to remove rumble before gate/suppressor stages.
            pre_filter: new_input_pre_filter(sample_rate_hz),
            adaptive_cleanup: AdaptiveInputCleanupState::new(sample_rate as f32),
            recording,
            gate,
            gate_controls: FixedAudioBuffer::new(),
            suppressor: SuppressorStage {
                engine: suppressor_engine,
                commands: suppressor_commands,
                retired: suppressor_retire,
                deferred_retire: None,
                starvation: SuppressorStarvation::new(sample_rate),
                recovery_fade: SuppressorRecoveryFade::new(duration_samples(sample_rate, 6)),
                output: FixedAudioBuffer::new(),
                non_finite_window_started_at: None,
                non_finite_window_count: 0,
            },
            chain: DownstreamChain {
                deesser,
                correction_eq,
                tone_eq,
                compressor,
                limiter,
                #[cfg(feature = "vad")]
                sample_rate,
            },
            // Metering state (IIR smoothing for RMS)
            meters: LevelMeters {
                input: MeterState::default(),
                output: MeterState::default(),
                output_short_term_power: 0.0,
                coeff: smoothing_coeff_for_time_constant(sample_rate as f32, 100.0),
                sample_rate,
            },
            output: output_stage,
            limiter_lookahead_samples,
            previous_path,
            #[cfg(feature = "vad")]
            vad_queue,
            #[cfg(feature = "vad")]
            vad_source_sample_clock: 0,
        };
        Self {
            shared,
            input: input_stage,
            pacing: Pacing {
                last_latency_update: now,
                last_input_smoothing: now,
                smoothed_input_len: 0.0,
                consecutive_idle_wakeups: 0,
                input_resampler_delay_samples,
                output_resampler_delay_samples,
                output_sample_rate,
                true_peak_lookahead_samples,
                sample_rate,
            },
            pipeline,
        }
    }

    /// Run until `running` is cleared. A panic is caught and reported
    /// (recovery state plus a restart request) instead of leaving a dead
    /// thread behind a running processor.
    pub(super) fn run(mut self) {
        let shared = Arc::clone(&self.shared);
        // Run the whole loop with denormals flushed to zero so tiny values
        // cannot stall the CPU.
        // SAFETY: `no_denormals` sets the FTZ/DAZ bits of this thread's FP
        // control register and restores them when the closure returns or
        // unwinds. Its own documentation calls this technically undefined
        // behaviour, because the compiler may assume the default FP
        // environment. The risk is confined to this thread; the loop does
        // not rely on exact results of denormal arithmetic.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        let dsp_loop = || unsafe { no_denormals::no_denormals(|| self.run_passes()) };
        run_guarded_dsp_loop(
            dsp_loop,
            &shared.stream.recovery_state,
            &shared.stream.restart_requested,
        );
    }
}

// RT_REGION_START: dsp_processing_loop
impl DspLoop {
    fn run_passes(&mut self) {
        while self.shared.stream.running.load(Ordering::SeqCst) {
            if let Pass::Idle { sleep_us } = self.pass() {
                std::thread::sleep(Duration::from_micros(sleep_us));
            }
        }
    }

    /// One pass of the loop: housekeeping, then at most one block of input.
    pub(super) fn pass(&mut self) -> Pass {
        let shared = &*self.shared;
        // Publish latency on every pass, before any path branch or early
        // return, so Raw monitor, Bypass and idle wakeups all keep it current.
        let pass_now = Instant::now();
        self.pacing.publish_latency(
            shared,
            pass_now,
            self.pipeline
                .limiter_lookahead_samples
                .load(Ordering::Relaxed),
        );

        // Installs engines queued by set_noise_model / rebuild and applies a
        // pending suppressor control change. It runs at the top of every pass,
        // whether or not the suppressor is in use and whether or not input
        // arrived, so the bounded command queue cannot fill up behind a
        // disabled suppressor and make later switches fail.
        // `suppressor_in_use` is true only when this pass feeds the
        // suppressor; otherwise the swap is silent (no fade), as the output
        // carries no suppressor audio.
        let suppressor_in_use = shared.suppressor.enabled.load(Ordering::Acquire)
            && select_processing_path(
                shared.mode.raw_monitor.load(Ordering::Acquire),
                shared.mode.bypass.load(Ordering::SeqCst),
            ) == ProcessingPath::Full;
        self.pipeline
            .suppressor
            .install_pending(shared, suppressor_in_use, &self.pipeline.output);

        // Record input buffer fill level (samples waiting to be processed)
        self.pacing
            .record_input_fill(shared, self.input.consumer.len() as u32, pass_now);

        let input_read = self.input.read_block();
        if input_read.discontinuity {
            shared
                .input
                .backlog_recovery_count
                .fetch_add(1, Ordering::Relaxed);
            shared
                .input
                .backlog_dropped_samples
                .fetch_add(input_read.discarded as u64, Ordering::Relaxed);
            store_rt_error(
                shared.dsp.rt_error_code.as_ref(),
                RtErrorCode::InputBacklogDropped,
            );
            reset_input_continuity(&mut self.input, &mut self.pipeline, shared);
        }
        let n_raw = input_read.samples;
        if n_raw == 0 {
            return self.pacing.idle(shared);
        }

        if self.input.stage_block(shared, n_raw) == 0 {
            return self.pacing.idle(shared);
        }
        self.pacing.consecutive_idle_wakeups = 0;

        if !self
            .pipeline
            .process_block(shared, self.input.block.as_mut_slice())
        {
            reset_input_continuity(&mut self.input, &mut self.pipeline, shared);
        }
        Pass::Busy
    }
}

/// An input discontinuity (capture gap, backlog drop, dropped chunk) leaves
/// the resampler, gate and suppressor out of step with the audio: clear them.
fn reset_input_continuity(input: &mut InputStage, pipeline: &mut Pipeline, shared: &Shared) {
    input.reset();
    pipeline.reset_after_input_gap(shared);
}

impl Pacing {
    fn publish_latency(&mut self, shared: &Shared, now: Instant, limiter_lookahead_samples: u64) {
        if now.duration_since(self.last_latency_update) < LATENCY_UPDATE_INTERVAL {
            return;
        }
        self.last_latency_update = now;
        let suppressor_latency_samples = if shared.suppressor.enabled.load(Ordering::Acquire) {
            shared.suppressor.latency_samples.load(Ordering::Relaxed) as u64
        } else {
            0
        };
        publish_reported_latency(
            &shared.dsp.latency_us,
            LatencyComponents {
                input_resampler_delay_samples: self.input_resampler_delay_samples,
                output_buffer_samples: shared.output.buffer_len.load(Ordering::Relaxed) as u64,
                output_sample_rate: self.output_sample_rate,
                output_resampler_delay_samples: self.output_resampler_delay_samples,
                suppressor_latency_samples,
                limiter_lookahead_samples,
                true_peak_lookahead_samples: self.true_peak_lookahead_samples,
                processing_sample_rate: self.sample_rate,
            },
            select_processing_path(
                shared.mode.raw_monitor.load(Ordering::Acquire),
                shared.mode.bypass.load(Ordering::SeqCst),
            ),
            shared.dsp.latency_compensation_us.load(Ordering::Relaxed),
        );
    }

    /// The input fill level advances by measured wall time on every loop pass.
    fn record_input_fill(&mut self, shared: &Shared, raw_input_len: u32, now: Instant) {
        shared
            .input
            .buffer_len
            .store(raw_input_len, Ordering::Relaxed);
        advance_time_ema(
            &mut self.smoothed_input_len,
            raw_input_len as f32,
            now.duration_since(self.last_input_smoothing).as_secs_f32() * 1000.0,
            TAU_MS,
        );
        self.last_input_smoothing = now;
        shared
            .input
            .smoothed_buffer_len
            .store(self.smoothed_input_len.round() as u32, Ordering::Relaxed);
    }

    /// No usable input: note the wakeup and choose how long to sleep.
    fn idle(&mut self, shared: &Shared) -> Pass {
        self.consecutive_idle_wakeups = self.consecutive_idle_wakeups.saturating_add(1);
        let last_input_callback = shared.input.callback_time_us.load(Ordering::Relaxed);
        let input_callback_age_us = if last_input_callback == 0 {
            u64::MAX
        } else {
            now_micros().saturating_sub(last_input_callback)
        };
        let idle_sleep_us =
            next_process_idle_sleep_us(self.consecutive_idle_wakeups, input_callback_age_us);
        shared.dsp.idle_wakeup_count.fetch_add(1, Ordering::Relaxed);
        shared
            .dsp
            .idle_sleep_us
            .store(idle_sleep_us, Ordering::Relaxed);
        Pass::Idle {
            sleep_us: idle_sleep_us,
        }
    }
}

impl InputStage {
    fn read_block(&mut self) -> ContiguousInputRead {
        read_contiguous_input(
            &mut self.consumer,
            self.capture_dropped.as_ref(),
            &mut self.observed_capture_drops,
            self.raw.as_mut_capacity_slice(),
            self.backlog_limits,
        )
    }

    /// Turn the `n_raw` samples just read into an engine-rate block, resampling
    /// when the device rate differs. Returns the block length.
    fn stage_block(&mut self, shared: &Shared, n_raw: usize) -> usize {
        let Some(resampler) = self.resampler.as_mut() else {
            self.block.clear();
            let written = self
                .block
                .extend_from_slice(&self.raw.as_mut_capacity_slice()[..n_raw]);
            if written < n_raw {
                shared
                    .dsp
                    .rt_buffer_overflow_count
                    .fetch_add(1, Ordering::Relaxed);
                store_rt_error(
                    shared.dsp.rt_error_code.as_ref(),
                    RtErrorCode::FixedBufferOverflow,
                );
            }
            return written;
        };
        self.block.clear();
        for &sample in self.raw.as_mut_capacity_slice()[..n_raw].iter() {
            if !self.resample_queue.push(sample as f64) {
                shared
                    .dsp
                    .rt_buffer_overflow_count
                    .fetch_add(1, Ordering::Relaxed);
                store_rt_error(
                    shared.dsp.rt_error_code.as_ref(),
                    RtErrorCode::FixedBufferOverflow,
                );
                break;
            }
        }

        let input_frames_needed = resampler.input_frames_next();
        while self.resample_queue.len() >= input_frames_needed
            && input_frames_needed <= self.resampler_frame.len()
        {
            let Some(outbuf) = self.resampler_out.as_mut() else {
                break;
            };
            if !has_resampler_output_capacity(&self.block, outbuf) {
                break;
            }
            let read = self
                .resample_queue
                .pop_into(&mut self.resampler_frame[..input_frames_needed]);
            if read != input_frames_needed {
                break;
            }
            let in_slices = [&self.resampler_frame[..input_frames_needed]];
            match resampler.process_into_buffer(&in_slices, outbuf, None) {
                Ok((_nbr_in, nbr_out)) => {
                    let channel_out = &outbuf[0];
                    for &sample in channel_out.iter().take(nbr_out) {
                        if !self.block.push(sample as f32) {
                            shared
                                .dsp
                                .rt_buffer_overflow_count
                                .fetch_add(1, Ordering::Relaxed);
                            store_rt_error(
                                shared.dsp.rt_error_code.as_ref(),
                                RtErrorCode::FixedBufferOverflow,
                            );
                            break;
                        }
                    }
                }
                Err(_) => {
                    signal_input_resampler_failure(
                        shared.dsp.rt_error_code.as_ref(),
                        shared.stream.restart_requested.as_ref(),
                    );
                    self.resample_queue.clear();
                    self.block.clear();
                    break;
                }
            }
        }
        self.block.len()
    }

    /// Drop everything buffered ahead of the resampler.
    fn reset(&mut self) {
        self.resample_queue.clear();
        if let Some(resampler) = self.resampler.as_mut() {
            resampler.reset();
        }
        if let Some(outbuf) = self.resampler_out.as_mut() {
            clear_resampler_output_samples(outbuf);
        }
    }
}

impl Pipeline {
    /// Process one engine-rate block in place and queue it for output. Returns
    /// false when the block overran the suppressor's input buffer; the caller
    /// then resets the input continuity.
    fn process_block(&mut self, shared: &Shared, buffer: &mut [f32]) -> bool {
        // Closest tap to the converter in every processing path: after channel
        // mix-down and any resampling, before sanitizing, the pre-filter and
        // cleanup.
        record_level_extremes(
            buffer,
            &shared.input.peak_hold,
            Some(&shared.input.full_scale_count),
        );

        // Start DSP timing
        let dsp_start = Instant::now();
        let processing_path = select_processing_path(
            shared.mode.raw_monitor.load(Ordering::Acquire),
            shared.mode.bypass.load(Ordering::SeqCst),
        );
        if processing_path != self.previous_path {
            self.switch_path(shared, processing_path);
        }

        apply_pending_limiter_control(
            &shared.limiter.dirty,
            &shared.limiter.rt_control,
            &mut self.chain.limiter,
            &self.output.limiter_release_ms,
            &self.output.ceiling_linear,
            &self.limiter_lookahead_samples,
            &shared.dsp.lock_contention_count,
        );

        if processing_path == ProcessingPath::RawMonitor {
            self.process_raw_monitor(shared, buffer, dsp_start);
            return true;
        }

        sanitize_and_clamp_input_inplace(
            buffer,
            &shared.input.clip_event_count,
            &shared.input.clip_peak_db,
        );
        let cleanup_mode =
            InputCleanupMode::from_u8(shared.input.cleanup_mode.load(Ordering::Acquire))
                .unwrap_or_default();
        let recording_active_now = shared.recording.active.load(Ordering::Relaxed);
        let raw_tap_before_cleanup = shared.recording.before_cleanup.load(Ordering::Relaxed);
        let adaptive_cleanup_enabled = processing_path == ProcessingPath::Full
            && cleanup_mode.is_enabled()
            && (!recording_active_now || raw_tap_before_cleanup);
        self.adaptive_cleanup.set_mode(cleanup_mode);
        if adaptive_cleanup_enabled {
            self.adaptive_cleanup.analyze_input(buffer);
        } else {
            self.adaptive_cleanup.reset_dynamic_state();
            publish_input_cleanup_bypassed(
                &shared.input.cleanup_hum_detected,
                &shared.input.cleanup_rumble_detected,
                &shared.input.cleanup_high_pass_hz,
            );
        }
        // An explicit pre-cleanup audition captures the sanitized microphone
        // samples before either the fixed DC/80 Hz pre-filter or adaptive
        // cleanup.
        if recording_active_now && raw_tap_before_cleanup {
            self.record_raw_tap(shared, buffer);
        }
        apply_input_pre_filter(
            buffer,
            &mut self.pre_filter_state,
            &mut self.pre_filter,
            !adaptive_cleanup_enabled,
        );

        // === RAW AUDIO RECORDING TAP (for calibration) ===
        // Capture audio AFTER pre-filter, BEFORE noise gate
        // This is the raw microphone response needed for EQ analysis
        if recording_active_now && !raw_tap_before_cleanup {
            self.record_raw_tap(shared, buffer);
        }
        // === END RECORDING TAP ===

        if adaptive_cleanup_enabled {
            self.adaptive_cleanup.process_block(buffer);
            publish_input_cleanup_diagnostics(
                &self.adaptive_cleanup,
                &shared.input.cleanup_hum_detected,
                &shared.input.cleanup_rumble_detected,
                &shared.input.cleanup_high_pass_hz,
            );
        }

        // Measure INPUT levels after fixed pre-filter and optional adaptive cleanup.
        self.meters.measure_input(shared, buffer);

        if processing_path == ProcessingPath::Bypass {
            self.process_bypass(shared, buffer, dsp_start);
            true
        } else {
            self.process_full(shared, buffer, dsp_start)
        }
    }

    fn record_raw_tap(&mut self, shared: &Shared, buffer: &[f32]) {
        record_raw_recording_block(
            buffer,
            self.sample_rate,
            &mut self.recording,
            &shared.recording.pos,
            &shared.recording.target,
            &shared.recording.level_db,
        );
    }

    /// Publish how long the block took, raw and smoothed.
    fn record_dsp_time(&self, shared: &Shared, dsp_start: Instant) {
        let raw_dsp_us = dsp_start.elapsed().as_micros() as u64;
        let prev_smoothed = shared.dsp.time_smoothed_us.load(Ordering::Relaxed);
        let smoothed = smooth_dsp_time(self.metric_alpha, raw_dsp_us, prev_smoothed);
        shared.dsp.time_us.store(raw_dsp_us, Ordering::Relaxed);
        shared
            .dsp
            .time_smoothed_us
            .store(smoothed, Ordering::Relaxed);
    }

    /// The processing path changed (Raw monitor, Bypass or Full): every stage
    /// and meter restarts so nothing from the old path leaks into the new one.
    fn switch_path(&mut self, shared: &Shared, processing_path: ProcessingPath) {
        #[cfg(feature = "vad")]
        publish_vad_processing_path_discontinuity(
            self.previous_path,
            processing_path,
            &shared.vad.source_discontinuity,
        );
        #[cfg(feature = "vad")]
        shared.vad.available.store(false, Ordering::Release);
        self.pre_filter_state = InputPreFilterState::default();
        self.adaptive_cleanup.reset_dynamic_state();
        publish_input_cleanup_bypassed(
            &shared.input.cleanup_hum_detected,
            &shared.input.cleanup_rumble_detected,
            &shared.input.cleanup_high_pass_hz,
        );
        self.meters.restart();
        shared
            .gate
            .gain_meter
            .store(1.0_f32.to_bits(), Ordering::Relaxed);
        shared
            .gate
            .auto_relax_active
            .store(false, Ordering::Relaxed);
        shared
            .compressor
            .gain_reduction
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        shared
            .deesser
            .gain_reduction
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        shared
            .deesser
            .detector_confidence
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        shared
            .compressor
            .current_release_ms
            .store(COMPRESSOR_DEFAULT_RELEASE_TENTH_MS, Ordering::Relaxed);
        shared
            .compressor
            .current_lufs
            .store((-100.0_f64).to_bits(), Ordering::Relaxed);
        shared
            .compressor
            .current_makeup_gain
            .store(0.0_f64.to_bits(), Ordering::Relaxed);
        shared
            .limiter
            .gain_reduction_db
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        shared
            .limiter
            .peak_gain_reduction_db
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        shared
            .limiter
            .gain_reduction_history_db
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        shared
            .output
            .true_peak_gain_reduction_history_db
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        shared.suppressor.buffer_len.store(0, Ordering::Relaxed);
        shared
            .suppressor
            .latency_samples
            .store(0, Ordering::Relaxed);
        shared
            .suppressor
            .successful_inference_frames
            .store(0, Ordering::Relaxed);
        shared
            .suppressor
            .smoothed_buffer_len
            .store(0, Ordering::Relaxed);
        self.output
            .discontinuity_fade_remaining
            .set(self.output.discontinuity_fade_samples);

        self.gate.reset();
        if let Some(control) = rearm_unless_stable(
            shared.gate.rt_control.snapshot(),
            &shared.gate.dirty,
            &shared.dsp.lock_contention_count,
        ) {
            apply_gate_control(&mut self.gate, &control);
        }
        self.chain.correction_eq.reset();
        self.chain.tone_eq.reset();
        self.chain.compressor.reset();
        self.chain.deesser.reset();
        self.chain.limiter.reset();
        self.suppressor.engine.soft_reset();
        self.suppressor.starvation.restart();
        if processing_path == ProcessingPath::Full
            && shared.suppressor.enabled.load(Ordering::Acquire)
        {
            self.suppressor.restart_fade(&self.output);
        } else {
            self.suppressor.recovery_fade.restart(0);
        }
        update_backend_status_rt(
            &shared.suppressor.backend_available,
            &shared.suppressor.backend_failed,
            shared.dsp.rt_error_code.as_ref(),
            &self.suppressor.engine,
        );
        self.previous_path = processing_path;
    }

    /// Raw monitor: the sanitized input goes straight to the output writer.
    fn process_raw_monitor(&mut self, shared: &Shared, buffer: &mut [f32], dsp_start: Instant) {
        sanitize_non_finite_inplace(buffer);
        self.adaptive_cleanup.reset_dynamic_state();
        publish_input_cleanup_bypassed(
            &shared.input.cleanup_hum_detected,
            &shared.input.cleanup_rumble_detected,
            &shared.input.cleanup_high_pass_hz,
        );

        self.meters.measure_input(shared, buffer);

        if shared.recording.active.load(Ordering::Relaxed) {
            self.record_raw_tap(shared, buffer);
        }

        self.meters.measure_output(shared, buffer);
        shared
            .compressor
            .gain_reduction
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        shared
            .deesser
            .gain_reduction
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        shared
            .deesser
            .detector_confidence
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        shared
            .gate
            .gain_meter
            .store(1.0_f32.to_bits(), Ordering::Relaxed);
        shared
            .gate
            .auto_relax_active
            .store(false, Ordering::Relaxed);
        shared
            .compressor
            .current_release_ms
            .store(COMPRESSOR_DEFAULT_RELEASE_TENTH_MS, Ordering::Relaxed);
        shared
            .compressor
            .current_lufs
            .store((-100.0_f64).to_bits(), Ordering::Relaxed);
        shared
            .compressor
            .current_makeup_gain
            .store(0.0_f64.to_bits(), Ordering::Relaxed);
        shared
            .limiter
            .gain_reduction_db
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        shared
            .limiter
            .peak_gain_reduction_db
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        update_decaying_peak_db(0.0, &shared.limiter.gain_reduction_history_db, 0.15);
        update_decaying_peak_db(
            0.0,
            &shared.output.true_peak_gain_reduction_history_db,
            0.15,
        );
        shared.suppressor.buffer_len.store(0, Ordering::Relaxed);
        shared
            .suppressor
            .latency_samples
            .store(0, Ordering::Relaxed);
        shared
            .suppressor
            .successful_inference_frames
            .store(0, Ordering::Relaxed);
        shared
            .suppressor
            .smoothed_buffer_len
            .store(0, Ordering::Relaxed);

        self.output.write(
            shared,
            buffer,
            uses_clean_write_path(ProcessingPath::RawMonitor),
        );
        self.record_dsp_time(shared, dsp_start);
    }

    /// Bypass: measure output = input and send it directly.
    fn process_bypass(&mut self, shared: &Shared, buffer: &mut [f32], dsp_start: Instant) {
        self.meters.measure_output(shared, buffer);
        shared
            .compressor
            .gain_reduction
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        shared
            .deesser
            .gain_reduction
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        shared
            .deesser
            .detector_confidence
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        shared
            .gate
            .auto_relax_active
            .store(false, Ordering::Relaxed);
        shared
            .compressor
            .current_release_ms
            .store(COMPRESSOR_DEFAULT_RELEASE_TENTH_MS, Ordering::Relaxed); // Default 200ms
        shared
            .compressor
            .current_lufs
            .store((-100.0_f64).to_bits(), Ordering::Relaxed);
        shared
            .compressor
            .current_makeup_gain
            .store(0.0_f64.to_bits(), Ordering::Relaxed);
        shared
            .limiter
            .gain_reduction_db
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        shared
            .limiter
            .peak_gain_reduction_db
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        update_decaying_peak_db(0.0, &shared.limiter.gain_reduction_history_db, 0.15);
        update_decaying_peak_db(
            0.0,
            &shared.output.true_peak_gain_reduction_history_db,
            0.15,
        );
        shared.suppressor.buffer_len.store(0, Ordering::Relaxed);

        self.output.write(
            shared,
            buffer,
            uses_clean_write_path(ProcessingPath::Bypass),
        );
        self.record_dsp_time(shared, dsp_start);
    }

    /// Full chain: gate, suppressor, then the downstream chain. Returns false
    /// when the suppressor's input buffer overflowed.
    fn process_full(&mut self, shared: &Shared, buffer: &mut [f32], dsp_start: Instant) -> bool {
        #[cfg(feature = "vad")]
        {
            let written = enqueue_vad_block(
                &mut self.vad_queue,
                buffer,
                &mut self.vad_source_sample_clock,
                &shared.vad.source_discontinuity,
            );
            // Keep the source clock on samples the worker can actually
            // consume. Dropped blocks are absent from this analysis timeline;
            // the wall-clock watchdog still invalidates results if the worker
            // stalls.
            if written < buffer.len() {
                shared
                    .dsp
                    .rt_buffer_overflow_count
                    .fetch_add(1, Ordering::Relaxed);
                store_rt_error(
                    shared.dsp.rt_error_code.as_ref(),
                    RtErrorCode::FixedBufferOverflow,
                );
            }
        }

        #[cfg(feature = "vad")]
        let vad_snapshot = shared.vad.result().snapshot();
        #[cfg(feature = "vad")]
        let vad_worker_available = vad_snapshot
            .and_then(|snapshot| {
                snapshot
                    .is_fresh(self.vad_source_sample_clock, now_micros(), self.sample_rate)
                    .then_some(snapshot.backend_available)
            })
            .unwrap_or(false);
        #[cfg(feature = "vad")]
        shared
            .vad
            .available
            .store(vad_worker_available, Ordering::Relaxed);
        let gate_controls = &mut self.gate_controls.as_mut_capacity_slice()[..buffer.len()];
        gate_controls.fill(GateControl::BYPASS);
        if shared.gate.enabled.load(Ordering::Acquire) {
            if let Some(control) = take_dirty(
                &shared.gate.dirty,
                || shared.gate.rt_control.snapshot(),
                &shared.dsp.lock_contention_count,
            ) {
                apply_gate_control(&mut self.gate, &control);
            }

            #[cfg(feature = "vad")]
            {
                let latest_prob = vad_snapshot
                    .map(|snapshot| snapshot.probability)
                    .unwrap_or(0.0);
                self.gate
                    .set_external_vad_probability(latest_prob, vad_worker_available);
            }
            self.gate
                .process_block_with_gate_control(buffer, gate_controls);
            shared.gate.gain_meter.store(
                self.gate.current_gain().clamp(0.0, 1.0).to_bits(),
                Ordering::Relaxed,
            );
            shared
                .gate
                .chatter_event_count
                .store(self.gate.chatter_event_count(), Ordering::Relaxed);
            shared
                .gate
                .auto_relax_active
                .store(self.gate.auto_relax_active(), Ordering::Relaxed);
            #[cfg(feature = "vad")]
            {
                let prob = self.gate.get_vad_probability();
                shared
                    .vad
                    .meter_probability
                    .store(prob.to_bits(), Ordering::Relaxed);
                shared
                    .gate
                    .noise_floor_db
                    .store(self.gate.noise_floor().to_bits(), Ordering::Relaxed);
                shared
                    .gate
                    .fused_score
                    .store(self.gate.fused_gate_score().to_bits(), Ordering::Relaxed);
            }
        } else {
            shared
                .gate
                .gain_meter
                .store(1.0_f32.to_bits(), Ordering::Relaxed);
            shared
                .gate
                .auto_relax_active
                .store(false, Ordering::Relaxed);
        }

        // Stage 2: Noise Suppression (RNNoise or DeepFilterNet)
        if shared.suppressor.enabled.load(Ordering::Acquire) {
            if shared
                .suppressor
                .reset_requested
                .swap(false, Ordering::AcqRel)
            {
                self.suppressor.engine.soft_reset();
                self.suppressor.starvation.restart();
                self.suppressor.restart_fade(&self.output);
                update_backend_status_rt(
                    &shared.suppressor.backend_available,
                    &shared.suppressor.backend_failed,
                    shared.dsp.rt_error_code.as_ref(),
                    &self.suppressor.engine,
                );
            }
            // Always feed suppressor first so frame accumulation is correct.
            let accepted = self
                .suppressor
                .engine
                .push_controlled_samples(buffer, gate_controls);
            if accepted < buffer.len() {
                shared
                    .dsp
                    .rt_buffer_overflow_count
                    .fetch_add(1, Ordering::Relaxed);
                store_rt_error(
                    shared.dsp.rt_error_code.as_ref(),
                    RtErrorCode::FixedBufferOverflow,
                );
                return false;
            }
            self.suppressor.starvation.fed(accepted);
            self.suppressor.engine.process_frames();
            shared.suppressor.successful_inference_frames.store(
                self.suppressor.engine.successful_inference_frames(),
                Ordering::Relaxed,
            );
            update_backend_status_rt(
                &shared.suppressor.backend_available,
                &shared.suppressor.backend_failed,
                shared.dsp.rt_error_code.as_ref(),
                &self.suppressor.engine,
            );

            // Track suppressor internal buffer fill level after processing.
            let suppressor_buffered =
                self.suppressor.engine.pending_input() + self.suppressor.engine.available_samples();
            let suppressor_latency =
                self.suppressor.engine.latency_samples() + self.suppressor.engine.pending_input();
            let raw_buffer = suppressor_buffered as u32;
            let prev_smoothed = shared
                .suppressor
                .smoothed_buffer_len
                .load(Ordering::Relaxed);
            let smoothed = smooth_buffer(self.metric_alpha, raw_buffer, prev_smoothed);
            shared
                .suppressor
                .buffer_len
                .store(raw_buffer, Ordering::Relaxed);
            shared
                .suppressor
                .latency_samples
                .store(suppressor_latency as u32, Ordering::Relaxed);
            shared
                .suppressor
                .smoothed_buffer_len
                .store(smoothed, Ordering::Relaxed);

            let available = self.suppressor.engine.available_samples();
            if available > 0 {
                self.suppressor.starvation.restart();
                self.suppressor.output.clear();
                let count = available.min(self.suppressor.output.capacity());
                let _ = self.suppressor.output.set_len_zeroed(count);
                let processed = self
                    .suppressor
                    .engine
                    .pop_samples_into(self.suppressor.output.as_mut_slice());
                let _ = self.suppressor.output.set_len_zeroed(processed);
                let output_slice = self.suppressor.output.as_mut_slice();

                if contain_non_finite(output_slice) {
                    shared
                        .suppressor
                        .non_finite_count
                        .fetch_add(1, Ordering::Relaxed);
                    store_rt_error(
                        shared.dsp.rt_error_code.as_ref(),
                        RtErrorCode::SuppressorNonFinite,
                    );
                    let now = Instant::now();
                    match self.suppressor.non_finite_window_started_at {
                        Some(start)
                            if now.duration_since(start).as_millis() as u64
                                <= NON_FINITE_REBUILD_WINDOW_MS => {}
                        _ => {
                            self.suppressor.non_finite_window_started_at = Some(now);
                            self.suppressor.non_finite_window_count = 0;
                        }
                    }
                    self.suppressor.non_finite_window_count =
                        self.suppressor.non_finite_window_count.saturating_add(1);
                    if self.suppressor.non_finite_window_count >= NON_FINITE_REBUILD_THRESHOLD {
                        shared
                            .stream
                            .restart_requested
                            .store(true, Ordering::Release);
                        self.suppressor.non_finite_window_started_at = None;
                        self.suppressor.non_finite_window_count = 0;
                    }
                    update_backend_status_rt(
                        &shared.suppressor.backend_available,
                        &shared.suppressor.backend_failed,
                        shared.dsp.rt_error_code.as_ref(),
                        &self.suppressor.engine,
                    );
                }

                self.suppressor.recovery_fade.apply(output_slice);
                self.chain.process(
                    shared,
                    output_slice,
                    #[cfg(feature = "vad")]
                    &self.gate,
                    #[cfg(feature = "vad")]
                    self.vad_source_sample_clock,
                );

                // Measure OUTPUT levels
                self.meters.measure_output(shared, output_slice);

                // Send processed samples to output
                self.output.write(shared, output_slice, false);
            } else if accepted > 0
                && !shared.recording.active.load(Ordering::Relaxed)
                && self.suppressor.starvation.starved()
            {
                store_rt_error(
                    shared.dsp.rt_error_code.as_ref(),
                    RtErrorCode::SuppressorBackendFailed,
                );
                shared
                    .stream
                    .restart_requested
                    .store(true, Ordering::Release);
                self.suppressor.starvation.restart();
            }

            // Record DSP processing time
            self.record_dsp_time(shared, dsp_start);
        } else {
            // Suppressor disabled: clear buffer counter
            self.suppressor.starvation.restart();
            shared.suppressor.buffer_len.store(0, Ordering::Relaxed);
            shared
                .suppressor
                .latency_samples
                .store(0, Ordering::Relaxed);
            shared
                .suppressor
                .successful_inference_frames
                .store(0, Ordering::Relaxed);
            shared
                .suppressor
                .smoothed_buffer_len
                .store(0, Ordering::Relaxed);
            if shared
                .suppressor
                .reset_requested
                .swap(false, Ordering::AcqRel)
            {
                self.suppressor.engine.soft_reset();
                self.suppressor.recovery_fade.restart(0);
                self.output
                    .discontinuity_fade_remaining
                    .set(self.output.discontinuity_fade_samples);
                update_backend_status_rt(
                    &shared.suppressor.backend_available,
                    &shared.suppressor.backend_failed,
                    shared.dsp.rt_error_code.as_ref(),
                    &self.suppressor.engine,
                );
            }
            // Suppressor disabled: apply remaining stages directly
            self.chain.process(
                shared,
                buffer,
                #[cfg(feature = "vad")]
                &self.gate,
                #[cfg(feature = "vad")]
                self.vad_source_sample_clock,
            );

            // Measure OUTPUT levels
            self.meters.measure_output(shared, buffer);

            // Send to output
            self.output.write(shared, buffer, false);
            // Record DSP processing time
            self.record_dsp_time(shared, dsp_start);
        }
        true
    }

    /// Restart everything that depends on contiguous input after a gap.
    fn reset_after_input_gap(&mut self, shared: &Shared) {
        self.gate.reset();
        self.suppressor.engine.soft_reset();
        self.suppressor.output.clear();
        shared.suppressor.buffer_len.store(0, Ordering::Relaxed);
        shared
            .suppressor
            .smoothed_buffer_len
            .store(0, Ordering::Relaxed);
        shared.suppressor.latency_samples.store(
            if shared.suppressor.enabled.load(Ordering::Acquire) {
                self.suppressor.engine.latency_samples() as u32
            } else {
                0
            },
            Ordering::Relaxed,
        );
        shared.suppressor.successful_inference_frames.store(
            self.suppressor.engine.successful_inference_frames(),
            Ordering::Relaxed,
        );
        shared
            .gate
            .gain_meter
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        shared
            .gate
            .auto_relax_active
            .store(false, Ordering::Relaxed);
        self.suppressor.starvation.restart();
        #[cfg(feature = "vad")]
        {
            shared
                .vad
                .source_discontinuity
                .fetch_add(1, Ordering::Release);
            shared.vad.available.store(false, Ordering::Release);
        }
        if !shared.mode.raw_monitor.load(Ordering::Acquire)
            && !shared.mode.bypass.load(Ordering::SeqCst)
            && shared.suppressor.enabled.load(Ordering::Acquire)
        {
            self.suppressor.restart_fade(&self.output);
        } else {
            self.suppressor.recovery_fade.restart(0);
            if !shared.mode.raw_monitor.load(Ordering::Acquire) {
                self.output
                    .discontinuity_fade_remaining
                    .set(self.output.discontinuity_fade_samples);
            }
        }
    }
}

impl SuppressorStage {
    /// Fade the suppressor back in after a reset or an engine swap.
    fn restart_fade(&mut self, output: &OutputStage) {
        self.recovery_fade.restart_for_suppressor(
            &self.engine,
            &output.discontinuity_fade_remaining,
            output.discontinuity_fade_samples,
        );
    }

    /// Install engines queued by `set_noise_model` / rebuild and apply a
    /// pending suppressor control change. `in_use` is true only when this pass
    /// feeds the suppressor; otherwise the swap is silent (no fade).
    fn install_pending(&mut self, shared: &Shared, in_use: bool, output: &OutputStage) {
        if let Some(retired) = self.deferred_retire.take() {
            if let Err(retired) = self.retired.try_push(retired) {
                self.deferred_retire = Some(retired);
            }
        }

        if let Some(control) = take_dirty(
            &shared.suppressor.dirty,
            || shared.suppressor.rt_control.snapshot(),
            &shared.dsp.lock_contention_count,
        ) {
            while self.deferred_retire.is_none() && !self.retired.is_full() {
                let candidate_matches = match self.commands.iter().next() {
                    Some(candidate) => candidate.model_type() == control.model,
                    None => break,
                };
                let Some(candidate) = self.commands.try_pop() else {
                    break;
                };

                if candidate_matches {
                    let retired = std::mem::replace(&mut self.engine, candidate);
                    self.engine.soft_reset();
                    self.starvation.restart();
                    if in_use {
                        self.restart_fade(output);
                    }
                    if let Err(retired) = self.retired.try_push(retired) {
                        self.deferred_retire = Some(retired);
                        shared
                            .dsp
                            .rt_buffer_overflow_count
                            .fetch_add(1, Ordering::Relaxed);
                        store_rt_error(
                            shared.dsp.rt_error_code.as_ref(),
                            RtErrorCode::FixedBufferOverflow,
                        );
                        break;
                    }
                } else if let Err(candidate) = self.retired.try_push(candidate) {
                    self.deferred_retire = Some(candidate);
                    shared
                        .dsp
                        .rt_buffer_overflow_count
                        .fetch_add(1, Ordering::Relaxed);
                    store_rt_error(
                        shared.dsp.rt_error_code.as_ref(),
                        RtErrorCode::FixedBufferOverflow,
                    );
                    break;
                }
            }
            if self.engine.model_type() == control.model {
                apply_suppressor_control(&mut self.engine, &control);
                update_backend_status_rt(
                    &shared.suppressor.backend_available,
                    &shared.suppressor.backend_failed,
                    shared.dsp.rt_error_code.as_ref(),
                    &self.engine,
                );
            } else {
                shared.suppressor.dirty.store(true, Ordering::Release);
            }
        }
    }
}

// RT_REGION_END: dsp_processing_loop

// RT_REGION_START: dsp_level_meter
impl LevelMeters {
    fn restart(&mut self) {
        self.input = MeterState::default();
        self.output = MeterState::default();
    }

    fn measure_input(&mut self, shared: &Shared, buffer: &[f32]) {
        let stats = update_meter_block_stats(buffer, &mut self.input, self.coeff);
        shared
            .input
            .peak
            .store(stats.peak_db.to_bits(), Ordering::Relaxed);
        shared
            .input
            .rms
            .store(stats.rms_db.to_bits(), Ordering::Relaxed);
        shared
            .input
            .crest_factor_db
            .store(stats.crest_factor_db.to_bits(), Ordering::Relaxed);
    }

    /// The OUT tap: before the final true-peak limiter and clamp.
    fn measure_output(&mut self, shared: &Shared, buffer: &[f32]) {
        let stats = update_meter_block_stats(buffer, &mut self.output, self.coeff);
        record_level_extremes(buffer, &shared.output.pre_limiter_peak_hold, None);
        shared
            .output
            .peak
            .store(stats.peak_db.to_bits(), Ordering::Relaxed);
        shared
            .output
            .rms
            .store(stats.rms_db.to_bits(), Ordering::Relaxed);
        shared
            .output
            .crest_factor_db
            .store(stats.crest_factor_db.to_bits(), Ordering::Relaxed);

        let block_coeff = smoothing_coeff_for_block(self.sample_rate as f32, 3000.0, buffer.len());
        self.output_short_term_power =
            block_coeff * self.output_short_term_power + (1.0 - block_coeff) * stats.mean_power;
        let loudness = if self.output_short_term_power > 1.0e-12 {
            10.0 * self.output_short_term_power.log10() - 0.691
        } else {
            -120.0
        };
        shared
            .output
            .short_term_lufs
            .store(loudness.to_bits(), Ordering::Relaxed);
    }
}

// RT_REGION_END: dsp_level_meter

// RT_REGION_START: dsp_downstream_chain
impl DownstreamChain {
    /// De-esser, EQ layers, compressor and limiter, in place. Pending control
    /// changes are applied first; a non-finite result is contained and the
    /// stateful stages are cleared.
    fn process(
        &mut self,
        shared: &Shared,
        buffer: &mut [f32],
        #[cfg(feature = "vad")] gate: &NoiseGate,
        #[cfg(feature = "vad")] vad_source_sample_clock: u64,
    ) {
        if let Some(control) = take_dirty(
            &shared.deesser.dirty,
            || shared.deesser.rt_control.snapshot(),
            &shared.dsp.lock_contention_count,
        ) {
            apply_deesser_control(&mut self.deesser, &control);
        }
        if let Some(eq_snapshot) = take_dirty(
            &shared.eq.dirty,
            || shared.eq.control.snapshot(),
            &shared.dsp.lock_contention_count,
        ) {
            apply_eq_control(&mut self.correction_eq, &mut self.tone_eq, &eq_snapshot);
        }
        if let Some(control) = take_dirty(
            &shared.compressor.dirty,
            || shared.compressor.rt_control.snapshot(),
            &shared.dsp.lock_contention_count,
        ) {
            apply_compressor_control(&mut self.compressor, &control);
        }

        // The de-esser owns its enable ramp, like the compressor.
        if self.deesser.is_active() {
            self.deesser.process_block_inplace(buffer);
            shared.deesser.gain_reduction.store(
                self.deesser.current_gain_reduction_db().to_bits(),
                Ordering::Relaxed,
            );
            shared.deesser.detector_confidence.store(
                self.deesser.detector_confidence().to_bits(),
                Ordering::Relaxed,
            );
        } else {
            shared
                .deesser
                .gain_reduction
                .store(0.0_f32.to_bits(), Ordering::Relaxed);
            shared
                .deesser
                .detector_confidence
                .store(0.0_f32.to_bits(), Ordering::Relaxed);
        }

        self.correction_eq.process_block_inplace(buffer);
        self.tone_eq.process_block_inplace(buffer);

        // The compressor owns its enable ramp; keep processing
        // until a disable has faded out.
        if self.compressor.is_active() {
            let true_peak_pressure = f32::from_bits(
                shared
                    .output
                    .true_peak_gain_reduction_db
                    .load(Ordering::Relaxed),
            ) as f64;
            self.compressor.set_limiter_feedback_gain_reduction_db(
                self.limiter
                    .current_gain_reduction()
                    .abs()
                    .max(true_peak_pressure),
            );
            #[cfg(feature = "vad")]
            {
                let vad_snapshot = shared.vad.result().snapshot();
                let now_us = now_micros();
                let (vad_probability, vad_reliability) =
                    vad_snapshot.map_or((0.0, 0.0), |snapshot| {
                        let wall_age_us = if snapshot.last_update_us > 0 {
                            now_us.saturating_sub(snapshot.last_update_us)
                        } else {
                            u64::MAX
                        };
                        let source_age_us = if snapshot.last_update_us > 0 {
                            vad_source_sample_clock
                                .saturating_sub(snapshot.source_sample_end)
                                .saturating_mul(1_000_000)
                                / u64::from(self.sample_rate.max(1))
                        } else {
                            u64::MAX
                        };
                        // Source age reflects audio actually waiting
                        // behind the worker; wall age remains a
                        // watchdog for a stalled callback or worker.
                        let age_us = source_age_us.max(wall_age_us);
                        let reliability =
                            if snapshot.backend_available && age_us <= VAD_PROBABILITY_STALE_US {
                                let freshness = if age_us <= VAD_PROBABILITY_STALE_US / 2 {
                                    1.0
                                } else {
                                    1.0 - (age_us - VAD_PROBABILITY_STALE_US / 2) as f64
                                        / (VAD_PROBABILITY_STALE_US / 2) as f64
                                };
                                let t = freshness.clamp(0.0, 1.0);
                                t * t * (3.0 - 2.0 * t)
                            } else {
                                0.0
                            };
                        (f64::from(snapshot.probability), reliability)
                    });
                let gate_is_enabled = shared.gate.enabled.load(Ordering::Acquire);
                self.compressor.process_block_inplace_with_activity_control(
                    buffer,
                    Some(AutoMakeupActivityInput {
                        vad_probability,
                        vad_reliability,
                        noise_floor_db: if gate_is_enabled {
                            gate.noise_floor() as f64
                        } else {
                            -60.0
                        },
                        live_noise_reliability: if gate_is_enabled {
                            gate.noise_floor_reliability() as f64
                        } else {
                            0.0
                        },
                    }),
                );
            }
            #[cfg(not(feature = "vad"))]
            self.compressor.process_block_inplace(buffer);
            shared.compressor.gain_reduction.store(
                (self.compressor.block_peak_gain_reduction() as f32).to_bits(),
                Ordering::Relaxed,
            );
            let current_release = self.compressor.current_release_time();
            shared
                .compressor
                .current_release_ms
                .store(release_ms_to_tenth_ms(current_release), Ordering::Relaxed);
            shared
                .compressor
                .current_lufs
                .store(self.compressor.current_lufs().to_bits(), Ordering::Relaxed);
            shared.compressor.current_makeup_gain.store(
                self.compressor.current_makeup_gain().to_bits(),
                Ordering::Relaxed,
            );
            shared
                .compressor
                .auto_makeup_error
                .store(self.compressor.auto_makeup_error_code(), Ordering::Relaxed);
        } else {
            shared
                .compressor
                .gain_reduction
                .store(0.0_f32.to_bits(), Ordering::Relaxed);
            shared
                .compressor
                .current_release_ms
                .store(COMPRESSOR_DEFAULT_RELEASE_TENTH_MS, Ordering::Relaxed);
            shared
                .compressor
                .current_lufs
                .store((-100.0_f64).to_bits(), Ordering::Relaxed);
            shared
                .compressor
                .current_makeup_gain
                .store(0.0_f64.to_bits(), Ordering::Relaxed);
            shared
                .compressor
                .auto_makeup_error
                .store(0, Ordering::Relaxed);
        }

        // Always run the limiter: disabled means unity gain through
        // the same delay, so toggling never shifts the timeline.
        self.limiter.process_block_inplace(buffer);
        // Read the peak every block so a disabled limiter's release
        // tail is not published when limiting is turned back on.
        let limiter_peak_gr = self.limiter.peak_gain_reduction_and_reset() as f32;
        if shared.limiter.enabled.load(Ordering::Acquire) {
            shared.limiter.gain_reduction_db.store(
                (self.limiter.current_gain_reduction().abs() as f32).to_bits(),
                Ordering::Relaxed,
            );
            shared
                .limiter
                .peak_gain_reduction_db
                .store(limiter_peak_gr.to_bits(), Ordering::Relaxed);
            update_decaying_peak_db(
                limiter_peak_gr,
                &shared.limiter.gain_reduction_history_db,
                0.15,
            );
        } else {
            shared
                .limiter
                .gain_reduction_db
                .store(0.0_f32.to_bits(), Ordering::Relaxed);
            shared
                .limiter
                .peak_gain_reduction_db
                .store(0.0_f32.to_bits(), Ordering::Relaxed);
            update_decaying_peak_db(0.0, &shared.limiter.gain_reduction_history_db, 0.15);
        }

        // One non-finite sample would poison the meters and latch in the
        // filter state of the de-esser and EQs, muting the stream for good,
        // and in the compressor's envelopes, which then never reduce gain
        // again. Count it, zero it and clear those stages; the compressor
        // clears only its detector and keeps its makeup state. The
        // limiter recovers on its own.
        if contain_non_finite(buffer) {
            shared
                .dsp
                .downstream_non_finite_count
                .fetch_add(1, Ordering::Relaxed);
            self.deesser.reset();
            self.correction_eq.reset();
            self.tone_eq.reset();
            self.compressor.reset_detector_state();
        }
    }
}

// RT_REGION_END: dsp_downstream_chain

// RT_REGION_START: dsp_output_writer
impl OutputStage {
    /// Queue `samples` for the output device, resampling to the device rate
    /// when it differs from the engine rate.
    fn write(&mut self, shared: &Shared, samples: &[f32], clean_path: bool) {
        let requested_release_ms = self.limiter_release_ms.get();
        if requested_release_ms != self.applied_limiter_release_ms {
            self.true_peak_limiter.set_release_ms(requested_release_ms);
            self.applied_limiter_release_ms = requested_release_ms;
        }
        let mut writer = OutputWriteContext {
            output_producer: &mut self.producer,
            output_queue_control_scratch: &mut self.queue_control_scratch,
            discontinuity_fade_scratch: &mut self.discontinuity_fade_scratch,
            output_safety_scratch: &mut self.safety_scratch,
            true_peak_detector: &mut self.true_peak_detector,
            true_peak_limiter: &mut self.true_peak_limiter,
            drift_error_ema: &mut self.drift_error_ema,
            drift_retimer: &mut self.drift_retimer,
            discontinuity_fade_remaining: &self.discontinuity_fade_remaining,
            output_fade_remaining: self.output_fade_remaining,
            limiter_enabled: &shared.limiter.enabled,
            output_ceiling_linear: &self.ceiling_linear,
            counters: OutputWriteCounters::new(shared),
            limits: self.limits,
        };
        if let Some(output_resampler) = self.resampler.as_mut() {
            for &sample in samples {
                if !self.resample_queue.push(sample as f64) {
                    shared
                        .dsp
                        .rt_buffer_overflow_count
                        .fetch_add(1, Ordering::Relaxed);
                    store_rt_error(
                        shared.dsp.rt_error_code.as_ref(),
                        RtErrorCode::FixedBufferOverflow,
                    );
                    break;
                }
            }

            loop {
                self.resampled_scratch.clear();
                let input_frames_needed = output_resampler.input_frames_next();
                while self.resample_queue.len() >= input_frames_needed
                    && input_frames_needed <= self.resampler_frame.len()
                {
                    let Some(outbuf) = self.resampler_out.as_mut() else {
                        break;
                    };
                    if !has_resampler_output_capacity(&self.resampled_scratch, outbuf) {
                        break;
                    }
                    let read = self
                        .resample_queue
                        .pop_into(&mut self.resampler_frame[..input_frames_needed]);
                    if read != input_frames_needed {
                        break;
                    }
                    let in_slices = [&self.resampler_frame[..input_frames_needed]];
                    match output_resampler.process_into_buffer(&in_slices, outbuf, None) {
                        Ok((_nbr_in, nbr_out)) => {
                            for &sample in outbuf[0].iter().take(nbr_out) {
                                if !self.resampled_scratch.push(sample as f32) {
                                    shared
                                        .dsp
                                        .rt_buffer_overflow_count
                                        .fetch_add(1, Ordering::Relaxed);
                                    store_rt_error(
                                        shared.dsp.rt_error_code.as_ref(),
                                        RtErrorCode::FixedBufferOverflow,
                                    );
                                    break;
                                }
                            }
                        }
                        Err(_) => {
                            // The dropped chunk is a hole in the output;
                            // count it and rebuild the stream like the
                            // input side does.
                            shared
                                .output
                                .resampler_error_count
                                .fetch_add(1, Ordering::Relaxed);
                            shared
                                .stream
                                .restart_requested
                                .store(true, Ordering::Release);
                            self.resample_queue.clear();
                            break;
                        }
                    }
                }

                if !writer.write_chunk(self.resampled_scratch.as_slice(), clean_path) {
                    break;
                }
                if self.resample_queue.len() < input_frames_needed {
                    break;
                }
            }
        } else {
            let _ = writer.write_chunk(samples, clean_path);
        }
        self.output_fade_remaining = writer.output_fade_remaining;
    }
}
// RT_REGION_END: dsp_output_writer

/// Exponential smoothing of a per-block `u64` metric.
fn smooth_dsp_time(alpha: f32, raw_us: u64, prev_smoothed: u64) -> u64 {
    let raw_f = raw_us as f32;
    let prev_f = prev_smoothed as f32;
    (alpha * raw_f + (1.0 - alpha) * prev_f) as u64
}

/// Exponential smoothing of a per-block `u32` metric.
fn smooth_buffer(alpha: f32, raw: u32, prev_smoothed: u32) -> u32 {
    let raw_f = raw as f32;
    let prev_f = prev_smoothed as f32;
    (alpha * raw_f + (1.0 - alpha) * prev_f) as u32
}

/// Queue a block for the VAD worker. A block that does not fit is dropped
/// whole and the gap published, so the model never sees a bridged hole.
#[cfg(feature = "vad")]
#[inline]
pub(super) fn enqueue_vad_block(
    producer: &mut AudioProducer,
    samples: &[f32],
    source_sample_clock: &mut u64,
    source_discontinuity: &AtomicU64,
) -> usize {
    // A partial write would make the next accepted sample look contiguous
    // to Silero even though the dropped tail is a source gap. Drop the
    // whole block and publish the boundary before any later write.
    let queue_would_overflow = producer.free_len() < samples.len();
    if queue_would_overflow {
        source_discontinuity.fetch_add(1, Ordering::Release);
        return 0;
    }

    let written = producer.write(samples);
    *source_sample_clock = source_sample_clock.saturating_add(written as u64);
    if written < samples.len() {
        // The free-space check is conservative; preserve the marker if a
        // short write is reported despite sufficient space.
        source_discontinuity.fetch_add(1, Ordering::Release);
    }
    written
}

#[cfg(test)]
mod input_continuity_tests {
    use super::*;
    use crate::audio::buffer::AudioRingBuffer;
    use crate::audio::processor::resampling::build_sinc_resampler;

    #[test]
    fn capture_gap_discards_only_observed_queue_and_invalidates_block() {
        let (mut producer, mut consumer) = AudioRingBuffer::new(8).split();
        let drops = producer.dropped_counter();
        producer.write(&[1.0; 10]);
        let mut observed = 0;
        let mut scratch = [0.0; 3];
        let read =
            read_contiguous_input(&mut consumer, &drops, &mut observed, &mut scratch, (8, 4));
        assert_eq!(read.samples, 0);
        assert_eq!(read.discarded, 8);
        assert!(read.discontinuity);
        assert_eq!(observed, 2);
        assert_eq!(drops.load(Ordering::Relaxed), 2);
        producer.write(&[2.0; 2]);
        let next =
            read_contiguous_input(&mut consumer, &drops, &mut observed, &mut scratch, (8, 4));
        assert_eq!(next.samples, 2);
        assert_eq!(scratch[..2], [2.0; 2]);
        assert!(!next.discontinuity);
    }

    #[test]
    fn capture_gap_during_read_cannot_join_partial_frames() {
        let (mut producer, mut consumer) = AudioRingBuffer::new(8).split();
        let drops = producer.dropped_counter();
        producer.write(&[1.0; 6]);
        let mut observed = 0;
        let mut scratch = [0.0; 3];
        INPUT_DROP_AFTER_READ.with(|flag| flag.set(true));
        let read =
            read_contiguous_input(&mut consumer, &drops, &mut observed, &mut scratch, (8, 4));
        assert_eq!(read.samples, 0);
        assert_eq!(read.discarded, 3);
        assert!(read.discontinuity);
        assert_eq!(observed, 0, "new gap remains pending for a bounded drain");
        let next =
            read_contiguous_input(&mut consumer, &drops, &mut observed, &mut scratch, (8, 4));
        assert_eq!(next.samples, 0);
        assert_eq!(next.discarded, 3);
        assert!(next.discontinuity);
        assert_eq!(observed, 1);
    }

    #[test]
    fn capture_gap_during_drain_is_not_acknowledged_as_recovered() {
        let (mut producer, mut consumer) = AudioRingBuffer::new(8).split();
        let drops = producer.dropped_counter();
        producer.write(&[1.0; 10]);
        let mut observed = 0;
        let mut scratch = [0.0; 3];
        INPUT_DROP_DURING_DRAIN.with(|flag| flag.set(true));
        let read =
            read_contiguous_input(&mut consumer, &drops, &mut observed, &mut scratch, (8, 4));
        assert_eq!(read.samples, 0);
        assert_eq!(read.discarded, 8);
        assert_eq!(observed, 2);
        assert_eq!(drops.load(Ordering::Relaxed), 3);
        producer.write(&[2.0; 2]);
        let next =
            read_contiguous_input(&mut consumer, &drops, &mut observed, &mut scratch, (8, 4));
        assert_eq!(next.samples, 0);
        assert_eq!(next.discarded, 2);
        assert!(next.discontinuity);
        assert_eq!(observed, 3);
    }

    #[test]
    fn backlog_drop_is_bounded_and_empty_input_is_not_a_gap() {
        let (mut producer, mut consumer) = AudioRingBuffer::new(16).split();
        let drops = producer.dropped_counter();
        producer.write(&[1.0; 12]);
        let mut observed = 0;
        let mut scratch = [0.0; 2];
        let read =
            read_contiguous_input(&mut consumer, &drops, &mut observed, &mut scratch, (10, 4));
        assert_eq!(read.discarded, 8);
        assert_eq!(read.samples, 2);
        assert_eq!(consumer.len(), 2);
        assert!(read.discontinuity);
        consumer.read(&mut scratch);
        let empty =
            read_contiguous_input(&mut consumer, &drops, &mut observed, &mut scratch, (10, 4));
        assert_eq!(empty.samples, 0);
        assert!(!empty.discontinuity);
        assert_eq!(empty.discarded, 0);
    }

    #[test]
    fn input_gap_resampler_and_gate_reset_are_allocation_free() {
        let mut resampler = build_sinc_resampler(44_100, 48_000, RESAMPLER_CHUNK_SIZE).unwrap();
        let mut output = resampler.output_buffer_allocate(true);
        resampler
            .process_into_buffer(&[&[0.25; RESAMPLER_CHUNK_SIZE]], &mut output, None)
            .unwrap();
        let mut gate = NoiseGate::new(-40.0, 10.0, 100.0, 48_000.0);
        #[cfg(feature = "vad")]
        gate.set_vad_auto_gate(Some(VadAutoGate::without_backend(48_000, 0.5)));
        let mut audio = [0.1; 480];
        gate.process_block_inplace(&mut audio);
        crate::test_alloc::assert_no_allocations("input gap resampler and gate reset", || {
            resampler.reset();
            gate.reset();
            audio.fill(0.0);
            gate.process_block_inplace(&mut audio);
            let (_, written) = resampler
                .process_into_buffer(&[&[0.0; RESAMPLER_CHUNK_SIZE]], &mut output, None)
                .unwrap();
            assert!(output[0][..written].iter().all(|sample| *sample == 0.0));
            assert!(audio.iter().all(|sample| *sample == 0.0));
        });
    }
}
