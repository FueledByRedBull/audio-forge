//! Tests that run the production DSP loop, supervisor and recovery on in-memory
//! queues (no audio devices): steady state, control changes through the
//! `AudioProcessor` setters, suppressor starvation, input discontinuities, idle
//! pacing, and the supervisor-to-recovery flow on a fake clock.

use super::control::{CompressorControlState, DeesserControlState, GateControlState};
use super::dsp_loop::Pass;
use super::lifecycle::{OpenStream, Streams};
use super::resampling::duration_samples;
use super::stress_harness::{OfflineLoop, OfflineLoopConfig};
use super::supervisor::{
    Supervisor, CALLBACK_STALL_MS, CHECK_INTERVAL_MS, CONSECUTIVE_STALL_CHECKS, STARTUP_GRACE_MS,
};
use super::{AudioProcessor, OUTPUT_PRIME_MS};
use crate::audio::rt::RtErrorCode;
use crate::dsp::eq::{EqBandConfig, EqFilterType, NUM_BANDS};
use crate::dsp::noise_suppressor::{
    ControlledNoiseSuppressor, GateControl, NoiseModel, NoiseSuppressor,
};
use ringbuf::traits::{Consumer, Producer};
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc};
use std::time::Duration;

const BLOCK: usize = 480;
const RATE: f32 = 48_000.0;

/// A tone, handed out block by block without phase breaks.
struct Tone {
    frequency: f32,
    amplitude: f32,
    position: usize,
}

impl Tone {
    fn new(frequency: f32, amplitude: f32) -> Self {
        Self {
            frequency,
            amplitude,
            position: 0,
        }
    }

    fn block(&mut self, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| {
                let phase = 2.0 * std::f32::consts::PI * self.frequency * self.position as f32;
                self.position += 1;
                self.amplitude * (phase / RATE).sin()
            })
            .collect()
    }
}

fn rms_db(samples: &[f32]) -> f32 {
    let power = samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32;
    10.0 * power.max(1.0e-12).log10()
}

fn peak(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0, |peak, s| peak.max(s.abs()))
}

/// Feed `blocks` blocks of `block` samples, letting the loop go idle after
/// each, and return everything it queued for the output.
fn feed(offline: &mut OfflineLoop, tone: &mut Tone, blocks: usize, block: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for _ in 0..blocks {
        assert_eq!(offline.push_input(&tone.block(block)), block);
        offline.run_until_idle();
        out.extend(offline.drain_output());
    }
    out
}

/// Gate, suppressor, compressor and de-esser out of the way, so EQ, limiter and
/// path changes can be measured on their own.
fn plain_chain(processor: &AudioProcessor) {
    processor.set_rnnoise_enabled(false);
    let mut gate = GateControlState::new();
    gate.enabled = false;
    processor.apply_gate_settings(&gate);
    let mut compressor = CompressorControlState::new();
    compressor.enabled = false;
    processor.apply_compressor_settings(&compressor);
    let mut deesser = DeesserControlState::new();
    deesser.enabled = false;
    processor.apply_deesser_settings(&deesser);
}

/// Output level once the loop has settled, from the last `tail` blocks.
fn settled_level(offline: &mut OfflineLoop, tone: &mut Tone, tail: usize) -> f32 {
    feed(offline, tone, 40, BLOCK);
    rms_db(&feed(offline, tone, tail, BLOCK))
}

#[test]
fn steady_state_tone_flows_through_the_whole_chain() {
    let processor = AudioProcessor::new();
    let mut offline = OfflineLoop::new(&processor, OfflineLoopConfig::default());
    let mut tone = Tone::new(440.0, 0.2);
    let blocks = 90;

    let out = feed(&mut offline, &mut tone, blocks, BLOCK);

    assert!(out.iter().all(|s| s.is_finite()));
    assert!(
        peak(&out) <= 1.0,
        "output beyond full scale: {}",
        peak(&out)
    );
    assert!(out.iter().any(|s| s.abs() > 1.0e-4), "the output is silent");
    // The output queue starts with the priming silence; drift control may trim
    // or stretch the stream by a sample here and there, never by a block.
    let expected = blocks * BLOCK + duration_samples(48_000, OUTPUT_PRIME_MS);
    assert!(
        out.len().abs_diff(expected) < BLOCK,
        "{} samples out for {expected} expected",
        out.len()
    );
    assert!(processor.get_input_peak_db() > -30.0);
    assert!(processor.get_output_peak_db() > -120.0);
    assert_eq!(processor.get_lock_contention_count(), 0);
    assert_eq!(processor.get_rt_error_code(), RtErrorCode::None as u32);
    assert_eq!(processor.get_output_resampler_error_count(), 0);
    assert!(!processor.is_recovery_requested());
    assert!(processor.get_dsp_idle_wakeup_count() >= blocks as u64);
}

#[test]
fn control_changes_reach_the_loop_through_the_processor_setters() {
    // EQ: +6 dB bell at the tone.
    let processor = AudioProcessor::new();
    plain_chain(&processor);
    let mut offline = OfflineLoop::new(&processor, OfflineLoopConfig::default());
    let mut tone = Tone::new(1_000.0, 0.1);
    let flat = settled_level(&mut offline, &mut tone, 20);
    let mut bands: Vec<EqBandConfig> = (0..NUM_BANDS)
        .map(EqBandConfig::default_for_index)
        .collect();
    bands[5] = EqBandConfig {
        filter_type: EqFilterType::Bell,
        frequency_hz: 1_000.0,
        gain_db: 6.0,
        q: 1.0,
        slope_db_per_octave: 12,
        enabled: true,
    };
    processor.apply_eq_settings_v2(bands).unwrap();
    let boosted = settled_level(&mut offline, &mut tone, 20);
    assert!(
        (boosted - flat - 6.0).abs() < 1.5,
        "EQ moved the level by {} dB",
        boosted - flat
    );
    assert!(!processor.shared.eq.dirty.load(Ordering::Acquire));

    // Limiter: a lower ceiling clamps a loud tone.
    let loud = settled_level(&mut offline, &mut Tone::new(1_000.0, 0.8), 20);
    processor.set_limiter_ceiling(-12.0);
    let mut loud_tone = Tone::new(1_000.0, 0.8);
    feed(&mut offline, &mut loud_tone, 40, BLOCK);
    let limited = feed(&mut offline, &mut loud_tone, 20, BLOCK);
    assert!(loud > rms_db(&limited) + 3.0);
    assert!(
        peak(&limited) < 0.30,
        "peak {} above the -12 dB ceiling",
        peak(&limited)
    );
    assert!(!processor.shared.limiter.dirty.load(Ordering::Acquire));

    // Gate: a threshold above the tone closes it.
    let processor = AudioProcessor::new();
    plain_chain(&processor);
    let mut gate = GateControlState::new();
    gate.enabled = true;
    gate.threshold_db = -80.0;
    processor.apply_gate_settings(&gate);
    let mut offline = OfflineLoop::new(&processor, OfflineLoopConfig::default());
    let mut quiet = Tone::new(1_000.0, 0.01);
    let open = settled_level(&mut offline, &mut quiet, 20);
    processor.set_gate_threshold(-20.0);
    let closed = settled_level(&mut offline, &mut quiet, 20);
    assert!(
        closed < open - 10.0,
        "gate left the level at {closed} dB (open: {open} dB)"
    );
    assert!(!processor.shared.gate.dirty.load(Ordering::Acquire));
    assert_eq!(processor.get_lock_contention_count(), 0);
}

#[test]
fn bypass_and_raw_monitor_skip_the_chain_and_switch_back() {
    let processor = AudioProcessor::new();
    let mut offline = OfflineLoop::new(&processor, OfflineLoopConfig::default());
    let mut tone = Tone::new(1_000.0, 0.1);
    let input_level = rms_db(&Tone::new(1_000.0, 0.1).block(4_800));
    feed(&mut offline, &mut tone, 40, BLOCK);

    for set in [
        AudioProcessor::set_bypass as fn(&AudioProcessor, bool),
        AudioProcessor::set_raw_monitor_enabled,
    ] {
        set(&processor, true);
        let level = settled_level(&mut offline, &mut tone, 20);
        assert!(
            (level - input_level).abs() < 1.0,
            "direct path changed the level to {level} dB from {input_level} dB"
        );
        assert_eq!(processor.get_compressor_gain_reduction_db(), 0.0);
        set(&processor, false);
        let out = feed(&mut offline, &mut tone, 40, BLOCK);
        assert!(out.iter().all(|s| s.is_finite()));
        assert!(peak(&out) <= 1.0);
    }
    assert_eq!(processor.get_rt_error_code(), RtErrorCode::None as u32);
}

fn offline_with_silent_suppressor(processor: &AudioProcessor) -> OfflineLoop {
    OfflineLoop::new(
        processor,
        OfflineLoopConfig {
            suppressor: Some(StubSuppressor::boxed(NoiseModel::RNNoise, &Arc::default())),
            ..OfflineLoopConfig::default()
        },
    )
}

#[test]
fn a_suppressor_that_stops_answering_is_restarted_after_400_ms_of_audio() {
    // 400 ms is counted in audio fed to the suppressor, not in blocks or time:
    // whatever the block size, the request comes with the block that reaches it.
    for block in [100, BLOCK, 1_000] {
        let processor = AudioProcessor::new();
        let mut offline = offline_with_silent_suppressor(&processor);
        let mut tone = Tone::new(440.0, 0.2);
        let mut fed = 0;
        while !processor.is_recovery_requested() {
            feed(&mut offline, &mut tone, 1, block);
            fed += block;
            assert!(
                fed < 48_000,
                "no restart after {fed} samples at {block} per block"
            );
        }
        assert!(
            (19_200..19_200 + block).contains(&fed),
            "restart requested after {fed} samples fed in blocks of {block}"
        );
        assert_eq!(
            processor.get_rt_error_code(),
            RtErrorCode::SuppressorBackendFailed as u32
        );
        assert_eq!(
            processor.noise_backend_error().as_deref(),
            Some("suppressor backend failed")
        );

        // The count starts over after the request.
        processor
            .shared
            .stream
            .restart_requested
            .store(false, Ordering::Release);
        let blocks_needed = 19_200_usize.div_ceil(block);
        feed(&mut offline, &mut tone, blocks_needed - 1, block);
        assert!(!processor.is_recovery_requested());
        feed(&mut offline, &mut tone, 1, block);
        assert!(processor.is_recovery_requested());
    }
}

#[test]
fn starvation_is_not_reported_while_the_suppressor_is_off_or_recording() {
    let processor = AudioProcessor::new();
    let mut offline = offline_with_silent_suppressor(&processor);
    let mut tone = Tone::new(440.0, 0.2);
    processor.set_rnnoise_enabled(false);
    feed(&mut offline, &mut tone, 80, BLOCK);
    assert!(!processor.is_recovery_requested(), "a disabled suppressor");

    processor.set_rnnoise_enabled(true);
    processor.start_raw_recording(5.0).unwrap();
    feed(&mut offline, &mut tone, 80, BLOCK);
    assert!(!processor.is_recovery_requested(), "while recording");

    let _ = processor.stop_raw_recording();
    feed(&mut offline, &mut tone, 80, BLOCK);
    assert!(processor.is_recovery_requested(), "once recording is over");
}

#[test]
fn a_capture_gap_is_counted_and_the_loop_keeps_running() {
    let processor = AudioProcessor::new();
    let mut offline = OfflineLoop::new(&processor, OfflineLoopConfig::default());
    let mut tone = Tone::new(440.0, 0.2);
    feed(&mut offline, &mut tone, 20, BLOCK);
    assert_eq!(
        processor
            .shared
            .input
            .backlog_recovery_count
            .load(Ordering::Relaxed),
        0
    );

    // More than the capture queue holds: the overflow is a gap in the stream.
    let capacity = 96_000;
    let written = offline.push_input(&tone.block(capacity + 4_000));
    assert_eq!(written, capacity);
    offline.run_until_idle();

    let shared = &processor.shared;
    assert_eq!(
        shared.input.backlog_recovery_count.load(Ordering::Relaxed),
        1
    );
    assert_eq!(
        shared.input.backlog_dropped_samples.load(Ordering::Relaxed),
        capacity as u64
    );
    assert_eq!(
        processor.get_rt_error_code(),
        RtErrorCode::InputBacklogDropped as u32
    );
    // The gate restarts closed, and the stages were cleared.
    assert_eq!(
        f32::from_bits(shared.gate.gain_meter.load(Ordering::Relaxed)),
        0.0
    );
    offline.drain_output();

    // A backlog above 250 ms is cut down to 100 ms.
    let before = shared.input.backlog_dropped_samples.load(Ordering::Relaxed);
    assert_eq!(offline.push_input(&tone.block(20_000)), 20_000);
    offline.run_until_idle();
    assert_eq!(
        shared.input.backlog_recovery_count.load(Ordering::Relaxed),
        2
    );
    assert_eq!(
        shared.input.backlog_dropped_samples.load(Ordering::Relaxed) - before,
        20_000 - 4_800
    );

    // Normal input flows again without further recoveries.
    offline.drain_output();
    let out = feed(&mut offline, &mut tone, 40, BLOCK);
    assert!(out.iter().all(|s| s.is_finite()));
    assert!(peak(&out) <= 1.0);
    assert!(out.iter().any(|s| s.abs() > 1.0e-4));
    assert_eq!(
        shared.input.backlog_recovery_count.load(Ordering::Relaxed),
        2
    );
}

#[test]
fn latency_is_published_on_idle_passes_and_in_every_path() {
    let processor = AudioProcessor::new();
    let mut offline = OfflineLoop::new(&processor, OfflineLoopConfig::default());
    let mut tone = Tone::new(440.0, 0.2);
    let latency = || processor.shared.dsp.latency_us.load(Ordering::Relaxed);
    let clear = || {
        processor
            .shared
            .dsp
            .latency_us
            .store(u64::MAX, Ordering::Relaxed)
    };
    let tick = || std::thread::sleep(Duration::from_millis(110));

    // No input at all: the pass still publishes.
    clear();
    tick();
    assert!(matches!(offline.pass(), Pass::Idle { .. }));
    assert_ne!(latency(), u64::MAX, "idle pass");
    let full = latency();

    type Switch = (&'static str, fn(&AudioProcessor));
    let switches: [Switch; 3] = [
        ("bypass", |p| p.set_bypass(true)),
        ("raw monitor", |p| {
            p.set_bypass(false);
            p.set_raw_monitor_enabled(true);
        }),
        ("full", |p| p.set_raw_monitor_enabled(false)),
    ];
    for (path, switch) in switches {
        switch(&processor);
        clear();
        tick();
        assert_eq!(offline.push_input(&tone.block(BLOCK)), BLOCK);
        assert_eq!(offline.pass(), Pass::Busy);
        assert_ne!(latency(), u64::MAX, "{path}");
        if path != "full" {
            assert!(
                latency() < full,
                "{path}: {} us vs full {full} us",
                latency()
            );
        }
        offline.run_until_idle();
        offline.drain_output();
    }
}

#[test]
fn idle_passes_back_off_and_input_resets_the_back_off() {
    let processor = AudioProcessor::new();
    let mut offline = OfflineLoop::new(&processor, OfflineLoopConfig::default());
    let idle = |offline: &mut OfflineLoop| match offline.pass() {
        Pass::Idle { sleep_us } => sleep_us,
        Pass::Busy => panic!("no input was queued"),
    };

    // No input callback has ever run: the sleep doubles up to its cap.
    let sleeps: Vec<u64> = (0..6).map(|_| idle(&mut offline)).collect();
    assert_eq!(sleeps, [200, 400, 800, 1_600, 1_600, 1_600]);
    assert_eq!(processor.get_dsp_idle_wakeup_count(), 6);
    assert_eq!(processor.get_dsp_idle_sleep_us(), 1_600);

    // Input arrives: the next idle pass starts the back-off over.
    assert_eq!(offline.push_input(&[0.1; BLOCK]), BLOCK);
    assert_eq!(offline.pass(), Pass::Busy);
    assert_eq!(idle(&mut offline), 200);

    // While the input callback is alive the loop polls at the base rate. A
    // callback time ahead of the clock reads as age 0 however late this runs.
    processor
        .shared
        .input
        .callback_time_us
        .store(u64::MAX, Ordering::Relaxed);
    assert_eq!(idle(&mut offline), 100);
}

#[test]
fn a_stalled_stream_goes_from_the_supervisor_to_recovery_on_a_fake_clock() {
    const SECOND: u64 = 1_000_000;
    let processor = AudioProcessor::new();
    let shared = Arc::clone(&processor.shared);
    let mut supervisor = Supervisor::new(Arc::clone(&shared));
    let started = 1_000 * SECOND;
    shared.stream.running.store(true, Ordering::Release);
    shared
        .stream
        .last_start_time_us
        .store(started, Ordering::Release);
    let beat = |at: u64| {
        shared.input.callback_time_us.store(at, Ordering::Relaxed);
        shared.output.callback_time_us.store(at, Ordering::Relaxed);
        shared
            .output
            .last_write_time_us
            .store(at, Ordering::Relaxed);
    };
    let tick = CHECK_INTERVAL_MS * 1_000;

    // Nothing is judged during the start-up grace period, however silent.
    assert!(!supervisor.check(started + STARTUP_GRACE_MS * 1_000 - 1));

    // Healthy callbacks are left alone.
    let last_beat = started + 10 * SECOND;
    beat(last_beat);
    for step in 0..9 {
        assert!(!supervisor.check(last_beat + step * tick));
    }
    assert!(!processor.is_recovery_requested());

    // The callbacks stop. It takes CONSECUTIVE_STALL_CHECKS late checks.
    let first_late = last_beat + (CALLBACK_STALL_MS + 1) * 1_000;
    for step in 0..u64::from(CONSECUTIVE_STALL_CHECKS) - 1 {
        assert!(!supervisor.check(first_late + step * tick), "check {step}");
        assert!(!processor.is_recovery_requested());
    }
    let requested_at = first_late + (u64::from(CONSECUTIVE_STALL_CHECKS) - 1) * tick;
    assert!(supervisor.check(requested_at));
    assert!(processor.is_recovery_requested());
    let reason = processor.get_last_restart_reason().unwrap();
    assert!(reason.starts_with("input_cb_age_ms="), "{reason}");
    // A pending request is not asked for again.
    assert!(!supervisor.check(requested_at + tick));

    // Recovery takes the request and restarts the streams (the start is faked).
    let reopened = |_: &AudioProcessor, _: &mut Streams| Ok("started".to_string());
    assert_eq!(
        processor.service_recovery_at(requested_at, reopened),
        Some(true)
    );
    assert_eq!(processor.get_stream_restart_count(), 1);
    assert!(!processor.is_recovery_requested());
    assert!(!processor.is_recovering());
    assert!(processor.get_last_stream_error().is_none());
    assert_eq!(
        processor.get_last_restart_reason().as_deref(),
        Some(reason.as_str())
    );
    // The stop inside the recovery clears `running`, so the supervisor stands down.
    assert!(!supervisor.check(requested_at + 10 * tick));

    // A second stall right behind the first waits out the 2 s back-off.
    shared
        .stream
        .restart_requested
        .store(true, Ordering::Release);
    assert_eq!(
        processor.service_recovery_at(requested_at + SECOND, reopened),
        None
    );
    assert!(processor.is_recovery_requested());
    assert_eq!(
        processor.service_recovery_at(requested_at + 3 * SECOND, reopened),
        Some(true)
    );
    assert_eq!(processor.get_stream_restart_count(), 2);
}

#[test]
fn getters_do_not_wait_for_a_start_in_progress() {
    // A start holds the lifecycle lock while it opens streams. Meters and
    // diagnostics are read on the GUI thread meanwhile and must not queue behind it.
    let processor = AudioProcessor::new();
    // The state a start is in while it waits for the DSP thread: running, the
    // opened devices published, the lifecycle lock still held.
    processor
        .shared
        .stream
        .running
        .store(true, Ordering::SeqCst);
    {
        let mut devices = processor.lock_devices();
        devices.input_name = Some("Microphone".to_string());
        devices.input_stream = Some(OpenStream {
            endpoint_id: Some("{input}".to_string()),
            sample_rate: 48_000,
        });
        devices.output_name_ordinal = 1;
    }
    let (locked_tx, locked_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let processor_ref = &processor;
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let _starting = processor_ref.lock_streams();
            locked_tx.send(()).unwrap();
            done_rx
                .recv_timeout(Duration::from_secs(10))
                .expect("a getter waited for the lifecycle lock");
        });
        locked_rx.recv().unwrap();

        assert!(processor.is_running());
        assert_eq!(
            processor.active_input_device_name().as_deref(),
            Some("Microphone")
        );
        assert_eq!(
            processor.active_input_device_endpoint_id().as_deref(),
            Some("{input}")
        );
        assert_eq!(processor.active_input_sample_rate(), Some(48_000));
        assert_eq!(processor.active_output_device_name_ordinal(), Some(1));
        assert!(processor.get_input_peak_db() <= 0.0);
        assert!(processor.get_output_rms_db() <= 0.0);
        assert!(processor.get_latency_ms() >= 0.0);
        assert_eq!(processor.get_rt_error_code(), 0);
        assert_eq!(processor.get_dropped_samples(), 0);
        assert!(!processor.is_recovery_requested());
        assert!(!processor.is_recovering());
        assert_eq!(processor.get_stream_restart_count(), 0);
        assert!(processor.get_last_restart_reason().is_none());
        assert!(processor.noise_backend_error().is_none());
        assert!(processor.is_compressor_enabled());
        assert_eq!(processor.get_noise_model(), NoiseModel::RNNoise);
        processor.set_bypass(true);
        processor.set_gate_threshold(-35.0);
        done_tx.send(()).unwrap();
    });
}

/// An engine that takes every sample, never produces any and notes when it
/// was retired.
struct StubSuppressor {
    model: NoiseModel,
    retired: Arc<std::sync::atomic::AtomicBool>,
    enabled: bool,
    available: bool,
}

impl StubSuppressor {
    fn boxed(model: NoiseModel, retired: &Arc<std::sync::atomic::AtomicBool>) -> Box<Self> {
        Box::new(Self {
            model,
            retired: Arc::clone(retired),
            enabled: true,
            available: true,
        })
    }

    /// An engine whose backend did not load.
    #[cfg(feature = "deepfilter")]
    fn unavailable(model: NoiseModel) -> Box<Self> {
        Box::new(Self {
            model,
            retired: Arc::default(),
            enabled: true,
            available: false,
        })
    }
}

impl NoiseSuppressor for StubSuppressor {
    fn push_samples(&mut self, samples: &[f32]) -> usize {
        samples.len()
    }
    fn process_frames(&mut self) {}
    fn available_samples(&self) -> usize {
        0
    }
    fn pop_samples_into(&mut self, _buffer: &mut [f32]) -> usize {
        0
    }
    fn set_strength(&self, _value: f32) {}
    fn get_strength(&self) -> f32 {
        1.0
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    fn soft_reset(&mut self) {}
    fn pending_input(&self) -> usize {
        0
    }
    fn model_type(&self) -> NoiseModel {
        self.model
    }
    fn latency_samples(&self) -> usize {
        480
    }
    fn backend_available(&self) -> bool {
        self.available
    }
    fn backend_error(&self) -> Option<&str> {
        None
    }
    fn backend_failed(&self) -> bool {
        false
    }
    fn retire(&mut self) {
        self.retired.store(true, Ordering::SeqCst);
    }
}

impl ControlledNoiseSuppressor for StubSuppressor {
    fn push_controlled_samples(&mut self, samples: &[f32], _controls: &[GateControl]) -> usize {
        samples.len()
    }
}

/// Queue `engine` for the loop the way `set_noise_model` does.
fn queue_engine(
    processor: &AudioProcessor,
    engine: crate::dsp::noise_suppressor::ControlledNoiseSuppressionEngine,
) {
    let mut queue = processor.shared.suppressor.pending_tx.lock().unwrap();
    assert!(queue.as_mut().unwrap().try_push(engine).is_ok());
    processor
        .shared
        .suppressor
        .dirty
        .store(true, Ordering::Release);
}

#[test]
fn an_engine_swapped_out_of_the_loop_is_retired_there_and_then() {
    use std::sync::atomic::AtomicBool;

    let processor = AudioProcessor::new();
    let old_retired = Arc::new(AtomicBool::new(false));
    let new_retired = Arc::new(AtomicBool::new(false));
    let mut offline = OfflineLoop::new(
        &processor,
        OfflineLoopConfig {
            suppressor: Some(StubSuppressor::boxed(NoiseModel::RNNoise, &old_retired)),
            ..OfflineLoopConfig::default()
        },
    );
    queue_engine(
        &processor,
        StubSuppressor::boxed(NoiseModel::RNNoise, &new_retired),
    );
    offline.run_until_idle();

    // The loop only hands the old engine to the queue the control thread drains
    // later; it must have told it to end its background work by then.
    assert!(old_retired.load(Ordering::SeqCst));
    assert!(!new_retired.load(Ordering::SeqCst));
    assert!(processor
        .shared
        .suppressor
        .retired_rx
        .lock()
        .unwrap()
        .as_mut()
        .unwrap()
        .try_pop()
        .is_some());
}

/// The production loop with DeepFilterNet inference at a worker thread, on the
/// stand-in library of `dsp::deepfilter_ffi::fake`.
#[cfg(feature = "deepfilter")]
mod worker_engine {
    use super::*;
    use crate::dsp::deepfilter_ffi::fake::{worker_engine, Fake};
    use crate::dsp::deepfilter_ffi::DeepFilterModel;
    use crate::dsp::deepfilter_ffi::TEST_WORKER_TRANSIT_SAMPLES as WORKER_TRANSIT_SAMPLES;
    use std::cell::{Cell, RefCell};
    use std::sync::atomic::AtomicBool;
    use std::time::Instant;

    fn select_model(processor: &AudioProcessor, model: NoiseModel) {
        processor.shared.suppressor.rt_control.set_model(model);
        processor
            .shared
            .suppressor
            .current_model
            .store(model as u8, Ordering::Release);
    }

    fn loop_with_worker(fake: &Arc<Fake>) -> (AudioProcessor, OfflineLoop) {
        let processor = AudioProcessor::new();
        select_model(&processor, NoiseModel::DeepFilterNetLL);
        let engine = worker_engine(fake, DeepFilterModel::LowLatency, 1.0);
        let offline = OfflineLoop::new(
            &processor,
            OfflineLoopConfig {
                suppressor: Some(engine),
                ..OfflineLoopConfig::default()
            },
        );
        (processor, offline)
    }

    /// Feed one block and let the worker take every frame it was offered.
    fn feed_block(
        offline: &mut OfflineLoop,
        tone: &mut Tone,
        fake: &Fake,
        frames_offered: usize,
    ) -> Vec<f32> {
        let out = feed(offline, tone, 1, BLOCK);
        wait_for("the worker to take a frame", || {
            fake.started.load(Ordering::Acquire) as usize >= frames_offered
        });
        out
    }

    fn wait_for(what: &str, mut condition: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !condition() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn the_loop_runs_with_inference_at_a_worker_and_publishes_its_latency() {
        let fake = Fake::new();
        let (processor, mut offline) = loop_with_worker(&fake);
        let mut tone = Tone::new(440.0, 0.2);
        let mut out = Vec::new();
        for block in 1..=60 {
            out.extend(feed_block(&mut offline, &mut tone, &fake, block));
        }
        out.extend(feed(&mut offline, &mut tone, 2, BLOCK));

        assert!(out.iter().all(|s| s.is_finite()));
        assert!(peak(&out) <= 1.0);
        assert!(
            rms_db(&out[out.len() - 4_800..]) > -60.0,
            "the output is silent"
        );
        assert!(!processor.noise_backend_failed());
        assert_eq!(processor.get_rt_error_code(), RtErrorCode::None as u32);
        assert_eq!(processor.get_rt_buffer_overflow_count(), 0);
        assert!(
            processor
                .shared
                .suppressor
                .successful_inference_frames
                .load(Ordering::Relaxed)
                >= 55
        );
        // The model's frame, nothing waiting for the next one (the blocks are
        // whole frames) and the worker's transit: the same at every moment.
        let latency = processor
            .shared
            .suppressor
            .latency_samples
            .load(Ordering::Relaxed) as usize;
        assert_eq!(latency, BLOCK + WORKER_TRANSIT_SAMPLES);
    }

    #[test]
    fn a_finished_frame_is_emitted_on_an_idle_pass_without_the_next_block() {
        let fake = Fake::new();
        let (_processor, mut offline) = loop_with_worker(&fake);
        let mut tone = Tone::new(440.0, 0.2);
        let primed = duration_samples(48_000, OUTPUT_PRIME_MS);
        let mut total = feed(&mut offline, &mut tone, 1, BLOCK).len();

        // No more input: only the idle passes can bring the frame out.
        let deadline = Instant::now() + Duration::from_secs(5);
        while total < primed + BLOCK - 8 {
            assert!(
                Instant::now() < deadline,
                "only {total} samples came out ({primed} of them priming)"
            );
            offline.run_until_idle();
            total += offline.drain_output().len();
            std::thread::sleep(Duration::from_micros(200));
        }
    }

    #[test]
    fn a_failing_worker_leaves_dry_passthrough_and_the_loop_reports_it() {
        let fake = Fake::new();
        fake.fail_from.store(5, Ordering::Release);
        let (processor, mut offline) = loop_with_worker(&fake);
        let mut tone = Tone::new(440.0, 0.2);
        let mut out = Vec::new();
        for block in 1..=40 {
            out.extend(feed_block(&mut offline, &mut tone, &fake, block.min(6)));
        }
        out.extend(feed(&mut offline, &mut tone, 2, BLOCK));

        assert!(processor.noise_backend_failed());
        assert_eq!(
            processor.get_rt_error_code(),
            RtErrorCode::SuppressorBackendFailed as u32
        );
        assert!(out.iter().all(|s| s.is_finite()));
        assert!(peak(&out) <= 1.0);
        assert!(
            rms_db(&out[out.len() - 4_800..]) > -30.0,
            "the dry audio is gone"
        );
        assert!(!processor.is_recovery_requested());
    }

    #[test]
    fn a_stalled_worker_is_declared_failed_before_the_starvation_restart() {
        let fake = Fake::new();
        fake.hold.store(true, Ordering::Release);
        let (processor, mut offline) = loop_with_worker(&fake);
        let mut tone = Tone::new(440.0, 0.2);
        let mut out = Vec::new();
        // Each look at the silent worker counts for 50 ms at most, so three
        // blocks cannot make the 300 ms, however long they took.
        for _ in 0..3 {
            out.extend(feed(&mut offline, &mut tone, 1, BLOCK));
        }
        assert!(!processor.noise_backend_failed());
        let deadline = Instant::now() + Duration::from_secs(10);
        while !processor.noise_backend_failed() {
            assert!(Instant::now() < deadline, "the stall was never declared");
            out.extend(feed(&mut offline, &mut tone, 1, BLOCK));
            std::thread::sleep(Duration::from_millis(10));
        }
        for _ in 0..8 {
            out.extend(feed(&mut offline, &mut tone, 1, BLOCK));
        }

        assert_eq!(
            processor.get_rt_error_code(),
            RtErrorCode::SuppressorBackendFailed as u32
        );
        assert!(out.iter().all(|s| s.is_finite()));
        assert!(
            rms_db(&out[out.len() - 4_800..]) > -30.0,
            "the dry audio is gone"
        );
        // The dry audio resets the starvation count, and the stall is declared
        // within 40 blocks anyway.
        assert!(!processor.is_recovery_requested());
        fake.hold.store(false, Ordering::Release);
        wait_for("the declared-failed worker to end", || {
            fake.freed.load(Ordering::Acquire)
        });
    }

    #[test]
    fn a_period_of_8192_samples_costs_latency_not_suppression() {
        let fake = Fake::new();
        let (processor, mut offline) = loop_with_worker(&fake);
        let mut tone = Tone::new(440.0, 0.2);
        for block in 1..=20 {
            feed_block(&mut offline, &mut tone, &fake, block);
        }
        // The longest period the opt-in fixed buffer allows, as one block.
        let mut out = feed(&mut offline, &mut tone, 1, 8192);
        for _ in 0..30 {
            out.extend(feed(&mut offline, &mut tone, 1, BLOCK));
            std::thread::sleep(Duration::from_millis(2));
        }

        assert!(out.iter().all(|s| s.is_finite()));
        assert!(
            rms_db(&out[out.len() - 4_800..]) > -60.0,
            "the output is silent"
        );
        assert!(!processor.noise_backend_failed());
        assert_eq!(processor.get_rt_buffer_overflow_count(), 0);
        assert_eq!(processor.get_rt_error_code(), RtErrorCode::None as u32);
    }

    #[test]
    fn an_input_overflow_does_not_hide_a_dead_worker() {
        let fake = Fake::new();
        fake.hold.store(true, Ordering::Release);
        let (processor, mut offline) = loop_with_worker(&fake);
        let mut tone = Tone::new(440.0, 0.2);
        // 24 of the 26 frames the suppressor input queue holds, and the worker is stuck
        // on the first. The blocks that follow overflow the queue and reset it
        // again and again; the worker's silence is counted through the resets.
        feed(&mut offline, &mut tone, 1, 16 * BLOCK);
        feed(&mut offline, &mut tone, 1, 8 * BLOCK);
        assert!(!processor.noise_backend_failed());
        let mut out = Vec::new();
        for _ in 0..30 {
            if processor.noise_backend_failed() {
                break;
            }
            std::thread::sleep(Duration::from_millis(60));
            out.extend(feed(&mut offline, &mut tone, 1, BLOCK));
        }
        assert!(processor.get_rt_buffer_overflow_count() >= 1);
        assert!(processor.noise_backend_failed());

        for _ in 0..6 {
            out.extend(feed(&mut offline, &mut tone, 1, BLOCK));
        }
        assert!(out.iter().all(|s| s.is_finite()));
        assert!(
            rms_db(&out[out.len() - 2_400..]) > -30.0,
            "the dry audio is gone"
        );
        fake.hold.store(false, Ordering::Release);
    }

    #[test]
    fn stopping_the_loop_does_not_wait_for_a_worker_inside_a_slow_frame() {
        let fake = Fake::new();
        fake.hold.store(true, Ordering::Release);
        let (_processor, mut offline) = loop_with_worker(&fake);
        let mut tone = Tone::new(440.0, 0.2);
        feed_block(&mut offline, &mut tone, &fake, 1);

        // On another thread, so that a drop that waits fails the test instead
        // of hanging it.
        let (dropped, wait_for_drop) = mpsc::channel();
        std::thread::spawn(move || {
            drop(offline);
            let _ = dropped.send(());
        });
        wait_for_drop
            .recv_timeout(Duration::from_secs(2))
            .expect("stopping the loop waited for the worker's slow frame");
        assert!(!fake.freed.load(Ordering::Acquire));

        fake.hold.store(false, Ordering::Release);
        wait_for("the library state to be freed", || {
            fake.freed.load(Ordering::Acquire)
        });
    }

    #[test]
    fn an_engine_the_loop_declines_is_retired_too() {
        let processor = AudioProcessor::new();
        let declined = Arc::new(AtomicBool::new(false));
        let mut offline = OfflineLoop::new(&processor, OfflineLoopConfig::default());
        // The selected model is RNNoise; this one is not it.
        queue_engine(
            &processor,
            StubSuppressor::boxed(NoiseModel::DeepFilterNet, &declined),
        );
        offline.run_until_idle();
        assert!(declined.load(Ordering::SeqCst));
    }

    #[test]
    fn repeated_starts_stops_and_model_switches_leave_no_worker_running() {
        let mut fakes: Vec<Arc<Fake>> = Vec::new();
        let mut tone = Tone::new(440.0, 0.2);
        for _ in 0..6 {
            let first = Fake::new();
            fakes.push(Arc::clone(&first));
            let (processor, mut offline) = loop_with_worker(&first);
            feed_block(&mut offline, &mut tone, &first, 1);

            for model in [DeepFilterModel::Standard, DeepFilterModel::LowLatency] {
                let engine_model = match model {
                    DeepFilterModel::Standard => NoiseModel::DeepFilterNet,
                    DeepFilterModel::LowLatency => NoiseModel::DeepFilterNetLL,
                };
                let next = Fake::new();
                fakes.push(Arc::clone(&next));
                select_model(&processor, engine_model);
                queue_engine(&processor, worker_engine(&next, model, 1.0));
                assert!(feed(&mut offline, &mut tone, 2, BLOCK)
                    .iter()
                    .all(|s| s.is_finite()));
            }
            // The swapped-out engines still wait, undropped, in the queue the
            // control thread drains: their workers must be gone already.
            for old in &fakes[fakes.len() - 3..fakes.len() - 1] {
                wait_for("a swapped-out worker to end", || {
                    old.freed.load(Ordering::Acquire)
                });
            }
            drop(offline);
            drop(processor);
        }
        for fake in &fakes {
            wait_for("a worker to end", || fake.freed.load(Ordering::Acquire));
        }
    }

    #[test]
    fn a_model_selected_while_the_stream_starts_reaches_the_loop() {
        // `start` does not wait for `set_noise_model` (the window calls it on
        // its own thread), and loading a model takes a few hundred
        // milliseconds. The stream begins on the model it finds, so a selection
        // made for a stopped processor must be redone for the running one.
        let processor = AudioProcessor::new();
        let started: RefCell<Option<OfflineLoop>> = RefCell::new(None);
        let builds = Cell::new(0);
        let selected = processor.select_noise_model(NoiseModel::DeepFilterNetLL, |_, _| {
            builds.set(builds.get() + 1);
            let engine = worker_engine(&Fake::new(), DeepFilterModel::LowLatency, 1.0);
            if started.borrow().is_none() {
                *started.borrow_mut() =
                    Some(OfflineLoop::new(&processor, OfflineLoopConfig::default()));
            }
            engine
        });

        assert!(selected);
        assert_eq!(
            builds.get(),
            2,
            "once for the stopped state, once for the loop"
        );
        let mut offline = started.into_inner().unwrap();
        offline.run_until_idle();
        assert!(!processor.shared.suppressor.dirty.load(Ordering::Acquire));
        assert!(processor
            .shared
            .suppressor
            .retired_rx
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .try_pop()
            .is_some());
    }

    #[test]
    fn a_selection_the_started_stream_already_has_needs_no_second_engine() {
        // `set_noise_model` recorded the model before `start` took the control,
        // so the loop began on it; queueing another engine would only load it
        // again and swap it in with a reset and a fade.
        let processor = AudioProcessor::new();
        processor.record_selected_model(NoiseModel::DeepFilterNetLL);
        let first = Arc::new(AtomicBool::new(false));
        let mut offline = OfflineLoop::new(
            &processor,
            OfflineLoopConfig {
                suppressor: Some(StubSuppressor::boxed(NoiseModel::DeepFilterNetLL, &first)),
                ..OfflineLoopConfig::default()
            },
        );
        let builds = Cell::new(0);
        let settled =
            processor.settle_selection_with_started_stream(NoiseModel::DeepFilterNetLL, |_, _| {
                builds.set(builds.get() + 1);
                StubSuppressor::boxed(NoiseModel::DeepFilterNetLL, &first)
            });

        assert!(settled);
        assert_eq!(builds.get(), 0);
        offline.run_until_idle();
        assert!(
            !first.load(Ordering::SeqCst),
            "the running engine was replaced"
        );
        assert!(processor
            .shared
            .suppressor
            .retired_rx
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .try_pop()
            .is_none());
    }

    #[test]
    fn a_selection_whose_engine_cannot_be_built_is_undone_for_the_started_stream() {
        // The stream began on RNNoise; the selection of another model was
        // recorded after that; the engine for it does not load. The control must
        // not go on saying the new model, with the loop waiting for an engine
        // that never comes, and the caller must hear that it failed.
        let processor = AudioProcessor::new();
        let first = Arc::new(AtomicBool::new(false));
        let mut offline = OfflineLoop::new(
            &processor,
            OfflineLoopConfig {
                suppressor: Some(StubSuppressor::boxed(NoiseModel::RNNoise, &first)),
                ..OfflineLoopConfig::default()
            },
        );
        processor.record_selected_model(NoiseModel::DeepFilterNetLL);
        processor
            .shared
            .suppressor
            .dirty
            .store(true, Ordering::Release);

        let settled = processor
            .settle_selection_with_started_stream(NoiseModel::DeepFilterNetLL, |model, _| {
                StubSuppressor::unavailable(model)
            });

        assert!(!settled);
        assert_eq!(processor.get_noise_model(), NoiseModel::RNNoise);
        assert_eq!(
            processor
                .shared
                .suppressor
                .rt_control
                .snapshot()
                .unwrap()
                .model,
            NoiseModel::RNNoise
        );
        offline.run_until_idle();
        assert!(!processor.shared.suppressor.dirty.load(Ordering::Acquire));
        assert!(
            !first.load(Ordering::SeqCst),
            "the running engine was replaced"
        );
    }

    #[test]
    fn the_drift_estimate_steps_once_per_block_however_the_worker_delivers() {
        // The estimate steers how the output is retimed. It is a smoothed
        // queue-level error with its time constant set per block, as it was when
        // every block wrote once. Two frames in one block come out of the worker
        // 25 ms apart, on separate idle passes: two writes, one step.
        let fake = Fake::new();
        fake.delay_ms.store(25, Ordering::Release);
        let (_processor, mut offline) = loop_with_worker(&fake);
        let mut tone = Tone::new(440.0, 0.2);
        let mut steps_per_block = Vec::new();
        for _ in 0..2 {
            assert_eq!(offline.push_input(&tone.block(2 * BLOCK)), 2 * BLOCK);
            let mut last = offline.drift_error_ema();
            let mut steps = 0;
            let until = Instant::now() + Duration::from_millis(400);
            while Instant::now() < until {
                let _ = offline.pass();
                let now = offline.drift_error_ema();
                if now != last {
                    steps += 1;
                    last = now;
                }
                std::thread::sleep(Duration::from_micros(300));
            }
            steps_per_block.push(steps);
        }
        assert_eq!(steps_per_block, [1, 1]);
    }

    #[test]
    fn work_done_on_an_idle_pass_is_processing_time_and_does_not_step_the_buffer_metric() {
        let fake = Fake::new();
        let (processor, mut offline) = loop_with_worker(&fake);
        let mut tone = Tone::new(440.0, 0.2);
        assert_eq!(offline.push_input(&tone.block(BLOCK)), BLOCK);
        let _ = offline.pass(); // the block: its frame goes to the worker
        let smoothed_after_block = processor
            .shared
            .suppressor
            .smoothed_buffer_len
            .load(Ordering::Relaxed);
        assert!(smoothed_after_block > 0);
        processor.shared.dsp.time_us.store(0, Ordering::Relaxed);
        processor
            .shared
            .dsp
            .time_smoothed_us
            .store(0, Ordering::Relaxed);

        // Only the idle passes that bring the frame out can add to either.
        wait_for("the frame to come out on an idle pass", || {
            let _ = offline.pass();
            std::thread::sleep(Duration::from_micros(200));
            processor.shared.dsp.time_us.load(Ordering::Relaxed) > 0
        });
        assert!(
            processor
                .shared
                .dsp
                .time_smoothed_us
                .load(Ordering::Relaxed)
                > 0
        );
        assert_eq!(
            processor
                .shared
                .suppressor
                .smoothed_buffer_len
                .load(Ordering::Relaxed),
            smoothed_after_block
        );
    }
}
