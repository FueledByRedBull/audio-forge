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

/// A suppressor that takes every sample and never produces any.
#[derive(Default)]
struct SilentSuppressor {
    enabled: bool,
}

impl NoiseSuppressor for SilentSuppressor {
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
        NoiseModel::RNNoise
    }
    fn latency_samples(&self) -> usize {
        480
    }
    fn backend_available(&self) -> bool {
        true
    }
    fn backend_error(&self) -> Option<&str> {
        None
    }
    fn backend_failed(&self) -> bool {
        false
    }
}

impl ControlledNoiseSuppressor for SilentSuppressor {
    fn push_controlled_samples(&mut self, samples: &[f32], _controls: &[GateControl]) -> usize {
        samples.len()
    }
}

fn offline_with_silent_suppressor(processor: &AudioProcessor) -> OfflineLoop {
    OfflineLoop::new(
        processor,
        OfflineLoopConfig {
            suppressor: Some(Box::new(SilentSuppressor::default())),
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
