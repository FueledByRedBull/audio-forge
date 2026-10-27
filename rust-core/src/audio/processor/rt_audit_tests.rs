// The shared imports serve tests gated on the default features.
#![cfg_attr(
    not(all(feature = "vad", feature = "deepfilter")),
    allow(unused_imports)
)]

use super::*;
use super::{
    control::*, diagnostics::*, dsp_loop::*, output_writer::*, python_api::*, resampling::*,
    routing::*, vad_worker::*,
};
use crate::audio::buffer::AudioRingBuffer;
use crate::audio::input::TARGET_SAMPLE_RATE;
use crate::audio::rt::RtCommandQueue;
use crate::dsp::noise_suppressor::{
    new_controlled_noise_suppression_engine as new_noise_suppression_engine,
    ControlledNoiseSuppressionEngine as NoiseSuppressionEngine, NoiseModel,
};
use crate::dsp::{Compressor, DeEsser, Limiter, NoiseGate, ParametricEQ};
use crate::rt_scan::{assert_rt_clean, marked_region, source_between};
use pyo3::prelude::*;
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
#[cfg(not(windows))]
use thread_priority::{set_current_thread_priority, ThreadPriority};

/// The production code that sets up and runs the DSP thread: dsp_loop.rs and
/// lifecycle.rs without their inline test modules.
fn dsp_loop_production() -> String {
    let dsp = include_str!("dsp_loop.rs");
    let lifecycle = include_str!("lifecycle.rs");
    let dsp_end = dsp
        .find("#[cfg(test)]\nmod input_continuity_tests")
        .unwrap();
    let lifecycle_end = lifecycle.find("#[cfg(test)]\nmod startup_tests").unwrap();
    format!("{}{}", &dsp[..dsp_end], &lifecycle[..lifecycle_end])
}

// --- RB-003 / RT-022: every snapshot site re-arms ------------------------

#[test]
fn every_dsp_loop_control_snapshot_site_rearms_or_goes_through_startup_control() {
    let production = dsp_loop_production();
    let sites: Vec<usize> = production
        .match_indices(".snapshot()")
        .map(|(i, _)| i)
        .collect();
    assert!(
        sites.len() >= 10,
        "expected the loop and start-up snapshot sites"
    );
    for &site in &sites {
        let before = &production[site.saturating_sub(60)..site];
        // The VAD snapshot is the lock-free result seqlock, not a control state.
        if before.contains("vad.result()") || before.contains("vad_snapshot") {
            continue;
        }
        // The statement holding this snapshot, up to it, must hand the
        // snapshot to a helper that re-arms an unstable one.
        let statement_start = production[..site]
            .rfind([';', '{', '}'])
            .map_or(0, |index| index + 1);
        let statement = &production[statement_start..site];
        assert!(
            ["startup_control(", "take_dirty(", "rearm_unless_stable("]
                .iter()
                .any(|helper| statement.contains(helper)),
            "snapshot site without a re-arm: ...{statement}"
        );
    }
    assert!(!production.contains("unwrap_or_else(GateControlState::new)"));
}

#[test]
fn unstable_start_up_snapshot_rearms_dirty_and_counts_contention() {
    let dirty = AtomicBool::new(false);
    let contention = AtomicU64::new(0);
    let stable = startup_control(Some(7_u32), || 0, &dirty, &contention);
    assert_eq!(stable, 7);
    assert!(!dirty.load(Ordering::Acquire));
    assert_eq!(contention.load(Ordering::Relaxed), 0);

    let fallback = startup_control(None, || 42_u32, &dirty, &contention);
    assert_eq!(fallback, 42);
    assert!(
        dirty.load(Ordering::Acquire),
        "defaults must not outlive the first pass"
    );
    assert_eq!(contention.load(Ordering::Relaxed), 1);
}

// --- RT-011: contention is measured, limiter reports instability -----------

#[test]
fn limiter_consume_counts_an_unstable_snapshot_as_contention() {
    let control = AtomicLimiterControlState::new();
    let dirty = AtomicBool::new(true);
    let mut limiter = Limiter::default_settings(48_000.0);
    let release = Cell::new(50.0_f32);
    let ceiling = Cell::new(1.0_f32);
    let lookahead = AtomicU64::new(0);
    let contention = AtomicU64::new(0);
    let mut consume = || {
        apply_pending_limiter_control(
            &dirty,
            &control,
            &mut limiter,
            &release,
            &ceiling,
            &lookahead,
            &contention,
        )
    };
    control.seq.store(1, Ordering::Release);
    consume();
    control.seq.store(2, Ordering::Release);
    consume();
    assert!(!dirty.load(Ordering::Acquire));
    assert_eq!(contention.load(Ordering::Relaxed), 1);

    // The loop feeds its own contention counter.
    let region = marked_region(include_str!("dsp_loop.rs"), "dsp_processing_loop");
    let call = &region[region.find("apply_pending_limiter_control(").unwrap()..];
    let call = &call[..call.find(");").unwrap()];
    assert!(call.contains("&shared.dsp.lock_contention_count"), "{call}");
}

// --- RB-004: seqlock fences ------------------------------------------------

#[test]
fn control_seqlock_orders_its_relaxed_data_with_fences() {
    let source = include_str!("control.rs");
    let writer = source_between(
        source,
        "fn locked_control_update",
        "fn stable_control_snapshot",
    );
    let first = writer.find("fetch_add(1, Ordering::Relaxed)").unwrap();
    let fence = writer.find("fence(Ordering::Release)").unwrap();
    let apply = writer.find("apply();").unwrap();
    assert!(first < fence && fence < apply);
    let reader = source_between(
        source,
        "fn stable_control_snapshot",
        "fn release_ms_to_tenth_ms",
    );
    let read = reader.find("read();").unwrap();
    let fence = reader.find("fence(Ordering::Acquire)").unwrap();
    let second = reader.rfind("seq.load(").unwrap();
    assert!(read < fence && fence < second);

    let vad = include_str!("vad_worker.rs");
    let begin = source_between(vad, "fn begin_write", "fn invalidate");
    assert!(begin.contains("fence(Ordering::Release)"));
    assert_eq!(vad.matches("self.begin_write()").count(), 2);
    assert!(!vad.contains("sequence.fetch_add(1, Ordering::AcqRel)"));
}

#[test]
fn control_seqlock_never_returns_a_torn_pair() {
    let writer = Mutex::new(());
    let seq = AtomicU64::new(0);
    let (a, b) = (AtomicU64::new(0), AtomicU64::new(0));
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for value in 1..=100_000_u64 {
                locked_control_update(&writer, &seq, || {
                    a.store(value, Ordering::Relaxed);
                    b.store(value, Ordering::Relaxed);
                });
            }
            done.store(true, Ordering::Release);
        });
        let mut seen = 0_u64;
        while !done.load(Ordering::Acquire) {
            if let Some((x, y)) = stable_control_snapshot(&seq, || {
                (a.load(Ordering::Relaxed), b.load(Ordering::Relaxed))
            }) {
                assert_eq!(x, y, "torn snapshot");
                seen += 1;
            }
        }
        assert!(seen > 0);
    });
}

// --- RB-005 / RT-012: start lifecycle --------------------------------------

#[test]
fn redundant_start_keeps_the_pending_restart_and_the_backoff() {
    let processor = AudioProcessor::new();
    processor
        .shared
        .stream
        .running
        .store(true, Ordering::SeqCst);
    processor
        .shared
        .stream
        .restart_requested
        .store(true, Ordering::Release);
    processor
        .shared
        .stream
        .restart_backoff_index
        .store(2, Ordering::Release);

    assert_eq!(
        processor.start(None, None),
        Err("Already running".to_string())
    );

    assert!(processor
        .shared
        .stream
        .restart_requested
        .load(Ordering::Acquire));
    assert_eq!(
        processor
            .shared
            .stream
            .restart_backoff_index
            .load(Ordering::Acquire),
        2
    );
    assert!(processor.lock_streams().supervisor_thread.is_none());
    processor
        .shared
        .stream
        .running
        .store(false, Ordering::SeqCst);
}

#[test]
fn failed_start_attempt_leaves_the_clean_stopped_state() {
    let processor = AudioProcessor::new();
    processor
        .shared
        .output
        .sample_rate
        .store(96_000, Ordering::Relaxed);
    let result = processor.start_with_cleanup(&mut processor.lock_streams(), |processor, _| {
        // What a half-built attempt has published when the output step fails.
        processor
            .shared
            .input
            .fixed_buffer_frames
            .store(480, Ordering::Relaxed);
        processor
            .shared
            .output
            .fixed_buffer_frames
            .store(480, Ordering::Relaxed);
        processor
            .shared
            .output
            .sample_rate
            .store(44_100, Ordering::Relaxed);
        processor
            .shared
            .stream
            .running
            .store(true, Ordering::SeqCst);
        processor
            .shared
            .stream
            .restart_requested
            .store(true, Ordering::Release);
        let (_producer, consumer) = AudioRingBuffer::new(16).split();
        *processor.shared.recording.consumer.lock().unwrap() = Some(consumer);
        let (probe, _consumer) = AudioRingBuffer::new(16).split();
        *processor.shared.output.probe_producer.lock().unwrap() = Some(probe);
        processor
            .shared
            .output
            .probe_active
            .store(true, Ordering::Release);
        Err("Failed to resolve audio output: test".to_string())
    });

    assert_eq!(
        result,
        Err("Failed to resolve audio output: test".to_string())
    );
    assert!(!processor.is_running());
    assert_eq!(processor.input_fixed_buffer_frames(), 0);
    assert_eq!(processor.output_fixed_buffer_frames(), 0);
    assert_eq!(
        processor.output_sample_rate(),
        96_000,
        "the failed attempt's rate must not stay"
    );
    assert!(processor
        .shared
        .recording
        .consumer
        .lock()
        .unwrap()
        .is_none());
    assert!(processor
        .shared
        .output
        .probe_producer
        .lock()
        .unwrap()
        .is_none());
    assert!(!processor.shared.output.probe_active.load(Ordering::Acquire));
    assert!(!processor
        .shared
        .stream
        .restart_requested
        .load(Ordering::Acquire));
}

#[test]
fn successful_start_attempt_is_not_cleaned_up() {
    let processor = AudioProcessor::new();
    let result = processor.start_with_cleanup(&mut processor.lock_streams(), |processor, _| {
        processor
            .shared
            .input
            .fixed_buffer_frames
            .store(480, Ordering::Relaxed);
        Ok("started".to_string())
    });
    assert_eq!(result, Ok("started".to_string()));
    assert_eq!(processor.input_fixed_buffer_frames(), 480);
}

#[test]
fn dsp_thread_spawn_failure_is_an_error_not_a_panic() {
    let production = dsp_loop_production();
    let start = source_between(
        &production,
        "fn start_streams_unguarded(",
        "/// Stop audio processing",
    );
    assert!(!start.contains("std::thread::spawn("));
    assert!(start.contains("std::thread::Builder::new().spawn("));
    let public = source_between(
        &production,
        "pub fn start_with_device_ordinals_and_ids(",
        "fn start_with_cleanup(",
    );
    assert!(public.contains("self.start_with_cleanup("));
    assert!(
        public.find("\"Already running\"").unwrap() < public.find("ensure_supervisor").unwrap(),
        "the running check must precede any state change"
    );
}

// --- RT-023: a late fault cannot restart a stopped processor ---------------

#[test]
fn stop_clears_a_restart_request_raised_while_the_dsp_thread_winds_down() {
    let processor = AudioProcessor::new();
    processor
        .shared
        .stream
        .running
        .store(true, Ordering::SeqCst);
    processor
        .shared
        .stream
        .restart_requested
        .store(true, Ordering::Release);
    let shared = Arc::clone(&processor.shared);
    processor.lock_streams().process_thread = Some(std::thread::spawn(move || {
        let restart_requested = &shared.stream.restart_requested;
        while shared.stream.running.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        // A fault in the pass after running=false raises its request only
        // once stop() has cleared the earlier one. Wait (bounded) for that
        // clear: the fixed stop() clears after the join, so this times out.
        let deadline = Instant::now() + Duration::from_millis(100);
        while restart_requested.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::yield_now();
        }
        restart_requested.store(true, Ordering::Release);
    }));

    processor.stop();

    assert!(
        !processor
            .shared
            .stream
            .restart_requested
            .load(Ordering::Acquire),
        "a request raised during shutdown survived stop()"
    );
    assert_eq!(processor.service_recovery(), None);
}

// --- RT-006: DSP thread panic ----------------------------------------------

#[test]
fn dsp_loop_panic_is_recorded_and_requests_a_restart() {
    let state = Mutex::new(StreamRecoveryState::default());
    let restart = AtomicBool::new(false);
    run_guarded_dsp_loop(|| panic!("index out of bounds"), &state, &restart);
    assert!(restart.load(Ordering::Acquire));
    assert_eq!(
        state.lock().unwrap().last_reason.as_deref(),
        Some("DSP thread panicked: index out of bounds")
    );

    let state = Mutex::new(StreamRecoveryState::default());
    let restart = AtomicBool::new(false);
    run_guarded_dsp_loop(|| {}, &state, &restart);
    assert!(!restart.load(Ordering::Acquire));
    assert!(state.lock().unwrap().last_reason.is_none());

    let source = include_str!("dsp_loop.rs");
    assert!(!marked_region(source, "dsp_processing_loop").contains("catch_unwind"));
    assert!(source
        .split_whitespace()
        .collect::<String>()
        .contains("run_guarded_dsp_loop(dsp_loop,"));
}

#[test]
fn stop_records_a_dsp_thread_that_ended_in_a_panic() {
    let processor = AudioProcessor::new();
    let thread = std::thread::spawn(|| panic!("late failure"));
    while !thread.is_finished() {
        std::thread::yield_now();
    }
    processor.lock_streams().process_thread = Some(thread);
    processor.stop();
    assert!(processor
        .get_last_restart_reason()
        .is_some_and(|reason| reason.contains("DSP thread panicked: late failure")));
    assert!(!processor
        .shared
        .stream
        .restart_requested
        .load(Ordering::Acquire));
}

// --- RT-007: start path ----------------------------------------------------

#[test]
fn missing_pro_audio_registration_degrades_instead_of_failing_start() {
    let degraded = AtomicBool::new(false);
    let guard = tolerate_scheduling_failure(Err::<u8, _>("no MMCSS task"), &degraded);
    assert!(guard.is_none());
    assert!(degraded.load(Ordering::Relaxed));
    let guard = tolerate_scheduling_failure(Ok::<_, &str>(5_u8), &degraded);
    assert_eq!(guard, Some(5));
    assert!(!degraded.load(Ordering::Relaxed));
    assert!(!dsp_loop_production().contains("Failed to register DSP thread with Pro Audio"));
    assert!(!AudioProcessor::new().is_dsp_scheduling_degraded());
}

#[test]
fn initial_suppressor_engine_is_built_before_the_dsp_thread_exists() {
    let production = dsp_loop_production();
    let start = source_between(
        &production,
        "fn start_streams_unguarded(",
        "/// Stop audio processing",
    );
    let spawn = start.find("std::thread::Builder::new().spawn(").unwrap();
    let build = start.find("new_noise_suppression_engine(").unwrap();
    assert!(
        build < spawn,
        "a model or DLL load must not delay DSP readiness"
    );
    assert!(!start[spawn..].contains("new_noise_suppression_engine("));
}

#[test]
fn live_suppressor_engines_come_from_the_factory_that_puts_inference_at_a_worker() {
    // The offline simulators need engines that answer inline; every engine the
    // live loop installs (first, selected or rebuilt) must not.
    let lifecycle = include_str!("lifecycle.rs");
    assert!(
        lifecycle.contains("new_realtime_noise_suppression_engine as new_noise_suppression_engine")
    );
    let controls = include_str!("noise_controls.rs");
    // The one place that picks between the two factories, and who asks for which.
    let builder = source_between(controls, "fn engine_builder(", "fn select_noise_model(");
    assert!(builder.contains("if realtime {"));
    assert!(builder.contains("new_realtime_noise_suppression_engine("));
    assert!(builder.contains("new_noise_suppression_engine("));
    let select = source_between(
        controls,
        "fn select_noise_model(",
        "pub(super) fn rebuild_noise_suppressor(",
    );
    // A stopped processor builds its engine only to ask whether it loads.
    assert!(select.contains("build(model, running)"));
    let rebuild = source_between(
        controls,
        "pub(super) fn rebuild_noise_suppressor(",
        "/// Get current noise suppression model",
    );
    assert!(rebuild.contains("self.rebuild_with(self.engine_builder())"));
    assert!(rebuild.contains("build(model, true)"));
    assert!(!controls[controls.find("fn list_noise_models").unwrap()..]
        .contains("new_realtime_noise_suppression_engine("));
}

// --- RT-004: suppressor command queue ----------------------------------------

#[test]
fn suppressor_queue_is_drained_at_the_top_of_every_pass() {
    let source = include_str!("dsp_loop.rs");
    let rt = marked_region(source, "dsp_processing_loop");
    let pass = source_between(rt, "fn pass(&mut self)", "fn reset_input_continuity(");
    let call = pass.find("let suppressor_in_use =").unwrap();
    let install = pass.find(".install_pending(").unwrap();
    let input_read = pass.find("self.input.read_block()").unwrap();
    let first_branch = pass.find("if n_raw == 0").unwrap();
    assert!(call < install && install < input_read && input_read < first_branch);
    // The old drain lived inside `if use_suppressor`, in the Full path only.
    let full_path = source_between(
        rt,
        "// Stage 2: Noise Suppression",
        "// Suppressor disabled: clear buffer counter",
    );
    let compact = |text: &str| text.split_whitespace().collect::<String>();
    assert!(!full_path.contains("self.suppressor.commands"));
    assert!(!compact(full_path).contains("take_dirty(&shared.suppressor.dirty"));
    let drain = &rt[rt.find("fn install_pending(").unwrap()..];
    assert!(compact(drain).contains("take_dirty(&shared.suppressor.dirty,"));
    assert!(drain.contains("self.retired"));
    assert!(
        drain.contains("if in_use"),
        "a swap with no suppressor audio must stay silent"
    );
}

#[test]
fn suppressor_queue_has_room_for_only_two_engines() {
    // Why the drain above must not depend on the suppressor being in use.
    let processor = AudioProcessor::new();
    let (tx, _rx) =
        RtCommandQueue::<NoiseSuppressionEngine, RT_SUPPRESSOR_COMMAND_CAPACITY>::new().split();
    *processor.shared.suppressor.pending_tx.lock().unwrap() = Some(tx);
    assert!(processor.rebuild_noise_suppressor());
    assert!(processor.rebuild_noise_suppressor());
    assert!(
        !processor.rebuild_noise_suppressor(),
        "nothing drains the queue here"
    );
}

#[cfg(feature = "vad")]
#[test]
fn offline_gate_clamps_vad_threshold_and_pre_gain_like_the_live_setters() {
    Python::initialize();
    Python::attach(|py| {
        let settings = pyo3::types::PyDict::new(py);
        for (threshold, pre_gain, expected) in [
            (5.0, 1e6, (VAD_THRESHOLD_MAX, VAD_PRE_GAIN_MAX)),
            (-1.0, 0.0, (VAD_THRESHOLD_MIN, VAD_PRE_GAIN_MIN)),
            (0.55, 2.5, (0.55, 2.5)),
        ] {
            settings.set_item("gate_vad_threshold", threshold).unwrap();
            settings.set_item("gate_vad_pre_gain", pre_gain).unwrap();
            let gate = py_gate(Some(&settings), 48_000.0, 1).unwrap();
            assert_eq!((gate.vad_threshold, gate.vad_pre_gain), expected);
        }
    });
}

// --- RT-008: non-finite downstream output ------------------------------------

#[test]
fn non_finite_samples_are_zeroed_and_reported() {
    let mut block = [0.25_f32, f32::NAN, -0.5, f32::INFINITY, f32::NEG_INFINITY];
    assert!(contain_non_finite(&mut block));
    assert_eq!(block, [0.25, 0.0, -0.5, 0.0, 0.0]);
    let mut clean = [0.25_f32, -0.5, 1.0e-30];
    let expected = clean;
    assert!(!contain_non_finite(&mut clean));
    assert_eq!(clean, expected, "finite input must stay bit-identical");
    crate::test_alloc::assert_no_allocations("contain_non_finite", || {
        contain_non_finite(&mut block);
    });
}

/// A Bypass/Raw -> Full transition resets both EQs (now clearing their
/// memory) and starts the output's discontinuity fade. The faded result must
/// not step harder than the settled signal itself.
#[test]
fn an_eq_reset_at_a_path_transition_stays_inside_the_discontinuity_fade() {
    let sample_rate = 48_000.0;
    let fade_samples = duration_samples(48_000, 6).max(1);
    let make_eq = || {
        let mut eq = ParametricEQ::new(sample_rate);
        for (index, frequency_hz, gain_db) in [(2, 200.0, 6.0), (5, 1_500.0, -4.0)] {
            let mut config = eq.get_band_config(index).unwrap();
            config.frequency_hz = frequency_hz;
            config.gain_db = gain_db;
            eq.set_band_config(index, config);
        }
        eq.reset();
        eq
    };
    let tone = |range: std::ops::Range<usize>| -> Vec<f32> {
        range
            .map(|n| {
                0.5 * (2.0 * std::f64::consts::PI * 200.0 * n as f64 / sample_rate).sin() as f32
            })
            .collect()
    };
    let max_step = |samples: &[f32]| {
        samples
            .windows(2)
            .fold(0.0_f32, |worst, pair| worst.max((pair[1] - pair[0]).abs()))
    };

    // The settled signal: an EQ that has run on the tone the whole time.
    let mut steady = make_eq();
    steady.process_block_inplace(&mut tone(0..24_000));
    let mut reference = tone(24_060..26_108);
    steady.process_block_inplace(&mut reference);
    let settled_step = max_step(&reference);

    // The transition: the same EQ is reset on the way into the Full path.
    let mut eq = make_eq();
    eq.process_block_inplace(&mut tone(0..24_000));
    eq.reset();
    let mut settled = tone(24_060..26_108);
    eq.process_block_inplace(&mut settled);
    // The output starts from silence, so the first sample is a step too.
    let from_silence = |samples: &[f32]| max_step(&[&[0.0_f32][..], samples].concat());
    let unfaded_step = from_silence(&settled);
    // Without the fade the restart is a hard step, so the fade is what bounds it.
    assert!(
        unfaded_step > settled_step * 5.0,
        "premise: an unfaded restart steps {unfaded_step}"
    );
    let mut remaining = fade_samples;
    fade_in(&mut settled, &mut remaining, fade_samples);
    assert_eq!(remaining, 0);
    let faded_step = from_silence(&settled);
    assert!(
        faded_step <= settled_step * 1.5,
        "faded transition stepped {faded_step}, settled signal steps {settled_step}"
    );
}

#[test]
fn a_nan_latches_the_stages_and_meters_unless_contained_and_cleared() {
    let poison = |buffer: &mut [f32; 64]| {
        buffer.fill(0.1);
        buffer[10] = f32::NAN;
    };
    let mut eq = ParametricEQ::new(48_000.0);
    eq.set_band_gain(4, 6.0);
    let mut poisoned = [0.0_f32; 64];
    poison(&mut poisoned);
    eq.process_block_inplace(&mut poisoned);
    let mut next = [0.1_f32; 64];
    eq.process_block_inplace(&mut next);
    assert!(
        next.iter().any(|s| !s.is_finite()),
        "premise: filter state latches NaN"
    );
    eq.reset();
    let mut recovered = [0.1_f32; 64];
    eq.process_block_inplace(&mut recovered);
    assert!(
        recovered.iter().all(|s| s.is_finite()),
        "the EQ must recover"
    );
    assert!(eq.is_enabled(), "a recovery must not leave the EQ disabled");
    let mut disabled = ParametricEQ::new(48_000.0);
    disabled.set_enabled(false);
    disabled.reset();
    assert!(!disabled.is_enabled(), "a user-disabled EQ stays disabled");

    let mut deesser = DeEsser::new(48_000.0);
    deesser.set_enabled(true);
    let mut block = [0.0_f32; 64];
    poison(&mut block);
    deesser.process_block_inplace(&mut block);
    deesser.reset();
    let mut recovered = [0.1_f32; 64];
    deesser.process_block_inplace(&mut recovered);
    assert!(
        recovered.iter().all(|s| s.is_finite()),
        "the de-esser must recover"
    );

    // The compressor's output stays finite after a NaN, but its envelopes
    // latch it and it never reduces gain again; clearing the detector
    // (field zeroing, no allocation) restores it.
    let loud_gain_reduction = |compressor: &mut Compressor| {
        let mut loud = [0.9_f32; 480];
        compressor.process_block_inplace(&mut loud);
        assert!(loud.iter().all(|s| s.is_finite()));
        compressor.current_gain_reduction()
    };
    // The release path after the loud block: gain reduction must fall back
    // smoothly, which a NaN release time or coefficient prevents.
    let release_trace = |compressor: &mut Compressor| -> Vec<f64> {
        (0..20)
            .map(|_| {
                let mut quiet = [0.001_f32; 480];
                compressor.process_block_inplace(&mut quiet);
                assert!(quiet.iter().all(|s| s.is_finite()));
                compressor.current_gain_reduction()
            })
            .collect()
    };
    let mut compressor = Compressor::default_voice(48_000.0);
    compressor.set_adaptive_release(true);
    poison(&mut block);
    compressor.process_block_inplace(&mut block);
    let latched = loud_gain_reduction(&mut compressor);
    assert!(
        latched.is_nan() || latched <= 0.0,
        "premise: a latched NaN stops gain reduction, got {latched}"
    );
    // The adaptive release never sees the NaN: the detector reads a NaN
    // envelope as its -200 dB floor.
    assert!(compressor.current_release_time().is_finite());
    poison(&mut block);
    compressor.process_block_inplace(&mut block);
    crate::test_alloc::assert_no_allocations("compressor detector reset", || {
        compressor.reset_detector_state();
    });
    let reduced = loud_gain_reduction(&mut compressor);
    assert!(reduced > 0.0);
    let release = release_trace(&mut compressor);
    assert!(compressor.current_release_time().is_finite());
    assert!(
        release.windows(2).all(|pair| pair[1] <= pair[0]) && release[19] < 0.5 * reduced,
        "the release must fall back from {reduced} dB: {release:?}"
    );

    // The limiter sheds a NaN without help.
    let mut limiter = Limiter::default_settings(48_000.0);
    for _ in 0..2 {
        poison(&mut block);
        limiter.process_block_inplace(&mut block);
        let mut clean = [0.1_f32; 64];
        limiter.process_block_inplace(&mut clean);
        assert!(clean.iter().all(|s| s.is_finite()));
    }

    let coeff = smoothing_coeff_for_time_constant(48_000.0, 100.0);
    let mut meter = MeterState::default();
    poison(&mut poisoned);
    update_meter_block_stats(&poisoned, &mut meter, coeff);
    assert!(
        meter.rms_acc.is_nan(),
        "premise: an unsanitized NaN latches the meter"
    );
    let mut meter = MeterState::default();
    contain_non_finite(&mut poisoned);
    let stats = update_meter_block_stats(&poisoned, &mut meter, coeff);
    assert!(meter.rms_acc.is_finite() && stats.rms_db.is_finite());

    // The chain must contain, count and clear before anything measures.
    let source = include_str!("dsp_loop.rs");
    let chain = marked_region(source, "dsp_downstream_chain");
    let contain = chain.find("if contain_non_finite(buffer)").unwrap();
    let tail = &chain[contain..];
    let tail: String = tail.split_whitespace().collect();
    for step in [
        "self.deesser.reset()",
        "self.correction_eq.reset()",
        "self.tone_eq.reset()",
        "self.compressor.reset_detector_state()",
        "downstream_non_finite_count.fetch_add",
    ] {
        assert!(tail.contains(step), "{step} missing from the NaN recovery");
    }
    assert_eq!(AudioProcessor::new().get_downstream_non_finite_count(), 0);
}

// --- RT-009 ---------------------------------------------------------------------

#[test]
fn input_ring_is_sized_from_the_device_rate_after_the_device_is_selected() {
    let production = dsp_loop_production();
    let start = source_between(
        &production,
        "fn start_streams_unguarded(",
        "/// Stop audio processing",
    );
    assert!(!start.contains("AudioRingBuffer::new(self.sample_rate as usize * 2)"));
    let open_input = source_between(start, "fn open_input(", "fn open_output(");
    let select = open_input
        .find("from_named_device_identity_setup(")
        .unwrap();
    let ring = open_input.find("self.input_ring(sample_rate)").unwrap();
    let open = open_input.find("AudioInput::from_setup(").unwrap();
    assert!(select < ring && ring < open);
    let input_ring = source_between(start, "fn input_ring(", "fn output_ring(");
    assert!(input_ring.contains("AudioRingBuffer::new(sample_rate as usize * 2)"));
}

// --- RT-013 --------------------------------------------------------------------

#[test]
fn no_denormals_safety_comment_states_the_documented_caveat() {
    let source = include_str!("dsp_loop.rs");
    let comment = source_between(
        source,
        "// SAFETY: `no_denormals`",
        "let dsp_loop = || unsafe",
    );
    assert!(comment.contains("technically undefined"));
    assert!(comment.contains("restores them when the closure returns or"));
}

// --- RT-020 ---------------------------------------------------------------------

#[test]
fn output_rates_beyond_the_scratch_capacity_are_rejected_at_start() {
    for (rate, fits) in [
        (96_000, true),
        (192_000, true),
        (352_800, false),
        (384_000, false),
    ] {
        let resampler = build_sinc_resampler(48_000, rate, RESAMPLER_CHUNK_SIZE).unwrap();
        let checked =
            ensure_resampler_scratch_capacity(&resampler, RT_OUTPUT_SCRATCH_CAPACITY, "output");
        assert_eq!(checked.is_ok(), fits, "{rate} Hz: {checked:?}");
        if let Err(message) = checked {
            assert!(message.contains("output"));
        }
    }
    let writer = marked_region(include_str!("dsp_loop.rs"), "dsp_output_writer");
    assert!(
        writer
            .split_whitespace()
            .collect::<String>()
            .contains("output.resampler_error_count"),
        "output resampler errors must be counted"
    );
    assert_eq!(AudioProcessor::new().get_output_resampler_error_count(), 0);
}

// --- RT-021 -----------------------------------------------------------------------

#[test]
fn input_fill_smoothing_follows_wall_time_not_pass_count() {
    let settle = |pass_ms: f32| {
        let mut state = 0.0_f32;
        let mut elapsed = 0.0_f32;
        while elapsed < 200.0 - 1e-3 {
            advance_time_ema(&mut state, 1000.0, pass_ms, 200.0);
            elapsed += pass_ms;
        }
        state
    };
    let expected = 1000.0 * (1.0 - (-1.0_f32).exp());
    for pass_ms in [0.1_f32, 1.6, 10.0, 20.0] {
        let value = settle(pass_ms);
        assert!(
            (value - expected).abs() < 2.0,
            "{pass_ms} ms passes gave {value}"
        );
    }
    let rt = marked_region(include_str!("dsp_loop.rs"), "dsp_processing_loop");
    let compact: String = rt.split_whitespace().collect();
    assert!(!compact.contains("smooth_buffer(raw_input_len"));
    assert!(compact.contains("advance_time_ema(&mutself.smoothed_input_len"));
}

// --- RT-025 -------------------------------------------------------------------------

#[test]
fn control_apply_and_stage_resets_do_not_allocate() {
    let sample_rate = TARGET_SAMPLE_RATE as f64;
    let mut gate = NoiseGate::new(-40.0, 10.0, 100.0, sample_rate);
    let mut correction_eq = ParametricEQ::new(sample_rate);
    let mut tone_eq = ParametricEQ::new(sample_rate);
    let mut compressor = Compressor::default_voice(sample_rate);
    let mut deesser = DeEsser::new(sample_rate);
    let mut limiter = Limiter::default_settings(sample_rate);
    let mut suppressor = new_noise_suppression_engine(
        NoiseModel::RNNoise,
        Arc::new(AtomicU32::new(1.0_f32.to_bits())),
    );
    let gate_control = GateControlState::new();
    let eq_control = EqControlSnapshot::new();
    let compressor_control = CompressorControlState::new();
    let deesser_control = DeesserControlState::new();
    let limiter_control = LimiterControlState::new();
    let suppressor_control = SuppressorControlState::new();
    let mut block = [0.1_f32; 480];
    for (index, sample) in block.iter_mut().enumerate() {
        *sample = (index as f32 * 0.037).sin() * 0.35;
    }

    // One pass through every path warms any lazily built state.
    gate.process_block_inplace(&mut block);
    correction_eq.process_block_inplace(&mut block);
    compressor.process_block_inplace(&mut block);
    deesser.process_block_inplace(&mut block);
    limiter.process_block_inplace(&mut block);
    let mut apply = || {
        apply_gate_control(&mut gate, &gate_control);
        apply_eq_control(&mut correction_eq, &mut tone_eq, &eq_control);
        apply_compressor_control(&mut compressor, &compressor_control);
        apply_deesser_control(&mut deesser, &deesser_control);
        apply_limiter_control(&mut limiter, &limiter_control);
        apply_suppressor_control(&mut suppressor, &suppressor_control);
    };
    apply();
    crate::test_alloc::assert_no_allocations("steady-state apply_*_control", apply);

    let counted = |function: &mut dyn FnMut()| crate::test_alloc::allocation_count_during(function);
    assert_eq!(counted(&mut || gate.reset()), 0, "gate reset");
    assert_eq!(counted(&mut || correction_eq.reset()), 0, "eq reset");
    assert_eq!(counted(&mut || deesser.reset()), 0, "deesser reset");
    assert_eq!(counted(&mut || limiter.reset()), 0, "limiter reset");
    assert_eq!(
        counted(&mut || suppressor.soft_reset()),
        0,
        "suppressor soft_reset"
    );
}

// RD-003: `Compressor::reset` rebuilds its EBU R128 meter (9 allocations), and the
// DSP loop calls it on every processing-path change. The fix is in src/dsp.
#[test]
#[ignore = "RD-003: Compressor::reset allocates; src/dsp/compressor is outside this change"]
fn compressor_reset_does_not_allocate() {
    let mut compressor = Compressor::default_voice(TARGET_SAMPLE_RATE as f64);
    compressor.reset();
    assert_eq!(
        crate::test_alloc::allocation_count_during(|| compressor.reset()),
        0
    );
}

#[test]
fn dsp_loop_meters_cleanup_and_sanitizers_do_not_allocate() {
    let mut block = [0.0_f32; 480];
    for (index, sample) in block.iter_mut().enumerate() {
        *sample = (index as f32 * 0.037).sin() * 0.35;
    }
    block[9] = 1.5;
    let coeff = smoothing_coeff_for_time_constant(48_000.0, 100.0);
    let mut meter = MeterState::default();
    let clip_count = AtomicU64::new(0);
    let clip_peak = AtomicU32::new((-120.0_f32).to_bits());
    let mut pre_state = InputPreFilterState::default();
    let mut pre_filter = new_input_pre_filter(48_000.0);
    let mut cleanup = AdaptiveInputCleanupState::new(48_000.0);
    cleanup.set_mode(InputCleanupMode::Gentle);
    let long: Vec<f32> = (0..cleanup.hum_window_samples * 2)
        .map(|n| 0.05 * (2.0 * std::f32::consts::PI * 50.0 * n as f32 / 48_000.0).sin())
        .collect();

    let mut run = |block: &mut [f32; 480]| {
        block[7] = f32::NAN;
        sanitize_and_clamp_input_inplace(block, &clip_count, &clip_peak);
        block[7] = f32::NAN;
        sanitize_and_clamp_output_inplace(block, 0.9);
        apply_input_pre_filter(block, &mut pre_state, &mut pre_filter, true);
        update_meter_block_stats(block, &mut meter, coeff);
        cleanup.analyze_input(&long);
        cleanup.process_block(block);
    };
    // Warm-up pass, then the measured pass over every listed function.
    run(&mut block);
    crate::test_alloc::assert_no_allocations("meters, cleanup and sanitizers", || run(&mut block));

    let source = include_str!("dsp_loop.rs");
    for (label, start, end) in [
        (
            "contain_non_finite",
            "fn contain_non_finite(",
            "/// Advance an exponential",
        ),
        (
            "advance_time_ema",
            "fn advance_time_ema(",
            "/// Keep why the DSP thread died",
        ),
        (
            "startup_control",
            "fn startup_control<",
            "/// Consume a pending limiter",
        ),
    ] {
        assert_rt_clean(label, source_between(source, start, end));
    }
    let routing = include_str!("routing.rs");
    assert_rt_clean(
        "routing.rs meters",
        source_between(
            routing,
            "fn update_meter_block_stats(",
            "fn record_level_extremes(",
        ),
    );
    assert_rt_clean(
        "routing.rs sanitizers",
        source_between(
            routing,
            "fn sanitize_and_clamp_output_inplace(",
            "fn new_input_pre_filter(",
        ),
    );
    assert_rt_clean(
        "routing.rs adaptive cleanup",
        source_between(routing, "fn analyze_input(", "fn wrap_phase("),
    );
    let control = include_str!("control.rs");
    assert_rt_clean(
        "control.rs apply_*_control",
        source_between(
            control,
            "fn apply_eq_control(",
            "#[cfg(test)]\nmod compressor_control_tests",
        ),
    );
}

// --- RT-027 ---------------------------------------------------------------------------

#[cfg(feature = "vad")]
#[test]
fn vad_load_retries_back_off_and_the_last_error_is_published() {
    let mut retry = VadRetryBackoff::default();
    let delays: Vec<u64> = (0..9)
        .map(|_| retry.next_delay().as_millis() as u64)
        .collect();
    assert_eq!(delays, [50, 100, 200, 400, 800, 1_600, 3_200, 5_000, 5_000]);
    retry.reset();
    assert_eq!(retry.next_delay().as_millis(), 50);

    let slot = Mutex::new(None);
    let error = crate::dsp::vad::VadError::InvalidSampleRate(44_100);
    publish_vad_worker_error(&slot, Some(error.to_string()));
    assert_eq!(
        slot.lock().unwrap().as_deref(),
        Some("Unsupported VAD sample rate: 44100 Hz")
    );
    publish_vad_worker_error(&slot, None);
    assert!(AudioProcessor::new().get_vad_worker_error().is_none());

    let worker = include_str!("vad_worker.rs");
    assert!(
        !worker.contains("from_millis(50)"),
        "fixed 50 ms retry loops are gone"
    );
}

#[cfg(feature = "vad")]
#[test]
fn a_stopped_vad_worker_does_not_wait_out_its_maximum_back_off() {
    let running = Arc::new(AtomicBool::new(true));
    let mut retry = VadRetryBackoff::default();
    let max_delay = (0..9).map(|_| retry.next_delay()).last().unwrap();
    assert_eq!(max_delay, Duration::from_millis(5_000));

    let worker_running = Arc::clone(&running);
    let worker = std::thread::spawn(move || sleep_while_running(&worker_running, max_delay));
    std::thread::sleep(Duration::from_millis(30));
    let started = Instant::now();
    running.store(false, Ordering::Release);
    worker.join().unwrap();
    assert!(
        started.elapsed() < Duration::from_millis(100),
        "stop waited {:?}",
        started.elapsed()
    );

    // An unmolested back-off still waits its full delay.
    let started = Instant::now();
    sleep_while_running(&AtomicBool::new(true), Duration::from_millis(60));
    assert!(started.elapsed() >= Duration::from_millis(60));
}

// --- RB-012 -----------------------------------------------------------------------------

#[test]
fn processor_keeps_no_write_only_gate_shadow() {
    let source = include_str!("../processor.rs");
    let fields = source_between(
        source,
        "pub struct AudioProcessor {",
        "impl AudioProcessor {",
    );
    assert!(!fields.contains("gate: Arc<Mutex<NoiseGate>>"));
    assert!(!include_str!("shared.rs").contains("Mutex<NoiseGate>"));
    let stop = source_between(include_str!("lifecycle.rs"), "pub fn stop(", "\n    }\n}\n");
    assert!(!stop.contains("self.gate.lock()"));
}
