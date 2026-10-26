//! The worker-backed processor against the inline one, on the stand-in
//! library of `fake`. The stand-in halves a frame, so every output value says
//! which input produced it: `0.5 * input` for a processed frame, the input of
//! the frame before it for a frame passed through dry (the model's one-frame
//! delay is kept), and 0 for the frame a reset leaves invalid.

use super::fake::{inline_processor, worker_processor, Fake};
use super::worker::{TestFault, WORKER_QUEUE_FRAMES};
use super::*;
use std::time::{Duration, Instant};

const FRAME: usize = DEEPFILTER_FRAME_SIZE;
const LL: DeepFilterModel = DeepFilterModel::LowLatency;

fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !condition() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn drain(processor: &mut DeepFilterProcessor, output: &mut Vec<f32>) {
    let mut block = [0.0_f32; FRAME];
    while processor.available_samples() > 0 {
        let count = processor.pop_samples_into(&mut block);
        output.extend_from_slice(&block[..count]);
    }
}

/// Input value of frame `n` in the constant-frame tests.
fn level(n: usize) -> f32 {
    0.1 * (n + 1) as f32
}

fn push_level(processor: &mut DeepFilterProcessor, n: usize) {
    assert_eq!(processor.push_samples(&[level(n); FRAME]), FRAME);
    processor.process_frames();
}

/// Call `process_frames` until `frames` frames of output have come out.
fn collect_frames(processor: &mut DeepFilterProcessor, frames: usize) -> Vec<f32> {
    let mut output = Vec::new();
    wait_until("the output", || {
        processor.process_frames();
        drain(processor, &mut output);
        output.len() >= frames * FRAME
    });
    assert_eq!(output.len(), frames * FRAME);
    output
}

fn assert_frame(output: &[f32], n: usize, expected: f32) {
    let frame = &output[n * FRAME..(n + 1) * FRAME];
    assert!(
        frame.iter().all(|sample| *sample == expected),
        "frame {n} is {:?}, expected {expected}",
        &frame[..3]
    );
}

fn link(processor: &DeepFilterProcessor) -> &worker::WorkerLink {
    processor.link.as_ref().expect("a worker-backed processor")
}

/// Feed `input` in blocks of the given sizes and return everything that came
/// out. The worker answers each block before the next arrives, which keeps the
/// run independent of its speed; inline that wait is a no-op.
fn render(
    processor: &mut DeepFilterProcessor,
    fake: &Fake,
    input: &[f32],
    controls: &[GateControl],
    blocks: &[usize],
) -> Vec<f32> {
    let mut output = Vec::new();
    let mut pushed = 0;
    for &block in blocks.iter().cycle() {
        if pushed == input.len() {
            break;
        }
        let end = (pushed + block).min(input.len());
        assert_eq!(
            processor.push_controlled_samples(&input[pushed..end], &controls[pushed..end]),
            end - pushed
        );
        pushed = end;
        let fail_from = fake.fail_from.load(Ordering::Acquire) as usize;
        let succeeded = (pushed / FRAME).min(fail_from);
        // A block longer than the worker's queue is offered over several calls.
        wait_until("the worker", || {
            processor.process_frames();
            drain(processor, &mut output);
            processor.successful_inference_frames() as usize >= succeeded
                && started(fake) >= (pushed / FRAME).min(fail_from.saturating_add(1))
        });
        // Every pushed sample is out, or still pending.
        assert_eq!(
            pushed,
            output.len() + processor.available_samples() + processor.pending_input()
        );
    }
    let complete = pushed / FRAME * FRAME;
    wait_until("the last frames", || {
        processor.process_frames();
        drain(processor, &mut output);
        output.len() >= complete
    });
    assert_eq!(output.len(), complete);
    output
}

fn assert_same_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "sample {index}: {actual} against {expected}"
        );
    }
}

#[test]
fn worker_output_is_bit_identical_to_the_inline_path() {
    // No tolerance: the worker runs the same code on the same frames in the
    // same order as the inline path, so the outputs match to the bit. Only the
    // time at which a frame comes out differs.
    let input: Vec<f32> = (0..FRAME * 37 + 123)
        .map(|i| (i as f32 * 0.031).sin() * 0.3 + (i as f32 * 0.0007).cos() * 0.1)
        .collect();
    let controls: Vec<GateControl> = (0..input.len())
        .map(|i| GateControl {
            gain: 0.02 + 0.98 * (0.5 + 0.5 * (i as f32 * 0.0011).sin()),
            open: (i / 700) % 3 != 0,
        })
        .collect();
    let blocks = [1000, 137, 480, 2500, 2];
    for model in [LL, DeepFilterModel::Standard] {
        for strength in [1.0, 0.37] {
            let (inline_fake, worker_fake) = (Fake::new(), Fake::new());
            let mut inline = inline_processor(&inline_fake, model, strength);
            let mut worker = worker_processor(&worker_fake, model, strength);
            let expected = render(&mut inline, &inline_fake, &input, &controls, &blocks);
            let actual = render(&mut worker, &worker_fake, &input, &controls, &blocks);

            assert_eq!(expected.len(), 37 * FRAME);
            assert_same_bits(&actual, &expected);
            assert!(actual.iter().any(|sample| *sample != 0.0));
            assert_eq!(worker.pending_input(), input.len() % FRAME);
            assert_eq!(worker.successful_inference_frames(), 37);
            assert!(!worker.backend_failed() && worker.backend_error().is_none());
            assert_eq!(worker.latency_samples(), inline.latency_samples());
            assert_eq!(link(&worker).owed_results(), 0);
        }
    }
}

#[test]
fn a_failing_frame_degrades_exactly_like_the_inline_path() {
    let input: Vec<f32> = (0..FRAME * 12)
        .map(|i| (i as f32 * 0.05).sin() * 0.2)
        .collect();
    let controls = vec![GateControl::BYPASS; input.len()];
    for fail_from in [0_u32, 2, 5] {
        let (inline_fake, worker_fake) = (Fake::new(), Fake::new());
        inline_fake.fail_from.store(fail_from, Ordering::Release);
        worker_fake.fail_from.store(fail_from, Ordering::Release);
        let mut inline = inline_processor(&inline_fake, LL, 0.5);
        let mut worker = worker_processor(&worker_fake, LL, 0.5);
        let expected = render(&mut inline, &inline_fake, &input, &controls, &[700]);
        let actual = render(&mut worker, &worker_fake, &input, &controls, &[700]);

        assert_same_bits(&actual, &expected);
        assert!(actual.iter().all(|sample| sample.is_finite()));
        for processor in [&inline, &worker] {
            assert!(processor.backend_failed() && !processor.is_ffi_available());
            assert_eq!(
                processor.backend_error(),
                Some("DeepFilterNet processing failed with non-finite SNR")
            );
            assert_eq!(processor.successful_inference_frames(), fail_from as u64);
            assert_eq!(processor.latency_samples(), 480);
        }
    }
}

#[test]
fn a_non_finite_frame_from_the_library_is_refused_like_the_inline_path() {
    let (inline_fake, worker_fake) = (Fake::new(), Fake::new());
    inline_fake.non_finite_from.store(3, Ordering::Release);
    worker_fake.non_finite_from.store(3, Ordering::Release);
    let mut inline = inline_processor(&inline_fake, LL, 1.0);
    let mut worker = worker_processor(&worker_fake, LL, 1.0);
    for n in 0..6 {
        push_level(&mut inline, n);
        push_level(&mut worker, n);
        let frames = n + 1;
        wait_until("the worker", || started(&worker_fake) >= frames.min(4));
    }
    let expected = collect_frames(&mut inline, 6);
    let actual = collect_frames(&mut worker, 6);

    assert_same_bits(&actual, &expected);
    assert!(actual.iter().all(|sample| sample.is_finite()));
    assert_eq!(
        worker.backend_error(),
        Some("DeepFilterNet processing produced non-finite output")
    );
    assert_eq!(worker.successful_inference_frames(), 3);
}

fn started(fake: &Fake) -> usize {
    fake.started.load(Ordering::Acquire) as usize
}

#[test]
fn a_worker_that_dies_or_panics_falls_back_to_dry_passthrough_and_says_why() {
    for (fault, reason) in [
        (TestFault::Panic, "DeepFilterNet worker panicked"),
        (TestFault::Exit, "DeepFilterNet worker stopped unexpectedly"),
    ] {
        let fake = Fake::new();
        let mut processor = worker_processor(&fake, LL, 1.0);
        for n in 0..2 {
            push_level(&mut processor, n);
            wait_until("the worker", || {
                processor.successful_inference_frames() as usize > n
            });
        }
        link(&processor).worker().inject_fault(fault);
        for n in 2..7 {
            push_level(&mut processor, n);
        }
        let output = collect_frames(&mut processor, 7);

        for n in 0..2 {
            assert_frame(&output, n, level(n) * 0.5);
        }
        for n in 2..7 {
            assert_frame(&output, n, level(n - 1));
        }
        assert!(processor.backend_failed() && !processor.is_ffi_available());
        assert_eq!(processor.backend_error(), Some(reason));
        assert_eq!(processor.successful_inference_frames(), 2);
        assert_eq!(processor.pending_input(), 0);
    }
}

#[test]
fn a_worker_that_stops_answering_is_declared_stalled_and_never_blocks_the_caller() {
    let fake = Fake::new();
    fake.hold.store(true, Ordering::Release);
    let mut processor = worker_processor(&fake, LL, 1.0);
    let begun = Instant::now();
    for n in 0..14 {
        push_level(&mut processor, n);
    }
    assert!(
        begun.elapsed() < Duration::from_secs(2),
        "the calls waited for the stalled worker"
    );
    let output = collect_frames(&mut processor, 14);

    assert!(processor.backend_failed() && !processor.is_ffi_available());
    assert_eq!(
        processor.backend_error(),
        Some("DeepFilterNet worker stopped answering")
    );
    assert!(output.iter().all(|sample| sample.is_finite()));
    // Nothing was lost: every frame is out as the dry input, one frame late.
    assert_frame(&output, 0, 0.0);
    for n in 1..14 {
        assert_frame(&output, n, level(n - 1));
    }
    assert_eq!(processor.pending_input(), 0);
    fake.hold.store(false, Ordering::Release);
}

#[test]
fn a_worker_that_falls_behind_makes_the_frames_wait_and_loses_none() {
    let fake = Fake::new();
    fake.hold.store(true, Ordering::Release);
    let mut processor = worker_processor(&fake, LL, 1.0);
    let frames = WORKER_QUEUE_FRAMES + 1;
    for n in 0..frames {
        push_level(&mut processor, n);
    }
    // The frames wait where they are, so the time they spend at the worker is
    // part of the pending input (and of the published latency); the worker owes
    // its maximum and the last frame waits for a slot.
    assert_eq!(processor.pending_input(), frames * FRAME);
    assert_eq!(processor.latency_samples(), 480);
    assert_eq!(processor.available_samples(), 0);
    assert_eq!(link(&processor).owed_results(), WORKER_QUEUE_FRAMES);
    assert!(!processor.backend_failed() && processor.backend_error().is_none());
    assert!(processor.is_ffi_available());

    fake.hold.store(false, Ordering::Release);
    let output = collect_frames(&mut processor, frames);

    // Every frame is processed, including the one that had to wait.
    for n in 0..frames {
        assert_frame(&output, n, level(n) * 0.5);
    }
    assert!(!processor.backend_failed());
    assert_eq!(processor.pending_input(), 0);
    assert_eq!(link(&processor).owed_results(), 0);
}

#[test]
fn a_block_longer_than_the_workers_queue_waits_instead_of_going_dry() {
    // What the loop reads after a delay: one frame more than the queue holds.
    // (With ten or more frames waiting, a worker starved for 100 ms on an
    // overloaded machine would count as stalled.)
    let frames = WORKER_QUEUE_FRAMES + 1;
    let input: Vec<f32> = (0..FRAME * frames + 40)
        .map(|i| (i as f32 * 0.037).sin() * 0.3)
        .collect();
    let controls: Vec<GateControl> = (0..input.len())
        .map(|i| GateControl {
            gain: 0.02 + 0.98 * (0.5 + 0.5 * (i as f32 * 0.0013).sin()),
            open: (i / 600) % 3 != 0,
        })
        .collect();
    let (inline_fake, worker_fake) = (Fake::new(), Fake::new());
    let mut inline = inline_processor(&inline_fake, LL, 0.4);
    let mut worker = worker_processor(&worker_fake, LL, 0.4);
    let expected = render(&mut inline, &inline_fake, &input, &controls, &[input.len()]);

    assert_eq!(
        worker.push_controlled_samples(&input, &controls),
        input.len()
    );
    worker.process_frames();
    assert_eq!(link(&worker).owed_results(), WORKER_QUEUE_FRAMES);
    assert_eq!(worker.pending_input(), input.len());
    let actual = collect_frames(&mut worker, frames);

    // The same audio, only later: nothing went through dry.
    assert_same_bits(&actual, &expected);
    assert_eq!(worker.successful_inference_frames(), frames as u64);
    assert!(!worker.backend_failed() && worker.backend_error().is_none());
    assert_eq!(worker.pending_input(), input.len() % FRAME);
}

#[test]
fn a_dead_worker_is_found_when_the_input_queue_overflows() {
    // The loop answers an overflow by resetting the queue, which would restart
    // the worker's silence clock; large blocks reach the overflow before they
    // reach another `process_frames`, so the push itself has to notice.
    let fake = Fake::new();
    fake.hold.store(true, Ordering::Release);
    let mut processor = worker_processor(&fake, LL, 1.0);
    for n in 0..16 {
        assert_eq!(processor.push_samples(&[level(n); FRAME]), FRAME);
    }
    processor.process_frames();
    assert!(!processor.backend_failed());
    std::thread::sleep(Duration::from_millis(150));

    let accepted = processor.push_samples(&[0.1; 4 * FRAME]);
    assert!(accepted < 4 * FRAME, "the queue was meant to overflow");
    assert!(processor.backend_failed());
    assert_eq!(
        processor.backend_error(),
        Some("DeepFilterNet worker stopped answering")
    );

    processor.soft_reset();
    assert!(processor.backend_failed());
    push_level(&mut processor, 0);
    assert_eq!(collect_frames(&mut processor, 1).len(), FRAME);
    fake.hold.store(false, Ordering::Release);
}

/// Three frames are at the worker (it holds the first) when `interrupt` runs;
/// afterwards the worker answers them, and then a fresh stream is fed.
fn run_interruption(
    interrupt: impl FnOnce(&mut DeepFilterProcessor),
    pending_after: usize,
) -> Vec<f32> {
    let fake = Fake::new();
    fake.hold.store(true, Ordering::Release);
    let mut processor = worker_processor(&fake, LL, 1.0);
    for _ in 0..3 {
        assert_eq!(processor.push_samples(&[9.0; FRAME]), FRAME);
        processor.process_frames();
    }
    interrupt(&mut processor);
    assert_eq!(processor.pending_input(), pending_after);
    assert_eq!(processor.available_samples(), 0);
    fake.hold.store(false, Ordering::Release);
    wait_until("the stale frames", || {
        processor.successful_inference_frames() == 3
    });
    for n in 0..4 {
        push_level(&mut processor, n);
    }
    let output = collect_frames(&mut processor, 4 + pending_after / FRAME);
    assert_eq!(processor.pending_input(), 0);
    wait_until("the stale results to be dropped", || {
        processor.process_frames();
        link(&processor).owed_results() == 0
    });
    output
}

#[test]
fn a_reset_drops_the_frames_at_the_worker_and_their_results() {
    let output = run_interruption(|processor| processor.soft_reset(), 0);
    // The first frame after a reset is the model's invalid overlap; nothing of
    // the 9.0 frames the worker answered after the reset appears.
    assert_frame(&output, 0, 0.0);
    for n in 1..4 {
        assert_frame(&output, n, level(n) * 0.5);
    }
}

#[test]
fn disabling_and_enabling_never_replays_or_loses_a_frame() {
    // The frames stay queued across the toggle and are processed once, after
    // it; the answers the worker owed before it are dropped.
    let output = run_interruption(
        |processor| {
            processor.set_enabled(false);
            processor.set_enabled(true);
        },
        3 * FRAME,
    );
    assert_frame(&output, 0, 0.0);
    assert_frame(&output, 1, 9.0 * 0.5);
    assert_frame(&output, 2, 9.0 * 0.5);
    for n in 0..4 {
        assert_frame(&output, 3 + n, level(n) * 0.5);
    }
}

#[test]
fn the_dsp_side_of_worker_mode_does_not_allocate() {
    let fake = Fake::new();
    let mut processor = worker_processor(&fake, LL, 0.5);
    let block = [0.1_f32; 1000];
    let controls = [GateControl::BYPASS; 1000];
    let mut output = [0.0_f32; FRAME];
    let mut run = |processor: &mut DeepFilterProcessor, rounds: usize| {
        for _ in 0..rounds {
            processor.push_controlled_samples(&block, &controls);
            processor.process_frames();
            while processor.pop_samples_into(&mut output) > 0 {}
            processor.soft_reset();
        }
    };
    run(&mut processor, 5);

    // The rounds follow each other faster than the worker answers, so they
    // cover frames waiting for a slot and the abandoned results.
    crate::test_alloc::assert_no_allocations("worker mode, healthy", || run(&mut processor, 300));

    link(&processor).worker().inject_fault(TestFault::Exit);
    wait_until("the worker to stop", || {
        link(&processor).worker().failure().is_some()
    });
    crate::test_alloc::assert_no_allocations("worker mode, failed", || run(&mut processor, 50));
    assert!(processor.backend_failed());
}

#[test]
fn dropping_an_engine_whose_worker_is_in_a_slow_frame_returns_at_once() {
    let fake = Fake::new();
    fake.hold.store(true, Ordering::Release);
    let mut processor = worker_processor(&fake, LL, 1.0);
    push_level(&mut processor, 0);
    wait_until("the library to be entered", || started(&fake) == 1);

    // On another thread, so that a drop that waits fails the test instead of
    // hanging it.
    let (dropped, wait_for_drop) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        drop(processor);
        let _ = dropped.send(());
    });
    wait_for_drop
        .recv_timeout(Duration::from_secs(2))
        .expect("the drop waited for the worker's slow frame");
    // The frame is still being processed: the drop did not wait for it.
    assert!(!fake.freed.load(Ordering::Acquire));

    // Once the frame ends the worker stops and frees the library state.
    fake.hold.store(false, Ordering::Release);
    wait_until("the library state to be freed", || {
        fake.freed.load(Ordering::Acquire)
    });
}

#[test]
fn dropping_an_idle_worker_frees_the_library_state() {
    let fake = Fake::new();
    let mut processor = worker_processor(&fake, LL, 1.0);
    push_level(&mut processor, 0);
    drop(processor);
    wait_until("the library state to be freed", || {
        fake.freed.load(Ordering::Acquire)
    });
}

#[test]
fn a_retired_engine_ends_its_worker_before_it_is_dropped() {
    let fake = Fake::new();
    let mut engine = super::fake::worker_engine(&fake, LL, 1.0);
    engine.push_samples(&[0.1; FRAME]);
    engine.process_frames();
    engine.retire();
    wait_until("the library state to be freed", || {
        fake.freed.load(Ordering::Acquire)
    });
    drop(engine);
}

/// The real library, where the DeepFilter assets are enabled
/// (`AUDIOFORGE_ENABLE_DEEPFILTER`, `AUDIOFORGE_ALLOW_EXTERNAL_DF`,
/// `DEEPFILTER_LIB_PATH`, `DEEPFILTER_MODEL_PATH`). The inline run uses the
/// floating-point mode the worker does, as the DSP loop did for the inline
/// model; with that, the two agree to the bit.
#[test]
fn the_real_model_gives_the_same_output_at_the_worker_as_inline() {
    const FRAMES: usize = 300;
    let mut seed = 0x2545_f491_u32;
    let input: Vec<f32> = (0..FRAMES * FRAME)
        .map(|i| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (seed >> 8) as f32 / (1 << 24) as f32 - 0.5;
            let voice = (i as f32 * 0.043).sin() * (0.5 + 0.5 * (i as f32 * 0.0004).sin());
            0.2 * voice + 0.05 * noise
        })
        .collect();
    for model in [LL, DeepFilterModel::Standard] {
        let strength = || Arc::new(AtomicU32::new(0.8_f32.to_bits()));
        let mut inline = DeepFilterProcessor::new(strength(), model);
        if !inline.is_ffi_available() {
            eprintln!(
                "SKIPPED: the real {model:?} backend is unavailable: {:?}",
                inline.backend_error()
            );
            return;
        }
        let mut worker = DeepFilterProcessor::new(strength(), model).with_worker();
        assert!(worker.is_ffi_available(), "{:?}", worker.backend_error());

        let mut expected = Vec::new();
        // SAFETY: the same scoped floating-point mode the worker thread runs in,
        // with only stack-owned buffers.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        let inline_time = unsafe {
            no_denormals::no_denormals(|| {
                let begun = Instant::now();
                for block in input.chunks(FRAME) {
                    inline.push_samples(block);
                    inline.process_frames();
                    drain(&mut inline, &mut expected);
                }
                begun.elapsed()
            })
        };

        let mut actual = Vec::new();
        for (index, block) in input.chunks(FRAME).enumerate() {
            worker.push_samples(block);
            worker.process_frames();
            wait_until("the worker", || {
                worker.successful_inference_frames() as usize > index
            });
            drain(&mut worker, &mut actual);
        }
        actual.extend(collect_frames(&mut worker, FRAMES - actual.len() / FRAME));

        eprintln!(
            "{model:?}: {:.2} ms per frame inline",
            inline_time.as_secs_f64() * 1000.0 / FRAMES as f64
        );
        assert_same_bits(&actual, &expected);
        assert!(actual.iter().any(|sample| *sample != 0.0));
        assert_eq!(worker.successful_inference_frames(), FRAMES as u64);
        assert!(!worker.backend_failed());
    }
}

#[test]
fn the_realtime_factory_puts_inference_at_a_worker_and_the_default_stays_inline() {
    use crate::dsp::noise_suppressor::{
        new_controlled_noise_suppression_engine, new_realtime_noise_suppression_engine, NoiseModel,
    };
    let strength = || Arc::new(AtomicU32::new(1.0_f32.to_bits()));
    let mut inline =
        new_controlled_noise_suppression_engine(NoiseModel::DeepFilterNetLL, strength());
    if !inline.backend_available() {
        eprintln!("SKIPPED: the real DeepFilterNet LL backend is unavailable");
        return;
    }
    let mut realtime =
        new_realtime_noise_suppression_engine(NoiseModel::DeepFilterNetLL, strength());
    assert!(realtime.backend_available());
    for engine in [&mut inline, &mut realtime] {
        engine.push_samples(&[0.1; FRAME]);
        engine.process_frames();
    }

    // The inline engine has answered by the time the call returns; the worker's
    // answer is collected by a later call.
    assert_eq!(inline.available_samples(), FRAME);
    assert_eq!(realtime.available_samples(), 0);
    wait_until("the worker's answer", || {
        realtime.process_frames();
        realtime.available_samples() == FRAME
    });
}
