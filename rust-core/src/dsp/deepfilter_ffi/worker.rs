//! Worker thread that runs DeepFilterNet inference for the live DSP loop.
//!
//! The loop must neither run the model nor wait for it. It offers whole frames
//! to this worker through one bounded queue and collects the results from a
//! second; both are allocated before the worker starts, and the loop only
//! pushes, pops and counts. The worker polls its queue with a short sleep, as
//! the VAD worker does, so the loop never has to wake it. It answers every
//! frame in order, so the loop pairs results with frames by position alone.

use super::{DeepFilterFFI, DeepFilterProcessError, DEEPFILTER_FRAME_SIZE};
use crate::audio::buffer::{AudioConsumer, AudioProducer, AudioRingBuffer};
#[cfg(windows)]
use crate::audio::scheduling::ProAudioThread;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Frames the worker may owe the loop at once (10 ms each): enough for one pass
/// to offer the longest device period the opt-in fixed buffer allows (8192
/// samples, 17 frames and a part that the remainder of the pass before completes
/// to 18). It is also the most latency a slow worker can add. Frames beyond it
/// wait in the processor's input queue.
pub(super) const WORKER_QUEUE_FRAMES: usize = 8192_usize.div_ceil(DEEPFILTER_FRAME_SIZE);
const QUEUE_SAMPLES: usize = WORKER_QUEUE_FRAMES * DEEPFILTER_FRAME_SIZE;
/// What the published suppressor latency adds for the time a frame spends at the
/// worker (5 ms): the worker's poll, the model, and the loop's next look on an
/// idle pass. Measured with the real library, input paced in real time, blocks
/// of 128, 256 and 480 samples: from the frame being offered to its output being
/// emitted, DeepFilterNet LL took 4.1 ms on average (95th percentile 4.8, worst
/// 6.2) and the standard model 2.6 ms (4.2, 4.8), whatever the block size. The
/// tools that check the published figure (`health_check.py`,
/// `evaluate_hardware_validation.py`) mirror this value, and a test compares them.
pub(crate) const WORKER_TRANSIT_SAMPLES: usize = 240;
/// A frame that has been at the worker longer than this (a frame's own length)
/// counts as a backlog in the published latency: the healthy transit is half of
/// it at most, and the worker is behind if a frame outlasts the next one.
const WORKER_LATE_AFTER: Duration = Duration::from_millis(10);
/// The worker is stalled once it has owed a result for this long without
/// sending one, as the loop saw it. The loop's backlog recovery keeps 100 ms
/// of input, so a freeze of that length must never look like a stall.
const WORKER_STALL_TIME: Duration = Duration::from_millis(300);
/// What one look at the worker can add to that silence. A pause of the whole
/// process (the machine freezing, a suspend) stops the worker as well as the
/// loop, and it is no evidence against the worker: it counts for no more than
/// this.
const WORKER_STALL_STEP: Duration = Duration::from_millis(50);
/// How often an idle worker looks for a frame.
const POLL_INTERVAL: Duration = Duration::from_millis(1);

#[derive(Default)]
struct WorkerShared {
    /// `DeepFilterProcessError` code of why the worker stopped, 0 while it runs.
    /// Stored once, after the last result it produced is queued.
    failure: AtomicU8,
    stop: AtomicBool,
    inferred_frames: AtomicU64,
    /// A `TestFault` the tests ask for; always 0 in production.
    #[cfg_attr(not(test), allow(dead_code))]
    fault: AtomicU8,
}

/// The loop's end of the worker: its queues and a way to stop it. The thread
/// is detached: it owns the library state and frees it when it ends, so
/// dropping this never waits for the worker, however slow its current frame.
pub(super) struct InferenceWorker {
    shared: Arc<WorkerShared>,
    input: AudioProducer,
    output: AudioConsumer,
}

impl InferenceWorker {
    /// Start a worker that owns `df`. Runs on the control thread, where model
    /// loading already happens.
    pub(super) fn spawn(df: DeepFilterFFI) -> Result<Self, String> {
        let shared = Arc::new(WorkerShared::default());
        let (input, worker_input) = AudioRingBuffer::new(QUEUE_SAMPLES).split();
        let (worker_output, output) = AudioRingBuffer::new(QUEUE_SAMPLES).split();
        let worker_shared = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("deepfilter-worker".to_string())
            .spawn(move || run(df, &worker_shared, worker_input, worker_output))
            .map_err(|error| error.to_string())?;
        Ok(Self {
            shared,
            input,
            output,
        })
    }

    /// Why the worker stopped, once it has. Read this before collecting
    /// results: it is stored after the last result is queued.
    pub(super) fn failure(&self) -> Option<DeepFilterProcessError> {
        DeepFilterProcessError::from_code(self.shared.failure.load(Ordering::Acquire))
    }

    /// Ask the worker to finish its current frame, free the library state and
    /// end. Does not wait for it.
    pub(super) fn stop(&self) {
        self.shared.stop.store(true, Ordering::Release);
    }

    /// Queue one frame. The caller keeps at most `WORKER_QUEUE_FRAMES` frames
    /// unanswered, so there is always room; false means there was none.
    pub(super) fn offer(&mut self, frame: &[f32; DEEPFILTER_FRAME_SIZE]) -> bool {
        self.input.write(frame) == DEEPFILTER_FRAME_SIZE
    }

    /// Whether a finished frame is waiting.
    pub(super) fn has_result(&self) -> bool {
        self.output.len() >= DEEPFILTER_FRAME_SIZE
    }

    /// Take the oldest finished frame.
    pub(super) fn take(&mut self, frame: &mut [f32; DEEPFILTER_FRAME_SIZE]) -> bool {
        self.has_result() && self.output.read(frame) == DEEPFILTER_FRAME_SIZE
    }

    pub(super) fn inferred_frames(&self) -> u64 {
        self.shared.inferred_frames.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(super) fn inject_fault(&self, fault: TestFault) {
        self.shared.fault.store(fault as u8, Ordering::Release);
    }
}

impl Drop for InferenceWorker {
    fn drop(&mut self) {
        self.stop();
    }
}

/// The bookkeeping of the frames handed to the worker. The frames themselves
/// stay in the processor's input queue until their result is back, so nothing
/// is ever lost between the two.
pub(super) struct WorkerLink {
    worker: InferenceWorker,
    /// Frames at the front of the input queue already handed to the worker.
    offered: usize,
    /// Results the worker still owes for frames a reset dropped. They come
    /// first, and are discarded.
    stale: usize,
    /// How long the worker has owed a result without sending one, as seen in
    /// the loop's looks, and when it last looked.
    silent_for: Duration,
    last_look: Option<Instant>,
    /// When each owed result was asked for, oldest first from `owed_head`.
    owed_since: [Instant; WORKER_QUEUE_FRAMES],
    owed_head: usize,
    /// `WORKER_LATE_AFTER` unless a test sets its own. Unconditional: a
    /// `#[cfg(test)]` field would blank this impl out of the RT source scan.
    late_after: Duration,
}

impl WorkerLink {
    pub(super) fn new(worker: InferenceWorker) -> Self {
        Self {
            worker,
            offered: 0,
            stale: 0,
            silent_for: Duration::ZERO,
            last_look: None,
            owed_since: [Instant::now(); WORKER_QUEUE_FRAMES],
            owed_head: 0,
            late_after: WORKER_LATE_AFTER,
        }
    }

    /// Results the worker owes, stale ones included.
    fn owed(&self) -> usize {
        self.offered + self.stale
    }

    /// A result came back.
    fn heard_from_worker(&mut self) {
        self.forget_silence();
        self.owed_head = (self.owed_head + 1) % WORKER_QUEUE_FRAMES;
    }

    fn forget_silence(&mut self) {
        self.silent_for = Duration::ZERO;
        self.last_look = None;
    }

    /// How many of the results the worker owes were asked for longer ago than
    /// `WORKER_LATE_AFTER`: the frames it is behind with, which the published
    /// latency counts one by one. Frames offered together in one pass are not
    /// late while the worker is working through them.
    pub(super) fn late_frames(&self) -> usize {
        let late_after = self.late_after;
        let now = Instant::now();
        (0..self.owed())
            .take_while(|index| {
                let since = self.owed_since[(self.owed_head + index) % WORKER_QUEUE_FRAMES];
                now.saturating_duration_since(since) > late_after
            })
            .count()
    }

    /// Add the time since the last look to the worker's silence, if it owes a
    /// result and none is waiting to be collected.
    fn look_at_silence(&mut self) {
        if self.owed() == 0 || self.worker.has_result() {
            self.forget_silence();
            return;
        }
        let now = Instant::now();
        if let Some(last) = self.last_look {
            self.silent_for += now.saturating_duration_since(last).min(WORKER_STALL_STEP);
        }
        self.last_look = Some(now);
    }

    /// The frames offered so far no longer matter (the queue was cleared, or
    /// the stage was switched off): their results are discarded on arrival.
    /// The worker's silence goes on, so a reset (an input overflow resets the
    /// queue) does not hide a stall.
    pub(super) fn abandon(&mut self) {
        self.stale += self.offered;
        self.offered = 0;
    }

    pub(super) fn offered(&self) -> usize {
        self.offered
    }

    pub(super) fn has_room(&self) -> bool {
        self.owed() < WORKER_QUEUE_FRAMES
    }

    pub(super) fn inferred_frames(&self) -> u64 {
        self.worker.inferred_frames()
    }

    pub(super) fn stop(&self) {
        self.worker.stop();
    }

    /// Drop the results of abandoned frames that are back. `scratch` takes them.
    fn reap(&mut self, scratch: &mut [f32; DEEPFILTER_FRAME_SIZE]) {
        while self.stale > 0 && self.worker.take(scratch) {
            self.stale -= 1;
            self.heard_from_worker();
        }
    }

    /// Why no more results will come, if that is known: the worker stopped, or
    /// it has been silent for `WORKER_STALL_TIME`. Call before `take_head`: the
    /// worker stores its failure after its last result.
    pub(super) fn failure(
        &mut self,
        scratch: &mut [f32; DEEPFILTER_FRAME_SIZE],
    ) -> Option<DeepFilterProcessError> {
        self.reap(scratch);
        if let Some(error) = self.worker.failure() {
            return Some(error);
        }
        self.look_at_silence();
        (self.silent_for >= WORKER_STALL_TIME).then_some(DeepFilterProcessError::WorkerStalled)
    }

    /// Take the result of the oldest offered frame into `wet`. False while it
    /// is not back, or when no frame has been offered.
    pub(super) fn take_head(&mut self, wet: &mut [f32; DEEPFILTER_FRAME_SIZE]) -> bool {
        if self.offered == 0 {
            return false;
        }
        self.reap(wet);
        if self.stale > 0 || !self.worker.take(wet) {
            return false;
        }
        self.offered -= 1;
        self.heard_from_worker();
        true
    }

    /// Hand the next complete frame to the worker. False when it already owes
    /// `WORKER_QUEUE_FRAMES` results: the frame waits for the next call.
    pub(super) fn offer(&mut self, samples: &[f32; DEEPFILTER_FRAME_SIZE]) -> bool {
        if !self.has_room() || !self.worker.offer(samples) {
            return false;
        }
        self.owed_since[(self.owed_head + self.owed()) % WORKER_QUEUE_FRAMES] = Instant::now();
        self.offered += 1;
        true
    }

    #[cfg(test)]
    pub(super) fn worker(&self) -> &InferenceWorker {
        &self.worker
    }

    /// Count a frame late after `late_after` instead of the usual time.
    #[cfg(test)]
    pub(super) fn set_late_after(&mut self, late_after: Duration) {
        self.late_after = late_after;
    }

    /// Results the worker still owes, stale ones included.
    #[cfg(test)]
    pub(super) fn owed_results(&self) -> usize {
        self.owed()
    }
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(super) enum TestFault {
    Panic = 1,
    Exit = 2,
}

/// Records why the worker ended, unless it was asked to stop, so the loop never
/// waits on a worker that no longer answers.
struct Exit<'a>(&'a WorkerShared);

impl Drop for Exit<'_> {
    fn drop(&mut self) {
        if !self.0.stop.load(Ordering::Acquire) {
            let _ = self.0.failure.compare_exchange(
                0,
                DeepFilterProcessError::WorkerStopped.code(),
                Ordering::AcqRel,
                Ordering::Relaxed,
            );
        }
    }
}

fn run(df: DeepFilterFFI, shared: &WorkerShared, input: AudioConsumer, output: AudioProducer) {
    let _exit = Exit(shared);
    // Like the DSP thread, the worker needs the audio scheduling class to keep
    // up with it; without it the worker runs at normal priority, which still
    // works.
    #[cfg(windows)]
    let scheduling = ProAudioThread::enter().ok();
    // The model ran under the DSP loop's floating-point mode and must keep
    // doing so: it is the mode the library's long-silence behaviour was
    // verified in.
    // SAFETY: `no_denormals` sets the FTZ/DAZ bits of this thread's FP control
    // register and restores them when the closure returns or unwinds. Its own
    // documentation calls this technically undefined behaviour, because the
    // compiler may assume the default FP environment; the risk is confined to
    // this thread, which does not rely on exact results of denormal arithmetic.
    // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
    unsafe { no_denormals::no_denormals(|| serve(df, shared, input, output)) };
    #[cfg(windows)]
    if let Some(scheduling) = scheduling {
        let _ = scheduling.finish();
    }
}

/// Answer frames until asked to stop or the library fails. `df` is dropped
/// here, so the library state is freed when the last frame ends.
fn serve(
    mut df: DeepFilterFFI,
    shared: &WorkerShared,
    mut input: AudioConsumer,
    mut output: AudioProducer,
) {
    let mut frame = [0.0_f32; DEEPFILTER_FRAME_SIZE];
    let mut wet = [0.0_f32; DEEPFILTER_FRAME_SIZE];
    while !shared.stop.load(Ordering::Acquire) {
        #[cfg(test)]
        if shared.fault.load(Ordering::Acquire) == TestFault::Exit as u8 {
            return;
        }
        if input.len() < DEEPFILTER_FRAME_SIZE {
            std::thread::sleep(POLL_INTERVAL);
            continue;
        }
        input.read(&mut frame);
        let inference = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            #[cfg(test)]
            if shared.fault.load(Ordering::Acquire) == TestFault::Panic as u8 {
                panic!("injected worker panic");
            }
            df.process_into(&mut frame, &mut wet)
        }));
        let failure = match inference {
            Ok(Ok(_)) => {
                output.write(&wet);
                shared.inferred_frames.fetch_add(1, Ordering::Release);
                continue;
            }
            Ok(Err(error)) => error,
            Err(_) => DeepFilterProcessError::WorkerPanicked,
        };
        shared.failure.store(failure.code(), Ordering::Release);
        return;
    }
}
