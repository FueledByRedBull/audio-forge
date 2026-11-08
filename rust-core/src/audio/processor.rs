//! Unified audio processor - DSP chain runs entirely in Rust
//!
//! Processing chain: input cleanup -> gate -> suppressor -> de-esser -> EQ ->
//! compressor -> sample/true-peak limiting -> output.
//!
//! Adapted from Spectral Workbench for AudioForge.
//!
//! Layout: `shared` holds every cross-thread value (`Arc<Shared>`, grouped by
//! stage); `lifecycle` starts and stops the streams and the DSP thread under the
//! `streams` lock; `dsp_loop` is the realtime loop (`DspLoop`, built on the DSP
//! thread from a `DspLoopSetup` the starting thread assembles); `control`,
//! `*_controls` and `routing` publish parameter changes to it through seqlock
//! snapshots and dirty flags;
//! `supervisor` and `recovery` restart after a stall or device change; the rest
//! are workers, diagnostics and the Python API (`python_api`).

#![allow(clippy::useless_conversion)] // PyO3 proc-macro wrappers trigger false positives.

use crate::audio::input::TARGET_SAMPLE_RATE;
use crate::config_ranges as ranges;
use crate::config_ranges::{
    COMPRESSOR_ATTACK_MAX_MS, COMPRESSOR_ATTACK_MIN_MS, COMPRESSOR_BASE_RELEASE_MAX_MS,
    COMPRESSOR_BASE_RELEASE_MIN_MS, COMPRESSOR_MAKEUP_MAX_DB, COMPRESSOR_MAKEUP_MIN_DB,
    COMPRESSOR_RATIO_MAX, COMPRESSOR_RATIO_MIN, COMPRESSOR_RELEASE_MAX_MS,
    COMPRESSOR_RELEASE_MIN_MS, COMPRESSOR_TARGET_LUFS_MAX, COMPRESSOR_TARGET_LUFS_MIN,
    COMPRESSOR_THRESHOLD_MAX_DB, COMPRESSOR_THRESHOLD_MIN_DB, DEESSER_ATTACK_MAX_MS,
    DEESSER_ATTACK_MIN_MS, DEESSER_AUTO_AMOUNT_MAX, DEESSER_AUTO_AMOUNT_MIN,
    DEESSER_HIGH_CUT_MAX_HZ, DEESSER_HIGH_CUT_MIN_HZ, DEESSER_LOW_CUT_MAX_HZ,
    DEESSER_LOW_CUT_MIN_HZ, DEESSER_MAX_REDUCTION_MAX_DB, DEESSER_MAX_REDUCTION_MIN_DB,
    DEESSER_RATIO_MAX, DEESSER_RATIO_MIN, DEESSER_RELEASE_MAX_MS, DEESSER_RELEASE_MIN_MS,
    DEESSER_THRESHOLD_MAX_DB, DEESSER_THRESHOLD_MIN_DB, GATE_ATTACK_MAX_MS, GATE_ATTACK_MIN_MS,
    GATE_RELEASE_MAX_MS, GATE_RELEASE_MIN_MS, GATE_THRESHOLD_MAX_DB, GATE_THRESHOLD_MIN_DB,
    LIMITER_CEILING_MAX_DB, LIMITER_CEILING_MIN_DB, LIMITER_RELEASE_MAX_MS, LIMITER_RELEASE_MIN_MS,
};
use crate::dsp::eq::{EqBandConfig, NUM_BANDS};
use crate::dsp::rnnoise::RNNOISE_FRAME_SIZE;
use lifecycle::{ActiveDevices, Streams};
use shared::Shared;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

#[cfg(all(debug_assertions, not(windows)))]
macro_rules! processor_debug_log {
    ($($arg:tt)*) => {
        eprintln!($($arg)*);
    };
}

#[cfg(all(not(debug_assertions), not(windows)))]
macro_rules! processor_debug_log {
    ($($arg:tt)*) => {};
}

const RT_INPUT_CHUNK_CAPACITY: usize = 8192;
const RT_PROCESS_BUFFER_CAPACITY: usize = 8192;
const RT_RESAMPLE_QUEUE_CAPACITY: usize = 65_536;
const RT_SUPPRESSOR_OUTPUT_CAPACITY: usize = RT_PROCESS_BUFFER_CAPACITY + RNNOISE_FRAME_SIZE;
const RT_OUTPUT_SCRATCH_CAPACITY: usize = RT_SUPPRESSOR_OUTPUT_CAPACITY;
// One drain of the suppressor engine moves everything its worker can hand back
// in a pass into the scratch buffer above, so a pass's results go down the
// chain in one call; a larger worker queue needs a larger scratch.
#[cfg(feature = "deepfilter")]
const _: () = assert!(
    crate::dsp::deepfilter_ffi::WORKER_OWED_SAMPLES <= RT_SUPPRESSOR_OUTPUT_CAPACITY,
    "the worker may owe more than the suppressor output scratch holds"
);
const RT_SUPPRESSOR_COMMAND_CAPACITY: usize = 2;
const RT_SUPPRESSOR_RETIRE_CAPACITY: usize = 8;
const RESAMPLER_CHUNK_SIZE: usize = 1024;
const PRODUCT_RESAMPLER_SINC_LEN: usize = 128;
const PRODUCT_RESAMPLER_WINDOW_NAME: &str = "blackman";
const PROCESS_IDLE_SLEEP_US: u64 = 100;
const PROCESS_IDLE_MAX_SLEEP_US: u64 = 1_600;
const PROCESS_IDLE_RECENT_INPUT_WINDOW_US: u64 = 2_000;
const COMPRESSOR_DEFAULT_RELEASE_TENTH_MS: u64 = 2000; // 200ms * 10

// The Windows default-buffer path produced 10/20 ms callbacks during sustained
// qualification. Keep at least 10 ms beyond a 20 ms callback at startup and a
// 15 ms reserve at the steady-state center so ordinary scheduler jitter cannot
// drain the queue between DSP writes. This intentionally trades 10 ms of
// engine latency for deterministic callback headroom.
#[cfg(not(target_os = "linux"))]
const OUTPUT_PRIME_MS: u32 = 30;
#[cfg(not(target_os = "linux"))]
const OUTPUT_TARGET_HIGH_MS: u32 = 40;
#[cfg(not(target_os = "linux"))]
const OUTPUT_HARD_BACKLOG_MS: u32 = 60;
// While the device really calls back every 10 ms and no resampler writes in
// 1024-sample chunks, the queue needs only one callback plus a reserve.
const OUTPUT_SHORT_CALLBACK_MAX_MS: u32 = 12;
const OUTPUT_SHORT_TARGET_LOW_MS: u32 = 10;
const OUTPUT_SHORT_TARGET_HIGH_MS: u32 = 20;
const OUTPUT_SHORT_HARD_BACKLOG_MS: u32 = 40;
// On a short engine period (see `wasapi_render.rs`) a callback takes a few ms of
// audio, so the same reserve sits on a much smaller callback.
const OUTPUT_LOW_PERIOD_CALLBACK_MAX_MS: u32 = 6;
const OUTPUT_LOW_PERIOD_TARGET_LOW_MS: u32 = 6;
const OUTPUT_LOW_PERIOD_TARGET_HIGH_MS: u32 = 10;
const OUTPUT_LOW_PERIOD_HARD_BACKLOG_MS: u32 = 30;
// PipeWire runs the input and output callbacks in one graph cycle, so the queue
// needs no callback-sized reserve. A 15 ms center was the smallest with no
// underruns at quanta of both 256 and 1024; 10 ms already dropped audio at 256.
#[cfg(target_os = "linux")]
const OUTPUT_PRIME_MS: u32 = 14;
#[cfg(target_os = "linux")]
const OUTPUT_TARGET_HIGH_MS: u32 = 16;
#[cfg(target_os = "linux")]
const OUTPUT_HARD_BACKLOG_MS: u32 = 35;
// A write can land before that cycle's read, so the fill it sees may hold a whole
// cycle more than the target. Room for that keeps long quanta out of the catch-up.
const OUTPUT_HARD_BACKLOG_RESERVE_MS: u32 = 5;
// pw_stream delivers a capture buffer and takes a playback buffer one cycle later
// each. The queues do not hold these two cycles; delay measured on the graph does.
const PIPEWIRE_STREAM_DELAY_CYCLES: u64 = 2;
const OUTPUT_DRIFT_MAX_RATIO_ADJUST: f32 = 0.008;
const OUTPUT_DRIFT_MAX_EXPANSION_RATIO: f32 = 0.96;
const DSP_THREAD_READY_TIMEOUT_MS: u64 = 5_000;
const MAX_RECORDING_SECONDS: usize = 30;
const MAX_OUTPUT_PROBE_SECONDS: usize = 2;
const INPUT_DC_BLOCK_COEFF: f32 = 0.995;
const INPUT_PREFILTER_HZ: f64 = 80.0;
const INPUT_PREFILTER_Q: f64 = 0.707;
#[cfg(feature = "vad")]
const VAD_WORKER_MAX_BUFFER_SAMPLES: usize = 48_000;
#[cfg(feature = "vad")]
const VAD_PROBABILITY_STALE_US: u64 = 500_000;
const DEESSER_MIN_BANDWIDTH_HZ: f64 = 200.0;

// The live setters clamp to the table Python validates presets against
// (`config_ranges`); the values stored in single precision are cast from it.
#[cfg(feature = "vad")]
const VAD_THRESHOLD_MIN: f32 = ranges::VAD_THRESHOLD_MIN as f32;
#[cfg(feature = "vad")]
const VAD_THRESHOLD_MAX: f32 = ranges::VAD_THRESHOLD_MAX as f32;
#[cfg(feature = "vad")]
const VAD_HOLD_MIN_MS: f32 = ranges::VAD_HOLD_MIN_MS as f32;
#[cfg(feature = "vad")]
const VAD_HOLD_MAX_MS: f32 = ranges::VAD_HOLD_MAX_MS as f32;
#[cfg(feature = "vad")]
const VAD_PRE_GAIN_MIN: f32 = ranges::VAD_PRE_GAIN_MIN as f32;
#[cfg(feature = "vad")]
const VAD_PRE_GAIN_MAX: f32 = ranges::VAD_PRE_GAIN_MAX as f32;
#[cfg(feature = "vad")]
const GATE_MARGIN_MIN_DB: f32 = ranges::GATE_MARGIN_MIN_DB as f32;
#[cfg(feature = "vad")]
const GATE_MARGIN_MAX_DB: f32 = ranges::GATE_MARGIN_MAX_DB as f32;
const RNNOISE_STRENGTH_MIN: f32 = ranges::RNNOISE_STRENGTH_MIN as f32;
const RNNOISE_STRENGTH_MAX: f32 = ranges::RNNOISE_STRENGTH_MAX as f32;

mod block_processor;
mod control;
mod diagnostics;
mod dsp_loop;
mod dynamics_controls;
mod eq_controls;
mod gate_controls;
mod lifecycle;
#[cfg(test)]
mod loop_tests;
mod noise_controls;
mod output_writer;
mod python_api;
mod raw_recording;
mod recovery;
mod resampling;
mod routing;
#[cfg(test)]
mod rt_audit_tests;
mod runtime_metrics;
mod shared;
mod status;
mod stress_harness;
mod supervisor;
#[cfg(test)]
mod tests;
mod vad_worker;

pub use block_processor::{OfflineDspBlockProcessor, OfflineDspBlockStats};
#[cfg(feature = "vad")]
pub use python_api::{analyze_vad_probabilities, simulate_gate_suppressor_order, PyGateMode};
pub use python_api::{
    input_prefilter_response_db, simulate_auto_eq_chain, simulate_auto_makeup_control,
    PyAudioProcessor,
};
pub use resampling::{
    product_resampler_configuration, simulate_product_resampler, simulate_product_resampler_py,
};
pub use routing::InputCleanupMode;
pub use stress_harness::{run_seeded_control_dsp_stress, ControlDspStressReport};

/// Main audio processor combining all DSP stages.
///
/// Every method takes `&self`, so the processor can be shared between threads: a
/// worker may start, stop or recover the streams while the GUI thread reads
/// meters and diagnostics. Those reads only touch `shared` and `devices`, never
/// the `streams` lock a start holds.
pub struct AudioProcessor {
    /// State shared with the DSP thread, the supervisor and the VAD worker.
    shared: Arc<Shared>,

    /// Sample rate
    sample_rate: u32,

    /// Stream and worker-thread handles. The one lock serializes start, stop and
    /// recovery.
    streams: Mutex<Streams>,

    /// Devices the streams are bound to. Held only for short reads and writes.
    devices: Mutex<ActiveDevices>,
}

// The Python class is `frozen` and called from worker threads.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AudioProcessor>();
};

impl AudioProcessor {
    fn validate_eq_band_index(&self, band: usize) -> Result<(), String> {
        if band >= NUM_BANDS {
            return Err(format!("Band {} out of range [0, {})", band, NUM_BANDS));
        }
        Ok(())
    }

    fn validate_eq_band_config(&self, band: usize, config: EqBandConfig) -> Result<(), String> {
        self.validate_eq_band_index(band)?;
        config.validate(band, self.sample_rate as f64)
    }

    /// Create a new audio processor
    pub fn new() -> Self {
        let sample_rate = TARGET_SAMPLE_RATE;

        Self {
            shared: Arc::new(Shared::new(sample_rate)),
            sample_rate,
            streams: Mutex::new(Streams::default()),
            devices: Mutex::new(ActiveDevices::default()),
        }
    }
}

impl Default for AudioProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AudioProcessor {
    fn drop(&mut self) {
        self.stop();
        self.shared
            .stream
            .supervisor_running
            .store(false, Ordering::Release);
        let supervisor = self
            .streams
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .supervisor_thread
            .take();
        if let Some(handle) = supervisor {
            let _ = handle.join();
        }
    }
}
