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
const OUTPUT_PRIME_MS: u32 = 30;
const OUTPUT_TARGET_HIGH_MS: u32 = 40;
const OUTPUT_HARD_BACKLOG_MS: u32 = 60;
const OUTPUT_DRIFT_MAX_RATIO_ADJUST: f32 = 0.008;
const OUTPUT_DRIFT_MAX_EXPANSION_RATIO: f32 = 0.96;
const DSP_THREAD_READY_TIMEOUT_MS: u64 = 5_000;
const MAX_RECORDING_SECONDS: usize = 30;
const MAX_OUTPUT_PROBE_SECONDS: usize = 2;
const INPUT_DC_BLOCK_COEFF: f32 = 0.995;
const INPUT_PREFILTER_HZ: f64 = 80.0;
const INPUT_PREFILTER_Q: f64 = 0.707;
const GATE_THRESHOLD_MIN_DB: f64 = -80.0;
const GATE_THRESHOLD_MAX_DB: f64 = -10.0;
const GATE_ATTACK_MIN_MS: f64 = 0.1;
const GATE_ATTACK_MAX_MS: f64 = 100.0;
const GATE_RELEASE_MIN_MS: f64 = 10.0;
const GATE_RELEASE_MAX_MS: f64 = 1000.0;
#[cfg(feature = "vad")]
const VAD_THRESHOLD_MIN: f32 = 0.3;
#[cfg(feature = "vad")]
const VAD_THRESHOLD_MAX: f32 = 0.7;
#[cfg(feature = "vad")]
const VAD_HOLD_MIN_MS: f32 = 0.0;
#[cfg(feature = "vad")]
const VAD_HOLD_MAX_MS: f32 = 500.0;
#[cfg(feature = "vad")]
const VAD_PRE_GAIN_MIN: f32 = 1.0;
#[cfg(feature = "vad")]
const VAD_PRE_GAIN_MAX: f32 = 10.0;
#[cfg(feature = "vad")]
const VAD_WORKER_MAX_BUFFER_SAMPLES: usize = 48_000;
#[cfg(feature = "vad")]
const VAD_PROBABILITY_STALE_US: u64 = 500_000;
#[cfg(feature = "vad")]
const GATE_MARGIN_MIN_DB: f32 = 0.0;
#[cfg(feature = "vad")]
const GATE_MARGIN_MAX_DB: f32 = 20.0;
const RNNOISE_STRENGTH_MIN: f32 = 0.0;
const RNNOISE_STRENGTH_MAX: f32 = 1.0;
const DEESSER_AUTO_AMOUNT_MIN: f64 = 0.0;
const DEESSER_AUTO_AMOUNT_MAX: f64 = 1.0;
const DEESSER_LOW_CUT_MIN_HZ: f64 = 2000.0;
const DEESSER_LOW_CUT_MAX_HZ: f64 = 12000.0;
const DEESSER_HIGH_CUT_MIN_HZ: f64 = 2200.0;
const DEESSER_HIGH_CUT_MAX_HZ: f64 = 16000.0;
const DEESSER_MIN_BANDWIDTH_HZ: f64 = 200.0;
const DEESSER_THRESHOLD_MIN_DB: f64 = -60.0;
const DEESSER_THRESHOLD_MAX_DB: f64 = -6.0;
const DEESSER_RATIO_MIN: f64 = 1.0;
const DEESSER_RATIO_MAX: f64 = 20.0;
const DEESSER_ATTACK_MIN_MS: f64 = 0.1;
const DEESSER_ATTACK_MAX_MS: f64 = 50.0;
const DEESSER_RELEASE_MIN_MS: f64 = 5.0;
const DEESSER_RELEASE_MAX_MS: f64 = 500.0;
const DEESSER_MAX_REDUCTION_MIN_DB: f64 = 0.0;
const DEESSER_MAX_REDUCTION_MAX_DB: f64 = 24.0;
const COMPRESSOR_THRESHOLD_MIN_DB: f64 = -60.0;
const COMPRESSOR_THRESHOLD_MAX_DB: f64 = 0.0;
const COMPRESSOR_RATIO_MIN: f64 = 1.0;
const COMPRESSOR_RATIO_MAX: f64 = 20.0;
const COMPRESSOR_ATTACK_MIN_MS: f64 = 0.1;
const COMPRESSOR_ATTACK_MAX_MS: f64 = 100.0;
const COMPRESSOR_RELEASE_MIN_MS: f64 = 10.0;
const COMPRESSOR_RELEASE_MAX_MS: f64 = 1000.0;
const COMPRESSOR_BASE_RELEASE_MIN_MS: f64 = 20.0;
const COMPRESSOR_BASE_RELEASE_MAX_MS: f64 = 200.0;
const COMPRESSOR_MAKEUP_MIN_DB: f64 = 0.0;
const COMPRESSOR_MAKEUP_MAX_DB: f64 = 24.0;
const COMPRESSOR_TARGET_LUFS_MIN: f64 = -24.0;
const COMPRESSOR_TARGET_LUFS_MAX: f64 = -12.0;
const LIMITER_CEILING_MIN_DB: f64 = -12.0;
const LIMITER_CEILING_MAX_DB: f64 = 0.0;
const LIMITER_RELEASE_MIN_MS: f64 = 10.0;
const LIMITER_RELEASE_MAX_MS: f64 = 500.0;

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
