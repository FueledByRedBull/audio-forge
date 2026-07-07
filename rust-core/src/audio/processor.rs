//! Unified audio processor - DSP chain runs entirely in Rust
//!
//! Processing chain: Mic Input Ã¢â€ â€™ Noise Gate Ã¢â€ â€™ RNNoise Ã¢â€ â€™ 10-Band EQ Ã¢â€ â€™ Output
//!
//! Adapted from Spectral Workbench for AudioForge.

#![allow(clippy::useless_conversion)] // PyO3 proc-macro wrappers trigger false positives.

use pyo3::prelude::*;
use rubato::{
    calculate_cutoff, Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thread_priority::{set_current_thread_priority, ThreadPriority};

use std::sync::atomic::AtomicU8;

use super::buffer::{AudioProducer, AudioRingBuffer};
use super::clock::now_micros;
use super::input::{AudioInput, InputChannelMode, InputStreamOptions, TARGET_SAMPLE_RATE};
use super::output::AudioOutput;
use super::rt::{store_rt_error, FixedAudioBuffer, RtCommandQueue, RtErrorCode};
use crate::dsp::biquad::{Biquad, BiquadType};
use crate::dsp::eq::{DEFAULT_FREQUENCIES, DEFAULT_Q, NUM_BANDS};
use crate::dsp::noise_suppressor::{NoiseModel, NoiseSuppressionEngine, NoiseSuppressor};
use crate::dsp::rnnoise::RNNOISE_FRAME_SIZE;
use crate::dsp::{Compressor, DeEsser, Limiter, NoiseGate, ParametricEQ, TruePeakDetector};

#[cfg(feature = "vad")]
use crate::dsp::vad::{GateMode, SileroVAD, VadAutoGate};

const RT_INPUT_CHUNK_CAPACITY: usize = 8192;
const RT_PROCESS_BUFFER_CAPACITY: usize = 8192;
const RT_RESAMPLE_QUEUE_CAPACITY: usize = 65_536;
const RT_SUPPRESSOR_OUTPUT_CAPACITY: usize = RT_PROCESS_BUFFER_CAPACITY + RNNOISE_FRAME_SIZE;
const RT_OUTPUT_SCRATCH_CAPACITY: usize = RT_SUPPRESSOR_OUTPUT_CAPACITY;
const RT_SUPPRESSOR_COMMAND_CAPACITY: usize = 2;
const RT_SUPPRESSOR_RETIRE_CAPACITY: usize = 8;
const PROCESS_IDLE_SLEEP_US: u64 = 100;
const PROCESS_IDLE_MAX_SLEEP_US: u64 = 1_600;
const PROCESS_IDLE_RECENT_INPUT_WINDOW_US: u64 = 2_000;
const COMPRESSOR_DEFAULT_RELEASE_TENTH_MS: u64 = 2000; // 200ms * 10
const OUTPUT_PRIME_MS: u32 = 20;
const OUTPUT_TARGET_HIGH_MS: u32 = 30;
const OUTPUT_HARD_BACKLOG_MS: u32 = 60;
const OUTPUT_DRIFT_MAX_RATIO_ADJUST: f32 = 0.008;
const OUTPUT_DRIFT_MAX_EXPANSION_RATIO: f32 = 0.96;
const DSP_THREAD_READY_TIMEOUT_MS: u64 = 5_000;
const MAX_RECORDING_SECONDS: usize = 30;
const INPUT_DC_BLOCK_COEFF: f32 = 0.995;
const INPUT_PREFILTER_HZ: f64 = 80.0;
const INPUT_PREFILTER_Q: f64 = 0.707;
const EQ_GAIN_MIN_DB: f64 = -12.0;
const EQ_GAIN_MAX_DB: f64 = 12.0;
const EQ_Q_MIN: f64 = 0.1;
const EQ_Q_MAX: f64 = 10.0;
const EQ_FREQ_MIN_HZ: f64 = 20.0;
const EQ_NYQUIST_MARGIN_HZ: f64 = 1.0;
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

#[cfg(debug_assertions)]
macro_rules! processor_debug_log {
    ($($arg:tt)*) => {
        eprintln!($($arg)*);
    };
}

#[cfg(not(debug_assertions))]
macro_rules! processor_debug_log {
    ($($arg:tt)*) => {};
}

include!("processor/control.rs");
include!("processor/block_processor.rs");
include!("processor/output_writer.rs");

/// Main audio processor combining all DSP stages
pub struct AudioProcessor {
    /// Noise gate
    gate: Arc<Mutex<NoiseGate>>,
    gate_enabled: Arc<AtomicBool>,
    gate_control: Arc<Mutex<GateControlState>>,
    gate_rt_control: Arc<AtomicGateControlState>,
    gate_dirty: Arc<AtomicBool>,

    /// Noise suppression engine (RNNoise or DeepFilterNet)
    suppressor: Arc<Mutex<NoiseSuppressionEngine>>,
    suppressor_enabled: Arc<AtomicBool>,
    suppressor_control: Arc<Mutex<SuppressorControlState>>,
    suppressor_rt_control: Arc<AtomicSuppressorControlState>,
    suppressor_dirty: Arc<AtomicBool>,
    suppressor_reset_requested: Arc<AtomicBool>,
    pending_suppressor_tx: Arc<Mutex<Option<ringbuf::HeapProducer<NoiseSuppressionEngine>>>>,
    retired_suppressor_rx: Arc<Mutex<Option<ringbuf::HeapConsumer<NoiseSuppressionEngine>>>>,
    suppressor_strength: Arc<AtomicU32>, // f32 bits stored as u32
    current_model: Arc<AtomicU8>,        // NoiseModel as u8

    /// 10-band parametric EQ
    eq: Arc<Mutex<ParametricEQ>>,
    eq_enabled: Arc<AtomicBool>,
    eq_control: Arc<EqControlState>,
    eq_dirty: Arc<AtomicBool>,

    /// Compressor
    compressor: Arc<Mutex<Compressor>>,
    compressor_enabled: Arc<AtomicBool>,
    compressor_control: Arc<Mutex<CompressorControlState>>,
    compressor_rt_control: Arc<AtomicCompressorControlState>,
    compressor_dirty: Arc<AtomicBool>,

    /// De-esser
    deesser: Arc<Mutex<DeEsser>>,
    deesser_enabled: Arc<AtomicBool>,
    deesser_control: Arc<Mutex<DeesserControlState>>,
    deesser_rt_control: Arc<AtomicDeesserControlState>,
    deesser_dirty: Arc<AtomicBool>,

    /// Hard limiter (brick-wall ceiling)
    limiter: Arc<Mutex<Limiter>>,
    limiter_enabled: Arc<AtomicBool>,
    limiter_control: Arc<Mutex<LimiterControlState>>,
    limiter_rt_control: Arc<AtomicLimiterControlState>,
    limiter_dirty: Arc<AtomicBool>,

    /// Audio input stream
    audio_input: Option<AudioInput>,

    /// Audio output stream
    audio_output: Option<AudioOutput>,

    /// Processing thread handle
    process_thread: Option<std::thread::JoinHandle<()>>,

    /// Supervisor thread for callback-based recovery detection.
    supervisor_thread: Option<std::thread::JoinHandle<()>>,
    supervisor_running: Arc<AtomicBool>,
    restart_requested: Arc<AtomicBool>,
    recovering: Arc<AtomicBool>,
    restart_backoff_index: Arc<AtomicU32>,
    last_restart_attempt_us: Arc<AtomicU64>,
    last_start_time_us: Arc<AtomicU64>,
    recovery_state: Arc<Mutex<StreamRecoveryState>>,

    /// Running flag
    running: Arc<AtomicBool>,

    /// Master bypass flag
    bypass: Arc<AtomicBool>,
    /// True raw monitor path: bypasses pre-filter/DSP and uses minimal output write path.
    raw_monitor_enabled: Arc<AtomicBool>,
    /// Input channel mixdown mode consumed by the CPAL input callback.
    input_channel_mode: Arc<AtomicU8>,

    /// Sample rate
    sample_rate: u32,
    /// Actual active output device sample rate
    output_sample_rate: Arc<AtomicU32>,
    /// Latest stereo input correlation reported by the input callback.
    input_stereo_correlation: Arc<AtomicU32>,
    /// Number of stereo input blocks with strong negative correlation.
    input_phase_warning_count: Arc<AtomicU64>,

    /// Input device name
    input_device_name: Option<String>,

    /// Output device name
    output_device_name: Option<String>,

    // === Level Metering (lock-free atomics) ===
    // Stored as f32 bits via to_bits()/from_bits()
    /// Input peak level at processor tap (pre-filtered in normal mode, raw in raw-monitor mode)
    input_peak: Arc<AtomicU32>,
    /// Input RMS level
    input_rms: Arc<AtomicU32>,
    /// Output peak level (after all processing)
    output_peak: Arc<AtomicU32>,
    /// Output RMS level
    output_rms: Arc<AtomicU32>,
    /// Compressor gain reduction in dB
    compressor_gain_reduction: Arc<AtomicU32>,
    /// De-esser gain reduction in dB
    deesser_gain_reduction: Arc<AtomicU32>,
    /// De-esser detector confidence (0.0-1.0)
    deesser_detector_confidence: Arc<AtomicU32>,
    /// Last known gate gain (0.0-1.0) for metering.
    gate_gain_meter: Arc<AtomicU32>,
    /// VAD speech probability (0.0-1.0) for metering
    #[cfg_attr(not(feature = "vad"), allow(dead_code))]
    vad_probability: Arc<AtomicU32>,
    /// Latest gate noise-floor estimate in dB.
    #[cfg_attr(not(feature = "vad"), allow(dead_code))]
    gate_noise_floor_db: Arc<AtomicU32>,
    /// Latest fused gate open score (0.0-1.0) for diagnostics.
    #[cfg_attr(not(feature = "vad"), allow(dead_code))]
    gate_fused_score: Arc<AtomicU32>,
    /// Lifetime count of detected rapid gate open/close chatter.
    gate_chatter_event_count: Arc<AtomicU64>,
    /// Whether the current VAD backend is available.
    #[cfg_attr(not(feature = "vad"), allow(dead_code))]
    vad_available: Arc<AtomicBool>,
    #[cfg(feature = "vad")]
    /// Non-realtime VAD worker thread handle.
    vad_worker_thread: Option<std::thread::JoinHandle<()>>,
    #[cfg(feature = "vad")]
    /// Non-realtime VAD worker running flag.
    vad_worker_running: Arc<AtomicBool>,
    #[cfg(feature = "vad")]
    /// Last successful non-realtime VAD inference timestamp.
    vad_last_update_us: Arc<AtomicU64>,

    /// Compressor current release time in milliseconds (for metering)
    compressor_current_release_ms: Arc<AtomicU64>,
    /// Compressor current measured loudness in LUFS (f64 bits).
    compressor_current_lufs: Arc<AtomicU64>,
    /// Compressor current applied makeup gain in dB (f64 bits).
    compressor_current_makeup_gain: Arc<AtomicU64>,

    /// Processing latency in microseconds (measured from input to output)
    latency_us: Arc<AtomicU64>,
    /// User-configured compensation added to reported latency (microseconds).
    latency_compensation_us: Arc<AtomicU64>,

    // === DSP Performance Metrics (lock-free atomics) ===
    /// DSP processing time in microseconds (for last processed chunk)
    dsp_time_us: Arc<AtomicU64>,
    /// Input ring buffer fill level (samples)
    input_buffer_len: Arc<AtomicU32>,
    /// Output ring buffer fill level (samples)
    output_buffer_len: Arc<AtomicU32>,
    /// Noise suppressor internal buffer fill level (samples)
    suppressor_buffer_len: Arc<AtomicU32>,
    /// Suppressor latency + pending samples mirrored from the processing thread.
    suppressor_latency_samples: Arc<AtomicU32>,
    /// Last successful output write time (for detecting stalled processing)
    last_output_write_time: Arc<AtomicU64>,
    /// Last input callback heartbeat timestamp (unix micros)
    last_input_callback_time_us: Arc<AtomicU64>,
    /// Last output callback heartbeat timestamp (unix micros)
    last_output_callback_time_us: Arc<AtomicU64>,
    /// Consecutive output callback underrun count
    output_underrun_streak: Arc<AtomicU32>,
    /// Total output callback underruns since start
    output_underrun_total: Arc<AtomicU64>,
    /// Samples dropped by DSP-side jitter buffer control
    jitter_dropped_samples: Arc<AtomicU64>,
    /// Number of normal DSP-side drift-retime length adjustments.
    output_retime_adjustment_count: Arc<AtomicU64>,
    /// Number of true DSP-side output recovery events.
    output_recovery_event_count: Arc<AtomicU64>,
    /// Samples discarded because the output ring buffer could not accept the full write.
    output_short_write_dropped_samples: Arc<AtomicU64>,
    /// Number of times the DSP loop entered its idle sleep path.
    dsp_idle_wakeup_count: Arc<AtomicU64>,
    /// Last idle sleep duration selected by the DSP loop.
    dsp_idle_sleep_us: Arc<AtomicU64>,

    /// Smoothed DSP processing time in microseconds (EMA, 200ms time constant)
    dsp_time_smoothed_us: Arc<AtomicU64>,
    /// Smoothed input buffer fill level (EMA, 200ms time constant)
    smoothed_input_buffer_len: Arc<AtomicU32>,
    /// Smoothed suppressor buffer fill level (EMA, 200ms time constant)
    smoothed_buffer_len: Arc<AtomicU32>,

    /// Dropped samples counter from input ring buffer
    input_dropped: Arc<AtomicU64>,
    /// Number of proactive input backlog recoveries.
    input_backlog_recovery_count: Arc<AtomicU64>,
    /// Samples proactively dropped from the input backlog.
    input_backlog_dropped_samples: Arc<AtomicU64>,

    /// Total lock contention events in the real-time processing loop.
    lock_contention_count: Arc<AtomicU64>,
    /// Number of times suppressor output contained non-finite samples.
    suppressor_non_finite_count: Arc<AtomicU64>,
    /// Number of clipped input samples before safety clamp.
    clip_event_count: Arc<AtomicU64>,
    /// Peak pre-clamp level in dBFS.
    clip_peak_db: Arc<AtomicU32>,
    /// Number of final output samples clamped by output protection.
    output_clip_event_count: Arc<AtomicU64>,
    /// Peak final pre-clamp output level in dBFS.
    output_clip_peak_db: Arc<AtomicU32>,
    /// Number of output blocks with estimated true peak above the active ceiling.
    output_true_peak_event_count: Arc<AtomicU64>,
    /// Peak estimated true-peak level in dBFS.
    output_true_peak_db: Arc<AtomicU32>,
    /// Whether input resampling is active.
    input_resampler_active: Arc<AtomicBool>,
    /// Whether output resampling is active.
    output_resampler_active: Arc<AtomicBool>,
    /// Whether the current noise backend is operational.
    noise_backend_available: Arc<AtomicBool>,
    /// Whether the backend has entered a failed state.
    noise_backend_failed: Arc<AtomicBool>,
    /// Last backend error string.
    noise_backend_error: Arc<Mutex<Option<String>>>,
    /// Last RT-safe error code published by callbacks or the DSP loop.
    rt_error_code: Arc<AtomicU32>,
    /// Input callback stream error count.
    input_callback_error_count: Arc<AtomicU64>,
    /// Output callback stream error count.
    output_callback_error_count: Arc<AtomicU64>,
    /// Fixed-buffer overflow count in AudioForge RT regions.
    rt_buffer_overflow_count: Arc<AtomicU64>,

    // === RAW AUDIO RECORDING (for calibration) ===
    /// Raw audio recording consumer (for calibration - captures audio AFTER pre-filter, BEFORE gate)
    raw_recording_consumer: Arc<Mutex<Option<super::buffer::AudioConsumer>>>,
    /// Current recording position (samples recorded so far)
    raw_recording_pos: Arc<AtomicUsize>,
    /// Target recording length (total samples to record)
    raw_recording_target: Arc<AtomicUsize>,
    /// Current recording level in dB for UI/self-test metering.
    recording_level_db: Arc<AtomicU32>,
    /// Manual output mute flag (independent of recording).
    output_muted: Arc<AtomicBool>,
    /// Flag indicating recording is active (used to mute output to prevent user from hearing themselves)
    recording_active: Arc<AtomicBool>,
    /// Temporarily disables watchdog-driven recovery during intrusive UI workflows.
    recovery_suppressed: Arc<AtomicBool>,
}

impl AudioProcessor {
    #[cfg(feature = "deepfilter")]
    fn deepfilter_experimental_enabled() -> bool {
        std::env::var("AUDIOFORGE_ENABLE_DEEPFILTER")
            .map(|v| {
                let normalized = v.trim().to_ascii_lowercase();
                normalized == "1" || normalized == "true" || normalized == "yes"
            })
            .unwrap_or(false)
    }

    fn eq_nyquist_limit_hz(&self) -> f64 {
        (self.sample_rate as f64 / 2.0 - EQ_NYQUIST_MARGIN_HZ).max(EQ_FREQ_MIN_HZ)
    }

    fn validate_eq_band_index(&self, band: usize) -> Result<(), String> {
        if band >= NUM_BANDS {
            return Err(format!("Band {} out of range [0, {})", band, NUM_BANDS));
        }
        Ok(())
    }

    fn validate_eq_frequency(&self, band: usize, frequency: f64) -> Result<(), String> {
        if !frequency.is_finite() {
            return Err(format!("Band {}: frequency must be finite", band));
        }
        let max_frequency = self.eq_nyquist_limit_hz();
        if !(EQ_FREQ_MIN_HZ..=max_frequency).contains(&frequency) {
            return Err(format!(
                "Band {}: frequency {} Hz out of range [{}, {}]",
                band, frequency, EQ_FREQ_MIN_HZ, max_frequency
            ));
        }
        Ok(())
    }

    fn validate_eq_gain(&self, band: usize, gain_db: f64) -> Result<(), String> {
        if !gain_db.is_finite() {
            return Err(format!("Band {}: gain must be finite", band));
        }
        if !(EQ_GAIN_MIN_DB..=EQ_GAIN_MAX_DB).contains(&gain_db) {
            return Err(format!(
                "Band {}: gain {} dB out of range [{}, {}]",
                band, gain_db, EQ_GAIN_MIN_DB, EQ_GAIN_MAX_DB
            ));
        }
        Ok(())
    }

    fn validate_eq_q(&self, band: usize, q: f64) -> Result<(), String> {
        if !q.is_finite() {
            return Err(format!("Band {}: Q must be finite", band));
        }
        if !(EQ_Q_MIN..=EQ_Q_MAX).contains(&q) {
            return Err(format!(
                "Band {}: Q {} out of range [{}, {}]",
                band, q, EQ_Q_MIN, EQ_Q_MAX
            ));
        }
        Ok(())
    }

    /// Create a new audio processor
    pub fn new() -> Self {
        let sample_rate = TARGET_SAMPLE_RATE;
        let gate_control_state = GateControlState::new();
        let suppressor_control_state = SuppressorControlState::new();
        let compressor_control_state = CompressorControlState::new();

        // Create strength Arc BEFORE noise suppressor (share reference)
        let suppressor_strength = Arc::new(AtomicU32::new(1.0_f32.to_bits()));

        // Create noise suppression engine (default to RNNoise)
        let suppressor =
            NoiseSuppressionEngine::new(NoiseModel::RNNoise, suppressor_strength.clone());

        Self {
            gate: {
                let gate = Arc::new(Mutex::new(NoiseGate::new(
                    -40.0,
                    10.0,
                    100.0,
                    sample_rate as f64,
                )));

                #[cfg(feature = "vad")]
                {
                    let vad_auto_gate = VadAutoGate::without_backend(sample_rate, 0.5);
                    if let Ok(mut g) = gate.lock() {
                        g.set_vad_auto_gate(Some(vad_auto_gate));
                        apply_gate_control(&mut g, &gate_control_state);
                    }
                }

                gate
            },
            gate_enabled: Arc::new(AtomicBool::new(true)),
            gate_control: Arc::new(Mutex::new(gate_control_state)),
            gate_rt_control: Arc::new(AtomicGateControlState::new()),
            gate_dirty: Arc::new(AtomicBool::new(false)),
            suppressor: Arc::new(Mutex::new(suppressor)),
            suppressor_enabled: Arc::new(AtomicBool::new(true)),
            suppressor_control: Arc::new(Mutex::new(suppressor_control_state)),
            suppressor_rt_control: Arc::new(AtomicSuppressorControlState::new()),
            suppressor_dirty: Arc::new(AtomicBool::new(false)),
            suppressor_reset_requested: Arc::new(AtomicBool::new(false)),
            pending_suppressor_tx: Arc::new(Mutex::new(None)),
            retired_suppressor_rx: Arc::new(Mutex::new(None)),
            suppressor_strength, // Store Arc for PyO3 bindings
            current_model: Arc::new(AtomicU8::new(NoiseModel::RNNoise as u8)),
            eq: Arc::new(Mutex::new(ParametricEQ::new(sample_rate as f64))),
            eq_enabled: Arc::new(AtomicBool::new(true)),
            eq_control: Arc::new(EqControlState::new()),
            eq_dirty: Arc::new(AtomicBool::new(false)),
            compressor: {
                let mut compressor = Compressor::default_voice(sample_rate as f64);
                apply_compressor_control(&mut compressor, &compressor_control_state);
                Arc::new(Mutex::new(compressor))
            },
            compressor_enabled: Arc::new(AtomicBool::new(true)),
            compressor_control: Arc::new(Mutex::new(compressor_control_state)),
            compressor_rt_control: Arc::new(AtomicCompressorControlState::new()),
            compressor_dirty: Arc::new(AtomicBool::new(false)),
            deesser: Arc::new(Mutex::new(DeEsser::new(sample_rate as f64))),
            deesser_enabled: Arc::new(AtomicBool::new(false)),
            deesser_control: Arc::new(Mutex::new(DeesserControlState::new())),
            deesser_rt_control: Arc::new(AtomicDeesserControlState::new()),
            deesser_dirty: Arc::new(AtomicBool::new(false)),
            limiter: Arc::new(Mutex::new(Limiter::new(
                CAREFUL_OUTPUT_CEILING_DB,
                50.0,
                sample_rate as f64,
            ))),
            limiter_enabled: Arc::new(AtomicBool::new(true)),
            limiter_control: Arc::new(Mutex::new(LimiterControlState::new())),
            limiter_rt_control: Arc::new(AtomicLimiterControlState::new()),
            limiter_dirty: Arc::new(AtomicBool::new(false)),
            audio_input: None,
            audio_output: None,
            process_thread: None,
            supervisor_thread: None,
            supervisor_running: Arc::new(AtomicBool::new(false)),
            restart_requested: Arc::new(AtomicBool::new(false)),
            recovering: Arc::new(AtomicBool::new(false)),
            restart_backoff_index: Arc::new(AtomicU32::new(0)),
            last_restart_attempt_us: Arc::new(AtomicU64::new(0)),
            last_start_time_us: Arc::new(AtomicU64::new(0)),
            recovery_state: Arc::new(Mutex::new(StreamRecoveryState::default())),
            running: Arc::new(AtomicBool::new(false)),
            bypass: Arc::new(AtomicBool::new(false)),
            raw_monitor_enabled: Arc::new(AtomicBool::new(false)),
            input_channel_mode: Arc::new(AtomicU8::new(InputChannelMode::Average as u8)),
            sample_rate,
            output_sample_rate: Arc::new(AtomicU32::new(sample_rate)),
            input_stereo_correlation: Arc::new(AtomicU32::new(1.0_f32.to_bits())),
            input_phase_warning_count: Arc::new(AtomicU64::new(0)),
            input_device_name: None,
            output_device_name: None,
            // Initialize metering atomics with -infinity (no signal)
            input_peak: Arc::new(AtomicU32::new((-120.0_f32).to_bits())),
            input_rms: Arc::new(AtomicU32::new((-120.0_f32).to_bits())),
            output_peak: Arc::new(AtomicU32::new((-120.0_f32).to_bits())),
            output_rms: Arc::new(AtomicU32::new((-120.0_f32).to_bits())),
            compressor_gain_reduction: Arc::new(AtomicU32::new(0.0_f32.to_bits())),
            deesser_gain_reduction: Arc::new(AtomicU32::new(0.0_f32.to_bits())),
            deesser_detector_confidence: Arc::new(AtomicU32::new(0.0_f32.to_bits())),
            gate_gain_meter: Arc::new(AtomicU32::new(1.0_f32.to_bits())),
            vad_probability: Arc::new(AtomicU32::new(0.0_f32.to_bits())),
            gate_noise_floor_db: Arc::new(AtomicU32::new((-60.0_f32).to_bits())),
            gate_fused_score: Arc::new(AtomicU32::new(0.0_f32.to_bits())),
            gate_chatter_event_count: Arc::new(AtomicU64::new(0)),
            vad_available: Arc::new(AtomicBool::new(false)),
            #[cfg(feature = "vad")]
            vad_worker_thread: None,
            #[cfg(feature = "vad")]
            vad_worker_running: Arc::new(AtomicBool::new(false)),
            #[cfg(feature = "vad")]
            vad_last_update_us: Arc::new(AtomicU64::new(0)),
            compressor_current_release_ms: Arc::new(AtomicU64::new(
                COMPRESSOR_DEFAULT_RELEASE_TENTH_MS,
            )),
            compressor_current_lufs: Arc::new(AtomicU64::new((-100.0_f64).to_bits())),
            compressor_current_makeup_gain: Arc::new(AtomicU64::new(0.0_f64.to_bits())),
            latency_us: Arc::new(AtomicU64::new(0)),
            latency_compensation_us: Arc::new(AtomicU64::new(0)),
            // Initialize DSP performance metrics
            dsp_time_us: Arc::new(AtomicU64::new(0)),
            input_buffer_len: Arc::new(AtomicU32::new(0)),
            smoothed_input_buffer_len: Arc::new(AtomicU32::new(0)),
            output_buffer_len: Arc::new(AtomicU32::new(0)),
            suppressor_buffer_len: Arc::new(AtomicU32::new(0)),
            suppressor_latency_samples: Arc::new(AtomicU32::new(0)),
            last_output_write_time: Arc::new(AtomicU64::new(0)),
            last_input_callback_time_us: Arc::new(AtomicU64::new(0)),
            last_output_callback_time_us: Arc::new(AtomicU64::new(0)),
            output_underrun_streak: Arc::new(AtomicU32::new(0)),
            output_underrun_total: Arc::new(AtomicU64::new(0)),
            jitter_dropped_samples: Arc::new(AtomicU64::new(0)),
            output_retime_adjustment_count: Arc::new(AtomicU64::new(0)),
            output_recovery_event_count: Arc::new(AtomicU64::new(0)),
            output_short_write_dropped_samples: Arc::new(AtomicU64::new(0)),
            dsp_idle_wakeup_count: Arc::new(AtomicU64::new(0)),
            dsp_idle_sleep_us: Arc::new(AtomicU64::new(PROCESS_IDLE_SLEEP_US)),
            dsp_time_smoothed_us: Arc::new(AtomicU64::new(0)),
            smoothed_buffer_len: Arc::new(AtomicU32::new(0)),

            // Initialize dropped samples counter
            input_dropped: Arc::new(AtomicU64::new(0)),
            input_backlog_recovery_count: Arc::new(AtomicU64::new(0)),
            input_backlog_dropped_samples: Arc::new(AtomicU64::new(0)),

            // Initialize lock contention counter
            lock_contention_count: Arc::new(AtomicU64::new(0)),
            suppressor_non_finite_count: Arc::new(AtomicU64::new(0)),
            clip_event_count: Arc::new(AtomicU64::new(0)),
            clip_peak_db: Arc::new(AtomicU32::new((-120.0_f32).to_bits())),
            output_clip_event_count: Arc::new(AtomicU64::new(0)),
            output_clip_peak_db: Arc::new(AtomicU32::new((-120.0_f32).to_bits())),
            output_true_peak_event_count: Arc::new(AtomicU64::new(0)),
            output_true_peak_db: Arc::new(AtomicU32::new((-120.0_f32).to_bits())),
            input_resampler_active: Arc::new(AtomicBool::new(false)),
            output_resampler_active: Arc::new(AtomicBool::new(false)),
            noise_backend_available: Arc::new(AtomicBool::new(true)),
            noise_backend_failed: Arc::new(AtomicBool::new(false)),
            noise_backend_error: Arc::new(Mutex::new(None)),
            rt_error_code: Arc::new(AtomicU32::new(RtErrorCode::None as u32)),
            input_callback_error_count: Arc::new(AtomicU64::new(0)),
            output_callback_error_count: Arc::new(AtomicU64::new(0)),
            rt_buffer_overflow_count: Arc::new(AtomicU64::new(0)),

            // Initialize raw recording buffer
            raw_recording_consumer: Arc::new(Mutex::new(None)),
            raw_recording_pos: Arc::new(AtomicUsize::new(0)),
            raw_recording_target: Arc::new(AtomicUsize::new(0)),
            recording_level_db: Arc::new(AtomicU32::new((-120.0_f32).to_bits())),
            output_muted: Arc::new(AtomicBool::new(false)),
            recording_active: Arc::new(AtomicBool::new(false)),
            recovery_suppressed: Arc::new(AtomicBool::new(false)),
        }
    }
}

include!("processor/dsp_loop.rs");
include!("processor/status.rs");
include!("processor/gate_controls.rs");
include!("processor/noise_controls.rs");
include!("processor/eq_controls.rs");
include!("processor/dynamics_controls.rs");
include!("processor/runtime_metrics.rs");
include!("processor/supervisor.rs");
include!("processor/vad_worker.rs");
include!("processor/recovery.rs");
include!("processor/raw_recording.rs");

impl Default for AudioProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AudioProcessor {
    fn drop(&mut self) {
        self.stop();
        self.supervisor_running.store(false, Ordering::Release);
        if let Some(handle) = self.supervisor_thread.take() {
            let _ = handle.join();
        }
    }
}

include!("processor/tests.rs");

// === Python Bindings ===
include!("processor/python_api.rs");
