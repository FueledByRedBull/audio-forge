//! Unified audio processor - DSP chain runs entirely in Rust
//!
//! Processing chain: Mic Input → Noise Gate → RNNoise → 10-Band EQ → Output
//!
//! Adapted from Spectral Workbench project for MicEq.

#![allow(clippy::useless_conversion)] // PyO3 proc-macro wrappers trigger false positives.

use pyo3::prelude::*;
use rubato::{
    calculate_cutoff, Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use thread_priority::{set_current_thread_priority, ThreadPriority};

use std::sync::atomic::AtomicU8;

use super::buffer::AudioRingBuffer;
use super::clock::now_micros;
use super::input::{AudioInput, TARGET_SAMPLE_RATE};
use super::output::AudioOutput;
use crate::dsp::biquad::{Biquad, BiquadType};
use crate::dsp::eq::{DEFAULT_FREQUENCIES, DEFAULT_Q, NUM_BANDS};
use crate::dsp::noise_suppressor::{NoiseModel, NoiseSuppressionEngine, NoiseSuppressor};
use crate::dsp::rnnoise::RNNOISE_FRAME_SIZE;
use crate::dsp::{Compressor, DeEsser, Limiter, NoiseGate, ParametricEQ};

#[cfg(feature = "vad")]
use crate::dsp::vad::{GateMode, VadAutoGate};

const RESAMPLE_COMPACT_THRESHOLD: usize = 16384;
const PROCESS_IDLE_SLEEP_US: u64 = 100;
const GATE_FALLBACK_APPLY_THRESHOLD: f32 = 0.9999;
const COMPRESSOR_DEFAULT_RELEASE_TENTH_MS: u64 = 2000; // 200ms * 10
const OUTPUT_PRIME_MS: u32 = 10;
const OUTPUT_TARGET_HIGH_MS: u32 = 20;
const OUTPUT_HARD_BACKLOG_MS: u32 = 40;
const MAX_RECORDING_SECONDS: usize = 30;

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

fn duration_samples(sample_rate: u32, duration_ms: u32) -> usize {
    (((sample_rate as u64) * duration_ms as u64 + 500) / 1000).max(1) as usize
}

fn samples_to_micros(samples: u64, sample_rate: u32) -> u64 {
    if sample_rate == 0 {
        0
    } else {
        samples.saturating_mul(1_000_000) / sample_rate as u64
    }
}

struct StreamRecoveryState {
    last_error: Option<String>,
    last_reason: Option<String>,
    restart_count: u64,
}

impl Default for StreamRecoveryState {
    fn default() -> Self {
        Self {
            last_error: None,
            last_reason: None,
            restart_count: 0,
        }
    }
}

#[derive(Clone)]
struct EqControlState {
    enabled: bool,
    bands: [(f64, f64, f64); NUM_BANDS],
}

impl EqControlState {
    fn new() -> Self {
        Self {
            enabled: true,
            bands: std::array::from_fn(|index| (DEFAULT_FREQUENCIES[index], 0.0, DEFAULT_Q)),
        }
    }
}

#[derive(Clone)]
struct DeesserControlState {
    enabled: bool,
    auto_enabled: bool,
    auto_amount: f64,
    low_cut_hz: f64,
    high_cut_hz: f64,
    threshold_db: f64,
    ratio: f64,
    attack_ms: f64,
    release_ms: f64,
    max_reduction_db: f64,
}

impl DeesserControlState {
    fn new() -> Self {
        Self {
            enabled: false,
            auto_enabled: true,
            auto_amount: 0.5,
            low_cut_hz: 4000.0,
            high_cut_hz: 9000.0,
            threshold_db: -28.0,
            ratio: 4.0,
            attack_ms: 2.0,
            release_ms: 80.0,
            max_reduction_db: 6.0,
        }
    }
}

#[derive(Clone)]
struct CompressorControlState {
    enabled: bool,
    threshold_db: f64,
    ratio: f64,
    attack_ms: f64,
    base_release_ms: f64,
    makeup_gain_db: f64,
    adaptive_release: bool,
    auto_makeup_enabled: bool,
    target_lufs: f64,
}

impl CompressorControlState {
    fn new() -> Self {
        Self {
            enabled: true,
            threshold_db: -20.0,
            ratio: 4.0,
            attack_ms: 10.0,
            base_release_ms: 200.0,
            makeup_gain_db: 0.0,
            adaptive_release: false,
            auto_makeup_enabled: false,
            target_lufs: -18.0,
        }
    }
}

#[derive(Clone)]
struct LimiterControlState {
    enabled: bool,
    ceiling_db: f64,
    release_ms: f64,
}

impl LimiterControlState {
    fn new() -> Self {
        Self {
            enabled: true,
            ceiling_db: -0.5,
            release_ms: 50.0,
        }
    }
}

fn apply_eq_control(eq: &mut ParametricEQ, control: &EqControlState) {
    eq.set_enabled(control.enabled);
    for (index, (frequency, gain_db, q)) in control.bands.iter().copied().enumerate() {
        eq.set_band_frequency(index, frequency);
        eq.set_band_gain(index, gain_db);
        eq.set_band_q(index, q);
    }
}

fn apply_deesser_control(deesser: &mut DeEsser, control: &DeesserControlState) {
    deesser.set_enabled(control.enabled);
    deesser.set_auto_enabled(control.auto_enabled);
    deesser.set_auto_amount(control.auto_amount);
    deesser.set_low_cut_hz(control.low_cut_hz);
    deesser.set_high_cut_hz(control.high_cut_hz);
    deesser.set_threshold_db(control.threshold_db);
    deesser.set_ratio(control.ratio);
    deesser.set_attack_ms(control.attack_ms);
    deesser.set_release_ms(control.release_ms);
    deesser.set_max_reduction_db(control.max_reduction_db);
}

fn apply_compressor_control(compressor: &mut Compressor, control: &CompressorControlState) {
    compressor.set_enabled(control.enabled);
    compressor.set_threshold(control.threshold_db);
    compressor.set_ratio(control.ratio);
    compressor.set_attack_time(control.attack_ms);
    compressor.set_base_release_time(control.base_release_ms);
    if !control.adaptive_release {
        compressor.set_release_time(control.base_release_ms);
    }
    compressor.set_makeup_gain(control.makeup_gain_db);
    compressor.set_adaptive_release(control.adaptive_release);
    compressor.set_auto_makeup_enabled(control.auto_makeup_enabled);
    compressor.set_target_lufs(control.target_lufs);
}

fn apply_limiter_control(limiter: &mut Limiter, control: &LimiterControlState) {
    limiter.set_enabled(control.enabled);
    limiter.set_ceiling(control.ceiling_db);
    limiter.set_release_time(control.release_ms);
}

/// Main audio processor combining all DSP stages
pub struct AudioProcessor {
    /// Noise gate
    gate: Arc<Mutex<NoiseGate>>,
    gate_enabled: Arc<AtomicBool>,

    /// Noise suppression engine (RNNoise or DeepFilterNet)
    suppressor: Arc<Mutex<NoiseSuppressionEngine>>,
    suppressor_enabled: Arc<AtomicBool>,
    suppressor_strength: Arc<AtomicU32>, // f32 bits stored as u32
    current_model: Arc<AtomicU8>,        // NoiseModel as u8

    /// 10-band parametric EQ
    eq: Arc<Mutex<ParametricEQ>>,
    eq_enabled: Arc<AtomicBool>,
    eq_control: Arc<Mutex<EqControlState>>,
    eq_dirty: Arc<AtomicBool>,

    /// Compressor
    compressor: Arc<Mutex<Compressor>>,
    compressor_enabled: Arc<AtomicBool>,
    compressor_control: Arc<Mutex<CompressorControlState>>,
    compressor_dirty: Arc<AtomicBool>,

    /// De-esser
    deesser: Arc<Mutex<DeEsser>>,
    deesser_enabled: Arc<AtomicBool>,
    deesser_control: Arc<Mutex<DeesserControlState>>,
    deesser_dirty: Arc<AtomicBool>,

    /// Hard limiter (brick-wall ceiling)
    limiter: Arc<Mutex<Limiter>>,
    limiter_enabled: Arc<AtomicBool>,
    limiter_control: Arc<Mutex<LimiterControlState>>,
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

    /// Sample rate
    sample_rate: u32,
    /// Actual active output device sample rate
    output_sample_rate: Arc<AtomicU32>,

    /// Input device name
    input_device_name: Option<String>,

    /// Output device name
    output_device_name: Option<String>,

    // === Level Metering (lock-free atomics) ===
    // Stored as f32 bits via to_bits()/from_bits()
    /// Input peak level (after pre-filter, before processing)
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
    /// Last known gate gain (0.0-1.0), used as RT-safe fallback on lock contention.
    gate_fallback_gain: Arc<AtomicU32>,
    /// VAD speech probability (0.0-1.0) for metering
    vad_probability: Arc<AtomicU32>,

    /// Compressor current release time in milliseconds (for metering)
    compressor_current_release_ms: Arc<AtomicU64>,

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
    /// Number of DSP-side jitter recovery events
    output_recovery_count: Arc<AtomicU64>,

    /// Smoothed DSP processing time in microseconds (EMA, 200ms time constant)
    dsp_time_smoothed_us: Arc<AtomicU64>,
    /// Smoothed input buffer fill level (EMA, 200ms time constant)
    smoothed_input_buffer_len: Arc<AtomicU32>,
    /// Smoothed suppressor buffer fill level (EMA, 200ms time constant)
    smoothed_buffer_len: Arc<AtomicU32>,

    /// Dropped samples counter from input ring buffer
    input_dropped: Arc<AtomicU64>,

    /// Total lock contention events in the real-time processing loop.
    lock_contention_count: Arc<AtomicU64>,
    /// Number of times suppressor output contained non-finite samples.
    suppressor_non_finite_count: Arc<AtomicU64>,

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

    /// Create a new audio processor
    pub fn new() -> Self {
        let sample_rate = TARGET_SAMPLE_RATE;

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
                    let vad_auto_gate = VadAutoGate::new(sample_rate, 0.5);
                    if let Ok(mut g) = gate.lock() {
                        g.set_vad_auto_gate(Some(vad_auto_gate));
                    }
                }

                gate
            },
            gate_enabled: Arc::new(AtomicBool::new(true)),
            suppressor: Arc::new(Mutex::new(suppressor)),
            suppressor_enabled: Arc::new(AtomicBool::new(true)),
            suppressor_strength, // Store Arc for PyO3 bindings
            current_model: Arc::new(AtomicU8::new(NoiseModel::RNNoise as u8)),
            eq: Arc::new(Mutex::new(ParametricEQ::new(sample_rate as f64))),
            eq_enabled: Arc::new(AtomicBool::new(true)),
            eq_control: Arc::new(Mutex::new(EqControlState::new())),
            eq_dirty: Arc::new(AtomicBool::new(false)),
            compressor: Arc::new(Mutex::new(Compressor::default_voice(sample_rate as f64))),
            compressor_enabled: Arc::new(AtomicBool::new(true)),
            compressor_control: Arc::new(Mutex::new(CompressorControlState::new())),
            compressor_dirty: Arc::new(AtomicBool::new(false)),
            deesser: Arc::new(Mutex::new(DeEsser::new(sample_rate as f64))),
            deesser_enabled: Arc::new(AtomicBool::new(false)),
            deesser_control: Arc::new(Mutex::new(DeesserControlState::new())),
            deesser_dirty: Arc::new(AtomicBool::new(false)),
            limiter: Arc::new(Mutex::new(Limiter::default_settings(sample_rate as f64))),
            limiter_enabled: Arc::new(AtomicBool::new(true)),
            limiter_control: Arc::new(Mutex::new(LimiterControlState::new())),
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
            sample_rate,
            output_sample_rate: Arc::new(AtomicU32::new(sample_rate)),
            input_device_name: None,
            output_device_name: None,
            // Initialize metering atomics with -infinity (no signal)
            input_peak: Arc::new(AtomicU32::new((-120.0_f32).to_bits())),
            input_rms: Arc::new(AtomicU32::new((-120.0_f32).to_bits())),
            output_peak: Arc::new(AtomicU32::new((-120.0_f32).to_bits())),
            output_rms: Arc::new(AtomicU32::new((-120.0_f32).to_bits())),
            compressor_gain_reduction: Arc::new(AtomicU32::new(0.0_f32.to_bits())),
            deesser_gain_reduction: Arc::new(AtomicU32::new(0.0_f32.to_bits())),
            gate_fallback_gain: Arc::new(AtomicU32::new(1.0_f32.to_bits())),
            vad_probability: Arc::new(AtomicU32::new(0.0_f32.to_bits())),
            compressor_current_release_ms: Arc::new(AtomicU64::new(200)), // Default 200ms
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
            output_recovery_count: Arc::new(AtomicU64::new(0)),
            dsp_time_smoothed_us: Arc::new(AtomicU64::new(0)),
            smoothed_buffer_len: Arc::new(AtomicU32::new(0)),

            // Initialize dropped samples counter
            input_dropped: Arc::new(AtomicU64::new(0)),

            // Initialize lock contention counter
            lock_contention_count: Arc::new(AtomicU64::new(0)),
            suppressor_non_finite_count: Arc::new(AtomicU64::new(0)),

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

    fn ensure_supervisor(&mut self) {
        if self.supervisor_thread.is_some() {
            return;
        }

        self.supervisor_running.store(true, Ordering::Release);

        let running = Arc::clone(&self.running);
        let supervisor_running = Arc::clone(&self.supervisor_running);
        let restart_requested = Arc::clone(&self.restart_requested);
        let recovering = Arc::clone(&self.recovering);
        let last_input_callback_time_us = Arc::clone(&self.last_input_callback_time_us);
        let last_output_callback_time_us = Arc::clone(&self.last_output_callback_time_us);
        let last_output_write_time = Arc::clone(&self.last_output_write_time);
        let last_start_time_us = Arc::clone(&self.last_start_time_us);
        let recovery_state = Arc::clone(&self.recovery_state);
        let recording_active = Arc::clone(&self.recording_active);
        let output_muted = Arc::clone(&self.output_muted);
        let recovery_suppressed = Arc::clone(&self.recovery_suppressed);

        self.supervisor_thread = Some(std::thread::spawn(move || {
            const CHECK_INTERVAL_MS: u64 = 250;
            const CALLBACK_STALL_MS: u64 = 2500;
            const WRITE_STALL_MS: u64 = 3000;
            const STARTUP_GRACE_MS: u64 = 5000;
            const CONSECUTIVE_STALL_CHECKS: u32 = 3;
            let mut consecutive_stalls = 0u32;

            while supervisor_running.load(Ordering::Acquire) {
                std::thread::sleep(std::time::Duration::from_millis(CHECK_INTERVAL_MS));

                if !running.load(Ordering::Acquire) {
                    continue;
                }

                if recovering.load(Ordering::Acquire) || restart_requested.load(Ordering::Acquire) {
                    consecutive_stalls = 0;
                    continue;
                }

                if recovery_suppressed.load(Ordering::Acquire)
                    || recording_active.load(Ordering::Acquire)
                    || output_muted.load(Ordering::Acquire)
                {
                    consecutive_stalls = 0;
                    continue;
                }

                let now = now_micros();
                let last_start = last_start_time_us.load(Ordering::Relaxed);
                if last_start > 0 && now.saturating_sub(last_start) < STARTUP_GRACE_MS * 1000 {
                    consecutive_stalls = 0;
                    continue;
                }

                let last_in = last_input_callback_time_us.load(Ordering::Relaxed);
                let last_out = last_output_callback_time_us.load(Ordering::Relaxed);
                let last_write = last_output_write_time.load(Ordering::Relaxed);

                let input_age_ms = if last_in > 0 {
                    now.saturating_sub(last_in) / 1000
                } else {
                    u64::MAX
                };
                let output_age_ms = if last_out > 0 {
                    now.saturating_sub(last_out) / 1000
                } else {
                    u64::MAX
                };
                let write_age_ms = if last_write > 0 {
                    now.saturating_sub(last_write) / 1000
                } else {
                    u64::MAX
                };

                if input_age_ms > CALLBACK_STALL_MS
                    || output_age_ms > CALLBACK_STALL_MS
                    || write_age_ms > WRITE_STALL_MS
                {
                    consecutive_stalls = consecutive_stalls.saturating_add(1);
                    if consecutive_stalls < CONSECUTIVE_STALL_CHECKS {
                        continue;
                    }
                    let reason = format!(
                        "input_cb_age_ms={}, output_cb_age_ms={}, output_write_age_ms={}",
                        input_age_ms, output_age_ms, write_age_ms
                    );
                    if let Ok(mut state) = recovery_state.lock() {
                        state.last_reason = Some(reason);
                    }
                    restart_requested.store(true, Ordering::Release);
                    consecutive_stalls = 0;
                } else {
                    consecutive_stalls = 0;
                }
            }
        }));
    }

    /// Start audio processing
    ///
    /// # Arguments
    /// * `input_device` - Input device name (None for default)
    /// * `output_device` - Output device name (None for default)
    pub fn start(
        &mut self,
        input_device: Option<&str>,
        output_device: Option<&str>,
    ) -> Result<String, String> {
        self.ensure_supervisor();
        self.restart_requested.store(false, Ordering::Release);
        if !self.recovering.load(Ordering::Acquire) {
            self.restart_backoff_index.store(0, Ordering::Release);
        }

        if self.running.load(Ordering::SeqCst) {
            return Err("Already running".to_string());
        }

        // Ensure any stale recording/mute state is cleared before starting.
        self.recording_active.store(false, Ordering::Release);
        self.output_muted.store(false, Ordering::Release);
        self.raw_recording_pos.store(0, Ordering::Release);
        self.raw_recording_target.store(0, Ordering::Release);
        self.recording_level_db
            .store((-120.0_f32).to_bits(), Ordering::Relaxed);
        if let Ok(mut consumer_guard) = self.raw_recording_consumer.lock() {
            *consumer_guard = None;
        }

        // Create input ring buffer (2 seconds capacity)
        let input_rb = AudioRingBuffer::new(self.sample_rate as usize * 2);
        let (input_producer, input_consumer) = input_rb.split();

        // Create a dedicated raw recording tap buffer. Capacity is fixed so the DSP thread
        // can write without locking during steady-state capture.
        let recording_rb = AudioRingBuffer::new(self.sample_rate as usize * MAX_RECORDING_SECONDS);
        let (mut recording_producer, recording_consumer) = recording_rb.split();
        recording_producer.reset_dropped_count();
        if let Ok(mut consumer_guard) = self.raw_recording_consumer.lock() {
            *consumer_guard = Some(recording_consumer);
        }

        // Capture the dropped counter before producer is moved
        let input_dropped_counter = input_producer.dropped_counter();
        self.input_dropped = input_dropped_counter;

        // Reset the dropped counter at start
        self.input_dropped.store(0, Ordering::Relaxed);
        self.output_underrun_streak.store(0, Ordering::Relaxed);
        self.output_underrun_total.store(0, Ordering::Relaxed);
        self.jitter_dropped_samples.store(0, Ordering::Relaxed);
        self.output_recovery_count.store(0, Ordering::Relaxed);
        self.suppressor_non_finite_count.store(0, Ordering::Relaxed);
        self.last_input_callback_time_us.store(0, Ordering::Relaxed);
        self.last_output_callback_time_us
            .store(0, Ordering::Relaxed);
        self.last_output_write_time.store(0, Ordering::Relaxed);

        // Start audio input
        let last_input_callback_time_us = Arc::clone(&self.last_input_callback_time_us);
        let input = match input_device {
            Some(name) => {
                AudioInput::from_device_name(name, input_producer, last_input_callback_time_us)
            }
            None => AudioInput::from_default_device(input_producer, last_input_callback_time_us),
        }
        .map_err(|e| format!("Failed to start audio input: {}", e))?;

        let input_device_name = input.device_info().name.clone();
        let input_sample_rate_for_thread = input.device_info().sample_rate;

        if let Err(e) = input.start() {
            return Err(format!("Failed to start input stream: {}", e));
        }

        let output_setup = match output_device {
            Some(name) => AudioOutput::from_named_device_setup(name),
            None => AudioOutput::from_default_device_setup(),
        };
        let output_setup = match output_setup {
            Ok(setup) => setup,
            Err(e) => {
                let _ = input.pause();
                return Err(format!("Failed to resolve audio output: {}", e));
            }
        };

        let output_device_name = output_setup.device_info.name.clone();
        let output_sample_rate_for_thread = output_setup.device_info.sample_rate;
        self.output_sample_rate
            .store(output_sample_rate_for_thread, Ordering::Relaxed);

        // Create output ring buffer sized for the actual playback device rate.
        let output_rb = AudioRingBuffer::new(output_sample_rate_for_thread as usize * 2);
        let (output_producer, output_consumer) = output_rb.split();

        // Start audio output
        let recording_active = Arc::clone(&self.recording_active);
        let output_muted = Arc::clone(&self.output_muted);
        let last_output_callback_time_us = Arc::clone(&self.last_output_callback_time_us);
        let output_underrun_streak = Arc::clone(&self.output_underrun_streak);
        let output_underrun_total = Arc::clone(&self.output_underrun_total);
        let output_result = AudioOutput::from_setup(
            output_setup,
            output_consumer,
            recording_active.clone(),
            output_muted.clone(),
            last_output_callback_time_us,
            output_underrun_streak,
            output_underrun_total,
        );
        let output = match output_result {
            Ok(output) => output,
            Err(e) => {
                let _ = input.pause();
                return Err(format!("Failed to start audio output: {}", e));
            }
        };

        let output_prime_samples = duration_samples(output_sample_rate_for_thread, OUTPUT_PRIME_MS);
        let mut prod = output_producer;
        let silence = vec![0.0f32; output_prime_samples];
        prod.write(&silence);
        self.output_buffer_len
            .store(output_prime_samples as u32, Ordering::Relaxed);
        let output_producer = prod;

        if let Err(e) = output.start() {
            let _ = input.pause();
            let _ = output.pause();
            return Err(format!("Failed to start output stream: {}", e));
        }

        self.input_device_name = Some(input_device_name.clone());
        self.output_device_name = Some(output_device_name.clone());
        self.audio_input = Some(input);
        self.audio_output = Some(output);

        // Start processing thread
        self.running.store(true, Ordering::SeqCst);
        self.last_start_time_us
            .store(now_micros(), Ordering::Release);

        let gate = Arc::clone(&self.gate);
        let gate_enabled = Arc::clone(&self.gate_enabled);
        let suppressor = Arc::clone(&self.suppressor);
        let suppressor_enabled = Arc::clone(&self.suppressor_enabled);
        let eq_enabled = Arc::clone(&self.eq_enabled);
        let eq_control = Arc::clone(&self.eq_control);
        let eq_dirty = Arc::clone(&self.eq_dirty);
        let compressor_enabled = Arc::clone(&self.compressor_enabled);
        let compressor_control = Arc::clone(&self.compressor_control);
        let compressor_dirty = Arc::clone(&self.compressor_dirty);
        let deesser_enabled = Arc::clone(&self.deesser_enabled);
        let deesser_control = Arc::clone(&self.deesser_control);
        let deesser_dirty = Arc::clone(&self.deesser_dirty);
        let limiter_enabled = Arc::clone(&self.limiter_enabled);
        let limiter_control = Arc::clone(&self.limiter_control);
        let limiter_dirty = Arc::clone(&self.limiter_dirty);
        let mut output_producer = output_producer;
        let running = Arc::clone(&self.running);
        let bypass = Arc::clone(&self.bypass);

        // Clone metering atomics
        let input_peak = Arc::clone(&self.input_peak);
        let input_rms = Arc::clone(&self.input_rms);
        let output_peak = Arc::clone(&self.output_peak);
        let output_rms = Arc::clone(&self.output_rms);
        let compressor_gain_reduction = Arc::clone(&self.compressor_gain_reduction);
        let deesser_gain_reduction = Arc::clone(&self.deesser_gain_reduction);
        let gate_fallback_gain = Arc::clone(&self.gate_fallback_gain);
        let vad_probability = Arc::clone(&self.vad_probability);
        let compressor_current_release_ms = Arc::clone(&self.compressor_current_release_ms);
        let latency_us = Arc::clone(&self.latency_us);
        let latency_compensation_us = Arc::clone(&self.latency_compensation_us);
        let sample_rate_for_latency = self.sample_rate;
        let output_sample_rate_for_latency = output_sample_rate_for_thread;

        // Clone DSP performance metric atomics
        let dsp_time_us = Arc::clone(&self.dsp_time_us);
        let input_buffer_len = Arc::clone(&self.input_buffer_len);
        let smoothed_input_buffer_len = Arc::clone(&self.smoothed_input_buffer_len);
        let output_buffer_len = Arc::clone(&self.output_buffer_len);
        let suppressor_buffer_len = Arc::clone(&self.suppressor_buffer_len);
        let suppressor_latency_samples = Arc::clone(&self.suppressor_latency_samples);
        let last_output_write_time = Arc::clone(&self.last_output_write_time);
        let dsp_time_smoothed_us = Arc::clone(&self.dsp_time_smoothed_us);
        let smoothed_buffer_len = Arc::clone(&self.smoothed_buffer_len);
        let lock_contention_count = Arc::clone(&self.lock_contention_count);
        let jitter_dropped_samples = Arc::clone(&self.jitter_dropped_samples);
        let output_recovery_count = Arc::clone(&self.output_recovery_count);
        let suppressor_non_finite_count = Arc::clone(&self.suppressor_non_finite_count);
        let recording_active_thread = Arc::clone(&recording_active);
        let suppressor_strength_for_thread = Arc::clone(&self.suppressor_strength);

        // Clone raw recording buffer atomics
        let raw_recording_pos = Arc::clone(&self.raw_recording_pos);
        let raw_recording_target = Arc::clone(&self.raw_recording_target);
        let recording_level_db = Arc::clone(&self.recording_level_db);

        let handle = std::thread::spawn(move || {
            // Set high thread priority for real-time audio processing
            if let Err(_e) = set_current_thread_priority(ThreadPriority::Max) {
                processor_debug_log!("Warning: Could not set audio thread priority: {:?}", _e);
            }

            let mut consumer = input_consumer;
            let mut input_buffer = vec![0.0f32; 2048];
            let mut temp_buffer = vec![0.0f32; 4096];
            let mut rnnoise_output = vec![0.0f32; 2048];
            let mut resample_input: Vec<f64> = Vec::with_capacity(65536);
            let mut resample_read_pos: usize = 0;
            const RESAMPLER_CHUNK_SIZE: usize = 1024;
            let mut resampler = if input_sample_rate_for_thread != sample_rate_for_latency {
                processor_debug_log!(
                    "[PROCESSING] Input device sample rate {} Hz; resampling to {} Hz in DSP thread",
                    input_sample_rate_for_thread, sample_rate_for_latency
                );
                let ratio = sample_rate_for_latency as f64 / input_sample_rate_for_thread as f64;
                let sinc_len = 128;
                let window = WindowFunction::BlackmanHarris2;
                let params = SincInterpolationParameters {
                    sinc_len,
                    f_cutoff: calculate_cutoff(sinc_len, window),
                    interpolation: SincInterpolationType::Cubic,
                    oversampling_factor: 256,
                    window,
                };
                match SincFixedIn::<f64>::new(ratio, 1.2, params, RESAMPLER_CHUNK_SIZE, 1) {
                    Ok(r) => Some(r),
                    Err(_e) => {
                        processor_debug_log!(
                            "[PROCESSING] WARNING: Failed to init resampler ({}). Processing raw input rate.",
                            _e
                        );
                        None
                    }
                }
            } else {
                None
            };
            let mut resampler_out = resampler.as_ref().map(|r| r.output_buffer_allocate(false));
            let mut output_resample_input: Vec<f64> = Vec::with_capacity(65536);
            let mut output_resample_read_pos: usize = 0;
            let mut output_resampler = if output_sample_rate_for_latency != sample_rate_for_latency
            {
                processor_debug_log!(
                    "[PROCESSING] Output device sample rate {} Hz; resampling from {} Hz in DSP thread",
                    output_sample_rate_for_latency, sample_rate_for_latency
                );
                let ratio = output_sample_rate_for_latency as f64 / sample_rate_for_latency as f64;
                let sinc_len = 128;
                let window = WindowFunction::BlackmanHarris2;
                let params = SincInterpolationParameters {
                    sinc_len,
                    f_cutoff: calculate_cutoff(sinc_len, window),
                    interpolation: SincInterpolationType::Cubic,
                    oversampling_factor: 256,
                    window,
                };
                match SincFixedIn::<f64>::new(ratio, 1.2, params, RESAMPLER_CHUNK_SIZE, 1) {
                    Ok(r) => Some(r),
                    Err(_e) => {
                        processor_debug_log!(
                            "[PROCESSING] WARNING: Failed to init output resampler ({}). Writing 48kHz output directly.",
                            _e
                        );
                        None
                    }
                }
            } else {
                None
            };
            let mut output_resampler_out = output_resampler
                .as_ref()
                .map(|r| r.output_buffer_allocate(false));
            let mut eq_rt = ParametricEQ::new(sample_rate_for_latency as f64);
            let mut compressor_rt = Compressor::default_voice(sample_rate_for_latency as f64);
            let mut deesser_rt = DeEsser::new(sample_rate_for_latency as f64);
            let mut limiter_rt = Limiter::default_settings(sample_rate_for_latency as f64);
            if let Ok(control) = eq_control.lock() {
                apply_eq_control(&mut eq_rt, &control);
            }
            if let Ok(control) = compressor_control.lock() {
                apply_compressor_control(&mut compressor_rt, &control);
            }
            if let Ok(control) = deesser_control.lock() {
                apply_deesser_control(&mut deesser_rt, &control);
            }
            if let Ok(control) = limiter_control.lock() {
                apply_limiter_control(&mut limiter_rt, &control);
            }
            let mut output_resampled_scratch: Vec<f32> = Vec::with_capacity(8192);

            // DC Blocker state variables
            // This removes electrical DC offset which causes static/clicks
            let mut dc_x1: f32 = 0.0;
            let mut dc_y1: f32 = 0.0;
            const DC_COEFF: f32 = 0.995; // DC blocking coefficient

            // Pre-filter at 80Hz to remove rumble before gate/suppressor stages.
            let mut pre_filter = Biquad::new(
                BiquadType::HighPass,
                80.0,
                0.0,
                0.707,
                sample_rate_for_latency as f64,
            );

            // Metering state (IIR smoothing for RMS)
            let mut input_rms_acc: f32 = 0.0;
            let mut output_rms_acc: f32 = 0.0;
            const METER_COEFF: f32 = 0.99; // ~100ms time constant at 48kHz

            // Latency tracking
            let mut last_latency_update = Instant::now();
            let mut last_heartbeat = Instant::now();
            let latency_update_interval = std::time::Duration::from_millis(100); // Update every 100ms
            const HEARTBEAT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);
            const STALL_THRESHOLD_MS: u64 = 3000; // 3 seconds without write = stall
            const SUPPRESSOR_STARVATION_MS: u64 = 400; // Reset suppressor if no output this long
            const SUPPRESSOR_RECOVERY_COOLDOWN_MS: u64 = 2000;
            let mut last_suppressor_recovery =
                Instant::now() - std::time::Duration::from_millis(SUPPRESSOR_RECOVERY_COOLDOWN_MS);

            // Helper: compute peak and RMS from buffer, update atomics
            let measure_levels = |buffer: &[f32],
                                  rms_acc: &mut f32,
                                  peak_atomic: &AtomicU32,
                                  rms_atomic: &AtomicU32| {
                let mut peak: f32 = 0.0;
                for &sample in buffer.iter() {
                    let abs = sample.abs();
                    if abs > peak {
                        peak = abs;
                    }
                    // IIR RMS accumulator
                    *rms_acc = METER_COEFF * *rms_acc + (1.0 - METER_COEFF) * (sample * sample);
                }
                // Convert to dB
                let peak_db = if peak > 0.0 {
                    20.0 * peak.log10()
                } else {
                    -120.0
                };
                let rms_db = if *rms_acc > 0.0 {
                    10.0 * rms_acc.log10() // RMS is sqrt of mean squared, so 10*log10 not 20
                } else {
                    -120.0
                };
                peak_atomic.store(peak_db.to_bits(), Ordering::Relaxed);
                rms_atomic.store(rms_db.to_bits(), Ordering::Relaxed);
            };

            // Helper: apply downstream DSP stages (de-esser -> EQ -> compressor -> limiter).
            let mut apply_downstream_chain = |buffer: &mut [f32]| {
                if deesser_dirty.swap(false, Ordering::AcqRel) {
                    if let Ok(control) = deesser_control.lock() {
                        apply_deesser_control(&mut deesser_rt, &control);
                    }
                }
                if eq_dirty.swap(false, Ordering::AcqRel) {
                    if let Ok(control) = eq_control.lock() {
                        apply_eq_control(&mut eq_rt, &control);
                    }
                }
                if compressor_dirty.swap(false, Ordering::AcqRel) {
                    if let Ok(control) = compressor_control.lock() {
                        apply_compressor_control(&mut compressor_rt, &control);
                    }
                }
                if limiter_dirty.swap(false, Ordering::AcqRel) {
                    if let Ok(control) = limiter_control.lock() {
                        apply_limiter_control(&mut limiter_rt, &control);
                    }
                }

                // Stage 3: De-esser
                if deesser_enabled.load(Ordering::Acquire) {
                    deesser_rt.process_block_inplace(buffer);
                    deesser_gain_reduction.store(
                        deesser_rt.current_gain_reduction_db().to_bits(),
                        Ordering::Relaxed,
                    );
                } else {
                    deesser_gain_reduction.store(0.0_f32.to_bits(), Ordering::Relaxed);
                }

                // Stage 4: EQ
                if eq_enabled.load(Ordering::Acquire) {
                    eq_rt.process_block_inplace(buffer);
                }

                // Stage 5: Compressor
                if compressor_enabled.load(Ordering::Acquire) {
                    compressor_rt.process_block_inplace(buffer);
                    compressor_gain_reduction.store(
                        (compressor_rt.current_gain_reduction() as f32).to_bits(),
                        Ordering::Relaxed,
                    );
                    let current_release = compressor_rt.current_release_time();
                    compressor_current_release_ms
                        .store((current_release * 10.0) as u64, Ordering::Relaxed);
                } else {
                    compressor_gain_reduction.store(0.0_f32.to_bits(), Ordering::Relaxed);
                    compressor_current_release_ms
                        .store(COMPRESSOR_DEFAULT_RELEASE_TENTH_MS, Ordering::Relaxed);
                }

                // Stage 6: Hard Limiter (LAST - safety ceiling)
                if limiter_enabled.load(Ordering::Acquire) {
                    limiter_rt.process_block_inplace(buffer);
                }
            };

            // Time-based EMA smoothing for metrics (200ms time constant)
            const TAU_MS: f32 = 200.0; // Time constant in milliseconds
            let dt_ms = 10.0; // Processing interval (480 samples @ 48kHz)
            let alpha = 1.0 - (-dt_ms / TAU_MS).exp(); // Smoothing factor

            let smooth_dsp_time = |raw_us: u64, prev_smoothed: u64| -> u64 {
                let raw_f = raw_us as f32;
                let prev_f = prev_smoothed as f32;
                (alpha * raw_f + (1.0 - alpha) * prev_f) as u64
            };

            let smooth_buffer = |raw: u32, prev_smoothed: u32| -> u32 {
                let raw_f = raw as f32;
                let prev_f = prev_smoothed as f32;
                (alpha * raw_f + (1.0 - alpha) * prev_f) as u32
            };

            // Helper: update last write time (call after successful output write)
            let update_write_time = || {
                last_output_write_time.store(now_micros(), Ordering::Relaxed);
            };

            // Jitter-buffer write control: keep output queue in a healthy range.
            let output_target_low_samples =
                duration_samples(output_sample_rate_for_latency, OUTPUT_PRIME_MS);
            let output_target_high_samples =
                duration_samples(output_sample_rate_for_latency, OUTPUT_TARGET_HIGH_MS);
            let output_hard_backlog_samples =
                duration_samples(output_sample_rate_for_latency, OUTPUT_HARD_BACKLOG_MS);
            const OUTPUT_MAX_CATCHUP_RATIO: f32 = 1.08; // Up to 8% faster in normal backlog.
            const OUTPUT_MAX_EMERGENCY_CATCHUP_RATIO: f32 = 1.20; // Up to 20% in hard backlog.
            const CATCHUP_SCRATCH_CAPACITY: usize = 4096;
            let mut catchup_scratch: Vec<f32> = vec![0.0; CATCHUP_SCRATCH_CAPACITY];
            let mut write_with_jitter_control = |samples: &[f32]| {
                let write_source = if let Some(output_resampler) = output_resampler.as_mut() {
                    output_resampled_scratch.clear();

                    for &sample in samples {
                        if output_resample_input.len() == output_resample_input.capacity() {
                            if output_resample_read_pos > 0 {
                                let unread = output_resample_input.len() - output_resample_read_pos;
                                output_resample_input.copy_within(output_resample_read_pos.., 0);
                                output_resample_input.truncate(unread);
                                output_resample_read_pos = 0;
                            } else {
                                let keep = output_resample_input.len() / 2;
                                let drop = output_resample_input.len() - keep;
                                output_resample_input.copy_within(drop.., 0);
                                output_resample_input.truncate(keep);
                            }
                        }
                        output_resample_input.push(sample as f64);
                    }

                    let input_frames_needed = output_resampler.input_frames_next();
                    while output_resample_input
                        .len()
                        .saturating_sub(output_resample_read_pos)
                        >= input_frames_needed
                    {
                        if let Some(outbuf) = output_resampler_out.as_mut() {
                            let in_slices = [&output_resample_input[output_resample_read_pos
                                ..output_resample_read_pos + input_frames_needed]];
                            if let Ok((_nbr_in, nbr_out)) =
                                output_resampler.process_into_buffer(&in_slices, outbuf, None)
                            {
                                output_resampled_scratch.extend(
                                    outbuf[0].iter().take(nbr_out).map(|&sample| sample as f32),
                                );
                            }
                        }
                        output_resample_read_pos += input_frames_needed;
                    }

                    if output_resample_read_pos > RESAMPLE_COMPACT_THRESHOLD
                        && output_resample_read_pos * 2 >= output_resample_input.len()
                    {
                        let unread = output_resample_input.len() - output_resample_read_pos;
                        output_resample_input.copy_within(output_resample_read_pos.., 0);
                        output_resample_input.truncate(unread);
                        output_resample_read_pos = 0;
                    }

                    output_resampled_scratch.as_slice()
                } else {
                    samples
                };

                if write_source.is_empty() {
                    return;
                }

                let capacity = output_producer.capacity();
                let free = output_producer.free_len();
                let fill = capacity.saturating_sub(free);

                // Pro catch-up: smoothly time-compress this block when backlog is high.
                // This avoids discontinuities/pops from hard sample drops.
                let mut write_slice = write_source;
                if fill > output_target_high_samples {
                    let excess = fill - output_target_high_samples;
                    let zone = output_hard_backlog_samples
                        .saturating_sub(output_target_high_samples)
                        .max(1);
                    let severity = (excess as f32 / zone as f32).clamp(0.0, 1.0);
                    let max_ratio = if fill >= output_hard_backlog_samples {
                        OUTPUT_MAX_EMERGENCY_CATCHUP_RATIO
                    } else {
                        OUTPUT_MAX_CATCHUP_RATIO
                    };
                    let catchup_ratio = 1.0 + severity * (max_ratio - 1.0);
                    let out_len = ((write_source.len() as f32) / catchup_ratio)
                        .round()
                        .max(1.0) as usize;
                    let out_len = out_len.min(CATCHUP_SCRATCH_CAPACITY);

                    if out_len < write_source.len() {
                        let max_src = (write_source.len() - 1) as f32;
                        for (i, out_sample) in catchup_scratch.iter_mut().enumerate().take(out_len)
                        {
                            let src_pos = (i as f32 * catchup_ratio).min(max_src);
                            let idx = src_pos.floor() as isize;
                            let frac = src_pos - idx as f32;

                            // 4-point Hermite interpolation with clamped endpoints.
                            let clamp_idx = |k: isize| -> usize {
                                k.clamp(0, (write_source.len() - 1) as isize) as usize
                            };
                            let y0 = write_source[clamp_idx(idx - 1)];
                            let y1 = write_source[clamp_idx(idx)];
                            let y2 = write_source[clamp_idx(idx + 1)];
                            let y3 = write_source[clamp_idx(idx + 2)];

                            // Catmull-Rom style cubic Hermite coefficients.
                            let c0 = y1;
                            let c1 = 0.5 * (y2 - y0);
                            let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
                            let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
                            *out_sample = ((c3 * frac + c2) * frac + c1) * frac + c0;
                        }

                        let skipped = write_source.len() - out_len;
                        jitter_dropped_samples.fetch_add(skipped as u64, Ordering::Relaxed);
                        output_recovery_count.fetch_add(1, Ordering::Relaxed);
                        write_slice = &catchup_scratch[..out_len];
                    }
                }

                // Low watermark currently informational; could be used for adaptive refill.
                let _below_low_target = fill < output_target_low_samples;

                if !write_slice.is_empty() {
                    let written = output_producer.write(write_slice);
                    if written > 0 {
                        update_write_time();
                    }
                }

                let new_fill = output_producer
                    .capacity()
                    .saturating_sub(output_producer.free_len());
                output_buffer_len.store(new_fill as u32, Ordering::Relaxed);
            };

            // Run entire processing loop with denormals flushed to zero
            // This prevents tiny floating point values from causing CPU stalls and audio artifacts
            // SAFETY: This only modifies floating point control flags for this thread
            unsafe {
                no_denormals::no_denormals(|| {
                    while running.load(Ordering::SeqCst) {
                        // Record input buffer fill level (samples waiting to be processed)
                        let raw_input_len = consumer.len() as u32;
                        input_buffer_len.store(raw_input_len, Ordering::Relaxed);
                        let smoothed_input = smooth_buffer(
                            raw_input_len,
                            smoothed_input_buffer_len.load(Ordering::Relaxed),
                        );
                        smoothed_input_buffer_len.store(smoothed_input, Ordering::Relaxed);

                        // Read audio samples
                        let n_raw = consumer.read(&mut input_buffer);

                        if n_raw > 0 {
                            let n = if let Some(resampler) = resampler.as_mut() {
                                for &sample in input_buffer[..n_raw].iter() {
                                    if resample_input.len() == resample_input.capacity() {
                                        // Rare overflow guard: compact unread samples in-place first.
                                        if resample_read_pos > 0 {
                                            let unread = resample_input.len() - resample_read_pos;
                                            resample_input.copy_within(resample_read_pos.., 0);
                                            resample_input.truncate(unread);
                                            resample_read_pos = 0;
                                        } else {
                                            // If still full with no consumed prefix, drop oldest half.
                                            let keep = resample_input.len() / 2;
                                            let drop = resample_input.len() - keep;
                                            resample_input.copy_within(drop.., 0);
                                            resample_input.truncate(keep);
                                        }
                                    }
                                    resample_input.push(sample as f64);
                                }

                                let mut produced = 0usize;
                                let input_frames_needed = resampler.input_frames_next();
                                while resample_input.len().saturating_sub(resample_read_pos)
                                    >= input_frames_needed
                                    && produced < temp_buffer.len()
                                {
                                    if let Some(outbuf) = resampler_out.as_mut() {
                                        let in_slices = [&resample_input[resample_read_pos
                                            ..resample_read_pos + input_frames_needed]];
                                        if let Ok((_nbr_in, nbr_out)) =
                                            resampler.process_into_buffer(&in_slices, outbuf, None)
                                        {
                                            let channel_out = &outbuf[0];
                                            for &sample in channel_out.iter().take(nbr_out) {
                                                if produced >= temp_buffer.len() {
                                                    break;
                                                }
                                                temp_buffer[produced] = sample as f32;
                                                produced += 1;
                                            }
                                        }
                                    }
                                    // Advance by one consumed input block.
                                    resample_read_pos += input_frames_needed;
                                }

                                // Compact consumed prefix occasionally (amortized O(n), infrequent).
                                if resample_read_pos > RESAMPLE_COMPACT_THRESHOLD
                                    && resample_read_pos * 2 >= resample_input.len()
                                {
                                    let unread = resample_input.len() - resample_read_pos;
                                    resample_input.copy_within(resample_read_pos.., 0);
                                    resample_input.truncate(unread);
                                    resample_read_pos = 0;
                                }
                                produced
                            } else {
                                let copy_n = n_raw.min(temp_buffer.len());
                                temp_buffer[..copy_n].copy_from_slice(&input_buffer[..copy_n]);
                                copy_n
                            };

                            if n == 0 {
                                std::thread::sleep(std::time::Duration::from_micros(
                                    PROCESS_IDLE_SLEEP_US,
                                ));
                                continue;
                            }

                            let buffer = &mut temp_buffer[..n];

                            // Start DSP timing
                            let dsp_start = Instant::now();

                            // Clamp any out-of-range samples to prevent distortion
                            for sample in buffer.iter_mut() {
                                if !sample.is_finite() {
                                    *sample = 0.0;
                                } else {
                                    *sample = (*sample).clamp(-1.0, 1.0);
                                }
                            }

                            // === PRE-PROCESSING: Clean the input before DSP chain ===
                            // This removes fan rumble and DC offset that cause RNNoise artifacts
                            for sample in buffer.iter_mut() {
                                // A. DC Blocker (removes electrical DC offset causing static/clicks)
                                let input = *sample;
                                let output = input - dc_x1 + DC_COEFF * dc_y1;
                                dc_x1 = input;
                                dc_y1 = output;
                                *sample = output;

                                // B. High-Pass Filter at 80Hz (kills fan rumble before RNNoise)
                                *sample = pre_filter.process_sample(*sample);
                            }
                            // === END PRE-PROCESSING ===

                            // Measure INPUT levels (after pre-filter, before main processing)
                            measure_levels(buffer, &mut input_rms_acc, &input_peak, &input_rms);

                            // === RAW AUDIO RECORDING TAP (for calibration) ===
                            // Capture audio AFTER pre-filter, BEFORE noise gate
                            // This is the raw microphone response needed for EQ analysis
                            if recording_active.load(Ordering::Relaxed) {
                                let target = raw_recording_target.load(Ordering::Acquire);
                                let pos = raw_recording_pos.load(Ordering::Acquire);
                                if pos < target {
                                    let remaining = target - pos;
                                    let to_copy = n.min(remaining);
                                    let written = recording_producer.write(&buffer[..to_copy]);
                                    let new_pos = pos.saturating_add(written);
                                    raw_recording_pos.store(new_pos, Ordering::Release);

                                    let window_len =
                                        (sample_rate_for_latency as usize / 10).max(1);
                                    let level_start = to_copy.saturating_sub(window_len);
                                    let level_slice = &buffer[level_start..to_copy];
                                    let level_rms = if level_slice.is_empty() {
                                        -120.0
                                    } else {
                                        let sum_sq: f32 =
                                            level_slice.iter().map(|sample| sample * sample).sum();
                                        let rms = (sum_sq / level_slice.len() as f32).sqrt();
                                        if rms > 1e-6 {
                                            20.0 * rms.log10()
                                        } else {
                                            -120.0
                                        }
                                    };
                                    recording_level_db
                                        .store(level_rms.to_bits(), Ordering::Relaxed);
                                }
                            }
                            // === END RECORDING TAP ===

                            if bypass.load(Ordering::SeqCst) {
                                // Bypass mode: measure output = input, send directly
                                measure_levels(
                                    buffer,
                                    &mut output_rms_acc,
                                    &output_peak,
                                    &output_rms,
                                );
                                compressor_gain_reduction
                                    .store(0.0_f32.to_bits(), Ordering::Relaxed);
                                deesser_gain_reduction.store(0.0_f32.to_bits(), Ordering::Relaxed);
                                compressor_current_release_ms
                                    .store(COMPRESSOR_DEFAULT_RELEASE_TENTH_MS, Ordering::Relaxed); // Default 200ms
                                suppressor_buffer_len.store(0, Ordering::Relaxed);

                                write_with_jitter_control(buffer);
                                // Record DSP processing time
                                let raw_dsp_us = dsp_start.elapsed().as_micros() as u64;
                                let prev_smoothed = dsp_time_smoothed_us.load(Ordering::Relaxed);
                                let smoothed = smooth_dsp_time(raw_dsp_us, prev_smoothed);
                                dsp_time_us.store(raw_dsp_us, Ordering::Relaxed);
                                dsp_time_smoothed_us.store(smoothed, Ordering::Relaxed);
                            } else {
                                // Stage 1: Noise Gate
                                if gate_enabled.load(Ordering::Acquire) {
                                    if let Ok(mut g) = gate.try_lock() {
                                        g.process_block_inplace(buffer);
                                        gate_fallback_gain.store(
                                            g.current_gain().clamp(0.0, 1.0).to_bits(),
                                            Ordering::Relaxed,
                                        );

                                        // Update VAD probability for metering (if available)
                                        #[cfg(feature = "vad")]
                                        {
                                            let prob = g.get_vad_probability();
                                            vad_probability
                                                .store(prob.to_bits(), Ordering::Relaxed);
                                        }
                                    } else {
                                        lock_contention_count.fetch_add(1, Ordering::Relaxed);
                                        // RT-safe fallback: apply last known gate gain instead of
                                        // bypassing the stage entirely for this block.
                                        let gain = f32::from_bits(
                                            gate_fallback_gain.load(Ordering::Relaxed),
                                        )
                                        .clamp(0.0, 1.0);
                                        if gain < GATE_FALLBACK_APPLY_THRESHOLD {
                                            for sample in buffer.iter_mut() {
                                                *sample *= gain;
                                            }
                                        }
                                    }
                                }

                                // Stage 2: Noise Suppression (RNNoise or DeepFilterNet)
                                let use_suppressor = suppressor_enabled.load(Ordering::Acquire);
                                if use_suppressor {
                                    if let Ok(mut s) = suppressor.try_lock() {
                                        // Always feed suppressor first so frame accumulation is correct.
                                        s.push_samples(buffer);
                                        s.process_frames();

                                        // Track suppressor internal buffer fill level after processing.
                                        let suppressor_buffered =
                                            s.pending_input() + s.available_samples();
                                        let suppressor_latency =
                                            s.latency_samples() + s.pending_input();
                                        let raw_buffer = suppressor_buffered as u32;
                                        let prev_smoothed =
                                            smoothed_buffer_len.load(Ordering::Relaxed);
                                        let smoothed = smooth_buffer(raw_buffer, prev_smoothed);
                                        suppressor_buffer_len.store(raw_buffer, Ordering::Relaxed);
                                        suppressor_latency_samples
                                            .store(suppressor_latency as u32, Ordering::Relaxed);
                                        smoothed_buffer_len.store(smoothed, Ordering::Relaxed);

                                        let available = s.available_samples();
                                        if available == 0 {
                                            let pending = s.pending_input();
                                            if pending >= RNNOISE_FRAME_SIZE
                                                && !recording_active_thread.load(Ordering::Relaxed)
                                            {
                                                let last_write =
                                                    last_output_write_time.load(Ordering::Relaxed);
                                                if last_write > 0 {
                                                    let now = now_micros();
                                                    let since_write_ms =
                                                        now.saturating_sub(last_write) / 1000;
                                                    if since_write_ms > SUPPRESSOR_STARVATION_MS
                                                        && last_suppressor_recovery
                                                            .elapsed()
                                                            .as_millis()
                                                            as u64
                                                            > SUPPRESSOR_RECOVERY_COOLDOWN_MS
                                                    {
                                                        processor_debug_log!(
                                                            "[PROCESSING] WARNING: Suppressor starvation detected (pending={}, no output for {} ms). Soft-resetting suppressor.",
                                                            pending, since_write_ms
                                                        );
                                                        s.soft_reset();
                                                        last_suppressor_recovery = Instant::now();
                                                    }
                                                }
                                            }
                                        }
                                        if available > 0 {
                                            let count = available.min(rnnoise_output.len());
                                            let processed =
                                                s.pop_samples_into(&mut rnnoise_output[..count]);
                                            let output_slice = &mut rnnoise_output[..processed];

                                            let mut detected_non_finite = false;
                                            for sample in output_slice.iter_mut() {
                                                if !sample.is_finite() {
                                                    *sample = 0.0;
                                                    detected_non_finite = true;
                                                }
                                            }
                                            if detected_non_finite {
                                                suppressor_non_finite_count
                                                    .fetch_add(1, Ordering::Relaxed);
                                                processor_debug_log!(
                                                    "[PROCESSING] WARNING: Non-finite suppressor output detected. Reinitializing suppressor state."
                                                );
                                                let was_enabled = s.is_enabled();
                                                let model = s.model_type();
                                                *s = NoiseSuppressionEngine::new(
                                                    model,
                                                    Arc::clone(&suppressor_strength_for_thread),
                                                );
                                                s.set_enabled(was_enabled);
                                            }

                                            apply_downstream_chain(output_slice);

                                            // Measure OUTPUT levels
                                            measure_levels(
                                                output_slice,
                                                &mut output_rms_acc,
                                                &output_peak,
                                                &output_rms,
                                            );

                                            // Send processed samples to output
                                            write_with_jitter_control(output_slice);
                                        }

                                        // Record DSP processing time
                                        let raw_dsp_us = dsp_start.elapsed().as_micros() as u64;
                                        let prev_smoothed =
                                            dsp_time_smoothed_us.load(Ordering::Relaxed);
                                        let smoothed = smooth_dsp_time(raw_dsp_us, prev_smoothed);
                                        dsp_time_us.store(raw_dsp_us, Ordering::Relaxed);
                                        dsp_time_smoothed_us.store(smoothed, Ordering::Relaxed);
                                    } else {
                                        // RT-safe deterministic fallback: suppressor temporarily unavailable.
                                        // Process current block through downstream stages without suppressor
                                        // instead of dropping work for this callback.
                                        lock_contention_count.fetch_add(1, Ordering::Relaxed);
                                        suppressor_buffer_len.store(0, Ordering::Relaxed);
                                        suppressor_latency_samples.store(0, Ordering::Relaxed);

                                        apply_downstream_chain(buffer);

                                        measure_levels(
                                            buffer,
                                            &mut output_rms_acc,
                                            &output_peak,
                                            &output_rms,
                                        );

                                        write_with_jitter_control(buffer);

                                        let raw_dsp_us = dsp_start.elapsed().as_micros() as u64;
                                        let prev_smoothed =
                                            dsp_time_smoothed_us.load(Ordering::Relaxed);
                                        let smoothed = smooth_dsp_time(raw_dsp_us, prev_smoothed);
                                        dsp_time_us.store(raw_dsp_us, Ordering::Relaxed);
                                        dsp_time_smoothed_us.store(smoothed, Ordering::Relaxed);
                                    }
                                } else {
                                    // Suppressor disabled: clear buffer counter
                                    suppressor_buffer_len.store(0, Ordering::Relaxed);
                                    suppressor_latency_samples.store(0, Ordering::Relaxed);
                                    // Suppressor disabled: apply remaining stages directly

                                    apply_downstream_chain(buffer);

                                    // Measure OUTPUT levels
                                    measure_levels(
                                        buffer,
                                        &mut output_rms_acc,
                                        &output_peak,
                                        &output_rms,
                                    );

                                    // Send to output
                                    write_with_jitter_control(buffer);
                                    // Record DSP processing time
                                    let raw_dsp_us = dsp_start.elapsed().as_micros() as u64;
                                    let prev_smoothed =
                                        dsp_time_smoothed_us.load(Ordering::Relaxed);
                                    let smoothed = smooth_dsp_time(raw_dsp_us, prev_smoothed);
                                    dsp_time_us.store(raw_dsp_us, Ordering::Relaxed);
                                    dsp_time_smoothed_us.store(smoothed, Ordering::Relaxed);
                                }
                            }
                            // Update latency periodically
                            if last_latency_update.elapsed() >= latency_update_interval {
                                last_latency_update = Instant::now();

                                let output_buffer_samples =
                                    output_buffer_len.load(Ordering::Relaxed) as u64;
                                let output_latency_us = samples_to_micros(
                                    output_buffer_samples,
                                    output_sample_rate_for_latency,
                                );

                                let suppressor_latency_samples = if suppressor_enabled
                                    .load(Ordering::Acquire)
                                {
                                    suppressor_latency_samples.load(Ordering::Relaxed) as u64
                                } else {
                                    0
                                };
                                let suppressor_latency_us = samples_to_micros(
                                    suppressor_latency_samples,
                                    sample_rate_for_latency,
                                );

                                let compensation_us =
                                    latency_compensation_us.load(Ordering::Relaxed);
                                let total_latency = output_latency_us
                                    .saturating_add(suppressor_latency_us)
                                    .saturating_add(compensation_us);
                                latency_us.store(total_latency, Ordering::Relaxed);
                            }
                        } else {
                            // No data available, sleep briefly to avoid busy-wait
                            if last_heartbeat.elapsed() >= HEARTBEAT_INTERVAL {
                                last_heartbeat = Instant::now();

                                // Check for output stall
                                let last_write = last_output_write_time.load(Ordering::Relaxed);
                                let now = now_micros();
                                let time_since_write_ms = now.saturating_sub(last_write) / 1000;

                                if last_write > 0 && time_since_write_ms > STALL_THRESHOLD_MS {
                                    processor_debug_log!(
                                        "[PROCESSING] WARNING: Output stall detected! No write for {} ms (input_buf={}, suppressor_enabled={})",
                                        time_since_write_ms,
                                        raw_input_len,
                                        suppressor_enabled.load(Ordering::Acquire)
                                    );
                                }
                            }
                            std::thread::sleep(std::time::Duration::from_micros(
                                PROCESS_IDLE_SLEEP_US,
                            ));
                        }
                    }
                }); // End no_denormals block
            } // End unsafe block
        });

        self.process_thread = Some(handle);

        Ok(format!(
            "Started: {} -> {}",
            input_device_name, output_device_name
        ))
    }

    /// Stop audio processing
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        self.restart_requested.store(false, Ordering::Release);
        self.recovering.store(false, Ordering::Release);
        self.last_start_time_us.store(0, Ordering::Release);

        if let Some(handle) = self.process_thread.take() {
            let _ = handle.join();
        }

        if let Some(input) = &self.audio_input {
            let _ = input.pause();
        }

        if let Some(output) = &self.audio_output {
            let _ = output.pause();
        }

        self.audio_input = None;
        self.audio_output = None;

        // Ensure output is unmuted and recording state is cleared.
        self.recording_active.store(false, Ordering::Release);
        self.output_muted.store(false, Ordering::Release);
        self.raw_recording_pos.store(0, Ordering::Release);
        self.raw_recording_target.store(0, Ordering::Release);
        self.recording_level_db
            .store((-120.0_f32).to_bits(), Ordering::Relaxed);
        self.suppressor_latency_samples.store(0, Ordering::Relaxed);
        if let Ok(mut consumer_guard) = self.raw_recording_consumer.lock() {
            *consumer_guard = None;
        }

        // Reinitialize suppressor state so stop/start can recover from poisoned model state.
        if let Ok(mut s) = self.suppressor.lock() {
            let was_enabled = s.is_enabled();
            let model = s.model_type();
            *s = NoiseSuppressionEngine::new(model, Arc::clone(&self.suppressor_strength));
            s.set_enabled(was_enabled);
        }

        // Reset DSP state so stop/start can recover from stuck envelopes.
        if let Ok(mut g) = self.gate.lock() {
            g.reset();
        }
        if let Ok(mut e) = self.eq.lock() {
            e.reset();
        }
        if let Ok(mut c) = self.compressor.lock() {
            c.reset();
        }
        if let Ok(mut d) = self.deesser.lock() {
            d.reset();
        }
        if let Ok(mut l) = self.limiter.lock() {
            l.reset();
        }
    }

    /// Service pending stream recovery requests.
    ///
    /// Returns:
    /// - None: no recovery attempt was made
    /// - Some(true): recovery succeeded
    /// - Some(false): recovery attempt failed
    pub fn service_recovery(&mut self) -> Option<bool> {
        if self.recovery_suppressed.load(Ordering::Acquire) {
            return None;
        }

        if !self.restart_requested.load(Ordering::Acquire) {
            return None;
        }

        let now = now_micros();
        let last_attempt = self.last_restart_attempt_us.load(Ordering::Acquire);
        let backoff_idx = self.restart_backoff_index.load(Ordering::Acquire);
        let backoff_ms = match backoff_idx {
            0 => 0,
            1 => 2000,
            2 => 5000,
            _ => 10000,
        };

        if last_attempt > 0 && now.saturating_sub(last_attempt) < backoff_ms * 1000 {
            return None;
        }

        self.restart_requested.store(false, Ordering::Release);
        self.recovering.store(true, Ordering::Release);
        self.last_restart_attempt_us.store(now, Ordering::Release);

        let input_name = self.input_device_name.clone();
        let output_name = self.output_device_name.clone();

        self.stop();

        let mut success = false;
        let mut last_error: Option<String> = None;

        match self.start(input_name.as_deref(), output_name.as_deref()) {
            Ok(_) => {
                success = true;
            }
            Err(err) => {
                last_error = Some(format!("Restart failed for selected devices: {}", err));
                match self.start(None, None) {
                    Ok(_) => {
                        success = true;
                    }
                    Err(fallback_err) => {
                        last_error = Some(format!(
                            "Restart failed for selected + default devices: {}",
                            fallback_err
                        ));
                    }
                }
            }
        }

        if let Ok(mut state) = self.recovery_state.lock() {
            if success {
                state.last_error = None;
                state.restart_count = state.restart_count.saturating_add(1);
            } else {
                state.last_error = last_error.clone();
            }
        }

        if success {
            self.restart_backoff_index.store(0, Ordering::Release);
        } else {
            let next = (backoff_idx + 1).min(3);
            self.restart_backoff_index.store(next, Ordering::Release);
            self.restart_requested.store(true, Ordering::Release);
        }

        self.recovering.store(false, Ordering::Release);
        Some(success)
    }

    /// Whether a restart has been requested by the supervisor.
    pub fn is_recovery_requested(&self) -> bool {
        self.restart_requested.load(Ordering::Acquire)
    }

    /// Whether a recovery attempt is in progress.
    pub fn is_recovering(&self) -> bool {
        self.recovering.load(Ordering::Acquire)
    }

    /// Number of successful stream restarts.
    pub fn get_stream_restart_count(&self) -> u64 {
        self.recovery_state
            .lock()
            .map(|s| s.restart_count)
            .unwrap_or(0)
    }

    /// Last recovery error string, if any.
    pub fn get_last_stream_error(&self) -> Option<String> {
        self.recovery_state
            .lock()
            .ok()
            .and_then(|s| s.last_error.clone())
    }

    /// Last recovery trigger reason string, if any.
    pub fn get_last_restart_reason(&self) -> Option<String> {
        self.recovery_state
            .lock()
            .ok()
            .and_then(|s| s.last_reason.clone())
    }

    /// Check if processing is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Set master bypass
    pub fn set_bypass(&self, bypass: bool) {
        self.bypass.store(bypass, Ordering::SeqCst);
    }

    /// Get bypass state
    pub fn is_bypass(&self) -> bool {
        self.bypass.load(Ordering::SeqCst)
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    // === Noise Gate Controls ===

    /// Enable/disable noise gate
    pub fn set_gate_enabled(&self, enabled: bool) {
        self.gate_enabled.store(enabled, Ordering::Release);
        if let Ok(mut g) = self.gate.lock() {
            g.set_enabled(enabled);
        }
    }

    /// Set noise gate threshold
    pub fn set_gate_threshold(&self, threshold_db: f64) {
        if let Ok(mut g) = self.gate.lock() {
            g.set_threshold(threshold_db);
        }
    }

    /// Set noise gate attack time
    pub fn set_gate_attack(&self, attack_ms: f64) {
        if let Ok(mut g) = self.gate.lock() {
            g.set_attack_time(attack_ms);
        }
    }

    /// Set noise gate release time
    pub fn set_gate_release(&self, release_ms: f64) {
        if let Ok(mut g) = self.gate.lock() {
            g.set_release_time(release_ms);
        }
    }

    /// Check if gate is enabled
    pub fn is_gate_enabled(&self) -> bool {
        self.gate_enabled.load(Ordering::Acquire)
    }

    // === VAD Gate Controls (VAD feature only) ===

    #[cfg(feature = "vad")]
    /// Set gate mode
    pub fn set_gate_mode(&self, mode: u8) -> Result<(), String> {
        let gate_mode = match mode {
            0 => GateMode::ThresholdOnly,
            1 => GateMode::VadAssisted,
            2 => GateMode::VadOnly,
            _ => return Err("Invalid gate mode".to_string()),
        };
        if let Ok(mut gate) = self.gate.lock() {
            gate.set_gate_mode(gate_mode);
        }
        Ok(())
    }

    #[cfg(feature = "vad")]
    /// Get VAD speech probability (0.0-1.0)
    pub fn get_vad_probability(&self) -> f32 {
        f32::from_bits(self.vad_probability.load(Ordering::Relaxed))
    }

    #[cfg(feature = "vad")]
    /// Check whether VAD backend is available (model/runtime loaded)
    pub fn is_vad_available(&self) -> bool {
        if let Ok(gate) = self.gate.lock() {
            gate.is_vad_available()
        } else {
            false
        }
    }

    #[cfg(feature = "vad")]
    /// Set VAD probability threshold (0.0-1.0)
    pub fn set_vad_threshold(&self, threshold: f32) {
        if let Ok(mut gate) = self.gate.lock() {
            gate.set_vad_threshold(threshold);
        }
    }

    #[cfg(feature = "vad")]
    /// Set VAD hold time in milliseconds
    pub fn set_vad_hold_time(&self, hold_ms: f32) {
        if let Ok(mut gate) = self.gate.lock() {
            gate.set_hold_time(hold_ms);
        }
    }

    #[cfg(feature = "vad")]
    /// Set VAD pre-gain to boost weak signals for better speech detection
    /// Default is 1.0 (no gain). Values > 1.0 boost the signal.
    /// This helps with quiet microphones where VAD can't detect speech.
    pub fn set_vad_pre_gain(&self, gain: f32) {
        if let Ok(mut gate) = self.gate.lock() {
            gate.set_vad_pre_gain(gain);
        }
    }

    #[cfg(feature = "vad")]
    /// Get current VAD pre-gain
    pub fn vad_pre_gain(&self) -> f32 {
        if let Ok(gate) = self.gate.lock() {
            gate.vad_pre_gain()
        } else {
            1.0
        }
    }

    #[cfg(feature = "vad")]
    /// Enable/disable auto-threshold mode (automatically adjusts gate threshold based on noise floor)
    pub fn set_auto_threshold(&self, enabled: bool) {
        if let Ok(mut gate) = self.gate.lock() {
            gate.set_auto_threshold(enabled);
        }
    }

    #[cfg(feature = "vad")]
    /// Set margin above noise floor for auto-threshold (in dB)
    pub fn set_gate_margin(&self, margin_db: f32) {
        if let Ok(mut gate) = self.gate.lock() {
            gate.set_margin(margin_db);
        }
    }

    #[cfg(feature = "vad")]
    /// Get current noise floor estimate (in dB)
    pub fn get_noise_floor(&self) -> f32 {
        if let Ok(gate) = self.gate.lock() {
            gate.noise_floor()
        } else {
            -60.0
        }
    }

    #[cfg(feature = "vad")]
    /// Get current gate margin (in dB)
    pub fn gate_margin(&self) -> f32 {
        if let Ok(gate) = self.gate.lock() {
            gate.margin()
        } else {
            6.0
        }
    }

    #[cfg(feature = "vad")]
    /// Check if auto-threshold is enabled
    pub fn auto_threshold_enabled(&self) -> bool {
        if let Ok(gate) = self.gate.lock() {
            gate.auto_threshold_enabled()
        } else {
            false
        }
    }

    // === Noise Suppression Controls ===

    /// Enable/disable noise suppression
    pub fn set_rnnoise_enabled(&self, enabled: bool) {
        // Check if we're transitioning from enabled to disabled
        let was_enabled = self.suppressor_enabled.load(Ordering::Acquire);
        self.suppressor_enabled.store(enabled, Ordering::Release);

        if let Ok(mut s) = self.suppressor.lock() {
            s.set_enabled(enabled);
            // Flush buffers on disable to prevent stale audio on re-enable
            // This avoids delayed/duplicated samples when toggling rapidly
            if was_enabled && !enabled {
                s.soft_reset();
            }
        }
    }

    /// Check if noise suppression is enabled
    pub fn is_rnnoise_enabled(&self) -> bool {
        self.suppressor_enabled.load(Ordering::Acquire)
    }

    /// Set noise suppression wet/dry mix strength (0.0 = fully dry, 1.0 = fully wet)
    pub fn set_rnnoise_strength(&self, strength: f32) {
        let clamped = strength.clamp(0.0, 1.0);
        let bits = clamped.to_bits();
        self.suppressor_strength.store(bits, Ordering::Relaxed);

        // Also update suppressor directly for consistency
        if let Ok(s) = self.suppressor.lock() {
            s.set_strength(clamped);
        }
    }

    /// Get current noise suppression strength
    pub fn get_rnnoise_strength(&self) -> f32 {
        f32::from_bits(self.suppressor_strength.load(Ordering::Relaxed))
    }

    /// Set noise suppression model
    /// Returns true if successful
    pub fn set_noise_model(&self, model: NoiseModel) -> bool {
        #[cfg(feature = "deepfilter")]
        {
            if matches!(
                model,
                NoiseModel::DeepFilterNetLL | NoiseModel::DeepFilterNet
            ) && !Self::deepfilter_experimental_enabled()
            {
                return false;
            }
        }

        // Create new suppressor engine with current strength
        let strength = Arc::clone(&self.suppressor_strength);
        let new_engine = NoiseSuppressionEngine::new(model, strength);

        #[cfg(feature = "deepfilter")]
        {
            if matches!(
                model,
                NoiseModel::DeepFilterNetLL | NoiseModel::DeepFilterNet
            ) && !new_engine.backend_available()
            {
                // DeepFilter is present in code but runtime backend failed to initialize.
                // Report failure so UI can revert to RNNoise instead of silent passthrough.
                return false;
            }
        }

        if let Ok(mut s) = self.suppressor.lock() {
            // Preserve enabled state
            let was_enabled = s.is_enabled();
            *s = new_engine;
            s.set_enabled(was_enabled);

            // Update model indicator
            self.current_model.store(model as u8, Ordering::Release);
            return true;
        }
        false
    }

    /// Get current noise suppression model
    pub fn get_noise_model(&self) -> NoiseModel {
        let model_u8 = self.current_model.load(Ordering::Acquire);
        match model_u8 {
            0 => NoiseModel::RNNoise,
            #[cfg(feature = "deepfilter")]
            1 => NoiseModel::DeepFilterNetLL,
            #[cfg(feature = "deepfilter")]
            2 => NoiseModel::DeepFilterNet,
            _ => NoiseModel::RNNoise,
        }
    }

    /// Get list of available noise models
    pub fn list_noise_models(&self) -> Vec<(String, String)> {
        let mut models = vec![(
            NoiseModel::RNNoise.id().to_string(),
            NoiseModel::RNNoise.display_name().to_string(),
        )];

        #[cfg(feature = "deepfilter")]
        {
            if Self::deepfilter_experimental_enabled() {
                let strength = Arc::clone(&self.suppressor_strength);

                let ll =
                    NoiseSuppressionEngine::new(NoiseModel::DeepFilterNetLL, Arc::clone(&strength));
                if ll.backend_available() {
                    models.push((
                        NoiseModel::DeepFilterNetLL.id().to_string(),
                        NoiseModel::DeepFilterNetLL.display_name().to_string(),
                    ));
                }

                let std = NoiseSuppressionEngine::new(NoiseModel::DeepFilterNet, strength);
                if std.backend_available() {
                    models.push((
                        NoiseModel::DeepFilterNet.id().to_string(),
                        NoiseModel::DeepFilterNet.display_name().to_string(),
                    ));
                }
            }
        }

        models
    }

    // === EQ Controls ===

    /// Enable/disable EQ
    pub fn set_eq_enabled(&self, enabled: bool) {
        self.eq_enabled.store(enabled, Ordering::Release);
        if let Ok(mut e) = self.eq.lock() {
            e.set_enabled(enabled);
        }
        if let Ok(mut control) = self.eq_control.lock() {
            control.enabled = enabled;
        }
        self.eq_dirty.store(true, Ordering::Release);
    }

    /// Check if EQ is enabled
    pub fn is_eq_enabled(&self) -> bool {
        self.eq_enabled.load(Ordering::Acquire)
    }

    /// Set EQ band gain
    pub fn set_eq_band_gain(&self, band: usize, gain_db: f64) {
        if let Ok(mut e) = self.eq.lock() {
            e.set_band_gain(band, gain_db);
        }
        if band < NUM_BANDS {
            if let Ok(mut control) = self.eq_control.lock() {
                control.bands[band].1 = gain_db;
            }
            self.eq_dirty.store(true, Ordering::Release);
        }
    }

    /// Set EQ band frequency
    pub fn set_eq_band_frequency(&self, band: usize, frequency: f64) {
        if let Ok(mut e) = self.eq.lock() {
            e.set_band_frequency(band, frequency);
        }
        if band < NUM_BANDS {
            if let Ok(mut control) = self.eq_control.lock() {
                control.bands[band].0 = frequency;
            }
            self.eq_dirty.store(true, Ordering::Release);
        }
    }

    /// Set EQ band Q
    pub fn set_eq_band_q(&self, band: usize, q: f64) {
        if let Ok(mut e) = self.eq.lock() {
            e.set_band_q(band, q);
        }
        if band < NUM_BANDS {
            if let Ok(mut control) = self.eq_control.lock() {
                control.bands[band].2 = q;
            }
            self.eq_dirty.store(true, Ordering::Release);
        }
    }

    /// Get EQ band parameters (frequency, gain_db, q)
    pub fn get_eq_band_params(&self, band: usize) -> Option<(f64, f64, f64)> {
        if let Ok(e) = self.eq.lock() {
            e.get_band_params(band)
        } else {
            None
        }
    }

    /// Apply EQ settings for all 10 bands in a single atomic call
    ///
    /// # Arguments
    /// * `bands` - Vector of (frequency_hz, gain_db, q) tuples for each band (must be 10)
    ///
    /// # Returns
    /// * PyResult<()> - Ok(()) on success, Err if validation fails
    ///
    /// # Validation
    /// * bands.len() must equal 10
    /// * frequency: 20.0 to 20000.0 Hz
    /// * gain_db: -12.0 to +12.0 dB
    /// * q: 0.1 to 10.0
    pub fn apply_eq_settings(&self, bands: Vec<(f64, f64, f64)>) -> PyResult<()> {
        // Validate band count
        if bands.len() != 10 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected 10 bands, got {}",
                bands.len()
            )));
        }

        // Validate each band's parameters
        for (i, (freq, gain, q)) in bands.iter().enumerate() {
            if *freq < 20.0 || *freq > 20000.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Band {}: frequency {} Hz out of range [20, 20000]",
                    i, freq
                )));
            }
            if *gain < -12.0 || *gain > 12.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Band {}: gain {} dB out of range [-12, 12]",
                    i, gain
                )));
            }
            if *q < 0.1 || *q > 10.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Band {}: Q {} out of range [0.1, 10.0]",
                    i, q
                )));
            }
        }

        // All validation passed - apply atomically
        if let Ok(mut eq) = self.eq.lock() {
            for (i, (freq, gain, q)) in bands.iter().enumerate() {
                eq.set_band_frequency(i, *freq);
                eq.set_band_gain(i, *gain);
                eq.set_band_q(i, *q);
            }
        }
        if let Ok(mut control) = self.eq_control.lock() {
            for (i, (freq, gain, q)) in bands.iter().copied().enumerate() {
                control.bands[i] = (freq, gain, q);
            }
        }
        self.eq_dirty.store(true, Ordering::Release);

        Ok(())
    }

    // === De-Esser Controls ===

    /// Enable/disable de-esser.
    pub fn set_deesser_enabled(&self, enabled: bool) {
        self.deesser_enabled.store(enabled, Ordering::Release);
        if let Ok(mut d) = self.deesser.lock() {
            d.set_enabled(enabled);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.enabled = enabled;
        }
        self.deesser_dirty.store(true, Ordering::Release);
    }

    /// Check if de-esser is enabled.
    pub fn is_deesser_enabled(&self) -> bool {
        self.deesser_enabled.load(Ordering::Acquire)
    }

    pub fn set_deesser_low_cut_hz(&self, hz: f64) {
        if let Ok(mut d) = self.deesser.lock() {
            d.set_low_cut_hz(hz);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.low_cut_hz = hz;
        }
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn set_deesser_high_cut_hz(&self, hz: f64) {
        if let Ok(mut d) = self.deesser.lock() {
            d.set_high_cut_hz(hz);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.high_cut_hz = hz;
        }
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn set_deesser_threshold_db(&self, threshold_db: f64) {
        if let Ok(mut d) = self.deesser.lock() {
            d.set_threshold_db(threshold_db);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.threshold_db = threshold_db;
        }
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn set_deesser_ratio(&self, ratio: f64) {
        if let Ok(mut d) = self.deesser.lock() {
            d.set_ratio(ratio);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.ratio = ratio;
        }
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn set_deesser_attack_ms(&self, attack_ms: f64) {
        if let Ok(mut d) = self.deesser.lock() {
            d.set_attack_ms(attack_ms);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.attack_ms = attack_ms;
        }
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn set_deesser_release_ms(&self, release_ms: f64) {
        if let Ok(mut d) = self.deesser.lock() {
            d.set_release_ms(release_ms);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.release_ms = release_ms;
        }
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn set_deesser_max_reduction_db(&self, max_reduction_db: f64) {
        if let Ok(mut d) = self.deesser.lock() {
            d.set_max_reduction_db(max_reduction_db);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.max_reduction_db = max_reduction_db;
        }
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn set_deesser_auto_enabled(&self, auto_enabled: bool) {
        if let Ok(mut d) = self.deesser.lock() {
            d.set_auto_enabled(auto_enabled);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.auto_enabled = auto_enabled;
        }
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn is_deesser_auto_enabled(&self) -> bool {
        if let Ok(d) = self.deesser.lock() {
            d.is_auto_enabled()
        } else {
            true
        }
    }

    pub fn set_deesser_auto_amount(&self, amount: f64) {
        if let Ok(mut d) = self.deesser.lock() {
            d.set_auto_amount(amount);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.auto_amount = amount;
        }
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn get_deesser_auto_amount(&self) -> f64 {
        if let Ok(d) = self.deesser.lock() {
            d.auto_amount()
        } else {
            0.5
        }
    }

    pub fn get_deesser_low_cut_hz(&self) -> f64 {
        if let Ok(d) = self.deesser.lock() {
            d.low_cut_hz()
        } else {
            4000.0
        }
    }

    pub fn get_deesser_high_cut_hz(&self) -> f64 {
        if let Ok(d) = self.deesser.lock() {
            d.high_cut_hz()
        } else {
            9000.0
        }
    }

    pub fn get_deesser_threshold_db(&self) -> f64 {
        if let Ok(d) = self.deesser.lock() {
            d.threshold_db()
        } else {
            -28.0
        }
    }

    pub fn get_deesser_ratio(&self) -> f64 {
        if let Ok(d) = self.deesser.lock() {
            d.ratio()
        } else {
            4.0
        }
    }

    pub fn get_deesser_max_reduction_db(&self) -> f64 {
        if let Ok(d) = self.deesser.lock() {
            d.max_reduction_db()
        } else {
            6.0
        }
    }

    pub fn get_deesser_gain_reduction_db(&self) -> f32 {
        f32::from_bits(self.deesser_gain_reduction.load(Ordering::Relaxed))
    }

    // === Compressor Controls ===

    /// Enable/disable compressor
    pub fn set_compressor_enabled(&self, enabled: bool) {
        self.compressor_enabled.store(enabled, Ordering::Release);
        if let Ok(mut c) = self.compressor.lock() {
            c.set_enabled(enabled);
        }
        if let Ok(mut control) = self.compressor_control.lock() {
            control.enabled = enabled;
        }
        self.compressor_dirty.store(true, Ordering::Release);
    }

    /// Check if compressor is enabled
    pub fn is_compressor_enabled(&self) -> bool {
        self.compressor_enabled.load(Ordering::Acquire)
    }

    /// Set compressor threshold in dB
    pub fn set_compressor_threshold(&self, threshold_db: f64) {
        if let Ok(mut c) = self.compressor.lock() {
            c.set_threshold(threshold_db);
        }
        if let Ok(mut control) = self.compressor_control.lock() {
            control.threshold_db = threshold_db;
        }
        self.compressor_dirty.store(true, Ordering::Release);
    }

    /// Set compressor ratio
    pub fn set_compressor_ratio(&self, ratio: f64) {
        if let Ok(mut c) = self.compressor.lock() {
            c.set_ratio(ratio);
        }
        if let Ok(mut control) = self.compressor_control.lock() {
            control.ratio = ratio;
        }
        self.compressor_dirty.store(true, Ordering::Release);
    }

    /// Set compressor attack time in ms
    pub fn set_compressor_attack(&self, attack_ms: f64) {
        if let Ok(mut c) = self.compressor.lock() {
            c.set_attack_time(attack_ms);
        }
        if let Ok(mut control) = self.compressor_control.lock() {
            control.attack_ms = attack_ms;
        }
        self.compressor_dirty.store(true, Ordering::Release);
    }

    /// Set compressor release time in ms
    ///
    /// Note: When adaptive release is enabled, this updates the base release time.
    /// Use set_compressor_adaptive_release(false) to disable adaptive mode and
    /// use set_compressor_base_release() to set the base release time directly.
    pub fn set_compressor_release(&self, release_ms: f64) {
        if let Ok(mut c) = self.compressor.lock() {
            // Update base release time regardless of adaptive mode
            c.set_base_release_time(release_ms);
            // Also update current release when not in adaptive mode
            if !c.adaptive_release() {
                c.set_release_time(release_ms);
            }
        }
        if let Ok(mut control) = self.compressor_control.lock() {
            control.base_release_ms = release_ms;
        }
        self.compressor_dirty.store(true, Ordering::Release);
    }

    /// Get compressor release time.
    ///
    /// Note: When adaptive release is enabled, this returns the base release time.
    /// Use get_compressor_current_release() for the actual adaptive release time.
    pub fn get_compressor_release(&self) -> f64 {
        if let Ok(c) = self.compressor.lock() {
            if c.adaptive_release() {
                // Return base release when adaptive mode is active
                c.base_release_ms()
            } else {
                // Return current release when in manual mode
                c.current_release_time()
            }
        } else {
            200.0
        }
    }

    /// Set compressor makeup gain in dB
    pub fn set_compressor_makeup_gain(&self, makeup_gain_db: f64) {
        if let Ok(mut c) = self.compressor.lock() {
            c.set_makeup_gain(makeup_gain_db);
        }
        if let Ok(mut control) = self.compressor_control.lock() {
            control.makeup_gain_db = makeup_gain_db;
        }
        self.compressor_dirty.store(true, Ordering::Release);
    }

    pub fn set_compressor_adaptive_release(&self, enabled: bool) {
        if let Ok(mut c) = self.compressor.lock() {
            c.set_adaptive_release(enabled);
        }
        if let Ok(mut control) = self.compressor_control.lock() {
            control.adaptive_release = enabled;
        }
        self.compressor_dirty.store(true, Ordering::Release);
    }

    pub fn get_compressor_adaptive_release(&self) -> bool {
        if let Ok(c) = self.compressor.lock() {
            c.adaptive_release()
        } else {
            false
        }
    }

    pub fn set_compressor_base_release(&self, release_ms: f64) {
        if let Ok(mut c) = self.compressor.lock() {
            c.set_base_release_time(release_ms);
        }
        if let Ok(mut control) = self.compressor_control.lock() {
            control.base_release_ms = release_ms;
        }
        self.compressor_dirty.store(true, Ordering::Release);
    }

    pub fn get_compressor_base_release(&self) -> f64 {
        if let Ok(c) = self.compressor.lock() {
            c.base_release_ms()
        } else {
            200.0
        }
    }

    // === Limiter Controls ===

    /// Enable/disable limiter
    pub fn set_limiter_enabled(&self, enabled: bool) {
        self.limiter_enabled.store(enabled, Ordering::Release);
        if let Ok(mut l) = self.limiter.lock() {
            l.set_enabled(enabled);
        }
        if let Ok(mut control) = self.limiter_control.lock() {
            control.enabled = enabled;
        }
        self.limiter_dirty.store(true, Ordering::Release);
    }

    /// Check if limiter is enabled
    pub fn is_limiter_enabled(&self) -> bool {
        self.limiter_enabled.load(Ordering::Acquire)
    }

    /// Set limiter ceiling in dB
    pub fn set_limiter_ceiling(&self, ceiling_db: f64) {
        if let Ok(mut l) = self.limiter.lock() {
            l.set_ceiling(ceiling_db);
        }
        if let Ok(mut control) = self.limiter_control.lock() {
            control.ceiling_db = ceiling_db;
        }
        self.limiter_dirty.store(true, Ordering::Release);
    }

    /// Set limiter release time in ms
    pub fn set_limiter_release(&self, release_ms: f64) {
        if let Ok(mut l) = self.limiter.lock() {
            l.set_release_time(release_ms);
        }
        if let Ok(mut control) = self.limiter_control.lock() {
            control.release_ms = release_ms;
        }
        self.limiter_dirty.store(true, Ordering::Release);
    }

    // === Metering ===

    /// Get input peak level in dB
    pub fn get_input_peak_db(&self) -> f32 {
        f32::from_bits(self.input_peak.load(Ordering::Relaxed))
    }

    /// Get input RMS level in dB
    pub fn get_input_rms_db(&self) -> f32 {
        f32::from_bits(self.input_rms.load(Ordering::Relaxed))
    }

    /// Get output peak level in dB
    pub fn get_output_peak_db(&self) -> f32 {
        f32::from_bits(self.output_peak.load(Ordering::Relaxed))
    }

    /// Get output RMS level in dB
    pub fn get_output_rms_db(&self) -> f32 {
        f32::from_bits(self.output_rms.load(Ordering::Relaxed))
    }

    /// Get compressor gain reduction in dB
    pub fn get_compressor_gain_reduction_db(&self) -> f32 {
        f32::from_bits(self.compressor_gain_reduction.load(Ordering::Relaxed))
    }

    // === Auto Makeup Gain Controls ===

    /// Set compressor auto makeup gain mode
    pub fn set_compressor_auto_makeup_enabled(&self, enabled: bool) {
        if let Ok(mut c) = self.compressor.lock() {
            c.set_auto_makeup_enabled(enabled);
        }
        if let Ok(mut control) = self.compressor_control.lock() {
            control.auto_makeup_enabled = enabled;
        }
        self.compressor_dirty.store(true, Ordering::Release);
    }

    /// Get compressor auto makeup gain mode
    pub fn get_compressor_auto_makeup_enabled(&self) -> bool {
        if let Ok(c) = self.compressor.lock() {
            c.auto_makeup_enabled()
        } else {
            false
        }
    }

    /// Set compressor target LUFS
    pub fn set_compressor_target_lufs(&self, target_lufs: f64) {
        if let Ok(mut c) = self.compressor.lock() {
            c.set_target_lufs(target_lufs);
        }
        if let Ok(mut control) = self.compressor_control.lock() {
            control.target_lufs = target_lufs;
        }
        self.compressor_dirty.store(true, Ordering::Release);
    }

    /// Get compressor target LUFS
    pub fn get_compressor_target_lufs(&self) -> f64 {
        if let Ok(c) = self.compressor.lock() {
            c.target_lufs()
        } else {
            -18.0
        }
    }

    /// Get compressor current LUFS
    pub fn get_compressor_current_lufs(&self) -> f64 {
        if let Ok(c) = self.compressor.lock() {
            c.current_lufs()
        } else {
            -18.0
        }
    }

    /// Get compressor current makeup gain
    pub fn get_compressor_current_makeup_gain(&self) -> f64 {
        if let Ok(c) = self.compressor.lock() {
            c.current_makeup_gain()
        } else {
            0.0
        }
    }

    /// Get current processing latency in milliseconds
    pub fn get_latency_ms(&self) -> f32 {
        let latency_us = self.latency_us.load(Ordering::Relaxed);
        latency_us as f32 / 1000.0
    }

    /// Set user latency compensation in milliseconds (added to reported latency).
    pub fn set_latency_compensation_ms(&self, compensation_ms: f32) {
        let clamped_ms = compensation_ms.clamp(0.0, 5000.0);
        let compensation_us = (clamped_ms * 1000.0) as u64;
        self.latency_compensation_us
            .store(compensation_us, Ordering::Relaxed);
    }

    /// Get configured latency compensation in milliseconds.
    pub fn get_latency_compensation_ms(&self) -> f32 {
        let compensation_us = self.latency_compensation_us.load(Ordering::Relaxed);
        compensation_us as f32 / 1000.0
    }

    // === DSP Performance Metrics ===

    /// Get DSP processing time in milliseconds
    pub fn get_dsp_time_ms(&self) -> f32 {
        let us = self.dsp_time_us.load(Ordering::Relaxed);
        us as f32 / 1000.0
    }

    /// Get input buffer fill level in samples
    pub fn get_input_buffer_samples(&self) -> u32 {
        self.input_buffer_len.load(Ordering::Relaxed)
    }

    /// Get smoothed input buffer fill level in samples
    pub fn get_input_buffer_smoothed_samples(&self) -> u32 {
        self.smoothed_input_buffer_len.load(Ordering::Relaxed)
    }

    /// Get output buffer fill level in samples
    pub fn get_output_buffer_samples(&self) -> u32 {
        self.output_buffer_len.load(Ordering::Relaxed)
    }

    /// Get active output device sample rate in Hz.
    pub fn output_sample_rate(&self) -> u32 {
        self.output_sample_rate.load(Ordering::Relaxed)
    }

    /// Get noise suppressor buffer fill level in samples
    pub fn get_rnnoise_buffer_samples(&self) -> u32 {
        self.suppressor_buffer_len.load(Ordering::Relaxed)
    }

    // === Dropped Sample Tracking ===

    /// Get dropped sample count (samples lost due to buffer overflow)
    pub fn get_dropped_samples(&self) -> u64 {
        self.input_dropped.load(Ordering::Relaxed)
    }

    /// Reset dropped sample counter
    pub fn reset_dropped_samples(&self) {
        self.input_dropped.store(0, Ordering::Relaxed);
    }

    /// Get total real-time lock contention events.
    pub fn get_lock_contention_count(&self) -> u64 {
        self.lock_contention_count.load(Ordering::Relaxed)
    }

    /// Reset lock contention counter.
    pub fn reset_lock_contention_count(&self) {
        self.lock_contention_count.store(0, Ordering::Relaxed);
    }

    pub fn get_suppressor_non_finite_count(&self) -> u64 {
        self.suppressor_non_finite_count.load(Ordering::Relaxed)
    }

    /// Age of last input callback heartbeat in milliseconds.
    pub fn get_input_callback_age_ms(&self) -> u64 {
        let last = self.last_input_callback_time_us.load(Ordering::Relaxed);
        if last == 0 {
            return u64::MAX;
        }
        let now = now_micros();
        now.saturating_sub(last) / 1000
    }

    /// Age of last output callback heartbeat in milliseconds.
    pub fn get_output_callback_age_ms(&self) -> u64 {
        let last = self.last_output_callback_time_us.load(Ordering::Relaxed);
        if last == 0 {
            return u64::MAX;
        }
        let now = now_micros();
        now.saturating_sub(last) / 1000
    }

    /// Current consecutive output underrun streak.
    pub fn get_output_underrun_streak(&self) -> u32 {
        self.output_underrun_streak.load(Ordering::Relaxed)
    }

    /// Total output underrun callbacks since start.
    pub fn get_output_underrun_total(&self) -> u64 {
        self.output_underrun_total.load(Ordering::Relaxed)
    }

    /// Samples dropped by DSP-side jitter control.
    pub fn get_jitter_dropped_samples(&self) -> u64 {
        self.jitter_dropped_samples.load(Ordering::Relaxed)
    }

    /// Number of jitter recovery events.
    pub fn get_output_recovery_count(&self) -> u64 {
        self.output_recovery_count.load(Ordering::Relaxed)
    }

    /// Whether the currently selected suppressor backend is operational.
    pub fn is_noise_backend_available(&self) -> bool {
        if let Ok(suppressor) = self.suppressor.lock() {
            suppressor.backend_available()
        } else {
            false
        }
    }

    /// Whether the suppressor backend has failed after startup.
    pub fn noise_backend_failed(&self) -> bool {
        if let Ok(suppressor) = self.suppressor.lock() {
            suppressor.backend_failed()
        } else {
            false
        }
    }

    /// Last suppressor backend error, if any.
    pub fn noise_backend_error(&self) -> Option<String> {
        self.suppressor
            .lock()
            .ok()
            .and_then(|suppressor| suppressor.backend_error().map(str::to_string))
    }

    /// Suppress watchdog-driven stream recovery during intrusive UI workflows.
    pub fn set_recovery_suppressed(&self, suppressed: bool) {
        self.recovery_suppressed.store(suppressed, Ordering::Release);
    }

    /// Whether watchdog-driven stream recovery is currently suppressed.
    pub fn is_recovery_suppressed(&self) -> bool {
        self.recovery_suppressed.load(Ordering::Acquire)
    }

    // === RAW AUDIO RECORDING (for calibration) ===

    /// Start recording raw audio for calibration
    /// Taps audio AFTER pre-filter (DC blocker + 80Hz HP) but BEFORE noise gate
    pub fn start_raw_recording(&mut self, duration_secs: f64) -> Result<(), String> {
        let num_samples = (duration_secs * self.sample_rate as f64) as usize;
        let max_samples = self.sample_rate as usize * MAX_RECORDING_SECONDS;
        if num_samples > max_samples {
            return Err(format!(
                "Requested recording length exceeds max supported capture window ({}s)",
                MAX_RECORDING_SECONDS
            ));
        }

        let mut drained = vec![0.0f32; 4096];
        if let Ok(mut consumer_guard) = self.raw_recording_consumer.lock() {
            let Some(ref mut consumer) = *consumer_guard else {
                return Err("Recording buffer unavailable. Start the processor first.".to_string());
            };
            while !consumer.is_empty() {
                let read = consumer.read(&mut drained);
                if read == 0 {
                    break;
                }
            }
            self.raw_recording_pos.store(0, Ordering::Release);
            self.raw_recording_target.store(num_samples, Ordering::Release);
            self.recording_level_db
                .store((-120.0_f32).to_bits(), Ordering::Relaxed);
            self.recording_active.store(true, Ordering::Release);
            Ok(())
        } else {
            Err("Failed to access recording buffer".to_string())
        }
    }

    /// Stop recording and return captured audio (truncated to actual length)
    pub fn stop_raw_recording(&mut self) -> Option<Vec<f32>> {
        self.recording_active.store(false, Ordering::Release);
        self.recording_level_db
            .store((-120.0_f32).to_bits(), Ordering::Relaxed);
        if let Ok(mut consumer_guard) = self.raw_recording_consumer.lock() {
            if let Some(ref mut consumer) = *consumer_guard {
                let pos = self.raw_recording_pos.load(Ordering::Acquire);
                let mut buffer = vec![0.0f32; pos];
                let mut read_total = 0usize;
                while read_total < pos {
                    let read = consumer.read(&mut buffer[read_total..]);
                    if read == 0 {
                        break;
                    }
                    read_total += read;
                }
                buffer.truncate(read_total);
                self.raw_recording_target.store(0, Ordering::Release);
                return Some(buffer);
            }
        }
        None
    }

    /// Check if recording is complete
    pub fn is_recording_complete(&self) -> bool {
        let target = self.raw_recording_target.load(Ordering::Acquire);
        let pos = self.raw_recording_pos.load(Ordering::Acquire);
        target > 0 && pos >= target
    }

    /// Get recording progress (0.0 to 1.0)
    pub fn recording_progress(&self) -> f32 {
        let target = self.raw_recording_target.load(Ordering::Acquire);
        if target == 0 {
            return 0.0;
        }
        let pos = self.raw_recording_pos.load(Ordering::Acquire);
        (pos as f32 / target as f32).min(1.0)
    }

    /// Get current recording level as RMS in dB (for level meter visualization)
    /// Returns -inf dB if no recording is active
    pub fn recording_level_db(&self) -> f32 {
        f32::from_bits(self.recording_level_db.load(Ordering::Relaxed))
    }

    /// Manually set output mute state (useful for calibration workflow)
    pub fn set_output_mute(&self, muted: bool) {
        self.output_muted.store(muted, Ordering::Release);
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
        self.supervisor_running.store(false, Ordering::Release);
        if let Some(handle) = self.supervisor_thread.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_duration_samples_for_44k1_output() {
        assert_eq!(duration_samples(44_100, OUTPUT_PRIME_MS), 441);
        assert_eq!(duration_samples(44_100, OUTPUT_TARGET_HIGH_MS), 882);
        assert_eq!(duration_samples(44_100, OUTPUT_HARD_BACKLOG_MS), 1764);
    }

    #[test]
    fn test_samples_to_micros_uses_device_rate() {
        assert_eq!(samples_to_micros(441, 44_100), 10_000);
        assert_eq!(samples_to_micros(480, 48_000), 10_000);
    }
}

// === Python Bindings ===

/// Gate operating modes
#[cfg(feature = "vad")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[pyclass(eq, eq_int)]
pub enum PyGateMode {
    /// Traditional gate using only level threshold
    ThresholdOnly = 0,
    /// Hybrid: gate opens when level exceeded OR speech detected
    VadAssisted = 1,
    /// VAD-only: gate opens solely based on speech probability
    VadOnly = 2,
}

/// Python-exposed audio processor
#[pyclass(name = "AudioProcessor", unsendable)]
pub struct PyAudioProcessor {
    processor: AudioProcessor,
}

#[pymethods]
impl PyAudioProcessor {
    #[new]
    fn new() -> Self {
        Self {
            processor: AudioProcessor::new(),
        }
    }

    /// Start audio processing
    #[pyo3(signature = (input_device=None, output_device=None))]
    fn start(
        &mut self,
        input_device: Option<&str>,
        output_device: Option<&str>,
    ) -> PyResult<String> {
        self.processor
            .start(input_device, output_device)
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    /// Stop audio processing
    fn stop(&mut self) {
        self.processor.stop();
    }

    /// Check if running
    fn is_running(&self) -> bool {
        self.processor.is_running()
    }

    /// Get sample rate
    fn sample_rate(&self) -> u32 {
        self.processor.sample_rate()
    }

    /// Set master bypass
    fn set_bypass(&self, bypass: bool) {
        self.processor.set_bypass(bypass);
    }

    /// Get bypass state
    fn is_bypass(&self) -> bool {
        self.processor.is_bypass()
    }

    // === Noise Gate ===

    fn set_gate_enabled(&self, enabled: bool) {
        self.processor.set_gate_enabled(enabled);
    }

    fn is_gate_enabled(&self) -> bool {
        self.processor.is_gate_enabled()
    }

    fn set_gate_threshold(&self, threshold_db: f64) {
        self.processor.set_gate_threshold(threshold_db);
    }

    fn set_gate_attack(&self, attack_ms: f64) {
        self.processor.set_gate_attack(attack_ms);
    }

    fn set_gate_release(&self, release_ms: f64) {
        self.processor.set_gate_release(release_ms);
    }

    // === VAD Gate Controls ===

    /// Set gate mode (0 = ThresholdOnly, 1 = VadAssisted, 2 = VadOnly)
    #[cfg(feature = "vad")]
    #[pyo3(signature = (mode))]
    fn set_gate_mode(&self, mode: u8) -> PyResult<()> {
        self.processor
            .set_gate_mode(mode)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    /// Get VAD speech probability (0.0-1.0)
    #[cfg(feature = "vad")]
    fn get_vad_probability(&self) -> f32 {
        self.processor.get_vad_probability()
    }

    /// Check whether VAD backend is available (model/runtime loaded)
    #[cfg(feature = "vad")]
    fn is_vad_available(&self) -> bool {
        self.processor.is_vad_available()
    }

    /// Set VAD probability threshold (0.0-1.0)
    #[cfg(feature = "vad")]
    fn set_vad_threshold(&self, threshold: f32) {
        self.processor.set_vad_threshold(threshold);
    }

    /// Set VAD hold time in milliseconds
    #[cfg(feature = "vad")]
    fn set_vad_hold_time(&self, hold_ms: f32) {
        self.processor.set_vad_hold_time(hold_ms);
    }

    /// Set VAD pre-gain to boost weak signals for better speech detection
    /// Default is 1.0 (no gain). Values > 1.0 boost the signal.
    /// This helps with quiet microphones where VAD can't detect speech.
    #[cfg(feature = "vad")]
    fn set_vad_pre_gain(&self, gain: f32) {
        self.processor.set_vad_pre_gain(gain);
    }

    /// Enable/disable auto-threshold mode (automatically adjusts gate threshold based on noise floor)
    #[cfg(feature = "vad")]
    fn set_auto_threshold(&self, enabled: bool) {
        self.processor.set_auto_threshold(enabled);
    }

    /// Set margin above noise floor for auto-threshold (in dB)
    #[cfg(feature = "vad")]
    fn set_gate_margin(&self, margin_db: f32) {
        self.processor.set_gate_margin(margin_db);
    }

    /// Get current noise floor estimate (in dB)
    #[cfg(feature = "vad")]
    fn get_noise_floor(&self) -> f32 {
        self.processor.get_noise_floor()
    }

    /// Get current gate margin (in dB)
    #[cfg(feature = "vad")]
    fn gate_margin(&self) -> f32 {
        self.processor.gate_margin()
    }

    /// Check if auto-threshold is enabled
    #[cfg(feature = "vad")]
    fn auto_threshold_enabled(&self) -> bool {
        self.processor.auto_threshold_enabled()
    }

    /// Get current VAD pre-gain
    #[cfg(feature = "vad")]
    fn vad_pre_gain(&self) -> f32 {
        self.processor.vad_pre_gain()
    }

    // === RNNoise ===

    fn set_rnnoise_enabled(&self, enabled: bool) {
        self.processor.set_rnnoise_enabled(enabled);
    }

    fn is_rnnoise_enabled(&self) -> bool {
        self.processor.is_rnnoise_enabled()
    }

    /// Set RNNoise wet/dry mix strength (0.0 = fully dry, 1.0 = fully wet)
    fn set_rnnoise_strength(&self, strength: f64) {
        self.processor.set_rnnoise_strength(strength as f32);
    }

    /// Get current RNNoise strength
    fn get_rnnoise_strength(&self) -> f64 {
        self.processor.get_rnnoise_strength() as f64
    }

    /// Set noise suppression model by name ("rnnoise" or "deepfilter")
    fn set_noise_model(&self, model: &str) -> bool {
        match NoiseModel::from_id(model) {
            Some(m) => self.processor.set_noise_model(m),
            None => false,
        }
    }

    /// Get current noise model name
    fn get_noise_model(&self) -> String {
        self.processor.get_noise_model().id().to_string()
    }

    /// Get current noise model display name
    fn get_noise_model_display_name(&self) -> String {
        self.processor.get_noise_model().display_name().to_string()
    }

    /// List available noise models: [(id, display_name), ...]
    fn list_noise_models(&self) -> Vec<(String, String)> {
        self.processor.list_noise_models()
    }

    // === EQ ===

    fn set_eq_enabled(&self, enabled: bool) {
        self.processor.set_eq_enabled(enabled);
    }

    fn is_eq_enabled(&self) -> bool {
        self.processor.is_eq_enabled()
    }

    fn set_eq_band_gain(&self, band: usize, gain_db: f64) {
        self.processor.set_eq_band_gain(band, gain_db);
    }

    fn set_eq_band_frequency(&self, band: usize, frequency: f64) {
        self.processor.set_eq_band_frequency(band, frequency);
    }

    fn set_eq_band_q(&self, band: usize, q: f64) {
        self.processor.set_eq_band_q(band, q);
    }

    fn get_eq_band_params(&self, band: usize) -> Option<(f64, f64, f64)> {
        self.processor.get_eq_band_params(band)
    }

    /// Apply EQ settings for all 10 bands in a single atomic call
    ///
    /// Args:
    ///     bands: List of (frequency_hz, gain_db, q) tuples for each band (must be 10)
    ///
    /// Raises:
    ///     ValueError: If band count is not 10 or parameters are out of range
    fn apply_eq_settings(&self, bands: Vec<(f64, f64, f64)>) -> PyResult<()> {
        self.processor.apply_eq_settings(bands)
    }

    // === De-Esser ===

    fn set_deesser_enabled(&self, enabled: bool) {
        self.processor.set_deesser_enabled(enabled);
    }

    fn is_deesser_enabled(&self) -> bool {
        self.processor.is_deesser_enabled()
    }

    fn set_deesser_low_cut_hz(&self, hz: f64) {
        self.processor.set_deesser_low_cut_hz(hz);
    }

    fn set_deesser_high_cut_hz(&self, hz: f64) {
        self.processor.set_deesser_high_cut_hz(hz);
    }

    fn set_deesser_threshold_db(&self, threshold_db: f64) {
        self.processor.set_deesser_threshold_db(threshold_db);
    }

    fn set_deesser_ratio(&self, ratio: f64) {
        self.processor.set_deesser_ratio(ratio);
    }

    fn set_deesser_attack_ms(&self, attack_ms: f64) {
        self.processor.set_deesser_attack_ms(attack_ms);
    }

    fn set_deesser_release_ms(&self, release_ms: f64) {
        self.processor.set_deesser_release_ms(release_ms);
    }

    fn set_deesser_max_reduction_db(&self, max_reduction_db: f64) {
        self.processor
            .set_deesser_max_reduction_db(max_reduction_db);
    }

    fn set_deesser_auto_enabled(&self, auto_enabled: bool) {
        self.processor.set_deesser_auto_enabled(auto_enabled);
    }

    fn is_deesser_auto_enabled(&self) -> bool {
        self.processor.is_deesser_auto_enabled()
    }

    fn set_deesser_auto_amount(&self, amount: f64) {
        self.processor.set_deesser_auto_amount(amount);
    }

    fn get_deesser_low_cut_hz(&self) -> f64 {
        self.processor.get_deesser_low_cut_hz()
    }

    fn get_deesser_high_cut_hz(&self) -> f64 {
        self.processor.get_deesser_high_cut_hz()
    }

    fn get_deesser_threshold_db(&self) -> f64 {
        self.processor.get_deesser_threshold_db()
    }

    fn get_deesser_ratio(&self) -> f64 {
        self.processor.get_deesser_ratio()
    }

    fn get_deesser_max_reduction_db(&self) -> f64 {
        self.processor.get_deesser_max_reduction_db()
    }

    fn get_deesser_auto_amount(&self) -> f64 {
        self.processor.get_deesser_auto_amount()
    }

    fn get_deesser_gain_reduction_db(&self) -> f32 {
        self.processor.get_deesser_gain_reduction_db()
    }

    // === Compressor ===

    fn set_compressor_enabled(&self, enabled: bool) {
        self.processor.set_compressor_enabled(enabled);
    }

    fn is_compressor_enabled(&self) -> bool {
        self.processor.is_compressor_enabled()
    }

    fn set_compressor_threshold(&self, threshold_db: f64) {
        self.processor.set_compressor_threshold(threshold_db);
    }

    fn set_compressor_ratio(&self, ratio: f64) {
        self.processor.set_compressor_ratio(ratio);
    }

    fn set_compressor_attack(&self, attack_ms: f64) {
        self.processor.set_compressor_attack(attack_ms);
    }

    fn set_compressor_release(&self, release_ms: f64) {
        self.processor.set_compressor_release(release_ms);
    }

    /// Get compressor release time.
    ///
    /// Note: When adaptive release is enabled, this returns the base release time.
    /// Use get_compressor_current_release() for the actual adaptive release time.
    fn get_compressor_release(&self) -> f64 {
        self.processor.get_compressor_release()
    }

    fn set_compressor_makeup_gain(&self, makeup_gain_db: f64) {
        self.processor.set_compressor_makeup_gain(makeup_gain_db);
    }

    /// Set compressor adaptive release mode
    fn set_compressor_adaptive_release(&self, enabled: bool) {
        self.processor.set_compressor_adaptive_release(enabled);
    }

    /// Get compressor adaptive release mode
    fn get_compressor_adaptive_release(&self) -> bool {
        self.processor.get_compressor_adaptive_release()
    }

    /// Set compressor base release time (milliseconds)
    fn set_compressor_base_release(&self, release_ms: f64) {
        self.processor.set_compressor_base_release(release_ms);
    }

    /// Get compressor base release time (milliseconds)
    fn get_compressor_base_release(&self) -> f64 {
        self.processor.get_compressor_base_release()
    }

    /// Get current compressor release time (adaptive or base, in milliseconds)
    fn get_compressor_current_release(&self) -> f64 {
        let release_raw = self
            .processor
            .compressor_current_release_ms
            .load(Ordering::Relaxed);
        release_raw as f64 / 10.0 // Convert back from 0.1ms resolution
    }

    // === Auto Makeup Gain ===

    /// Set compressor auto makeup gain mode
    fn set_compressor_auto_makeup_enabled(&self, enabled: bool) {
        self.processor.set_compressor_auto_makeup_enabled(enabled);
    }

    /// Get compressor auto makeup gain mode
    fn get_compressor_auto_makeup_enabled(&self) -> bool {
        self.processor.get_compressor_auto_makeup_enabled()
    }

    /// Set compressor target LUFS
    fn set_compressor_target_lufs(&self, target_lufs: f64) {
        self.processor.set_compressor_target_lufs(target_lufs);
    }

    /// Get compressor target LUFS
    fn get_compressor_target_lufs(&self) -> f64 {
        self.processor.get_compressor_target_lufs()
    }

    /// Get compressor current LUFS
    fn get_compressor_current_lufs(&self) -> f64 {
        self.processor.get_compressor_current_lufs()
    }

    /// Get compressor current makeup gain
    fn get_compressor_current_makeup_gain(&self) -> f64 {
        self.processor.get_compressor_current_makeup_gain()
    }

    // === Limiter ===

    fn set_limiter_enabled(&self, enabled: bool) {
        self.processor.set_limiter_enabled(enabled);
    }

    fn is_limiter_enabled(&self) -> bool {
        self.processor.is_limiter_enabled()
    }

    fn set_limiter_ceiling(&self, ceiling_db: f64) {
        self.processor.set_limiter_ceiling(ceiling_db);
    }

    fn set_limiter_release(&self, release_ms: f64) {
        self.processor.set_limiter_release(release_ms);
    }

    // === Metering ===

    fn get_input_peak_db(&self) -> f32 {
        self.processor.get_input_peak_db()
    }

    fn get_input_rms_db(&self) -> f32 {
        self.processor.get_input_rms_db()
    }

    fn get_output_peak_db(&self) -> f32 {
        self.processor.get_output_peak_db()
    }

    fn get_output_rms_db(&self) -> f32 {
        self.processor.get_output_rms_db()
    }

    fn get_compressor_gain_reduction_db(&self) -> f32 {
        self.processor.get_compressor_gain_reduction_db()
    }

    fn get_latency_ms(&self) -> f32 {
        self.processor.get_latency_ms()
    }

    fn set_latency_compensation_ms(&self, compensation_ms: f32) {
        self.processor.set_latency_compensation_ms(compensation_ms);
    }

    fn get_latency_compensation_ms(&self) -> f32 {
        self.processor.get_latency_compensation_ms()
    }

    // === DSP Performance Metrics ===

    fn get_dsp_time_ms(&self) -> f32 {
        self.processor.get_dsp_time_ms()
    }

    fn get_input_buffer_samples(&self) -> u32 {
        self.processor.get_input_buffer_samples()
    }

    fn get_input_buffer_smoothed_samples(&self) -> u32 {
        self.processor.get_input_buffer_smoothed_samples()
    }

    fn get_output_buffer_samples(&self) -> u32 {
        self.processor.get_output_buffer_samples()
    }

    fn output_sample_rate(&self) -> u32 {
        self.processor.output_sample_rate()
    }

    fn get_rnnoise_buffer_samples(&self) -> u32 {
        self.processor.get_rnnoise_buffer_samples()
    }

    /// Get smoothed DSP processing time in milliseconds
    fn get_dsp_time_smoothed_ms(&self) -> f32 {
        let us = self.processor.dsp_time_smoothed_us.load(Ordering::Relaxed);
        us as f32 / 1000.0
    }

    /// Get smoothed suppressor buffer fill level in samples
    fn get_buffer_smoothed_samples(&self) -> u32 {
        self.processor.smoothed_buffer_len.load(Ordering::Relaxed)
    }

    // === Dropped Sample Tracking ===

    fn get_dropped_samples(&self) -> u64 {
        self.processor.get_dropped_samples()
    }

    fn reset_dropped_samples(&self) {
        self.processor.reset_dropped_samples();
    }

    fn get_lock_contention_count(&self) -> u64 {
        self.processor.get_lock_contention_count()
    }

    fn reset_lock_contention_count(&self) {
        self.processor.reset_lock_contention_count();
    }

    fn get_input_callback_age_ms(&self) -> u64 {
        self.processor.get_input_callback_age_ms()
    }

    fn get_output_callback_age_ms(&self) -> u64 {
        self.processor.get_output_callback_age_ms()
    }

    fn get_output_underrun_streak(&self) -> u32 {
        self.processor.get_output_underrun_streak()
    }

    fn get_output_underrun_total(&self) -> u64 {
        self.processor.get_output_underrun_total()
    }

    fn get_jitter_dropped_samples(&self) -> u64 {
        self.processor.get_jitter_dropped_samples()
    }

    fn get_output_recovery_count(&self) -> u64 {
        self.processor.get_output_recovery_count()
    }

    fn get_suppressor_non_finite_count(&self) -> u64 {
        self.processor.get_suppressor_non_finite_count()
    }

    fn is_noise_backend_available(&self) -> bool {
        self.processor.is_noise_backend_available()
    }

    fn noise_backend_failed(&self) -> bool {
        self.processor.noise_backend_failed()
    }

    fn noise_backend_error(&self) -> Option<String> {
        self.processor.noise_backend_error()
    }

    fn set_recovery_suppressed(&self, suppressed: bool) {
        self.processor.set_recovery_suppressed(suppressed);
    }

    fn is_recovery_suppressed(&self) -> bool {
        self.processor.is_recovery_suppressed()
    }

    fn get_runtime_diagnostics(&self, py: Python) -> PyResult<PyObject> {
        let diagnostics = pyo3::types::PyDict::new_bound(py);
        diagnostics.set_item("noise_model", self.processor.get_noise_model().id())?;
        diagnostics.set_item(
            "noise_backend_available",
            self.processor.is_noise_backend_available(),
        )?;
        diagnostics.set_item("noise_backend_failed", self.processor.noise_backend_failed())?;
        diagnostics.set_item(
            "noise_backend_error",
            self.processor.noise_backend_error(),
        )?;
        diagnostics.set_item("input_dropped_samples", self.processor.get_dropped_samples())?;
        diagnostics.set_item(
            "lock_contention_count",
            self.processor.get_lock_contention_count(),
        )?;
        diagnostics.set_item(
            "output_underrun_total",
            self.processor.get_output_underrun_total(),
        )?;
        diagnostics.set_item(
            "jitter_dropped_samples",
            self.processor.get_jitter_dropped_samples(),
        )?;
        diagnostics.set_item(
            "stream_restart_count",
            self.processor.get_stream_restart_count(),
        )?;
        diagnostics.set_item(
            "last_restart_reason",
            self.processor.get_last_restart_reason(),
        )?;
        diagnostics.set_item(
            "last_stream_error",
            self.processor.get_last_stream_error(),
        )?;
        diagnostics.set_item(
            "suppressor_non_finite_count",
            self.processor.get_suppressor_non_finite_count(),
        )?;
        diagnostics.set_item("output_sample_rate", self.processor.output_sample_rate())?;
        diagnostics.set_item(
            "recovery_suppressed",
            self.processor.is_recovery_suppressed(),
        )?;
        Ok(diagnostics.into_any().unbind())
    }

    // === Stream Recovery Status ===

    /// Service pending recovery requests (returns None if no attempt).
    fn service_recovery(&mut self) -> Option<bool> {
        self.processor.service_recovery()
    }

    fn is_recovery_requested(&self) -> bool {
        self.processor.is_recovery_requested()
    }

    fn is_recovering(&self) -> bool {
        self.processor.is_recovering()
    }

    fn get_stream_restart_count(&self) -> u64 {
        self.processor.get_stream_restart_count()
    }

    fn get_last_stream_error(&self) -> Option<String> {
        self.processor.get_last_stream_error()
    }

    fn get_last_restart_reason(&self) -> Option<String> {
        self.processor.get_last_restart_reason()
    }

    // === RAW AUDIO RECORDING (for calibration) ===

    /// Start recording raw audio for calibration (10 seconds @ 48kHz)
    fn start_raw_recording(&mut self, duration_secs: f64) -> PyResult<()> {
        self.processor
            .start_raw_recording(duration_secs)
            .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
    }

    /// Stop recording and return audio data as NumPy array
    fn stop_raw_recording(&mut self, py: Python) -> PyResult<PyObject> {
        if let Some(audio) = self.processor.stop_raw_recording() {
            // Zero-copy transfer to NumPy
            use numpy::PyArray1;
            let array = PyArray1::from_vec_bound(py, audio);
            Ok(array.into_any().unbind())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "No recording in progress",
            ))
        }
    }

    /// Check if recording is complete
    fn is_recording_complete(&mut self) -> bool {
        self.processor.is_recording_complete()
    }

    /// Get recording progress (0.0 to 1.0)
    fn recording_progress(&mut self) -> f32 {
        self.processor.recording_progress()
    }

    /// Get current recording level as RMS in dB (for level meter visualization)
    fn recording_level_db(&mut self) -> f32 {
        self.processor.recording_level_db()
    }

    /// Manually set output mute state (useful for calibration workflow)
    fn set_output_mute(&mut self, muted: bool) {
        self.processor.set_output_mute(muted);
    }
}
