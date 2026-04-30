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
use std::cell::Cell;
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
use crate::dsp::vad::{GateMode, SileroVAD, VadAutoGate};

const RESAMPLE_COMPACT_THRESHOLD: usize = 16384;
const PROCESS_IDLE_SLEEP_US: u64 = 100;
const COMPRESSOR_DEFAULT_RELEASE_TENTH_MS: u64 = 2000; // 200ms * 10
const OUTPUT_PRIME_MS: u32 = 20;
const OUTPUT_TARGET_HIGH_MS: u32 = 30;
const OUTPUT_HARD_BACKLOG_MS: u32 = 60;
const OUTPUT_DRIFT_MAX_RATIO_ADJUST: f32 = 0.008;
const OUTPUT_DRIFT_MAX_EXPANSION_RATIO: f32 = 0.96;
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

fn smoothing_coeff_for_time_constant(sample_rate_hz: f32, time_constant_ms: f32) -> f32 {
    if !sample_rate_hz.is_finite()
        || sample_rate_hz <= 0.0
        || !time_constant_ms.is_finite()
        || time_constant_ms <= 0.0
    {
        0.0
    } else {
        (-1.0 / (sample_rate_hz * (time_constant_ms / 1000.0))).exp()
    }
}

fn total_reported_latency_us(
    output_buffer_samples: u64,
    output_sample_rate: u32,
    suppressor_latency_samples: u64,
    limiter_lookahead_samples: u64,
    limiter_enabled: bool,
    processing_sample_rate: u32,
    compensation_us: u64,
) -> u64 {
    let output_latency_us = samples_to_micros(output_buffer_samples, output_sample_rate);
    let suppressor_latency_us =
        samples_to_micros(suppressor_latency_samples, processing_sample_rate);
    let limiter_latency_us = if limiter_enabled {
        samples_to_micros(limiter_lookahead_samples, processing_sample_rate)
    } else {
        0
    };

    output_latency_us
        .saturating_add(suppressor_latency_us)
        .saturating_add(limiter_latency_us)
        .saturating_add(compensation_us)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProcessingPath {
    RawMonitor,
    Bypass,
    Full,
}

#[derive(Default)]
struct InputPreFilterState {
    dc_x1: f32,
    dc_y1: f32,
}

#[inline]
fn select_processing_path(raw_monitor_enabled: bool, bypass_enabled: bool) -> ProcessingPath {
    if raw_monitor_enabled {
        ProcessingPath::RawMonitor
    } else if bypass_enabled {
        ProcessingPath::Bypass
    } else {
        ProcessingPath::Full
    }
}

#[inline]
fn uses_clean_write_path(path: ProcessingPath) -> bool {
    matches!(path, ProcessingPath::RawMonitor)
}

#[inline]
fn sanitize_non_finite_inplace(buffer: &mut [f32]) {
    for sample in buffer.iter_mut() {
        if !sample.is_finite() {
            *sample = 0.0;
        }
    }
}

#[inline]
fn sanitize_and_clamp_output_inplace(buffer: &mut [f32], ceiling_linear: f32) {
    let ceiling = ceiling_linear.clamp(0.0, 1.0);
    for sample in buffer.iter_mut() {
        if !sample.is_finite() {
            *sample = 0.0;
        } else {
            *sample = sample.clamp(-ceiling, ceiling);
        }
    }
}

#[inline]
fn sanitize_and_clamp_input_inplace(
    buffer: &mut [f32],
    clip_event_count: &AtomicU64,
    clip_peak_db: &AtomicU32,
) {
    for sample in buffer.iter_mut() {
        if !sample.is_finite() {
            *sample = 0.0;
            continue;
        }
        let amplitude = sample.abs();
        if amplitude > 1.0 {
            clip_event_count.fetch_add(1, Ordering::Relaxed);
            let peak_db = 20.0 * amplitude.log10();
            let current_peak = f32::from_bits(clip_peak_db.load(Ordering::Relaxed));
            if peak_db > current_peak {
                clip_peak_db.store(peak_db.to_bits(), Ordering::Relaxed);
            }
        }
        *sample = sample.clamp(-1.0, 1.0);
    }
}

#[inline]
fn apply_input_pre_filter(
    buffer: &mut [f32],
    dc_state: &mut InputPreFilterState,
    pre_filter: &mut Biquad,
) {
    for sample in buffer.iter_mut() {
        let input = *sample;
        let output = input - dc_state.dc_x1 + INPUT_DC_BLOCK_COEFF * dc_state.dc_y1;
        dc_state.dc_x1 = input;
        dc_state.dc_y1 = output;
        *sample = pre_filter.process_sample(output);
    }
}

fn retime_audio_block<'a>(
    input: &'a [f32],
    speed_ratio: f32,
    max_output_len: usize,
    output: &'a mut Vec<f32>,
) -> &'a [f32] {
    if input.is_empty() || max_output_len == 0 {
        output.clear();
        return output.as_slice();
    }

    let clamped_ratio = speed_ratio.max(0.5);
    let desired_len = ((input.len() as f32) / clamped_ratio).round().max(1.0) as usize;
    let out_len = desired_len.min(max_output_len);
    if out_len == input.len() {
        return input;
    }

    output.clear();
    output.resize(out_len, 0.0);

    let max_src = (input.len() - 1) as f32;
    for (i, out_sample) in output.iter_mut().enumerate() {
        let src_pos = if out_len == 1 {
            0.0
        } else {
            (i as f32 * clamped_ratio).min(max_src)
        };
        let idx0 = src_pos.floor() as usize;
        let idx1 = (idx0 + 1).min(input.len() - 1);
        let frac = src_pos - idx0 as f32;
        let y0 = input[idx0];
        let y1 = input[idx1];
        *out_sample = y0 + (y1 - y0) * frac;
    }

    output.as_slice()
}

#[inline]
fn release_ms_to_tenth_ms(release_ms: f64) -> u64 {
    if !release_ms.is_finite() {
        return 0;
    }
    (release_ms.max(0.0) * 10.0).round() as u64
}

#[inline]
fn clamp_control_value(value: f64, min_value: f64, max_value: f64) -> Option<f64> {
    value.is_finite().then(|| value.clamp(min_value, max_value))
}

#[inline]
fn clamp_control_value_f32(value: f32, min_value: f32, max_value: f32) -> Option<f32> {
    value.is_finite().then(|| value.clamp(min_value, max_value))
}

fn lock_rt<'a, T>(
    mutex: &'a Mutex<T>,
    lock_contention_count: &AtomicU64,
) -> Option<std::sync::MutexGuard<'a, T>> {
    match mutex.try_lock() {
        Ok(guard) => Some(guard),
        Err(std::sync::TryLockError::WouldBlock) => {
            lock_contention_count.fetch_add(1, Ordering::Relaxed);
            None
        }
        Err(std::sync::TryLockError::Poisoned(_)) => None,
    }
}

fn build_sinc_resampler(
    input_rate: u32,
    output_rate: u32,
    chunk_size: usize,
) -> Result<SincFixedIn<f64>, String> {
    let ratio = output_rate as f64 / input_rate as f64;
    let sinc_len = 128;
    let window = WindowFunction::BlackmanHarris2;
    let params = SincInterpolationParameters {
        sinc_len,
        f_cutoff: calculate_cutoff(sinc_len, window),
        interpolation: SincInterpolationType::Cubic,
        oversampling_factor: 256,
        window,
    };
    SincFixedIn::<f64>::new(ratio, 1.2, params, chunk_size, 1).map_err(|e| e.to_string())
}

fn update_backend_diagnostics(
    available: &AtomicBool,
    failed: &AtomicBool,
    error: &Mutex<Option<String>>,
    suppressor: &NoiseSuppressionEngine,
) {
    available.store(suppressor.backend_available(), Ordering::Relaxed);
    failed.store(suppressor.backend_failed(), Ordering::Relaxed);
    if let Ok(mut guard) = error.lock() {
        *guard = suppressor.backend_error().map(str::to_string);
    }
}

#[derive(Default)]
struct StreamRecoveryState {
    last_error: Option<String>,
    last_reason: Option<String>,
    restart_count: u64,
}

#[derive(Clone)]
struct GateControlState {
    enabled: bool,
    threshold_db: f64,
    attack_ms: f64,
    release_ms: f64,
    #[cfg(feature = "vad")]
    gate_mode: GateMode,
    #[cfg(feature = "vad")]
    vad_threshold: f32,
    #[cfg(feature = "vad")]
    hold_ms: f32,
    #[cfg(feature = "vad")]
    pre_gain: f32,
    #[cfg(feature = "vad")]
    auto_threshold: bool,
    #[cfg(feature = "vad")]
    margin_db: f32,
}

impl GateControlState {
    fn new() -> Self {
        Self {
            enabled: true,
            threshold_db: -40.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            #[cfg(feature = "vad")]
            gate_mode: GateMode::ThresholdOnly,
            #[cfg(feature = "vad")]
            vad_threshold: 0.4,
            #[cfg(feature = "vad")]
            hold_ms: 200.0,
            #[cfg(feature = "vad")]
            pre_gain: 1.0,
            #[cfg(feature = "vad")]
            auto_threshold: true,
            #[cfg(feature = "vad")]
            margin_db: 10.0,
        }
    }
}

#[derive(Clone)]
struct SuppressorControlState {
    enabled: bool,
    model: NoiseModel,
}

impl SuppressorControlState {
    fn new() -> Self {
        Self {
            enabled: true,
            model: NoiseModel::RNNoise,
        }
    }
}

#[derive(Clone, Copy)]
struct EqControlSnapshot {
    enabled: bool,
    bands: [(f64, f64, f64); NUM_BANDS],
}

impl EqControlSnapshot {
    fn new() -> Self {
        Self {
            enabled: true,
            bands: std::array::from_fn(|index| (DEFAULT_FREQUENCIES[index], 0.0, DEFAULT_Q)),
        }
    }
}

struct EqControlState {
    seq: AtomicU64,
    enabled: AtomicBool,
    frequency_bits: [AtomicU64; NUM_BANDS],
    gain_bits: [AtomicU64; NUM_BANDS],
    q_bits: [AtomicU64; NUM_BANDS],
}

impl EqControlState {
    fn new() -> Self {
        let snapshot = EqControlSnapshot::new();
        Self {
            seq: AtomicU64::new(0),
            enabled: AtomicBool::new(snapshot.enabled),
            frequency_bits: std::array::from_fn(|index| {
                AtomicU64::new(snapshot.bands[index].0.to_bits())
            }),
            gain_bits: std::array::from_fn(|index| {
                AtomicU64::new(snapshot.bands[index].1.to_bits())
            }),
            q_bits: std::array::from_fn(|index| AtomicU64::new(snapshot.bands[index].2.to_bits())),
        }
    }

    fn update<F>(&self, apply: F)
    where
        F: FnOnce(&Self),
    {
        self.seq.fetch_add(1, Ordering::AcqRel);
        apply(self);
        self.seq.fetch_add(1, Ordering::Release);
    }

    fn snapshot(&self) -> EqControlSnapshot {
        loop {
            let seq_before = self.seq.load(Ordering::Acquire);
            if (seq_before & 1) != 0 {
                std::hint::spin_loop();
                continue;
            }

            let enabled = self.enabled.load(Ordering::Relaxed);
            let bands = std::array::from_fn(|index| {
                let frequency = f64::from_bits(self.frequency_bits[index].load(Ordering::Relaxed));
                let gain = f64::from_bits(self.gain_bits[index].load(Ordering::Relaxed));
                let q = f64::from_bits(self.q_bits[index].load(Ordering::Relaxed));
                (frequency, gain, q)
            });

            let seq_after = self.seq.load(Ordering::Acquire);
            if seq_before == seq_after {
                return EqControlSnapshot { enabled, bands };
            }
        }
    }

    fn set_enabled(&self, enabled: bool) {
        self.update(|state| {
            state.enabled.store(enabled, Ordering::Relaxed);
        });
    }

    fn set_band_frequency(&self, band: usize, frequency: f64) {
        self.update(|state| {
            state.frequency_bits[band].store(frequency.to_bits(), Ordering::Relaxed);
        });
    }

    fn set_band_gain(&self, band: usize, gain_db: f64) {
        self.update(|state| {
            state.gain_bits[band].store(gain_db.to_bits(), Ordering::Relaxed);
        });
    }

    fn set_band_q(&self, band: usize, q: f64) {
        self.update(|state| {
            state.q_bits[band].store(q.to_bits(), Ordering::Relaxed);
        });
    }

    fn set_bands(&self, bands: &[(f64, f64, f64); NUM_BANDS]) {
        self.update(|state| {
            for (index, (frequency, gain, q)) in bands.iter().copied().enumerate() {
                state.frequency_bits[index].store(frequency.to_bits(), Ordering::Relaxed);
                state.gain_bits[index].store(gain.to_bits(), Ordering::Relaxed);
                state.q_bits[index].store(q.to_bits(), Ordering::Relaxed);
            }
        });
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

fn apply_eq_control(eq: &mut ParametricEQ, control: &EqControlSnapshot) {
    eq.set_enabled(control.enabled);
    for (index, (frequency, gain_db, q)) in control.bands.iter().copied().enumerate() {
        eq.set_band_frequency(index, frequency);
        eq.set_band_gain(index, gain_db);
        eq.set_band_q(index, q);
    }
}

fn apply_gate_control(gate: &mut NoiseGate, control: &GateControlState) {
    gate.set_enabled(control.enabled);
    gate.set_threshold(control.threshold_db);
    gate.set_attack_time(control.attack_ms);
    gate.set_release_time(control.release_ms);
    #[cfg(feature = "vad")]
    {
        gate.set_gate_mode(control.gate_mode);
        gate.set_vad_threshold(control.vad_threshold);
        gate.set_hold_time(control.hold_ms);
        gate.set_vad_pre_gain(control.pre_gain);
        gate.set_auto_threshold(control.auto_threshold);
        gate.set_margin(control.margin_db);
    }
}

fn apply_suppressor_control(
    suppressor: &mut NoiseSuppressionEngine,
    control: &SuppressorControlState,
) {
    suppressor.set_enabled(control.enabled);
}

fn swap_pending_suppressor_if_ready(
    suppressor: &mut NoiseSuppressionEngine,
    control: &SuppressorControlState,
    pending: &mut Option<NoiseSuppressionEngine>,
) -> bool {
    if suppressor.model_type() == control.model {
        return true;
    }

    let Some(candidate) = pending.as_ref() else {
        return false;
    };
    if candidate.model_type() != control.model {
        return false;
    }

    *suppressor = pending
        .take()
        .expect("pending suppressor was checked above");
    true
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
    limiter.set_ceiling(control.ceiling_db);
    limiter.set_release_time(control.release_ms);
    limiter.set_enabled(control.enabled);
}

/// Main audio processor combining all DSP stages
pub struct AudioProcessor {
    /// Noise gate
    gate: Arc<Mutex<NoiseGate>>,
    gate_enabled: Arc<AtomicBool>,
    gate_control: Arc<Mutex<GateControlState>>,
    gate_dirty: Arc<AtomicBool>,

    /// Noise suppression engine (RNNoise or DeepFilterNet)
    suppressor: Arc<Mutex<NoiseSuppressionEngine>>,
    suppressor_enabled: Arc<AtomicBool>,
    suppressor_control: Arc<Mutex<SuppressorControlState>>,
    suppressor_dirty: Arc<AtomicBool>,
    suppressor_reset_requested: Arc<AtomicBool>,
    pending_suppressor: Arc<Mutex<Option<NoiseSuppressionEngine>>>,
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
    /// True raw monitor path: bypasses pre-filter/DSP and uses minimal output write path.
    raw_monitor_enabled: Arc<AtomicBool>,

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
    /// Last known gate gain (0.0-1.0) for metering.
    gate_gain_meter: Arc<AtomicU32>,
    /// VAD speech probability (0.0-1.0) for metering
    vad_probability: Arc<AtomicU32>,
    /// Latest gate noise-floor estimate in dB.
    gate_noise_floor_db: Arc<AtomicU32>,
    /// Whether the current VAD backend is available.
    vad_available: Arc<AtomicBool>,
    #[cfg(feature = "vad")]
    /// Audio copied from the DSP loop for non-realtime VAD inference.
    vad_worker_buffer: Arc<Mutex<Vec<f32>>>,
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
    /// Samples discarded because the output ring buffer could not accept the full write.
    output_short_write_dropped_samples: Arc<AtomicU64>,

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
            gate_dirty: Arc::new(AtomicBool::new(false)),
            suppressor: Arc::new(Mutex::new(suppressor)),
            suppressor_enabled: Arc::new(AtomicBool::new(true)),
            suppressor_control: Arc::new(Mutex::new(suppressor_control_state)),
            suppressor_dirty: Arc::new(AtomicBool::new(false)),
            suppressor_reset_requested: Arc::new(AtomicBool::new(false)),
            pending_suppressor: Arc::new(Mutex::new(None)),
            suppressor_strength, // Store Arc for PyO3 bindings
            current_model: Arc::new(AtomicU8::new(NoiseModel::RNNoise as u8)),
            eq: Arc::new(Mutex::new(ParametricEQ::new(sample_rate as f64))),
            eq_enabled: Arc::new(AtomicBool::new(true)),
            eq_control: Arc::new(EqControlState::new()),
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
            raw_monitor_enabled: Arc::new(AtomicBool::new(false)),
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
            gate_gain_meter: Arc::new(AtomicU32::new(1.0_f32.to_bits())),
            vad_probability: Arc::new(AtomicU32::new(0.0_f32.to_bits())),
            gate_noise_floor_db: Arc::new(AtomicU32::new((-60.0_f32).to_bits())),
            vad_available: Arc::new(AtomicBool::new(false)),
            #[cfg(feature = "vad")]
            vad_worker_buffer: Arc::new(Mutex::new(Vec::with_capacity(
                VAD_WORKER_MAX_BUFFER_SAMPLES,
            ))),
            #[cfg(feature = "vad")]
            vad_worker_thread: None,
            #[cfg(feature = "vad")]
            vad_worker_running: Arc::new(AtomicBool::new(false)),
            #[cfg(feature = "vad")]
            vad_last_update_us: Arc::new(AtomicU64::new(0)),
            compressor_current_release_ms: Arc::new(AtomicU64::new(
                COMPRESSOR_DEFAULT_RELEASE_TENTH_MS,
            )),
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
            output_short_write_dropped_samples: Arc::new(AtomicU64::new(0)),
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
            input_resampler_active: Arc::new(AtomicBool::new(false)),
            output_resampler_active: Arc::new(AtomicBool::new(false)),
            noise_backend_available: Arc::new(AtomicBool::new(true)),
            noise_backend_failed: Arc::new(AtomicBool::new(false)),
            noise_backend_error: Arc::new(Mutex::new(None)),

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

    #[cfg(feature = "vad")]
    fn ensure_vad_worker(&mut self) {
        if self.vad_worker_thread.is_some() {
            return;
        }

        self.vad_worker_running.store(true, Ordering::Release);
        let running = Arc::clone(&self.vad_worker_running);
        let worker_buffer = Arc::clone(&self.vad_worker_buffer);
        let probability = Arc::clone(&self.vad_probability);
        let available = Arc::clone(&self.vad_available);
        let last_update_us = Arc::clone(&self.vad_last_update_us);
        let sample_rate = self.sample_rate;
        let threshold = self
            .gate_control
            .lock()
            .map(|control| control.vad_threshold)
            .unwrap_or(0.5);

        self.vad_worker_thread = Some(std::thread::spawn(move || {
            let mut vad = match SileroVAD::new(sample_rate, threshold) {
                Ok(vad) => {
                    available.store(true, Ordering::Release);
                    vad
                }
                Err(_) => {
                    available.store(false, Ordering::Release);
                    while running.load(Ordering::Acquire) {
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                    return;
                }
            };

            let mut local = Vec::with_capacity(VAD_WORKER_MAX_BUFFER_SAMPLES);
            while running.load(Ordering::Acquire) {
                let mut has_audio = false;
                if let Ok(mut shared) = worker_buffer.lock() {
                    if !shared.is_empty() {
                        std::mem::swap(&mut local, &mut *shared);
                        has_audio = true;
                    }
                }

                if has_audio {
                    match vad.process(&local) {
                        Ok(prob) => {
                            probability.store(prob.clamp(0.0, 1.0).to_bits(), Ordering::Release);
                            last_update_us.store(now_micros(), Ordering::Release);
                            available.store(true, Ordering::Release);
                        }
                        Err(_) => {
                            available.store(false, Ordering::Release);
                        }
                    }
                    local.clear();
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            }
        }));
    }

    #[cfg(feature = "vad")]
    fn stop_vad_worker(&mut self) {
        self.vad_worker_running.store(false, Ordering::Release);
        if let Some(handle) = self.vad_worker_thread.take() {
            let _ = handle.join();
        }
        self.vad_available.store(false, Ordering::Release);
        self.vad_last_update_us.store(0, Ordering::Release);
        if let Ok(mut buffer) = self.vad_worker_buffer.lock() {
            buffer.clear();
        }
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
        self.output_short_write_dropped_samples
            .store(0, Ordering::Relaxed);
        self.lock_contention_count.store(0, Ordering::Relaxed);
        self.suppressor_non_finite_count.store(0, Ordering::Relaxed);
        self.input_backlog_recovery_count
            .store(0, Ordering::Relaxed);
        self.input_backlog_dropped_samples
            .store(0, Ordering::Relaxed);
        self.clip_event_count.store(0, Ordering::Relaxed);
        self.clip_peak_db
            .store((-120.0_f32).to_bits(), Ordering::Relaxed);
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

        const RESAMPLER_CHUNK_SIZE: usize = 1024;
        let input_resampler = if input_sample_rate_for_thread != self.sample_rate {
            Some(
                build_sinc_resampler(
                    input_sample_rate_for_thread,
                    self.sample_rate,
                    RESAMPLER_CHUNK_SIZE,
                )
                .map_err(|e| {
                    self.stop();
                    format!(
                        "Failed to initialize input resampler ({} Hz -> {} Hz): {}",
                        input_sample_rate_for_thread, self.sample_rate, e
                    )
                })?,
            )
        } else {
            None
        };
        let output_resampler = if output_sample_rate_for_thread != self.sample_rate {
            Some(
                build_sinc_resampler(
                    self.sample_rate,
                    output_sample_rate_for_thread,
                    RESAMPLER_CHUNK_SIZE,
                )
                .map_err(|e| {
                    self.stop();
                    format!(
                        "Failed to initialize output resampler ({} Hz -> {} Hz): {}",
                        self.sample_rate, output_sample_rate_for_thread, e
                    )
                })?,
            )
        } else {
            None
        };
        self.input_resampler_active
            .store(input_resampler.is_some(), Ordering::Relaxed);
        self.output_resampler_active
            .store(output_resampler.is_some(), Ordering::Relaxed);

        // Start processing thread
        #[cfg(feature = "vad")]
        self.ensure_vad_worker();
        self.running.store(true, Ordering::SeqCst);
        self.last_start_time_us
            .store(now_micros(), Ordering::Release);

        let gate = Arc::clone(&self.gate);
        let gate_enabled = Arc::clone(&self.gate_enabled);
        let gate_control = Arc::clone(&self.gate_control);
        let gate_dirty = Arc::clone(&self.gate_dirty);
        let suppressor = Arc::clone(&self.suppressor);
        let suppressor_enabled = Arc::clone(&self.suppressor_enabled);
        let suppressor_control = Arc::clone(&self.suppressor_control);
        let suppressor_dirty = Arc::clone(&self.suppressor_dirty);
        let suppressor_reset_requested = Arc::clone(&self.suppressor_reset_requested);
        let pending_suppressor = Arc::clone(&self.pending_suppressor);
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
        let raw_monitor_enabled = Arc::clone(&self.raw_monitor_enabled);

        // Clone metering atomics
        let input_peak = Arc::clone(&self.input_peak);
        let input_rms = Arc::clone(&self.input_rms);
        let output_peak = Arc::clone(&self.output_peak);
        let output_rms = Arc::clone(&self.output_rms);
        let compressor_gain_reduction = Arc::clone(&self.compressor_gain_reduction);
        let deesser_gain_reduction = Arc::clone(&self.deesser_gain_reduction);
        let gate_gain_meter = Arc::clone(&self.gate_gain_meter);
        let vad_probability = Arc::clone(&self.vad_probability);
        let gate_noise_floor_db = Arc::clone(&self.gate_noise_floor_db);
        let vad_available = Arc::clone(&self.vad_available);
        #[cfg(feature = "vad")]
        let vad_worker_buffer = Arc::clone(&self.vad_worker_buffer);
        #[cfg(feature = "vad")]
        let vad_last_update_us = Arc::clone(&self.vad_last_update_us);
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
        let jitter_dropped_samples = Arc::clone(&self.jitter_dropped_samples);
        let output_recovery_count = Arc::clone(&self.output_recovery_count);
        let output_short_write_dropped_samples =
            Arc::clone(&self.output_short_write_dropped_samples);
        let lock_contention_count = Arc::clone(&self.lock_contention_count);
        let suppressor_non_finite_count = Arc::clone(&self.suppressor_non_finite_count);
        let input_backlog_recovery_count = Arc::clone(&self.input_backlog_recovery_count);
        let input_backlog_dropped_samples = Arc::clone(&self.input_backlog_dropped_samples);
        let clip_event_count = Arc::clone(&self.clip_event_count);
        let clip_peak_db = Arc::clone(&self.clip_peak_db);
        let noise_backend_available = Arc::clone(&self.noise_backend_available);
        let noise_backend_failed = Arc::clone(&self.noise_backend_failed);
        let noise_backend_error = Arc::clone(&self.noise_backend_error);
        let recording_active_thread = Arc::clone(&recording_active);

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
            let mut resampler = input_resampler;
            let mut resampler_out = resampler.as_ref().map(|r| r.output_buffer_allocate(false));
            let mut output_resample_input: Vec<f64> = Vec::with_capacity(65536);
            let mut output_resample_read_pos: usize = 0;
            let mut output_resampler = output_resampler;
            let mut output_resampler_out = output_resampler
                .as_ref()
                .map(|r| r.output_buffer_allocate(false));
            let mut eq_rt = ParametricEQ::new(sample_rate_for_latency as f64);
            let mut compressor_rt = Compressor::default_voice(sample_rate_for_latency as f64);
            let mut deesser_rt = DeEsser::new(sample_rate_for_latency as f64);
            let mut limiter_rt = Limiter::default_settings(sample_rate_for_latency as f64);
            let output_ceiling_linear =
                Cell::new(10.0_f32.powf(limiter_rt.ceiling_db() as f32 / 20.0));
            let limiter_lookahead_samples =
                Arc::new(AtomicU64::new(limiter_rt.lookahead_samples() as u64));
            let limiter_lookahead_samples_for_chain = Arc::clone(&limiter_lookahead_samples);
            if let Ok(control) = gate_control.lock() {
                if let Ok(mut gate_rt) = gate.lock() {
                    apply_gate_control(&mut gate_rt, &control);
                    #[cfg(feature = "vad")]
                    {
                        gate_noise_floor_db
                            .store(gate_rt.noise_floor().to_bits(), Ordering::Relaxed);
                        vad_available.store(gate_rt.is_vad_available(), Ordering::Relaxed);
                    }
                }
            }
            if let Ok(control) = suppressor_control.lock() {
                if let Ok(mut suppressor_rt) = suppressor.lock() {
                    apply_suppressor_control(&mut suppressor_rt, &control);
                    update_backend_diagnostics(
                        &noise_backend_available,
                        &noise_backend_failed,
                        noise_backend_error.as_ref(),
                        &suppressor_rt,
                    );
                }
            }
            let eq_snapshot = eq_control.snapshot();
            apply_eq_control(&mut eq_rt, &eq_snapshot);
            if let Ok(control) = compressor_control.lock() {
                apply_compressor_control(&mut compressor_rt, &control);
            }
            if let Ok(control) = deesser_control.lock() {
                apply_deesser_control(&mut deesser_rt, &control);
            }
            if let Ok(control) = limiter_control.lock() {
                apply_limiter_control(&mut limiter_rt, &control);
                output_ceiling_linear.set(10.0_f32.powf(limiter_rt.ceiling_db() as f32 / 20.0));
            }
            let mut output_resampled_scratch: Vec<f32> = Vec::with_capacity(8192);
            let mut output_queue_control_scratch: Vec<f32> = Vec::with_capacity(8192);

            let mut pre_filter_state = InputPreFilterState::default();

            // Pre-filter at 80Hz to remove rumble before gate/suppressor stages.
            let mut pre_filter = Biquad::new(
                BiquadType::HighPass,
                INPUT_PREFILTER_HZ,
                0.0,
                INPUT_PREFILTER_Q,
                sample_rate_for_latency as f64,
            );

            // Metering state (IIR smoothing for RMS)
            let mut input_rms_acc: f32 = 0.0;
            let mut output_rms_acc: f32 = 0.0;
            let meter_coeff =
                smoothing_coeff_for_time_constant(sample_rate_for_latency as f32, 100.0);

            // Latency tracking
            let mut last_latency_update = Instant::now();
            let mut last_heartbeat = Instant::now();
            let latency_update_interval = std::time::Duration::from_millis(100); // Update every 100ms
            const HEARTBEAT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);
            const STALL_THRESHOLD_MS: u64 = 3000; // 3 seconds without write = stall
            const SUPPRESSOR_STARVATION_MS: u64 = 400; // Reset suppressor if no output this long
            const SUPPRESSOR_RECOVERY_COOLDOWN_MS: u64 = 2000;
            const NON_FINITE_REBUILD_THRESHOLD: u32 = 3;
            const NON_FINITE_REBUILD_WINDOW_MS: u64 = 2000;
            let mut last_suppressor_recovery =
                Instant::now() - std::time::Duration::from_millis(SUPPRESSOR_RECOVERY_COOLDOWN_MS);
            let mut suppressor_soft_reset_pending = false;
            let mut non_finite_window_started_at: Option<Instant> = None;
            let mut non_finite_window_count: u32 = 0;

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
                    *rms_acc = meter_coeff * *rms_acc + (1.0 - meter_coeff) * (sample * sample);
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

            macro_rules! apply_downstream_chain_rt {
                ($buffer:expr) => {{
                    if deesser_dirty.load(Ordering::Acquire) {
                        if let Some(control) =
                            lock_rt(deesser_control.as_ref(), &lock_contention_count)
                        {
                            apply_deesser_control(&mut deesser_rt, &control);
                            deesser_dirty.store(false, Ordering::Release);
                        }
                    }
                    if eq_dirty.load(Ordering::Acquire) {
                        let eq_snapshot = eq_control.snapshot();
                        apply_eq_control(&mut eq_rt, &eq_snapshot);
                        eq_dirty.store(false, Ordering::Release);
                    }
                    if compressor_dirty.load(Ordering::Acquire) {
                        if let Some(control) =
                            lock_rt(compressor_control.as_ref(), &lock_contention_count)
                        {
                            apply_compressor_control(&mut compressor_rt, &control);
                            compressor_dirty.store(false, Ordering::Release);
                        }
                    }
                    if limiter_dirty.load(Ordering::Acquire) {
                        if let Some(control) =
                            lock_rt(limiter_control.as_ref(), &lock_contention_count)
                        {
                            apply_limiter_control(&mut limiter_rt, &control);
                            output_ceiling_linear
                                .set(10.0_f32.powf(limiter_rt.ceiling_db() as f32 / 20.0));
                            limiter_lookahead_samples_for_chain
                                .store(limiter_rt.lookahead_samples() as u64, Ordering::Relaxed);
                            limiter_dirty.store(false, Ordering::Release);
                        }
                    }

                    if deesser_enabled.load(Ordering::Acquire) {
                        deesser_rt.process_block_inplace($buffer);
                        deesser_gain_reduction.store(
                            deesser_rt.current_gain_reduction_db().to_bits(),
                            Ordering::Relaxed,
                        );
                    } else {
                        deesser_gain_reduction.store(0.0_f32.to_bits(), Ordering::Relaxed);
                    }

                    if eq_enabled.load(Ordering::Acquire) {
                        eq_rt.process_block_inplace($buffer);
                    }

                    if compressor_enabled.load(Ordering::Acquire) {
                        compressor_rt.process_block_inplace($buffer);
                        compressor_gain_reduction.store(
                            (compressor_rt.current_gain_reduction() as f32).to_bits(),
                            Ordering::Relaxed,
                        );
                        let current_release = compressor_rt.current_release_time();
                        compressor_current_release_ms
                            .store(release_ms_to_tenth_ms(current_release), Ordering::Relaxed);
                    } else {
                        compressor_gain_reduction.store(0.0_f32.to_bits(), Ordering::Relaxed);
                        compressor_current_release_ms
                            .store(COMPRESSOR_DEFAULT_RELEASE_TENTH_MS, Ordering::Relaxed);
                    }

                    if limiter_enabled.load(Ordering::Acquire) {
                        limiter_rt.process_block_inplace($buffer);
                    }
                }};
            }

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
            let output_target_center_samples =
                (output_target_low_samples + output_target_high_samples) / 2;
            let output_hard_backlog_samples =
                duration_samples(output_sample_rate_for_latency, OUTPUT_HARD_BACKLOG_MS);
            const OUTPUT_MAX_CATCHUP_RATIO: f32 = 1.03;
            const OUTPUT_MAX_EMERGENCY_CATCHUP_RATIO: f32 = 1.06;
            let input_backlog_high_samples = duration_samples(input_sample_rate_for_thread, 250);
            let input_backlog_low_samples = duration_samples(input_sample_rate_for_thread, 100);
            let discontinuity_fade_samples =
                duration_samples(output_sample_rate_for_latency, 6).max(1);
            let discontinuity_fade_remaining = Cell::new(0usize);
            let mut discontinuity_fade_scratch: Vec<f32> = Vec::with_capacity(8192);
            let mut output_safety_scratch: Vec<f32> = Vec::with_capacity(8192);
            let mut output_drift_error_ema = 0.0_f32;
            let mut write_output = |samples: &[f32], clean_path: bool| {
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
                                output_resample_input
                                    .reserve(output_resample_input.capacity().max(1024));
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

                let mut write_slice = write_source;
                if !clean_path {
                    let error = fill as f32 - output_target_center_samples as f32;
                    output_drift_error_ema = output_drift_error_ema * 0.85 + error * 0.15;
                    let positive_zone = output_hard_backlog_samples
                        .saturating_sub(output_target_center_samples)
                        .max(1) as f32;
                    let negative_zone = output_target_center_samples.max(1) as f32;
                    let normalized_error = if output_drift_error_ema >= 0.0 {
                        (output_drift_error_ema / positive_zone).clamp(0.0, 1.0)
                    } else {
                        (output_drift_error_ema / negative_zone).clamp(-1.0, 0.0)
                    };
                    let mut queue_speed_ratio = (1.0
                        + normalized_error * OUTPUT_DRIFT_MAX_RATIO_ADJUST)
                        .clamp(OUTPUT_DRIFT_MAX_EXPANSION_RATIO, OUTPUT_MAX_CATCHUP_RATIO);
                    if fill >= output_hard_backlog_samples {
                        queue_speed_ratio = OUTPUT_MAX_EMERGENCY_CATCHUP_RATIO;
                    }

                    let adjusted_slice = retime_audio_block(
                        write_source,
                        queue_speed_ratio,
                        capacity.max(1),
                        &mut output_queue_control_scratch,
                    );
                    if adjusted_slice.len() != write_source.len() {
                        let delta = write_source.len().abs_diff(adjusted_slice.len());
                        if adjusted_slice.len() < write_source.len() {
                            jitter_dropped_samples.fetch_add(delta as u64, Ordering::Relaxed);
                        }
                        output_recovery_count.fetch_add(1, Ordering::Relaxed);
                    }
                    write_slice = adjusted_slice;

                    let fade_remaining = discontinuity_fade_remaining.get();
                    if fade_remaining > 0 && !write_slice.is_empty() {
                        discontinuity_fade_scratch.clear();
                        discontinuity_fade_scratch.extend_from_slice(write_slice);
                        let fade_count = fade_remaining.min(discontinuity_fade_scratch.len());
                        let elapsed = discontinuity_fade_samples.saturating_sub(fade_remaining);
                        let fade_total = discontinuity_fade_samples as f32;
                        for (i, sample) in discontinuity_fade_scratch
                            .iter_mut()
                            .enumerate()
                            .take(fade_count)
                        {
                            let progress = ((elapsed + i + 1) as f32 / fade_total).clamp(0.0, 1.0);
                            *sample *= progress;
                        }
                        discontinuity_fade_remaining.set(fade_remaining.saturating_sub(fade_count));
                        write_slice = discontinuity_fade_scratch.as_slice();
                    }

                    // Low watermark currently informational; could be used for adaptive refill.
                    let _below_low_target = fill < output_target_low_samples;
                }

                output_safety_scratch.clear();
                output_safety_scratch.extend_from_slice(write_slice);
                let output_ceiling = if limiter_enabled.load(Ordering::Acquire) {
                    output_ceiling_linear.get()
                } else {
                    1.0
                };
                sanitize_and_clamp_output_inplace(&mut output_safety_scratch, output_ceiling);

                let mut pending_slice = output_safety_scratch.as_slice();
                if pending_slice.len() > free {
                    let dropped = pending_slice.len() - free;
                    output_short_write_dropped_samples.fetch_add(dropped as u64, Ordering::Relaxed);
                    output_recovery_count.fetch_add(1, Ordering::Relaxed);
                    discontinuity_fade_remaining.set(discontinuity_fade_samples);
                    pending_slice = &pending_slice[..free];
                }

                if !pending_slice.is_empty() {
                    let written = output_producer.write(pending_slice);
                    if written > 0 {
                        update_write_time();
                    }
                    if written < pending_slice.len() {
                        let dropped = pending_slice.len() - written;
                        output_short_write_dropped_samples
                            .fetch_add(dropped as u64, Ordering::Relaxed);
                        output_recovery_count.fetch_add(1, Ordering::Relaxed);
                        discontinuity_fade_remaining.set(discontinuity_fade_samples);
                    }
                }

                let new_fill = output_producer
                    .capacity()
                    .saturating_sub(output_producer.free_len());
                output_buffer_len.store(new_fill as u32, Ordering::Relaxed);
            };
            let mut previous_processing_path = select_processing_path(
                raw_monitor_enabled.load(Ordering::Acquire),
                bypass.load(Ordering::SeqCst),
            );

            // Run entire processing loop with denormals flushed to zero
            // This prevents tiny floating point values from causing CPU stalls and audio artifacts
            // SAFETY: This only modifies floating point control flags for this thread
            unsafe {
                no_denormals::no_denormals(|| {
                    while running.load(Ordering::SeqCst) {
                        // Record input buffer fill level (samples waiting to be processed)
                        let mut raw_input_len = consumer.len();
                        if raw_input_len > input_backlog_high_samples {
                            let mut to_drop =
                                raw_input_len.saturating_sub(input_backlog_low_samples);
                            let mut dropped_total = 0usize;
                            while to_drop > 0 {
                                let batch = to_drop.min(input_buffer.len());
                                let dropped = consumer.read(&mut input_buffer[..batch]);
                                if dropped == 0 {
                                    break;
                                }
                                to_drop = to_drop.saturating_sub(dropped);
                                dropped_total += dropped;
                            }
                            if dropped_total > 0 {
                                input_backlog_recovery_count.fetch_add(1, Ordering::Relaxed);
                                input_backlog_dropped_samples
                                    .fetch_add(dropped_total as u64, Ordering::Relaxed);
                                resample_input.clear();
                                resample_read_pos = 0;
                                if let Some(outbuf) = resampler_out.as_mut() {
                                    outbuf[0].clear();
                                }
                                if !raw_monitor_enabled.load(Ordering::Acquire) {
                                    discontinuity_fade_remaining.set(discontinuity_fade_samples);
                                }
                                raw_input_len = consumer.len();
                            }
                        }
                        let raw_input_len = raw_input_len as u32;
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
                                            resample_input
                                                .reserve(resample_input.capacity().max(1024));
                                        }
                                    }
                                    resample_input.push(sample as f64);
                                }

                                let mut produced = 0usize;
                                let input_frames_needed = resampler.input_frames_next();
                                while resample_input.len().saturating_sub(resample_read_pos)
                                    >= input_frames_needed
                                {
                                    if let Some(outbuf) = resampler_out.as_mut() {
                                        let in_slices = [&resample_input[resample_read_pos
                                            ..resample_read_pos + input_frames_needed]];
                                        if let Ok((_nbr_in, nbr_out)) =
                                            resampler.process_into_buffer(&in_slices, outbuf, None)
                                        {
                                            let channel_out = &outbuf[0];
                                            let required = produced.saturating_add(nbr_out);
                                            if required > temp_buffer.len() {
                                                temp_buffer.resize(required, 0.0);
                                            }
                                            for &sample in channel_out.iter().take(nbr_out) {
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
                                if n_raw > temp_buffer.len() {
                                    temp_buffer.resize(n_raw, 0.0);
                                }
                                temp_buffer[..n_raw].copy_from_slice(&input_buffer[..n_raw]);
                                n_raw
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
                            let processing_path = select_processing_path(
                                raw_monitor_enabled.load(Ordering::Acquire),
                                bypass.load(Ordering::SeqCst),
                            );
                            if processing_path != previous_processing_path {
                                pre_filter_state = InputPreFilterState::default();
                                input_rms_acc = 0.0;
                                output_rms_acc = 0.0;
                                gate_gain_meter.store(1.0_f32.to_bits(), Ordering::Relaxed);
                                compressor_gain_reduction
                                    .store(0.0_f32.to_bits(), Ordering::Relaxed);
                                deesser_gain_reduction.store(0.0_f32.to_bits(), Ordering::Relaxed);
                                compressor_current_release_ms
                                    .store(COMPRESSOR_DEFAULT_RELEASE_TENTH_MS, Ordering::Relaxed);
                                suppressor_buffer_len.store(0, Ordering::Relaxed);
                                suppressor_latency_samples.store(0, Ordering::Relaxed);
                                smoothed_buffer_len.store(0, Ordering::Relaxed);
                                discontinuity_fade_remaining.set(discontinuity_fade_samples);

                                if let Some(mut gate_rt) =
                                    lock_rt(gate.as_ref(), &lock_contention_count)
                                {
                                    gate_rt.reset();
                                    if let Some(control) =
                                        lock_rt(gate_control.as_ref(), &lock_contention_count)
                                    {
                                        apply_gate_control(&mut gate_rt, &control);
                                    }
                                }
                                eq_rt.reset();
                                compressor_rt.reset();
                                deesser_rt.reset();
                                limiter_rt.reset();
                                if let Some(mut suppressor_rt) =
                                    lock_rt(suppressor.as_ref(), &lock_contention_count)
                                {
                                    suppressor_rt.soft_reset();
                                    update_backend_diagnostics(
                                        &noise_backend_available,
                                        &noise_backend_failed,
                                        noise_backend_error.as_ref(),
                                        &suppressor_rt,
                                    );
                                }
                                previous_processing_path = processing_path;
                            }

                            if processing_path == ProcessingPath::RawMonitor {
                                sanitize_non_finite_inplace(buffer);

                                measure_levels(buffer, &mut input_rms_acc, &input_peak, &input_rms);

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
                                            let sum_sq: f32 = level_slice
                                                .iter()
                                                .map(|sample| sample * sample)
                                                .sum();
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

                                measure_levels(
                                    buffer,
                                    &mut output_rms_acc,
                                    &output_peak,
                                    &output_rms,
                                );
                                compressor_gain_reduction
                                    .store(0.0_f32.to_bits(), Ordering::Relaxed);
                                deesser_gain_reduction.store(0.0_f32.to_bits(), Ordering::Relaxed);
                                gate_gain_meter.store(1.0_f32.to_bits(), Ordering::Relaxed);
                                compressor_current_release_ms
                                    .store(COMPRESSOR_DEFAULT_RELEASE_TENTH_MS, Ordering::Relaxed);
                                suppressor_buffer_len.store(0, Ordering::Relaxed);
                                suppressor_latency_samples.store(0, Ordering::Relaxed);
                                smoothed_buffer_len.store(0, Ordering::Relaxed);

                                write_output(buffer, uses_clean_write_path(processing_path));
                                let raw_dsp_us = dsp_start.elapsed().as_micros() as u64;
                                let prev_smoothed = dsp_time_smoothed_us.load(Ordering::Relaxed);
                                let smoothed = smooth_dsp_time(raw_dsp_us, prev_smoothed);
                                dsp_time_us.store(raw_dsp_us, Ordering::Relaxed);
                                dsp_time_smoothed_us.store(smoothed, Ordering::Relaxed);
                                continue;
                            }

                            sanitize_and_clamp_input_inplace(
                                buffer,
                                &clip_event_count,
                                &clip_peak_db,
                            );
                            apply_input_pre_filter(buffer, &mut pre_filter_state, &mut pre_filter);

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

                                    let window_len = (sample_rate_for_latency as usize / 10).max(1);
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

                            if processing_path == ProcessingPath::Bypass {
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

                                write_output(buffer, uses_clean_write_path(processing_path));
                                // Record DSP processing time
                                let raw_dsp_us = dsp_start.elapsed().as_micros() as u64;
                                let prev_smoothed = dsp_time_smoothed_us.load(Ordering::Relaxed);
                                let smoothed = smooth_dsp_time(raw_dsp_us, prev_smoothed);
                                dsp_time_us.store(raw_dsp_us, Ordering::Relaxed);
                                dsp_time_smoothed_us.store(smoothed, Ordering::Relaxed);
                            } else {
                                // Stage 1: Noise Gate
                                if gate_enabled.load(Ordering::Acquire) {
                                    if gate_dirty.load(Ordering::Acquire) {
                                        if let Some(control) =
                                            lock_rt(gate_control.as_ref(), &lock_contention_count)
                                        {
                                            if let Some(mut g) =
                                                lock_rt(gate.as_ref(), &lock_contention_count)
                                            {
                                                apply_gate_control(&mut g, &control);
                                                gate_dirty.store(false, Ordering::Release);
                                            }
                                        }
                                    }
                                    if let Some(mut g) =
                                        lock_rt(gate.as_ref(), &lock_contention_count)
                                    {
                                        #[cfg(feature = "vad")]
                                        {
                                            if let Ok(mut vad_input) = vad_worker_buffer.try_lock()
                                            {
                                                let remaining = VAD_WORKER_MAX_BUFFER_SAMPLES
                                                    .saturating_sub(vad_input.len());
                                                let to_copy = remaining.min(buffer.len());
                                                vad_input.extend_from_slice(&buffer[..to_copy]);
                                            } else {
                                                lock_contention_count
                                                    .fetch_add(1, Ordering::Relaxed);
                                            }

                                            let latest_prob = f32::from_bits(
                                                vad_probability.load(Ordering::Acquire),
                                            );
                                            let last_update =
                                                vad_last_update_us.load(Ordering::Acquire);
                                            let fresh = last_update > 0
                                                && now_micros().saturating_sub(last_update)
                                                    <= VAD_PROBABILITY_STALE_US;
                                            let worker_available =
                                                vad_available.load(Ordering::Acquire) && fresh;
                                            g.set_external_vad_probability(
                                                latest_prob,
                                                worker_available,
                                            );
                                        }
                                        g.process_block_inplace(buffer);
                                        gate_gain_meter.store(
                                            g.current_gain().clamp(0.0, 1.0).to_bits(),
                                            Ordering::Relaxed,
                                        );
                                        #[cfg(feature = "vad")]
                                        {
                                            let prob = g.get_vad_probability();
                                            vad_probability
                                                .store(prob.to_bits(), Ordering::Relaxed);
                                            gate_noise_floor_db.store(
                                                g.noise_floor().to_bits(),
                                                Ordering::Relaxed,
                                            );
                                            let last_update =
                                                vad_last_update_us.load(Ordering::Acquire);
                                            let fresh = last_update > 0
                                                && now_micros().saturating_sub(last_update)
                                                    <= VAD_PROBABILITY_STALE_US;
                                            vad_available.store(
                                                g.is_vad_available() && fresh,
                                                Ordering::Relaxed,
                                            );
                                        }
                                    }
                                }

                                // Stage 2: Noise Suppression (RNNoise or DeepFilterNet)
                                let use_suppressor = suppressor_enabled.load(Ordering::Acquire);
                                if use_suppressor {
                                    if suppressor_dirty.load(Ordering::Acquire) {
                                        if let Some(control) = lock_rt(
                                            suppressor_control.as_ref(),
                                            &lock_contention_count,
                                        ) {
                                            if let Some(mut s) =
                                                lock_rt(suppressor.as_ref(), &lock_contention_count)
                                            {
                                                let model_ready = if s.model_type() != control.model
                                                {
                                                    if let Some(mut pending) = lock_rt(
                                                        pending_suppressor.as_ref(),
                                                        &lock_contention_count,
                                                    ) {
                                                        swap_pending_suppressor_if_ready(
                                                            &mut s,
                                                            &control,
                                                            &mut pending,
                                                        )
                                                    } else {
                                                        false
                                                    }
                                                } else {
                                                    true
                                                };

                                                if model_ready {
                                                    apply_suppressor_control(&mut s, &control);
                                                    update_backend_diagnostics(
                                                        &noise_backend_available,
                                                        &noise_backend_failed,
                                                        noise_backend_error.as_ref(),
                                                        &s,
                                                    );
                                                    suppressor_dirty
                                                        .store(false, Ordering::Release);
                                                }
                                            }
                                        }
                                    }
                                    if suppressor_reset_requested.swap(false, Ordering::AcqRel) {
                                        if let Some(mut s) =
                                            lock_rt(suppressor.as_ref(), &lock_contention_count)
                                        {
                                            s.soft_reset();
                                            update_backend_diagnostics(
                                                &noise_backend_available,
                                                &noise_backend_failed,
                                                noise_backend_error.as_ref(),
                                                &s,
                                            );
                                        }
                                    }
                                    if let Some(mut s) =
                                        lock_rt(suppressor.as_ref(), &lock_contention_count)
                                    {
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
                                                        if suppressor_soft_reset_pending {
                                                            s.soft_reset();
                                                            suppressor_soft_reset_pending = false;
                                                        } else {
                                                            processor_debug_log!(
                                                                "[PROCESSING] WARNING: Suppressor starvation detected (pending={}, no output for {} ms). Soft-resetting suppressor.",
                                                                pending, since_write_ms
                                                            );
                                                            s.soft_reset();
                                                            suppressor_soft_reset_pending = true;
                                                        }
                                                        update_backend_diagnostics(
                                                            &noise_backend_available,
                                                            &noise_backend_failed,
                                                            noise_backend_error.as_ref(),
                                                            &s,
                                                        );
                                                        last_suppressor_recovery = Instant::now();
                                                    }
                                                }
                                            }
                                        }
                                        if available > 0 {
                                            suppressor_soft_reset_pending = false;
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
                                                let now = Instant::now();
                                                match non_finite_window_started_at {
                                                    Some(start)
                                                        if now.duration_since(start).as_millis()
                                                            as u64
                                                            <= NON_FINITE_REBUILD_WINDOW_MS => {}
                                                    _ => {
                                                        non_finite_window_started_at = Some(now);
                                                        non_finite_window_count = 0;
                                                    }
                                                }
                                                non_finite_window_count =
                                                    non_finite_window_count.saturating_add(1);
                                                if non_finite_window_count
                                                    >= NON_FINITE_REBUILD_THRESHOLD
                                                {
                                                    s.soft_reset();
                                                    non_finite_window_started_at = None;
                                                    non_finite_window_count = 0;
                                                }
                                                update_backend_diagnostics(
                                                    &noise_backend_available,
                                                    &noise_backend_failed,
                                                    noise_backend_error.as_ref(),
                                                    &s,
                                                );
                                            }

                                            apply_downstream_chain_rt!(output_slice);

                                            // Measure OUTPUT levels
                                            measure_levels(
                                                output_slice,
                                                &mut output_rms_acc,
                                                &output_peak,
                                                &output_rms,
                                            );

                                            // Send processed samples to output
                                            write_output(output_slice, false);
                                        }

                                        // Record DSP processing time
                                        let raw_dsp_us = dsp_start.elapsed().as_micros() as u64;
                                        let prev_smoothed =
                                            dsp_time_smoothed_us.load(Ordering::Relaxed);
                                        let smoothed = smooth_dsp_time(raw_dsp_us, prev_smoothed);
                                        dsp_time_us.store(raw_dsp_us, Ordering::Relaxed);
                                        dsp_time_smoothed_us.store(smoothed, Ordering::Relaxed);
                                    } else {
                                        suppressor_buffer_len.store(0, Ordering::Relaxed);
                                        suppressor_latency_samples.store(0, Ordering::Relaxed);
                                        smoothed_buffer_len.store(0, Ordering::Relaxed);

                                        apply_downstream_chain_rt!(buffer);

                                        measure_levels(
                                            buffer,
                                            &mut output_rms_acc,
                                            &output_peak,
                                            &output_rms,
                                        );

                                        write_output(buffer, false);

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
                                    smoothed_buffer_len.store(0, Ordering::Relaxed);
                                    if suppressor_reset_requested.swap(false, Ordering::AcqRel) {
                                        if let Some(mut s) =
                                            lock_rt(suppressor.as_ref(), &lock_contention_count)
                                        {
                                            s.soft_reset();
                                            update_backend_diagnostics(
                                                &noise_backend_available,
                                                &noise_backend_failed,
                                                noise_backend_error.as_ref(),
                                                &s,
                                            );
                                        }
                                    }
                                    // Suppressor disabled: apply remaining stages directly

                                    apply_downstream_chain_rt!(buffer);

                                    // Measure OUTPUT levels
                                    measure_levels(
                                        buffer,
                                        &mut output_rms_acc,
                                        &output_peak,
                                        &output_rms,
                                    );

                                    // Send to output
                                    write_output(buffer, false);
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

                                let suppressor_latency_samples =
                                    if suppressor_enabled.load(Ordering::Acquire) {
                                        suppressor_latency_samples.load(Ordering::Relaxed) as u64
                                    } else {
                                        0
                                    };
                                let total_latency = total_reported_latency_us(
                                    output_buffer_samples,
                                    output_sample_rate_for_latency,
                                    suppressor_latency_samples,
                                    limiter_lookahead_samples.load(Ordering::Relaxed),
                                    limiter_enabled.load(Ordering::Acquire),
                                    sample_rate_for_latency,
                                    latency_compensation_us.load(Ordering::Relaxed),
                                );
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
        #[cfg(feature = "vad")]
        self.stop_vad_worker();

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
        self.input_resampler_active.store(false, Ordering::Relaxed);
        self.output_resampler_active.store(false, Ordering::Relaxed);
        if let Ok(mut consumer_guard) = self.raw_recording_consumer.lock() {
            *consumer_guard = None;
        }
        if let Ok(mut pending) = self.pending_suppressor.lock() {
            *pending = None;
        }

        // Reinitialize suppressor state so stop/start can recover from poisoned model state.
        if let Ok(mut s) = self.suppressor.lock() {
            let was_enabled = s.is_enabled();
            let model = s.model_type();
            *s = NoiseSuppressionEngine::new(model, Arc::clone(&self.suppressor_strength));
            s.set_enabled(was_enabled);
            update_backend_diagnostics(
                &self.noise_backend_available,
                &self.noise_backend_failed,
                self.noise_backend_error.as_ref(),
                &s,
            );
        }

        // Reset DSP state so stop/start can recover from stuck envelopes.
        if let Ok(mut g) = self.gate.lock() {
            g.reset();
            if let Ok(control) = self.gate_control.lock() {
                apply_gate_control(&mut g, &control);
            }
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

    /// Get active input device name for the running stream.
    pub fn active_input_device_name(&self) -> Option<String> {
        if self.is_running() {
            self.input_device_name.clone()
        } else {
            None
        }
    }

    /// Get active output device name for the running stream.
    pub fn active_output_device_name(&self) -> Option<String> {
        if self.is_running() {
            self.output_device_name.clone()
        } else {
            None
        }
    }

    /// Set master bypass
    pub fn set_bypass(&self, bypass: bool) {
        self.bypass.store(bypass, Ordering::SeqCst);
    }

    /// Get bypass state
    pub fn is_bypass(&self) -> bool {
        self.bypass.load(Ordering::SeqCst)
    }

    /// Enable/disable true raw monitor path.
    pub fn set_raw_monitor_enabled(&self, enabled: bool) {
        self.raw_monitor_enabled.store(enabled, Ordering::Release);
    }

    /// Get true raw monitor path state.
    pub fn is_raw_monitor_enabled(&self) -> bool {
        self.raw_monitor_enabled.load(Ordering::Acquire)
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    // === Noise Gate Controls ===

    /// Enable/disable noise gate
    pub fn set_gate_enabled(&self, enabled: bool) {
        self.gate_enabled.store(enabled, Ordering::Release);
        if let Ok(mut control) = self.gate_control.lock() {
            control.enabled = enabled;
        }
        self.gate_dirty.store(true, Ordering::Release);
    }

    /// Set noise gate threshold
    pub fn set_gate_threshold(&self, threshold_db: f64) {
        let Some(threshold_db) =
            clamp_control_value(threshold_db, GATE_THRESHOLD_MIN_DB, GATE_THRESHOLD_MAX_DB)
        else {
            return;
        };
        if let Ok(mut control) = self.gate_control.lock() {
            control.threshold_db = threshold_db;
        }
        self.gate_dirty.store(true, Ordering::Release);
    }

    /// Set noise gate attack time
    pub fn set_gate_attack(&self, attack_ms: f64) {
        let Some(attack_ms) =
            clamp_control_value(attack_ms, GATE_ATTACK_MIN_MS, GATE_ATTACK_MAX_MS)
        else {
            return;
        };
        if let Ok(mut control) = self.gate_control.lock() {
            control.attack_ms = attack_ms;
        }
        self.gate_dirty.store(true, Ordering::Release);
    }

    /// Set noise gate release time
    pub fn set_gate_release(&self, release_ms: f64) {
        let Some(release_ms) =
            clamp_control_value(release_ms, GATE_RELEASE_MIN_MS, GATE_RELEASE_MAX_MS)
        else {
            return;
        };
        if let Ok(mut control) = self.gate_control.lock() {
            control.release_ms = release_ms;
        }
        self.gate_dirty.store(true, Ordering::Release);
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
        if let Ok(mut control) = self.gate_control.lock() {
            control.gate_mode = gate_mode;
        }
        self.gate_dirty.store(true, Ordering::Release);
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
        self.vad_available.load(Ordering::Relaxed)
    }

    #[cfg(feature = "vad")]
    /// Set VAD probability threshold (0.0-1.0)
    pub fn set_vad_threshold(&self, threshold: f32) {
        let Some(threshold) =
            clamp_control_value_f32(threshold, VAD_THRESHOLD_MIN, VAD_THRESHOLD_MAX)
        else {
            return;
        };
        if let Ok(mut control) = self.gate_control.lock() {
            control.vad_threshold = threshold;
        }
        self.gate_dirty.store(true, Ordering::Release);
    }

    #[cfg(feature = "vad")]
    /// Set VAD hold time in milliseconds
    pub fn set_vad_hold_time(&self, hold_ms: f32) {
        let Some(hold_ms) = clamp_control_value_f32(hold_ms, VAD_HOLD_MIN_MS, VAD_HOLD_MAX_MS)
        else {
            return;
        };
        if let Ok(mut control) = self.gate_control.lock() {
            control.hold_ms = hold_ms;
        }
        self.gate_dirty.store(true, Ordering::Release);
    }

    #[cfg(feature = "vad")]
    /// Set VAD pre-gain to boost weak signals for better speech detection
    /// Default is 1.0 (no gain). Values > 1.0 boost the signal.
    /// This helps with quiet microphones where VAD can't detect speech.
    pub fn set_vad_pre_gain(&self, gain: f32) {
        let Some(gain) = clamp_control_value_f32(gain, VAD_PRE_GAIN_MIN, VAD_PRE_GAIN_MAX) else {
            return;
        };
        if let Ok(mut control) = self.gate_control.lock() {
            control.pre_gain = gain;
        }
        self.gate_dirty.store(true, Ordering::Release);
    }

    #[cfg(feature = "vad")]
    /// Get current VAD pre-gain
    pub fn vad_pre_gain(&self) -> f32 {
        self.gate_control
            .lock()
            .map(|control| control.pre_gain)
            .unwrap_or(1.0)
    }

    #[cfg(feature = "vad")]
    /// Enable/disable auto-threshold mode (automatically adjusts gate threshold based on noise floor)
    pub fn set_auto_threshold(&self, enabled: bool) {
        if let Ok(mut control) = self.gate_control.lock() {
            control.auto_threshold = enabled;
        }
        self.gate_dirty.store(true, Ordering::Release);
    }

    #[cfg(feature = "vad")]
    /// Set margin above noise floor for auto-threshold (in dB)
    pub fn set_gate_margin(&self, margin_db: f32) {
        let Some(margin_db) =
            clamp_control_value_f32(margin_db, GATE_MARGIN_MIN_DB, GATE_MARGIN_MAX_DB)
        else {
            return;
        };
        if let Ok(mut control) = self.gate_control.lock() {
            control.margin_db = margin_db;
        }
        self.gate_dirty.store(true, Ordering::Release);
    }

    #[cfg(feature = "vad")]
    /// Get current noise floor estimate (in dB)
    pub fn get_noise_floor(&self) -> f32 {
        f32::from_bits(self.gate_noise_floor_db.load(Ordering::Relaxed))
    }

    #[cfg(feature = "vad")]
    /// Get current gate margin (in dB)
    pub fn gate_margin(&self) -> f32 {
        self.gate_control
            .lock()
            .map(|control| control.margin_db)
            .unwrap_or(10.0)
    }

    #[cfg(feature = "vad")]
    /// Check if auto-threshold is enabled
    pub fn auto_threshold_enabled(&self) -> bool {
        self.gate_control
            .lock()
            .map(|control| control.auto_threshold)
            .unwrap_or(false)
    }

    // === Noise Suppression Controls ===

    /// Enable/disable noise suppression
    pub fn set_rnnoise_enabled(&self, enabled: bool) {
        self.suppressor_enabled.store(enabled, Ordering::Release);
        if let Ok(mut control) = self.suppressor_control.lock() {
            control.enabled = enabled;
        }
        if !enabled {
            self.suppressor_reset_requested
                .store(true, Ordering::Release);
        }
        self.suppressor_dirty.store(true, Ordering::Release);
    }

    /// Check if noise suppression is enabled
    pub fn is_rnnoise_enabled(&self) -> bool {
        self.suppressor_enabled.load(Ordering::Acquire)
    }

    /// Set noise suppression wet/dry mix strength (0.0 = fully dry, 1.0 = fully wet)
    pub fn set_rnnoise_strength(&self, strength: f32) {
        let Some(clamped) =
            clamp_control_value_f32(strength, RNNOISE_STRENGTH_MIN, RNNOISE_STRENGTH_MAX)
        else {
            return;
        };
        let bits = clamped.to_bits();
        self.suppressor_strength.store(bits, Ordering::Relaxed);
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
                update_backend_diagnostics(
                    &self.noise_backend_available,
                    &self.noise_backend_failed,
                    self.noise_backend_error.as_ref(),
                    &new_engine,
                );
                return false;
            }
        }

        update_backend_diagnostics(
            &self.noise_backend_available,
            &self.noise_backend_failed,
            self.noise_backend_error.as_ref(),
            &new_engine,
        );

        if let Ok(mut pending) = self.pending_suppressor.lock() {
            *pending = Some(new_engine);
        } else {
            return false;
        }

        if let Ok(mut control) = self.suppressor_control.lock() {
            control.model = model;
        }
        self.current_model.store(model as u8, Ordering::Release);
        self.suppressor_reset_requested
            .store(true, Ordering::Release);
        self.suppressor_dirty.store(true, Ordering::Release);
        true
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
        self.eq_control.set_enabled(enabled);
        self.eq_dirty.store(true, Ordering::Release);
    }

    /// Check if EQ is enabled
    pub fn is_eq_enabled(&self) -> bool {
        self.eq_enabled.load(Ordering::Acquire)
    }

    /// Set EQ band gain
    pub fn set_eq_band_gain(&self, band: usize, gain_db: f64) -> Result<(), String> {
        self.validate_eq_band_index(band)?;
        self.validate_eq_gain(band, gain_db)?;
        if let Ok(mut e) = self.eq.lock() {
            e.set_band_gain(band, gain_db);
        }
        self.eq_control.set_band_gain(band, gain_db);
        self.eq_dirty.store(true, Ordering::Release);
        Ok(())
    }

    /// Set EQ band frequency
    pub fn set_eq_band_frequency(&self, band: usize, frequency: f64) -> Result<(), String> {
        self.validate_eq_band_index(band)?;
        self.validate_eq_frequency(band, frequency)?;
        if let Ok(mut e) = self.eq.lock() {
            e.set_band_frequency(band, frequency);
        }
        self.eq_control.set_band_frequency(band, frequency);
        self.eq_dirty.store(true, Ordering::Release);
        Ok(())
    }

    /// Set EQ band Q
    pub fn set_eq_band_q(&self, band: usize, q: f64) -> Result<(), String> {
        self.validate_eq_band_index(band)?;
        self.validate_eq_q(band, q)?;
        if let Ok(mut e) = self.eq.lock() {
            e.set_band_q(band, q);
        }
        self.eq_control.set_band_q(band, q);
        self.eq_dirty.store(true, Ordering::Release);
        Ok(())
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
    /// * bands.len() must equal NUM_BANDS
    /// * frequency: 20.0 Hz to Nyquist minus a small margin
    /// * gain_db: -12.0 to +12.0 dB
    /// * q: 0.1 to 10.0
    pub fn apply_eq_settings(&self, bands: Vec<(f64, f64, f64)>) -> PyResult<()> {
        // Validate band count
        if bands.len() != NUM_BANDS {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected {} bands, got {}",
                NUM_BANDS,
                bands.len()
            )));
        }

        // Validate each band's parameters
        for (i, (freq, gain, q)) in bands.iter().enumerate() {
            self.validate_eq_frequency(i, *freq)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
            self.validate_eq_gain(i, *gain)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
            self.validate_eq_q(i, *q)
                .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        }

        // All validation passed - apply atomically
        if let Ok(mut eq) = self.eq.lock() {
            for (i, (freq, gain, q)) in bands.iter().enumerate() {
                eq.set_band_frequency(i, *freq);
                eq.set_band_gain(i, *gain);
                eq.set_band_q(i, *q);
            }
        }
        let snapshot_bands = std::array::from_fn(|index| bands[index]);
        self.eq_control.set_bands(&snapshot_bands);
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
        let Some(mut hz) = clamp_control_value(hz, DEESSER_LOW_CUT_MIN_HZ, DEESSER_LOW_CUT_MAX_HZ)
        else {
            return;
        };
        let high_cut_hz = self
            .deesser_control
            .lock()
            .map(|control| control.high_cut_hz)
            .unwrap_or(9000.0);
        if high_cut_hz <= hz + DEESSER_MIN_BANDWIDTH_HZ {
            hz = (high_cut_hz - DEESSER_MIN_BANDWIDTH_HZ)
                .clamp(DEESSER_LOW_CUT_MIN_HZ, DEESSER_LOW_CUT_MAX_HZ);
        }
        if let Ok(mut d) = self.deesser.lock() {
            d.set_low_cut_hz(hz);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.low_cut_hz = hz;
        }
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn set_deesser_high_cut_hz(&self, hz: f64) {
        let Some(mut hz) =
            clamp_control_value(hz, DEESSER_HIGH_CUT_MIN_HZ, DEESSER_HIGH_CUT_MAX_HZ)
        else {
            return;
        };
        let low_cut_hz = self
            .deesser_control
            .lock()
            .map(|control| control.low_cut_hz)
            .unwrap_or(4000.0);
        if hz <= low_cut_hz + DEESSER_MIN_BANDWIDTH_HZ {
            hz = (low_cut_hz + DEESSER_MIN_BANDWIDTH_HZ)
                .clamp(DEESSER_HIGH_CUT_MIN_HZ, DEESSER_HIGH_CUT_MAX_HZ);
        }
        if let Ok(mut d) = self.deesser.lock() {
            d.set_high_cut_hz(hz);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.high_cut_hz = hz;
        }
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn set_deesser_threshold_db(&self, threshold_db: f64) {
        let Some(threshold_db) = clamp_control_value(
            threshold_db,
            DEESSER_THRESHOLD_MIN_DB,
            DEESSER_THRESHOLD_MAX_DB,
        ) else {
            return;
        };
        if let Ok(mut d) = self.deesser.lock() {
            d.set_threshold_db(threshold_db);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.threshold_db = threshold_db;
        }
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn set_deesser_ratio(&self, ratio: f64) {
        let Some(ratio) = clamp_control_value(ratio, DEESSER_RATIO_MIN, DEESSER_RATIO_MAX) else {
            return;
        };
        if let Ok(mut d) = self.deesser.lock() {
            d.set_ratio(ratio);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.ratio = ratio;
        }
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn set_deesser_attack_ms(&self, attack_ms: f64) {
        let Some(attack_ms) =
            clamp_control_value(attack_ms, DEESSER_ATTACK_MIN_MS, DEESSER_ATTACK_MAX_MS)
        else {
            return;
        };
        if let Ok(mut d) = self.deesser.lock() {
            d.set_attack_ms(attack_ms);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.attack_ms = attack_ms;
        }
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn set_deesser_release_ms(&self, release_ms: f64) {
        let Some(release_ms) =
            clamp_control_value(release_ms, DEESSER_RELEASE_MIN_MS, DEESSER_RELEASE_MAX_MS)
        else {
            return;
        };
        if let Ok(mut d) = self.deesser.lock() {
            d.set_release_ms(release_ms);
        }
        if let Ok(mut control) = self.deesser_control.lock() {
            control.release_ms = release_ms;
        }
        self.deesser_dirty.store(true, Ordering::Release);
    }

    pub fn set_deesser_max_reduction_db(&self, max_reduction_db: f64) {
        let Some(max_reduction_db) = clamp_control_value(
            max_reduction_db,
            DEESSER_MAX_REDUCTION_MIN_DB,
            DEESSER_MAX_REDUCTION_MAX_DB,
        ) else {
            return;
        };
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
        let Some(amount) =
            clamp_control_value(amount, DEESSER_AUTO_AMOUNT_MIN, DEESSER_AUTO_AMOUNT_MAX)
        else {
            return;
        };
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
        let Some(threshold_db) = clamp_control_value(
            threshold_db,
            COMPRESSOR_THRESHOLD_MIN_DB,
            COMPRESSOR_THRESHOLD_MAX_DB,
        ) else {
            return;
        };
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
        let Some(ratio) = clamp_control_value(ratio, COMPRESSOR_RATIO_MIN, COMPRESSOR_RATIO_MAX)
        else {
            return;
        };
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
        let Some(attack_ms) = clamp_control_value(
            attack_ms,
            COMPRESSOR_ATTACK_MIN_MS,
            COMPRESSOR_ATTACK_MAX_MS,
        ) else {
            return;
        };
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
        let Some(release_ms) = clamp_control_value(
            release_ms,
            COMPRESSOR_RELEASE_MIN_MS,
            COMPRESSOR_RELEASE_MAX_MS,
        ) else {
            return;
        };
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
        let Some(makeup_gain_db) = clamp_control_value(
            makeup_gain_db,
            COMPRESSOR_MAKEUP_MIN_DB,
            COMPRESSOR_MAKEUP_MAX_DB,
        ) else {
            return;
        };
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
        let Some(release_ms) = clamp_control_value(
            release_ms,
            COMPRESSOR_BASE_RELEASE_MIN_MS,
            COMPRESSOR_BASE_RELEASE_MAX_MS,
        ) else {
            return;
        };
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
        let Some(ceiling_db) =
            clamp_control_value(ceiling_db, LIMITER_CEILING_MIN_DB, LIMITER_CEILING_MAX_DB)
        else {
            return;
        };
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
        let Some(release_ms) =
            clamp_control_value(release_ms, LIMITER_RELEASE_MIN_MS, LIMITER_RELEASE_MAX_MS)
        else {
            return;
        };
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
        let Some(target_lufs) = clamp_control_value(
            target_lufs,
            COMPRESSOR_TARGET_LUFS_MIN,
            COMPRESSOR_TARGET_LUFS_MAX,
        ) else {
            return;
        };
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
        self.noise_backend_available.load(Ordering::Relaxed)
    }

    /// Whether the suppressor backend has failed after startup.
    pub fn noise_backend_failed(&self) -> bool {
        self.noise_backend_failed.load(Ordering::Relaxed)
    }

    /// Last suppressor backend error, if any.
    pub fn noise_backend_error(&self) -> Option<String> {
        self.noise_backend_error
            .lock()
            .ok()
            .and_then(|error| error.clone())
    }

    /// Suppress watchdog-driven stream recovery during intrusive UI workflows.
    pub fn set_recovery_suppressed(&self, suppressed: bool) {
        self.recovery_suppressed
            .store(suppressed, Ordering::Release);
    }

    /// Whether watchdog-driven stream recovery is currently suppressed.
    pub fn is_recovery_suppressed(&self) -> bool {
        self.recovery_suppressed.load(Ordering::Acquire)
    }

    // === RAW AUDIO RECORDING (for calibration) ===

    /// Start recording raw audio for calibration
    /// Taps audio AFTER pre-filter (DC blocker + 80Hz HP) but BEFORE noise gate
    pub fn start_raw_recording(&mut self, duration_secs: f64) -> Result<(), String> {
        if !duration_secs.is_finite() || duration_secs <= 0.0 {
            return Err("Recording duration must be a finite positive value".to_string());
        }

        let num_samples_f = duration_secs * self.sample_rate as f64;
        if !num_samples_f.is_finite() || num_samples_f < 1.0 {
            return Err("Recording duration is too short".to_string());
        }

        let num_samples = num_samples_f.round() as usize;
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
            self.raw_recording_target
                .store(num_samples, Ordering::Release);
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
    use pyo3::types::PyDict;
    use pyo3::Python;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn test_duration_samples_for_44k1_output() {
        assert_eq!(duration_samples(44_100, OUTPUT_PRIME_MS), 882);
        assert_eq!(duration_samples(44_100, OUTPUT_TARGET_HIGH_MS), 1323);
        assert_eq!(duration_samples(44_100, OUTPUT_HARD_BACKLOG_MS), 2646);
    }

    #[test]
    fn test_samples_to_micros_uses_device_rate() {
        assert_eq!(samples_to_micros(441, 44_100), 10_000);
        assert_eq!(samples_to_micros(480, 48_000), 10_000);
    }

    #[test]
    fn test_total_reported_latency_includes_limiter_lookahead_across_sample_rates() {
        for (sample_rate, expected_us) in
            [(44_100_u32, 1_995_u64), (48_000, 2_000), (96_000, 2_000)]
        {
            let limiter = Limiter::default_settings(sample_rate as f64);
            let lookahead_samples = limiter.lookahead_samples() as u64;
            let total = total_reported_latency_us(
                0,
                sample_rate,
                0,
                lookahead_samples,
                true,
                sample_rate,
                0,
            );
            assert_eq!(total, expected_us);
        }
    }

    #[test]
    fn test_total_reported_latency_respects_output_vs_processing_rates() {
        let total = total_reported_latency_us(
            882,    // 20ms @ 44.1kHz output buffer
            44_100, // output sample rate
            480,    // 10ms suppressor latency @ 48kHz
            96,     // 2ms limiter lookahead @ 48kHz
            true, 48_000, // processing sample rate
            500,    // fixed compensation
        );
        assert_eq!(total, 20_000 + 10_000 + 2_000 + 500);
    }

    #[test]
    fn test_build_sinc_resampler_for_valid_rates() {
        let resampler = build_sinc_resampler(44_100, 48_000, 1024);
        assert!(resampler.is_ok());
    }

    #[test]
    fn test_active_device_names_only_report_when_running() {
        let mut processor = AudioProcessor::new();
        processor.input_device_name = Some("Mic A".to_string());
        processor.output_device_name = Some("Out B".to_string());

        assert_eq!(processor.active_input_device_name(), None);
        assert_eq!(processor.active_output_device_name(), None);

        processor.running.store(true, Ordering::SeqCst);
        assert_eq!(
            processor.active_input_device_name().as_deref(),
            Some("Mic A")
        );
        assert_eq!(
            processor.active_output_device_name().as_deref(),
            Some("Out B")
        );
    }

    #[test]
    fn test_raw_monitor_toggle_round_trip() {
        let processor = AudioProcessor::new();
        assert!(!processor.is_raw_monitor_enabled());
        processor.set_raw_monitor_enabled(true);
        assert!(processor.is_raw_monitor_enabled());
        processor.set_raw_monitor_enabled(false);
        assert!(!processor.is_raw_monitor_enabled());
    }

    fn install_raw_recording_consumer(processor: &AudioProcessor) {
        let rb = crate::audio::AudioRingBuffer::new(processor.sample_rate as usize);
        let (_producer, consumer) = rb.split();
        *processor.raw_recording_consumer.lock().unwrap() = Some(consumer);
    }

    #[test]
    fn test_raw_recording_rejects_invalid_durations_without_activating() {
        for duration in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut processor = AudioProcessor::new();
            install_raw_recording_consumer(&processor);

            let err = processor.start_raw_recording(duration).unwrap_err();

            assert!(err.contains("finite positive"));
            assert!(!processor.recording_active.load(Ordering::Acquire));
            assert_eq!(processor.raw_recording_target.load(Ordering::Acquire), 0);
        }
    }

    #[test]
    fn test_raw_recording_rejects_too_short_duration_without_activating() {
        let mut processor = AudioProcessor::new();
        install_raw_recording_consumer(&processor);

        let err = processor.start_raw_recording(0.25 / processor.sample_rate as f64);

        assert_eq!(err.unwrap_err(), "Recording duration is too short");
        assert!(!processor.recording_active.load(Ordering::Acquire));
        assert_eq!(processor.raw_recording_target.load(Ordering::Acquire), 0);
    }

    #[test]
    fn test_raw_recording_accepts_valid_duration_with_nonzero_target() {
        let mut processor = AudioProcessor::new();
        install_raw_recording_consumer(&processor);

        processor.start_raw_recording(0.01).unwrap();

        assert!(processor.recording_active.load(Ordering::Acquire));
        assert!(processor.raw_recording_target.load(Ordering::Acquire) > 0);
    }

    #[test]
    fn test_pending_suppressor_swap_keeps_current_when_model_already_matches() {
        let strength = Arc::new(AtomicU32::new(1.0_f32.to_bits()));
        let mut current = NoiseSuppressionEngine::new(NoiseModel::RNNoise, Arc::clone(&strength));
        let mut pending = Some(NoiseSuppressionEngine::new(NoiseModel::RNNoise, strength));
        let control = SuppressorControlState {
            enabled: true,
            model: NoiseModel::RNNoise,
        };

        assert!(swap_pending_suppressor_if_ready(
            &mut current,
            &control,
            &mut pending
        ));
        assert_eq!(current.model_type(), NoiseModel::RNNoise);
        assert!(pending.is_some());
    }

    #[cfg(feature = "deepfilter")]
    #[test]
    fn test_pending_suppressor_swap_without_candidate_leaves_current_backend() {
        let strength = Arc::new(AtomicU32::new(1.0_f32.to_bits()));
        let mut current = NoiseSuppressionEngine::new(NoiseModel::RNNoise, strength);
        let mut pending = None;
        let control = SuppressorControlState {
            enabled: true,
            model: NoiseModel::DeepFilterNetLL,
        };

        assert!(!swap_pending_suppressor_if_ready(
            &mut current,
            &control,
            &mut pending
        ));
        assert_eq!(current.model_type(), NoiseModel::RNNoise);
    }

    #[test]
    fn test_raw_monitor_path_selection_and_clean_write_mode() {
        let raw = select_processing_path(true, false);
        let bypass = select_processing_path(false, true);
        let full = select_processing_path(false, false);

        assert_eq!(raw, ProcessingPath::RawMonitor);
        assert_eq!(bypass, ProcessingPath::Bypass);
        assert_eq!(full, ProcessingPath::Full);

        assert!(uses_clean_write_path(raw));
        assert!(!uses_clean_write_path(bypass));
        assert!(!uses_clean_write_path(full));
    }

    #[test]
    fn test_raw_monitor_sanitizes_but_skips_prefilter_shaping() {
        let mut non_finite = vec![f32::NAN, f32::INFINITY, -f32::INFINITY];
        sanitize_non_finite_inplace(&mut non_finite);
        assert_eq!(non_finite, vec![0.0, 0.0, 0.0]);

        let mut raw_buffer = vec![0.25_f32; 256];
        let mut normal_buffer = raw_buffer.clone();

        sanitize_non_finite_inplace(&mut raw_buffer);
        let mut pre_filter_state = InputPreFilterState::default();
        let mut pre_filter = Biquad::new(
            BiquadType::HighPass,
            INPUT_PREFILTER_HZ,
            0.0,
            INPUT_PREFILTER_Q,
            48_000.0,
        );
        apply_input_pre_filter(&mut normal_buffer, &mut pre_filter_state, &mut pre_filter);

        let max_abs_diff = normal_buffer
            .iter()
            .zip(raw_buffer.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_abs_diff > 0.05);
    }

    #[test]
    fn test_final_output_sanitizer_rejects_non_finite_and_limits_ceiling() {
        let mut buffer = vec![f32::NAN, f32::INFINITY, -f32::INFINITY, 0.75, -0.75];
        sanitize_and_clamp_output_inplace(&mut buffer, 0.5);
        assert_eq!(buffer, vec![0.0, 0.0, 0.0, 0.5, -0.5]);
    }

    #[test]
    fn test_retime_audio_block_can_expand_and_compress() {
        let input = [0.0_f32, 0.25, 0.5, 0.75, 1.0, 0.5];
        let mut scratch = Vec::new();

        let expanded = retime_audio_block(&input, 0.5, 32, &mut scratch);
        assert!(expanded.len() > input.len());

        let compressed = retime_audio_block(&input, 2.0, 32, &mut scratch);
        assert!(compressed.len() < input.len());
    }

    #[test]
    fn test_retime_audio_block_linear_interpolation_does_not_overshoot_neighbors() {
        let input = [0.0_f32, 0.5, 1.0, 0.5, 0.0];
        let mut scratch = Vec::new();

        let expanded = retime_audio_block(&input, 0.7, 32, &mut scratch);

        for sample in expanded {
            assert!((*sample >= 0.0) && (*sample <= 1.0));
        }
    }

    #[test]
    fn test_eq_single_band_validation_rejects_invalid_values_and_preserves_state() {
        let processor = AudioProcessor::new();
        let original = processor.get_eq_band_params(0).unwrap();

        assert!(processor.set_eq_band_gain(0, f64::NAN).is_err());
        assert!(processor.set_eq_band_frequency(0, 50_000.0).is_err());
        assert!(processor.set_eq_band_q(0, 0.01).is_err());
        assert_eq!(processor.get_eq_band_params(0).unwrap(), original);
    }

    #[test]
    fn test_gate_control_validation_rejects_non_finite_and_clamps_ranges() {
        let processor = AudioProcessor::new();
        processor.set_gate_threshold(f64::NAN);
        processor.set_gate_attack(f64::INFINITY);
        processor.set_gate_release(f64::NEG_INFINITY);

        {
            let control = processor.gate_control.lock().unwrap();
            assert_eq!(control.threshold_db, -40.0);
            assert_eq!(control.attack_ms, 10.0);
            assert_eq!(control.release_ms, 100.0);
        }

        processor.set_gate_threshold(-120.0);
        processor.set_gate_attack(0.01);
        processor.set_gate_release(5000.0);

        let control = processor.gate_control.lock().unwrap();
        assert_eq!(control.threshold_db, GATE_THRESHOLD_MIN_DB);
        assert_eq!(control.attack_ms, GATE_ATTACK_MIN_MS);
        assert_eq!(control.release_ms, GATE_RELEASE_MAX_MS);
    }

    #[test]
    fn test_rnnoise_strength_validation_rejects_non_finite_and_clamps_ranges() {
        let processor = AudioProcessor::new();
        let original = processor.get_rnnoise_strength();

        processor.set_rnnoise_strength(f32::NAN);
        assert_eq!(processor.get_rnnoise_strength(), original);

        processor.set_rnnoise_strength(2.0);
        assert_eq!(processor.get_rnnoise_strength(), RNNOISE_STRENGTH_MAX);

        processor.set_rnnoise_strength(-1.0);
        assert_eq!(processor.get_rnnoise_strength(), RNNOISE_STRENGTH_MIN);
    }

    #[test]
    fn test_deesser_control_validation_rejects_non_finite_and_clamps_ranges() {
        let processor = AudioProcessor::new();

        processor.set_deesser_low_cut_hz(f64::NAN);
        processor.set_deesser_high_cut_hz(f64::INFINITY);
        processor.set_deesser_threshold_db(f64::NEG_INFINITY);
        processor.set_deesser_ratio(f64::NAN);
        processor.set_deesser_attack_ms(f64::NAN);
        processor.set_deesser_release_ms(f64::NAN);
        processor.set_deesser_max_reduction_db(f64::NAN);
        processor.set_deesser_auto_amount(f64::NAN);

        assert_eq!(processor.get_deesser_low_cut_hz(), 4000.0);
        assert_eq!(processor.get_deesser_high_cut_hz(), 9000.0);
        assert_eq!(processor.get_deesser_threshold_db(), -28.0);
        assert_eq!(processor.get_deesser_ratio(), 4.0);
        assert_eq!(processor.get_deesser_max_reduction_db(), 6.0);
        assert_eq!(processor.get_deesser_auto_amount(), 0.5);

        processor.set_deesser_low_cut_hz(100.0);
        processor.set_deesser_high_cut_hz(80_000.0);
        processor.set_deesser_threshold_db(-100.0);
        processor.set_deesser_ratio(100.0);
        processor.set_deesser_attack_ms(0.01);
        processor.set_deesser_release_ms(10_000.0);
        processor.set_deesser_max_reduction_db(100.0);
        processor.set_deesser_auto_amount(2.0);

        assert_eq!(processor.get_deesser_low_cut_hz(), DEESSER_LOW_CUT_MIN_HZ);
        assert_eq!(processor.get_deesser_high_cut_hz(), DEESSER_HIGH_CUT_MAX_HZ);
        assert_eq!(
            processor.get_deesser_threshold_db(),
            DEESSER_THRESHOLD_MIN_DB
        );
        assert_eq!(processor.get_deesser_ratio(), DEESSER_RATIO_MAX);
        assert_eq!(
            processor.get_deesser_max_reduction_db(),
            DEESSER_MAX_REDUCTION_MAX_DB
        );
        assert_eq!(processor.get_deesser_auto_amount(), DEESSER_AUTO_AMOUNT_MAX);
    }

    #[test]
    fn test_apply_eq_settings_rejects_above_nyquist() {
        let processor = AudioProcessor::new();
        let mut bands = vec![(100.0, 0.0, 1.0); NUM_BANDS];
        bands[NUM_BANDS - 1] = (processor.eq_nyquist_limit_hz() + 100.0, 0.0, 1.0);

        pyo3::prepare_freethreaded_python();
        let err = processor.apply_eq_settings(bands).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn test_lock_rt_counts_contention() {
        let mutex = Arc::new(Mutex::new(1_u32));
        let contention = Arc::new(AtomicU64::new(0));
        let (started_tx, started_rx) = mpsc::channel();
        let mutex_for_thread = Arc::clone(&mutex);

        let holder = std::thread::spawn(move || {
            let _guard = mutex_for_thread.lock().unwrap();
            started_tx.send(()).unwrap();
            std::thread::sleep(Duration::from_millis(30));
        });

        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let guard = lock_rt(mutex.as_ref(), contention.as_ref());
        assert!(guard.is_none());
        assert_eq!(contention.load(Ordering::Relaxed), 1);
        holder.join().unwrap();
    }

    #[test]
    fn test_release_ms_to_tenth_ms_rounds_expected_values() {
        assert_eq!(release_ms_to_tenth_ms(200.0), 2000);
        assert_eq!(release_ms_to_tenth_ms(12.34), 123);
        assert_eq!(release_ms_to_tenth_ms(12.35), 124);
        assert_eq!(release_ms_to_tenth_ms(-5.0), 0);
        assert_eq!(release_ms_to_tenth_ms(f64::NAN), 0);
    }

    #[test]
    fn test_smoothing_coeff_for_time_constant_matches_100ms_meter_target() {
        let coeff = smoothing_coeff_for_time_constant(48_000.0, 100.0);
        assert!((coeff - 0.999_791_7).abs() < 1e-6);

        let invalid = smoothing_coeff_for_time_constant(0.0, 100.0);
        assert_eq!(invalid, 0.0);
    }

    #[test]
    fn test_runtime_diagnostics_include_output_recovery_count() {
        let wrapper = PyAudioProcessor {
            processor: AudioProcessor::new(),
        };
        wrapper
            .processor
            .output_recovery_count
            .store(7, Ordering::Relaxed);

        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let diagnostics = wrapper.get_runtime_diagnostics(py).unwrap();
            let diagnostics = diagnostics.bind(py);
            let diagnostics = diagnostics.downcast::<PyDict>().unwrap();
            assert_eq!(
                diagnostics
                    .get_item("output_recovery_count")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                7
            );
        });
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

    /// Get active input device name for the running stream.
    fn get_active_input_device(&self) -> Option<String> {
        self.processor.active_input_device_name()
    }

    /// Get active output device name for the running stream.
    fn get_active_output_device(&self) -> Option<String> {
        self.processor.active_output_device_name()
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

    fn set_raw_monitor_enabled(&self, enabled: bool) {
        self.processor.set_raw_monitor_enabled(enabled);
    }

    fn is_raw_monitor_enabled(&self) -> bool {
        self.processor.is_raw_monitor_enabled()
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

    fn set_eq_band_gain(&self, band: usize, gain_db: f64) -> PyResult<()> {
        self.processor
            .set_eq_band_gain(band, gain_db)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    fn set_eq_band_frequency(&self, band: usize, frequency: f64) -> PyResult<()> {
        self.processor
            .set_eq_band_frequency(band, frequency)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    fn set_eq_band_q(&self, band: usize, q: f64) -> PyResult<()> {
        self.processor
            .set_eq_band_q(band, q)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
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
        diagnostics.set_item(
            "noise_backend_failed",
            self.processor.noise_backend_failed(),
        )?;
        diagnostics.set_item("noise_backend_error", self.processor.noise_backend_error())?;
        diagnostics.set_item(
            "input_dropped_samples",
            self.processor.get_dropped_samples(),
        )?;
        diagnostics.set_item(
            "input_backlog_recovery_count",
            self.processor
                .input_backlog_recovery_count
                .load(Ordering::Relaxed),
        )?;
        diagnostics.set_item(
            "input_backlog_dropped_samples",
            self.processor
                .input_backlog_dropped_samples
                .load(Ordering::Relaxed),
        )?;
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
            "output_recovery_count",
            self.processor.get_output_recovery_count(),
        )?;
        diagnostics.set_item(
            "output_short_write_dropped_samples",
            self.processor
                .output_short_write_dropped_samples
                .load(Ordering::Relaxed),
        )?;
        diagnostics.set_item(
            "stream_restart_count",
            self.processor.get_stream_restart_count(),
        )?;
        diagnostics.set_item(
            "last_restart_reason",
            self.processor.get_last_restart_reason(),
        )?;
        diagnostics.set_item("last_stream_error", self.processor.get_last_stream_error())?;
        diagnostics.set_item(
            "suppressor_non_finite_count",
            self.processor.get_suppressor_non_finite_count(),
        )?;
        diagnostics.set_item(
            "clip_event_count",
            self.processor.clip_event_count.load(Ordering::Relaxed),
        )?;
        diagnostics.set_item(
            "clip_peak_db",
            f32::from_bits(self.processor.clip_peak_db.load(Ordering::Relaxed)),
        )?;
        diagnostics.set_item(
            "input_resampler_active",
            self.processor
                .input_resampler_active
                .load(Ordering::Relaxed),
        )?;
        diagnostics.set_item(
            "output_resampler_active",
            self.processor
                .output_resampler_active
                .load(Ordering::Relaxed),
        )?;
        diagnostics.set_item("output_sample_rate", self.processor.output_sample_rate())?;
        diagnostics.set_item(
            "recovery_suppressed",
            self.processor.is_recovery_suppressed(),
        )?;
        diagnostics.set_item(
            "raw_monitor_enabled",
            self.processor.is_raw_monitor_enabled(),
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
