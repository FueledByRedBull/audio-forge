//! Unified audio processor - DSP chain runs entirely in Rust
//!
//! Processing chain: Mic Input → Noise Gate → RNNoise → 10-Band EQ → Output
//!
//! Adapted from Spectral Workbench project for MicEq.

use pyo3::prelude::*;
use rubato::{FftFixedIn, Resampler};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use thread_priority::{set_current_thread_priority, ThreadPriority};

use std::sync::atomic::AtomicU8;

use super::buffer::{AudioProducer, AudioRingBuffer};
use super::input::{AudioInput, TARGET_SAMPLE_RATE};
use super::output::AudioOutput;
use crate::dsp::biquad::{Biquad, BiquadType};
use crate::dsp::noise_suppressor::{NoiseModel, NoiseSuppressionEngine, NoiseSuppressor};
use crate::dsp::{Compressor, DeEsser, Limiter, NoiseGate, ParametricEQ};

#[cfg(feature = "vad")]
use crate::dsp::vad::{GateMode, VadAutoGate};

/// Main audio processor combining all DSP stages
pub struct AudioProcessor {
    /// Pre-filter (High-pass at 80Hz to remove fan rumble before RNNoise)
    pre_filter: Arc<Mutex<Biquad>>,

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

    /// Compressor
    compressor: Arc<Mutex<Compressor>>,
    compressor_enabled: Arc<AtomicBool>,

    /// De-esser
    deesser: Arc<Mutex<DeEsser>>,
    deesser_enabled: Arc<AtomicBool>,

    /// Hard limiter (brick-wall ceiling)
    limiter: Arc<Mutex<Limiter>>,
    limiter_enabled: Arc<AtomicBool>,

    /// Audio input stream
    audio_input: Option<AudioInput>,

    /// Audio output stream
    audio_output: Option<AudioOutput>,

    /// Output ring buffer producer (for sending processed audio to output)
    output_producer: Arc<Mutex<Option<AudioProducer>>>,

    /// Processing thread handle
    process_thread: Option<std::thread::JoinHandle<()>>,

    /// Running flag
    running: Arc<AtomicBool>,

    /// Master bypass flag
    bypass: Arc<AtomicBool>,

    /// Sample rate
    sample_rate: u32,

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

    // === RAW AUDIO RECORDING (for calibration) ===
    /// Raw audio recording buffer (for calibration - captures audio AFTER pre-filter, BEFORE gate)
    raw_recording_buffer: Arc<Mutex<Option<Vec<f32>>>>,
    /// Current recording position (samples recorded so far)
    raw_recording_pos: Arc<AtomicUsize>,
    /// Target recording length (total samples to record)
    raw_recording_target: Arc<AtomicUsize>,
    /// Flag indicating recording is active (used to mute output to prevent user from hearing themselves)
    recording_active: Arc<AtomicBool>,
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

        // Create a HighPass filter at 80Hz to kill fan rumble before RNNoise
        // Q = 0.707 (Butterworth) for smooth frequency response
        let pre_filter = Arc::new(Mutex::new(Biquad::new(
            BiquadType::HighPass,
            80.0,  // Frequency (Hz) - cuts everything below this
            0.0,   // Gain (unused for HighPass)
            0.707, // Q Factor (Butterworth)
            sample_rate as f64,
        )));

        // Create strength Arc BEFORE noise suppressor (share reference)
        let suppressor_strength = Arc::new(AtomicU32::new(1.0_f32.to_bits()));

        // Create noise suppression engine (default to RNNoise)
        let suppressor =
            NoiseSuppressionEngine::new(NoiseModel::RNNoise, suppressor_strength.clone());

        Self {
            pre_filter,
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
            compressor: Arc::new(Mutex::new(Compressor::default_voice(sample_rate as f64))),
            compressor_enabled: Arc::new(AtomicBool::new(true)),
            deesser: Arc::new(Mutex::new(DeEsser::new(sample_rate as f64))),
            deesser_enabled: Arc::new(AtomicBool::new(false)),
            limiter: Arc::new(Mutex::new(Limiter::default_settings(sample_rate as f64))),
            limiter_enabled: Arc::new(AtomicBool::new(true)),
            audio_input: None,
            audio_output: None,
            output_producer: Arc::new(Mutex::new(None)),
            process_thread: None,
            running: Arc::new(AtomicBool::new(false)),
            bypass: Arc::new(AtomicBool::new(false)),
            sample_rate,
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

            // Initialize raw recording buffer
            raw_recording_buffer: Arc::new(Mutex::new(None)),
            raw_recording_pos: Arc::new(AtomicUsize::new(0)),
            raw_recording_target: Arc::new(AtomicUsize::new(0)),
            recording_active: Arc::new(AtomicBool::new(false)),
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
        if self.running.load(Ordering::SeqCst) {
            return Err("Already running".to_string());
        }

        // Create input ring buffer (2 seconds capacity)
        let input_rb = AudioRingBuffer::new(self.sample_rate as usize * 2);
        let (input_producer, input_consumer) = input_rb.split();

        // Capture the dropped counter before producer is moved
        let input_dropped_counter = input_producer.dropped_counter();
        self.input_dropped = input_dropped_counter;

        // Reset the dropped counter at start
        self.input_dropped.store(0, Ordering::Relaxed);
        self.output_underrun_streak.store(0, Ordering::Relaxed);
        self.output_underrun_total.store(0, Ordering::Relaxed);
        self.jitter_dropped_samples.store(0, Ordering::Relaxed);
        self.output_recovery_count.store(0, Ordering::Relaxed);
        self.last_input_callback_time_us.store(0, Ordering::Relaxed);
        self.last_output_callback_time_us.store(0, Ordering::Relaxed);

        // Create output ring buffer
        let output_rb = AudioRingBuffer::new(self.sample_rate as usize * 2);
        let (output_producer, output_consumer) = output_rb.split();

        // PRIME THE BUFFER: Write minimal silence to prevent initial underrun/scratch
        // 480 samples = 10ms at 48kHz (one RNNoise frame worth)
        let mut prod = output_producer;
        let silence = vec![0.0f32; 480]; // 10ms at 48kHz - minimal priming
        prod.write(&silence);
        let output_producer = prod;

        // Store output producer for processing thread
        if let Ok(mut prod) = self.output_producer.lock() {
            *prod = Some(output_producer);
        }

        // Start audio input
        let last_input_callback_time_us = Arc::clone(&self.last_input_callback_time_us);
        let input = match input_device {
            Some(name) => AudioInput::from_device_name(
                name,
                input_producer,
                last_input_callback_time_us,
            ),
            None => AudioInput::from_default_device(input_producer, last_input_callback_time_us),
        }
        .map_err(|e| format!("Failed to start audio input: {}", e))?;

        let input_device_name = input.device_info().name.clone();
        let input_sample_rate_for_thread = input.device_info().sample_rate;

        if let Err(e) = input.start() {
            if let Ok(mut prod) = self.output_producer.lock() {
                *prod = None;
            }
            return Err(format!("Failed to start input stream: {}", e));
        }

        // Start audio output
        let recording_active = Arc::clone(&self.recording_active);
        let last_output_callback_time_us = Arc::clone(&self.last_output_callback_time_us);
        let output_underrun_streak = Arc::clone(&self.output_underrun_streak);
        let output_underrun_total = Arc::clone(&self.output_underrun_total);
        let output_result = match output_device {
            Some(name) => AudioOutput::from_device_name(
                name,
                output_consumer,
                recording_active,
                last_output_callback_time_us,
                output_underrun_streak,
                output_underrun_total,
            ),
            None => AudioOutput::from_default_device(
                output_consumer,
                recording_active,
                last_output_callback_time_us,
                output_underrun_streak,
                output_underrun_total,
            ),
        };
        let output = match output_result {
            Ok(output) => output,
            Err(e) => {
                let _ = input.pause();
                if let Ok(mut prod) = self.output_producer.lock() {
                    *prod = None;
                }
                return Err(format!("Failed to start audio output: {}", e));
            }
        };

        let output_device_name = output.device_info().name.clone();

        if let Err(e) = output.start() {
            let _ = input.pause();
            let _ = output.pause();
            if let Ok(mut prod) = self.output_producer.lock() {
                *prod = None;
            }
            return Err(format!("Failed to start output stream: {}", e));
        }

        self.input_device_name = Some(input_device_name.clone());
        self.output_device_name = Some(output_device_name.clone());
        self.audio_input = Some(input);
        self.audio_output = Some(output);

        // Start processing thread
        self.running.store(true, Ordering::SeqCst);

        let gate = Arc::clone(&self.gate);
        let gate_enabled = Arc::clone(&self.gate_enabled);
        let suppressor = Arc::clone(&self.suppressor);
        let suppressor_enabled = Arc::clone(&self.suppressor_enabled);
        let eq = Arc::clone(&self.eq);
        let eq_enabled = Arc::clone(&self.eq_enabled);
        let compressor = Arc::clone(&self.compressor);
        let compressor_enabled = Arc::clone(&self.compressor_enabled);
        let deesser = Arc::clone(&self.deesser);
        let deesser_enabled = Arc::clone(&self.deesser_enabled);
        let limiter = Arc::clone(&self.limiter);
        let limiter_enabled = Arc::clone(&self.limiter_enabled);
        let output_producer = Arc::clone(&self.output_producer);
        let running = Arc::clone(&self.running);
        let bypass = Arc::clone(&self.bypass);
        let pre_filter = Arc::clone(&self.pre_filter);

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

        // Clone DSP performance metric atomics
        let dsp_time_us = Arc::clone(&self.dsp_time_us);
        let input_buffer_len = Arc::clone(&self.input_buffer_len);
        let smoothed_input_buffer_len = Arc::clone(&self.smoothed_input_buffer_len);
        let output_buffer_len = Arc::clone(&self.output_buffer_len);
        let suppressor_buffer_len = Arc::clone(&self.suppressor_buffer_len);
        let last_output_write_time = Arc::clone(&self.last_output_write_time);
        let dsp_time_smoothed_us = Arc::clone(&self.dsp_time_smoothed_us);
        let smoothed_buffer_len = Arc::clone(&self.smoothed_buffer_len);
        let lock_contention_count = Arc::clone(&self.lock_contention_count);
        let jitter_dropped_samples = Arc::clone(&self.jitter_dropped_samples);
        let output_recovery_count = Arc::clone(&self.output_recovery_count);

        // Clone raw recording buffer atomics
        let raw_recording_buffer = Arc::clone(&self.raw_recording_buffer);
        let raw_recording_pos = Arc::clone(&self.raw_recording_pos);
        let raw_recording_target = Arc::clone(&self.raw_recording_target);

        let handle = std::thread::spawn(move || {
            // Set high thread priority for real-time audio processing
            if let Err(e) = set_current_thread_priority(ThreadPriority::Max) {
                eprintln!("Warning: Could not set audio thread priority: {:?}", e);
            }

            let mut consumer = input_consumer;
            let mut input_buffer = vec![0.0f32; 2048];
            let mut temp_buffer = vec![0.0f32; 4096];
            let mut rnnoise_output = vec![0.0f32; 2048];
            let mut resample_input: Vec<f64> = Vec::with_capacity(4096);
            let mut resampler = if input_sample_rate_for_thread != sample_rate_for_latency {
                eprintln!(
                    "[PROCESSING] Input device sample rate {} Hz; resampling to {} Hz in DSP thread",
                    input_sample_rate_for_thread, sample_rate_for_latency
                );
                match FftFixedIn::<f64>::new(
                    input_sample_rate_for_thread as usize,
                    sample_rate_for_latency as usize,
                    1024,
                    2,
                    1,
                ) {
                    Ok(r) => Some(r),
                    Err(e) => {
                        eprintln!(
                            "[PROCESSING] WARNING: Failed to init resampler ({}). Processing raw input rate.",
                            e
                        );
                        None
                    }
                }
            } else {
                None
            };

            // DC Blocker state variables
            // This removes electrical DC offset which causes static/clicks
            let mut dc_x1: f32 = 0.0;
            let mut dc_y1: f32 = 0.0;
            const DC_COEFF: f32 = 0.995; // DC blocking coefficient

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
                if let Ok(now) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                {
                    last_output_write_time.store(now.as_micros() as u64, Ordering::Relaxed);
                }
            };

            // Jitter-buffer write control: keep output queue in a healthy range.
            const OUTPUT_TARGET_LOW_SAMPLES: usize = 480; // ~10ms @ 48k
            const OUTPUT_TARGET_HIGH_SAMPLES: usize = 960; // ~20ms @ 48k
            const OUTPUT_HARD_BACKLOG_SAMPLES: usize = 1920; // ~40ms backlog
            let write_with_jitter_control = |samples: &[f32]| {
                if let Ok(mut prod_guard) = output_producer.try_lock() {
                    if let Some(prod) = prod_guard.as_mut() {
                        let capacity = prod.capacity();
                        let free = prod.free_len();
                        let fill = capacity.saturating_sub(free);

                        // If backlog is too large, drop part/all of this block to reduce latency drift.
                        let mut start_idx = 0usize;
                        if fill >= OUTPUT_HARD_BACKLOG_SAMPLES {
                            jitter_dropped_samples
                                .fetch_add(samples.len() as u64, Ordering::Relaxed);
                            output_recovery_count.fetch_add(1, Ordering::Relaxed);
                            output_buffer_len.store(fill as u32, Ordering::Relaxed);
                            return;
                        } else if fill > OUTPUT_TARGET_HIGH_SAMPLES {
                            let drop = (fill - OUTPUT_TARGET_HIGH_SAMPLES).min(samples.len());
                            if drop > 0 {
                                start_idx = drop;
                                jitter_dropped_samples.fetch_add(drop as u64, Ordering::Relaxed);
                            }
                        }

                        let to_write = &samples[start_idx..];
                        if !to_write.is_empty() {
                            let written = prod.write(to_write);
                            if written > 0 {
                                update_write_time();
                            }
                        }

                        let new_fill = prod.capacity().saturating_sub(prod.free_len());
                        let _ = OUTPUT_TARGET_LOW_SAMPLES; // Documented target band lower bound.
                        output_buffer_len.store(new_fill as u32, Ordering::Relaxed);
                    }
                } else {
                    lock_contention_count.fetch_add(1, Ordering::Relaxed);
                }
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
                                    resample_input.push(sample as f64);
                                }

                                let mut produced = 0usize;
                                let input_frames_needed = resampler.input_frames_next();
                                while resample_input.len() >= input_frames_needed
                                    && produced < temp_buffer.len()
                                {
                                    let input_chunk: Vec<f64> =
                                        resample_input.drain(..input_frames_needed).collect();
                                    if let Ok(output) = resampler.process(&[input_chunk], None) {
                                        if !output.is_empty() && !output[0].is_empty() {
                                            for &sample in output[0].iter() {
                                                if produced >= temp_buffer.len() {
                                                    break;
                                                }
                                                temp_buffer[produced] = sample as f32;
                                                produced += 1;
                                            }
                                        }
                                    }
                                }
                                produced
                            } else {
                                let copy_n = n_raw.min(temp_buffer.len());
                                temp_buffer[..copy_n].copy_from_slice(&input_buffer[..copy_n]);
                                copy_n
                            };

                            if n == 0 {
                                std::thread::sleep(std::time::Duration::from_micros(100));
                                continue;
                            }

                            let buffer = &mut temp_buffer[..n];

                            // Start DSP timing
                            let dsp_start = Instant::now();

                            // Clamp any out-of-range samples to prevent distortion
                            for sample in buffer.iter_mut() {
                                if sample.is_nan() || sample.is_infinite() {
                                    *sample = 0.0;
                                } else if *sample > 1.0 {
                                    *sample = 1.0;
                                } else if *sample < -1.0 {
                                    *sample = -1.0;
                                }
                            }

                            // === PRE-PROCESSING: Clean the input before DSP chain ===
                            // This removes fan rumble and DC offset that cause RNNoise artifacts
                            if let Ok(mut pf) = pre_filter.try_lock() {
                                for sample in buffer.iter_mut() {
                                    // A. DC Blocker (removes electrical DC offset causing static/clicks)
                                    let input = *sample;
                                    let output = input - dc_x1 + DC_COEFF * dc_y1;
                                    dc_x1 = input;
                                    dc_y1 = output;
                                    *sample = output;

                                    // B. High-Pass Filter at 80Hz (kills fan rumble before RNNoise)
                                    *sample = pf.process_sample(*sample);
                                }
                            } else {
                                lock_contention_count.fetch_add(1, Ordering::Relaxed);
                            }
                            // === END PRE-PROCESSING ===

                            // Measure INPUT levels (after pre-filter, before main processing)
                            measure_levels(buffer, &mut input_rms_acc, &input_peak, &input_rms);

                            // === RAW AUDIO RECORDING TAP (for calibration) ===
                            // Capture audio AFTER pre-filter, BEFORE noise gate
                            // This is the raw microphone response needed for EQ analysis
                            {
                                if let Ok(mut buf_guard) = raw_recording_buffer.try_lock() {
                                    if let Some(ref mut buffer_rec) = *buf_guard {
                                        let target = raw_recording_target.load(Ordering::Acquire);
                                        let mut pos = raw_recording_pos.load(Ordering::Acquire);

                                        if pos < target {
                                            // Calculate how many samples we can copy
                                            let remaining = target - pos;
                                            let to_copy = n.min(remaining);

                                            // Copy samples to recording buffer
                                            if pos + to_copy <= buffer_rec.len() {
                                                buffer_rec[pos..pos + to_copy]
                                                    .copy_from_slice(&buffer[..to_copy]);
                                                pos += to_copy;
                                                raw_recording_pos.store(pos, Ordering::Release);
                                            }
                                        }
                                    }
                                } else {
                                    lock_contention_count.fetch_add(1, Ordering::Relaxed);
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
                                compressor_current_release_ms.store(2000, Ordering::Relaxed); // Default 200ms
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
                                        if gain < 0.9999 {
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
                                        let raw_buffer = suppressor_buffered as u32;
                                        let prev_smoothed =
                                            smoothed_buffer_len.load(Ordering::Relaxed);
                                        let smoothed = smooth_buffer(raw_buffer, prev_smoothed);
                                        suppressor_buffer_len.store(raw_buffer, Ordering::Relaxed);
                                        smoothed_buffer_len.store(smoothed, Ordering::Relaxed);

                                        let available = s.available_samples();
                                        if available > 0 {
                                            let count = available.min(rnnoise_output.len());
                                            let processed = s.pop_samples(count);
                                            let output_slice =
                                                &mut rnnoise_output[..processed.len()];
                                            output_slice.copy_from_slice(&processed);

                                            // Stage 3: De-esser
                                            if deesser_enabled.load(Ordering::Acquire) {
                                                if let Ok(mut d) = deesser.try_lock() {
                                                    d.process_block_inplace(output_slice);
                                                    deesser_gain_reduction.store(
                                                        d.current_gain_reduction_db().to_bits(),
                                                        Ordering::Relaxed,
                                                    );
                                                } else {
                                                    lock_contention_count
                                                        .fetch_add(1, Ordering::Relaxed);
                                                }
                                            } else {
                                                deesser_gain_reduction
                                                    .store(0.0_f32.to_bits(), Ordering::Relaxed);
                                            }

                                            // Stage 4: EQ
                                            if eq_enabled.load(Ordering::Acquire) {
                                                if let Ok(mut e) = eq.try_lock() {
                                                    e.process_block_inplace(output_slice);
                                                } else {
                                                    lock_contention_count
                                                        .fetch_add(1, Ordering::Relaxed);
                                                }
                                            }

                                            // Stage 5: Compressor
                                            if compressor_enabled.load(Ordering::Acquire) {
                                                if let Ok(mut c) = compressor.try_lock() {
                                                    c.process_block_inplace(output_slice);
                                                    // Store gain reduction for metering
                                                    compressor_gain_reduction.store(
                                                        (c.current_gain_reduction() as f32)
                                                            .to_bits(),
                                                        Ordering::Relaxed,
                                                    );
                                                    // Store current release time for metering
                                                    let current_release = c.current_release_time();
                                                    // Convert to u64 (0.1ms resolution = multiply by 10)
                                                    compressor_current_release_ms.store(
                                                        (current_release * 10.0) as u64,
                                                        Ordering::Relaxed,
                                                    );
                                                } else {
                                                    lock_contention_count
                                                        .fetch_add(1, Ordering::Relaxed);
                                                }
                                            } else {
                                                compressor_gain_reduction
                                                    .store(0.0_f32.to_bits(), Ordering::Relaxed);
                                                compressor_current_release_ms
                                                    .store(2000, Ordering::Relaxed);
                                                // Default 200ms
                                            }

                                            // Stage 6: Hard Limiter (LAST - safety ceiling)
                                            if limiter_enabled.load(Ordering::Acquire) {
                                                if let Ok(mut l) = limiter.try_lock() {
                                                    l.process_block_inplace(output_slice);
                                                } else {
                                                    lock_contention_count
                                                        .fetch_add(1, Ordering::Relaxed);
                                                }
                                            }

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

                                        if deesser_enabled.load(Ordering::Acquire) {
                                            if let Ok(mut d) = deesser.try_lock() {
                                                d.process_block_inplace(buffer);
                                                deesser_gain_reduction.store(
                                                    d.current_gain_reduction_db().to_bits(),
                                                    Ordering::Relaxed,
                                                );
                                            } else {
                                                lock_contention_count
                                                    .fetch_add(1, Ordering::Relaxed);
                                            }
                                        } else {
                                            deesser_gain_reduction
                                                .store(0.0_f32.to_bits(), Ordering::Relaxed);
                                        }

                                        if eq_enabled.load(Ordering::Acquire) {
                                            if let Ok(mut e) = eq.try_lock() {
                                                e.process_block_inplace(buffer);
                                            } else {
                                                lock_contention_count
                                                    .fetch_add(1, Ordering::Relaxed);
                                            }
                                        }

                                        if compressor_enabled.load(Ordering::Acquire) {
                                            if let Ok(mut c) = compressor.try_lock() {
                                                c.process_block_inplace(buffer);
                                                compressor_gain_reduction.store(
                                                    (c.current_gain_reduction() as f32).to_bits(),
                                                    Ordering::Relaxed,
                                                );
                                                let current_release = c.current_release_time();
                                                compressor_current_release_ms.store(
                                                    (current_release * 10.0) as u64,
                                                    Ordering::Relaxed,
                                                );
                                            } else {
                                                lock_contention_count
                                                    .fetch_add(1, Ordering::Relaxed);
                                            }
                                        } else {
                                            compressor_gain_reduction
                                                .store(0.0_f32.to_bits(), Ordering::Relaxed);
                                            compressor_current_release_ms
                                                .store(2000, Ordering::Relaxed);
                                        }

                                        if limiter_enabled.load(Ordering::Acquire) {
                                            if let Ok(mut l) = limiter.try_lock() {
                                                l.process_block_inplace(buffer);
                                            } else {
                                                lock_contention_count
                                                    .fetch_add(1, Ordering::Relaxed);
                                            }
                                        }

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
                                    // Suppressor disabled: apply remaining stages directly

                                    // Stage 3: De-esser
                                    if deesser_enabled.load(Ordering::Acquire) {
                                        if let Ok(mut d) = deesser.try_lock() {
                                            d.process_block_inplace(buffer);
                                            deesser_gain_reduction.store(
                                                d.current_gain_reduction_db().to_bits(),
                                                Ordering::Relaxed,
                                            );
                                        } else {
                                            lock_contention_count.fetch_add(1, Ordering::Relaxed);
                                        }
                                    } else {
                                        deesser_gain_reduction
                                            .store(0.0_f32.to_bits(), Ordering::Relaxed);
                                    }

                                    // Stage 4: EQ
                                    if eq_enabled.load(Ordering::Acquire) {
                                        if let Ok(mut e) = eq.try_lock() {
                                            e.process_block_inplace(buffer);
                                        } else {
                                            lock_contention_count.fetch_add(1, Ordering::Relaxed);
                                        }
                                    }

                                    // Stage 5: Compressor
                                    if compressor_enabled.load(Ordering::Acquire) {
                                        if let Ok(mut c) = compressor.try_lock() {
                                            c.process_block_inplace(buffer);
                                            compressor_gain_reduction.store(
                                                (c.current_gain_reduction() as f32).to_bits(),
                                                Ordering::Relaxed,
                                            );
                                            // Store current release time for metering
                                            let current_release = c.current_release_time();
                                            // Convert to u64 (0.1ms resolution = multiply by 10)
                                            compressor_current_release_ms.store(
                                                (current_release * 10.0) as u64,
                                                Ordering::Relaxed,
                                            );
                                        } else {
                                            lock_contention_count.fetch_add(1, Ordering::Relaxed);
                                        }
                                    } else {
                                        compressor_gain_reduction
                                            .store(0.0_f32.to_bits(), Ordering::Relaxed);
                                        compressor_current_release_ms
                                            .store(2000, Ordering::Relaxed); // Default 200ms
                                    }

                                    // Stage 6: Hard Limiter (LAST - safety ceiling)
                                    if limiter_enabled.load(Ordering::Acquire) {
                                        if let Ok(mut l) = limiter.try_lock() {
                                            l.process_block_inplace(buffer);
                                        } else {
                                            lock_contention_count.fetch_add(1, Ordering::Relaxed);
                                        }
                                    }

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

                                // Calculate total processing latency in samples:
                                // 1. Output buffer priming: 480 samples (10ms)
                                let output_buffer_samples: u64 = 480;

                                // 2. Noise suppressor frame buffering (when enabled)
                                let suppressor_latency_samples =
                                    if suppressor_enabled.load(Ordering::Acquire) {
                                        // Suppressor needs a full frame before processing
                                        // Plus any pending samples waiting for the next frame
                                        if let Ok(s) = suppressor.try_lock() {
                                            s.latency_samples() as u64 + s.pending_input() as u64
                                        } else {
                                            lock_contention_count.fetch_add(1, Ordering::Relaxed);
                                            480 // Default assumption
                                        }
                                    } else {
                                        0
                                    };

                                // Total latency in samples
                                let total_samples =
                                    output_buffer_samples + suppressor_latency_samples;

                                // Convert to microseconds: samples / sample_rate * 1_000_000
                                let latency_microseconds =
                                    (total_samples * 1_000_000) / sample_rate_for_latency as u64;

                                let compensation_us =
                                    latency_compensation_us.load(Ordering::Relaxed);
                                let total_latency =
                                    latency_microseconds.saturating_add(compensation_us);
                                latency_us.store(total_latency, Ordering::Relaxed);
                            }
                        } else {
                            // No data available, sleep briefly to avoid busy-wait
                            if last_heartbeat.elapsed() >= HEARTBEAT_INTERVAL {
                                last_heartbeat = Instant::now();

                                // Check for output stall
                                let last_write = last_output_write_time.load(Ordering::Relaxed);
                                let now = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .map(|dur| dur.as_micros() as u64)
                                    .unwrap_or(last_write);
                                let time_since_write_ms = now.saturating_sub(last_write) / 1000;

                                if last_write > 0 && time_since_write_ms > STALL_THRESHOLD_MS {
                                    eprintln!(
                                        "[PROCESSING] WARNING: Output stall detected! No write for {} ms (input_buf={}, suppressor_enabled={})",
                                        time_since_write_ms,
                                        raw_input_len,
                                        suppressor_enabled.load(Ordering::Acquire)
                                    );
                                }
                            }
                            std::thread::sleep(std::time::Duration::from_micros(100));
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

        if let Ok(mut prod) = self.output_producer.lock() {
            *prod = None;
        }

        // Soft reset suppressor to clear stale buffers without model convergence penalty
        if let Ok(mut s) = self.suppressor.lock() {
            s.soft_reset();
        }
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
    }

    /// Set EQ band frequency
    pub fn set_eq_band_frequency(&self, band: usize, frequency: f64) {
        if let Ok(mut e) = self.eq.lock() {
            e.set_band_frequency(band, frequency);
        }
    }

    /// Set EQ band Q
    pub fn set_eq_band_q(&self, band: usize, q: f64) {
        if let Ok(mut e) = self.eq.lock() {
            e.set_band_q(band, q);
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

        Ok(())
    }

    // === De-Esser Controls ===

    /// Enable/disable de-esser.
    pub fn set_deesser_enabled(&self, enabled: bool) {
        self.deesser_enabled.store(enabled, Ordering::Release);
        if let Ok(mut d) = self.deesser.lock() {
            d.set_enabled(enabled);
        }
    }

    /// Check if de-esser is enabled.
    pub fn is_deesser_enabled(&self) -> bool {
        self.deesser_enabled.load(Ordering::Acquire)
    }

    pub fn set_deesser_low_cut_hz(&self, hz: f64) {
        if let Ok(mut d) = self.deesser.lock() {
            d.set_low_cut_hz(hz);
        }
    }

    pub fn set_deesser_high_cut_hz(&self, hz: f64) {
        if let Ok(mut d) = self.deesser.lock() {
            d.set_high_cut_hz(hz);
        }
    }

    pub fn set_deesser_threshold_db(&self, threshold_db: f64) {
        if let Ok(mut d) = self.deesser.lock() {
            d.set_threshold_db(threshold_db);
        }
    }

    pub fn set_deesser_ratio(&self, ratio: f64) {
        if let Ok(mut d) = self.deesser.lock() {
            d.set_ratio(ratio);
        }
    }

    pub fn set_deesser_attack_ms(&self, attack_ms: f64) {
        if let Ok(mut d) = self.deesser.lock() {
            d.set_attack_ms(attack_ms);
        }
    }

    pub fn set_deesser_release_ms(&self, release_ms: f64) {
        if let Ok(mut d) = self.deesser.lock() {
            d.set_release_ms(release_ms);
        }
    }

    pub fn set_deesser_max_reduction_db(&self, max_reduction_db: f64) {
        if let Ok(mut d) = self.deesser.lock() {
            d.set_max_reduction_db(max_reduction_db);
        }
    }

    pub fn set_deesser_auto_enabled(&self, auto_enabled: bool) {
        if let Ok(mut d) = self.deesser.lock() {
            d.set_auto_enabled(auto_enabled);
        }
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
    }

    /// Set compressor ratio
    pub fn set_compressor_ratio(&self, ratio: f64) {
        if let Ok(mut c) = self.compressor.lock() {
            c.set_ratio(ratio);
        }
    }

    /// Set compressor attack time in ms
    pub fn set_compressor_attack(&self, attack_ms: f64) {
        if let Ok(mut c) = self.compressor.lock() {
            c.set_attack_time(attack_ms);
        }
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
    }

    // === Limiter Controls ===

    /// Enable/disable limiter
    pub fn set_limiter_enabled(&self, enabled: bool) {
        self.limiter_enabled.store(enabled, Ordering::Release);
        if let Ok(mut l) = self.limiter.lock() {
            l.set_enabled(enabled);
        }
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
    }

    /// Set limiter release time in ms
    pub fn set_limiter_release(&self, release_ms: f64) {
        if let Ok(mut l) = self.limiter.lock() {
            l.set_release_time(release_ms);
        }
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

    /// Age of last input callback heartbeat in milliseconds.
    pub fn get_input_callback_age_ms(&self) -> u64 {
        let last = self.last_input_callback_time_us.load(Ordering::Relaxed);
        if last == 0 {
            return u64::MAX;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|dur| dur.as_micros() as u64)
            .unwrap_or(last);
        now.saturating_sub(last) / 1000
    }

    /// Age of last output callback heartbeat in milliseconds.
    pub fn get_output_callback_age_ms(&self) -> u64 {
        let last = self.last_output_callback_time_us.load(Ordering::Relaxed);
        if last == 0 {
            return u64::MAX;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|dur| dur.as_micros() as u64)
            .unwrap_or(last);
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

    // === RAW AUDIO RECORDING (for calibration) ===

    /// Start recording raw audio for calibration
    /// Taps audio AFTER pre-filter (DC blocker + 80Hz HP) but BEFORE noise gate
    pub fn start_raw_recording(&mut self, duration_secs: f64) -> Result<(), String> {
        let num_samples = (duration_secs * self.sample_rate as f64) as usize;

        // Allocate recording buffer
        let buffer = vec![0.0f32; num_samples];

        // Lock and set recording state
        if let Ok(mut buf_guard) = self.raw_recording_buffer.lock() {
            *buf_guard = Some(buffer);
            self.raw_recording_pos.store(0, Ordering::Release);
            self.raw_recording_target
                .store(num_samples, Ordering::Release);
            self.recording_active.store(true, Ordering::Release); // Mute output during recording
            Ok(())
        } else {
            Err("Failed to lock recording buffer".to_string())
        }
    }

    /// Stop recording and return captured audio (truncated to actual length)
    pub fn stop_raw_recording(&mut self) -> Option<Vec<f32>> {
        if let Ok(mut buf_guard) = self.raw_recording_buffer.lock() {
            if let Some(mut buffer) = buf_guard.take() {
                let pos = self.raw_recording_pos.load(Ordering::Acquire);
                buffer.truncate(pos);
                self.recording_active.store(false, Ordering::Release); // Unmute output after recording
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
        if let Ok(buf_guard) = self.raw_recording_buffer.lock() {
            if let Some(ref buffer) = *buf_guard {
                let pos = self.raw_recording_pos.load(Ordering::Acquire);
                if pos == 0 {
                    return -120.0; // -infinity
                }

                // Calculate RMS from recorded samples so far
                let len = pos.min(buffer.len());
                let slice = &buffer[..len];

                // RMS calculation
                let sum_sq: f32 = slice.iter().map(|&x| x * x).sum();
                let rms = (sum_sq / len as f32).sqrt();

                // Convert to dB (with floor at -120 to prevent log(0))
                if rms > 1e-6 {
                    20.0 * rms.log10()
                } else {
                    -120.0
                }
            } else {
                -120.0
            }
        } else {
            -120.0
        }
    }

    /// Manually set output mute state (useful for calibration workflow)
    pub fn set_output_mute(&self, muted: bool) {
        self.recording_active.store(muted, Ordering::Release);
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
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
    fn set_vad_threshold(&self, threshold: f32) -> PyResult<()> {
        self.processor.set_vad_threshold(threshold);
        Ok(())
    }

    /// Set VAD hold time in milliseconds
    #[cfg(feature = "vad")]
    fn set_vad_hold_time(&self, hold_ms: f32) -> PyResult<()> {
        self.processor.set_vad_hold_time(hold_ms);
        Ok(())
    }

    /// Set VAD pre-gain to boost weak signals for better speech detection
    /// Default is 1.0 (no gain). Values > 1.0 boost the signal.
    /// This helps with quiet microphones where VAD can't detect speech.
    #[cfg(feature = "vad")]
    fn set_vad_pre_gain(&self, gain: f32) -> PyResult<()> {
        self.processor.set_vad_pre_gain(gain);
        Ok(())
    }

    /// Enable/disable auto-threshold mode (automatically adjusts gate threshold based on noise floor)
    #[cfg(feature = "vad")]
    fn set_auto_threshold(&self, enabled: bool) -> PyResult<()> {
        self.processor.set_auto_threshold(enabled);
        Ok(())
    }

    /// Set margin above noise floor for auto-threshold (in dB)
    #[cfg(feature = "vad")]
    fn set_gate_margin(&self, margin_db: f32) -> PyResult<()> {
        self.processor.set_gate_margin(margin_db);
        Ok(())
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
    fn vad_pre_gain(&self) -> PyResult<f32> {
        Ok(self.processor.vad_pre_gain())
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
        self.processor.set_deesser_max_reduction_db(max_reduction_db);
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
        if let Ok(mut comp) = self.processor.compressor.lock() {
            comp.set_adaptive_release(enabled);
        }
    }

    /// Get compressor adaptive release mode
    fn get_compressor_adaptive_release(&self) -> bool {
        if let Ok(comp) = self.processor.compressor.lock() {
            comp.adaptive_release()
        } else {
            false
        }
    }

    /// Set compressor base release time (milliseconds)
    fn set_compressor_base_release(&self, release_ms: f64) {
        if let Ok(mut comp) = self.processor.compressor.lock() {
            comp.set_base_release_time(release_ms);
        }
    }

    /// Get compressor base release time (milliseconds)
    fn get_compressor_base_release(&self) -> f64 {
        if let Ok(comp) = self.processor.compressor.lock() {
            comp.base_release_ms()
        } else {
            200.0
        }
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

    // === RAW AUDIO RECORDING (for calibration) ===

    /// Start recording raw audio for calibration (10 seconds @ 48kHz)
    fn start_raw_recording(&mut self, duration_secs: f64) -> PyResult<()> {
        self.processor
            .start_raw_recording(duration_secs)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
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
    fn is_recording_complete(&mut self) -> PyResult<bool> {
        Ok(self.processor.is_recording_complete())
    }

    /// Get recording progress (0.0 to 1.0)
    fn recording_progress(&mut self) -> PyResult<f32> {
        Ok(self.processor.recording_progress())
    }

    /// Get current recording level as RMS in dB (for level meter visualization)
    fn recording_level_db(&mut self) -> PyResult<f32> {
        Ok(self.processor.recording_level_db())
    }

    /// Manually set output mute state (useful for calibration workflow)
    fn set_output_mute(&mut self, muted: bool) -> PyResult<()> {
        self.processor.set_output_mute(muted);
        Ok(())
    }
}
