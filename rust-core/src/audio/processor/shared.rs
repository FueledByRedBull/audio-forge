//! State shared between the control side (Python-facing `AudioProcessor`), the DSP
//! thread, the supervisor and the VAD worker.
//!
//! Fields are grouped by concern. All of them are plain atomics, mutexes or
//! seqlock snapshots, so the DSP thread reads and writes them without blocking.
//! A few stay `Arc` because a CPAL callback, a suppressor engine or a ring
//! buffer owns its own handle.

use super::control::{
    apply_compressor_control, AtomicCompressorControlState, AtomicDeesserControlState,
    AtomicGateControlState, AtomicLimiterControlState, AtomicSuppressorControlState,
    CompressorControlState, DeesserControlState, EqControlState, GateControlState,
    LimiterControlState, SuppressorControlState, CAREFUL_OUTPUT_CEILING_DB,
};
use super::diagnostics::StreamRecoveryState;
use super::routing::InputCleanupMode;
use super::{
    COMPRESSOR_DEFAULT_RELEASE_TENTH_MS, INPUT_PREFILTER_HZ, PROCESS_IDLE_SLEEP_US,
    RT_SUPPRESSOR_COMMAND_CAPACITY, RT_SUPPRESSOR_RETIRE_CAPACITY,
};
use crate::audio::buffer::AudioProducer;
use crate::audio::input::InputChannelMode;
use crate::audio::rt::{RtCommandQueue, RtErrorCode};
use crate::dsp::noise_suppressor::{
    ControlledNoiseSuppressionEngine as NoiseSuppressionEngine, NoiseModel,
};
use crate::dsp::{Compressor, DeEsser, Limiter, ParametricEQ};
use ringbuf::{HeapCons, HeapProd};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Noise gate controls and the state it publishes.
pub(super) struct GateShared {
    pub(super) enabled: AtomicBool,
    pub(super) control: Mutex<GateControlState>,
    pub(super) rt_control: AtomicGateControlState,
    pub(super) dirty: AtomicBool,
    /// Last known gate gain (0.0-1.0) for metering.
    pub(super) gain_meter: AtomicU32,
    /// Latest gate noise-floor estimate in dB.
    #[cfg_attr(not(feature = "vad"), allow(dead_code))]
    pub(super) noise_floor_db: AtomicU32,
    /// Latest fused gate open score (0.0-1.0) for diagnostics.
    #[cfg_attr(not(feature = "vad"), allow(dead_code))]
    pub(super) fused_score: AtomicU32,
    /// Whether VAD-mode chatter mitigation is temporarily relaxing the gate.
    pub(super) auto_relax_active: AtomicBool,
    /// Lifetime count of detected rapid gate open/close chatter.
    pub(super) chatter_event_count: AtomicU64,
}

/// Noise suppressor controls, engine hand-off queues and health.
pub(super) struct SuppressorShared {
    pub(super) enabled: AtomicBool,
    pub(super) control: Mutex<SuppressorControlState>,
    pub(super) rt_control: AtomicSuppressorControlState,
    pub(super) dirty: AtomicBool,
    pub(super) reset_requested: AtomicBool,
    /// Control side of the queue that hands new engines to the loop.
    pub(super) pending_tx: Mutex<Option<HeapProd<NoiseSuppressionEngine>>>,
    /// Control side of the queue that takes retired engines back, so they are
    /// dropped off the DSP thread.
    pub(super) retired_rx: Mutex<Option<HeapCons<NoiseSuppressionEngine>>>,
    /// Wet/dry strength shared with every engine.
    pub(super) strength: Arc<AtomicU32>,
    pub(super) current_model: AtomicU8,
    /// The model of the engine a started loop runs: published by `start` before
    /// the stream is marked running, and by the loop when it swaps engines.
    pub(super) engine_model: AtomicU8,
    /// Noise suppressor internal buffer fill level (samples)
    pub(super) buffer_len: AtomicU32,
    /// Suppressor latency + pending samples mirrored from the processing thread.
    pub(super) latency_samples: AtomicU32,
    /// Successful external suppressor inference frames mirrored from the processing thread.
    pub(super) successful_inference_frames: AtomicU64,
    /// Smoothed suppressor buffer fill level (EMA per processed block, nominal 200ms time constant)
    pub(super) smoothed_buffer_len: AtomicU32,
    /// Number of times suppressor output contained non-finite samples.
    pub(super) non_finite_count: AtomicU64,
    /// Selections dropped after the stream started because the engine for the
    /// model did not build or could not be queued.
    pub(super) dropped_selection_count: AtomicU64,
    /// Whether the current noise backend is operational.
    pub(super) backend_available: AtomicBool,
    /// Whether the backend has entered a failed state.
    pub(super) backend_failed: AtomicBool,
    /// Last backend error string.
    pub(super) backend_error: Mutex<Option<String>>,
}

/// Correction and tone EQ controls (the shadows serve getters on the control side).
pub(super) struct EqShared {
    /// Control-side copies of the independent microphone correction and user
    /// tone EQ stages; the loop owns its own.
    pub(super) correction_shadow: Mutex<ParametricEQ>,
    pub(super) tone_shadow: Mutex<ParametricEQ>,
    pub(super) enabled: AtomicBool,
    pub(super) control: EqControlState,
    pub(super) dirty: AtomicBool,
}

/// Compressor controls and the state it publishes.
pub(super) struct CompressorShared {
    /// Control-side copy of the compressor; the loop owns its own.
    pub(super) shadow: Mutex<Compressor>,
    pub(super) enabled: AtomicBool,
    pub(super) control: Mutex<CompressorControlState>,
    pub(super) rt_control: AtomicCompressorControlState,
    pub(super) dirty: AtomicBool,
    /// Compressor gain reduction in dB
    pub(super) gain_reduction: AtomicU32,
    /// Compressor current release time in milliseconds (for metering)
    pub(super) current_release_ms: AtomicU64,
    /// Compressor current measured loudness in LUFS (f64 bits).
    pub(super) current_lufs: AtomicU64,
    /// `Compressor::auto_makeup_error_code` of the live compressor (0 healthy).
    pub(super) auto_makeup_error: AtomicU8,
    /// Compressor current applied makeup gain in dB (f64 bits).
    pub(super) current_makeup_gain: AtomicU64,
}

/// De-esser controls and the state it publishes.
pub(super) struct DeesserShared {
    /// Control-side copy of the de-esser; the loop owns its own.
    pub(super) shadow: Mutex<DeEsser>,
    pub(super) control: Mutex<DeesserControlState>,
    pub(super) rt_control: AtomicDeesserControlState,
    pub(super) dirty: AtomicBool,
    /// De-esser gain reduction in dB
    pub(super) gain_reduction: AtomicU32,
    /// De-esser detector confidence (0.0-1.0)
    pub(super) detector_confidence: AtomicU32,
}

/// Sample limiter controls and the state it publishes.
pub(super) struct LimiterShared {
    /// Control-side copy of the hard limiter (brick-wall ceiling); the loop
    /// owns its own.
    pub(super) shadow: Mutex<Limiter>,
    pub(super) enabled: AtomicBool,
    pub(super) control: Mutex<LimiterControlState>,
    pub(super) rt_control: AtomicLimiterControlState,
    pub(super) dirty: AtomicBool,
    /// Current normal limiter gain reduction in dB.
    pub(super) gain_reduction_db: AtomicU32,
    /// Peak normal limiter gain reduction since the last DSP block in dB.
    pub(super) peak_gain_reduction_db: AtomicU32,
    /// Decaying recent normal limiter gain-reduction peak in dB.
    pub(super) gain_reduction_history_db: AtomicU32,
}

/// Processing path selection.
pub(super) struct ModeShared {
    /// Master bypass flag
    pub(super) bypass: AtomicBool,
    /// True raw monitor path: bypasses pre-filter/DSP and uses minimal output write path.
    pub(super) raw_monitor: AtomicBool,
}

/// Capture side: callback health, level meters, clipping and backlog counters.
pub(super) struct InputShared {
    /// Input channel mixdown mode consumed by the CPAL input callback.
    pub(super) channel_mode: Arc<AtomicU8>,
    /// Optional adaptive cleanup mode that owns the input high-pass topology.
    pub(super) cleanup_mode: AtomicU8,
    /// Negotiated fixed input callback buffer size, or zero for driver default.
    pub(super) fixed_buffer_frames: AtomicU32,
    /// Latest stereo input correlation reported by the input callback.
    pub(super) stereo_correlation: Arc<AtomicU32>,
    /// Number of stereo input blocks with strong negative correlation.
    pub(super) phase_warning_count: Arc<AtomicU64>,
    /// Last phase-safe mono rescue strategy selected by the input callback.
    pub(super) phase_rescue_strategy: Arc<AtomicU8>,
    /// Last estimated stereo channel delay in samples for phase-safe mono.
    pub(super) phase_estimated_delay_samples: Arc<AtomicU32>,
    /// Whether the last phase-safe mono rescue flipped polarity.
    pub(super) phase_polarity_flipped: Arc<AtomicBool>,
    /// Whether adaptive input cleanup is currently suppressing line hum.
    pub(super) cleanup_hum_detected: AtomicBool,
    /// Whether adaptive input cleanup is currently reacting to rumble/plosive energy.
    pub(super) cleanup_rumble_detected: AtomicBool,
    /// Selected effective input cleanup high-pass cutoff in Hz.
    pub(super) cleanup_high_pass_hz: AtomicU32,
    // Levels are f32 bits (to_bits/from_bits).
    /// Input peak level at processor tap (pre-filtered in normal mode, raw in raw-monitor mode)
    pub(super) peak: AtomicU32,
    /// Input RMS level
    pub(super) rms: AtomicU32,
    /// Input crest factor in dB (decaying sample peak over RMS, same 100 ms window).
    pub(super) crest_factor_db: AtomicU32,
    /// Input ring buffer fill level (samples)
    pub(super) buffer_len: AtomicU32,
    /// Last input callback heartbeat timestamp (unix micros)
    pub(super) callback_time_us: Arc<AtomicU64>,
    /// Smoothed input buffer fill level (EMA over measured wall time, 200ms time constant)
    pub(super) smoothed_buffer_len: AtomicU32,
    /// Dropped samples counter from input ring buffer
    dropped: Mutex<Arc<AtomicU64>>,
    /// Counter value at the last user reset. The DSP loop detects capture
    /// gaps from `input_dropped` itself, so a reset must not rewind it.
    pub(super) dropped_baseline: AtomicU64,
    /// Number of proactive input backlog recoveries.
    pub(super) backlog_recovery_count: AtomicU64,
    /// Samples proactively dropped from the input backlog.
    pub(super) backlog_dropped_samples: AtomicU64,
    /// Number of clipped input samples before safety clamp.
    pub(super) clip_event_count: AtomicU64,
    /// Peak pre-clamp level in dBFS.
    pub(super) clip_peak_db: AtomicU32,
    /// Largest absolute input sample since `take_input_peak_since_read`
    /// (f32 bits, linear; measured on the mono engine-rate signal before the
    /// pre-filter, in every processing path).
    pub(super) peak_hold: AtomicU32,
    /// Input samples at or above `FULL_SCALE_LINEAR`, cumulative since start.
    pub(super) full_scale_count: AtomicU64,
    /// Whether input resampling is active.
    pub(super) resampler_active: AtomicBool,
    /// Input callback stream error count.
    pub(super) callback_error_count: Arc<AtomicU64>,
}

impl InputShared {
    /// The capture ring's dropped-sample counter; each start brings a new ring.
    pub(super) fn dropped_counter(&self) -> Arc<AtomicU64> {
        let counter = self
            .dropped
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Arc::clone(&counter)
    }

    pub(super) fn set_dropped_counter(&self, counter: Arc<AtomicU64>) {
        *self
            .dropped
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = counter;
    }
}

/// Playback side: queue health, level meters, output protection and the probe.
pub(super) struct OutputShared {
    /// Actual active output device sample rate
    pub(super) sample_rate: AtomicU32,
    /// Negotiated fixed output callback buffer size, or zero for driver default.
    pub(super) fixed_buffer_frames: AtomicU32,
    /// Output peak level (after all processing)
    pub(super) peak: AtomicU32,
    /// Output RMS level
    pub(super) rms: AtomicU32,
    /// Output crest factor in dB (decaying sample peak over RMS, same 100 ms window).
    pub(super) crest_factor_db: AtomicU32,
    /// Output short-term loudness estimate in LUFS-like dB.
    pub(super) short_term_lufs: AtomicU32,
    /// Output ring buffer fill level (samples)
    pub(super) buffer_len: AtomicU32,
    /// Last successful output write time (for detecting stalled processing)
    pub(super) last_write_time_us: Arc<AtomicU64>,
    /// Last time the writer queued a block that was not all exact zeros
    /// (unix micros; 0 = none since the start).
    pub(super) last_signal_write_time_us: AtomicU64,
    /// Last output callback heartbeat timestamp (unix micros)
    pub(super) callback_time_us: Arc<AtomicU64>,
    /// Consecutive output callback underrun count
    pub(super) underrun_streak: Arc<AtomicU32>,
    /// Total output callback underruns since start
    pub(super) underrun_total: Arc<AtomicU64>,
    /// Output callbacks that ran dry before the stream was primed
    pub(super) priming_underrun_total: Arc<AtomicU64>,
    /// Output callback has fully served a request after real audio was queued
    pub(super) primed: Arc<AtomicBool>,
    /// Callback time the current start's priming bound runs from (0 until first underrun after real audio)
    pub(super) priming_started_at_us: Arc<AtomicU64>,
    /// Samples dropped by DSP-side jitter buffer control
    pub(super) jitter_dropped_samples: AtomicU64,
    /// Number of normal DSP-side drift-retime length adjustments.
    pub(super) retime_adjustment_count: AtomicU64,
    /// Number of true DSP-side output recovery events.
    pub(super) recovery_event_count: AtomicU64,
    /// Samples discarded because the output ring buffer could not accept the full write.
    pub(super) short_write_dropped_samples: AtomicU64,
    /// Output resampler failures; each drops a chunk and requests a restart.
    pub(super) resampler_error_count: AtomicU64,
    /// Largest absolute sample of the meter's OUT tap (before the final
    /// true-peak limiter and clamp) since the last take (f32 bits, linear).
    pub(super) pre_limiter_peak_hold: AtomicU32,
    /// Largest absolute sample written to the output queue, after the final
    /// limiter, clamp and fade, since the last take (f32 bits, linear).
    pub(super) final_peak_hold: AtomicU32,
    /// Queued output samples at or above `FULL_SCALE_LINEAR`, cumulative since start.
    pub(super) full_scale_count: AtomicU64,
    /// Number of final output samples clamped by output protection.
    pub(super) clip_event_count: AtomicU64,
    /// Peak final pre-clamp output level in dBFS.
    pub(super) clip_peak_db: AtomicU32,
    /// Number of output blocks with estimated true peak above the active ceiling.
    pub(super) true_peak_event_count: AtomicU64,
    /// Peak estimated true-peak level in dBFS.
    pub(super) true_peak_db: AtomicU32,
    /// Peak estimated true-peak level before final true-peak limiting in dBFS.
    pub(super) true_peak_input_db: AtomicU32,
    /// Current final true-peak limiter gain reduction in dB.
    pub(super) true_peak_gain_reduction_db: AtomicU32,
    /// Output true-peak headroom relative to the active ceiling in dB.
    pub(super) true_peak_headroom_db: AtomicU32,
    /// Decaying recent final true-peak limiter gain-reduction peak in dB.
    pub(super) true_peak_gain_reduction_history_db: AtomicU32,
    /// Whether output resampling is active.
    pub(super) resampler_active: AtomicBool,
    /// Output callback stream error count.
    pub(super) callback_error_count: Arc<AtomicU64>,
    /// Manual output mute flag (independent of recording).
    pub(super) muted: Arc<AtomicBool>,
    /// Producer for calibration probes rendered by the selected CPAL output stream.
    pub(super) probe_producer: Mutex<Option<AudioProducer>>,
    /// Whether the selected output callback should render the queued calibration probe.
    pub(super) probe_active: Arc<AtomicBool>,
    /// Whether the last queued output probe has been fully rendered.
    pub(super) probe_complete: Arc<AtomicBool>,
    /// Requests that the output callback discard any remaining queued probe.
    pub(super) probe_cancel_requested: Arc<AtomicBool>,
}

impl OutputShared {
    /// Nothing queued or rendering, and the last probe counts as complete.
    pub(super) fn reset_probe(&self) {
        self.probe_active.store(false, Ordering::Release);
        self.probe_complete.store(true, Ordering::Release);
        self.probe_cancel_requested.store(false, Ordering::Release);
    }
}

/// Raw recording tap used for calibration.
pub(super) struct RecordingShared {
    /// Raw audio recording consumer (for calibration - captures audio AFTER pre-filter, BEFORE gate)
    pub(super) consumer: Mutex<Option<crate::audio::buffer::AudioConsumer>>,
    /// Current recording position (samples recorded so far)
    pub(super) pos: AtomicUsize,
    /// Target recording length (total samples to record)
    pub(super) target: AtomicUsize,
    /// Current recording level in dB for UI/self-test metering.
    pub(super) level_db: AtomicU32,
    /// Flag indicating recording is active (used to mute output to prevent user from hearing themselves)
    pub(super) active: Arc<AtomicBool>,
    /// Whether the raw recording tap should run before adaptive input cleanup.
    /// The default calibration tap remains after the fixed pre-filter.
    pub(super) before_cleanup: AtomicBool,
}

impl RecordingShared {
    /// Stale capture state cleared: no recording, nothing recorded, level at
    /// the floor. The consumer ring and the caller's output mute are left alone.
    pub(super) fn clear_capture(&self) {
        self.active.store(false, Ordering::Release);
        self.before_cleanup.store(false, Ordering::Release);
        self.pos.store(0, Ordering::Release);
        self.target.store(0, Ordering::Release);
        self.level_db
            .store((-120.0_f32).to_bits(), Ordering::Relaxed);
    }
}

/// Voice-activity worker results and health.
pub(super) struct VadShared {
    /// Latest raw VAD speech probability (0.0-1.0), written only by the worker.
    #[cfg_attr(not(feature = "vad"), allow(dead_code))]
    pub(super) raw_probability: AtomicU32,
    /// Smoothed VAD speech probability (0.0-1.0) for metering.
    #[cfg_attr(not(feature = "vad"), allow(dead_code))]
    pub(super) meter_probability: AtomicU32,
    /// Whether the current VAD backend is loaded, written only by the worker.
    #[cfg_attr(not(feature = "vad"), allow(dead_code))]
    pub(super) backend_available: AtomicBool,
    /// Whether a fresh VAD result is available to the realtime gate.
    #[cfg_attr(not(feature = "vad"), allow(dead_code))]
    pub(super) available: AtomicBool,
    #[cfg(feature = "vad")]
    /// Non-realtime VAD worker running flag.
    pub(super) worker_running: AtomicBool,
    #[cfg(feature = "vad")]
    /// Last successful non-realtime VAD inference timestamp.
    pub(super) last_update_us: AtomicU64,
    #[cfg(feature = "vad")]
    /// Input-rate source position covered by the last successful VAD result.
    /// This lets the DSP thread measure audio age independently of worker
    /// scheduling and publication time.
    pub(super) source_sample_end: AtomicU64,
    #[cfg(feature = "vad")]
    /// Incremented when the bounded VAD queue drops a block. The stateful
    /// model must restart at the next contiguous source sample.
    pub(super) source_discontinuity: AtomicU64,
    #[cfg(feature = "vad")]
    /// Seqlock for the worker-owned VAD result fields.
    pub(super) result_sequence: AtomicU64,
    #[cfg(feature = "vad")]
    /// Source discontinuity generation bound to the published result.
    pub(super) result_generation: AtomicU64,
    #[cfg(feature = "vad")]
    /// Last reason the VAD worker could not provide results.
    pub(super) worker_error: Mutex<Option<String>>,
}

/// DSP loop timing, latency and RT error counters.
pub(super) struct DspShared {
    /// Processing latency in microseconds (measured from input to output)
    pub(super) latency_us: AtomicU64,
    /// User-configured compensation added to reported latency (microseconds).
    pub(super) latency_compensation_us: AtomicU64,
    /// DSP processing time in microseconds (for last processed chunk)
    pub(super) time_us: AtomicU64,
    /// Number of times the DSP loop entered its idle sleep path.
    pub(super) idle_wakeup_count: AtomicU64,
    /// Last idle sleep duration selected by the DSP loop.
    pub(super) idle_sleep_us: AtomicU64,
    /// Smoothed DSP processing time in microseconds (EMA per processed block, nominal 200ms time constant)
    pub(super) time_smoothed_us: AtomicU64,
    /// Control-snapshot reads in the real-time loop that stayed unstable and were re-armed.
    pub(super) lock_contention_count: AtomicU64,
    /// Number of blocks where the downstream chain (de-esser, EQ, compressor,
    /// limiter) produced non-finite samples; the stages were reset each time.
    pub(super) downstream_non_finite_count: AtomicU64,
    /// True when the DSP thread could not register for MMCSS Pro Audio
    /// scheduling and runs at normal priority.
    pub(super) scheduling_degraded: AtomicBool,
    /// Last RT-safe error code published by callbacks or the DSP loop.
    pub(super) rt_error_code: Arc<AtomicU32>,
    /// Fixed-buffer overflow count in AudioForge RT regions.
    pub(super) rt_buffer_overflow_count: AtomicU64,
    /// Non-finite values the clamping live setters ignored.
    pub(super) rejected_control_value_count: AtomicU64,
}

/// Stream lifecycle, restart requests and recovery state.
pub(super) struct StreamShared {
    pub(super) supervisor_running: AtomicBool,
    pub(super) restart_requested: Arc<AtomicBool>,
    pub(super) recovering: AtomicBool,
    pub(super) restart_backoff_index: AtomicU32,
    pub(super) last_restart_attempt_us: AtomicU64,
    pub(super) last_start_time_us: AtomicU64,
    /// When the supervisor last saw the output silent under a live microphone
    /// (`now_micros`; 0 = never). A restart only counts as recovered once this
    /// is as old as the healthy-stream time.
    pub(super) silent_output_seen_us: AtomicU64,
    pub(super) recovery_state: Mutex<StreamRecoveryState>,
    /// Running flag
    pub(super) running: AtomicBool,
    /// Temporarily disables watchdog-driven recovery during intrusive UI workflows.
    pub(super) recovery_suppressed: AtomicBool,
    /// Counts stops, so a recovery attempt can tell that a stop ran while it
    /// waited for the lifecycle lock.
    pub(super) stop_epoch: AtomicU64,
}

impl SuppressorShared {
    /// Open the engine hand-off queues for a run. The control side's ends are
    /// published (`pending_tx`: new engines to the loop; `retired_rx`: old ones
    /// back), the loop's ends are returned.
    pub(super) fn open_queues(
        &self,
    ) -> (
        HeapCons<NoiseSuppressionEngine>,
        HeapProd<NoiseSuppressionEngine>,
    ) {
        let (commands_tx, commands_rx) =
            RtCommandQueue::<NoiseSuppressionEngine, RT_SUPPRESSOR_COMMAND_CAPACITY>::new().split();
        if let Ok(mut tx) = self.pending_tx.lock() {
            *tx = Some(commands_tx);
        }
        let (retired_tx, retired_rx) =
            RtCommandQueue::<NoiseSuppressionEngine, RT_SUPPRESSOR_RETIRE_CAPACITY>::new().split();
        if let Ok(mut rx) = self.retired_rx.lock() {
            *rx = Some(retired_rx);
        }
        (commands_rx, retired_tx)
    }
}

/// Everything the control side and the worker threads share.
pub(super) struct Shared {
    pub(super) gate: GateShared,
    pub(super) suppressor: SuppressorShared,
    pub(super) eq: EqShared,
    pub(super) compressor: CompressorShared,
    pub(super) deesser: DeesserShared,
    pub(super) limiter: LimiterShared,
    pub(super) mode: ModeShared,
    pub(super) input: InputShared,
    pub(super) output: OutputShared,
    pub(super) recording: RecordingShared,
    #[cfg_attr(not(feature = "vad"), allow(dead_code))]
    pub(super) vad: VadShared,
    pub(super) dsp: DspShared,
    pub(super) stream: StreamShared,
}

impl Shared {
    pub(super) fn new(sample_rate: u32) -> Self {
        let suppressor_control_state = SuppressorControlState::new();
        let compressor_control_state = CompressorControlState::new();
        // The strength handle is shared with every suppressor engine.
        let suppressor_strength = Arc::new(AtomicU32::new(1.0_f32.to_bits()));

        Self {
            gate: GateShared {
                enabled: AtomicBool::new(true),
                control: Mutex::new(GateControlState::new()),
                rt_control: AtomicGateControlState::new(),
                dirty: AtomicBool::new(false),
                gain_meter: AtomicU32::new(1.0_f32.to_bits()),
                noise_floor_db: AtomicU32::new((-60.0_f32).to_bits()),
                fused_score: AtomicU32::new(0.0_f32.to_bits()),
                auto_relax_active: AtomicBool::new(false),
                chatter_event_count: AtomicU64::new(0),
            },
            suppressor: SuppressorShared {
                enabled: AtomicBool::new(true),
                control: Mutex::new(suppressor_control_state),
                rt_control: AtomicSuppressorControlState::new(),
                dirty: AtomicBool::new(false),
                reset_requested: AtomicBool::new(false),
                pending_tx: Mutex::new(None),
                retired_rx: Mutex::new(None),
                strength: suppressor_strength,
                current_model: AtomicU8::new(NoiseModel::RNNoise as u8),
                engine_model: AtomicU8::new(NoiseModel::RNNoise as u8),
                buffer_len: AtomicU32::new(0),
                latency_samples: AtomicU32::new(0),
                successful_inference_frames: AtomicU64::new(0),
                smoothed_buffer_len: AtomicU32::new(0),
                non_finite_count: AtomicU64::new(0),
                dropped_selection_count: AtomicU64::new(0),
                backend_available: AtomicBool::new(true),
                backend_failed: AtomicBool::new(false),
                backend_error: Mutex::new(None),
            },
            eq: EqShared {
                correction_shadow: Mutex::new(ParametricEQ::new(sample_rate as f64)),
                tone_shadow: Mutex::new(ParametricEQ::new(sample_rate as f64)),
                enabled: AtomicBool::new(true),
                control: EqControlState::new(),
                dirty: AtomicBool::new(false),
            },
            compressor: CompressorShared {
                shadow: {
                    let mut compressor = Compressor::default_voice(sample_rate as f64);
                    apply_compressor_control(&mut compressor, &compressor_control_state);
                    Mutex::new(compressor)
                },
                enabled: AtomicBool::new(true),
                control: Mutex::new(compressor_control_state),
                rt_control: AtomicCompressorControlState::new(),
                dirty: AtomicBool::new(false),
                gain_reduction: AtomicU32::new(0.0_f32.to_bits()),
                current_release_ms: AtomicU64::new(COMPRESSOR_DEFAULT_RELEASE_TENTH_MS),
                current_lufs: AtomicU64::new((-100.0_f64).to_bits()),
                auto_makeup_error: AtomicU8::new(0),
                current_makeup_gain: AtomicU64::new(0.0_f64.to_bits()),
            },
            deesser: DeesserShared {
                shadow: Mutex::new(DeEsser::new(sample_rate as f64)),
                control: Mutex::new(DeesserControlState::new()),
                rt_control: AtomicDeesserControlState::new(),
                dirty: AtomicBool::new(false),
                gain_reduction: AtomicU32::new(0.0_f32.to_bits()),
                detector_confidence: AtomicU32::new(0.0_f32.to_bits()),
            },
            limiter: LimiterShared {
                shadow: Mutex::new(Limiter::new(
                    CAREFUL_OUTPUT_CEILING_DB,
                    50.0,
                    sample_rate as f64,
                )),
                enabled: AtomicBool::new(true),
                control: Mutex::new(LimiterControlState::new()),
                rt_control: AtomicLimiterControlState::new(),
                dirty: AtomicBool::new(false),
                gain_reduction_db: AtomicU32::new(0.0_f32.to_bits()),
                peak_gain_reduction_db: AtomicU32::new(0.0_f32.to_bits()),
                gain_reduction_history_db: AtomicU32::new(0.0_f32.to_bits()),
            },
            mode: ModeShared {
                bypass: AtomicBool::new(false),
                raw_monitor: AtomicBool::new(false),
            },
            input: InputShared {
                channel_mode: Arc::new(AtomicU8::new(InputChannelMode::PhaseSafeMono as u8)),
                cleanup_mode: AtomicU8::new(InputCleanupMode::Off as u8),
                fixed_buffer_frames: AtomicU32::new(0),
                stereo_correlation: Arc::new(AtomicU32::new(1.0_f32.to_bits())),
                phase_warning_count: Arc::new(AtomicU64::new(0)),
                phase_rescue_strategy: Arc::new(AtomicU8::new(0)),
                phase_estimated_delay_samples: Arc::new(AtomicU32::new(0.0_f32.to_bits())),
                phase_polarity_flipped: Arc::new(AtomicBool::new(false)),
                cleanup_hum_detected: AtomicBool::new(false),
                cleanup_rumble_detected: AtomicBool::new(false),
                cleanup_high_pass_hz: AtomicU32::new((INPUT_PREFILTER_HZ as f32).to_bits()),
                peak: AtomicU32::new((-120.0_f32).to_bits()),
                rms: AtomicU32::new((-120.0_f32).to_bits()),
                crest_factor_db: AtomicU32::new(0.0_f32.to_bits()),
                buffer_len: AtomicU32::new(0),
                callback_time_us: Arc::new(AtomicU64::new(0)),
                smoothed_buffer_len: AtomicU32::new(0),
                dropped: Mutex::new(Arc::new(AtomicU64::new(0))),
                dropped_baseline: AtomicU64::new(0),
                backlog_recovery_count: AtomicU64::new(0),
                backlog_dropped_samples: AtomicU64::new(0),
                clip_event_count: AtomicU64::new(0),
                clip_peak_db: AtomicU32::new((-120.0_f32).to_bits()),
                peak_hold: AtomicU32::new(0),
                full_scale_count: AtomicU64::new(0),
                resampler_active: AtomicBool::new(false),
                callback_error_count: Arc::new(AtomicU64::new(0)),
            },
            output: OutputShared {
                sample_rate: AtomicU32::new(sample_rate),
                fixed_buffer_frames: AtomicU32::new(0),
                peak: AtomicU32::new((-120.0_f32).to_bits()),
                rms: AtomicU32::new((-120.0_f32).to_bits()),
                crest_factor_db: AtomicU32::new(0.0_f32.to_bits()),
                short_term_lufs: AtomicU32::new((-120.0_f32).to_bits()),
                buffer_len: AtomicU32::new(0),
                last_write_time_us: Arc::new(AtomicU64::new(0)),
                last_signal_write_time_us: AtomicU64::new(0),
                callback_time_us: Arc::new(AtomicU64::new(0)),
                underrun_streak: Arc::new(AtomicU32::new(0)),
                underrun_total: Arc::new(AtomicU64::new(0)),
                priming_underrun_total: Arc::new(AtomicU64::new(0)),
                primed: Arc::new(AtomicBool::new(false)),
                priming_started_at_us: Arc::new(AtomicU64::new(0)),
                jitter_dropped_samples: AtomicU64::new(0),
                retime_adjustment_count: AtomicU64::new(0),
                recovery_event_count: AtomicU64::new(0),
                short_write_dropped_samples: AtomicU64::new(0),
                resampler_error_count: AtomicU64::new(0),
                pre_limiter_peak_hold: AtomicU32::new(0),
                final_peak_hold: AtomicU32::new(0),
                full_scale_count: AtomicU64::new(0),
                clip_event_count: AtomicU64::new(0),
                clip_peak_db: AtomicU32::new((-120.0_f32).to_bits()),
                true_peak_event_count: AtomicU64::new(0),
                true_peak_db: AtomicU32::new((-120.0_f32).to_bits()),
                true_peak_input_db: AtomicU32::new((-120.0_f32).to_bits()),
                true_peak_gain_reduction_db: AtomicU32::new(0.0_f32.to_bits()),
                true_peak_headroom_db: AtomicU32::new(120.0_f32.to_bits()),
                true_peak_gain_reduction_history_db: AtomicU32::new(0.0_f32.to_bits()),
                resampler_active: AtomicBool::new(false),
                callback_error_count: Arc::new(AtomicU64::new(0)),
                muted: Arc::new(AtomicBool::new(false)),
                probe_producer: Mutex::new(None),
                probe_active: Arc::new(AtomicBool::new(false)),
                probe_complete: Arc::new(AtomicBool::new(true)),
                probe_cancel_requested: Arc::new(AtomicBool::new(false)),
            },
            recording: RecordingShared {
                consumer: Mutex::new(None),
                pos: AtomicUsize::new(0),
                target: AtomicUsize::new(0),
                level_db: AtomicU32::new((-120.0_f32).to_bits()),
                active: Arc::new(AtomicBool::new(false)),
                before_cleanup: AtomicBool::new(false),
            },
            vad: VadShared {
                raw_probability: AtomicU32::new(0.0_f32.to_bits()),
                meter_probability: AtomicU32::new(0.0_f32.to_bits()),
                backend_available: AtomicBool::new(false),
                available: AtomicBool::new(false),
                #[cfg(feature = "vad")]
                worker_running: AtomicBool::new(false),
                #[cfg(feature = "vad")]
                last_update_us: AtomicU64::new(0),
                #[cfg(feature = "vad")]
                source_sample_end: AtomicU64::new(0),
                #[cfg(feature = "vad")]
                source_discontinuity: AtomicU64::new(0),
                #[cfg(feature = "vad")]
                result_sequence: AtomicU64::new(0),
                #[cfg(feature = "vad")]
                result_generation: AtomicU64::new(0),
                #[cfg(feature = "vad")]
                worker_error: Mutex::new(None),
            },
            dsp: DspShared {
                latency_us: AtomicU64::new(0),
                latency_compensation_us: AtomicU64::new(0),
                time_us: AtomicU64::new(0),
                idle_wakeup_count: AtomicU64::new(0),
                idle_sleep_us: AtomicU64::new(PROCESS_IDLE_SLEEP_US),
                time_smoothed_us: AtomicU64::new(0),
                lock_contention_count: AtomicU64::new(0),
                downstream_non_finite_count: AtomicU64::new(0),
                scheduling_degraded: AtomicBool::new(false),
                rt_error_code: Arc::new(AtomicU32::new(RtErrorCode::None as u32)),
                rt_buffer_overflow_count: AtomicU64::new(0),
                rejected_control_value_count: AtomicU64::new(0),
            },
            stream: StreamShared {
                supervisor_running: AtomicBool::new(false),
                restart_requested: Arc::new(AtomicBool::new(false)),
                recovering: AtomicBool::new(false),
                restart_backoff_index: AtomicU32::new(0),
                last_restart_attempt_us: AtomicU64::new(0),
                last_start_time_us: AtomicU64::new(0),
                silent_output_seen_us: AtomicU64::new(0),
                recovery_state: Mutex::new(StreamRecoveryState::default()),
                running: AtomicBool::new(false),
                recovery_suppressed: AtomicBool::new(false),
                stop_epoch: AtomicU64::new(0),
            },
        }
    }
}

impl Shared {
    /// Zero the per-run counters and meters before a (re)start. Explicit
    /// settings (mute, controls) are left alone.
    pub(super) fn reset_for_start(&self) {
        // The new input ring brings a fresh dropped counter.
        self.input.dropped_baseline.store(0, Ordering::Relaxed);
        self.output.underrun_streak.store(0, Ordering::Relaxed);
        self.output.underrun_total.store(0, Ordering::Relaxed);
        self.output
            .priming_underrun_total
            .store(0, Ordering::Relaxed);
        self.output
            .jitter_dropped_samples
            .store(0, Ordering::Relaxed);
        self.output
            .retime_adjustment_count
            .store(0, Ordering::Relaxed);
        self.output.recovery_event_count.store(0, Ordering::Relaxed);
        self.output
            .short_write_dropped_samples
            .store(0, Ordering::Relaxed);
        self.dsp.idle_wakeup_count.store(0, Ordering::Relaxed);
        self.dsp
            .idle_sleep_us
            .store(PROCESS_IDLE_SLEEP_US, Ordering::Relaxed);
        self.dsp.lock_contention_count.store(0, Ordering::Relaxed);
        self.suppressor.non_finite_count.store(0, Ordering::Relaxed);
        self.dsp
            .downstream_non_finite_count
            .store(0, Ordering::Relaxed);
        self.output
            .resampler_error_count
            .store(0, Ordering::Relaxed);
        self.dsp.scheduling_degraded.store(false, Ordering::Relaxed);
        self.compressor
            .auto_makeup_error
            .store(0, Ordering::Relaxed);
        self.input
            .backlog_recovery_count
            .store(0, Ordering::Relaxed);
        self.input
            .backlog_dropped_samples
            .store(0, Ordering::Relaxed);
        self.input.peak_hold.store(0, Ordering::Relaxed);
        self.input.full_scale_count.store(0, Ordering::Relaxed);
        self.output
            .pre_limiter_peak_hold
            .store(0, Ordering::Relaxed);
        self.output.final_peak_hold.store(0, Ordering::Relaxed);
        self.output.full_scale_count.store(0, Ordering::Relaxed);
        self.input.clip_event_count.store(0, Ordering::Relaxed);
        self.input
            .clip_peak_db
            .store((-120.0_f32).to_bits(), Ordering::Relaxed);
        self.input
            .crest_factor_db
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        self.output
            .crest_factor_db
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        self.output
            .short_term_lufs
            .store((-120.0_f32).to_bits(), Ordering::Relaxed);
        self.output.clip_event_count.store(0, Ordering::Relaxed);
        self.output
            .clip_peak_db
            .store((-120.0_f32).to_bits(), Ordering::Relaxed);
        self.output
            .true_peak_event_count
            .store(0, Ordering::Relaxed);
        self.output
            .true_peak_db
            .store((-120.0_f32).to_bits(), Ordering::Relaxed);
        self.output
            .true_peak_input_db
            .store((-120.0_f32).to_bits(), Ordering::Relaxed);
        self.output
            .true_peak_gain_reduction_db
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        self.output
            .true_peak_headroom_db
            .store(120.0_f32.to_bits(), Ordering::Relaxed);
        self.limiter
            .gain_reduction_db
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        self.limiter
            .peak_gain_reduction_db
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        self.limiter
            .gain_reduction_history_db
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        self.output
            .true_peak_gain_reduction_history_db
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        self.dsp
            .rt_error_code
            .store(RtErrorCode::None as u32, Ordering::Relaxed);
        self.input.callback_error_count.store(0, Ordering::Relaxed);
        self.output.callback_error_count.store(0, Ordering::Relaxed);
        self.dsp
            .rt_buffer_overflow_count
            .store(0, Ordering::Relaxed);
        self.input
            .stereo_correlation
            .store(1.0_f32.to_bits(), Ordering::Relaxed);
        self.input.phase_warning_count.store(0, Ordering::Relaxed);
        self.input.phase_rescue_strategy.store(0, Ordering::Relaxed);
        self.input
            .phase_estimated_delay_samples
            .store(0.0_f32.to_bits(), Ordering::Relaxed);
        self.input
            .phase_polarity_flipped
            .store(false, Ordering::Relaxed);
        self.input
            .cleanup_hum_detected
            .store(false, Ordering::Relaxed);
        self.input
            .cleanup_rumble_detected
            .store(false, Ordering::Relaxed);
        self.input
            .cleanup_high_pass_hz
            .store((INPUT_PREFILTER_HZ as f32).to_bits(), Ordering::Relaxed);
        self.gate.chatter_event_count.store(0, Ordering::Relaxed);
        self.gate.auto_relax_active.store(false, Ordering::Relaxed);
        self.input.callback_time_us.store(0, Ordering::Relaxed);
        self.output.callback_time_us.store(0, Ordering::Relaxed);
        self.output.last_write_time_us.store(0, Ordering::Relaxed);
        self.output
            .last_signal_write_time_us
            .store(0, Ordering::Relaxed);
        self.output.primed.store(false, Ordering::Relaxed);
        self.output
            .priming_started_at_us
            .store(0, Ordering::Relaxed);
    }
}
