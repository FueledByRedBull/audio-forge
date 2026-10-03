//! Downward compressor with blended peak/RMS detection.
//!
//! Reduces dynamic range by attenuating signals above the threshold.

use crate::dsp::util;

const DETECTOR_PEAK_WEIGHT: f64 = 0.6;
const DETECTOR_RMS_WEIGHT: f64 = 0.4;
const ADAPTIVE_FAST_RELEASE_MS: f64 = 50.0;
const ADAPTIVE_SLOW_CHARGE_MS: f64 = 250.0;
const ADAPTIVE_SLOW_RELEASE_MS: f64 = 400.0;
const SLOW_RELEASE_TRIGGER_DB: f64 = 3.0;
const SPEECH_ACTIVE_RMS_MIN_DB: f64 = -55.0;
const SPEECH_ACTIVE_RMS_MAX_DB: f64 = -6.0;
const AUTO_MAKEUP_ACTIVE_MIN: f64 = 0.20;
const AUTO_MAKEUP_RELIABILITY_MIN: f64 = 0.35;
const AUTO_MAKEUP_ACTIVITY_SMOOTH_MS: f64 = 200.0;
const NOISE_RELATIVE_ACTIVITY_START_DB: f64 = 3.0;
const NOISE_RELATIVE_ACTIVITY_FULL_DB: f64 = 15.0;
const MAKEUP_SILENCE_RELAX_MS: f64 = 1500.0;
const AUTO_MAKEUP_SAMPLE_WINDOW_MS: f64 = 10.0;
// Compressor callers accept generic rates through 192 kHz; reserve the full
// 10 ms window at that rate even when the optional loudness meter is absent.
const AUTO_MAKEUP_SAMPLE_WINDOW_MAX_SAMPLES: usize = 1_920;
const SIDECHAIN_HIGHPASS_DEFAULT_HZ: f64 = 120.0;
const SIDECHAIN_BAND_ENV_MS: f64 = 18.0;
const PRESENCE_HIGHPASS_HZ: f64 = 2_000.0;
const PEAK_DETECTOR_RELEASE_MS: f64 = 5.0;
const ADAPTIVE_SLOW_RELEASE_MULTIPLIER: f64 = 8.0;
const ADAPTIVE_RELEASE_MAX_MS: f64 = 1_000.0;
const ADAPTIVE_RELEASE_COEFF_UPDATE_MS: f64 = 1.0;
const PLOSIVE_RATIO_START: f64 = 1.25;
const PLOSIVE_RATIO_FULL: f64 = 5.0;
const PLOSIVE_MIN_DETECTOR_GAIN: f64 = 0.35;
/// Enable/disable ramps between processed and unity gain instead of stepping.
const ENABLE_TRANSITION_MS: f64 = 10.0;

/// Non-realtime speech evidence used only by auto-makeup measurement/control.
///
/// All fields are soft evidence in `[0, 1]` except `noise_floor_db`. Invalid
/// values are rejected locally so they cannot poison realtime compressor state.
#[derive(Clone, Copy, Debug)]
pub struct AutoMakeupActivityInput {
    pub vad_probability: f64,
    pub vad_reliability: f64,
    pub noise_floor_db: f64,
    pub live_noise_reliability: f64,
}

#[derive(Clone, Copy, Debug)]
struct AutoMakeupActivityEstimate {
    activity: f64,
    reliability: f64,
}

/// Downward compressor with soft-knee gain reduction
pub struct Compressor {
    /// Threshold in dB - compression starts above this level
    threshold_db: f64,
    /// Compression ratio (e.g., 4.0 = 4:1 ratio)
    ratio: f64,
    /// Attack time constant (exponential smoothing coefficient)
    attack_coeff: f64,
    /// Release time constant for gain-reduction smoothing
    release_coeff: f64,
    /// Fixed coefficients used by the adaptive release controller.
    adaptive_fast_release_coeff: f64,
    adaptive_slow_charge_coeff: f64,
    adaptive_slow_release_coeff: f64,
    /// Fixed smoothing coefficient for sidechain-band energy measurements.
    sidechain_band_env_coeff: f64,
    /// Makeup gain in dB to compensate for gain reduction
    makeup_gain_db: f64,
    /// Knee width in dB for soft-knee transition
    knee_db: f64,
    /// Instant-attack peak capture with fixed-time amplitude release smoothing.
    peak_envelope: f64,
    /// Peak-envelope decay is independent of configured gain reduction.
    detector_release_coeff: f64,
    /// Fixed-time RMS detector state (squared amplitude)
    rms_envelope_sq: f64,
    /// RMS smoothing coefficient (single-pole IIR, fixed 20ms)
    rms_coeff: f64,
    /// Current gain reduction in dB (for metering)
    current_gain_reduction_db: f64,
    /// Maximum gain reduction observed in the most recently processed block.
    block_peak_gain_reduction_db: f64,
    /// Sample rate
    sample_rate: f64,
    /// Whether compressor is enabled
    enabled: bool,
    /// Enable-ramp position: 0 is bypassed, `enable_ramp_samples` is fully processed.
    enable_ramp_position: usize,
    enable_ramp_samples: usize,
    /// Whether adaptive release is enabled
    adaptive_release: bool,
    /// Base release time in milliseconds (user-controlled)
    base_release_ms: f64,
    /// Current release time in milliseconds (adaptive value)
    current_release_ms: f64,
    /// Target release time (for smoothing)
    target_release_ms: f64,
    /// Release smoothing coefficient (100ms hysteresis)
    release_smoothing_coeff: f64,
    /// Recompute the adaptive gain-release coefficient at 1 ms cadence.
    adaptive_release_coeff_update_samples: usize,
    adaptive_release_coeff_samples_until_update: usize,
    /// Fast adaptive release envelope in dB.
    fast_release_env_db: f64,
    /// Slow adaptive release envelope in dB.
    slow_release_env_db: f64,
    /// Loudness meter for auto makeup gain
    loudness_meter: Option<crate::dsp::loudness::LoudnessMeter>,
    /// Fixed storage for the sample API's auto-makeup control window.
    auto_makeup_sample_window: [f32; AUTO_MAKEUP_SAMPLE_WINDOW_MAX_SAMPLES],
    /// Number of output samples currently held for the sample API window.
    auto_makeup_sample_window_len: usize,
    /// Sum of per-sample activity evidence in the current sample window.
    auto_makeup_sample_activity_sum: f64,
    /// Number of samples in one sample API auto-makeup control window.
    auto_makeup_sample_window_samples: usize,
    /// Auto makeup gain enabled
    auto_makeup_enabled: bool,
    /// Target LUFS for auto makeup gain
    target_lufs: f64,
    /// Smoothed makeup gain (for transitions)
    smoothed_makeup_gain: f64,
    /// Makeup gain smoothing coefficient (200ms time constant)
    makeup_smoothing_coeff: f64,
    /// Current measured loudness (for metering)
    current_lufs: f64,
    /// Smoothed speech activity score for auto makeup.
    speech_activity_score: f64,
    /// Speech-activity smoothing coefficient, expressed per sample.
    speech_activity_smoothing_coeff: f64,
    /// Reliability of the most recent auto-makeup activity estimate.
    auto_makeup_activity_reliability: f64,
    /// Reliability of the room-noise reference supplied by Auto Voice Setup.
    noise_reference_reliability: f64,
    /// Slow relaxation coefficient used when auto makeup sees silence/noise.
    makeup_silence_relax_coeff: f64,
    /// Whether the detector sidechain ignores most plosive/rumble energy.
    sidechain_highpass_enabled: bool,
    /// Sidechain high-pass coefficient.
    sidechain_highpass_coeff: f64,
    /// Presence-band high-pass coefficient applied to the voiced sidechain.
    presence_highpass_coeff: f64,
    /// Previous sidechain high-pass input sample.
    sidechain_highpass_prev_input: f64,
    /// Previous sidechain high-pass output sample.
    sidechain_highpass_prev_output: f64,
    /// Previous presence high-pass input and output samples.
    presence_highpass_prev_input: f64,
    presence_highpass_prev_output: f64,
    /// Low-band detector energy used for plosive discrimination.
    low_band_env_sq: f64,
    /// Voiced-band detector energy used for plosive discrimination.
    voiced_band_env_sq: f64,
    /// Presence-band detector energy used to keep consonants forward.
    presence_band_env_sq: f64,
    /// Low/mid portion of the voiced band, complementary to the presence band.
    non_presence_band_env_sq: f64,
    /// Smoothed low/voiced ratio exposed for diagnostics and tests.
    plosive_ratio: f64,
    /// Previous limiter pressure used to keep auto makeup inside headroom.
    limiter_feedback_gain_reduction_db: f64,
}

impl Compressor {
    /// Create a new compressor
    pub fn new(
        threshold_db: f64,
        ratio: f64,
        attack_ms: f64,
        release_ms: f64,
        makeup_gain_db: f64,
        knee_db: f64,
        sample_rate: f64,
    ) -> Self {
        let attack_coeff = util::time_constant_to_coeff(attack_ms, sample_rate);
        let release_coeff = util::time_constant_to_coeff(release_ms, sample_rate);
        let detector_release_coeff =
            util::time_constant_to_coeff(PEAK_DETECTOR_RELEASE_MS, sample_rate);
        let rms_coeff = util::time_constant_to_coeff(20.0, sample_rate);
        let release_smoothing_coeff = util::time_constant_to_coeff(100.0, sample_rate);
        let makeup_smoothing_coeff = util::time_constant_to_coeff(200.0, sample_rate);
        let adaptive_fast_release_coeff =
            util::time_constant_to_coeff(ADAPTIVE_FAST_RELEASE_MS, sample_rate);
        let adaptive_slow_charge_coeff =
            util::time_constant_to_coeff(ADAPTIVE_SLOW_CHARGE_MS, sample_rate);
        let adaptive_slow_release_coeff =
            util::time_constant_to_coeff(ADAPTIVE_SLOW_RELEASE_MS, sample_rate);
        let sidechain_band_env_coeff =
            util::time_constant_to_coeff(SIDECHAIN_BAND_ENV_MS, sample_rate);
        let adaptive_release_coeff_update_samples = if sample_rate.is_finite() && sample_rate > 0.0
        {
            (sample_rate * ADAPTIVE_RELEASE_COEFF_UPDATE_MS / 1_000.0)
                .round()
                .clamp(1.0, 1_000_000.0) as usize
        } else {
            1
        };

        let loudness_meter = crate::dsp::loudness::LoudnessMeter::new(sample_rate as u32).ok();
        let enable_ramp_samples = if sample_rate.is_finite() && sample_rate > 0.0 {
            (sample_rate * ENABLE_TRANSITION_MS / 1_000.0)
                .round()
                .max(1.0) as usize
        } else {
            1
        };
        let auto_makeup_sample_window_samples = if sample_rate.is_finite() && sample_rate > 0.0 {
            (sample_rate * AUTO_MAKEUP_SAMPLE_WINDOW_MS / 1_000.0)
                .round()
                .clamp(1.0, AUTO_MAKEUP_SAMPLE_WINDOW_MAX_SAMPLES as f64) as usize
        } else {
            1
        };

        Self {
            threshold_db,
            ratio: ratio.max(1.0),
            attack_coeff,
            release_coeff,
            detector_release_coeff,
            adaptive_fast_release_coeff,
            adaptive_slow_charge_coeff,
            adaptive_slow_release_coeff,
            sidechain_band_env_coeff,
            makeup_gain_db,
            knee_db: knee_db.max(0.0),
            peak_envelope: 0.0,
            rms_envelope_sq: 0.0,
            rms_coeff,
            current_gain_reduction_db: 0.0,
            block_peak_gain_reduction_db: 0.0,
            sample_rate,
            enabled: true,
            enable_ramp_position: enable_ramp_samples,
            enable_ramp_samples,
            adaptive_release: false,
            base_release_ms: release_ms,
            current_release_ms: release_ms,
            target_release_ms: release_ms,
            release_smoothing_coeff,
            adaptive_release_coeff_update_samples,
            adaptive_release_coeff_samples_until_update: 1,
            fast_release_env_db: 0.0,
            slow_release_env_db: 0.0,
            loudness_meter,
            auto_makeup_sample_window: [0.0; AUTO_MAKEUP_SAMPLE_WINDOW_MAX_SAMPLES],
            auto_makeup_sample_window_len: 0,
            auto_makeup_sample_activity_sum: 0.0,
            auto_makeup_sample_window_samples,
            auto_makeup_enabled: false,
            target_lufs: -18.0,
            smoothed_makeup_gain: makeup_gain_db,
            makeup_smoothing_coeff,
            current_lufs: -100.0,
            speech_activity_score: 0.0,
            speech_activity_smoothing_coeff: util::time_constant_to_coeff(
                AUTO_MAKEUP_ACTIVITY_SMOOTH_MS,
                sample_rate,
            ),
            auto_makeup_activity_reliability: 0.0,
            noise_reference_reliability: 0.0,
            makeup_silence_relax_coeff: util::time_constant_to_coeff(
                MAKEUP_SILENCE_RELAX_MS,
                sample_rate,
            ),
            sidechain_highpass_enabled: false,
            sidechain_highpass_coeff: Self::sidechain_highpass_coeff(
                SIDECHAIN_HIGHPASS_DEFAULT_HZ,
                sample_rate,
            ),
            presence_highpass_coeff: Self::sidechain_highpass_coeff(
                PRESENCE_HIGHPASS_HZ,
                sample_rate,
            ),
            sidechain_highpass_prev_input: 0.0,
            sidechain_highpass_prev_output: 0.0,
            presence_highpass_prev_input: 0.0,
            presence_highpass_prev_output: 0.0,
            low_band_env_sq: 0.0,
            voiced_band_env_sq: 0.0,
            presence_band_env_sq: 0.0,
            non_presence_band_env_sq: 0.0,
            plosive_ratio: 0.0,
            limiter_feedback_gain_reduction_db: 0.0,
        }
    }

    /// Create with default parameters suitable for voice
    pub fn default_voice(sample_rate: f64) -> Self {
        Self::new(-20.0, 4.0, 10.0, 200.0, 0.0, 6.0, sample_rate)
    }

    /// Set threshold in dB
    pub fn set_threshold(&mut self, threshold_db: f64) {
        if self.threshold_db == threshold_db {
            return;
        }
        self.threshold_db = threshold_db;
        self.reset_adaptive_release_state();
    }

    /// Get current threshold in dB
    pub fn threshold_db(&self) -> f64 {
        self.threshold_db
    }

    /// Set compression ratio
    pub fn set_ratio(&mut self, ratio: f64) {
        self.ratio = ratio.max(1.0);
    }

    /// Get current ratio
    pub fn ratio(&self) -> f64 {
        self.ratio
    }

    /// Set attack time in ms
    pub fn set_attack_time(&mut self, attack_ms: f64) {
        self.attack_coeff = util::time_constant_to_coeff(attack_ms, self.sample_rate);
    }

    /// Set release time in ms
    pub fn set_release_time(&mut self, release_ms: f64) {
        if self.base_release_ms == release_ms {
            return;
        }
        self.base_release_ms = release_ms;
        if self.adaptive_release {
            self.update_adaptive_release_time_meter();
            self.adaptive_release_coeff_samples_until_update = 1;
        } else {
            self.current_release_ms = release_ms;
            self.target_release_ms = release_ms;
            self.release_coeff = util::time_constant_to_coeff(release_ms, self.sample_rate);
        }
    }

    /// Enable or disable adaptive release
    pub fn set_adaptive_release(&mut self, enabled: bool) {
        if self.adaptive_release == enabled {
            return;
        }
        self.adaptive_release = enabled;
        if !enabled {
            self.current_release_ms = self.base_release_ms;
            self.target_release_ms = self.base_release_ms;
            self.fast_release_env_db = self.current_gain_reduction_db;
            self.slow_release_env_db = 0.0;
            self.release_coeff =
                util::time_constant_to_coeff(self.current_release_ms, self.sample_rate);
        } else {
            self.fast_release_env_db = self.current_gain_reduction_db;
            self.slow_release_env_db = 0.0;
            self.update_adaptive_release_time_meter();
        }
        self.adaptive_release_coeff_samples_until_update = 1;
    }

    /// Check if adaptive release is enabled
    pub fn adaptive_release(&self) -> bool {
        self.adaptive_release
    }

    /// Set base release time
    pub fn set_base_release_time(&mut self, release_ms: f64) {
        if self.base_release_ms == release_ms {
            return;
        }
        self.base_release_ms = release_ms;
        if self.adaptive_release {
            self.update_adaptive_release_time_meter();
            self.adaptive_release_coeff_samples_until_update = 1;
        } else {
            self.current_release_ms = release_ms;
            self.target_release_ms = release_ms;
            self.release_coeff = util::time_constant_to_coeff(release_ms, self.sample_rate);
        }
    }

    /// Get current release time (adaptive or base)
    pub fn current_release_time(&self) -> f64 {
        self.current_release_ms
    }

    /// Get base release time in milliseconds
    pub fn base_release_ms(&self) -> f64 {
        self.base_release_ms
    }

    /// Reset adaptive release state (call when threshold changes)
    pub fn reset_adaptive_release_state(&mut self) {
        self.fast_release_env_db = self.current_gain_reduction_db;
        self.slow_release_env_db = 0.0;
    }

    /// Set makeup gain in dB
    pub fn set_makeup_gain(&mut self, makeup_gain_db: f64) {
        self.makeup_gain_db = makeup_gain_db;
    }

    /// Enable or disable the compressor with a short gain ramp.
    pub fn set_enabled(&mut self, enabled: bool) {
        if enabled && !self.enabled && self.enable_ramp_position == 0 {
            // Resume from full bypass with fresh detector state, not the
            // envelope frozen when processing stopped.
            self.reset_detector_state();
        }
        self.enabled = enabled;
        if !enabled {
            self.clear_auto_makeup_sample_window();
        }
    }

    /// Check if compressor is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Whether the compressor still affects audio (enabled or fading out).
    pub fn is_active(&self) -> bool {
        self.enabled || self.enable_ramp_position > 0
    }

    /// Complete any enable/disable ramp now, for state set before audio starts.
    pub fn finish_enable_transition(&mut self) {
        self.enable_ramp_position = if self.enabled {
            self.enable_ramp_samples
        } else {
            0
        };
    }

    /// Get current gain reduction in dB (for metering)
    pub fn current_gain_reduction(&self) -> f64 {
        self.current_gain_reduction_db
    }

    /// Maximum gain reduction reached during the most recent block.
    pub fn block_peak_gain_reduction(&self) -> f64 {
        self.block_peak_gain_reduction_db
    }

    /// Enable or disable auto makeup gain
    pub fn set_auto_makeup_enabled(&mut self, enabled: bool) {
        let enabled = enabled && self.loudness_meter.is_some();
        if self.auto_makeup_enabled == enabled {
            return;
        }
        self.auto_makeup_enabled = enabled;
        if !enabled {
            self.clear_auto_makeup_sample_window();
        }
    }

    /// Check if auto makeup is enabled
    pub fn auto_makeup_enabled(&self) -> bool {
        self.auto_makeup_enabled
    }

    /// Set target LUFS for auto makeup gain
    pub fn set_target_lufs(&mut self, target: f64) {
        let target = target.clamp(-24.0, -12.0);
        if self.target_lufs != target {
            self.target_lufs = target;
        }
    }

    /// Get target LUFS
    pub fn target_lufs(&self) -> f64 {
        self.target_lufs
    }

    /// Get current measured loudness (for metering)
    pub fn current_lufs(&self) -> f64 {
        self.current_lufs
    }

    /// Get current applied makeup gain (for metering)
    pub fn current_makeup_gain(&self) -> f64 {
        self.smoothed_makeup_gain
    }

    /// Set confidence in the room-noise reference used by auto makeup.
    pub fn set_noise_reference_reliability(&mut self, reliability: f64) {
        self.noise_reference_reliability = Self::finite_unit(reliability).unwrap_or(0.0);
    }

    /// Return the latest soft speech-activity estimate used by auto makeup.
    pub fn auto_makeup_activity(&self) -> f64 {
        self.speech_activity_score
    }

    /// Return the latest reliability attached to the auto-makeup estimate.
    pub fn auto_makeup_activity_reliability(&self) -> f64 {
        self.auto_makeup_activity_reliability
    }

    /// Enable or disable the detector sidechain high-pass.
    pub fn set_sidechain_highpass_enabled(&mut self, enabled: bool) {
        if self.sidechain_highpass_enabled != enabled {
            self.reset_sidechain_highpass_state();
        }
        self.sidechain_highpass_enabled = enabled;
    }

    /// Check whether the detector sidechain high-pass is enabled.
    pub fn sidechain_highpass_enabled(&self) -> bool {
        self.sidechain_highpass_enabled
    }

    /// Current low/voiced sidechain ratio used to de-emphasize plosives.
    pub fn plosive_ratio(&self) -> f64 {
        self.plosive_ratio
    }

    /// Feed previous limiter pressure into auto makeup so it does not chase
    /// loudness targets through unavailable headroom.
    pub fn set_limiter_feedback_gain_reduction_db(&mut self, gain_reduction_db: f64) {
        self.limiter_feedback_gain_reduction_db = gain_reduction_db.clamp(0.0, 24.0);
    }

    #[inline]
    fn sidechain_highpass_coeff(cutoff_hz: f64, sample_rate: f64) -> f64 {
        let cutoff_hz = cutoff_hz.clamp(20.0, sample_rate * 0.45);
        let omega = 2.0 * std::f64::consts::PI * cutoff_hz / sample_rate.max(1.0);
        1.0 / (1.0 + omega)
    }

    #[inline]
    fn reset_sidechain_highpass_state(&mut self) {
        self.sidechain_highpass_prev_input = 0.0;
        self.sidechain_highpass_prev_output = 0.0;
        self.presence_highpass_prev_input = 0.0;
        self.presence_highpass_prev_output = 0.0;
        self.low_band_env_sq = 0.0;
        self.voiced_band_env_sq = 0.0;
        self.presence_band_env_sq = 0.0;
        self.non_presence_band_env_sq = 0.0;
        self.plosive_ratio = 0.0;
    }

    fn reset_detector_state(&mut self) {
        self.peak_envelope = 0.0;
        self.rms_envelope_sq = 0.0;
        self.current_gain_reduction_db = 0.0;
        self.block_peak_gain_reduction_db = 0.0;
        self.fast_release_env_db = 0.0;
        self.slow_release_env_db = 0.0;
        self.reset_sidechain_highpass_state();
    }

    #[inline]
    fn process_sidechain_sample(&mut self, input: f64) -> f64 {
        if !self.sidechain_highpass_enabled {
            return input;
        }

        let output = self.sidechain_highpass_coeff
            * (self.sidechain_highpass_prev_output + input - self.sidechain_highpass_prev_input);
        self.sidechain_highpass_prev_input = input;
        self.sidechain_highpass_prev_output = output;
        output
    }

    #[inline]
    fn process_presence_highpass_sample(&mut self, input: f64) -> f64 {
        let output = self.presence_highpass_coeff
            * (self.presence_highpass_prev_output + input - self.presence_highpass_prev_input);
        self.presence_highpass_prev_input = input;
        self.presence_highpass_prev_output = output;
        output
    }

    #[inline]
    fn update_peak_envelope(&mut self, input_abs: f64) {
        if input_abs >= self.peak_envelope {
            self.peak_envelope = input_abs;
        } else {
            let coeff = self.detector_release_coeff;
            self.peak_envelope = coeff * self.peak_envelope + (1.0 - coeff) * input_abs;
        }
    }

    #[inline]
    fn update_sidechain_band_metrics(&mut self, full_band_input: f64, detector_input: f64) -> f64 {
        if !self.sidechain_highpass_enabled {
            self.plosive_ratio = 0.0;
            return 1.0;
        }

        let low_component = full_band_input - detector_input;
        let voiced_component = detector_input;
        let presence_component = self.process_presence_highpass_sample(voiced_component);
        let non_presence_component = voiced_component - presence_component;
        let coeff = self.sidechain_band_env_coeff;

        self.low_band_env_sq =
            coeff * self.low_band_env_sq + (1.0 - coeff) * low_component * low_component;
        self.voiced_band_env_sq =
            coeff * self.voiced_band_env_sq + (1.0 - coeff) * voiced_component * voiced_component;
        self.presence_band_env_sq = coeff * self.presence_band_env_sq
            + (1.0 - coeff) * presence_component * presence_component;
        self.non_presence_band_env_sq = coeff * self.non_presence_band_env_sq
            + (1.0 - coeff) * non_presence_component * non_presence_component;

        let low_rms = self.low_band_env_sq.sqrt();
        let voiced_rms = self.voiced_band_env_sq.sqrt().max(1e-8);
        let presence_rms = self.presence_band_env_sq.sqrt();
        let non_presence_rms = self.non_presence_band_env_sq.sqrt().max(1e-8);
        self.plosive_ratio = (low_rms / voiced_rms).clamp(0.0, 32.0);

        let plosive_amount = ((self.plosive_ratio - PLOSIVE_RATIO_START)
            / (PLOSIVE_RATIO_FULL - PLOSIVE_RATIO_START))
            .clamp(0.0, 1.0);
        let plosive_penalty = 1.0 - plosive_amount * (1.0 - PLOSIVE_MIN_DETECTOR_GAIN);
        let presence_ratio = (presence_rms / non_presence_rms).clamp(0.0, 4.0);
        let presence_weight = 1.0 + 0.18 * (presence_ratio - 0.75).clamp(0.0, 1.0);
        (plosive_penalty * presence_weight).clamp(PLOSIVE_MIN_DETECTOR_GAIN, 1.15)
    }

    fn update_adaptive_release_time_meter(&mut self) {
        if !self.adaptive_release {
            self.target_release_ms = self.base_release_ms;
            return;
        }

        let sustained =
            (self.slow_release_env_db / (SLOW_RELEASE_TRIGGER_DB + 3.0)).clamp(0.0, 1.0);
        let transient_bias = ((self.fast_release_env_db - self.slow_release_env_db)
            / (SLOW_RELEASE_TRIGGER_DB + 4.0))
            .clamp(0.0, 1.0);
        let syllabic = (sustained * sustained * (1.0 - 0.35 * transient_bias)).clamp(0.0, 1.0);
        let minimum_release_ms = self.base_release_ms.max(0.001);
        let maximum_release_ms = (minimum_release_ms * ADAPTIVE_SLOW_RELEASE_MULTIPLIER)
            .min(ADAPTIVE_RELEASE_MAX_MS)
            .max(minimum_release_ms);
        self.target_release_ms =
            minimum_release_ms + syllabic * (maximum_release_ms - minimum_release_ms);
    }

    fn smooth_gain_reduction(&mut self, target_gain_reduction_db: f64) {
        if !self.adaptive_release {
            let gr_coeff = if target_gain_reduction_db > self.current_gain_reduction_db {
                self.attack_coeff
            } else {
                self.release_coeff
            };
            self.current_gain_reduction_db = gr_coeff * self.current_gain_reduction_db
                + (1.0 - gr_coeff) * target_gain_reduction_db;
            self.fast_release_env_db = self.current_gain_reduction_db;
            self.slow_release_env_db = 0.0;
            return;
        }

        if target_gain_reduction_db > self.current_gain_reduction_db {
            self.fast_release_env_db = self.attack_coeff * self.current_gain_reduction_db
                + (1.0 - self.attack_coeff) * target_gain_reduction_db;
        } else {
            self.fast_release_env_db = self.adaptive_fast_release_coeff * self.fast_release_env_db
                + (1.0 - self.adaptive_fast_release_coeff) * target_gain_reduction_db;
        }

        if target_gain_reduction_db > SLOW_RELEASE_TRIGGER_DB {
            self.slow_release_env_db = self.adaptive_slow_charge_coeff * self.slow_release_env_db
                + (1.0 - self.adaptive_slow_charge_coeff) * target_gain_reduction_db;
        } else {
            self.slow_release_env_db *= self.adaptive_slow_release_coeff;
        }

        let gr_coeff = if target_gain_reduction_db > self.current_gain_reduction_db {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.current_gain_reduction_db =
            gr_coeff * self.current_gain_reduction_db + (1.0 - gr_coeff) * target_gain_reduction_db;
    }

    fn speech_activity_from_rms_db(rms_db: f64) -> f64 {
        if !(SPEECH_ACTIVE_RMS_MIN_DB..=SPEECH_ACTIVE_RMS_MAX_DB).contains(&rms_db) {
            return 0.0;
        }
        let onset = ((rms_db - SPEECH_ACTIVE_RMS_MIN_DB) / 12.0).clamp(0.0, 1.0);
        let overload = ((SPEECH_ACTIVE_RMS_MAX_DB - rms_db) / 6.0).clamp(0.0, 1.0);
        onset.min(overload)
    }

    fn finite_unit(value: f64) -> Option<f64> {
        value.is_finite().then(|| value.clamp(0.0, 1.0))
    }

    fn smoothstep(edge0: f64, edge1: f64, value: f64) -> f64 {
        if !value.is_finite() || !edge0.is_finite() || !edge1.is_finite() || edge1 <= edge0 {
            return 0.0;
        }
        let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    fn estimate_auto_makeup_activity(
        &self,
        rms_db: f64,
        evidence: Option<AutoMakeupActivityInput>,
    ) -> AutoMakeupActivityEstimate {
        let absolute_activity = Self::speech_activity_from_rms_db(rms_db);
        let Some(evidence) = evidence else {
            return AutoMakeupActivityEstimate {
                activity: absolute_activity,
                reliability: 1.0,
            };
        };

        let mut vad_reliability = Self::finite_unit(evidence.vad_reliability).unwrap_or(0.0);
        let vad_probability = match Self::finite_unit(evidence.vad_probability) {
            Some(probability) => probability,
            None => {
                vad_reliability = 0.0;
                0.0
            }
        };
        let configured_noise_reliability =
            Self::finite_unit(self.noise_reference_reliability).unwrap_or(0.0);
        let live_noise_reliability =
            Self::finite_unit(evidence.live_noise_reliability).unwrap_or(0.0);
        let mut noise_reliability = if configured_noise_reliability > 0.0 {
            live_noise_reliability.min(configured_noise_reliability)
        } else {
            live_noise_reliability
        };
        let relative_activity = if evidence.noise_floor_db.is_finite()
            && (-120.0..=0.0).contains(&evidence.noise_floor_db)
        {
            Self::smoothstep(
                evidence.noise_floor_db + NOISE_RELATIVE_ACTIVITY_START_DB,
                evidence.noise_floor_db + NOISE_RELATIVE_ACTIVITY_FULL_DB,
                rms_db,
            )
        } else {
            noise_reliability = 0.0;
            0.0
        };

        let fallback_activity =
            noise_reliability * relative_activity + (1.0 - noise_reliability) * absolute_activity;
        let activity =
            vad_reliability * vad_probability + (1.0 - vad_reliability) * fallback_activity;
        let reliability = vad_reliability.max(0.75 * noise_reliability);

        AutoMakeupActivityEstimate {
            activity: activity.clamp(0.0, 1.0),
            reliability: reliability.clamp(0.0, 1.0),
        }
    }

    fn block_rms_db(buffer: &[f32]) -> f64 {
        if buffer.is_empty() {
            return -120.0;
        }
        let power = buffer
            .iter()
            .map(|sample| {
                let sample = *sample as f64;
                sample * sample
            })
            .sum::<f64>()
            / buffer.len() as f64;
        util::linear_to_db(power.sqrt(), 1e-10)
    }

    fn update_auto_makeup_gain(
        &mut self,
        speech_activity: f64,
        reliability: f64,
        elapsed_samples: usize,
    ) {
        let elapsed_samples = elapsed_samples.max(1);
        let makeup_coeff = if elapsed_samples == 1 {
            self.makeup_smoothing_coeff
        } else {
            self.makeup_smoothing_coeff.powf(elapsed_samples as f64)
        };
        let silence_relax_coeff = if elapsed_samples == 1 {
            self.makeup_silence_relax_coeff
        } else {
            self.makeup_silence_relax_coeff.powf(elapsed_samples as f64)
        };
        if !self.auto_makeup_enabled {
            let target = self.makeup_gain_db;
            self.smoothed_makeup_gain =
                makeup_coeff * self.smoothed_makeup_gain + (1.0 - makeup_coeff) * target;
            return;
        }

        if let Some(meter) = &self.loudness_meter {
            self.current_lufs = meter.loudness_momentary() as f64;
            let activity_coeff = if elapsed_samples == 1 {
                self.speech_activity_smoothing_coeff
            } else {
                self.speech_activity_smoothing_coeff
                    .powf(elapsed_samples as f64)
            };
            self.speech_activity_score = activity_coeff * self.speech_activity_score
                + (1.0 - activity_coeff) * speech_activity.clamp(0.0, 1.0);
            self.auto_makeup_activity_reliability = reliability.clamp(0.0, 1.0);
            if self.speech_activity_score < AUTO_MAKEUP_ACTIVE_MIN {
                self.smoothed_makeup_gain = silence_relax_coeff * self.smoothed_makeup_gain
                    + (1.0 - silence_relax_coeff) * self.makeup_gain_db;
                return;
            }
            if self.auto_makeup_activity_reliability < AUTO_MAKEUP_RELIABILITY_MIN {
                let conservative_cap = self.makeup_gain_db
                    + 3.0 * (self.auto_makeup_activity_reliability / AUTO_MAKEUP_RELIABILITY_MIN);
                if self.smoothed_makeup_gain > conservative_cap {
                    self.smoothed_makeup_gain = makeup_coeff * self.smoothed_makeup_gain
                        + (1.0 - makeup_coeff) * conservative_cap;
                }
                return;
            }
            // The meter receives the compressor output after the currently
            // applied makeup gain. Remove that gain before calculating the
            // correction, otherwise feedback settles below the requested
            // target by approximately the applied makeup amount.
            let pre_makeup_lufs = self.current_lufs - self.smoothed_makeup_gain;
            let required_gain = self.target_lufs - pre_makeup_lufs;
            let reliability_cap = (12.0 * self.auto_makeup_activity_reliability).clamp(3.0, 12.0);
            let headroom_cap =
                (12.0 - self.limiter_feedback_gain_reduction_db * 2.0).clamp(0.0, reliability_cap);
            let clamped_gain = required_gain.clamp(0.0, headroom_cap);

            self.smoothed_makeup_gain =
                makeup_coeff * self.smoothed_makeup_gain + (1.0 - makeup_coeff) * clamped_gain;
        }
    }

    /// Calculate gain reduction in dB for a given detector level.
    #[inline]
    fn compute_gain_reduction(&self, detector_db: f64) -> f64 {
        let comp_factor = 1.0 - 1.0 / self.ratio;
        if self.knee_db <= 0.0 {
            if detector_db <= self.threshold_db {
                return 0.0;
            }
            return (detector_db - self.threshold_db) * comp_factor;
        }

        let knee_half = self.knee_db / 2.0;
        let knee_start = self.threshold_db - knee_half;
        let knee_end = self.threshold_db + knee_half;

        if detector_db <= knee_start {
            0.0
        } else if detector_db >= knee_end {
            (detector_db - self.threshold_db) * comp_factor
        } else {
            let x = detector_db - knee_start;
            comp_factor * x * x / (2.0 * self.knee_db)
        }
    }

    #[inline]
    fn blended_detector_db(peak_db: f64, rms_db: f64) -> f64 {
        let peak_lin = util::db_to_linear(peak_db);
        let rms_lin = util::db_to_linear(rms_db);
        let blended = DETECTOR_PEAK_WEIGHT * peak_lin + DETECTOR_RMS_WEIGHT * rms_lin;
        util::linear_to_db(blended, 1e-10)
    }

    /// Process a single sample
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        self.block_peak_gain_reduction_db = 0.0;
        self.process_sample_impl(input, true)
    }

    /// Process a block of samples in-place
    pub fn process_block_inplace(&mut self, buffer: &mut [f32]) {
        self.process_block_inplace_with_activity_control(buffer, None);
    }

    /// Process a block using shared VAD/noise evidence for auto-makeup only.
    pub fn process_block_inplace_with_activity_control(
        &mut self,
        buffer: &mut [f32],
        evidence: Option<AutoMakeupActivityInput>,
    ) {
        self.block_peak_gain_reduction_db = 0.0;
        // A caller switching from the sample API must not mix an incomplete
        // sample window into this block's loudness measurement.
        self.clear_auto_makeup_sample_window();
        if !self.is_active() {
            self.current_gain_reduction_db = 0.0;
            return;
        }

        let activity = self.estimate_auto_makeup_activity(Self::block_rms_db(buffer), evidence);
        for sample in buffer.iter_mut() {
            *sample = self.process_sample_impl(*sample, false);
        }
        if !self.enabled {
            // A fade-out block is partly dry; keep it out of loudness control.
            return;
        }
        if activity.activity > AUTO_MAKEUP_ACTIVE_MIN
            && activity.reliability >= AUTO_MAKEUP_RELIABILITY_MIN
        {
            if let Some(meter) = &mut self.loudness_meter {
                meter.process(buffer);
            }
        }
        if self.auto_makeup_enabled {
            self.update_auto_makeup_gain(activity.activity, activity.reliability, buffer.len());
        }
    }

    #[inline]
    fn clear_auto_makeup_sample_window(&mut self) {
        self.auto_makeup_sample_window_len = 0;
        self.auto_makeup_sample_activity_sum = 0.0;
    }

    #[inline]
    fn process_sample_impl(&mut self, input: f32, collect_sample_auto_makeup: bool) -> f32 {
        if !self.is_active() {
            self.current_gain_reduction_db = 0.0;
            return input;
        }

        let input_f64 = input as f64;
        let detector_input = self.process_sidechain_sample(input_f64);
        let detector_weight = self.update_sidechain_band_metrics(input_f64, detector_input);
        self.update_peak_envelope(detector_input.abs());

        let input_squared = detector_input * detector_input;
        self.rms_envelope_sq =
            self.rms_coeff * self.rms_envelope_sq + (1.0 - self.rms_coeff) * input_squared;
        let rms_db = util::linear_to_db(self.rms_envelope_sq.sqrt(), 1e-10);

        let peak_db = util::linear_to_db(self.peak_envelope, 1e-10);
        let detector_db =
            Self::blended_detector_db(peak_db, rms_db) + util::linear_to_db(detector_weight, 1e-10);

        self.update_adaptive_release_time_meter();
        let release_diff = self.target_release_ms - self.current_release_ms;
        if release_diff.abs() > 1.0 {
            self.current_release_ms = self.release_smoothing_coeff * self.current_release_ms
                + (1.0 - self.release_smoothing_coeff) * self.target_release_ms;
        } else {
            self.current_release_ms = self.target_release_ms;
        }
        if self.adaptive_release {
            if self.adaptive_release_coeff_samples_until_update <= 1 {
                self.release_coeff =
                    util::time_constant_to_coeff(self.current_release_ms, self.sample_rate);
                self.adaptive_release_coeff_samples_until_update =
                    self.adaptive_release_coeff_update_samples;
            } else {
                self.adaptive_release_coeff_samples_until_update -= 1;
            }
        }

        let target_gain_reduction_db = self.compute_gain_reduction(detector_db);
        self.smooth_gain_reduction(target_gain_reduction_db);
        self.block_peak_gain_reduction_db = self
            .block_peak_gain_reduction_db
            .max(self.current_gain_reduction_db);

        if !self.auto_makeup_enabled {
            let speech_activity = Self::speech_activity_from_rms_db(detector_db);
            self.update_auto_makeup_gain(speech_activity, 1.0, 1);
        }

        let mut output_gain = util::db_to_linear(-self.current_gain_reduction_db)
            * util::db_to_linear(self.smoothed_makeup_gain);
        let ramp_target = if self.enabled {
            self.enable_ramp_samples
        } else {
            0
        };
        if self.enable_ramp_position != ramp_target {
            if self.enabled {
                self.enable_ramp_position += 1;
            } else {
                self.enable_ramp_position -= 1;
            }
            let mix = self.enable_ramp_position as f64 / self.enable_ramp_samples as f64;
            output_gain = 1.0 + mix * (output_gain - 1.0);
        }
        let output = (input_f64 * output_gain) as f32;

        if collect_sample_auto_makeup && self.auto_makeup_enabled && self.enabled {
            let speech_activity = Self::speech_activity_from_rms_db(detector_db);
            let sample_index = self.auto_makeup_sample_window_len;
            self.auto_makeup_sample_window[sample_index] = output;
            self.auto_makeup_sample_activity_sum += speech_activity;
            self.auto_makeup_sample_window_len += 1;

            if self.auto_makeup_sample_window_len >= self.auto_makeup_sample_window_samples {
                let window_len = self.auto_makeup_sample_window_len;
                if let Some(meter) = &mut self.loudness_meter {
                    meter.process(&self.auto_makeup_sample_window[..window_len]);
                }
                let activity = self.auto_makeup_sample_activity_sum / window_len as f64;
                self.clear_auto_makeup_sample_window();
                self.update_auto_makeup_gain(activity, 1.0, window_len);
            }
        }

        output
    }

    /// Reset compressor state
    pub fn reset(&mut self) {
        self.reset_detector_state();
        self.finish_enable_transition();
        self.current_release_ms = self.base_release_ms;
        self.target_release_ms = self.base_release_ms;
        self.release_coeff =
            util::time_constant_to_coeff(self.current_release_ms, self.sample_rate);
        self.adaptive_release_coeff_samples_until_update = 1;
        self.limiter_feedback_gain_reduction_db = 0.0;
        self.speech_activity_score = 0.0;
        self.auto_makeup_activity_reliability = 0.0;
        self.clear_auto_makeup_sample_window();
        if let Some(meter) = &mut self.loudness_meter {
            if meter.reset().is_ok() {
                self.current_lufs = -100.0;
            }
        } else {
            self.current_lufs = -100.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressor_no_compression_below_threshold() {
        let mut comp = Compressor::new(-20.0, 4.0, 10.0, 200.0, 0.0, 0.0, 48_000.0);
        let input = 0.001f32;
        let output = comp.process_sample(input);
        assert!((output - input).abs() < 0.0001);
    }

    #[test]
    #[ignore = "release-mode compressor hot-path cost measurement"]
    fn benchmark_compressor_process_sample_cost() {
        const SAMPLES: usize = 480_000;
        const REPEATS: usize = 5;
        let input: Vec<f32> = (0..SAMPLES)
            .map(|index| {
                let phase = 2.0 * std::f64::consts::PI * 187.0 * index as f64 / 48_000.0;
                (0.3 * phase.sin()) as f32
            })
            .collect();

        for adaptive_release in [false, true] {
            for sample_rate in [44_100.0, 48_000.0, 96_000.0] {
                let mut durations = [std::time::Duration::ZERO; REPEATS];
                for duration in &mut durations {
                    let mut compressor =
                        Compressor::new(-24.0, 3.0, 10.0, 180.0, 0.0, 6.0, sample_rate);
                    compressor.set_adaptive_release(adaptive_release);
                    for sample in input.iter().take(10_000) {
                        std::hint::black_box(
                            compressor.process_sample(std::hint::black_box(*sample)),
                        );
                    }
                    let started = std::time::Instant::now();
                    for sample in &input {
                        std::hint::black_box(
                            compressor.process_sample(std::hint::black_box(*sample)),
                        );
                    }
                    *duration = started.elapsed();
                }
                durations.sort_unstable();
                println!(
                    "compressor {sample_rate:.0} Hz adaptive={adaptive_release}: {:.2} ns/sample (median)",
                    durations[REPEATS / 2].as_nanos() as f64 / SAMPLES as f64
                );
            }
        }
    }

    #[test]
    fn test_sample_path_remains_finite_across_control_transitions() {
        let mut compressor = Compressor::new(-27.0, 4.5, 4.0, 110.0, 1.25, 7.0, 48_000.0);
        compressor.set_sidechain_highpass_enabled(true);

        for index in 0..4_096 {
            match index {
                1_024 => {
                    compressor.set_attack_time(1.5);
                    compressor.set_release_time(80.0);
                    compressor.set_adaptive_release(true);
                }
                2_048 => compressor.set_base_release_time(240.0),
                3_072 => compressor.set_adaptive_release(false),
                3_584 => compressor.set_release_time(65.0),
                _ => {}
            }

            let amplitude = match index {
                0..=511 => 0.03,
                512..=1_535 => 0.65,
                1_536..=2_047 => 0.12,
                2_048..=3_071 => 0.42,
                _ => 0.08,
            };
            let phase = 2.0 * std::f64::consts::PI * 223.0 * index as f64 / 48_000.0;
            let sample = (amplitude * (phase.sin() + 0.17 * (phase * 5.1).sin())) as f32;
            let output = compressor.process_sample(sample);
            assert!(output.is_finite());
            assert!(output.abs() <= sample.abs() * 1.2 + 1e-6);
        }

        assert!((compressor.current_release_time() - 65.0).abs() < 1e-12);
        assert!(compressor.current_gain_reduction().is_finite());
        assert!((0.0..=32.0).contains(&compressor.plosive_ratio()));
    }

    #[test]
    fn test_compressor_reduces_gain_above_threshold() {
        let mut comp = Compressor::new(-20.0, 4.0, 0.1, 200.0, 0.0, 0.0, 48_000.0);
        let loud_signal = vec![0.3f32; 5_000];
        for sample in &loud_signal {
            comp.process_sample(*sample);
        }
        assert!(comp.current_gain_reduction() > 0.0);
    }

    #[test]
    fn test_compressor_makeup_gain() {
        let mut comp = Compressor::new(-20.0, 4.0, 10.0, 200.0, 6.0, 0.0, 48_000.0);
        let input = 0.001f32;
        for _ in 0..1000 {
            comp.process_sample(input);
        }
        let output = comp.process_sample(input);
        assert!(output > input * 1.5);
    }

    #[test]
    fn test_compressor_disabled() {
        let mut comp = Compressor::new(-20.0, 4.0, 10.0, 200.0, 6.0, 0.0, 48_000.0);
        comp.set_enabled(false);
        let input = 0.5f32;
        for _ in 0..480 {
            comp.process_sample(input);
        }
        assert!(!comp.is_active());
        assert_eq!(comp.process_sample(input), input);
    }

    #[test]
    fn test_compressor_toggle_ramps_active_reduction_and_makeup() {
        let sample_rate = 48_000.0;
        let mut comp = Compressor::new(-30.0, 8.0, 20.0, 100.0, 9.0, 0.0, sample_rate);
        let signal = |n: usize| {
            (0.8 * (2.0 * std::f64::consts::PI * 150.0 * n as f64 / sample_rate).sin()) as f32
        };
        let mut previous_gain: Option<f64> = None;
        // Largest per-sample gain change in steady compression, around the
        // disable, and around the re-enable.
        let mut max_step_db = [0.0_f64; 3];
        for n in 0..48_000 {
            match n {
                12_000 => {
                    assert!(comp.current_gain_reduction() > 6.0);
                    comp.set_enabled(false);
                }
                30_000 => comp.set_enabled(true),
                _ => {}
            }
            let input = signal(n);
            let output = comp.process_sample(input);
            if input.abs() < 0.05 {
                previous_gain = None;
                continue;
            }
            let gain_db = 20.0 * (output / input).abs().log10() as f64;
            let region = match n {
                6_000..=11_999 => Some(0),
                12_000..=12_999 => Some(1),
                30_000..=30_999 => Some(2),
                _ => None,
            };
            if let (Some(region), Some(previous)) = (region, previous_gain) {
                max_step_db[region] = max_step_db[region].max((gain_db - previous).abs());
            }
            previous_gain = Some(gain_db);
        }
        // Steady compression moves ~0.0004 dB per sample; a 10 ms ramp across
        // the ~16 dB processed/unity swing stays below 0.1 dB per sample,
        // where an instantaneous toggle would step by the whole swing.
        assert!(max_step_db[0] < 0.01, "steady {:.4} dB", max_step_db[0]);
        assert!(max_step_db[1] < 0.25, "disable {:.4} dB", max_step_db[1]);
        assert!(max_step_db[2] < 0.25, "enable {:.4} dB", max_step_db[2]);
    }

    #[test]
    fn test_soft_knee() {
        let comp_hard = Compressor::new(-20.0, 4.0, 10.0, 200.0, 0.0, 0.0, 48_000.0);
        let comp_soft = Compressor::new(-20.0, 4.0, 10.0, 200.0, 0.0, 12.0, 48_000.0);

        // Inside the knee but below threshold, soft knee should start compressing
        // while hard knee still applies no gain reduction.
        let at_minus_22_hard = comp_hard.compute_gain_reduction(-22.0);
        let at_minus_22_soft = comp_soft.compute_gain_reduction(-22.0);
        assert!((at_minus_22_hard - 0.0).abs() < 1e-12);
        assert!(at_minus_22_soft > 0.0);

        let well_above_hard = comp_hard.compute_gain_reduction(-5.0);
        let well_above_soft = comp_soft.compute_gain_reduction(-5.0);
        assert!((well_above_hard - well_above_soft).abs() < 0.5);
    }

    #[test]
    fn test_soft_knee_exact_boundaries() {
        let comp = Compressor::new(-20.0, 4.0, 10.0, 200.0, 0.0, 12.0, 48_000.0);
        let w = 12.0;
        let t = -20.0;
        let comp_factor = 1.0 - 1.0 / 4.0;

        let at_knee_start = comp.compute_gain_reduction(t - w / 2.0);
        let at_knee_end = comp.compute_gain_reduction(t + w / 2.0);

        assert!((at_knee_start - 0.0).abs() < 1e-12);
        assert!((at_knee_end - ((w / 2.0) * comp_factor)).abs() < 1e-12);
    }

    #[test]
    fn test_detector_blend_uses_linear_domain() {
        let detector = Compressor::blended_detector_db(-6.0, -18.0);

        assert!(detector < -6.0);
        assert!(detector > -18.0);
        assert!((Compressor::blended_detector_db(-12.0, -12.0) + 12.0).abs() < 1e-9);
        assert!(Compressor::blended_detector_db(-160.0, -160.0).is_finite());
    }

    fn process_sine(
        compressor: &mut Compressor,
        frequency_hz: f64,
        amplitude: f32,
        samples: usize,
    ) {
        for index in 0..samples {
            let phase = 2.0 * std::f64::consts::PI * frequency_hz * index as f64 / 48_000.0;
            compressor.process_sample((phase.sin() as f32) * amplitude);
        }
    }

    #[test]
    fn test_sidechain_highpass_disabled_preserves_existing_detection() {
        let mut default_off = Compressor::new(-28.0, 6.0, 0.1, 120.0, 0.0, 0.0, 48_000.0);
        let mut explicitly_off = Compressor::new(-28.0, 6.0, 0.1, 120.0, 0.0, 0.0, 48_000.0);
        explicitly_off.set_sidechain_highpass_enabled(false);

        for index in 0..24_000 {
            let phase = 2.0 * std::f64::consts::PI * 55.0 * index as f64 / 48_000.0;
            let sample = (phase.sin() as f32) * 0.65;
            let default_output = default_off.process_sample(sample);
            let explicit_output = explicitly_off.process_sample(sample);
            assert!((default_output - explicit_output).abs() < 1e-12);
        }

        assert!(
            (default_off.current_gain_reduction() - explicitly_off.current_gain_reduction()).abs()
                < 1e-12
        );
    }

    #[test]
    fn test_sidechain_highpass_reduces_plosive_driven_gain_reduction() {
        let mut full_band = Compressor::new(-30.0, 8.0, 0.1, 180.0, 0.0, 0.0, 48_000.0);
        let mut highpassed = Compressor::new(-30.0, 8.0, 0.1, 180.0, 0.0, 0.0, 48_000.0);
        highpassed.set_sidechain_highpass_enabled(true);

        process_sine(&mut full_band, 55.0, 0.7, 48_000);
        process_sine(&mut highpassed, 55.0, 0.7, 48_000);

        assert!(
            highpassed.current_gain_reduction() + 2.0 < full_band.current_gain_reduction(),
            "highpassed={} full_band={}",
            highpassed.current_gain_reduction(),
            full_band.current_gain_reduction()
        );
    }

    #[test]
    fn test_plosive_ratio_tracks_low_band_bursts() {
        let mut comp = Compressor::new(-30.0, 8.0, 0.1, 180.0, 0.0, 0.0, 48_000.0);
        comp.set_sidechain_highpass_enabled(true);

        process_sine(&mut comp, 55.0, 0.7, 12_000);
        let plosive_ratio = comp.plosive_ratio();

        assert!(
            plosive_ratio > 1.5,
            "low-band burst should raise plosive ratio, got {plosive_ratio}"
        );
    }

    #[test]
    fn test_sidechain_highpass_preserves_speech_band_compression() {
        let mut full_band = Compressor::new(-30.0, 8.0, 0.1, 180.0, 0.0, 0.0, 48_000.0);
        let mut highpassed = Compressor::new(-30.0, 8.0, 0.1, 180.0, 0.0, 0.0, 48_000.0);
        highpassed.set_sidechain_highpass_enabled(true);

        process_sine(&mut full_band, 1_000.0, 0.3, 48_000);
        process_sine(&mut highpassed, 1_000.0, 0.3, 48_000);

        assert!(highpassed.current_gain_reduction() > 1.0);
        assert!(
            highpassed.current_gain_reduction() > full_band.current_gain_reduction() * 0.8,
            "highpassed={} full_band={}",
            highpassed.current_gain_reduction(),
            full_band.current_gain_reduction()
        );
    }

    #[test]
    fn test_adaptive_release_enables() {
        let mut comp = Compressor::new(-20.0, 4.0, 10.0, 50.0, 0.0, 0.0, 48_000.0);
        comp.set_adaptive_release(true);
        assert!(comp.adaptive_release());

        let loud_signal = vec![0.3f32; 96_000];
        for sample in &loud_signal {
            comp.process_sample(*sample);
        }
        let current_release = comp.current_release_time();
        assert!(current_release > 300.0);
    }

    #[test]
    fn test_release_time_changes_recovery_speed() {
        let mut fast = Compressor::new(-20.0, 4.0, 0.1, 20.0, 0.0, 0.0, 48_000.0);
        let mut slow = Compressor::new(-20.0, 4.0, 0.1, 200.0, 0.0, 0.0, 48_000.0);

        for _ in 0..4_000 {
            fast.process_sample(0.5);
            slow.process_sample(0.5);
        }

        for _ in 0..1_440 {
            fast.process_sample(0.001);
            slow.process_sample(0.001);
        }

        assert!(fast.current_gain_reduction() < slow.current_gain_reduction());
    }

    #[test]
    fn test_adaptive_release_does_not_change_detector_release_decay() {
        let mut fixed = Compressor::new(-20.0, 4.0, 1.0, 80.0, 0.0, 0.0, 48_000.0);
        let mut adaptive = Compressor::new(-20.0, 4.0, 1.0, 80.0, 0.0, 0.0, 48_000.0);
        adaptive.set_adaptive_release(true);

        for _ in 0..96_000 {
            fixed.process_sample(0.4);
            adaptive.process_sample(0.4);
        }

        assert!(adaptive.current_release_time() > fixed.current_release_time());

        let fixed_peak_before = fixed.peak_envelope;
        let adaptive_peak_before = adaptive.peak_envelope;
        for _ in 0..2_400 {
            fixed.process_sample(0.001);
            adaptive.process_sample(0.001);
        }

        let fixed_drop = fixed_peak_before - fixed.peak_envelope;
        let adaptive_drop = adaptive_peak_before - adaptive.peak_envelope;
        assert!((fixed_drop - adaptive_drop).abs() < 1e-9);
    }

    #[test]
    fn test_adaptive_release_slow_envelope_ignores_light_compression() {
        let mut comp = Compressor::new(-20.0, 4.0, 1.0, 80.0, 0.0, 0.0, 48_000.0);
        comp.set_adaptive_release(true);

        for _ in 0..4_800 {
            comp.smooth_gain_reduction(SLOW_RELEASE_TRIGGER_DB - 0.5);
        }

        assert!(comp.slow_release_env_db < 0.1);
        assert!(comp.current_gain_reduction() > 0.0);
    }

    #[test]
    fn test_adaptive_release_slow_envelope_charges_on_deep_compression() {
        let mut comp = Compressor::new(-20.0, 4.0, 1.0, 80.0, 0.0, 0.0, 48_000.0);
        comp.set_adaptive_release(true);

        for _ in 0..24_000 {
            comp.smooth_gain_reduction(SLOW_RELEASE_TRIGGER_DB + 4.0);
        }

        assert!(comp.slow_release_env_db > SLOW_RELEASE_TRIGGER_DB);

        let held = comp.current_gain_reduction();
        for _ in 0..2_400 {
            comp.smooth_gain_reduction(0.0);
        }

        assert!(comp.current_gain_reduction() > 0.0);
        assert!(comp.current_gain_reduction() < held);
    }

    #[test]
    fn test_continuous_adaptive_release_maps_transient_faster_than_sustained() {
        let mut transient = Compressor::new(-20.0, 4.0, 1.0, 80.0, 0.0, 0.0, 48_000.0);
        transient.set_adaptive_release(true);
        transient.fast_release_env_db = 6.0;
        transient.slow_release_env_db = 0.5;
        transient.update_adaptive_release_time_meter();

        let mut sustained = Compressor::new(-20.0, 4.0, 1.0, 80.0, 0.0, 0.0, 48_000.0);
        sustained.set_adaptive_release(true);
        sustained.fast_release_env_db = 6.0;
        sustained.slow_release_env_db = 6.0;
        sustained.update_adaptive_release_time_meter();

        assert!(transient.target_release_ms < sustained.target_release_ms);
        assert!(sustained.target_release_ms > ADAPTIVE_FAST_RELEASE_MS);
    }

    #[test]
    fn test_auto_makeup_activity_smoothing_is_block_size_invariant() {
        fn activity_after_one_second(block_size: usize) -> f64 {
            let mut compressor = Compressor::default_voice(48_000.0);
            compressor.set_auto_makeup_enabled(true);
            let mut remaining = 48_000;
            while remaining > 0 {
                let elapsed = remaining.min(block_size);
                compressor.update_auto_makeup_gain(1.0, 1.0, elapsed);
                remaining -= elapsed;
            }
            compressor.auto_makeup_activity()
        }

        let reference = activity_after_one_second(480);
        for block_size in [1, 48, 240, 960, 4_096, 48_000] {
            let candidate = activity_after_one_second(block_size);
            assert!(
                (candidate - reference).abs() < 1e-10,
                "block size {block_size} changed activity from {reference} to {candidate}"
            );
        }
    }

    #[test]
    fn test_auto_makeup_does_not_rise_during_silence() {
        let mut comp = Compressor::default_voice(48_000.0);
        comp.set_auto_makeup_enabled(true);
        comp.set_target_lufs(-12.0);

        let mut silence = vec![0.0_f32; 48_000];
        for _ in 0..4 {
            comp.process_block_inplace(&mut silence);
        }

        assert!(comp.current_makeup_gain() < 0.5);
    }

    #[test]
    fn test_auto_makeup_follows_speech_like_blocks() {
        let mut comp = Compressor::default_voice(48_000.0);
        comp.set_auto_makeup_enabled(true);
        comp.set_target_lufs(-12.0);

        let mut speech_like = vec![0.04_f32; 48_000];
        for _ in 0..10 {
            comp.process_block_inplace(&mut speech_like);
        }

        assert!(comp.current_makeup_gain() > 0.1);
    }

    #[test]
    fn test_auto_makeup_sample_api_updates_loudness_meter() {
        for sample_rate in [44_100.0, 48_000.0, 96_000.0] {
            let mut compressor = Compressor::new(0.0, 1.0, 0.1, 200.0, 0.0, 0.0, sample_rate);
            compressor.set_auto_makeup_enabled(true);
            compressor.set_target_lufs(-14.0);

            let amplitude = 10.0_f32.powf(-24.0 / 20.0) * 2.0_f32.sqrt();
            let sample_count = sample_rate as usize;
            for sample_index in 0..sample_count {
                let phase =
                    2.0 * std::f32::consts::PI * 1_000.0 * sample_index as f32 / sample_rate as f32;
                compressor.process_sample(amplitude * phase.sin());
            }

            assert!(
                compressor.current_lufs() > -90.0,
                "sample processing must feed the auto-makeup loudness meter at {sample_rate} Hz: {:.2} LUFS",
                compressor.current_lufs()
            );
        }
    }

    #[test]
    fn test_auto_makeup_sample_and_block_paths_converge_to_same_level() {
        for sample_rate in [44_100.0, 48_000.0, 96_000.0] {
            let mut sample_compressor =
                Compressor::new(0.0, 1.0, 0.1, 200.0, 0.0, 0.0, sample_rate);
            sample_compressor.set_auto_makeup_enabled(true);
            sample_compressor.set_target_lufs(-14.0);

            let mut block_compressor = Compressor::new(0.0, 1.0, 0.1, 200.0, 0.0, 0.0, sample_rate);
            block_compressor.set_auto_makeup_enabled(true);
            block_compressor.set_target_lufs(-14.0);

            let amplitude = 10.0_f32.powf(-24.0 / 20.0) * 2.0_f32.sqrt();
            let sample_count = sample_rate as usize * 4;
            for sample_index in 0..sample_count {
                let phase =
                    2.0 * std::f32::consts::PI * 1_000.0 * sample_index as f32 / sample_rate as f32;
                sample_compressor.process_sample(amplitude * phase.sin());
            }

            let block_size =
                (sample_rate * AUTO_MAKEUP_SAMPLE_WINDOW_MS / 1_000.0).round() as usize;
            let mut block = vec![0.0_f32; block_size];
            for block_index in 0..(sample_count / block_size) {
                for (sample_index, sample) in block.iter_mut().enumerate() {
                    let absolute_index = block_index * block_size + sample_index;
                    let phase = 2.0 * std::f32::consts::PI * 1_000.0 * absolute_index as f32
                        / sample_rate as f32;
                    *sample = amplitude * phase.sin();
                }
                block_compressor.process_block_inplace(&mut block);
            }

            assert!(
                (sample_compressor.current_lufs() - block_compressor.current_lufs()).abs() < 1.0,
                "sample/block meter mismatch at {sample_rate} Hz: sample={:.2} block={:.2}",
                sample_compressor.current_lufs(),
                block_compressor.current_lufs()
            );
            assert!(
                (sample_compressor.current_makeup_gain() - block_compressor.current_makeup_gain()).abs()
                    < 1.0,
                "sample/block auto-makeup mismatch at {sample_rate} Hz: sample={:.2} block={:.2}, lufs={:.2}/{:.2}, activity={:.3}/{:.3}",
                sample_compressor.current_makeup_gain(),
                block_compressor.current_makeup_gain(),
                sample_compressor.current_lufs(),
                block_compressor.current_lufs(),
                sample_compressor.auto_makeup_activity(),
                block_compressor.auto_makeup_activity()
            );
        }
    }

    #[test]
    fn test_auto_makeup_sample_window_covers_generic_rates() {
        for (sample_rate, expected_window) in [
            (44_100.0, 441),
            (48_000.0, 480),
            (96_000.0, 960),
            (192_000.0, 1_920),
        ] {
            let compressor = Compressor::new(0.0, 1.0, 0.1, 200.0, 0.0, 0.0, sample_rate);
            assert_eq!(
                compressor.auto_makeup_sample_window_samples, expected_window,
                "sample API auto-makeup window at {sample_rate} Hz"
            );
        }
    }

    #[test]
    fn test_auto_makeup_sample_window_resets_when_api_or_state_changes() {
        let mut compressor = Compressor::new(0.0, 1.0, 0.1, 200.0, 0.0, 0.0, 48_000.0);
        compressor.set_auto_makeup_enabled(true);
        compressor.process_sample(0.1);
        assert_eq!(compressor.auto_makeup_sample_window_len, 1);

        let mut block = vec![0.0_f32; 480];
        compressor.process_block_inplace(&mut block);
        assert_eq!(compressor.auto_makeup_sample_window_len, 0);

        compressor.process_sample(0.1);
        assert_eq!(compressor.auto_makeup_sample_window_len, 1);
        compressor.reset();
        assert_eq!(compressor.auto_makeup_sample_window_len, 0);
    }

    #[test]
    fn test_reliable_vad_prevents_loud_noise_from_driving_auto_makeup() {
        let mut comp = Compressor::default_voice(48_000.0);
        comp.set_auto_makeup_enabled(true);
        comp.set_target_lufs(-12.0);
        comp.set_noise_reference_reliability(1.0);
        let evidence = AutoMakeupActivityInput {
            vad_probability: 0.01,
            vad_reliability: 1.0,
            noise_floor_db: -32.0,
            live_noise_reliability: 1.0,
        };

        for _ in 0..10 {
            let mut loud_noise = vec![0.08_f32; 48_000];
            comp.process_block_inplace_with_activity_control(&mut loud_noise, Some(evidence));
        }

        assert!(comp.auto_makeup_activity() < AUTO_MAKEUP_ACTIVE_MIN);
        assert!(comp.current_makeup_gain() < 0.1);
    }

    #[test]
    fn test_reliable_vad_allows_quiet_speech_to_drive_auto_makeup() {
        let mut comp = Compressor::default_voice(48_000.0);
        comp.set_auto_makeup_enabled(true);
        comp.set_target_lufs(-12.0);
        let evidence = AutoMakeupActivityInput {
            vad_probability: 0.92,
            vad_reliability: 1.0,
            noise_floor_db: -68.0,
            live_noise_reliability: 0.8,
        };

        for _ in 0..10 {
            let mut quiet_speech = vec![0.003_f32; 48_000];
            comp.process_block_inplace_with_activity_control(&mut quiet_speech, Some(evidence));
        }

        assert!(comp.auto_makeup_activity() > AUTO_MAKEUP_ACTIVE_MIN);
        assert_eq!(comp.auto_makeup_activity_reliability(), 1.0);
        assert!(comp.current_makeup_gain() > 0.1);
    }

    #[test]
    fn test_stale_vad_degrades_continuously_to_noise_relative_fallback() {
        let comp = Compressor::default_voice(48_000.0);
        let rms_db = -52.0;
        let fresh = comp.estimate_auto_makeup_activity(
            rms_db,
            Some(AutoMakeupActivityInput {
                vad_probability: 0.9,
                vad_reliability: 1.0,
                noise_floor_db: -55.0,
                live_noise_reliability: 1.0,
            }),
        );
        let fading = comp.estimate_auto_makeup_activity(
            rms_db,
            Some(AutoMakeupActivityInput {
                vad_probability: 0.9,
                vad_reliability: 0.5,
                noise_floor_db: -55.0,
                live_noise_reliability: 1.0,
            }),
        );
        let stale = comp.estimate_auto_makeup_activity(
            rms_db,
            Some(AutoMakeupActivityInput {
                vad_probability: 0.9,
                vad_reliability: 0.0,
                noise_floor_db: -55.0,
                live_noise_reliability: 1.0,
            }),
        );

        assert!(fresh.activity > fading.activity);
        assert!(fading.activity > stale.activity);
        assert!(fresh.reliability >= fading.reliability);
        assert!(fading.reliability >= stale.reliability);
    }

    #[test]
    fn test_configured_noise_reliability_cannot_elevate_live_evidence() {
        let mut comp = Compressor::default_voice(48_000.0);
        comp.set_noise_reference_reliability(1.0);

        let estimate = comp.estimate_auto_makeup_activity(
            -53.0,
            Some(AutoMakeupActivityInput {
                vad_probability: 0.0,
                vad_reliability: 0.0,
                noise_floor_db: -60.0,
                live_noise_reliability: 0.0,
            }),
        );

        assert_eq!(estimate.reliability, 0.0);
        assert_eq!(
            estimate.activity,
            Compressor::speech_activity_from_rms_db(-53.0)
        );
    }

    #[test]
    fn test_configured_noise_reliability_caps_live_evidence() {
        let mut comp = Compressor::default_voice(48_000.0);
        comp.set_noise_reference_reliability(0.25);

        let estimate = comp.estimate_auto_makeup_activity(
            -53.0,
            Some(AutoMakeupActivityInput {
                vad_probability: 0.0,
                vad_reliability: 0.0,
                noise_floor_db: -60.0,
                live_noise_reliability: 1.0,
            }),
        );

        assert!((estimate.reliability - 0.1875).abs() < f64::EPSILON);
    }

    #[test]
    fn test_invalid_activity_evidence_cannot_poison_compressor_state() {
        let mut comp = Compressor::default_voice(48_000.0);
        comp.set_auto_makeup_enabled(true);
        comp.set_noise_reference_reliability(f64::NAN);
        let evidence = AutoMakeupActivityInput {
            vad_probability: f64::NAN,
            vad_reliability: f64::INFINITY,
            noise_floor_db: f64::NEG_INFINITY,
            live_noise_reliability: f64::NAN,
        };
        let mut block = vec![0.02_f32; 48_000];

        comp.process_block_inplace_with_activity_control(&mut block, Some(evidence));

        assert!(comp.auto_makeup_activity().is_finite());
        assert!(comp.auto_makeup_activity_reliability().is_finite());
        assert!(comp.current_makeup_gain().is_finite());
    }

    #[test]
    fn test_auto_makeup_targets_post_compression_output_level() {
        let mut compressed = Compressor::new(-36.0, 20.0, 0.1, 200.0, 0.0, 0.0, 48_000.0);
        compressed.set_auto_makeup_enabled(true);
        compressed.set_target_lufs(-12.0);

        let mut uncompressed = Compressor::new(0.0, 1.0, 0.1, 200.0, 0.0, 0.0, 48_000.0);
        uncompressed.set_auto_makeup_enabled(true);
        uncompressed.set_target_lufs(-12.0);

        let speech_like = vec![0.04_f32; 48_000];
        for _ in 0..10 {
            let mut compressed_block = speech_like.clone();
            compressed.process_block_inplace(&mut compressed_block);
            let mut uncompressed_block = speech_like.clone();
            uncompressed.process_block_inplace(&mut uncompressed_block);
        }

        assert!(compressed.current_gain_reduction() > 1.0);
        assert!(compressed.current_makeup_gain() >= uncompressed.current_makeup_gain());
    }

    #[test]
    fn test_auto_makeup_converges_to_target_after_feedback_compensation() {
        let sample_rate = 48_000.0;
        let mut compressor = Compressor::new(0.0, 1.0, 0.1, 200.0, 0.0, 0.0, sample_rate);
        compressor.set_auto_makeup_enabled(true);
        compressor.set_target_lufs(-14.0);

        let amplitude = 10.0_f32.powf(-24.0 / 20.0) * 2.0_f32.sqrt();
        let block_size = 480;
        for block_index in 0..1_200 {
            let mut block = vec![0.0_f32; block_size];
            for (sample_index, sample) in block.iter_mut().enumerate() {
                let phase = 2.0
                    * std::f32::consts::PI
                    * 1_000.0
                    * (block_index * block_size + sample_index) as f32
                    / sample_rate as f32;
                *sample = amplitude * phase.sin();
            }
            compressor.process_block_inplace(&mut block);
        }

        assert!(
            (compressor.current_lufs() + 14.0).abs() < 1.0,
            "auto makeup failed to converge to target: {:.2} LUFS",
            compressor.current_lufs()
        );
    }

    #[test]
    fn test_auto_makeup_caps_against_limiter_feedback() {
        let mut uncapped = Compressor::default_voice(48_000.0);
        uncapped.set_auto_makeup_enabled(true);
        uncapped.set_target_lufs(-12.0);

        let mut capped = Compressor::default_voice(48_000.0);
        capped.set_auto_makeup_enabled(true);
        capped.set_target_lufs(-12.0);
        capped.set_limiter_feedback_gain_reduction_db(5.0);

        let mut block = vec![0.04_f32; 48_000];
        for _ in 0..12 {
            uncapped.process_block_inplace(&mut block);
            block.fill(0.04);
            capped.process_block_inplace(&mut block);
            block.fill(0.04);
        }

        assert!(
            capped.current_makeup_gain() < uncapped.current_makeup_gain(),
            "limiter feedback should cap makeup: capped={} uncapped={}",
            capped.current_makeup_gain(),
            uncapped.current_makeup_gain()
        );
        assert!(capped.current_makeup_gain() <= 2.5);
    }

    #[test]
    fn test_manual_makeup_gain_stays_fixed_when_auto_makeup_disabled_for_blocks() {
        let mut comp = Compressor::default_voice(48_000.0);
        comp.set_makeup_gain(6.0);
        comp.set_auto_makeup_enabled(false);

        let mut block = vec![0.04_f32; 48_000];
        for _ in 0..4 {
            comp.process_block_inplace(&mut block);
            block.fill(0.04);
        }

        assert!((comp.current_makeup_gain() - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_block_peak_gain_reduction_preserves_transient_maximum() {
        let mut comp = Compressor::new(-30.0, 6.0, 0.1, 3.0, 0.0, 0.0, 48_000.0);
        let mut block = vec![0.0_f32; 2_400];
        block[..240].fill(0.8);

        comp.process_block_inplace(&mut block);

        assert!(
            comp.block_peak_gain_reduction() > comp.current_gain_reduction() + 0.5,
            "peak={} endpoint={}",
            comp.block_peak_gain_reduction(),
            comp.current_gain_reduction()
        );
    }

    #[test]
    fn test_peak_detector_tracks_single_sample_and_has_rate_independent_release() {
        for sample_rate in [48_000.0, 96_000.0, 192_000.0] {
            let mut comp = Compressor::new(-20.0, 4.0, 10.0, 200.0, 0.0, 0.0, sample_rate);
            let transient = 0.5_f32;
            comp.process_sample(transient);
            assert_eq!(
                comp.peak_envelope,
                f64::from(transient),
                "one-sample peak was under-read at {sample_rate:.0} Hz"
            );

            let peak_before = comp.peak_envelope;
            for _ in 0..(sample_rate * PEAK_DETECTOR_RELEASE_MS / 1_000.0) as usize {
                comp.process_sample(0.0);
            }
            let remaining = comp.peak_envelope / peak_before;
            assert!(
                (0.33..=0.41).contains(&remaining),
                "5 ms peak-envelope release should leave about e^-1 amplitude at {sample_rate:.0} Hz, got {remaining:.4}"
            );
        }
    }

    #[test]
    fn test_peak_detector_tracks_sine_peaks_independently_of_gr_controls() {
        const AMPLITUDE: f64 = 0.8;
        for frequency_hz in [100.0, 1_000.0] {
            for attack_ms in [5.0, 20.0] {
                for release_ms in [20.0, 200.0] {
                    let mut comp =
                        Compressor::new(-20.0, 4.0, attack_ms, release_ms, 0.0, 0.0, 48_000.0);
                    let mut min_peak = f64::INFINITY;
                    let mut max_peak = 0.0_f64;
                    let mut peak_sum = 0.0;
                    let mut min_gain_reduction = f64::INFINITY;
                    let mut count = 0;
                    for index in 0..48_000 {
                        let phase =
                            2.0 * std::f64::consts::PI * frequency_hz * index as f64 / 48_000.0;
                        comp.process_sample((AMPLITUDE * phase.sin()) as f32);
                        if index >= 43_200 {
                            let peak = comp.peak_envelope;
                            min_peak = min_peak.min(peak);
                            max_peak = max_peak.max(peak);
                            peak_sum += peak;
                            min_gain_reduction =
                                min_gain_reduction.min(comp.current_gain_reduction());
                            count += 1;
                        }
                    }
                    let mean_peak = peak_sum / count as f64;
                    let ripple = max_peak - min_peak;
                    assert!(
                        (AMPLITUDE * 0.985..=AMPLITUDE * 1.001).contains(&max_peak),
                        "sample-peak capture missed or exceeded the {frequency_hz:.0} Hz sine peak: max={max_peak:.4}, mean={mean_peak:.4}, ripple={ripple:.4}, attack={attack_ms} ms, release={release_ms} ms"
                    );
                    assert!(mean_peak > AMPLITUDE * 0.5 && mean_peak < max_peak);
                    assert!(ripple < AMPLITUDE * 0.75);
                    assert!(
                        min_gain_reduction > 0.0,
                        "gain reduction collapsed at sine zero crossings: min={min_gain_reduction:.3} dB, frequency={frequency_hz:.0} Hz, attack={attack_ms} ms, release={release_ms} ms"
                    );
                }
            }
        }
    }

    #[test]
    fn test_configured_attack_and_release_calibrate_gain_reduction_smoothing() {
        for (attack_ms, release_ms) in [(5.0, 20.0), (10.0, 80.0), (20.0, 200.0)] {
            let mut comp = Compressor::new(-20.0, 4.0, attack_ms, release_ms, 0.0, 0.0, 48_000.0);
            for _ in 0..(attack_ms * 48.0) as usize {
                comp.smooth_gain_reduction(6.0);
            }
            let expected_attack = 6.0 * (1.0 - (-1.0_f64).exp());
            assert!((comp.current_gain_reduction() - expected_attack).abs() < 0.02);

            for _ in 0..(release_ms * 48.0) as usize {
                comp.smooth_gain_reduction(0.0);
            }
            let expected_release = expected_attack * (-1.0_f64).exp();
            assert!(
                (comp.current_gain_reduction() - expected_release).abs() < 0.02,
                "user release was not applied once at {release_ms} ms: actual={:.4}, expected={expected_release:.4}",
                comp.current_gain_reduction()
            );
        }
    }

    #[test]
    fn test_end_to_end_silence_recovery_follows_short_vs_default_release() {
        fn reduction_after_early_silence(release_ms: f64) -> f64 {
            let mut comp = Compressor::new(-30.0, 4.0, 1.0, release_ms, 0.0, 0.0, 48_000.0);
            for _ in 0..24_000 {
                comp.process_sample(0.5);
            }
            for _ in 0..960 {
                comp.process_sample(0.0);
            }
            comp.current_gain_reduction()
        }

        let fast = reduction_after_early_silence(20.0);
        let default = reduction_after_early_silence(200.0);
        assert!(
            fast < default,
            "20 ms release did not recover faster than 200 ms: fast={fast:.3} dB default={default:.3} dB"
        );
    }

    #[test]
    fn test_user_release_is_applied_once_after_detector_falls_below_threshold() {
        fn gain_reduction_after_one_release_time(release_ms: f64) -> f64 {
            let mut comp = Compressor::new(-30.0, 4.0, 1.0, release_ms, 0.0, 0.0, 48_000.0);
            for _ in 0..24_000 {
                comp.process_sample(0.5);
            }
            for _ in 0..4_800 {
                comp.process_sample(0.0);
            }

            let peak_db = util::linear_to_db(comp.peak_envelope, 1e-10);
            let rms_db = util::linear_to_db(comp.rms_envelope_sq.sqrt(), 1e-10);
            let detector_db = Compressor::blended_detector_db(peak_db, rms_db);
            assert!(
                comp.compute_gain_reduction(detector_db) == 0.0,
                "detector is not below threshold after 100 ms: {detector_db:.2} dB"
            );

            let before = comp.current_gain_reduction();
            assert!(before > 0.0);
            for _ in 0..(release_ms * 48.0) as usize {
                comp.process_sample(0.0);
            }
            comp.current_gain_reduction() / before
        }

        for release_ms in [20.0, 200.0] {
            let remaining = gain_reduction_after_one_release_time(release_ms);
            assert!(
                (remaining - (-1.0_f64).exp()).abs() < 0.03,
                "after one {release_ms} ms user release, GR fraction={remaining:.4}, expected e^-1"
            );
        }
    }

    #[test]
    fn test_reapplying_unchanged_release_controls_is_idempotent() {
        let mut comp = Compressor::new(-20.0, 4.0, 10.0, 80.0, 0.0, 0.0, 48_000.0);
        comp.set_adaptive_release(true);
        comp.fast_release_env_db = 4.0;
        comp.slow_release_env_db = 3.0;
        comp.set_threshold(comp.threshold_db());
        assert_eq!(comp.fast_release_env_db, 4.0);
        assert_eq!(comp.slow_release_env_db, 3.0);
        comp.set_adaptive_release(true);
        assert_eq!(comp.fast_release_env_db, 4.0);
        assert_eq!(comp.slow_release_env_db, 3.0);
    }

    #[test]
    fn test_adaptive_release_uses_base_release_as_its_lower_bound() {
        fn response_for_base(base_release_ms: f64) -> (f64, f64) {
            let mut comp = Compressor::new(-30.0, 4.0, 1.0, 200.0, 0.0, 0.0, 48_000.0);
            comp.set_base_release_time(base_release_ms);
            comp.set_adaptive_release(true);
            for _ in 0..96_000 {
                comp.process_sample(0.7);
            }
            let release_ms = comp.current_release_time();
            let gain_before_silence = comp.current_gain_reduction();
            for _ in 0..12_000 {
                comp.process_sample(0.0);
            }
            (
                release_ms,
                comp.current_gain_reduction() / gain_before_silence,
            )
        }

        let (short, short_remaining) = response_for_base(40.0);
        let (long, long_remaining) = response_for_base(100.0);
        assert!(
            long > short * 1.8,
            "adaptive release readout ignored base release: short={short:.1} ms long={long:.1} ms"
        );
        assert!(
            long_remaining > short_remaining * 1.2,
            "base release did not change the audible recovery: short={short_remaining:.3} long={long_remaining:.3}"
        );
    }

    #[test]
    fn test_manual_makeup_gain_changes_are_click_safe() {
        let mut comp = Compressor::new(-20.0, 4.0, 10.0, 200.0, 0.0, 0.0, 48_000.0);
        let input = 0.1_f32;
        let before = comp.process_sample(input);

        comp.set_makeup_gain(12.0);
        assert!(
            comp.current_makeup_gain() < 0.01,
            "manual makeup jumped immediately to {} dB",
            comp.current_makeup_gain()
        );
        let first_after_edit = comp.process_sample(input);
        assert!(
            first_after_edit / before < 1.01,
            "first output after a manual gain edit stepped by {:.2} dB",
            20.0 * (first_after_edit / before).abs().log10()
        );

        for _ in 1..9_600 {
            comp.process_sample(input);
        }
        assert!(
            (7.2..=8.0).contains(&comp.current_makeup_gain()),
            "manual makeup did not follow its 200 ms smoothing time: {:.3} dB",
            comp.current_makeup_gain()
        );
    }

    #[test]
    fn test_manual_makeup_gain_changes_are_sample_smoothed_in_block_api() {
        let mut comp = Compressor::new(-20.0, 4.0, 10.0, 200.0, 0.0, 0.0, 48_000.0);
        let before = comp.process_sample(0.1);
        comp.set_makeup_gain(12.0);
        let mut block = vec![0.1_f32; 480];
        comp.process_block_inplace(&mut block);

        let first_step_db = 20.0 * (f64::from(block[0]) / f64::from(before)).abs().log10();
        assert!(
            first_step_db < 0.01,
            "first block sample stepped by {first_step_db:.4} dB"
        );
        let largest_step_db = block
            .windows(2)
            .map(|samples| {
                20.0 * (f64::from(samples[1]) / f64::from(samples[0]))
                    .abs()
                    .log10()
            })
            .fold(0.0_f64, f64::max);
        assert!(
            largest_step_db < 0.01,
            "manual makeup created a block-path gain step of {largest_step_db:.4} dB"
        );
        assert!(
            (0.55..=0.65).contains(&comp.current_makeup_gain()),
            "10 ms block did not advance the 200 ms makeup ramp smoothly: {:.3} dB",
            comp.current_makeup_gain()
        );
    }

    #[test]
    fn test_presence_metric_is_high_frequency_weighted_not_bass_weighted() {
        fn presence_metric_at(frequency_hz: f64) -> (f64, f64) {
            let mut comp = Compressor::new(-30.0, 4.0, 1.0, 100.0, 0.0, 0.0, 48_000.0);
            comp.set_sidechain_highpass_enabled(true);
            let mut detector_weight = 1.0;
            for index in 0..48_000 {
                let phase = 2.0 * std::f64::consts::PI * frequency_hz * index as f64 / 48_000.0;
                let input = 0.5 * phase.sin();
                let detector = comp.process_sidechain_sample(input);
                detector_weight = comp.update_sidechain_band_metrics(input, detector);
            }
            let ratio = (comp.presence_band_env_sq / comp.non_presence_band_env_sq).sqrt();
            (ratio, detector_weight)
        }

        let (bass_ratio, bass_weight) = presence_metric_at(250.0);
        let (presence_ratio, presence_weight) = presence_metric_at(4_000.0);
        assert!(
            presence_ratio > bass_ratio * 2.0,
            "presence detector should favor presence over bass: bass={bass_ratio:.3} presence={presence_ratio:.3}"
        );
        assert!(
            presence_weight >= bass_weight + 0.14,
            "presence weighting did not favor high-frequency content: bass={bass_weight:.3} presence={presence_weight:.3}"
        );
    }

    #[test]
    fn test_reset_clears_reported_loudness() {
        let mut comp = Compressor::default_voice(48_000.0);
        comp.current_lufs = -18.0;

        comp.reset();

        assert_eq!(comp.current_lufs(), -100.0);
    }

    #[test]
    fn test_reset_clears_reported_loudness_without_meter() {
        let mut comp = Compressor::new(-20.0, 4.0, 10.0, 200.0, 0.0, 0.0, 12_345.0);
        assert!(comp.loudness_meter.is_none());
        comp.current_lufs = -18.0;

        comp.reset();

        assert_eq!(comp.current_lufs(), -100.0);
    }
}
