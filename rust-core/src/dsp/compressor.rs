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
const MAKEUP_SILENCE_RELAX_MS: f64 = 1500.0;
const SIDECHAIN_HIGHPASS_DEFAULT_HZ: f64 = 120.0;
const SIDECHAIN_BAND_ENV_MS: f64 = 18.0;
const PLOSIVE_RATIO_START: f64 = 1.25;
const PLOSIVE_RATIO_FULL: f64 = 5.0;
const PLOSIVE_MIN_DETECTOR_GAIN: f64 = 0.35;

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
    /// Release time constant for the peak detector envelope
    detector_release_coeff: f64,
    /// Makeup gain in dB to compensate for gain reduction
    makeup_gain_db: f64,
    /// Makeup gain as linear multiplier (cached)
    makeup_gain_linear: f64,
    /// Knee width in dB for soft-knee transition
    knee_db: f64,
    /// AR-smoothed log-domain peak detector (dBFS)
    peak_envelope_db: f64,
    /// Fixed-time RMS detector state (squared amplitude)
    rms_envelope_sq: f64,
    /// RMS smoothing coefficient (single-pole IIR, fixed 20ms)
    rms_coeff: f64,
    /// Current gain reduction in dB (for metering)
    current_gain_reduction_db: f64,
    /// Sample rate
    sample_rate: f64,
    /// Whether compressor is enabled
    enabled: bool,
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
    /// Fast adaptive release envelope in dB.
    fast_release_env_db: f64,
    /// Slow adaptive release envelope in dB.
    slow_release_env_db: f64,
    /// Loudness meter for auto makeup gain
    loudness_meter: Option<crate::dsp::loudness::LoudnessMeter>,
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
    /// Slow relaxation coefficient used when auto makeup sees silence/noise.
    makeup_silence_relax_coeff: f64,
    /// Whether the detector sidechain ignores most plosive/rumble energy.
    sidechain_highpass_enabled: bool,
    /// Sidechain high-pass coefficient.
    sidechain_highpass_coeff: f64,
    /// Previous sidechain high-pass input sample.
    sidechain_highpass_prev_input: f64,
    /// Previous sidechain high-pass output sample.
    sidechain_highpass_prev_output: f64,
    /// Low-band detector energy used for plosive discrimination.
    low_band_env_sq: f64,
    /// Voiced-band detector energy used for plosive discrimination.
    voiced_band_env_sq: f64,
    /// Presence-band detector energy used to keep consonants forward.
    presence_band_env_sq: f64,
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
        let rms_coeff = util::time_constant_to_coeff(20.0, sample_rate);
        let release_smoothing_coeff = util::time_constant_to_coeff(100.0, sample_rate);
        let makeup_smoothing_coeff = util::time_constant_to_coeff(200.0, sample_rate);

        let loudness_meter = match crate::dsp::loudness::LoudnessMeter::new(sample_rate as u32) {
            Ok(meter) => Some(meter),
            Err(e) => {
                eprintln!("Failed to initialize loudness meter: {}", e);
                None
            }
        };

        Self {
            threshold_db,
            ratio: ratio.max(1.0),
            attack_coeff,
            release_coeff,
            detector_release_coeff: release_coeff,
            makeup_gain_db,
            makeup_gain_linear: util::db_to_linear(makeup_gain_db),
            knee_db: knee_db.max(0.0),
            peak_envelope_db: -120.0,
            rms_envelope_sq: 0.0,
            rms_coeff,
            current_gain_reduction_db: 0.0,
            sample_rate,
            enabled: true,
            adaptive_release: false,
            base_release_ms: release_ms,
            current_release_ms: release_ms,
            target_release_ms: release_ms,
            release_smoothing_coeff,
            fast_release_env_db: 0.0,
            slow_release_env_db: 0.0,
            loudness_meter,
            auto_makeup_enabled: false,
            target_lufs: -18.0,
            smoothed_makeup_gain: makeup_gain_db,
            makeup_smoothing_coeff,
            current_lufs: -100.0,
            speech_activity_score: 0.0,
            makeup_silence_relax_coeff: util::time_constant_to_coeff(
                MAKEUP_SILENCE_RELAX_MS,
                sample_rate,
            ),
            sidechain_highpass_enabled: false,
            sidechain_highpass_coeff: Self::sidechain_highpass_coeff(
                SIDECHAIN_HIGHPASS_DEFAULT_HZ,
                sample_rate,
            ),
            sidechain_highpass_prev_input: 0.0,
            sidechain_highpass_prev_output: 0.0,
            low_band_env_sq: 0.0,
            voiced_band_env_sq: 0.0,
            presence_band_env_sq: 0.0,
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
        self.base_release_ms = release_ms;
        if !self.adaptive_release {
            self.current_release_ms = release_ms;
            self.target_release_ms = release_ms;
            self.release_coeff = util::time_constant_to_coeff(release_ms, self.sample_rate);
        }
        self.detector_release_coeff = util::time_constant_to_coeff(release_ms, self.sample_rate);
    }

    /// Enable or disable adaptive release
    pub fn set_adaptive_release(&mut self, enabled: bool) {
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
        }
    }

    /// Check if adaptive release is enabled
    pub fn adaptive_release(&self) -> bool {
        self.adaptive_release
    }

    /// Set base release time
    pub fn set_base_release_time(&mut self, release_ms: f64) {
        self.base_release_ms = release_ms;
        if !self.adaptive_release {
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
        self.makeup_gain_linear = util::db_to_linear(makeup_gain_db);
        if !self.auto_makeup_enabled {
            self.smoothed_makeup_gain = makeup_gain_db;
        }
    }

    /// Get makeup gain in dB
    pub fn makeup_gain_db(&self) -> f64 {
        self.makeup_gain_db
    }

    /// Set knee width in dB
    pub fn set_knee(&mut self, knee_db: f64) {
        self.knee_db = knee_db.max(0.0);
    }

    /// Enable or disable the compressor
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if compressor is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get current gain reduction in dB (for metering)
    pub fn current_gain_reduction(&self) -> f64 {
        self.current_gain_reduction_db
    }

    /// Enable or disable auto makeup gain
    pub fn set_auto_makeup_enabled(&mut self, enabled: bool) {
        self.auto_makeup_enabled = enabled && self.loudness_meter.is_some();
        if !enabled {
            self.smoothed_makeup_gain = self.makeup_gain_db;
        }
    }

    /// Check if auto makeup is enabled
    pub fn auto_makeup_enabled(&self) -> bool {
        self.auto_makeup_enabled
    }

    /// Set target LUFS for auto makeup gain
    pub fn set_target_lufs(&mut self, target: f64) {
        self.target_lufs = target.clamp(-24.0, -12.0);
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
        self.low_band_env_sq = 0.0;
        self.voiced_band_env_sq = 0.0;
        self.presence_band_env_sq = 0.0;
        self.plosive_ratio = 0.0;
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
    fn update_sidechain_band_metrics(&mut self, full_band_input: f64, detector_input: f64) -> f64 {
        if !self.sidechain_highpass_enabled {
            self.plosive_ratio = 0.0;
            return 1.0;
        }

        let low_component = full_band_input - detector_input;
        let voiced_component = detector_input;
        let presence_component = 0.65 * detector_input + 0.35 * (detector_input - low_component);
        let coeff = util::time_constant_to_coeff(SIDECHAIN_BAND_ENV_MS, self.sample_rate);

        self.low_band_env_sq =
            coeff * self.low_band_env_sq + (1.0 - coeff) * low_component * low_component;
        self.voiced_band_env_sq =
            coeff * self.voiced_band_env_sq + (1.0 - coeff) * voiced_component * voiced_component;
        self.presence_band_env_sq = coeff * self.presence_band_env_sq
            + (1.0 - coeff) * presence_component * presence_component;

        let low_rms = self.low_band_env_sq.sqrt();
        let voiced_rms = self.voiced_band_env_sq.sqrt().max(1e-8);
        let presence_rms = self.presence_band_env_sq.sqrt();
        self.plosive_ratio = (low_rms / voiced_rms).clamp(0.0, 32.0);

        let plosive_amount = ((self.plosive_ratio - PLOSIVE_RATIO_START)
            / (PLOSIVE_RATIO_FULL - PLOSIVE_RATIO_START))
            .clamp(0.0, 1.0);
        let plosive_penalty = 1.0 - plosive_amount * (1.0 - PLOSIVE_MIN_DETECTOR_GAIN);
        let presence_ratio = (presence_rms / voiced_rms).clamp(0.0, 4.0);
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
        self.target_release_ms = ADAPTIVE_FAST_RELEASE_MS
            + syllabic * (ADAPTIVE_SLOW_RELEASE_MS - ADAPTIVE_FAST_RELEASE_MS);
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

        let fast_release_coeff =
            util::time_constant_to_coeff(ADAPTIVE_FAST_RELEASE_MS, self.sample_rate);
        let slow_charge_coeff =
            util::time_constant_to_coeff(ADAPTIVE_SLOW_CHARGE_MS, self.sample_rate);
        let slow_release_coeff =
            util::time_constant_to_coeff(ADAPTIVE_SLOW_RELEASE_MS, self.sample_rate);

        if target_gain_reduction_db > self.current_gain_reduction_db {
            self.fast_release_env_db = self.attack_coeff * self.current_gain_reduction_db
                + (1.0 - self.attack_coeff) * target_gain_reduction_db;
        } else {
            self.fast_release_env_db = fast_release_coeff * self.fast_release_env_db
                + (1.0 - fast_release_coeff) * target_gain_reduction_db;
        }

        if target_gain_reduction_db > SLOW_RELEASE_TRIGGER_DB {
            self.slow_release_env_db = slow_charge_coeff * self.slow_release_env_db
                + (1.0 - slow_charge_coeff) * target_gain_reduction_db;
        } else {
            self.slow_release_env_db *= slow_release_coeff;
        }

        self.current_gain_reduction_db = self.fast_release_env_db.max(self.slow_release_env_db);
    }

    fn speech_activity_from_rms_db(rms_db: f64) -> f64 {
        if !(SPEECH_ACTIVE_RMS_MIN_DB..=SPEECH_ACTIVE_RMS_MAX_DB).contains(&rms_db) {
            return 0.0;
        }
        let onset = ((rms_db - SPEECH_ACTIVE_RMS_MIN_DB) / 12.0).clamp(0.0, 1.0);
        let overload = ((SPEECH_ACTIVE_RMS_MAX_DB - rms_db) / 6.0).clamp(0.0, 1.0);
        onset.min(overload)
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

    fn update_auto_makeup_gain(&mut self, speech_activity: f64) {
        if !self.auto_makeup_enabled {
            let target = self.makeup_gain_db;
            let diff = target - self.smoothed_makeup_gain;
            if diff.abs() > 0.1 {
                self.smoothed_makeup_gain = self.makeup_smoothing_coeff * self.smoothed_makeup_gain
                    + (1.0 - self.makeup_smoothing_coeff) * target;
            } else {
                self.smoothed_makeup_gain = target;
            }
            return;
        }

        if let Some(meter) = &self.loudness_meter {
            self.current_lufs = meter.loudness_momentary() as f64;
            self.speech_activity_score =
                0.95 * self.speech_activity_score + 0.05 * speech_activity.clamp(0.0, 1.0);
            if self.speech_activity_score < 0.20 {
                self.smoothed_makeup_gain = self.makeup_silence_relax_coeff
                    * self.smoothed_makeup_gain
                    + (1.0 - self.makeup_silence_relax_coeff) * self.makeup_gain_db;
                return;
            }
            let required_gain = self.target_lufs - self.current_lufs;
            let headroom_cap =
                (12.0 - self.limiter_feedback_gain_reduction_db * 2.0).clamp(0.0, 12.0);
            let clamped_gain = required_gain.clamp(0.0, headroom_cap);

            let diff = clamped_gain - self.smoothed_makeup_gain;
            if diff.abs() > 0.1 {
                let makeup_coeff = 0.90;
                self.smoothed_makeup_gain =
                    makeup_coeff * self.smoothed_makeup_gain + (1.0 - makeup_coeff) * clamped_gain;
            } else {
                self.smoothed_makeup_gain = clamped_gain;
            }
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
        self.process_sample_impl(input, true)
    }

    /// Process a block of samples in-place
    pub fn process_block_inplace(&mut self, buffer: &mut [f32]) {
        if !self.enabled {
            self.current_gain_reduction_db = 0.0;
            return;
        }

        let speech_activity = Self::speech_activity_from_rms_db(Self::block_rms_db(buffer));
        for sample in buffer.iter_mut() {
            *sample = self.process_sample_impl(*sample, false);
        }
        if speech_activity > 0.20 {
            if let Some(meter) = &mut self.loudness_meter {
                meter.process(buffer);
            }
        }
        self.update_auto_makeup_gain(speech_activity);
    }

    #[inline]
    fn process_sample_impl(&mut self, input: f32, update_makeup_gain: bool) -> f32 {
        if !self.enabled {
            self.current_gain_reduction_db = 0.0;
            return input;
        }

        let input_f64 = input as f64;
        let detector_input = self.process_sidechain_sample(input_f64);
        let detector_weight = self.update_sidechain_band_metrics(input_f64, detector_input);
        let detector_abs = detector_input.abs();
        let inst_peak_db = util::linear_to_db(detector_abs, 1e-10);
        let peak_coeff = if inst_peak_db > self.peak_envelope_db {
            self.attack_coeff
        } else {
            self.detector_release_coeff
        };
        self.peak_envelope_db =
            peak_coeff * self.peak_envelope_db + (1.0 - peak_coeff) * inst_peak_db;

        let input_squared = detector_input * detector_input;
        self.rms_envelope_sq =
            self.rms_coeff * self.rms_envelope_sq + (1.0 - self.rms_coeff) * input_squared;
        let rms_db = util::linear_to_db(self.rms_envelope_sq.sqrt(), 1e-10);

        let detector_db = Self::blended_detector_db(self.peak_envelope_db, rms_db)
            + util::linear_to_db(detector_weight, 1e-10);

        self.update_adaptive_release_time_meter();
        let release_diff = self.target_release_ms - self.current_release_ms;
        if release_diff.abs() > 1.0 {
            self.current_release_ms = self.release_smoothing_coeff * self.current_release_ms
                + (1.0 - self.release_smoothing_coeff) * self.target_release_ms;
        } else {
            self.current_release_ms = self.target_release_ms;
        }
        self.release_coeff =
            util::time_constant_to_coeff(self.current_release_ms, self.sample_rate);

        let target_gain_reduction_db = self.compute_gain_reduction(detector_db);
        self.smooth_gain_reduction(target_gain_reduction_db);

        if update_makeup_gain {
            let speech_activity = Self::speech_activity_from_rms_db(detector_db);
            self.update_auto_makeup_gain(speech_activity);
        }

        let output_gain = util::db_to_linear(-self.current_gain_reduction_db)
            * util::db_to_linear(self.smoothed_makeup_gain);
        (input_f64 * output_gain) as f32
    }

    /// Reset compressor state
    pub fn reset(&mut self) {
        self.peak_envelope_db = -120.0;
        self.rms_envelope_sq = 0.0;
        self.current_gain_reduction_db = 0.0;
        self.fast_release_env_db = 0.0;
        self.slow_release_env_db = 0.0;
        self.current_release_ms = self.base_release_ms;
        self.target_release_ms = self.base_release_ms;
        self.release_coeff =
            util::time_constant_to_coeff(self.current_release_ms, self.sample_rate);
        self.reset_sidechain_highpass_state();
        self.limiter_feedback_gain_reduction_db = 0.0;
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
        let output = comp.process_sample(input);
        assert_eq!(output, input);
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

        let fixed_peak_before = fixed.peak_envelope_db;
        let adaptive_peak_before = adaptive.peak_envelope_db;
        for _ in 0..2_400 {
            fixed.process_sample(0.001);
            adaptive.process_sample(0.001);
        }

        let fixed_drop = fixed_peak_before - fixed.peak_envelope_db;
        let adaptive_drop = adaptive_peak_before - adaptive.peak_envelope_db;
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

        assert!((comp.current_makeup_gain() - 6.0).abs() < 1e-9);
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
