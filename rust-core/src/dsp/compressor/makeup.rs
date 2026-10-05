//! Slow loudness, activity and makeup-gain control owned by the compressor.

use super::AutoMakeupActivityInput;
use crate::dsp::util;

const SPEECH_ACTIVE_RMS_MIN_DB: f64 = -55.0;
const SPEECH_ACTIVE_RMS_MAX_DB: f64 = -6.0;
pub(super) const AUTO_MAKEUP_ACTIVE_MIN: f64 = 0.20;
const AUTO_MAKEUP_RELIABILITY_MIN: f64 = 0.35;
const AUTO_MAKEUP_ACTIVITY_SMOOTH_MS: f64 = 200.0;
const NOISE_RELATIVE_ACTIVITY_START_DB: f64 = 3.0;
const NOISE_RELATIVE_ACTIVITY_FULL_DB: f64 = 15.0;
const MAKEUP_SILENCE_RELAX_MS: f64 = 1500.0;
pub(super) const AUTO_MAKEUP_SAMPLE_WINDOW_MS: f64 = 10.0;
// Compressor callers accept generic rates through 192 kHz; reserve the full
// 10 ms window at that rate even when the optional loudness meter is absent.
const AUTO_MAKEUP_SAMPLE_WINDOW_MAX_SAMPLES: usize = 1_920;

#[derive(Clone, Copy, Debug)]
pub(super) struct AutoMakeupActivityEstimate {
    pub(super) activity: f64,
    pub(super) reliability: f64,
}

pub(super) struct MakeupController {
    /// Makeup gain in dB to compensate for gain reduction
    makeup_gain_db: f64,
    /// Loudness meter for auto makeup gain
    pub(super) loudness_meter: Option<crate::dsp::loudness::LoudnessMeter>,
    /// Fixed storage for the sample API's auto-makeup control window.
    auto_makeup_sample_window: [f32; AUTO_MAKEUP_SAMPLE_WINDOW_MAX_SAMPLES],
    /// Number of output samples currently held for the sample API window.
    pub(super) auto_makeup_sample_window_len: usize,
    /// Sum of per-sample activity evidence in the current sample window.
    auto_makeup_sample_activity_sum: f64,
    /// Number of samples in one sample API auto-makeup control window.
    pub(super) auto_makeup_sample_window_samples: usize,
    /// Auto makeup gain enabled
    auto_makeup_enabled: bool,
    /// Target LUFS for auto makeup gain
    target_lufs: f64,
    /// Smoothed makeup gain (for transitions)
    smoothed_makeup_gain: f64,
    /// Makeup gain smoothing coefficient (200ms time constant)
    makeup_smoothing_coeff: f64,
    /// Current measured loudness (for metering)
    pub(super) current_lufs: f64,
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
    /// Previous limiter pressure used to keep auto makeup inside headroom.
    limiter_feedback_gain_reduction_db: f64,
}

impl MakeupController {
    pub(super) fn new(makeup_gain_db: f64, sample_rate: f64) -> Self {
        let makeup_smoothing_coeff = util::time_constant_to_coeff(200.0, sample_rate);
        let loudness_meter = crate::dsp::loudness::LoudnessMeter::new(sample_rate as u32).ok();
        let auto_makeup_sample_window_samples = if sample_rate.is_finite() && sample_rate > 0.0 {
            (sample_rate * AUTO_MAKEUP_SAMPLE_WINDOW_MS / 1_000.0)
                .round()
                .clamp(1.0, AUTO_MAKEUP_SAMPLE_WINDOW_MAX_SAMPLES as f64) as usize
        } else {
            1
        };

        Self {
            makeup_gain_db,
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
            limiter_feedback_gain_reduction_db: 0.0,
        }
    }

    pub(super) fn set_makeup_gain(&mut self, makeup_gain_db: f64) {
        self.makeup_gain_db = makeup_gain_db;
    }

    pub(super) fn set_auto_makeup_enabled(&mut self, enabled: bool) {
        let enabled = enabled && self.loudness_meter.is_some();
        if self.auto_makeup_enabled == enabled {
            return;
        }
        self.auto_makeup_enabled = enabled;
        if !enabled {
            self.clear_auto_makeup_sample_window();
        }
    }

    pub(super) fn auto_makeup_enabled(&self) -> bool {
        self.auto_makeup_enabled
    }

    pub(super) fn set_target_lufs(&mut self, target: f64) {
        let target = target.clamp(-24.0, -12.0);
        if self.target_lufs != target {
            self.target_lufs = target;
        }
    }

    pub(super) fn target_lufs(&self) -> f64 {
        self.target_lufs
    }

    pub(super) fn current_lufs(&self) -> f64 {
        self.current_lufs
    }

    pub(super) fn current_makeup_gain(&self) -> f64 {
        self.smoothed_makeup_gain
    }

    pub(super) fn set_noise_reference_reliability(&mut self, reliability: f64) {
        self.noise_reference_reliability = Self::finite_unit(reliability).unwrap_or(0.0);
    }

    pub(super) fn auto_makeup_activity(&self) -> f64 {
        self.speech_activity_score
    }

    pub(super) fn auto_makeup_activity_reliability(&self) -> f64 {
        self.auto_makeup_activity_reliability
    }

    pub(super) fn set_limiter_feedback_gain_reduction_db(&mut self, gain_reduction_db: f64) {
        self.limiter_feedback_gain_reduction_db = gain_reduction_db.clamp(0.0, 24.0);
    }

    pub(super) fn speech_activity_from_rms_db(rms_db: f64) -> f64 {
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

    pub(super) fn estimate_auto_makeup_activity(
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

    pub(super) fn block_rms_db(buffer: &[f32]) -> f64 {
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

    pub(super) fn update_auto_makeup_gain(
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

    #[inline]
    pub(super) fn clear_auto_makeup_sample_window(&mut self) {
        self.auto_makeup_sample_window_len = 0;
        self.auto_makeup_sample_activity_sum = 0.0;
    }

    pub(super) fn process_block_output(
        &mut self,
        buffer: &[f32],
        activity: AutoMakeupActivityEstimate,
    ) {
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
    pub(super) fn advance_manual_gain(&mut self, detector_db: f64) {
        if !self.auto_makeup_enabled {
            let speech_activity = Self::speech_activity_from_rms_db(detector_db);
            self.update_auto_makeup_gain(speech_activity, 1.0, 1);
        }
    }

    #[inline]
    pub(super) fn process_sample_output(&mut self, output: f32, detector_db: f64) {
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

    pub(super) fn reset(&mut self) {
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
