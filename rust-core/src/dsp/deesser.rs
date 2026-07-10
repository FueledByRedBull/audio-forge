//! Dynamic-EQ de-esser using sidechain sibilance detection.
//!
//! Detection path:
//! - High-pass at low cutoff (default 4kHz)
//! - Low-pass at high cutoff (default 9kHz)
//! - Envelope follower in dB domain
//!
//! Gain computer:
//! - Threshold/ratio above threshold
//! - Attack/release smoothing
//! - Max reduction clamp
//!
//! Apply path:
//! - Detector drives a dynamic peaking EQ in the sibilance region.

use super::biquad::{Biquad, BiquadType};
use crate::dsp::util;

const VOICE_REFERENCE_SIDECHAIN_DISCOUNT: f64 = 0.6;
const DETECTOR_RATIO_GATE_DB: f64 = 1.5;
const DETECTOR_RATIO_FULL_DB: f64 = 10.0;
const DETECTOR_LEVEL_GATE_DB: f64 = -62.0;
const DETECTOR_LEVEL_FULL_DB: f64 = -24.0;
const DETECTOR_VOICE_GATE_DB: f64 = -58.0;
const DETECTOR_VOICE_FULL_DB: f64 = -34.0;
const AUTO_BASELINE_FALL_MS: f64 = 13.88;
const AUTO_BASELINE_RISE_MS: f64 = 34.72;
const AUTO_BASELINE_INACTIVE_DECAY_MS: f64 = 20.82;
const DEESSER_BAND_COUNT: usize = 3;
const DEESSER_DEFAULT_HIGH_CUT_HZ: f64 = 11_000.0;
const BROADBAND_NARROWNESS_GATE: f64 = 0.34;
const BROADBAND_NARROWNESS_FULL: f64 = 0.68;

struct DeEsserBand {
    low_hz: f64,
    high_hz: f64,
    env: f64,
    confidence: f64,
    baseline_excess_db: f64,
    reduction_db: f64,
    detector_hp: Biquad,
    detector_lp: Biquad,
    dynamic_eq: Biquad,
}

impl DeEsserBand {
    fn new(low_hz: f64, high_hz: f64, sample_rate: f64) -> Self {
        let detector_q = 0.707;
        let center_hz = DeEsser::dynamic_eq_center_hz(low_hz, high_hz);
        let dynamic_q = DeEsser::dynamic_eq_q(low_hz, high_hz);
        Self {
            low_hz,
            high_hz,
            env: 0.0,
            confidence: 0.0,
            baseline_excess_db: 0.0,
            reduction_db: 0.0,
            detector_hp: Biquad::new(BiquadType::HighPass, low_hz, 0.0, detector_q, sample_rate),
            detector_lp: Biquad::new(BiquadType::LowPass, high_hz, 0.0, detector_q, sample_rate),
            dynamic_eq: Biquad::new(BiquadType::Peaking, center_hz, 0.0, dynamic_q, sample_rate),
        }
    }

    fn set_bounds(&mut self, low_hz: f64, high_hz: f64) {
        self.low_hz = low_hz;
        self.high_hz = high_hz;
        self.detector_hp.set_frequency(low_hz);
        self.detector_lp.set_frequency(high_hz);
        self.dynamic_eq
            .set_frequency(DeEsser::dynamic_eq_center_hz(low_hz, high_hz));
        self.dynamic_eq
            .set_q(DeEsser::dynamic_eq_q(low_hz, high_hz));
    }

    fn reset(&mut self) {
        self.env = 0.0;
        self.confidence = 0.0;
        self.baseline_excess_db = 0.0;
        self.reduction_db = 0.0;
        self.detector_hp.reset();
        self.detector_lp.reset();
        self.dynamic_eq.reset();
        self.dynamic_eq.set_gain_db_immediate(0.0);
    }
}

/// Real-time de-esser processor.
pub struct DeEsser {
    enabled: bool,
    auto_enabled: bool,
    auto_amount: f64,
    threshold_db: f64,
    ratio: f64,
    attack_coeff: f64,
    release_coeff: f64,
    detector_attack_coeff: f64,
    detector_release_coeff: f64,
    max_reduction_db: f64,
    current_reduction_db: f64,
    broadband_env: f64,
    detector_confidence: f64,
    low_cut_hz: f64,
    high_cut_hz: f64,
    sample_rate: f64,
    bands: [DeEsserBand; DEESSER_BAND_COUNT],
}

impl DeEsser {
    /// Create a de-esser with conservative voice defaults.
    pub fn new(sample_rate: f64) -> Self {
        let low_cut_hz = 4000.0;
        let high_cut_hz = DEESSER_DEFAULT_HIGH_CUT_HZ;
        let bands = Self::make_bands(low_cut_hz, high_cut_hz, sample_rate);

        Self {
            enabled: false,
            auto_enabled: true,
            auto_amount: 0.5,
            threshold_db: -28.0,
            ratio: 4.0,
            attack_coeff: util::time_constant_to_coeff(2.0, sample_rate),
            release_coeff: util::time_constant_to_coeff(80.0, sample_rate),
            detector_attack_coeff: util::time_constant_to_coeff(1.5, sample_rate),
            detector_release_coeff: util::time_constant_to_coeff(60.0, sample_rate),
            max_reduction_db: 6.0,
            current_reduction_db: 0.0,
            broadband_env: 0.0,
            detector_confidence: 0.0,
            low_cut_hz,
            high_cut_hz,
            sample_rate,
            bands,
        }
    }

    #[inline]
    fn update_env(&self, prev: f64, input: f64) -> f64 {
        Self::smooth_value(
            prev,
            input,
            self.detector_attack_coeff,
            self.detector_release_coeff,
        )
    }

    #[inline]
    fn smooth_value(prev: f64, input: f64, attack_coeff: f64, release_coeff: f64) -> f64 {
        let coeff = if input > prev {
            attack_coeff
        } else {
            release_coeff
        };
        coeff * prev + (1.0 - coeff) * input
    }

    #[inline]
    fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + (b - a) * t
    }

    #[inline]
    fn normalize_range(value: f64, start: f64, end: f64) -> f64 {
        ((value - start) / (end - start)).clamp(0.0, 1.0)
    }

    #[inline]
    fn confidence_reduction_gain(confidence: f64, floor: f64) -> f64 {
        Self::normalize_range(confidence, floor.clamp(0.0, 0.95), 1.0)
    }

    #[inline]
    fn detector_confidence_target(
        sidechain_level_db: f64,
        voice_reference_db: f64,
        narrowness: f64,
    ) -> f64 {
        let spectral_ratio_db = (sidechain_level_db - voice_reference_db).max(0.0);
        let ratio_conf = Self::normalize_range(
            spectral_ratio_db,
            DETECTOR_RATIO_GATE_DB,
            DETECTOR_RATIO_FULL_DB,
        );
        let level_conf = Self::normalize_range(
            sidechain_level_db,
            DETECTOR_LEVEL_GATE_DB,
            DETECTOR_LEVEL_FULL_DB,
        );
        let voice_conf = Self::normalize_range(
            voice_reference_db,
            DETECTOR_VOICE_GATE_DB,
            DETECTOR_VOICE_FULL_DB,
        );

        // Strong narrow-band sibilance should still be detected when the voice body is brief.
        let narrow_sibilance_support = if spectral_ratio_db > 6.0 && sidechain_level_db > -45.0 {
            0.75
        } else {
            0.0
        };
        let voice_support = voice_conf.max(narrow_sibilance_support);
        let balance_conf = if ratio_conf > 0.12 {
            ratio_conf.max(voice_support * 0.65)
        } else {
            ratio_conf
        };
        let broadband_penalty = Self::lerp(0.35, 1.0, balance_conf);
        let narrowness_gain = Self::lerp(
            0.35,
            1.0,
            Self::normalize_range(
                narrowness,
                BROADBAND_NARROWNESS_GATE,
                BROADBAND_NARROWNESS_FULL,
            ),
        );

        (0.62 * ratio_conf + 0.18 * level_conf + 0.20 * voice_support)
            * broadband_penalty
            * narrowness_gain
    }

    #[inline]
    fn detector_spectral_ratio_db(sidechain_level_db: f64, voice_reference_db: f64) -> f64 {
        (sidechain_level_db - voice_reference_db).max(0.0)
    }

    fn rebuild_detector_filters(&mut self) {
        let span = (self.high_cut_hz - self.low_cut_hz).max(600.0);
        let split_a = self.low_cut_hz + span / 3.0;
        let split_b = self.low_cut_hz + span * 2.0 / 3.0;
        let bounds = [
            (self.low_cut_hz, split_a),
            (split_a, split_b),
            (split_b, self.high_cut_hz),
        ];

        for (band, (low_hz, high_hz)) in self.bands.iter_mut().zip(bounds) {
            band.set_bounds(low_hz, high_hz);
        }
    }

    fn make_bands(
        low_cut_hz: f64,
        high_cut_hz: f64,
        sample_rate: f64,
    ) -> [DeEsserBand; DEESSER_BAND_COUNT] {
        let span = (high_cut_hz - low_cut_hz).max(600.0);
        let split_a = low_cut_hz + span / 3.0;
        let split_b = low_cut_hz + span * 2.0 / 3.0;
        [
            DeEsserBand::new(low_cut_hz, split_a, sample_rate),
            DeEsserBand::new(split_a, split_b, sample_rate),
            DeEsserBand::new(split_b, high_cut_hz, sample_rate),
        ]
    }

    #[inline]
    fn dynamic_eq_center_hz(low_cut_hz: f64, high_cut_hz: f64) -> f64 {
        (low_cut_hz * high_cut_hz).sqrt()
    }

    #[inline]
    fn dynamic_eq_q(low_cut_hz: f64, high_cut_hz: f64) -> f64 {
        let bandwidth = (high_cut_hz - low_cut_hz).max(200.0);
        (Self::dynamic_eq_center_hz(low_cut_hz, high_cut_hz) / bandwidth).clamp(0.5, 6.0)
    }

    #[inline]
    fn auto_baseline_fall_coeff(&self) -> f64 {
        util::time_constant_to_coeff(AUTO_BASELINE_FALL_MS, self.sample_rate)
    }

    #[inline]
    fn auto_baseline_rise_coeff(&self) -> f64 {
        util::time_constant_to_coeff(AUTO_BASELINE_RISE_MS, self.sample_rate)
    }

    #[inline]
    fn auto_baseline_inactive_decay_coeff(&self) -> f64 {
        util::time_constant_to_coeff(AUTO_BASELINE_INACTIVE_DECAY_MS, self.sample_rate)
    }

    /// Enable or disable de-essing.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check whether de-esser is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable/disable smart auto de-essing.
    pub fn set_auto_enabled(&mut self, enabled: bool) {
        self.auto_enabled = enabled;
    }

    pub fn is_auto_enabled(&self) -> bool {
        self.auto_enabled
    }

    /// Set auto mode amount [0.0, 1.0].
    pub fn set_auto_amount(&mut self, amount: f64) {
        self.auto_amount = amount.clamp(0.0, 1.0);
    }

    pub fn auto_amount(&self) -> f64 {
        self.auto_amount
    }

    /// Set detector low cutoff (high-pass edge) in Hz.
    pub fn set_low_cut_hz(&mut self, low_cut_hz: f64) {
        self.low_cut_hz = low_cut_hz.clamp(2000.0, 12000.0);
        if self.high_cut_hz <= self.low_cut_hz + 200.0 {
            self.high_cut_hz = (self.low_cut_hz + 200.0).clamp(2200.0, 16000.0);
        }
        self.rebuild_detector_filters();
    }

    /// Set detector high cutoff (low-pass edge) in Hz.
    pub fn set_high_cut_hz(&mut self, high_cut_hz: f64) {
        self.high_cut_hz = high_cut_hz.clamp(2200.0, 16000.0);
        if self.high_cut_hz <= self.low_cut_hz + 200.0 {
            self.low_cut_hz = (self.high_cut_hz - 200.0).clamp(2000.0, 12000.0);
        }
        self.rebuild_detector_filters();
    }

    /// Set threshold in dBFS.
    pub fn set_threshold_db(&mut self, threshold_db: f64) {
        self.threshold_db = threshold_db.clamp(-60.0, -6.0);
    }

    /// Set ratio (>= 1.0).
    pub fn set_ratio(&mut self, ratio: f64) {
        self.ratio = ratio.clamp(1.0, 20.0);
    }

    /// Set attack in milliseconds.
    pub fn set_attack_ms(&mut self, attack_ms: f64) {
        self.attack_coeff =
            util::time_constant_to_coeff(attack_ms.clamp(0.1, 50.0), self.sample_rate);
    }

    /// Set release in milliseconds.
    pub fn set_release_ms(&mut self, release_ms: f64) {
        self.release_coeff =
            util::time_constant_to_coeff(release_ms.clamp(5.0, 500.0), self.sample_rate);
    }

    /// Set max reduction cap in dB.
    pub fn set_max_reduction_db(&mut self, max_reduction_db: f64) {
        self.max_reduction_db = max_reduction_db.clamp(0.0, 24.0);
    }

    pub fn low_cut_hz(&self) -> f64 {
        self.low_cut_hz
    }

    pub fn high_cut_hz(&self) -> f64 {
        self.high_cut_hz
    }

    pub fn threshold_db(&self) -> f64 {
        self.threshold_db
    }

    pub fn ratio(&self) -> f64 {
        self.ratio
    }

    pub fn max_reduction_db(&self) -> f64 {
        self.max_reduction_db
    }

    /// Current smoothed gain reduction in dB.
    pub fn current_gain_reduction_db(&self) -> f32 {
        self.current_reduction_db as f32
    }

    /// Smoothed detector confidence (0.0-1.0) for diagnostics.
    pub fn detector_confidence(&self) -> f32 {
        self.detector_confidence as f32
    }

    /// Per-band detector confidence for lower/core/air sibilance diagnostics.
    pub fn band_detector_confidences(&self) -> [f32; 3] {
        [
            self.bands[0].confidence as f32,
            self.bands[1].confidence as f32,
            self.bands[2].confidence as f32,
        ]
    }

    /// Per-band smoothed reduction in dB for lower/core/air sibilance diagnostics.
    pub fn band_gain_reductions_db(&self) -> [f32; 3] {
        [
            self.bands[0].reduction_db as f32,
            self.bands[1].reduction_db as f32,
            self.bands[2].reduction_db as f32,
        ]
    }

    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        if !self.enabled {
            self.current_reduction_db = 0.0;
            self.detector_confidence = 0.0;
            return input;
        }

        let broadband_level = input.abs() as f64;
        self.broadband_env = self.update_env(self.broadband_env, broadband_level);

        let detector_attack = self.detector_attack_coeff;
        let detector_release = self.detector_release_coeff;
        let mut band_level_db = [0.0_f64; DEESSER_BAND_COUNT];
        let mut total_sibilance_env = 0.0_f64;
        let mut max_sibilance_env = 0.0_f64;

        for (index, band) in self.bands.iter_mut().enumerate() {
            let sidechain_hp = band.detector_hp.process_sample(input);
            let sidechain = band.detector_lp.process_sample(sidechain_hp);
            band.env = Self::smooth_value(
                band.env,
                sidechain.abs() as f64,
                detector_attack,
                detector_release,
            );
            total_sibilance_env += band.env;
            max_sibilance_env = max_sibilance_env.max(band.env);
            band_level_db[index] = util::linear_to_db(band.env, 1e-10);
        }

        // Estimate "voice body" reference by discounting all sibilance-band energy.
        let voice_reference_level = (self.broadband_env
            - total_sibilance_env * VOICE_REFERENCE_SIDECHAIN_DISCOUNT)
            .max(1e-8);
        let voice_reference_db = util::linear_to_db(voice_reference_level, 1e-10);
        let narrowness = if total_sibilance_env > 1e-10 {
            max_sibilance_env / total_sibilance_env
        } else {
            0.0
        };

        let amount = self.auto_amount.clamp(0.0, 1.0);
        let trigger_offset_db = Self::lerp(8.0, 0.8, amount);
        let slope = Self::lerp(0.08, 1.9, amount);
        let auto_cap = Self::lerp(0.8, 14.0, amount);
        let confidence_floor = Self::lerp(0.28, 0.06, amount);
        let baseline_fall = self.auto_baseline_fall_coeff();
        let baseline_rise = self.auto_baseline_rise_coeff();
        let baseline_inactive = self.auto_baseline_inactive_decay_coeff();
        let mut target_reductions = [0.0_f64; DEESSER_BAND_COUNT];
        let mut target_sum = 0.0_f64;
        let mut aggregate_confidence = 0.0_f64;

        for index in 0..DEESSER_BAND_COUNT {
            let sidechain_level_db = band_level_db[index];
            let spectral_ratio_db =
                Self::detector_spectral_ratio_db(sidechain_level_db, voice_reference_db);
            let band_dominance = if max_sibilance_env > 1e-10 {
                (self.bands[index].env / max_sibilance_env).sqrt()
            } else {
                0.0
            };
            let confidence_target = Self::detector_confidence_target(
                sidechain_level_db,
                voice_reference_db,
                narrowness,
            ) * band_dominance;
            let band = &mut self.bands[index];
            band.confidence = Self::smooth_value(
                band.confidence,
                confidence_target.clamp(0.0, 1.0),
                detector_attack,
                detector_release,
            );
            aggregate_confidence = aggregate_confidence.max(band.confidence);

            let target_reduction = if self.auto_enabled {
                let voice_active = voice_reference_db > -55.0 || sidechain_level_db > -55.0;
                if voice_active {
                    let baseline_target = (spectral_ratio_db * 0.45).clamp(0.0, 24.0);
                    let baseline_coeff = if baseline_target < band.baseline_excess_db {
                        baseline_fall
                    } else {
                        baseline_rise
                    };
                    band.baseline_excess_db = baseline_coeff * band.baseline_excess_db
                        + (1.0 - baseline_coeff) * baseline_target;
                } else {
                    band.baseline_excess_db *= baseline_inactive;
                }

                let cap_db = auto_cap.min(self.max_reduction_db * 0.75);
                let confidence_gain =
                    Self::confidence_reduction_gain(band.confidence, confidence_floor);
                let over_db =
                    (spectral_ratio_db - band.baseline_excess_db - trigger_offset_db).max(0.0);
                (over_db * slope * confidence_gain).clamp(0.0, cap_db)
            } else if sidechain_level_db > self.threshold_db {
                let ratio_threshold_db = ((self.threshold_db + 60.0) * 0.10).clamp(0.0, 6.0);
                let level_over_db = sidechain_level_db - self.threshold_db;
                let ratio_over_db = spectral_ratio_db - ratio_threshold_db;
                if ratio_over_db > 0.0 {
                    let over_db = level_over_db.min(ratio_over_db);
                    let confidence_gain = Self::confidence_reduction_gain(band.confidence, 0.22);
                    ((1.0 - (1.0 / self.ratio)) * over_db * confidence_gain)
                        .clamp(0.0, self.max_reduction_db * 0.75)
                } else {
                    0.0
                }
            } else {
                0.0
            };
            target_reductions[index] = target_reduction;
            target_sum += target_reduction;
        }

        if target_sum > self.max_reduction_db && target_sum > 0.0 {
            let scale = self.max_reduction_db / target_sum;
            for target in &mut target_reductions {
                *target *= scale;
            }
        }

        let mut processed = input;
        let mut total_reduction = 0.0_f64;
        for (band, target_reduction) in self.bands.iter_mut().zip(target_reductions) {
            band.reduction_db = Self::smooth_value(
                band.reduction_db,
                target_reduction,
                self.attack_coeff,
                self.release_coeff,
            );
            total_reduction += band.reduction_db;
            let dynamic_gain_db = -band.reduction_db;
            if (band.dynamic_eq.gain_db() - dynamic_gain_db).abs() > 0.001 {
                band.dynamic_eq.set_gain_db_immediate(dynamic_gain_db);
            }
            processed = band.dynamic_eq.process_sample(processed);
        }
        self.current_reduction_db = total_reduction.min(self.max_reduction_db);
        self.detector_confidence = aggregate_confidence.clamp(0.0, 1.0);
        processed
    }

    /// Process a full block in place.
    pub fn process_block_inplace(&mut self, buffer: &mut [f32]) {
        if !self.enabled {
            self.current_reduction_db = 0.0;
            self.detector_confidence = 0.0;
            return;
        }

        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.current_reduction_db = 0.0;
        self.broadband_env = 0.0;
        self.detector_confidence = 0.0;
        for band in &mut self.bands {
            band.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disabled_passthrough() {
        let mut deesser = DeEsser::new(48_000.0);
        deesser.set_enabled(false);
        let x = 0.25f32;
        let y = deesser.process_sample(x);
        assert_eq!(x, y);
    }

    #[test]
    fn test_reduction_on_sibilance_band() {
        let mut deesser = DeEsser::new(48_000.0);
        deesser.set_enabled(true);
        deesser.set_auto_enabled(false);
        deesser.set_threshold_db(-40.0);
        deesser.set_ratio(8.0);
        deesser.set_max_reduction_db(12.0);

        // 7kHz tone should trigger detector band.
        let freq = 7000.0f64;
        let sr = 48_000.0f64;
        let mut sum_in = 0.0f64;
        let mut sum_out = 0.0f64;
        for n in 0..4800 {
            let x = (2.0 * std::f64::consts::PI * freq * (n as f64 / sr)).sin() as f32 * 0.35;
            let y = deesser.process_sample(x);
            sum_in += (x as f64).abs();
            sum_out += (y as f64).abs();
        }

        assert!(sum_out < sum_in, "Expected de-esser attenuation");
        assert!(deesser.current_gain_reduction_db() > 0.1);
    }

    #[test]
    fn test_max_reduction_cap() {
        let mut deesser = DeEsser::new(48_000.0);
        deesser.set_enabled(true);
        deesser.set_threshold_db(-50.0);
        deesser.set_ratio(20.0);
        deesser.set_auto_enabled(false);
        deesser.set_max_reduction_db(3.0);

        let mut max_seen = 0.0f32;
        for _ in 0..10_000 {
            let _ = deesser.process_sample(0.8);
            max_seen = max_seen.max(deesser.current_gain_reduction_db());
        }

        assert!(max_seen <= 3.2, "Expected reduction to be capped near 3dB");
    }

    #[test]
    fn test_auto_amount_increases_reduction_strength() {
        fn render_with_amount(amount: f64) -> (f64, f32) {
            let mut deesser = DeEsser::new(48_000.0);
            deesser.set_enabled(true);
            deesser.set_auto_enabled(true);
            deesser.set_auto_amount(amount);
            deesser.set_max_reduction_db(12.0);

            let mut sum_out = 0.0f64;
            let mut peak_reduction = 0.0f32;
            let sr = 48_000.0f64;
            for n in 0..24_000 {
                // Sibilance-heavy synthetic signal: dominant 7kHz + mild low component.
                let t = n as f64 / sr;
                let x = (2.0 * std::f64::consts::PI * 7000.0 * t).sin() as f32 * 0.40
                    + (2.0 * std::f64::consts::PI * 500.0 * t).sin() as f32 * 0.02;
                let y = deesser.process_sample(x);
                sum_out += (y as f64).abs();
                peak_reduction = peak_reduction.max(deesser.current_gain_reduction_db());
            }

            (sum_out, peak_reduction)
        }

        let (sum_low, gr_low) = render_with_amount(0.2);
        let (sum_high, gr_high) = render_with_amount(1.0);

        assert!(
            sum_high < sum_low,
            "Higher auto amount should attenuate more"
        );
        assert!(gr_high > gr_low, "Higher auto amount should report more GR");
    }

    #[test]
    fn test_auto_baseline_coefficients_are_sample_rate_aware() {
        let deesser_44 = DeEsser::new(44_100.0);
        let deesser_96 = DeEsser::new(96_000.0);

        let one_second_44 = deesser_44
            .auto_baseline_rise_coeff()
            .powf(deesser_44.sample_rate);
        let one_second_96 = deesser_96
            .auto_baseline_rise_coeff()
            .powf(deesser_96.sample_rate);

        assert!((one_second_44 - one_second_96).abs() < 1e-6);
    }

    #[test]
    fn test_dynamic_eq_identity_when_reduction_is_zero() {
        let mut deesser = DeEsser::new(48_000.0);
        deesser.set_enabled(true);
        deesser.set_auto_enabled(false);
        deesser.set_threshold_db(-6.0);
        deesser.set_ratio(1.0);

        let mut max_err = 0.0f32;
        for n in 0..4_800 {
            let t = n as f64 / 48_000.0;
            let x = (2.0 * std::f64::consts::PI * 7_000.0 * t).sin() as f32 * 0.35
                + (2.0 * std::f64::consts::PI * 200.0 * t).sin() as f32 * 0.15;
            let y = deesser.process_sample(x);
            max_err = max_err.max((x - y).abs());
        }

        assert!(
            max_err < 1e-4,
            "dynamic EQ should be identity with zero reduction"
        );
    }

    #[test]
    fn test_dynamic_eq_output_stays_finite() {
        let mut deesser = DeEsser::new(48_000.0);
        deesser.set_enabled(true);
        deesser.set_auto_enabled(false);
        deesser.set_threshold_db(-45.0);
        deesser.set_ratio(12.0);
        deesser.set_max_reduction_db(12.0);

        let mut peak = 0.0f32;
        for n in 0..10_000 {
            let t = n as f64 / 48_000.0;
            let x = (2.0 * std::f64::consts::PI * 7_500.0 * t).sin() as f32 * 0.8
                + (2.0 * std::f64::consts::PI * 350.0 * t).sin() as f32 * 0.35;
            let y = deesser.process_sample(x);
            assert!(
                y.is_finite(),
                "dynamic-EQ de-esser produced non-finite sample"
            );
            peak = peak.max(y.abs());
        }

        assert!(
            peak < 2.0,
            "unexpectedly large dynamic-EQ excursion: {}",
            peak
        );
    }

    #[test]
    fn test_dynamic_eq_preserves_low_frequency_when_deessing() {
        let mut deesser = DeEsser::new(48_000.0);
        deesser.set_enabled(true);
        deesser.set_auto_enabled(false);
        deesser.set_threshold_db(-45.0);
        deesser.set_ratio(12.0);
        deesser.set_max_reduction_db(12.0);

        let sr = 48_000.0f64;
        let mut low_sum_in = 0.0f64;
        let mut low_sum_out = 0.0f64;
        for n in 0..12_000 {
            let t = n as f64 / sr;
            let low = (2.0 * std::f64::consts::PI * 250.0 * t).sin() as f32 * 0.25;
            let sib = (2.0 * std::f64::consts::PI * 7_000.0 * t).sin() as f32 * 0.35;
            let y = deesser.process_sample(low + sib);
            low_sum_in += low.abs() as f64;
            low_sum_out += (y as f64).abs();
        }

        assert!(low_sum_out > low_sum_in * 0.5);
    }

    #[test]
    fn test_ratio_detector_avoids_broadband_bright_over_reduction() {
        let mut deesser = DeEsser::new(48_000.0);
        deesser.set_enabled(true);
        deesser.set_auto_enabled(true);
        deesser.set_auto_amount(1.0);
        deesser.set_max_reduction_db(12.0);

        let sr = 48_000.0f64;
        for n in 0..24_000 {
            let t = n as f64 / sr;
            let x = (2.0 * std::f64::consts::PI * 500.0 * t).sin() as f32 * 0.12
                + (2.0 * std::f64::consts::PI * 7_000.0 * t).sin() as f32 * 0.06;
            deesser.process_sample(x);
        }

        let broadband_reduction = deesser.current_gain_reduction_db();
        let broadband_confidence = deesser.detector_confidence();

        deesser.reset();
        for n in 0..12_000 {
            let t = n as f64 / sr;
            let x = (2.0 * std::f64::consts::PI * 500.0 * t).sin() as f32 * 0.08;
            deesser.process_sample(x);
        }
        for n in 0..4_800 {
            let t = n as f64 / sr;
            let x = (2.0 * std::f64::consts::PI * 500.0 * t).sin() as f32 * 0.04
                + (2.0 * std::f64::consts::PI * 7_000.0 * t).sin() as f32 * 0.35;
            deesser.process_sample(x);
        }

        let sibilance_confidence = deesser.detector_confidence();
        assert!(deesser.current_gain_reduction_db() > broadband_reduction + 0.5);
        assert!(
            sibilance_confidence > broadband_confidence + 0.15,
            "sibilance confidence ({sibilance_confidence}) should exceed broadband confidence ({broadband_confidence})"
        );
    }

    #[test]
    fn test_voice_supported_sibilance_is_not_suppressed_by_confidence() {
        let mut deesser = DeEsser::new(48_000.0);
        deesser.set_enabled(true);
        deesser.set_auto_enabled(true);
        deesser.set_auto_amount(0.85);
        deesser.set_max_reduction_db(10.0);

        let sr = 48_000.0f64;
        for n in 0..12_000 {
            let t = n as f64 / sr;
            let voice_body = (2.0 * std::f64::consts::PI * 450.0 * t).sin() as f32 * 0.10;
            deesser.process_sample(voice_body);
        }
        for n in 0..6_000 {
            let t = n as f64 / sr;
            let voice_body = (2.0 * std::f64::consts::PI * 450.0 * t).sin() as f32 * 0.05;
            let sibilance = (2.0 * std::f64::consts::PI * 7_200.0 * t).sin() as f32 * 0.32;
            deesser.process_sample(voice_body + sibilance);
        }

        assert!(
            deesser.detector_confidence() > 0.25,
            "voice-supported sibilance should retain useful detector confidence"
        );
        assert!(
            deesser.current_gain_reduction_db() > 0.25,
            "voice-supported sibilance should still trigger de-essing"
        );
    }

    #[test]
    fn test_multiband_detector_follows_moving_sibilance_peak() {
        fn render_tone(freq_hz: f64) -> ([f32; 3], [f32; 3]) {
            let mut deesser = DeEsser::new(48_000.0);
            deesser.set_enabled(true);
            deesser.set_auto_enabled(true);
            deesser.set_auto_amount(1.0);
            deesser.set_max_reduction_db(12.0);

            let sr = 48_000.0f64;
            for n in 0..18_000 {
                let t = n as f64 / sr;
                let voice_body = (2.0 * std::f64::consts::PI * 420.0 * t).sin() as f32 * 0.08;
                let sibilance = (2.0 * std::f64::consts::PI * freq_hz * t).sin() as f32 * 0.34;
                deesser.process_sample(voice_body + sibilance);
            }

            (
                deesser.band_detector_confidences(),
                deesser.band_gain_reductions_db(),
            )
        }

        let (low_conf, low_reduction) = render_tone(5_000.0);
        let (air_conf, air_reduction) = render_tone(9_200.0);

        assert!(
            low_conf[0] > low_conf[2] && low_reduction[0] > low_reduction[2],
            "5kHz sibilance should favor lower band: conf={low_conf:?} gr={low_reduction:?}"
        );
        assert!(
            air_conf[2] > air_conf[0] && air_reduction[2] > air_reduction[0],
            "9.2kHz sibilance should favor air band: conf={air_conf:?} gr={air_reduction:?}"
        );
    }

    #[test]
    fn test_multiband_budget_limits_broadband_bright_voice() {
        let mut deesser = DeEsser::new(48_000.0);
        deesser.set_enabled(true);
        deesser.set_auto_enabled(true);
        deesser.set_auto_amount(1.0);
        deesser.set_max_reduction_db(6.0);

        let sr = 48_000.0f64;
        for n in 0..24_000 {
            let t = n as f64 / sr;
            let voice_body = (2.0 * std::f64::consts::PI * 480.0 * t).sin() as f32 * 0.14;
            let bright_low = (2.0 * std::f64::consts::PI * 4_800.0 * t).sin() as f32 * 0.05;
            let bright_core = (2.0 * std::f64::consts::PI * 7_000.0 * t).sin() as f32 * 0.05;
            let bright_air = (2.0 * std::f64::consts::PI * 9_500.0 * t).sin() as f32 * 0.05;
            deesser.process_sample(voice_body + bright_low + bright_core + bright_air);
        }

        let reductions = deesser.band_gain_reductions_db();
        let total_reduction: f32 = reductions.iter().sum();
        assert!(
            total_reduction <= 6.05,
            "stacked multiband reduction should honor budget: {reductions:?}"
        );
        assert!(
            deesser.detector_confidence() < 0.85,
            "broadband bright voice should not look like fully narrow sibilance"
        );
    }

    #[test]
    fn test_detector_confidence_and_reduction_stay_finite_at_extreme_settings() {
        let mut deesser = DeEsser::new(48_000.0);
        deesser.set_enabled(true);
        deesser.set_auto_enabled(true);
        deesser.set_auto_amount(1.0);
        deesser.set_low_cut_hz(12_000.0);
        deesser.set_high_cut_hz(12_050.0);
        deesser.set_attack_ms(0.1);
        deesser.set_release_ms(5.0);
        deesser.set_max_reduction_db(24.0);

        let sr = 48_000.0f64;
        for n in 0..20_000 {
            let t = n as f64 / sr;
            let x = (2.0 * std::f64::consts::PI * 11_900.0 * t).sin() as f32 * 0.95
                + (2.0 * std::f64::consts::PI * 300.0 * t).sin() as f32 * 0.30;
            let y = deesser.process_sample(x);
            assert!(y.is_finite(), "de-esser output should remain finite");
            assert!(
                deesser.current_gain_reduction_db().is_finite(),
                "gain reduction should remain finite"
            );
            assert!(
                deesser.detector_confidence().is_finite(),
                "detector confidence should remain finite"
            );
            assert!(
                (0.0..=1.0).contains(&deesser.detector_confidence()),
                "detector confidence should stay normalized"
            );
            assert!(
                deesser.current_gain_reduction_db() <= 24.1,
                "gain reduction should honor the configured cap"
            );
        }
    }
}
