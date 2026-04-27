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
const AUTO_BASELINE_FALL_MS: f64 = 13.88;
const AUTO_BASELINE_RISE_MS: f64 = 34.72;
const AUTO_BASELINE_INACTIVE_DECAY_MS: f64 = 20.82;

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
    sidechain_env: f64,
    broadband_env: f64,
    auto_baseline_excess_db: f64,
    low_cut_hz: f64,
    high_cut_hz: f64,
    sample_rate: f64,
    detector_hp: Biquad,
    detector_lp: Biquad,
    dynamic_eq: Biquad,
}

impl DeEsser {
    /// Create a de-esser with conservative voice defaults.
    pub fn new(sample_rate: f64) -> Self {
        let low_cut_hz = 4000.0;
        let high_cut_hz = 9000.0;
        let detector_q = 0.707;

        let center_hz = Self::dynamic_eq_center_hz(low_cut_hz, high_cut_hz);
        let dynamic_q = Self::dynamic_eq_q(low_cut_hz, high_cut_hz);

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
            sidechain_env: 0.0,
            broadband_env: 0.0,
            auto_baseline_excess_db: 0.0,
            low_cut_hz,
            high_cut_hz,
            sample_rate,
            detector_hp: Biquad::new(
                BiquadType::HighPass,
                low_cut_hz,
                0.0,
                detector_q,
                sample_rate,
            ),
            detector_lp: Biquad::new(
                BiquadType::LowPass,
                high_cut_hz,
                0.0,
                detector_q,
                sample_rate,
            ),
            dynamic_eq: Biquad::new(BiquadType::Peaking, center_hz, 0.0, dynamic_q, sample_rate),
        }
    }

    #[inline]
    fn update_env(&self, prev: f64, input: f64) -> f64 {
        let coeff = if input > prev {
            self.detector_attack_coeff
        } else {
            self.detector_release_coeff
        };
        coeff * prev + (1.0 - coeff) * input
    }

    #[inline]
    fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + (b - a) * t
    }

    fn rebuild_detector_filters(&mut self) {
        self.detector_hp.set_frequency(self.low_cut_hz);
        self.detector_lp.set_frequency(self.high_cut_hz);
        self.rebuild_dynamic_eq_shape();
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

    fn rebuild_dynamic_eq_shape(&mut self) {
        self.dynamic_eq.set_frequency(Self::dynamic_eq_center_hz(
            self.low_cut_hz,
            self.high_cut_hz,
        ));
        self.dynamic_eq
            .set_q(Self::dynamic_eq_q(self.low_cut_hz, self.high_cut_hz));
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

    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        if !self.enabled {
            self.current_reduction_db = 0.0;
            return input;
        }

        let sidechain_hp = self.detector_hp.process_sample(input);
        let sidechain = self.detector_lp.process_sample(sidechain_hp);
        let sidechain_level = sidechain.abs() as f64;
        let broadband_level = input.abs() as f64;
        self.sidechain_env = self.update_env(self.sidechain_env, sidechain_level);
        self.broadband_env = self.update_env(self.broadband_env, broadband_level);

        let sidechain_level_db = util::linear_to_db(self.sidechain_env, 1e-10);
        // Estimate "voice body" reference by discounting sidechain contribution.
        let voice_reference_level = (self.broadband_env
            - self.sidechain_env * VOICE_REFERENCE_SIDECHAIN_DISCOUNT)
            .max(1e-8);
        let voice_reference_db = util::linear_to_db(voice_reference_level, 1e-10);

        let target_reduction = if self.auto_enabled {
            let amount = self.auto_amount.clamp(0.0, 1.0);
            let excess_db = (sidechain_level_db - voice_reference_db).max(0.0);
            let voice_active = voice_reference_db > -55.0 || sidechain_level_db > -55.0;

            if voice_active {
                let baseline_target = excess_db.clamp(0.0, 24.0);
                // Learn baseline slowly; decay faster than rise to stay responsive.
                let baseline_coeff = if baseline_target < self.auto_baseline_excess_db {
                    self.auto_baseline_fall_coeff()
                } else {
                    self.auto_baseline_rise_coeff()
                };
                self.auto_baseline_excess_db = baseline_coeff * self.auto_baseline_excess_db
                    + (1.0 - baseline_coeff) * baseline_target;
            } else {
                self.auto_baseline_excess_db *= self.auto_baseline_inactive_decay_coeff();
            }

            let trigger_offset_db = Self::lerp(3.5, 0.8, amount);
            let slope = Self::lerp(0.55, 1.75, amount);
            let auto_cap = Self::lerp(3.0, 14.0, amount);
            let cap_db = auto_cap.min(self.max_reduction_db);

            let over_db = (excess_db - self.auto_baseline_excess_db - trigger_offset_db).max(0.0);
            (over_db * slope).clamp(0.0, cap_db)
        } else if sidechain_level_db > self.threshold_db {
            let over_db = sidechain_level_db - self.threshold_db;
            ((1.0 - (1.0 / self.ratio)) * over_db).clamp(0.0, self.max_reduction_db)
        } else {
            0.0
        };

        let coeff = if target_reduction > self.current_reduction_db {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.current_reduction_db =
            coeff * self.current_reduction_db + (1.0 - coeff) * target_reduction;

        let dynamic_gain_db = -self.current_reduction_db;
        if (self.dynamic_eq.gain_db() - dynamic_gain_db).abs() > 0.001 {
            self.dynamic_eq.set_gain_db_immediate(dynamic_gain_db);
        }
        self.dynamic_eq.process_sample(input)
    }

    /// Process a full block in place.
    pub fn process_block_inplace(&mut self, buffer: &mut [f32]) {
        if !self.enabled {
            self.current_reduction_db = 0.0;
            return;
        }

        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.current_reduction_db = 0.0;
        self.sidechain_env = 0.0;
        self.broadband_env = 0.0;
        self.auto_baseline_excess_db = 0.0;
        self.detector_hp.reset();
        self.detector_lp.reset();
        self.dynamic_eq.reset();
        self.dynamic_eq.set_gain_db_immediate(0.0);
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
            let sr = 48_000.0f64;
            for n in 0..24_000 {
                // Sibilance-heavy synthetic signal: dominant 7kHz + mild low component.
                let t = n as f64 / sr;
                let x = (2.0 * std::f64::consts::PI * 7000.0 * t).sin() as f32 * 0.40
                    + (2.0 * std::f64::consts::PI * 500.0 * t).sin() as f32 * 0.02;
                let y = deesser.process_sample(x);
                sum_out += (y as f64).abs();
            }

            (sum_out, deesser.current_gain_reduction_db())
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
}
