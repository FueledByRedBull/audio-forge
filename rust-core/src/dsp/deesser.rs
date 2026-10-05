//! Dynamic-EQ de-esser using sidechain sibilance detection.
//!
//! Detection path:
//! - Three detector bands partition the configured cutoff interval.
//! - Automatic mode uses unit-peak band-pass detectors with the configured edges.
//! - Manual mode retains the high-pass/low-pass detector bank.
//! - Absolute linear envelope smoothing followed by dB gain calculations.
//!
//! Gain computer:
//! - Threshold/ratio above threshold
//! - Attack/release smoothing
//! - Max reduction clamp
//!
//! Apply path:
//! - Detector drives a dynamic peaking EQ in the sibilance region.

use super::biquad::{Biquad, BiquadType};
use super::eq::EQ_NYQUIST_MARGIN_HZ;
use crate::dsp::util;

const DETECTOR_RATIO_GATE_DB: f64 = 1.5;
const DETECTOR_RATIO_FULL_DB: f64 = 10.0;
const DETECTOR_LEVEL_GATE_DB: f64 = -62.0;
const DETECTOR_LEVEL_FULL_DB: f64 = -24.0;
const DETECTOR_VOICE_GATE_DB: f64 = -58.0;
const DETECTOR_VOICE_FULL_DB: f64 = -34.0;
const NARROW_SIBILANCE_SUPPORT_START_DB: f64 = 6.0;
const NARROW_SIBILANCE_SUPPORT_START_LEVEL_DB: f64 = -45.0;
const DEESSER_BAND_COUNT: usize = 3;
const DEESSER_DEFAULT_HIGH_CUT_HZ: f64 = 11_000.0;
// Heuristic control-envelope time constants, chosen as simple response times
// rather than claimed psychoacoustic measurements.
const AUTO_BASELINE_FALL_MS: f64 = 14.0;
const AUTO_BASELINE_RISE_MS: f64 = 35.0;
const AUTO_BASELINE_INACTIVE_DECAY_MS: f64 = 21.0;
const VOICE_REFERENCE_LOW_HZ: f64 = 250.0;
const VOICE_REFERENCE_HIGH_HZ: f64 = 2_000.0;
// These section Qs form the fourth-order Butterworth body-reference low-pass.
const VOICE_REFERENCE_LOW_PASS_Q1: f64 = 0.541_196_100_146_197;
const VOICE_REFERENCE_LOW_PASS_Q2: f64 = 1.306_562_964_876_377;
const BROADBAND_NARROWNESS_GATE: f64 = 0.34;
const BROADBAND_NARROWNESS_FULL: f64 = 0.68;

struct DeEsserBand {
    low_hz: f64,
    high_hz: f64,
    manual_bounds: Option<(f64, f64)>,
    env: f64,
    auto_env: f64,
    confidence: f64,
    reduction_db: f64,
    detector_hp: Biquad,
    detector_lp: Biquad,
    auto_detector_notch: Option<Biquad>,
    dynamic_eq: Biquad,
}

impl DeEsserBand {
    fn new(low_hz: f64, high_hz: f64, manual_bounds: Option<(f64, f64)>, sample_rate: f64) -> Self {
        let parameters = Self::auto_detector_parameters(low_hz, high_hz, sample_rate);
        let (detector_hp, detector_lp) = Self::manual_detectors(manual_bounds, sample_rate);
        let (auto_detector_notch, dynamic_eq) = match parameters {
            Some((frequency, q)) => (
                Some(Biquad::new(
                    BiquadType::Notch,
                    frequency,
                    0.0,
                    q,
                    sample_rate,
                )),
                Biquad::new(BiquadType::Peaking, frequency, 0.0, q, sample_rate),
            ),
            None => (
                None,
                Biquad::new(BiquadType::Bypass, 0.0, 0.0, 1.0, sample_rate),
            ),
        };
        Self {
            low_hz,
            high_hz,
            manual_bounds,
            env: 0.0,
            auto_env: 0.0,
            confidence: 0.0,
            reduction_db: 0.0,
            detector_hp,
            detector_lp,
            auto_detector_notch,
            dynamic_eq,
        }
    }

    fn manual_detectors(bounds: Option<(f64, f64)>, sample_rate: f64) -> (Biquad, Biquad) {
        match bounds {
            Some((low, high)) => (
                Biquad::new(BiquadType::HighPass, low, 0.0, 0.707, sample_rate),
                Biquad::new(BiquadType::LowPass, high, 0.0, 0.707, sample_rate),
            ),
            None => (
                Biquad::new(BiquadType::Bypass, 0.0, 0.0, 1.0, sample_rate),
                Biquad::new(BiquadType::Bypass, 0.0, 0.0, 1.0, sample_rate),
            ),
        }
    }

    fn is_active(&self, automatic: bool) -> bool {
        if automatic {
            self.auto_detector_notch.is_some()
        } else {
            self.manual_bounds.is_some()
        }
    }

    fn auto_detector_parameters(low_hz: f64, high_hz: f64, sample_rate: f64) -> Option<(f64, f64)> {
        if !(0.0 < low_hz
            && low_hz < high_hz
            && high_hz <= sample_rate * 0.5 - EQ_NYQUIST_MARGIN_HZ)
        {
            return None;
        }
        // 1 - H_notch is a unit-peak band-pass. Prewarping both edges gives
        // exactly -3 dB there; equal Hz widths also have equal noise bandwidth.
        let low = (std::f64::consts::PI * low_hz / sample_rate).tan();
        let high = (std::f64::consts::PI * high_hz / sample_rate).tan();
        let center = (low * high).sqrt();
        let frequency = sample_rate / std::f64::consts::PI * center.atan();
        let q = center / (high - low);
        (frequency.is_finite() && q.is_finite() && q > 0.0).then_some((frequency, q))
    }

    fn set_eq_geometry(
        &mut self,
        automatic: bool,
        auto_parameters: Option<(f64, f64)>,
        sample_rate: f64,
    ) {
        let parameters = if automatic {
            auto_parameters
        } else {
            self.manual_bounds.map(|(low, high)| {
                (
                    DeEsser::dynamic_eq_center_hz(low, high),
                    DeEsser::dynamic_eq_q(low, high),
                )
            })
        };
        if let Some((frequency, q)) = parameters {
            self.dynamic_eq.set_parameters(
                BiquadType::Peaking,
                frequency,
                self.dynamic_eq.gain_db(),
                q,
            );
        } else {
            self.dynamic_eq = Biquad::new(BiquadType::Bypass, 0.0, 0.0, 1.0, sample_rate);
            self.confidence = 0.0;
            self.reduction_db = 0.0;
        }
    }

    fn set_bounds(
        &mut self,
        low_hz: f64,
        high_hz: f64,
        manual_bounds: Option<(f64, f64)>,
        sample_rate: f64,
        automatic: bool,
    ) {
        if let (Some(_), Some((low, high))) = (self.manual_bounds, manual_bounds) {
            self.detector_hp.set_frequency(low);
            self.detector_lp.set_frequency(high);
        } else {
            (self.detector_hp, self.detector_lp) =
                Self::manual_detectors(manual_bounds, sample_rate);
            self.env = 0.0;
        }
        self.manual_bounds = manual_bounds;
        self.low_hz = low_hz;
        self.high_hz = high_hz;
        let parameters = Self::auto_detector_parameters(low_hz, high_hz, sample_rate);
        if let (Some(detector), Some((frequency, q))) =
            (self.auto_detector_notch.as_mut(), parameters)
        {
            detector.set_parameters(BiquadType::Notch, frequency, 0.0, q);
        } else {
            self.auto_detector_notch = parameters.map(|(frequency, q)| {
                Biquad::new(BiquadType::Notch, frequency, 0.0, q, sample_rate)
            });
            self.auto_env = 0.0;
        }
        self.set_eq_geometry(automatic, parameters, sample_rate);
    }

    fn reset(&mut self) {
        self.env = 0.0;
        self.auto_env = 0.0;
        self.confidence = 0.0;
        self.reduction_db = 0.0;
        self.detector_hp.reset();
        self.detector_lp.reset();
        if let Some(detector) = &mut self.auto_detector_notch {
            detector.reset();
        }
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
    auto_baseline_fall_coeff: f64,
    auto_baseline_rise_coeff: f64,
    auto_baseline_inactive_decay_coeff: f64,
    auto_background_db: f64,
    max_reduction_db: f64,
    current_reduction_db: f64,
    voice_reference_env: f64,
    voice_reference_high_pass: Biquad,
    voice_reference_low_pass_q1: Biquad,
    voice_reference_low_pass_q2: Biquad,
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
            auto_background_db: 0.0,
            auto_amount: 0.5,
            threshold_db: -28.0,
            ratio: 4.0,
            attack_coeff: util::time_constant_to_coeff(2.0, sample_rate),
            release_coeff: util::time_constant_to_coeff(80.0, sample_rate),
            detector_attack_coeff: util::time_constant_to_coeff(1.5, sample_rate),
            detector_release_coeff: util::time_constant_to_coeff(60.0, sample_rate),
            auto_baseline_fall_coeff: util::time_constant_to_coeff(
                AUTO_BASELINE_FALL_MS,
                sample_rate,
            ),
            auto_baseline_rise_coeff: util::time_constant_to_coeff(
                AUTO_BASELINE_RISE_MS,
                sample_rate,
            ),
            auto_baseline_inactive_decay_coeff: util::time_constant_to_coeff(
                AUTO_BASELINE_INACTIVE_DECAY_MS,
                sample_rate,
            ),
            max_reduction_db: 6.0,
            current_reduction_db: 0.0,
            voice_reference_env: 0.0,
            voice_reference_high_pass: Biquad::new(
                BiquadType::HighPass,
                VOICE_REFERENCE_LOW_HZ,
                0.0,
                0.707,
                sample_rate,
            ),
            voice_reference_low_pass_q1: Biquad::new(
                BiquadType::LowPass,
                VOICE_REFERENCE_HIGH_HZ.min(sample_rate * 0.45),
                0.0,
                VOICE_REFERENCE_LOW_PASS_Q1,
                sample_rate,
            ),
            voice_reference_low_pass_q2: Biquad::new(
                BiquadType::LowPass,
                VOICE_REFERENCE_HIGH_HZ.min(sample_rate * 0.45),
                0.0,
                VOICE_REFERENCE_LOW_PASS_Q2,
                sample_rate,
            ),
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
        let narrow_sibilance_support =
            0.75 * Self::normalize_range(
                spectral_ratio_db,
                NARROW_SIBILANCE_SUPPORT_START_DB,
                DETECTOR_RATIO_FULL_DB,
            ) * Self::normalize_range(
                sidechain_level_db,
                NARROW_SIBILANCE_SUPPORT_START_LEVEL_DB,
                DETECTOR_LEVEL_FULL_DB,
            );
        let voice_support = voice_conf.max(narrow_sibilance_support);
        let voice_support_floor = voice_support * 0.65;
        let balance_conf = if voice_support_floor > 0.12 {
            let support_blend = Self::normalize_range(ratio_conf, 0.12, voice_support_floor);
            ratio_conf + (voice_support_floor - ratio_conf).max(0.0) * support_blend
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
        let bounds = Self::band_bounds(self.low_cut_hz, self.high_cut_hz, self.sample_rate);
        let manual = Self::manual_band_bounds(self.low_cut_hz, self.high_cut_hz, self.sample_rate);
        for ((band, (low_hz, high_hz)), manual_bounds) in
            self.bands.iter_mut().zip(bounds).zip(manual)
        {
            band.set_bounds(
                low_hz,
                high_hz,
                manual_bounds,
                self.sample_rate,
                self.auto_enabled,
            );
        }
    }

    fn band_bounds(
        low_cut_hz: f64,
        high_cut_hz: f64,
        sample_rate: f64,
    ) -> [(f64, f64); DEESSER_BAND_COUNT] {
        let high_cut_hz = high_cut_hz.min(sample_rate * 0.5 - EQ_NYQUIST_MARGIN_HZ);
        if high_cut_hz <= low_cut_hz {
            return [(0.0, 0.0); DEESSER_BAND_COUNT];
        }
        Self::partition_bounds(low_cut_hz, high_cut_hz)
    }

    fn partition_bounds(low_cut_hz: f64, high_cut_hz: f64) -> [(f64, f64); DEESSER_BAND_COUNT] {
        let span = high_cut_hz - low_cut_hz;
        let split_a = low_cut_hz + span / 3.0;
        let split_b = low_cut_hz + span * 2.0 / 3.0;
        [
            (low_cut_hz, split_a),
            (split_a, split_b),
            (split_b, high_cut_hz),
        ]
    }

    fn manual_band_bounds(
        low_cut_hz: f64,
        high_cut_hz: f64,
        sample_rate: f64,
    ) -> [Option<(f64, f64)>; DEESSER_BAND_COUNT] {
        // Preserve each valid original manual band; clipping an invalid upper
        // neighbor must not redistribute the lower band's detector or EQ.
        Self::partition_bounds(low_cut_hz, high_cut_hz).map(|(low, high)| {
            let high = high.min(sample_rate * 0.5 - EQ_NYQUIST_MARGIN_HZ);
            (low < high).then_some((low, high))
        })
    }

    fn make_bands(
        low_cut_hz: f64,
        high_cut_hz: f64,
        sample_rate: f64,
    ) -> [DeEsserBand; DEESSER_BAND_COUNT] {
        let automatic = Self::band_bounds(low_cut_hz, high_cut_hz, sample_rate);
        let manual = Self::manual_band_bounds(low_cut_hz, high_cut_hz, sample_rate);
        std::array::from_fn(|index| {
            DeEsserBand::new(
                automatic[index].0,
                automatic[index].1,
                manual[index],
                sample_rate,
            )
        })
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
        if self.auto_enabled == enabled {
            return;
        }
        self.auto_enabled = enabled;
        for band in &mut self.bands {
            let parameters =
                DeEsserBand::auto_detector_parameters(band.low_hz, band.high_hz, self.sample_rate);
            band.set_eq_geometry(enabled, parameters, self.sample_rate);
        }
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
        let low_cut_hz = low_cut_hz.clamp(2000.0, 12000.0);
        let mut high_cut_hz = self.high_cut_hz;
        if high_cut_hz <= low_cut_hz + 200.0 {
            high_cut_hz = (low_cut_hz + 200.0).clamp(2200.0, 16000.0);
        }
        if low_cut_hz == self.low_cut_hz && high_cut_hz == self.high_cut_hz {
            return;
        }
        self.low_cut_hz = low_cut_hz;
        self.high_cut_hz = high_cut_hz;
        self.rebuild_detector_filters();
    }

    /// Set detector high cutoff (low-pass edge) in Hz.
    pub fn set_high_cut_hz(&mut self, high_cut_hz: f64) {
        let high_cut_hz = high_cut_hz.clamp(2200.0, 16000.0);
        let mut low_cut_hz = self.low_cut_hz;
        if high_cut_hz <= low_cut_hz + 200.0 {
            low_cut_hz = (high_cut_hz - 200.0).clamp(2000.0, 12000.0);
        }
        if low_cut_hz == self.low_cut_hz && high_cut_hz == self.high_cut_hz {
            return;
        }
        self.low_cut_hz = low_cut_hz;
        self.high_cut_hz = high_cut_hz;
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

    fn project_auto_reductions(
        requested: [f64; DEESSER_BAND_COUNT],
        cap_db: f64,
        budget_db: f64,
    ) -> [f64; DEESSER_BAND_COUNT] {
        let reductions = requested.map(|value| value.clamp(0.0, cap_db));
        let mut sum: f64 = reductions.iter().sum();
        if sum <= budget_db {
            return reductions;
        }
        // Project the original requested dB depths onto the capped budget:
        // g_i = clamp(request_i - lambda, 0, cap). Clipping the requests first
        // would discard how strongly a saturated band still needs reduction.
        let mut breakpoints = [0.0; DEESSER_BAND_COUNT * 2];
        for (index, request) in requested.iter().enumerate() {
            breakpoints[index * 2] = (request - cap_db).max(0.0);
            breakpoints[index * 2 + 1] = request.max(0.0);
        }
        breakpoints.sort_unstable_by(f64::total_cmp);
        let mut lambda = 0.0;
        for upper in breakpoints {
            let upper_sum: f64 = requested
                .iter()
                .map(|request| (request - upper).clamp(0.0, cap_db))
                .sum();
            // The sum is linear between successive changes of active bounds.
            // The final breakpoint zeros every request, so it always brackets
            // a feasible solution for nonnegative caps and budget.
            if upper_sum <= budget_db {
                lambda += (upper - lambda) * (sum - budget_db) / (sum - upper_sum);
                break;
            }
            lambda = upper;
            sum = upper_sum;
        }
        requested.map(|request| (request - lambda).clamp(0.0, cap_db))
    }

    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        if !self.enabled {
            self.current_reduction_db = 0.0;
            self.detector_confidence = 0.0;
            return input;
        }
        // An empty intersection with the usable frequency range has no band
        // to detect or attenuate. Partial intersections retain their coverage.
        if self
            .bands
            .iter()
            .all(|band| !band.is_active(self.auto_enabled))
        {
            self.current_reduction_db = 0.0;
            self.detector_confidence = 0.0;
            for band in &mut self.bands {
                band.confidence = 0.0;
                band.reduction_db = 0.0;
            }
            return input;
        }

        let voice_body = self
            .voice_reference_low_pass_q2
            .process_sample(
                self.voice_reference_low_pass_q1
                    .process_sample(self.voice_reference_high_pass.process_sample(input)),
            )
            .abs() as f64;
        self.voice_reference_env = self.update_env(self.voice_reference_env, voice_body);

        let detector_attack = self.detector_attack_coeff;
        let detector_release = self.detector_release_coeff;
        let mut band_level_db = [0.0_f64; DEESSER_BAND_COUNT];
        let mut band_envelopes = [0.0_f64; DEESSER_BAND_COUNT];
        let mut total_sibilance_env = 0.0_f64;
        let mut total_sibilance_power = 0.0_f64;
        let mut max_sibilance_env = 0.0_f64;

        for (index, band) in self.bands.iter_mut().enumerate() {
            if band.manual_bounds.is_some() {
                let sidechain_hp = band.detector_hp.process_sample(input);
                let sidechain = band.detector_lp.process_sample(sidechain_hp);
                band.env = Self::smooth_value(
                    band.env,
                    sidechain.abs() as f64,
                    detector_attack,
                    detector_release,
                );
            }
            // Keep both detector banks warm when the selected mode changes.
            if let Some(detector) = &mut band.auto_detector_notch {
                let auto_sidechain = input - detector.process_sample(input);
                band.auto_env = Self::smooth_value(
                    band.auto_env,
                    auto_sidechain.abs() as f64,
                    detector_attack,
                    detector_release,
                );
            }
            let envelope = if self.auto_enabled {
                band.auto_env
            } else {
                band.env
            };
            band_envelopes[index] = envelope;
            total_sibilance_env += envelope;
            total_sibilance_power += envelope * envelope;
            max_sibilance_env = max_sibilance_env.max(envelope);
            band_level_db[index] = util::linear_to_db(envelope, 1e-10);
        }

        // Measure low voice-body energy on its own low-pass path; detector bands
        // are non-orthogonal, so subtracting their envelopes from broadband level
        // can erase this reference when sibilance is strong.
        let voice_reference_db = util::linear_to_db(self.voice_reference_env, 1e-10);
        let spectral_ratios =
            band_level_db.map(|level| Self::detector_spectral_ratio_db(level, voice_reference_db));
        if self.auto_enabled {
            // Estimate common brightness once per sample, so an isolated band
            // excess is not also learned as that same band's background.
            let mut ordered = spectral_ratios;
            ordered.sort_unstable_by(f64::total_cmp);
            let target = (0.45 * ordered[1]).clamp(0.0, 24.0);
            let active =
                voice_reference_db > -55.0 || band_level_db.iter().any(|&level| level > -55.0);
            if active {
                let coefficient = if target < self.auto_background_db {
                    self.auto_baseline_fall_coeff
                } else {
                    self.auto_baseline_rise_coeff
                };
                self.auto_background_db =
                    coefficient * self.auto_background_db + (1.0 - coefficient) * target;
            } else {
                self.auto_background_db *= self.auto_baseline_inactive_decay_coeff;
            }
        }
        let narrowness = if self.auto_enabled && total_sibilance_power > 1e-20 {
            max_sibilance_env * max_sibilance_env / total_sibilance_power
        } else if !self.auto_enabled && total_sibilance_env > 1e-10 {
            max_sibilance_env / total_sibilance_env
        } else {
            0.0
        };

        let amount = self.auto_amount.clamp(0.0, 1.0);
        let trigger_offset_db = Self::lerp(8.0, 0.8, amount);
        let slope = Self::lerp(0.08, 1.9, amount);
        let auto_cap = Self::lerp(0.8, 14.0, amount);
        let confidence_floor = Self::lerp(0.28, 0.06, amount);
        let mut target_reductions = [0.0_f64; DEESSER_BAND_COUNT];
        let mut target_sum = 0.0_f64;
        let mut aggregate_confidence = 0.0_f64;

        for index in 0..DEESSER_BAND_COUNT {
            let sidechain_level_db = band_level_db[index];
            let spectral_ratio_db = spectral_ratios[index];
            let band_dominance = if max_sibilance_env > 1e-10 {
                (band_envelopes[index] / max_sibilance_env).sqrt()
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
                let confidence_gain =
                    Self::confidence_reduction_gain(band.confidence, confidence_floor);
                let over_db =
                    (spectral_ratio_db - self.auto_background_db - trigger_offset_db).max(0.0);
                over_db * slope * confidence_gain
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

        if self.auto_enabled {
            target_reductions = Self::project_auto_reductions(
                target_reductions,
                auto_cap.min(self.max_reduction_db * 0.75),
                self.max_reduction_db,
            );
        } else if target_sum > self.max_reduction_db && target_sum > 0.0 {
            let scale = self.max_reduction_db / target_sum;
            for target in &mut target_reductions {
                *target *= scale;
            }
        }

        for (band, target_reduction) in self.bands.iter_mut().zip(target_reductions) {
            band.reduction_db = Self::smooth_value(
                band.reduction_db,
                target_reduction,
                self.attack_coeff,
                self.release_coeff,
            );
        }
        let smoothed_sum: f64 = self.bands.iter().map(|band| band.reduction_db).sum();
        if smoothed_sum > self.max_reduction_db && smoothed_sum > 0.0 {
            let scale = self.max_reduction_db / smoothed_sum;
            for band in &mut self.bands {
                band.reduction_db *= scale;
            }
        }

        let mut processed = input;
        let mut total_reduction = 0.0_f64;
        for band in &mut self.bands {
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
        self.auto_background_db = 0.0;
        self.current_reduction_db = 0.0;
        self.voice_reference_env = 0.0;
        self.detector_confidence = 0.0;
        self.voice_reference_high_pass.reset();
        self.voice_reference_low_pass_q1.reset();
        self.voice_reference_low_pass_q2.reset();
        for band in &mut self.bands {
            band.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_requests_follow_one_shared_background_across_rates() {
        for rate in [16_000.0, 44_100.0, 48_000.0, 96_000.0] {
            let mut deesser = DeEsser::new(rate);
            deesser.set_enabled(true);
            let mut background = 0.0;
            for n in 0..9_600 {
                let phase = 2.0 * std::f64::consts::PI * n as f64 / rate;
                let x = if !(317..=8_000).contains(&n) {
                    0.0
                } else {
                    (0.04 * (500.0 * phase).sin()
                        + 0.18 * ((rate * 0.37).min(6333.333333333334) * phase).sin()
                        + 0.07 * ((rate * 0.41).min(8666.666666666668) * phase).sin())
                        as f32
                };
                let before = deesser.bands.each_ref().map(|band| band.reduction_db);
                deesser.process_sample(x);
                let voice = util::linear_to_db(deesser.voice_reference_env, 1e-10);
                let levels = deesser
                    .bands
                    .each_ref()
                    .map(|band| util::linear_to_db(band.auto_env, 1e-10));
                let ratios = levels.map(|level| (level - voice).max(0.0));
                // Independent median expression and once-per-sample recurrence.
                let median = ratios[0]
                    .max(ratios[1].min(ratios[2]))
                    .min(ratios[1].max(ratios[2]));
                let target = (0.45 * median).clamp(0.0, 24.0);
                if voice > -55.0 || levels.iter().any(|&level| level > -55.0) {
                    let coefficient = if target < background {
                        deesser.auto_baseline_fall_coeff
                    } else {
                        deesser.auto_baseline_rise_coeff
                    };
                    background = coefficient * background + (1.0 - coefficient) * target;
                } else {
                    background *= deesser.auto_baseline_inactive_decay_coeff;
                }
                let requests = std::array::from_fn(|i| {
                    (ratios[i] - background - DeEsser::lerp(8.0, 0.8, 0.5)).max(0.0)
                        * DeEsser::lerp(0.08, 1.9, 0.5)
                        * DeEsser::confidence_reduction_gain(
                            deesser.bands[i].confidence,
                            DeEsser::lerp(0.28, 0.06, 0.5),
                        )
                });
                let projected = DeEsser::project_auto_reductions(requests, 4.5, 6.0);
                let mut expected = std::array::from_fn::<_, 3, _>(|i| {
                    DeEsser::smooth_value(
                        before[i],
                        projected[i],
                        deesser.attack_coeff,
                        deesser.release_coeff,
                    )
                });
                let total: f64 = expected.iter().sum();
                if total > 6.0 {
                    for value in &mut expected {
                        *value *= 6.0 / total;
                    }
                }
                for (i, expected) in expected.iter().enumerate() {
                    assert_eq!(
                        deesser.bands[i].reduction_db.to_bits(),
                        expected.to_bits(),
                        "shared background mismatch at rate={rate}, sample={n}, band={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_shared_background_lifecycle_and_inactive_decay() {
        let mut deesser = DeEsser::new(16_000.0);
        deesser.auto_background_db = 4.0;
        for automatic in [false, true] {
            deesser.set_auto_enabled(automatic);
            deesser.set_enabled(false);
            for _ in 0..480 {
                assert_eq!(deesser.process_sample(0.25), 0.25);
            }
            assert_eq!(deesser.auto_background_db, 4.0);
        }
        deesser.set_enabled(true);
        deesser.set_auto_enabled(false);
        for _ in 0..480 {
            deesser.process_sample(0.25);
        }
        assert_eq!(deesser.auto_background_db, 4.0);
        deesser.set_auto_enabled(true);
        deesser.set_low_cut_hz(12_000.0);
        deesser.set_high_cut_hz(16_000.0);
        for _ in 0..480 {
            assert_eq!(deesser.process_sample(0.25), 0.25);
        }
        assert_eq!(deesser.auto_background_db, 4.0);
        deesser.reset();
        assert_eq!(deesser.auto_background_db, 0.0);

        let mut silent = DeEsser::new(48_000.0);
        silent.set_enabled(true);
        silent.auto_background_db = 4.0;
        silent.process_sample(0.0);
        assert_eq!(
            silent.auto_background_db,
            4.0 * silent.auto_baseline_inactive_decay_coeff
        );
        let before = silent.auto_background_db;
        silent.set_auto_amount(1.0);
        silent.set_max_reduction_db(0.0);
        silent.set_low_cut_hz(2_000.0);
        assert_eq!(
            silent.auto_background_db, before,
            "parameter changes do not reset the observer state"
        );
        silent.process_sample(0.0);
        assert_eq!(
            silent.auto_background_db,
            before * silent.auto_baseline_inactive_decay_coeff
        );
    }

    #[test]
    fn test_auto_budget_projection_matches_independent_reference_and_kkt() {
        fn reference(requested: [f64; 3], cap: f64, budget: f64) -> ([f64; 3], f64) {
            let clipped = requested.map(|value| value.clamp(0.0, cap));
            if clipped.iter().sum::<f64>() <= budget {
                return (clipped, 0.0);
            }
            // Independent numerical root solve, not the production active set.
            let mut left = 0.0;
            let mut right = requested.into_iter().fold(0.0_f64, f64::max);
            for _ in 0..100 {
                let middle = (left + right) * 0.5;
                let sum: f64 = requested
                    .iter()
                    .map(|value| (value - middle).clamp(0.0, cap))
                    .sum();
                if sum > budget {
                    left = middle;
                } else {
                    right = middle;
                }
            }
            (
                requested.map(|value| (value - right).clamp(0.0, cap)),
                right,
            )
        }

        let mut cases = Vec::new();
        for budget in [0.0, 1e-6, 1.0, 6.0, 24.0] {
            for amount in [0.0, 0.5, 1.0] {
                let cap = DeEsser::lerp(0.8, 14.0, amount).min(0.75 * budget);
                for value in [0.0, cap, budget / 3.0, budget, 100.0] {
                    for offset in [-1e-10, 0.0, 1e-10] {
                        cases.push(([value, (value + offset).max(0.0), 0.0], cap, budget));
                        cases.push(([value, value, (value + offset).max(0.0)], cap, budget));
                    }
                }
            }
        }
        cases.push(([1.342282, 10.890507, 0.0], 4.5, 6.0));
        cases.push(([2.419224, 14.757074, 0.0064], 4.5, 6.0));
        let mut seed = 0x1a04_2026_u64;
        let mut random = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 11) as f64 / ((1_u64 << 53) as f64)
        };
        for _ in 0..1024 {
            let budget = random() * 24.0;
            let cap = DeEsser::lerp(0.8, 14.0, random()).min(0.75 * budget);
            cases.push((
                [random() * 100.0, random() * 100.0, random() * 100.0],
                cap,
                budget,
            ));
        }
        for (requested, cap, budget) in cases {
            let (expected, lambda) = reference(requested, cap, budget);
            let actual = DeEsser::project_auto_reductions(requested, cap, budget);
            assert_eq!(
                actual,
                DeEsser::project_auto_reductions(requested, cap, budget)
            );
            let tolerance = 1e-9;
            assert!(actual
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0 && *value <= cap));
            let sum: f64 = actual.iter().sum();
            assert!(sum <= budget + tolerance, "projection exceeded budget");
            let clipped = requested.map(|value| value.clamp(0.0, cap));
            if clipped.iter().sum::<f64>() <= budget {
                assert_eq!(actual, clipped, "feasible vector changed");
            }
            assert!(lambda * (budget - sum).abs() <= tolerance * (1.0 + lambda));
            for index in 0..3 {
                assert!(
                    (actual[index] - expected[index]).abs() < tolerance,
                    "request={requested:?}, cap={cap}, budget={budget}: {actual:?} != {expected:?}"
                );
                let gradient = actual[index] - requested[index] + lambda;
                if cap == 0.0 {
                    continue;
                } else if actual[index] <= tolerance {
                    assert!(gradient >= -tolerance, "lower-bound KKT violation");
                } else if actual[index] >= cap - tolerance {
                    assert!(gradient <= tolerance, "upper-bound KKT violation");
                } else {
                    assert!(gradient.abs() <= tolerance, "interior KKT violation");
                }
            }
        }
    }

    #[test]
    fn test_auto_budget_projection_does_not_allocate() {
        crate::test_alloc::assert_no_allocations("automatic gain budget", || {
            std::hint::black_box(DeEsser::project_auto_reductions(
                [2.419224, 14.757074, 0.0064],
                4.5,
                6.0,
            ));
        });
    }

    fn auto_tone_gains_db(sample_rate: f64, amount: f64, amplitudes: [f64; 4]) -> [f64; 4] {
        let mut deesser = DeEsser::new(sample_rate);
        deesser.set_enabled(true);
        deesser.set_auto_amount(amount);
        let frequencies = [500.0, 4_800.0, 7_000.0, 9_500.0];
        let mut projections = [[0.0_f64; 2]; 4];
        let sample_count = sample_rate as usize;
        for n in 0..sample_count {
            let phases = frequencies
                .map(|frequency| 2.0 * std::f64::consts::PI * frequency * n as f64 / sample_rate);
            let input = phases
                .iter()
                .zip(amplitudes)
                .map(|(phase, amplitude)| phase.sin() * amplitude)
                .sum::<f64>() as f32;
            let output = deesser.process_sample(input) as f64;
            assert!(output.is_finite());
            // A settled half-second contains integer periods of every tone.
            if n >= sample_count / 2 {
                for (projection, phase) in projections.iter_mut().zip(phases) {
                    projection[0] += output * phase.sin();
                    projection[1] += output * phase.cos();
                }
            }
        }
        std::array::from_fn(|index| {
            if amplitudes[index] == 0.0 {
                return 0.0;
            }
            let amplitude = 2.0 * projections[index][0].hypot(projections[index][1])
                / (sample_count - sample_count / 2) as f64;
            20.0 * (amplitude / amplitudes[index]).log10()
        })
    }

    #[test]
    fn test_auto_keeps_attenuating_sustained_moderate_sibilance() {
        for sample_rate in [44_100.0, 48_000.0, 96_000.0] {
            let gains = auto_tone_gains_db(sample_rate, 0.5, [0.06, 0.0, 0.20, 0.0]);
            assert!(
                gains[2] < -0.15,
                "sustained 7 kHz excess was absorbed at {sample_rate} Hz: {gains:?} dB"
            );
            assert!(
                gains[0].abs() < 0.15,
                "de-essing changed the voice body at {sample_rate} Hz: {gains:?} dB"
            );
        }
    }

    #[test]
    fn test_auto_preserves_body_and_broad_brightness_across_rates_and_amounts() {
        for sample_rate in [44_100.0, 48_000.0, 96_000.0] {
            for amount in [0.0, 0.5, 1.0] {
                for amplitudes in [
                    [0.12, 0.0, 0.0, 0.0],
                    [0.12, 0.0, 0.06, 0.0],
                    [0.14, 0.05, 0.05, 0.05],
                ] {
                    let gains = auto_tone_gains_db(sample_rate, amount, amplitudes);
                    assert!(
                        gains.iter().all(|gain| gain.abs() < 0.15),
                        "body/balanced/bright tones changed at {sample_rate} Hz, amount {amount}, input {amplitudes:?}: {gains:?} dB"
                    );
                }
            }
        }
    }

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
    fn test_recommended_fixture_sibilance_triggers_deessing() {
        let mut deesser = DeEsser::new(48_000.0);
        deesser.set_enabled(true);
        deesser.set_auto_enabled(true);
        deesser.set_auto_amount(0.831_076_770_5);
        deesser.set_low_cut_hz(4_800.0);
        deesser.set_high_cut_hz(8_600.0);
        deesser.set_threshold_db(-28.0);
        deesser.set_ratio(5.5);
        deesser.set_attack_ms(2.0);
        deesser.set_release_ms(80.0);
        deesser.set_max_reduction_db(8.0);

        let sample_rate = 48_000.0;
        let mut max_reduction = 0.0f32;
        for n in 0..240_000 {
            let t = n as f64 / sample_rate;
            let gate = ((t * 3.0).floor() as usize % 4) != 3;
            let envelope = if gate {
                0.55 + 0.25 * (2.0 * std::f64::consts::PI * 2.1 * t).sin().powi(2)
            } else {
                0.0
            };
            let phase = t % 1.25;
            let sibilance = if (0.72..=0.88).contains(&phase) {
                0.30 * (2.0 * std::f64::consts::PI * 6_500.0 * t).sin()
            } else {
                0.004 * (2.0 * std::f64::consts::PI * 6_500.0 * t).sin()
            };
            let voice = envelope
                * (0.11 * (2.0 * std::f64::consts::PI * 140.0 * t).sin()
                    + 0.07 * (2.0 * std::f64::consts::PI * 220.0 * t).sin()
                    + 0.05 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                    + sibilance);
            deesser.process_sample(voice as f32);
            max_reduction = max_reduction.max(deesser.current_gain_reduction_db());
        }

        assert!(
            max_reduction > 0.25,
            "recommended 6.5 kHz fixture bursts should engage the de-esser; max GR was {max_reduction} dB"
        );
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
        for n in 0..24_000 {
            let x = (2.0 * std::f64::consts::PI * 7_000.0 * n as f64 / 48_000.0).sin() as f32 * 0.8;
            let _ = deesser.process_sample(x);
            // Inspect the actual filter budget, not the separately clamped meter.
            max_seen = max_seen.max(deesser.band_gain_reductions_db().iter().sum());
        }

        assert!(
            max_seen > 0.5,
            "The cap must be tested with an engaged detector: {max_seen}"
        );
        assert!(
            max_seen <= 3.001,
            "Expected reduction to be capped near 3dB: {max_seen}"
        );
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
    fn test_voice_reference_survives_single_and_overlapping_sibilance() {
        fn reference_db(sibilance_amplitudes: [f32; 3]) -> f64 {
            let mut deesser = DeEsser::new(48_000.0);
            deesser.set_enabled(true);
            let sample_rate = 48_000.0;
            for n in 0..24_000 {
                let t = n as f64 / sample_rate;
                let body = (2.0 * std::f64::consts::PI * 250.0 * t).sin() as f32 * 0.02;
                let sibilance = [4_800.0, 7_000.0, 9_500.0]
                    .into_iter()
                    .zip(sibilance_amplitudes)
                    .map(|(frequency, amplitude)| {
                        (2.0 * std::f64::consts::PI * frequency * t).sin() as f32 * amplitude
                    })
                    .sum::<f32>();
                deesser.process_sample(body + sibilance);
            }

            util::linear_to_db(deesser.voice_reference_env, 1e-10)
        }

        let body_only = reference_db([0.0, 0.0, 0.0]);
        for sibilance in [[0.0, 0.32, 0.0], [0.08, 0.08, 0.08]] {
            let with_sibilance = reference_db(sibilance);
            assert!(
                (with_sibilance - body_only).abs() < 3.0,
                "the same 250 Hz voice body moved from {body_only:.2} dB to {with_sibilance:.2} dB for {sibilance:?}"
            );
        }
    }

    #[test]
    fn test_sibilance_confidence_is_monotonic_and_gain_reduction_bounded() {
        fn render(sibilance_amplitude: f32) -> (f32, f32) {
            let mut deesser = DeEsser::new(48_000.0);
            deesser.set_enabled(true);
            deesser.set_auto_enabled(true);
            deesser.set_auto_amount(1.0);
            let sample_rate = 48_000.0;
            for n in 0..24_000 {
                let t = n as f64 / sample_rate;
                let body = (2.0 * std::f64::consts::PI * 250.0 * t).sin() as f32 * 0.08;
                let sibilance =
                    (2.0 * std::f64::consts::PI * 7_000.0 * t).sin() as f32 * sibilance_amplitude;
                deesser.process_sample(body + sibilance);
            }
            (
                deesser.detector_confidence(),
                deesser.current_gain_reduction_db(),
            )
        }

        // These separately settled amplitudes test confidence ordering, not
        // temporal continuity. Rendered epsilon/ramp tests cover continuity
        // down to the existing actuator resolution without a meter slope.
        let mut previous_confidence = None;
        for step in 4..=10 {
            let amplitude = step as f32 * 0.02;
            let (confidence, reduction) = render(amplitude);
            assert!((0.0..=1.0).contains(&confidence));
            assert!(
                reduction <= 6.001,
                "reduction exceeded default cap: {reduction}"
            );
            if let Some(previous) = previous_confidence {
                assert!(
                    confidence + 0.005 >= previous,
                    "confidence fell from {previous:.4} to {confidence:.4} at amplitude {amplitude:.2}"
                );
            }
            previous_confidence = Some(confidence);
        }
    }

    #[test]
    fn test_detector_confidence_is_continuous_at_support_boundaries() {
        let target = |sidechain_level_db: f64, voice_reference_db: f64| {
            DeEsser::detector_confidence_target(sidechain_level_db, voice_reference_db, 0.8)
        };
        let epsilon = 1e-6;
        let boundaries = [
            // Narrow-band voice support starts at 6 dB of spectral contrast.
            (-44.0, -50.0, 6.0),
            // Narrow-band level support starts at -45 dBFS.
            (-45.0, -60.0, 15.0),
            // Ratio confidence crosses the former 0.12 balance gate at 2.52 dB.
            (-35.0, -37.52, 2.52),
        ];

        for (sidechain_level_db, voice_reference_db, ratio_db) in boundaries {
            let (before, after) = if ratio_db == 15.0 {
                (
                    target(sidechain_level_db - epsilon, voice_reference_db - epsilon),
                    target(sidechain_level_db + epsilon, voice_reference_db + epsilon),
                )
            } else {
                (
                    target(sidechain_level_db, voice_reference_db + epsilon),
                    target(sidechain_level_db, voice_reference_db - epsilon),
                )
            };
            assert!(
                (after - before).abs() < 1e-5,
                "confidence changed discontinuously around {ratio_db} dB contrast / {sidechain_level_db} dBFS level: {before} -> {after}"
            );
        }
    }

    fn render_sibilance_envelope(amplitudes: &[f64]) -> (Vec<f32>, Vec<f32>) {
        let mut deesser = DeEsser::new(48_000.0);
        deesser.set_enabled(true);
        deesser.set_auto_amount(1.0);
        let mut input = Vec::with_capacity(amplitudes.len());
        let mut output = Vec::with_capacity(amplitudes.len());
        for (n, amplitude) in amplitudes.iter().enumerate() {
            let phase = 2.0 * std::f64::consts::PI * n as f64 / 48_000.0;
            let sample =
                (0.08 * (250.0 * phase).sin() + amplitude * (7_000.0 * phase).sin()) as f32;
            let processed = deesser.process_sample(sample);
            assert!(processed.is_finite());
            assert!(deesser.current_gain_reduction_db() <= 6.001);
            input.push(sample);
            output.push(processed);
        }
        (input, output)
    }

    #[test]
    fn test_realized_sibilant_attenuation_is_monotonic_with_input_strength() {
        let mut previous_gain = f64::INFINITY;
        // Each of three bells updates only after a 0.001 dB control change.
        let gain_quantum_db = DEESSER_BAND_COUNT as f64 * 0.001;
        for step in 4..=10 {
            let amplitude = step as f64 * 0.02;
            let (_, output) = render_sibilance_envelope(&vec![amplitude; 24_000]);
            let mut sine = 0.0;
            let mut cosine = 0.0;
            for (n, sample) in output.iter().enumerate().skip(14_400) {
                let phase = 2.0 * std::f64::consts::PI * 7_000.0 * n as f64 / 48_000.0;
                sine += *sample as f64 * phase.sin();
                cosine += *sample as f64 * phase.cos();
            }
            let gain_db = 20.0 * (2.0 * sine.hypot(cosine) / 9_600.0 / amplitude).log10();
            assert!(gain_db <= previous_gain + gain_quantum_db,
                "attenuation decreased as sibilance rose to {amplitude}: {previous_gain} -> {gain_db} dB");
            previous_gain = gain_db;
        }
    }

    #[test]
    fn test_audio_epsilon_and_ramp_response_converges_to_actuator_resolution() {
        let mut envelopes: Vec<Vec<f64>> = (4..=10)
            .map(|step| vec![step as f64 * 0.02; 24_000])
            .collect();
        envelopes.push(
            (0..96_000)
                .map(|n| 0.20 * (1.0 - (n as f64 / 48_000.0 - 1.0).abs()))
                .collect(),
        );
        for (case, envelope) in envelopes.iter().enumerate() {
            let mut previous_rms = f64::INFINITY;
            for epsilon in [1e-4, 1e-5, 1e-6] {
                let lower: Vec<f64> = envelope.iter().map(|value| value - epsilon).collect();
                let upper: Vec<f64> = envelope.iter().map(|value| value + epsilon).collect();
                let (input, output_lower) = render_sibilance_envelope(&lower);
                let (_, output_upper) = render_sibilance_envelope(&upper);
                let input_rms = (input.iter().map(|x| (*x as f64).powi(2)).sum::<f64>()
                    / input.len() as f64)
                    .sqrt();
                let difference_rms = (output_lower
                    .iter()
                    .zip(output_upper)
                    .map(|(a, b)| (*a as f64 - b as f64).powi(2))
                    .sum::<f64>()
                    / input.len() as f64)
                    .sqrt();
                // Resolve audio changes down to the existing three-bell update
                // quantum; a discrete confidence jump would exceed this floor.
                let quantization_resolution =
                    input_rms * (10.0_f64.powf(DEESSER_BAND_COUNT as f64 * 0.001 / 20.0) - 1.0);
                assert!(
                    difference_rms <= previous_rms + quantization_resolution,
                    "epsilon audio diverged for case {case}: {previous_rms} -> {difference_rms}"
                );
                if epsilon == 1e-6 {
                    assert!(
                        difference_rms <= quantization_resolution + 2.0 * epsilon,
                        "audio jump exceeds actuator resolution for case {case}: {difference_rms}"
                    );
                    println!("audio continuity case {case}: rms={difference_rms:.9}, resolution={quantization_resolution:.9}");
                }
                previous_rms = difference_rms;
            }
        }
    }

    #[test]
    fn test_low_frequency_voice_body_alone_does_not_trigger_deessing() {
        let mut deesser = DeEsser::new(48_000.0);
        deesser.set_enabled(true);
        let mut max_reduction = 0.0_f32;
        for n in 0..24_000 {
            let voice =
                (2.0 * std::f64::consts::PI * 250.0 * n as f64 / 48_000.0).sin() as f32 * 0.2;
            deesser.process_sample(voice);
            max_reduction = max_reduction.max(deesser.current_gain_reduction_db());
        }
        assert!(
            max_reduction <= 0.01,
            "250 Hz voice body alone should not trigger de-essing: {max_reduction} dB"
        );
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
    fn test_detector_bands_partition_each_permitted_cutoff_width() {
        for width_hz in [200.0, 399.0, 400.0, 599.0, 600.0] {
            let low_cut_hz = 4000.0;
            let high_cut_hz = low_cut_hz + width_hz;
            let deesser = {
                let mut value = DeEsser::new(48_000.0);
                value.set_low_cut_hz(low_cut_hz);
                value.set_high_cut_hz(high_cut_hz);
                value
            };

            assert_eq!(deesser.low_cut_hz(), low_cut_hz);
            assert_eq!(deesser.high_cut_hz(), high_cut_hz);

            for (index, band) in deesser.bands.iter().enumerate() {
                assert!(
                    band.low_hz < band.high_hz,
                    "band {index} must be ordered for {low_cut_hz}-{high_cut_hz} Hz"
                );
                assert!(
                    band.low_hz >= low_cut_hz && band.high_hz <= high_cut_hz,
                    "band {index} must stay inside {low_cut_hz}-{high_cut_hz} Hz: {}-{}",
                    band.low_hz,
                    band.high_hz
                );
                if let Some(previous) = index.checked_sub(1).and_then(|i| deesser.bands.get(i)) {
                    assert_eq!(
                        previous.high_hz, band.low_hz,
                        "bands must remain contiguous for {low_cut_hz}-{high_cut_hz} Hz"
                    );
                }
            }
        }
    }

    #[test]
    fn test_reapplying_effective_cutoffs_does_not_restart_filter_crossfades() {
        fn all_filters_settled(deesser: &DeEsser) -> bool {
            !deesser.voice_reference_high_pass.is_crossfading()
                && !deesser.voice_reference_low_pass_q1.is_crossfading()
                && !deesser.voice_reference_low_pass_q2.is_crossfading()
                && deesser.bands.iter().all(|band| {
                    !band.detector_hp.is_crossfading()
                        && !band.detector_lp.is_crossfading()
                        && band
                            .auto_detector_notch
                            .as_ref()
                            .is_none_or(|detector| !detector.is_crossfading())
                        && !band.dynamic_eq.is_crossfading()
                })
        }

        for low_cut in [true, false] {
            let mut deesser = DeEsser::new(48_000.0);
            deesser.set_enabled(true);
            deesser.set_max_reduction_db(0.0);
            if low_cut {
                deesser.set_low_cut_hz(2_000.0);
            } else {
                deesser.set_high_cut_hz(16_000.0);
            }
            assert!(
                !deesser.voice_reference_high_pass.is_crossfading()
                    && !deesser.voice_reference_low_pass_q1.is_crossfading()
                    && !deesser.voice_reference_low_pass_q2.is_crossfading(),
                "detector cutoff changes must not reconfigure the fixed voice reference"
            );

            for _ in 0..60 {
                deesser.process_sample(0.1);
            }
            if low_cut {
                // Both requests clamp to the same effective 2 kHz low edge.
                deesser.set_low_cut_hz(-1_000.0);
            } else {
                // Both requests clamp to the same effective 16 kHz high edge.
                deesser.set_high_cut_hz(20_000.0);
            }
            for _ in 0..12 {
                deesser.process_sample(0.1);
            }

            assert!(
                all_filters_settled(&deesser),
                "reapplying the same effective {} cutoff restarted a filter transition",
                if low_cut { "low" } else { "high" }
            );
        }
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
        let mut low_projection = [0.0_f64; 2];
        let mut high_projection = [0.0_f64; 2];
        // Half a second of settled signal contains integer periods of both
        // tones. Quadrature projection separates LF preservation from HF loss.
        for n in 0..48_000 {
            let t = n as f64 / sr;
            let low = (2.0 * std::f64::consts::PI * 250.0 * t).sin() as f32 * 0.05;
            let sib = (2.0 * std::f64::consts::PI * 7_000.0 * t).sin() as f32 * 0.35;
            let y = deesser.process_sample(low + sib);
            if n >= 24_000 {
                for (frequency, projection) in [
                    (250.0, &mut low_projection),
                    (7_000.0, &mut high_projection),
                ] {
                    let phase = 2.0 * std::f64::consts::PI * frequency * t;
                    projection[0] += y as f64 * phase.sin();
                    projection[1] += y as f64 * phase.cos();
                }
            }
        }

        let amplitude = |projection: [f64; 2]| 2.0 * projection[0].hypot(projection[1]) / 24_000.0;
        let low_gain_db = 20.0 * (amplitude(low_projection) / 0.05).log10();
        let high_gain_db = 20.0 * (amplitude(high_projection) / 0.35).log10();
        assert!(
            deesser.current_gain_reduction_db() > 0.5,
            "Detector must engage"
        );
        assert!(low_gain_db.abs() < 0.15, "LF changed by {low_gain_db} dB");
        assert!(high_gain_db < -1.0, "HF attenuation only {high_gain_db} dB");
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
    fn test_smoothed_moving_band_reduction_never_exceeds_budget() {
        let mut deesser = DeEsser::new(48_000.0);
        deesser.set_enabled(true);
        deesser.set_auto_enabled(true);
        deesser.set_auto_amount(1.0);
        deesser.set_max_reduction_db(4.0);

        for &frequency in &[4_800.0, 7_000.0, 9_500.0, 4_800.0] {
            for n in 0..4_800 {
                let t = n as f64 / 48_000.0;
                let body = (2.0 * std::f64::consts::PI * 420.0 * t).sin() as f32 * 0.08;
                let sibilance = (2.0 * std::f64::consts::PI * frequency * t).sin() as f32 * 0.40;
                deesser.process_sample(body + sibilance);
                let total: f32 = deesser.band_gain_reductions_db().iter().sum();
                assert!(total <= 4.001, "smoothed band budget exceeded: {total}");
            }
        }
    }

    #[test]
    fn test_auto_bandpass_separates_center_tones_across_permitted_widths() {
        for sample_rate in [
            8_000.0, 16_000.0, 22_050.0, 32_000.0, 44_100.0, 48_000.0, 96_000.0, 192_000.0,
            768_000.0,
        ] {
            for (low_hz, high_hz) in [
                (4_000.0, 11_000.0),
                (5_625.0, 9_425.0),
                (6_300.0, 10_100.0),
                (7_000.0, 7_200.0),
                (2_000.0, 2_200.0),
                (12_000.0, 16_000.0),
            ] {
                if high_hz > sample_rate * 0.5 - EQ_NYQUIST_MARGIN_HZ {
                    continue;
                }
                let width = (high_hz - low_hz) / 3.0;
                for index in 0..3 {
                    let lo = low_hz + width * index as f64;
                    let hi = lo + width;
                    let prewarp =
                        |frequency: f64| (std::f64::consts::PI * frequency / sample_rate).tan();
                    let center = sample_rate / std::f64::consts::PI
                        * (prewarp(lo) * prewarp(hi)).sqrt().atan();
                    let mut deesser = DeEsser::new(sample_rate);
                    deesser.set_enabled(true);
                    deesser.set_auto_enabled(true);
                    deesser.set_low_cut_hz(low_hz);
                    deesser.set_high_cut_hz(high_hz);
                    deesser.set_max_reduction_db(0.0);
                    for n in 0..(sample_rate as usize / 4) {
                        let t = n as f64 / sample_rate;
                        let sample = (2.0 * std::f64::consts::PI * center * t).sin() * 0.25
                            + (2.0 * std::f64::consts::PI * 500.0 * t).sin() * 0.01;
                        assert!(deesser.process_sample(sample as f32).is_finite());
                    }
                    // Near 2 kHz the tone is also part of the unchanged body
                    // reference; concentration alone must not imply sibilance.
                    if low_hz >= 4_000.0 && sample_rate >= 44_100.0 {
                        let confidence = deesser.band_detector_confidences()[index];
                        assert!(
                            confidence > 0.65,
                            "center tone under-detected at {sample_rate} Hz, {low_hz}-{high_hz} Hz, band {index}: {confidence}"
                        );
                    }
                    let own_power = deesser.bands[index].auto_env.powi(2);
                    let total_power: f64 =
                        deesser.bands.iter().map(|band| band.auto_env.powi(2)).sum();
                    assert!(
                        own_power / total_power > BROADBAND_NARROWNESS_FULL,
                        "center tone did not dominate its physical band: {} at {sample_rate} Hz, {low_hz}-{high_hz} Hz, band {index}",
                        own_power / total_power
                    );
                }
            }
        }
    }

    #[test]
    fn test_auto_bandpass_keeps_partial_nyquist_coverage_and_bypasses_empty_coverage() {
        for sample_rate in [8_000.0, 16_000.0, 22_050.0, 32_000.0] {
            let mut deesser = DeEsser::new(sample_rate);
            deesser.set_enabled(true);
            deesser.set_auto_enabled(true);
            deesser.set_auto_amount(1.0);
            deesser.set_high_cut_hz(16_000.0);
            let empty = 4_000.0 >= sample_rate * 0.5 - EQ_NYQUIST_MARGIN_HZ;
            for n in 0..(sample_rate as usize / 10) {
                let sample = (2.0 * std::f64::consts::PI * 0.36 * n as f64).sin() as f32 * 0.25;
                let output = deesser.process_sample(sample);
                assert!(output.is_finite());
                if empty {
                    assert_eq!(output, sample);
                    assert_eq!(deesser.current_gain_reduction_db(), 0.0);
                    assert_eq!(deesser.detector_confidence(), 0.0);
                    assert_eq!(deesser.band_gain_reductions_db(), [0.0; 3]);
                    assert_eq!(deesser.band_detector_confidences(), [0.0; 3]);
                }
            }
            if !empty {
                assert!(
                    deesser.current_gain_reduction_db() > 0.0,
                    "valid partial coverage was lost at {sample_rate} Hz"
                );
            }
        }
    }

    #[test]
    fn test_auto_detector_and_bell_achieve_the_same_physical_band_edges() {
        for sample_rate in [
            8_000.0, 16_000.0, 22_050.0, 32_000.0, 44_100.0, 48_000.0, 96_000.0, 192_000.0,
            768_000.0,
        ] {
            for (low, requested_high) in [
                (2_000.0_f64, 2_200.0_f64),
                (4_000.0, 11_000.0),
                (7_000.0, 7_200.0),
                (12_000.0, 16_000.0),
            ] {
                let high = requested_high.min(sample_rate * 0.5 - EQ_NYQUIST_MARGIN_HZ);
                if high <= low {
                    continue;
                }
                let mut deesser = DeEsser::new(sample_rate);
                deesser.set_low_cut_hz(low);
                deesser.set_high_cut_hz(requested_high);
                for (index, band) in deesser.bands.iter_mut().enumerate() {
                    let left = low + (high - low) * index as f64 / 3.0;
                    let right = if index == 2 {
                        high
                    } else {
                        low + (high - low) * (index + 1) as f64 / 3.0
                    };
                    let prewarp =
                        |frequency: f64| (std::f64::consts::PI * frequency / sample_rate).tan();
                    let center = sample_rate / std::f64::consts::PI
                        * (prewarp(left) * prewarp(right)).sqrt().atan();
                    let detector = band
                        .auto_detector_notch
                        .as_mut()
                        .expect("nonempty physical band");
                    detector.reset();
                    band.dynamic_eq.set_gain_db_immediate(-6.0);
                    assert!(
                        (band.dynamic_eq.magnitude_response_db(center) + 6.0).abs() < 1e-5,
                        "bell peak missed detector center at {sample_rate} Hz, {left}-{right} Hz"
                    );
                    for edge in [left, right] {
                        assert!(
                            (detector.magnitude_response_db(edge) + 10.0 * 2.0_f64.log10()).abs()
                                < 1e-5,
                            "notch complement missed its half-power edge"
                        );
                        assert!(
                            (band.dynamic_eq.magnitude_response_db(edge) + 3.0).abs() < 1e-5,
                            "bell missed its half-gain edge at {sample_rate} Hz, {left}-{right} Hz"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_manual_preserves_valid_original_partitions_and_clips_each_invalid_band() {
        for sample_rate in [8_000.0, 16_000.0, 22_050.0, 32_000.0, 48_000.0, 96_000.0] {
            for (low, high) in [
                (2_000.0, 2_200.0),
                (4_000.0, 11_000.0),
                (7_000.0, 7_200.0),
                (12_000.0, 16_000.0),
            ] {
                let mut deesser = DeEsser::new(sample_rate);
                deesser.set_enabled(true);
                deesser.set_auto_enabled(false);
                deesser.set_low_cut_hz(low);
                deesser.set_high_cut_hz(high);
                let width = (high - low) / 3.0;
                for (index, band) in deesser.bands.iter_mut().enumerate() {
                    let left = low + width * index as f64;
                    let right = (low + width * (index + 1) as f64)
                        .min(sample_rate * 0.5 - EQ_NYQUIST_MARGIN_HZ);
                    band.dynamic_eq.set_gain_db_immediate(-6.0);
                    if left >= right {
                        for frequency in [sample_rate * 0.1, sample_rate * 0.3] {
                            assert_eq!(
                                band.detector_hp.target_magnitude_response_db(frequency),
                                0.0
                            );
                            assert_eq!(
                                band.detector_lp.target_magnitude_response_db(frequency),
                                0.0
                            );
                            assert_eq!(
                                band.dynamic_eq.target_magnitude_response_db(frequency),
                                0.0
                            );
                        }
                    } else {
                        let hp = Biquad::new(BiquadType::HighPass, left, 0.0, 0.707, sample_rate);
                        let lp = Biquad::new(BiquadType::LowPass, right, 0.0, 0.707, sample_rate);
                        let center = (left * right).sqrt();
                        let q = (center / (right - left).max(200.0)).clamp(0.5, 6.0);
                        let bell = Biquad::new(BiquadType::Peaking, center, -6.0, q, sample_rate);
                        for frequency in [left, center, right] {
                            assert!((band.detector_hp.target_magnitude_response_db(frequency)
                                - hp.target_magnitude_response_db(frequency)).abs() < 1e-8,
                                "manual high-pass repartitioned: rate={sample_rate} interval={low}-{high} band={index}");
                            assert!((band.detector_lp.target_magnitude_response_db(frequency)
                                - lp.target_magnitude_response_db(frequency)).abs() < 1e-8,
                                "manual low-pass repartitioned: rate={sample_rate} interval={low}-{high} band={index}");
                            assert!((band.dynamic_eq.target_magnitude_response_db(frequency)
                                - bell.target_magnitude_response_db(frequency)).abs() < 1e-8,
                                "manual bell center/Q changed: rate={sample_rate} interval={low}-{high} band={index}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_manual_out_of_range_bands_are_finite_and_never_warmed_invalid() {
        for sample_rate in [8_000.0, 16_000.0, 22_050.0, 32_000.0] {
            for low in [2_000.0, 4_000.0, 12_000.0] {
                let mut deesser = DeEsser::new(sample_rate);
                deesser.set_enabled(true);
                deesser.set_auto_enabled(false);
                deesser.set_low_cut_hz(low);
                deesser.set_high_cut_hz(16_000.0);
                let empty = low >= sample_rate * 0.5 - EQ_NYQUIST_MARGIN_HZ;
                for n in 0..(sample_rate as usize / 4) {
                    let sample = (2.0 * std::f64::consts::PI * 0.36 * n as f64).sin() as f32 * 0.25;
                    let output = deesser.process_sample(sample);
                    assert!(
                        output.is_finite(),
                        "invalid manual filter at {sample_rate} Hz"
                    );
                    assert!(deesser.current_gain_reduction_db() <= 6.001);
                    if empty {
                        assert_eq!(output, sample);
                    }
                }
                for band in &deesser.bands {
                    assert!(band.env.is_finite() && band.auto_env.is_finite());
                    if let Some((low, high)) = band.manual_bounds {
                        assert!(low > 0.0 && low < high);
                        assert!(high <= sample_rate * 0.5 - EQ_NYQUIST_MARGIN_HZ);
                    } else {
                        assert_eq!(band.env, 0.0, "invalid manual detector was warmed");
                        assert_eq!(band.detector_hp.target_magnitude_response_db(500.0), 0.0);
                        assert_eq!(band.detector_lp.target_magnitude_response_db(500.0), 0.0);
                    }
                    if !empty {
                        assert!(
                            band.low_hz > 0.0
                                && band.high_hz <= sample_rate * 0.5 - EQ_NYQUIST_MARGIN_HZ
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_auto_bandpass_updates_and_both_detector_banks_do_not_allocate() {
        let mut deesser = DeEsser::new(48_000.0);
        deesser.set_enabled(true);
        let mut audio = [0.1_f32; 480];
        crate::test_alloc::assert_no_allocations("band-pass detector updates and render", || {
            for index in 0..8 {
                deesser.set_low_cut_hz(4_000.0 + index as f64 * 100.0);
                deesser.set_high_cut_hz(11_000.0 - index as f64 * 100.0);
                deesser.set_auto_enabled(index % 2 == 0);
                deesser.process_block_inplace(&mut audio);
            }
        });
        assert!(audio.iter().all(|sample| sample.is_finite()));
        assert!(deesser
            .bands
            .iter()
            .all(|band| band.auto_env.is_finite() && band.env.is_finite()));
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

    #[test]
    fn test_auto_baseline_coefficients_are_cached_and_sample_rate_aware() {
        let deesser_44 = DeEsser::new(44_100.0);
        let deesser_96 = DeEsser::new(96_000.0);

        let one_second_44 = deesser_44
            .auto_baseline_rise_coeff
            .powf(deesser_44.sample_rate);
        let one_second_96 = deesser_96
            .auto_baseline_rise_coeff
            .powf(deesser_96.sample_rate);
        let expected = (-1000.0 / AUTO_BASELINE_RISE_MS).exp();

        assert!((one_second_44 - one_second_96).abs() < 1e-6);
        assert!((one_second_44 - expected).abs() < 1e-12);
        assert!(
            (deesser_44.auto_baseline_fall_coeff.powf(44_100.0)
                - (-1000.0 / AUTO_BASELINE_FALL_MS).exp())
            .abs()
                < 1e-12
        );
        assert!(
            (deesser_44.auto_baseline_inactive_decay_coeff.powf(44_100.0)
                - (-1000.0 / AUTO_BASELINE_INACTIVE_DECAY_MS).exp())
            .abs()
                < 1e-12
        );
    }

    #[test]
    fn test_auto_baseline_preserves_transient_response_and_reset() {
        let configure = || {
            let mut deesser = DeEsser::new(48_000.0);
            deesser.set_enabled(true);
            deesser.set_auto_amount(0.5);
            deesser
        };
        let input = |n: usize| {
            let phase = 2.0 * std::f64::consts::PI * n as f64 / 48_000.0;
            (0.04 * (300.0 * phase).sin() + 0.12 * (6800.0 * phase).sin()) as f32
        };
        let mut used = configure();
        let mut peak = 0.0_f32;
        for n in 0..9_600 {
            used.process_sample(input(n));
            peak = peak.max(used.current_gain_reduction_db());
        }
        assert!(
            peak > 0.01,
            "the warmed baseline erased the transient response"
        );
        assert!(used.auto_background_db > 0.0);
        used.reset();
        assert_eq!(used.auto_background_db, 0.0);
        let mut fresh = configure();
        for n in 0..48_000 {
            assert_eq!(
                used.process_sample(input(n)),
                fresh.process_sample(input(n))
            );
            assert_eq!(
                used.band_gain_reductions_db(),
                fresh.band_gain_reductions_db()
            );
        }
    }

    #[test]
    fn test_baseline_auto_amount_and_caps_remain_finite_and_bounded() {
        for amount in [0.0, 0.5, 1.0] {
            for cap in [0.0, 6.0, 12.0] {
                let mut deesser = DeEsser::new(48_000.0);
                deesser.set_enabled(true);
                deesser.set_auto_amount(amount);
                deesser.set_max_reduction_db(cap);
                for n in 0..48_000 {
                    let phase = 2.0 * std::f64::consts::PI * n as f64 / 48_000.0;
                    let input =
                        (0.04 * (300.0 * phase).sin() + 0.12 * (6800.0 * phase).sin()) as f32;
                    assert!(deesser.process_sample(input).is_finite());
                    assert!((0.0..=cap as f32).contains(&deesser.current_gain_reduction_db()));
                }
            }
        }
    }
}
