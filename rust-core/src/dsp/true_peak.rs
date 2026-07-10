//! Band-limited 4x true-peak detector and final safety limiter.
//!
//! The detector and limiter use a 127-tap Kaiser-windowed polyphase FIR. The
//! limiter delay exceeds the FIR group delay so gain reduction is known before
//! the corresponding source sample is emitted.

use crate::dsp::util;

const TRUE_PEAK_PHASES: usize = 4;
const TRUE_PEAK_TAPS_PER_PHASE: usize = 32;
const TRUE_PEAK_LIMITER_LOOKAHEAD_SAMPLES: usize = 20;

// 127-tap, 4x low-pass interpolator generated with scipy.signal.firwin(127,
// 0.25, window=("kaiser", 10.0)) and scaled by four. Coefficients are stored
// by polyphase branch so one input sample produces four band-limited points.
#[allow(clippy::excessive_precision)]
const TRUE_PEAK_FIR: [[f32; TRUE_PEAK_TAPS_PER_PHASE]; TRUE_PEAK_PHASES] = [
    [
        -5.075344514712e-06,
        4.020372649738e-05,
        -1.418964007042e-04,
        3.747307316742e-04,
        -8.371777494884e-04,
        1.668784190125e-03,
        -3.056982594553e-03,
        5.245357011035e-03,
        -8.548572061367e-03,
        1.338626682575e-02,
        -2.036503803793e-02,
        3.048451192740e-02,
        -4.570095330050e-02,
        7.075323958348e-02,
        -1.212808091387e-01,
        2.968932915018e-01,
        8.992408034658e-01,
        -1.747555596851e-01,
        9.076325896271e-02,
        -5.647326848442e-02,
        3.726378372716e-02,
        -2.494347922288e-02,
        1.655863614088e-02,
        -1.074359020134e-02,
        6.734459572706e-03,
        -4.033406903553e-03,
        2.279524360152e-03,
        -1.196325743299e-03,
        5.695732938322e-04,
        -2.366483298844e-04,
        7.939906635813e-05,
        -1.724035750980e-05,
    ],
    [
        -1.427009716384e-05,
        8.113594154575e-05,
        -2.610649871647e-04,
        6.563642916398e-04,
        -1.419899812495e-03,
        2.765037134336e-03,
        -4.975411710453e-03,
        8.418016422932e-03,
        -1.356880279857e-02,
        2.107227382417e-02,
        -3.188679543592e-02,
        4.765542618564e-02,
        -7.175397440485e-02,
        1.129012937293e-01,
        -2.032521146497e-01,
        6.335830989095e-01,
        6.335830989095e-01,
        -2.032521146497e-01,
        1.129012937293e-01,
        -7.175397440485e-02,
        4.765542618564e-02,
        -3.188679543592e-02,
        2.107227382417e-02,
        -1.356880279857e-02,
        8.418016422932e-03,
        -4.975411710453e-03,
        2.765037134336e-03,
        -1.419899812495e-03,
        6.563642916398e-04,
        -2.610649871647e-04,
        8.113594154575e-05,
        -1.427009716384e-05,
    ],
    [
        -1.724035750980e-05,
        7.939906635813e-05,
        -2.366483298844e-04,
        5.695732938322e-04,
        -1.196325743299e-03,
        2.279524360152e-03,
        -4.033406903553e-03,
        6.734459572706e-03,
        -1.074359020134e-02,
        1.655863614088e-02,
        -2.494347922288e-02,
        3.726378372716e-02,
        -5.647326848442e-02,
        9.076325896271e-02,
        -1.747555596851e-01,
        8.992408034658e-01,
        2.968932915018e-01,
        -1.212808091387e-01,
        7.075323958348e-02,
        -4.570095330050e-02,
        3.048451192740e-02,
        -2.036503803793e-02,
        1.338626682575e-02,
        -8.548572061367e-03,
        5.245357011035e-03,
        -3.056982594553e-03,
        1.668784190125e-03,
        -8.371777494884e-04,
        3.747307316742e-04,
        -1.418964007042e-04,
        4.020372649738e-05,
        -5.075344514712e-06,
    ],
    [
        2.063175246022e-19,
        -2.599797566132e-19,
        -8.304334830221e-19,
        -1.440619686029e-18,
        9.818323382399e-18,
        -4.581760436243e-18,
        7.181759012474e-18,
        -1.052671622728e-17,
        1.455323679291e-17,
        -1.909424846351e-17,
        2.388216297225e-17,
        -2.857014524452e-17,
        3.276992189968e-17,
        -3.610092443857e-17,
        3.824275546806e-17,
        9.999997738515e-01,
        3.824275546806e-17,
        -3.610092443857e-17,
        3.276992189968e-17,
        -2.857014524452e-17,
        2.388216297225e-17,
        -1.909424846351e-17,
        1.455323679291e-17,
        -1.052671622728e-17,
        7.181759012474e-18,
        -4.581760436243e-18,
        9.818323382399e-18,
        -1.440619686029e-18,
        -8.304334830221e-19,
        -2.599797566132e-19,
        2.063175246022e-19,
        0.0,
    ],
];

#[derive(Debug, Clone)]
struct Bandlimited4xPeak {
    history: [f32; TRUE_PEAK_TAPS_PER_PHASE],
}

impl Bandlimited4xPeak {
    fn new() -> Self {
        Self {
            history: [0.0; TRUE_PEAK_TAPS_PER_PHASE],
        }
    }

    fn reset(&mut self) {
        self.history = [0.0; TRUE_PEAK_TAPS_PER_PHASE];
    }

    #[inline]
    fn observe(&mut self, sample: f32) -> f32 {
        self.history.copy_within(..TRUE_PEAK_TAPS_PER_PHASE - 1, 1);
        self.history[0] = sample;

        let mut peak = sample.abs();
        for phase in TRUE_PEAK_FIR.iter() {
            let mut interpolated = 0.0_f32;
            for (coefficient, history) in phase.iter().zip(self.history.iter()) {
                interpolated = coefficient.mul_add(*history, interpolated);
            }
            peak = peak.max(interpolated.abs());
        }
        peak
    }
}

#[derive(Debug, Clone)]
pub struct TruePeakDetector {
    oversampler: Bandlimited4xPeak,
    last_peak: f32,
}

impl TruePeakDetector {
    pub fn new() -> Self {
        Self {
            oversampler: Bandlimited4xPeak::new(),
            last_peak: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.oversampler.reset();
        self.last_peak = 0.0;
    }

    pub fn process_block(&mut self, samples: &[f32]) -> f32 {
        let mut peak = 0.0_f32;

        for sample in samples.iter().copied() {
            let sample = if sample.is_finite() { sample } else { 0.0 };
            peak = peak.max(self.oversampler.observe(sample));
        }

        self.last_peak = peak;
        peak
    }

    pub fn last_peak(&self) -> f32 {
        self.last_peak
    }
}

impl Default for TruePeakDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TruePeakLimiterBlockStats {
    pub limited_events: u64,
    pub input_true_peak: f32,
    pub output_true_peak: f32,
    pub max_gain_reduction_db: f32,
}

impl Default for TruePeakLimiterBlockStats {
    fn default() -> Self {
        Self {
            limited_events: 0,
            input_true_peak: 0.0,
            output_true_peak: 0.0,
            max_gain_reduction_db: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TruePeakLimiter {
    ceiling_linear: f32,
    release_coeff: f32,
    gain_reduction: f32,
    delay: [f32; TRUE_PEAK_LIMITER_LOOKAHEAD_SAMPLES],
    write_idx: usize,
    input_oversampler: Bandlimited4xPeak,
    output_oversampler: Bandlimited4xPeak,
    last_input_true_peak: f32,
    last_output_true_peak: f32,
    peak_gain_reduction_db: f32,
    sample_rate: f32,
}

impl TruePeakLimiter {
    pub fn new(sample_rate: f32, ceiling_db: f32, release_ms: f32) -> Self {
        let mut limiter = Self {
            ceiling_linear: util::db_to_linear(ceiling_db as f64) as f32,
            release_coeff: util::time_constant_to_coeff(release_ms as f64, sample_rate as f64)
                as f32,
            gain_reduction: 1.0,
            delay: [0.0; TRUE_PEAK_LIMITER_LOOKAHEAD_SAMPLES],
            write_idx: 0,
            input_oversampler: Bandlimited4xPeak::new(),
            output_oversampler: Bandlimited4xPeak::new(),
            last_input_true_peak: 0.0,
            last_output_true_peak: 0.0,
            peak_gain_reduction_db: 0.0,
            sample_rate: sample_rate.max(1.0),
        };
        limiter.set_release_ms(release_ms);
        limiter
    }

    pub fn default_settings(sample_rate: f32) -> Self {
        Self::new(sample_rate, -1.5, 80.0)
    }

    pub fn reset(&mut self) {
        self.gain_reduction = 1.0;
        self.delay = [0.0; TRUE_PEAK_LIMITER_LOOKAHEAD_SAMPLES];
        self.write_idx = 0;
        self.input_oversampler.reset();
        self.output_oversampler.reset();
        self.last_input_true_peak = 0.0;
        self.last_output_true_peak = 0.0;
        self.peak_gain_reduction_db = 0.0;
    }

    pub fn lookahead_samples(&self) -> usize {
        TRUE_PEAK_LIMITER_LOOKAHEAD_SAMPLES
    }

    pub fn set_ceiling_linear(&mut self, ceiling_linear: f32) {
        self.ceiling_linear = ceiling_linear.clamp(0.000_001, 1.0);
    }

    pub fn set_release_ms(&mut self, release_ms: f32) {
        self.release_coeff = util::time_constant_to_coeff(
            release_ms.clamp(5.0, 500.0) as f64,
            self.sample_rate as f64,
        ) as f32;
    }

    pub fn current_gain_reduction_db(&self) -> f32 {
        if self.gain_reduction >= 1.0 {
            0.0
        } else {
            -20.0 * self.gain_reduction.max(1e-10).log10()
        }
    }

    pub fn peak_gain_reduction_and_reset(&mut self) -> f32 {
        let peak = self.peak_gain_reduction_db;
        self.peak_gain_reduction_db = 0.0;
        peak
    }

    pub fn last_input_true_peak(&self) -> f32 {
        self.last_input_true_peak
    }

    pub fn last_output_true_peak(&self) -> f32 {
        self.last_output_true_peak
    }

    pub fn process_block_inplace(&mut self, samples: &mut [f32]) -> TruePeakLimiterBlockStats {
        let mut stats = TruePeakLimiterBlockStats::default();
        let mut limited = false;

        for sample in samples.iter_mut() {
            let input = if sample.is_finite() { *sample } else { 0.0 };
            let delayed = self.delay[self.write_idx];
            self.delay[self.write_idx] = input;
            self.write_idx = (self.write_idx + 1) % self.delay.len();

            let input_true_peak = self.observe_input_true_peak(input);
            stats.input_true_peak = stats.input_true_peak.max(input_true_peak);

            let target_gain = if input_true_peak > self.ceiling_linear {
                ((self.ceiling_linear * 0.999) / input_true_peak).clamp(0.0, 1.0)
            } else {
                1.0
            };
            if target_gain < self.gain_reduction {
                self.gain_reduction = target_gain;
                limited = true;
            } else {
                self.gain_reduction = self.release_coeff * self.gain_reduction
                    + (1.0 - self.release_coeff) * target_gain;
            }

            let reduction_db = self.current_gain_reduction_db();
            self.peak_gain_reduction_db = self.peak_gain_reduction_db.max(reduction_db);
            stats.max_gain_reduction_db = stats.max_gain_reduction_db.max(reduction_db);

            let output =
                (delayed * self.gain_reduction).clamp(-self.ceiling_linear, self.ceiling_linear);
            let output = if output.is_finite() { output } else { 0.0 };
            stats.output_true_peak = stats
                .output_true_peak
                .max(self.observe_output_true_peak(output));
            *sample = output;
        }

        stats.limited_events = u64::from(limited);
        stats
    }

    #[inline]
    fn observe_input_true_peak(&mut self, sample: f32) -> f32 {
        let peak = self.input_oversampler.observe(sample);
        self.last_input_true_peak = peak;
        peak
    }

    #[inline]
    fn observe_output_true_peak(&mut self, sample: f32) -> f32 {
        let peak = self.output_oversampler.observe(sample);
        self.last_output_true_peak = peak;
        peak
    }
}

impl Default for TruePeakLimiter {
    fn default() -> Self {
        Self::default_settings(48_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_signal_matches_sample_peak() {
        let mut detector = TruePeakDetector::new();
        let peak = detector.process_block(&[0.5; 16]);

        assert!((peak - 0.5).abs() < 1e-6);
        assert!((detector.last_peak() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn bandlimited_oversampling_detects_intersample_overshoot() {
        let mut detector = TruePeakDetector::new();
        let mut signal = [0.0_f32; 64];
        signal[1] = 1.0;
        signal[2] = 1.0;
        let peak = detector.process_block(&signal);

        assert!(peak > 1.0);
    }

    #[test]
    fn reset_clears_history_and_peak() {
        let mut detector = TruePeakDetector::new();
        let mut signal = [0.0_f32; 64];
        signal[1] = 1.0;
        signal[2] = 1.0;
        assert!(detector.process_block(&signal) > 1.0);

        detector.reset();

        assert_eq!(detector.last_peak(), 0.0);
        assert_eq!(detector.process_block(&[0.25]), 0.25);
    }

    #[test]
    fn true_peak_limiter_attenuates_intersample_overs() {
        let ceiling = 1.0_f32;
        let mut limiter = TruePeakLimiter::new(48_000.0, 0.0, 60.0);
        limiter.set_ceiling_linear(ceiling);
        let mut block = [0.0_f32; 96];
        block[1] = 1.0;
        block[2] = 1.0;
        let stats = limiter.process_block_inplace(&mut block);
        let mut detector = TruePeakDetector::new();
        let mut out_peak = detector.process_block(&block);
        out_peak = out_peak.max(detector.process_block(&[0.0; 48]));

        assert_eq!(stats.limited_events, 1);
        assert!(stats.input_true_peak > ceiling);
        assert!(out_peak <= ceiling + 1e-4, "out_peak={out_peak}");
        assert!(stats.max_gain_reduction_db > 0.0);
    }

    #[test]
    fn true_peak_limiter_is_near_transparent_below_ceiling_after_delay() {
        let mut limiter = TruePeakLimiter::new(48_000.0, -1.5, 60.0);
        let input = [0.25_f32; 32];
        let mut block = input;
        let stats = limiter.process_block_inplace(&mut block);
        let delay = limiter.lookahead_samples();

        assert_eq!(stats.limited_events, 0);
        for sample in block.iter().skip(delay) {
            assert!((*sample - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn true_peak_limiter_keeps_extreme_input_finite_and_bounded() {
        let mut limiter = TruePeakLimiter::new(48_000.0, -1.5, 20.0);
        let ceiling = util::db_to_linear(-1.5) as f32;
        let mut block = [f32::NAN, 8.0, -9.0, f32::INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0];
        let stats = limiter.process_block_inplace(&mut block);

        assert!(stats.limited_events > 0);
        assert!(block.iter().all(|sample| sample.is_finite()));
        assert!(block.iter().all(|sample| sample.abs() <= ceiling + 1e-6));
    }

    fn reference_polyphase_coefficients() -> Vec<Vec<f64>> {
        const TAPS: usize = 511;
        let center = (TAPS - 1) as f64 / 2.0;
        let cutoff = 1.0 / (2.0 * TRUE_PEAK_PHASES as f64);
        let mut impulse = Vec::with_capacity(TAPS);
        for index in 0..TAPS {
            let offset = index as f64 - center;
            let sinc = if offset.abs() < f64::EPSILON {
                2.0 * cutoff
            } else {
                (2.0 * std::f64::consts::PI * cutoff * offset).sin()
                    / (std::f64::consts::PI * offset)
            };
            let phase = 2.0 * std::f64::consts::PI * index as f64 / (TAPS - 1) as f64;
            let blackman = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
            impulse.push(sinc * blackman);
        }
        let scale = TRUE_PEAK_PHASES as f64 / impulse.iter().sum::<f64>();
        for coefficient in impulse.iter_mut() {
            *coefficient *= scale;
        }

        (0..TRUE_PEAK_PHASES)
            .map(|phase| {
                (0..128)
                    .map(|tap| {
                        impulse
                            .get(phase + TRUE_PEAK_PHASES * tap)
                            .copied()
                            .unwrap_or(0.0)
                    })
                    .collect()
            })
            .collect()
    }

    fn reference_observe(history: &mut [f64], phases: &[Vec<f64>], sample: f32) -> f32 {
        history.copy_within(..history.len() - 1, 1);
        history[0] = sample as f64;
        phases
            .iter()
            .map(|phase| {
                phase
                    .iter()
                    .zip(history.iter())
                    .map(|(coefficient, delayed)| coefficient * delayed)
                    .sum::<f64>()
                    .abs() as f32
            })
            .fold(sample.abs(), f32::max)
    }

    #[test]
    fn bandlimited_estimator_matches_long_reference_within_point_zero_eight_db() {
        let reference_phases = reference_polyphase_coefficients();
        for frequency_hz in [
            6000.0, 8000.0, 12_000.0, 16_000.0, 18_000.0, 20_000.0, 22_000.0,
        ] {
            for phase_index in 0..8 {
                let initial_phase = phase_index as f64 * std::f64::consts::PI / 4.0;
                let mut estimator = Bandlimited4xPeak::new();
                let mut reference_history = [0.0_f64; 128];
                let mut estimated_peak = 0.0_f32;
                let mut reference_peak = 0.0_f32;
                for index in 0..1024 {
                    let phase = 2.0 * std::f64::consts::PI * frequency_hz * index as f64 / 48_000.0
                        + initial_phase;
                    let sample = (0.9 * phase.sin()) as f32;
                    let estimated = estimator.observe(sample);
                    let reference =
                        reference_observe(&mut reference_history, &reference_phases, sample);
                    if index >= 192 {
                        estimated_peak = estimated_peak.max(estimated);
                        reference_peak = reference_peak.max(reference);
                    }
                }
                let error_db =
                    20.0 * (estimated_peak.max(1e-12) / reference_peak.max(1e-12)).log10();
                assert!(
                    error_db.abs() <= 0.08,
                    "frequency={frequency_hz} phase={initial_phase} error_db={error_db} estimated={estimated_peak} reference={reference_peak}"
                );
            }
        }
    }
}
