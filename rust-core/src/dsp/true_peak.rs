//! Paired band-limited true-peak detection and final safety limiting.
//!
//! The limiter observes a long 16x/4095-tap FIR and a shorter 16x/2047-tap
//! FIR. The shorter observation is delayed by 64 samples before it enters the
//! causal hold window. A 320-sample delay gives the gain controller time to
//! protect the corresponding source sample before it is emitted.

use crate::dsp::util;
use std::sync::OnceLock;

const TRUE_PEAK_PHASES: usize = 16;
const TRUE_PEAK_LONG_TAPS: usize = 4095;
const TRUE_PEAK_LONG_TAPS_PER_PHASE: usize = 256;
const TRUE_PEAK_SHORT_TAPS: usize = 2047;
const TRUE_PEAK_SHORT_TAPS_PER_PHASE: usize = 128;
const TRUE_PEAK_SHORT_DELAY_SAMPLES: usize = 64;
const TRUE_PEAK_LIMITER_HOLD_SAMPLES: usize = 384;
const TRUE_PEAK_GAIN_AVERAGE_SAMPLES: usize = 128;
const TRUE_PEAK_LIMITER_LOOKAHEAD_SAMPLES: usize = 320;
const TRUE_PEAK_TARGET_MARGIN: f32 = 0.999;

#[derive(Debug)]
struct TruePeakFirBank {
    long: [[f32; TRUE_PEAK_LONG_TAPS_PER_PHASE]; TRUE_PEAK_PHASES],
    short: [[f32; TRUE_PEAK_SHORT_TAPS_PER_PHASE]; TRUE_PEAK_PHASES],
}

impl TruePeakFirBank {
    fn new() -> Self {
        Self {
            long: generate_fir::<TRUE_PEAK_LONG_TAPS_PER_PHASE>(TRUE_PEAK_LONG_TAPS),
            short: generate_fir::<TRUE_PEAK_SHORT_TAPS_PER_PHASE>(TRUE_PEAK_SHORT_TAPS),
        }
    }
}

static TRUE_PEAK_FIR_BANK: OnceLock<TruePeakFirBank> = OnceLock::new();

fn true_peak_fir_bank() -> &'static TruePeakFirBank {
    TRUE_PEAK_FIR_BANK.get_or_init(TruePeakFirBank::new)
}

fn generate_fir<const TAPS_PER_PHASE: usize>(
    taps: usize,
) -> [[f32; TAPS_PER_PHASE]; TRUE_PEAK_PHASES] {
    assert_eq!(taps % TRUE_PEAK_PHASES, TRUE_PEAK_PHASES - 1);
    let center = (taps - 1) as f64 / 2.0;
    let cutoff = 1.0 / (2.0 * TRUE_PEAK_PHASES as f64);
    let mut impulse = Vec::with_capacity(taps);
    for index in 0..taps {
        let offset = index as f64 - center;
        let sinc = if offset.abs() < f64::EPSILON {
            2.0 * cutoff
        } else {
            (2.0 * std::f64::consts::PI * cutoff * offset).sin() / (std::f64::consts::PI * offset)
        };
        let phase = 2.0 * std::f64::consts::PI * index as f64 / (taps - 1) as f64;
        let blackman = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
        impulse.push(sinc * blackman);
    }
    let scale = TRUE_PEAK_PHASES as f64 / impulse.iter().sum::<f64>();
    let mut coefficients = [[0.0_f32; TAPS_PER_PHASE]; TRUE_PEAK_PHASES];
    for (phase, phase_coefficients) in coefficients.iter_mut().enumerate() {
        for (tap, phase_coefficient) in phase_coefficients.iter_mut().enumerate() {
            let index = phase + TRUE_PEAK_PHASES * tap;
            if let Some(coefficient) = impulse.get(index) {
                *phase_coefficient = (*coefficient * scale) as f32;
            }
        }
    }
    coefficients
}

#[inline]
fn dot_interleaved8(coefficients: &[f32], history: &[f32]) -> f32 {
    debug_assert_eq!(coefficients.len(), history.len());
    let mut sums = [0.0_f32; 8];
    for (coefficient, sample) in coefficients.chunks_exact(8).zip(history.chunks_exact(8)) {
        sums[0] += coefficient[0] * sample[0];
        sums[1] += coefficient[1] * sample[1];
        sums[2] += coefficient[2] * sample[2];
        sums[3] += coefficient[3] * sample[3];
        sums[4] += coefficient[4] * sample[4];
        sums[5] += coefficient[5] * sample[5];
        sums[6] += coefficient[6] * sample[6];
        sums[7] += coefficient[7] * sample[7];
    }
    for (&coefficient, &sample) in coefficients
        .chunks_exact(8)
        .remainder()
        .iter()
        .zip(history.chunks_exact(8).remainder().iter())
    {
        sums[0] += coefficient * sample;
    }
    sums.iter().copied().sum()
}

#[derive(Debug, Clone)]
struct BandlimitedPeak<const TAPS_PER_PHASE: usize> {
    coefficients: &'static [[f32; TAPS_PER_PHASE]; TRUE_PEAK_PHASES],
    history: [f32; TAPS_PER_PHASE],
}

impl<const TAPS_PER_PHASE: usize> BandlimitedPeak<TAPS_PER_PHASE> {
    fn new(coefficients: &'static [[f32; TAPS_PER_PHASE]; TRUE_PEAK_PHASES]) -> Self {
        Self {
            coefficients,
            history: [0.0; TAPS_PER_PHASE],
        }
    }

    fn reset(&mut self) {
        self.history = [0.0; TAPS_PER_PHASE];
    }

    #[inline]
    fn observe(&mut self, sample: f32) -> f32 {
        self.observe_inner(sample, true)
    }

    #[inline]
    fn observe_fir_only(&mut self, sample: f32) -> f32 {
        self.observe_inner(sample, false)
    }

    #[inline]
    fn observe_inner(&mut self, sample: f32, include_sample: bool) -> f32 {
        self.history.copy_within(..TAPS_PER_PHASE - 1, 1);
        self.history[0] = sample;

        let mut peak = if include_sample { sample.abs() } else { 0.0 };
        for phase in self.coefficients.iter() {
            let interpolated = dot_interleaved8(phase, &self.history);
            peak = peak.max(interpolated.abs());
        }
        peak
    }
}

#[derive(Debug, Clone)]
pub struct TruePeakDetector {
    oversampler: BandlimitedPeak<TRUE_PEAK_LONG_TAPS_PER_PHASE>,
    last_peak: f32,
}

impl TruePeakDetector {
    pub fn new() -> Self {
        let bank = true_peak_fir_bank();
        Self {
            oversampler: BandlimitedPeak::new(&bank.long),
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
    /// One when the controller attacks during this block; zero otherwise.
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
struct MaxHoldDeque {
    indices: [u64; TRUE_PEAK_LIMITER_HOLD_SAMPLES],
    values: [f32; TRUE_PEAK_LIMITER_HOLD_SAMPLES],
    head: usize,
    len: usize,
}

impl MaxHoldDeque {
    fn new() -> Self {
        Self {
            indices: [0; TRUE_PEAK_LIMITER_HOLD_SAMPLES],
            values: [0.0; TRUE_PEAK_LIMITER_HOLD_SAMPLES],
            head: 0,
            len: 0,
        }
    }

    fn reset(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    fn expire_before(&mut self, current: u64) {
        while self.len != 0
            && current.wrapping_sub(self.indices[self.head])
                >= TRUE_PEAK_LIMITER_HOLD_SAMPLES as u64
        {
            self.head = (self.head + 1) % TRUE_PEAK_LIMITER_HOLD_SAMPLES;
            self.len -= 1;
        }
    }

    fn push(&mut self, index: u64, value: f32) {
        self.expire_before(index);
        while self.len != 0 {
            let back = (self.head + self.len - 1) % TRUE_PEAK_LIMITER_HOLD_SAMPLES;
            if self.values[back] >= value {
                break;
            }
            self.len -= 1;
        }
        debug_assert!(self.len < TRUE_PEAK_LIMITER_HOLD_SAMPLES);
        let back = (self.head + self.len) % TRUE_PEAK_LIMITER_HOLD_SAMPLES;
        self.indices[back] = index;
        self.values[back] = value;
        self.len += 1;
    }

    fn front(&self) -> f32 {
        debug_assert!(self.len != 0);
        self.values[self.head]
    }
}

#[derive(Debug, Clone)]
pub struct TruePeakLimiter {
    ceiling_linear: f32,
    release_coeff: f32,
    instant_gain: f32,
    gain_reduction: f32,
    delay: [f32; TRUE_PEAK_LIMITER_LOOKAHEAD_SAMPLES],
    write_idx: usize,
    input_oversampler: BandlimitedPeak<TRUE_PEAK_LONG_TAPS_PER_PHASE>,
    short_oversampler: BandlimitedPeak<TRUE_PEAK_SHORT_TAPS_PER_PHASE>,
    short_delay: [f32; TRUE_PEAK_SHORT_DELAY_SAMPLES],
    short_delay_idx: usize,
    held_peaks: MaxHoldDeque,
    sample_index: u64,
    gain_history: [f32; TRUE_PEAK_GAIN_AVERAGE_SAMPLES],
    gain_history_idx: usize,
    gain_history_sum: f64,
    output_oversampler: BandlimitedPeak<TRUE_PEAK_LONG_TAPS_PER_PHASE>,
    last_input_true_peak: f32,
    last_output_true_peak: f32,
    peak_gain_reduction_db: f32,
    sample_rate: f32,
}

impl TruePeakLimiter {
    pub fn new(sample_rate: f32, ceiling_db: f32, release_ms: f32) -> Self {
        let bank = true_peak_fir_bank();
        let mut limiter = Self {
            ceiling_linear: util::db_to_linear(ceiling_db as f64) as f32,
            release_coeff: 1.0,
            instant_gain: 1.0,
            gain_reduction: 1.0,
            delay: [0.0; TRUE_PEAK_LIMITER_LOOKAHEAD_SAMPLES],
            write_idx: 0,
            input_oversampler: BandlimitedPeak::new(&bank.long),
            short_oversampler: BandlimitedPeak::new(&bank.short),
            short_delay: [0.0; TRUE_PEAK_SHORT_DELAY_SAMPLES],
            short_delay_idx: 0,
            held_peaks: MaxHoldDeque::new(),
            sample_index: 0,
            gain_history: [1.0; TRUE_PEAK_GAIN_AVERAGE_SAMPLES],
            gain_history_idx: 0,
            gain_history_sum: TRUE_PEAK_GAIN_AVERAGE_SAMPLES as f64,
            output_oversampler: BandlimitedPeak::new(&bank.long),
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
        self.instant_gain = 1.0;
        self.gain_reduction = 1.0;
        self.delay = [0.0; TRUE_PEAK_LIMITER_LOOKAHEAD_SAMPLES];
        self.write_idx = 0;
        self.input_oversampler.reset();
        self.short_oversampler.reset();
        self.short_delay = [0.0; TRUE_PEAK_SHORT_DELAY_SAMPLES];
        self.short_delay_idx = 0;
        self.held_peaks.reset();
        self.sample_index = 0;
        self.gain_history = [1.0; TRUE_PEAK_GAIN_AVERAGE_SAMPLES];
        self.gain_history_idx = 0;
        self.gain_history_sum = TRUE_PEAK_GAIN_AVERAGE_SAMPLES as f64;
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

    #[cfg(test)]
    pub(crate) fn release_coefficient(&self) -> f32 {
        self.release_coeff
    }

    pub fn current_gain_reduction_db(&self) -> f32 {
        if !self.gain_reduction.is_finite() || self.gain_reduction >= 1.0 {
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

            let long_peak = self.input_oversampler.observe(input);
            let short_peak = self.short_oversampler.observe_fir_only(input);
            let delayed_short_peak = self.short_delay[self.short_delay_idx];
            self.short_delay[self.short_delay_idx] = short_peak;
            self.short_delay_idx = (self.short_delay_idx + 1) % self.short_delay.len();
            let detected_peak = long_peak.max(delayed_short_peak);
            self.last_input_true_peak = detected_peak;
            stats.input_true_peak = stats.input_true_peak.max(detected_peak);

            self.held_peaks.push(self.sample_index, detected_peak);
            self.sample_index = self.sample_index.wrapping_add(1);
            let held_peak = self.held_peaks.front();
            let target_gain = (self.ceiling_linear * TRUE_PEAK_TARGET_MARGIN
                / held_peak.max(1e-20))
            .clamp(0.0, 1.0);
            if target_gain < self.instant_gain {
                limited = true;
                self.instant_gain = target_gain;
            } else {
                self.instant_gain = self.release_coeff * self.instant_gain
                    + (1.0 - self.release_coeff) * target_gain;
            }

            let old_gain = self.gain_history[self.gain_history_idx];
            self.gain_history[self.gain_history_idx] = self.instant_gain;
            self.gain_history_idx = (self.gain_history_idx + 1) % self.gain_history.len();
            self.gain_history_sum += f64::from(self.instant_gain) - f64::from(old_gain);
            self.gain_reduction = (self.gain_history_sum / TRUE_PEAK_GAIN_AVERAGE_SAMPLES as f64)
                .clamp(0.0, 1.0) as f32;

            let reduction_db = self.current_gain_reduction_db();
            self.peak_gain_reduction_db = self.peak_gain_reduction_db.max(reduction_db);
            stats.max_gain_reduction_db = stats.max_gain_reduction_db.max(reduction_db);

            let output =
                (delayed * self.gain_reduction).clamp(-self.ceiling_linear, self.ceiling_linear);
            let output = if output.is_finite() { output } else { 0.0 };
            let output_peak = self.output_oversampler.observe(output);
            self.last_output_true_peak = output_peak;
            stats.output_true_peak = stats.output_true_peak.max(output_peak);
            *sample = output;
        }

        stats.limited_events = u64::from(limited);
        stats
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
        let signal: Vec<f32> = (0..256)
            .map(|index| (2.0 * std::f32::consts::PI * 0.49 * index as f32 + 2.2).sin())
            .collect();
        let peak = detector.process_block(&signal);

        assert!(peak > 1.0);
    }

    #[test]
    fn reset_clears_history_and_peak() {
        let mut detector = TruePeakDetector::new();
        let signal: Vec<f32> = (0..256)
            .map(|index| (2.0 * std::f32::consts::PI * 0.49 * index as f32 + 2.2).sin())
            .collect();
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
        let mut block = [0.0_f32; 640];
        block[1] = 1.0;
        block[2] = 1.0;
        let stats = limiter.process_block_inplace(&mut block);
        let mut detector = TruePeakDetector::new();
        let out_peak = detector.process_block(&block);

        assert_eq!(stats.limited_events, 1);
        assert!(stats.input_true_peak > ceiling);
        assert!(out_peak <= ceiling + 1e-4, "out_peak={out_peak}");
        assert!(stats.max_gain_reduction_db > 0.0);
    }

    #[test]
    fn true_peak_limiter_is_near_transparent_below_ceiling_after_delay() {
        let mut limiter = TruePeakLimiter::new(48_000.0, -1.5, 60.0);
        let input: Vec<f32> = (0..1024)
            .map(|index| 0.25 * (index as f32 * 0.037).sin())
            .collect();
        let mut block = input.clone();
        let stats = limiter.process_block_inplace(&mut block);
        let delay = limiter.lookahead_samples();

        assert_eq!(stats.limited_events, 0);
        for (&actual, &expected) in block.iter().skip(delay).zip(input.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn true_peak_limiter_keeps_extreme_input_finite_and_bounded() {
        let mut limiter = TruePeakLimiter::new(48_000.0, -1.5, 20.0);
        let ceiling = util::db_to_linear(-1.5) as f32;
        let mut block = vec![0.0_f32; 1024];
        block[..9].copy_from_slice(&[f32::NAN, 8.0, -9.0, f32::INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let stats = limiter.process_block_inplace(&mut block);

        assert!(stats.limited_events > 0);
        assert!(block.iter().all(|sample| sample.is_finite()));
        assert!(block.iter().all(|sample| sample.abs() <= ceiling + 1e-6));
        assert!(block.iter().any(|sample| sample.abs() > 0.1));
    }

    #[test]
    fn peak_hold_matches_sliding_window_across_index_rollover() {
        let mut deque = MaxHoldDeque::new();
        let mut window = std::collections::VecDeque::new();
        let mut seed = 0x1234_5678_u32;
        let start = u64::MAX - 2 * TRUE_PEAK_LIMITER_HOLD_SAMPLES as u64;
        for offset in 0..4 * TRUE_PEAK_LIMITER_HOLD_SAMPLES {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let value = if offset < TRUE_PEAK_LIMITER_HOLD_SAMPLES {
                (TRUE_PEAK_LIMITER_HOLD_SAMPLES - offset) as f32
            } else {
                (seed % 32) as f32
            };
            window.push_back(value);
            if window.len() > TRUE_PEAK_LIMITER_HOLD_SAMPLES {
                window.pop_front();
            }
            deque.push(start.wrapping_add(offset as u64), value);
            assert_eq!(
                deque.front(),
                window.iter().copied().fold(0.0_f32, f32::max)
            );
        }
    }

    #[test]
    fn paired_limiter_uses_causal_lookahead_and_nonvacuous_transparency() {
        let mut limiter = TruePeakLimiter::new(48_000.0, -1.5, 80.0);
        assert_eq!(limiter.lookahead_samples(), 320);

        let input: Vec<f32> = (0..1024)
            .map(|index| 0.25 * (index as f32 * 0.037).sin())
            .collect();
        let mut output = input.clone();
        limiter.process_block_inplace(&mut output);

        for (&actual, &expected) in output.iter().skip(320).zip(input.iter()) {
            assert!((actual - expected).abs() <= 1e-6);
        }
    }

    #[test]
    fn paired_limiter_is_reset_and_chunk_invariant() {
        let input: Vec<f32> = (0..1024)
            .map(|index| {
                let phase = index as f32 * 0.11;
                (0.75 * phase.sin() + 0.1 * (phase * 2.7).cos()).clamp(-1.0, 1.0)
            })
            .collect();
        let mut contiguous_limiter = TruePeakLimiter::new(48_000.0, -1.5, 80.0);
        let mut contiguous = input.clone();
        contiguous_limiter.process_block_inplace(&mut contiguous);

        let mut chunked_limiter = TruePeakLimiter::new(48_000.0, -1.5, 80.0);
        let mut chunked = input.clone();
        for range in [(0, 1), (1, 64), (64, 191), (191, 671), (671, 1024)] {
            chunked_limiter.process_block_inplace(&mut chunked[range.0..range.1]);
        }
        assert_eq!(contiguous, chunked);

        chunked_limiter.reset();
        let mut reset_output = input.clone();
        chunked_limiter.process_block_inplace(&mut reset_output);
        assert_eq!(contiguous, reset_output);
    }

    #[test]
    fn paired_limiter_reduces_near_nyquist_burst_against_independent_reference() {
        let mut limiter = TruePeakLimiter::new(48_000.0, -1.5, 80.0);
        let mut input = vec![0.0_f32; 2048];
        for (offset, sample) in input[479..527].iter_mut().enumerate() {
            *sample = (2.0 * std::f32::consts::PI * 0.49 * offset as f32 + 0.7).sin();
        }
        let mut output = input;
        let mut flushed = vec![0.0_f32; 640];
        limiter.process_block_inplace(&mut output);
        limiter.process_block_inplace(&mut flushed);
        output.extend_from_slice(&flushed);

        let peak_2047 = reference_true_peak(&output, 16, 2047);
        let peak_4095 = reference_true_peak(&output, 16, 4095);
        let peak_db = util::linear_to_db(peak_2047.max(peak_4095) as f64, 1e-10);
        assert!(
            peak_db <= -1.4,
            "independent peak={peak_db} dBTP (2047={peak_2047}, 4095={peak_4095})"
        );
    }

    #[test]
    fn release_setter_supports_bounded_range_and_long_recovery_is_unity() {
        let mut limiter = TruePeakLimiter::new(48_000.0, -1.5, 80.0);
        for release_ms in [5.0, 80.0, 500.0] {
            limiter.set_release_ms(release_ms);
            assert!(limiter.release_coefficient() > 0.0);
            assert!(limiter.release_coefficient() < 1.0);
        }

        let mut input = vec![0.0_f32; 32_000];
        input[80..160].fill(1.0);
        for (index, sample) in input[4096..].iter_mut().enumerate() {
            *sample = 0.25 * (index as f32 * 0.031).sin();
        }
        let expected = input.clone();
        limiter.set_release_ms(80.0);
        limiter.process_block_inplace(&mut input);
        let start = 30_000usize;
        let max_error = input[start..]
            .iter()
            .zip(expected[start - limiter.lookahead_samples()..].iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_error <= 1e-3, "recovery max_error={max_error}");
    }

    fn reference_true_peak(samples: &[f32], factor: usize, taps: usize) -> f32 {
        let coefficients = reference_fir(factor, taps);
        let taps_per_phase = taps.div_ceil(factor);
        let mut history = vec![0.0_f64; taps_per_phase];
        let mut peak = 0.0_f32;
        for sample in samples.iter().copied() {
            history.copy_within(..taps_per_phase - 1, 1);
            history[0] = sample as f64;
            peak = peak.max(sample.abs());
            for phase_coefficients in coefficients.iter().take(factor) {
                let interpolated = phase_coefficients
                    .iter()
                    .zip(history.iter())
                    .map(|(coefficient, delayed)| coefficient * delayed)
                    .sum::<f64>()
                    .abs() as f32;
                peak = peak.max(interpolated);
            }
        }
        peak
    }

    fn reference_fir(factor: usize, taps: usize) -> Vec<Vec<f64>> {
        let center = (taps - 1) as f64 / 2.0;
        let cutoff = 1.0 / (2.0 * factor as f64);
        let mut impulse = Vec::with_capacity(taps);
        for index in 0..taps {
            let offset = index as f64 - center;
            let sinc = if offset.abs() < f64::EPSILON {
                2.0 * cutoff
            } else {
                (2.0 * std::f64::consts::PI * cutoff * offset).sin()
                    / (std::f64::consts::PI * offset)
            };
            let phase = 2.0 * std::f64::consts::PI * index as f64 / (taps - 1) as f64;
            let blackman = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
            impulse.push(sinc * blackman);
        }
        let scale = factor as f64 / impulse.iter().sum::<f64>();
        (0..factor)
            .map(|phase| {
                (0..taps.div_ceil(factor))
                    .map(|tap| impulse.get(phase + factor * tap).copied().unwrap_or(0.0) * scale)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn bandlimited_estimator_matches_independent_reference() {
        for frequency_hz in [8_000.0, 16_000.0, 20_000.0, 22_000.0] {
            for phase_index in 0..4 {
                let initial_phase = phase_index as f64 * std::f64::consts::PI / 2.0;
                let mut estimator = TruePeakDetector::new();
                let mut samples = Vec::with_capacity(1024);
                for index in 0..1024 {
                    let phase = 2.0 * std::f64::consts::PI * frequency_hz * index as f64 / 48_000.0
                        + initial_phase;
                    samples.push((0.9 * phase.sin()) as f32);
                }
                let estimated = estimator.process_block(&samples);
                let reference = reference_true_peak(&samples, 16, 4095);
                let error_db =
                    util::linear_to_db(estimated as f64 / reference.max(1e-12) as f64, 1e-10);
                assert!(
                    error_db.abs() <= 0.08,
                    "frequency={frequency_hz} phase={initial_phase} error_db={error_db} estimated={estimated} reference={reference}"
                );
            }
        }
    }
}
