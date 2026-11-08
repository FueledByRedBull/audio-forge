//! Latency calibration: a repeated Barker-coded probe and the correlation
//! analysis that finds it in a recording of the selected output-to-input route.
//!
//! This replaced a NumPy implementation and keeps its arithmetic order. The
//! probe's Hann window comes from the caller (NumPy's `np.hanning` in the
//! bindings). The correlations use `realfft`, which rounds differently from
//! NumPy's pocketfft, so correlation-derived floats agree to rounding, not
//! bit for bit.

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::RealFftPlanner;

use super::numpy_mean;
use super::numpy_sum;

#[cfg(test)]
mod tests;

const BARKER_13: [f64; 13] = [
    1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
];
const REPETITIONS: i64 = 4;
pub const DEFAULT_ROUTE_KIND: &str = "output_to_input";
pub const COMPENSATION_BASIS: &str = "measured_output_to_input_route";

/// Why the probe or the analysis failed where the removed Python raised.
#[derive(Debug)]
pub enum LatencyError {
    /// Python's `ValueError`, as `int()` of NaN raised.
    Value(String),
    /// Python's `OverflowError`, as `int()` of infinity raised.
    Overflow(String),
    /// NumPy's `IndexError` for an index outside an array.
    Index(String),
    /// The caller's sample rate conversion or window failed with its own error.
    Caller(Box<dyn std::error::Error + Send + Sync>),
}

impl std::fmt::Display for LatencyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Value(message) | Self::Overflow(message) | Self::Index(message) => {
                formatter.write_str(message)
            }
            Self::Caller(error) => error.fmt(formatter),
        }
    }
}

/// `np.hanning(size)` for the probe's coded burst, or the caller's error.
pub type HannWindow<'a> = dyn FnMut(usize) -> Result<Vec<f64>, LatencyError> + Send + 'a;

/// What `analyze_latency` measured, or why it refused.
#[derive(Debug, Clone, PartialEq)]
pub struct LatencyAnalysis {
    pub success: bool,
    /// The measured selected output->input route delay; the name is kept for
    /// persisted profiles.
    pub measured_round_trip_ms: f64,
    pub confidence: f64,
    pub peak_sample_offset: i64,
    pub message: String,
    pub repetition_count: usize,
    pub agreement_ms: f64,
    pub ambiguity_score: f64,
    pub sub_sample_offset: f64,
    pub route_latency_ms: f64,
    pub route_kind: String,
}

impl LatencyAnalysis {
    fn refused(message: &str) -> Self {
        Self {
            success: false,
            measured_round_trip_ms: 0.0,
            confidence: 0.0,
            peak_sample_offset: 0,
            message: message.to_string(),
            repetition_count: 0,
            agreement_ms: 0.0,
            ambiguity_score: 0.0,
            sub_sample_offset: 0.0,
            route_latency_ms: 0.0,
            route_kind: DEFAULT_ROUTE_KIND.to_string(),
        }
    }
}

/// Where the probe may have started and how late the route may be.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchWindow {
    pub min_search_ms: f64,
    pub max_search_ms: f64,
    pub expected_playback_start_ms: Option<f64>,
    pub expected_playback_jitter_ms: Option<f64>,
    pub expected_latency_min_ms: Option<f64>,
    pub expected_latency_max_ms: Option<f64>,
}

/// Python's `max(a, b)`: `a` unless `b` is greater, so a NaN `b` yields `a`.
fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python's `int(x)` for a float: truncation, refusing NaN and infinity.
fn py_int(value: f64) -> Result<i64, LatencyError> {
    if value.is_nan() {
        Err(LatencyError::Value(
            "cannot convert float NaN to integer".to_string(),
        ))
    } else if value.is_infinite() {
        Err(LatencyError::Overflow(
            "cannot convert float infinity to integer".to_string(),
        ))
    } else {
        Ok(value.trunc() as i64)
    }
}

/// Python's `round(x)` for a float: half to even, as an integer.
fn py_round(value: f64) -> Result<i64, LatencyError> {
    py_int(value.round_ties_even())
}

/// `np.clip` for a scalar: NaN stays NaN.
fn clip(value: f64, low: f64, high: f64) -> f64 {
    if value.is_nan() {
        return value;
    }
    let raised = if value > low { value } else { low };
    if raised < high {
        raised
    } else {
        high
    }
}

/// `np.maximum`: NaN wins.
fn np_maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a > b {
        a
    } else {
        b
    }
}

/// `np.argmax` of a nonempty slice: the first NaN, else the first maximum.
fn argmax(values: &[f64]) -> usize {
    if let Some(index) = values.iter().position(|value| value.is_nan()) {
        return index;
    }
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

/// `np.max` of a nonempty slice: NaN if any value is NaN.
fn np_max(values: &[f64]) -> f64 {
    values[argmax(values)]
}

fn sorted(values: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted
}

/// `np.median` of a nonempty slice: NaN if any value is NaN.
fn median(values: &[f64]) -> f64 {
    if values.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }
    let sorted = sorted(values);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[middle]
    } else {
        numpy_mean(&sorted[middle - 1..=middle])
    }
}

/// `np.percentile(values, 75)` of a nonempty slice with linear interpolation.
fn percentile_75(values: &[f64]) -> f64 {
    if values.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }
    let sorted = sorted(values);
    let virtual_index = (sorted.len() - 1) as f64 * 0.75;
    let previous = virtual_index.floor();
    let gamma = virtual_index - previous;
    let below = sorted[previous as usize];
    let above = sorted[(previous as usize + 1).min(sorted.len() - 1)];
    let difference = above - below;
    if gamma >= 0.5 {
        above - difference * (1.0 - gamma)
    } else {
        below + difference * gamma
    }
}

/// NumPy's negative-index wraparound for a single index.
fn at(values: &[f64], index: i64) -> Result<f64, LatencyError> {
    let length = values.len() as i64;
    let wrapped = if index < 0 { index + length } else { index };
    if (0..length).contains(&wrapped) {
        Ok(values[wrapped as usize])
    } else {
        Err(LatencyError::Index(format!(
            "index {index} is out of bounds for axis 0 with size {length}"
        )))
    }
}

/// A single coded burst and its start offsets inside a probe of `total_samples`.
fn coded_probe_components(
    sample_rate: f64,
    total_samples: i64,
    hanning: &mut HannWindow<'_>,
) -> Result<(Vec<f64>, Vec<i64>), LatencyError> {
    let mut chip_samples = py_round(sample_rate * 0.0005)?.max(4);
    let min_spacing = chip_samples.max(py_round(sample_rate * 0.006)?);
    while chip_samples > 4 {
        let burst_len = BARKER_13.len() as i64 * chip_samples;
        if REPETITIONS * burst_len + (REPETITIONS - 1) * min_spacing <= total_samples {
            break;
        }
        chip_samples -= 1;
    }

    let code: Vec<f64> = BARKER_13
        .iter()
        .flat_map(|chip| std::iter::repeat_n(*chip, chip_samples as usize))
        .collect();
    let window = hanning(code.len())?;
    if window.len() != code.len() {
        return Err(LatencyError::Value(format!(
            "the Hann window has {} samples, not {}",
            window.len(),
            code.len()
        )));
    }
    let mut burst: Vec<f64> = code
        .iter()
        .zip(window)
        .map(|(chip, window)| chip * window)
        .collect();
    let mean = numpy_mean(&burst);
    for sample in &mut burst {
        *sample -= mean;
    }
    let peak = np_max(&burst.iter().map(|sample| sample.abs()).collect::<Vec<_>>());
    if peak > 0.0 {
        for sample in &mut burst {
            *sample /= peak;
        }
    }

    let burst_len = burst.len() as i64;
    if total_samples <= burst_len {
        burst.truncate(total_samples.max(0) as usize);
        return Ok((burst, vec![0]));
    }
    let available_gap = (total_samples - REPETITIONS * burst_len).max(0);
    let spacing = min_spacing.max(available_gap / (REPETITIONS - 1));
    let mut offsets = Vec::new();
    let mut cursor = 0;
    for _ in 0..REPETITIONS {
        if cursor + burst_len > total_samples {
            break;
        }
        offsets.push(cursor);
        cursor += burst_len + spacing;
    }
    if offsets.is_empty() {
        offsets.push(0);
    }
    Ok((burst, offsets))
}

/// A deterministic repeated Barker-coded probe, peak-normalized to `amplitude`.
pub fn generate_probe_signal(
    sample_rate: f64,
    duration_ms: f64,
    amplitude: f64,
    hanning: &mut HannWindow<'_>,
) -> Result<Vec<f32>, LatencyError> {
    let total_samples = py_int(sample_rate * (duration_ms / 1000.0))?.max(1);
    let (burst, offsets) = coded_probe_components(sample_rate, total_samples, hanning)?;
    let total = total_samples as usize;
    let mut probe = vec![0.0_f64; total];
    for offset in offsets {
        let offset = offset as usize;
        let end = total.min(offset + burst.len());
        if end > offset {
            for (sample, value) in probe[offset..end].iter_mut().zip(&burst) {
                *sample += value;
            }
        }
    }
    let peak = np_max(&probe.iter().map(|sample| sample.abs()).collect::<Vec<_>>());
    if peak > 0.0 {
        for sample in &mut probe {
            *sample = (*sample / peak) * amplitude;
        }
    }
    Ok(probe.into_iter().map(|sample| sample as f32).collect())
}

struct Correlator {
    planner: RealFftPlanner<f64>,
}

impl Correlator {
    fn new() -> Self {
        Self {
            planner: RealFftPlanner::new(),
        }
    }

    fn forward(&mut self, signal: &[f64], length: usize) -> Vec<Complex<f64>> {
        let plan: Arc<dyn realfft::RealToComplex<f64>> = self.planner.plan_fft_forward(length);
        let mut input = plan.make_input_vec();
        input[..signal.len()].copy_from_slice(signal);
        let mut output = plan.make_output_vec();
        plan.process(&mut input, &mut output)
            .expect("buffers come from the plan");
        output
    }

    /// `np.fft.irfft(spectrum, n=length)`: DC and Nyquist imaginary parts ignored.
    fn inverse(&mut self, mut spectrum: Vec<Complex<f64>>, length: usize) -> Vec<f64> {
        let plan = self.planner.plan_fft_inverse(length);
        spectrum[0].im = 0.0;
        if length.is_multiple_of(2) {
            if let Some(last) = spectrum.last_mut() {
                last.im = 0.0;
            }
        }
        let mut output = plan.make_output_vec();
        plan.process(&mut spectrum, &mut output)
            .expect("buffers come from the plan and DC/Nyquist are real");
        let scale = 1.0 / length as f64;
        for sample in &mut output {
            *sample *= scale;
        }
        output
    }

    /// Linear full correlation of nonempty real audio, lag `1 - reference.len()` first.
    fn full(&mut self, recording: &[f64], reference: &[f64]) -> Vec<f64> {
        let length = recording.len() + reference.len() - 1;
        let fft_length = length.next_power_of_two();
        let reversed: Vec<f64> = reference.iter().rev().copied().collect();
        let recording_spectrum = self.forward(recording, fft_length);
        let reference_spectrum = self.forward(&reversed, fft_length);
        let product = recording_spectrum
            .iter()
            .zip(&reference_spectrum)
            .map(|(a, b)| a * b)
            .collect();
        let mut correlation = self.inverse(product, fft_length);
        correlation.truncate(length);
        correlation
    }

    /// Lags where `reference` lies wholly inside `recording`, with their
    /// energy-normalized correlation magnitudes.
    fn normalized_scores(
        &mut self,
        recording: &[f64],
        reference: &[f64],
        min_lag: i64,
        max_lag: i64,
    ) -> (Vec<i64>, Vec<f64>) {
        let size = reference.len() as i64;
        let first = min_lag.max(0);
        let last = max_lag.min(recording.len() as i64 - size);
        if first > last {
            return (Vec::new(), Vec::new());
        }
        let correlation = self.full(recording, reference);
        let reference_energy =
            numpy_sum(&reference.iter().map(|x| x * x).collect::<Vec<_>>()) + 1e-12;
        let mut prefix = Vec::with_capacity(recording.len() + 1);
        prefix.push(0.0);
        let mut running = 0.0;
        for sample in recording {
            running += sample * sample;
            prefix.push(running);
        }
        let lags: Vec<i64> = (first..=last).collect();
        let scores = lags
            .iter()
            .map(|&lag| {
                let lag = lag as usize;
                let window_energy = prefix[lag + reference.len()] - prefix[lag];
                correlation[lag + reference.len() - 1].abs()
                    / (np_maximum(window_energy, 1e-12) * reference_energy).sqrt()
            })
            .collect();
        (lags, scores)
    }

    /// A GCC-PHAT lag inside the search range.
    fn phat_lag_hint(
        &mut self,
        recording: &[f64],
        reference: &[f64],
        min_lag: i64,
        max_lag: i64,
    ) -> Option<i64> {
        if recording.len() < reference.len() || reference.len() < 2 {
            return None;
        }
        let length = (recording.len() + reference.len()).next_power_of_two();
        let recording_spectrum = self.forward(recording, length);
        let reference_spectrum = self.forward(reference, length);
        let cross = recording_spectrum
            .iter()
            .zip(&reference_spectrum)
            .map(|(a, b)| {
                let product = a * b.conj();
                product / np_maximum(product.norm(), 1e-12)
            })
            .collect();
        let correlation = self.inverse(cross, length);
        let half = (length / 2) as i64;
        let (lags, scores): (Vec<i64>, Vec<f64>) = correlation
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let lag = index as i64;
                (
                    if lag > half { lag - length as i64 } else { lag },
                    value.abs(),
                )
            })
            .filter(|(lag, _)| (min_lag..=max_lag).contains(lag))
            .unzip();
        if lags.is_empty() {
            return None;
        }
        Some(lags[argmax(&scores)])
    }
}

fn parabolic_peak_offset(scores: &[f64], index: usize) -> f64 {
    if index == 0 || index + 1 >= scores.len() {
        return 0.0;
    }
    let (left, center, right) = (scores[index - 1], scores[index], scores[index + 1]);
    let denominator = left - 2.0 * center + right;
    if denominator.abs() < 1e-12 {
        return 0.0;
    }
    clip(0.5 * (left - right) / denominator, -0.5, 0.5)
}

struct Peak {
    lag: f64,
    score: f64,
    margin: f64,
    ambiguity: f64,
}

/// The earliest peak within `direct_path_bias` of the strongest, with its
/// margin over and ambiguity against the strongest off-peak score.
fn pick_peak(lags: &[i64], scores: &[f64], direct_path_bias: f64) -> Peak {
    if scores.is_empty() {
        return Peak {
            lag: 0.0,
            score: 0.0,
            margin: 0.0,
            ambiguity: 1.0,
        };
    }
    let max_index = argmax(scores);
    let threshold = scores[max_index] * direct_path_bias;
    let peak_index = scores
        .iter()
        .position(|score| *score >= threshold)
        .unwrap_or(max_index);
    let score = scores[peak_index];
    let lag = lags[peak_index] as f64 + parabolic_peak_offset(scores, peak_index);

    let radius = (scores.len() / 50).clamp(1, 128);
    let excluded = peak_index.saturating_sub(radius)..peak_index + radius + 1;
    let off_peak: Vec<f64> = scores
        .iter()
        .enumerate()
        .filter(|(index, _)| !excluded.contains(index))
        .map(|(_, score)| *score)
        .collect();
    let second_peak = if off_peak.is_empty() {
        0.0
    } else {
        np_max(&off_peak)
    };
    Peak {
        lag,
        score,
        margin: py_max(0.0, 1.0 - (second_peak / (score + 1e-6))),
        ambiguity: clip(second_peak / (score + 1e-6), 0.0, 1.0),
    }
}

/// The strongest probe match earlier than the selected one, as a noise
/// z-score, after projecting out the selected copy and any later one.
fn earlier_probe_match_z(
    correlator: &mut Correlator,
    recording: &[f64],
    reference: &[f64],
    min_lag: i64,
    max_lag: i64,
    late_center: i64,
    exclude_radius: i64,
) -> Result<f64, LatencyError> {
    let reference_energy: f64 = reference.iter().map(|x| x * x).sum();
    if reference_energy <= 1e-12 {
        return Ok(0.0);
    }
    let recording_correlations = correlator.full(recording, reference);
    let reference_correlations = correlator.full(reference, reference);
    let size = reference.len() as i64;
    let lag_offset = size - 1;
    let last_full_lag = recording.len() as i64 - size;
    let base_lags = (late_center - 5).max(0)..=(late_center + 5).min(last_full_lag);
    if base_lags.is_empty() {
        return Ok(0.0);
    }

    let autocorrelation_at = |lag: i64| -> f64 {
        if lag.abs() >= size {
            0.0
        } else {
            reference_correlations[(lag + lag_offset) as usize]
        }
    };
    let shifted_cross_correlations = |lags: &[i64], position: f64| -> Vec<f64> {
        lags.iter()
            .map(|&lag| {
                let relative = lag as f64 - position;
                let lower = relative.floor() as i64;
                let fraction = relative - lower as f64;
                let mut cross = 0.0;
                if lower.abs() < size {
                    cross +=
                        (1.0 - fraction) * reference_correlations[(lower + lag_offset) as usize];
                }
                if (lower + 1).abs() < size {
                    cross += fraction * reference_correlations[(lower + 1 + lag_offset) as usize];
                }
                cross
            })
            .collect()
    };
    let shifted_energy = |position: f64| -> f64 {
        let fraction = position - position.floor();
        reference_energy
            + 2.0 * fraction * (1.0 - fraction) * (autocorrelation_at(1) - reference_energy)
    };
    let recording_correlation_at = |position: f64| -> Result<f64, LatencyError> {
        let base = position.floor();
        let fraction = position - base;
        let base = base as i64;
        Ok(
            (1.0 - fraction) * at(&recording_correlations, base + lag_offset)?
                + fraction * at(&recording_correlations, base + 1 + lag_offset)?,
        )
    };

    let fractions: Vec<f64> = (0..=100)
        .map(|index| {
            if index == 100 {
                1.0
            } else {
                index as f64 * (1.0 / 100.0) + 0.0
            }
        })
        .collect();
    let adjacent_correlation = reference_correlations[(lag_offset + 1) as usize];
    let template_energies: Vec<f64> = fractions
        .iter()
        .map(|fraction| {
            reference_energy
                + 2.0 * fraction * (1.0 - fraction) * (adjacent_correlation - reference_energy)
        })
        .collect();
    let mut best_score = -1.0;
    let mut late_lag = late_center as f64;
    let mut late_energy = reference_energy;
    let mut late_correlation = 0.0;
    for base_lag in base_lags {
        let first = recording_correlations[(base_lag + lag_offset) as usize];
        let second = recording_correlations[(base_lag + 1 + lag_offset) as usize];
        let scores: Vec<f64> = fractions
            .iter()
            .zip(&template_energies)
            .map(|(fraction, energy)| {
                ((1.0 - fraction) * first + fraction * second).abs() / energy.sqrt()
            })
            .collect();
        let fraction_index = argmax(&scores);
        let score = scores[fraction_index];
        if score > best_score {
            let fraction = fractions[fraction_index];
            late_lag = base_lag as f64 + fraction;
            late_energy = template_energies[fraction_index];
            late_correlation = (1.0 - fraction) * first + fraction * second;
            best_score = score;
        }
    }

    let candidate_lags: Vec<i64> = (min_lag..=max_lag.min(last_full_lag)).collect();
    let first_cross = shifted_cross_correlations(&candidate_lags, late_lag);
    let first_projection: Vec<f64> = first_cross.iter().map(|x| x / late_energy).collect();
    let mut residual_energy: Vec<f64> = first_cross
        .iter()
        .zip(&first_projection)
        .map(|(cross, projection)| np_maximum(reference_energy - cross * projection, 1e-12))
        .collect();
    let candidate_correlations = candidate_lags
        .iter()
        .map(|lag| at(&recording_correlations, lag + lag_offset))
        .collect::<Result<Vec<f64>, LatencyError>>()?;
    let mut residual_correlations: Vec<f64> = candidate_correlations
        .iter()
        .zip(&first_projection)
        .map(|(correlation, projection)| correlation - projection * late_correlation)
        .collect();

    let recording_median = median(recording);
    let deviations: Vec<f64> = recording
        .iter()
        .map(|sample| (sample - recording_median).abs())
        .collect();
    let noise_sigma = py_max(1.4826 * median(&deviations), 1e-8);
    let z_scores = |correlations: &[f64], energies: &[f64]| -> Vec<f64> {
        correlations
            .iter()
            .zip(energies)
            .map(|(correlation, energy)| correlation.abs() / (noise_sigma * energy.sqrt()))
            .collect()
    };
    let gains = |correlations: &[f64], energies: &[f64]| -> Vec<f64> {
        correlations
            .iter()
            .zip(energies)
            .map(|(correlation, energy)| correlation.abs() / energy)
            .collect()
    };
    let first_z_scores = z_scores(&residual_correlations, &residual_energy);
    let first_gains = gains(&residual_correlations, &residual_energy);
    let reference_peak = np_max(&reference.iter().map(|x| x.abs()).collect::<Vec<_>>());
    let recording_peak = np_max(&recording.iter().map(|x| x.abs()).collect::<Vec<_>>());
    let gain_floor = 1e-5 * recording_peak / py_max(reference_peak, 1e-12);
    let later: Vec<bool> = candidate_lags
        .iter()
        .zip(&first_gains)
        .map(|(lag, gain)| *lag as f64 > late_lag + exclude_radius as f64 && *gain >= gain_floor)
        .collect();
    let masked: Vec<f64> = later
        .iter()
        .zip(&first_z_scores)
        .map(|(later, z)| if *later { *z } else { -1.0 })
        .collect();
    let later_index = argmax(&masked);

    if later.iter().any(|later| *later) && first_z_scores[later_index] >= 6.0 {
        let mut later_position = candidate_lags[later_index] as f64;
        if 0 < later_index && later_index + 1 < candidate_lags.len() {
            later_position += parabolic_peak_offset(&first_z_scores, later_index);
        }
        let second_energy = shifted_energy(later_position);
        let second_correlation = recording_correlation_at(later_position)?;
        let first_fraction = late_lag - late_lag.floor();
        let first_base = late_lag.floor() as i64;
        let second_base = later_position.floor() as i64;
        let second_fraction = later_position - second_base as f64;
        let first_second_cross = (1.0 - first_fraction)
            * ((1.0 - second_fraction) * autocorrelation_at(first_base - second_base)
                + second_fraction * autocorrelation_at(first_base - second_base - 1))
            + first_fraction
                * ((1.0 - second_fraction) * autocorrelation_at(first_base + 1 - second_base)
                    + second_fraction * autocorrelation_at(first_base - second_base));
        let determinant = late_energy * second_energy - first_second_cross * first_second_cross;
        if determinant > 1e-10 * late_energy * second_energy {
            let first_beta = (late_correlation * second_energy
                - second_correlation * first_second_cross)
                / determinant;
            let second_beta = (second_correlation * late_energy
                - late_correlation * first_second_cross)
                / determinant;
            if second_beta.abs() >= gain_floor {
                let second_cross = shifted_cross_correlations(&candidate_lags, later_position);
                residual_correlations = candidate_correlations
                    .iter()
                    .zip(first_cross.iter().zip(&second_cross))
                    .map(|(correlation, (first, second))| {
                        correlation - first * first_beta - second * second_beta
                    })
                    .collect();
                residual_energy = first_cross
                    .iter()
                    .zip(&second_cross)
                    .map(|(first, second)| {
                        let first_solution =
                            (second_energy * first - first_second_cross * second) / determinant;
                        let second_solution =
                            (late_energy * second - first_second_cross * first) / determinant;
                        np_maximum(
                            reference_energy - first * first_solution - second * second_solution,
                            1e-12,
                        )
                    })
                    .collect();
            }
        }
    }

    let final_z_scores = z_scores(&residual_correlations, &residual_energy);
    let final_gains = gains(&residual_correlations, &residual_energy);
    let mut earlier: Vec<f64> = vec![0.0];
    earlier.extend(
        candidate_lags
            .iter()
            .zip(final_z_scores.iter().zip(&final_gains))
            .filter(|(lag, (_, gain))| {
                (**lag as f64) < late_lag - exclude_radius as f64 && **gain >= gain_floor
            })
            .map(|(_, (z, _))| *z),
    );
    Ok(np_max(&earlier))
}

fn mean_removed(signal: &[f64]) -> Vec<f64> {
    if signal.is_empty() {
        return Vec::new();
    }
    let mean = numpy_mean(signal);
    signal.iter().map(|sample| sample - mean).collect()
}

/// Estimate the selected output-to-input route latency.
///
/// The probe is played on the selected output and measured on the selected
/// input: one concrete route, not a symmetric round trip, so the measured
/// delay is applied directly and never halved.
///
/// `sample_rate` is called once the route, the input and its length pass,
/// where the removed Python first used the rate.
pub fn analyze_latency(
    reference_probe: Option<&[f64]>,
    recorded_signal: Option<&[f64]>,
    sample_rate: impl FnOnce() -> Result<f64, LatencyError>,
    window: SearchWindow,
    route_kind: &str,
    hanning: &mut HannWindow<'_>,
) -> Result<LatencyAnalysis, LatencyError> {
    if route_kind != DEFAULT_ROUTE_KIND {
        return Ok(LatencyAnalysis {
            route_kind: route_kind.to_string(),
            ..LatencyAnalysis::refused("Unsupported latency route; expected output_to_input.")
        });
    }
    let (Some(reference_probe), Some(recorded_signal)) = (reference_probe, recorded_signal) else {
        return Ok(LatencyAnalysis::refused("Missing probe or recording."));
    };
    let reference = mean_removed(reference_probe);
    let recording = mean_removed(recorded_signal);
    if reference.len() < 16 || recording.len() < reference.len() {
        return Ok(LatencyAnalysis::refused(
            "Recording too short for reliable correlation.",
        ));
    }

    let rate = sample_rate()?;
    let mut min_lag = py_int((window.min_search_ms / 1000.0) * rate)?;
    let mut max_lag = py_int((window.max_search_ms / 1000.0) * rate)?;
    let expected_min = window
        .expected_latency_min_ms
        .unwrap_or(window.min_search_ms);
    let expected_max = window
        .expected_latency_max_ms
        .unwrap_or(window.max_search_ms);
    let mut playback_min_ms = 0.0;
    let mut playback_max_ms = 0.0;
    if let Some(start_ms) = window.expected_playback_start_ms {
        let jitter_ms = py_max(0.0, window.expected_playback_jitter_ms.unwrap_or(0.0));
        playback_min_ms = py_max(0.0, start_ms - jitter_ms);
        playback_max_ms = py_max(playback_min_ms, start_ms + jitter_ms);
        min_lag = py_int(((playback_min_ms + expected_min) / 1000.0) * rate)?;
        max_lag = py_int(((playback_max_ms + expected_max) / 1000.0) * rate)?;
    }
    if max_lag <= min_lag {
        return Ok(LatencyAnalysis::refused(
            "Search window is outside valid lag range.",
        ));
    }

    let (mut burst, mut offsets) = coded_probe_components(rate, reference.len() as i64, hanning)?;
    if burst.len() < 16 || offsets.is_empty() {
        burst = reference.clone();
        offsets = vec![0];
    }

    let mut correlator = Correlator::new();
    let (full_lags, full_scores) =
        correlator.normalized_scores(&recording, &reference, min_lag, max_lag);
    if full_lags.is_empty() {
        return Ok(LatencyAnalysis::refused(
            "Search window does not overlap captured audio.",
        ));
    }
    // Individual bursts reveal the earliest direct arrival even when a louder
    // delayed copy dominates correlation against the full repeated probe.
    let (coarse_lags, coarse_scores) =
        correlator.normalized_scores(&recording, &burst, min_lag, max_lag);
    let coarse = pick_peak(&coarse_lags, &coarse_scores, 0.94);
    let full = pick_peak(&full_lags, &full_scores, 0.985);
    let local_radius = py_round(rate * 0.010)?.max(burst.len() as i64);
    let full_max_lag = full_lags[argmax(&full_scores)];
    let mut earlier_probe_z = 0.0;
    if (full_max_lag as f64 - coarse.lag).abs() <= local_radius as f64 {
        earlier_probe_z = earlier_probe_match_z(
            &mut correlator,
            &recording,
            &reference,
            min_lag,
            max_lag,
            full_max_lag,
            local_radius,
        )?;
    }
    let unresolved_earlier_path = coarse_lags
        .iter()
        .zip(&coarse_scores)
        .any(|(lag, score)| (*lag as f64) < coarse.lag - local_radius as f64 && *score >= 0.25)
        || earlier_probe_z >= 6.0;

    let mut estimates = Vec::new();
    let mut peak_values = vec![full.score];
    let mut margins = vec![full.margin];
    let mut ambiguities = vec![full.ambiguity];
    let mut phat_matches = Vec::new();
    for &offset in &offsets {
        let expected_lag = coarse.lag + offset as f64;
        let lag_min = (min_lag + offset).max(py_round(expected_lag - local_radius as f64)?);
        let lag_max = (max_lag + offset).min(py_round(expected_lag + local_radius as f64)?);
        let (lag_window, scores) =
            correlator.normalized_scores(&recording, &burst, lag_min, lag_max);
        if lag_window.is_empty() {
            continue;
        }
        let peak = pick_peak(&lag_window, &scores, 0.94);
        if peak.score < 0.035 {
            continue;
        }
        let start_estimate = peak.lag - offset as f64;
        estimates.push(start_estimate);
        peak_values.push(peak.score);
        margins.push(peak.margin);
        ambiguities.push(peak.ambiguity);
        if let Some(hint) = correlator.phat_lag_hint(&recording, &burst, lag_min, lag_max) {
            let phat_start = (hint - offset) as f64;
            phat_matches.push(py_max(
                0.0,
                1.0 - (phat_start - start_estimate).abs() / py_max(1.0, rate * 0.006),
            ));
        }
    }

    if estimates.is_empty() {
        let peak = pick_peak(&full_lags, &full_scores, 0.985);
        estimates = vec![peak.lag];
        peak_values = vec![peak.score];
        margins = vec![peak.margin];
        ambiguities = vec![peak.ambiguity];
    }

    let median_start_samples = median(&estimates);
    let deviations: Vec<f64> = estimates
        .iter()
        .map(|estimate| (estimate - median_start_samples).abs())
        .collect();
    let agreement_ms = (percentile_75(&deviations) * 1000.0) / rate;

    let mut measured_route_ms = (median_start_samples * 1000.0) / rate;
    if let Some(start_ms) = window.expected_playback_start_ms {
        measured_route_ms = py_max(0.0, measured_route_ms - start_ms);
    }

    let peak_value = median(&peak_values);
    let margin_ratio = median(&margins);
    let ambiguity_score = median(&ambiguities);
    let phat_score = if phat_matches.is_empty() {
        0.5
    } else {
        median(&phat_matches)
    };

    let peak_strength_score = clip((peak_value - 0.06) / 0.24, 0.0, 1.0);
    let agreement_score = clip(1.0 - agreement_ms / 4.0, 0.0, 1.0);
    let repetition_score = clip(
        estimates.len() as f64 / offsets.len().clamp(1, 3) as f64,
        0.0,
        1.0,
    );
    let margin_score = clip(margin_ratio / 0.28, 0.0, 1.0);
    let ambiguity_confidence = clip(1.0 - ambiguity_score, 0.0, 1.0);
    let mut confidence = 0.24 * peak_strength_score
        + 0.24 * agreement_score
        + 0.18 * repetition_score
        + 0.14 * margin_score
        + 0.12 * ambiguity_confidence
        + 0.08 * phat_score;
    if window.expected_playback_start_ms.is_some() {
        let expected_center_ms =
            0.5 * (playback_min_ms + playback_max_ms + expected_min + expected_max);
        let expected_center_samples = py_int((expected_center_ms / 1000.0) * rate)?;
        let half_width_samples = ((max_lag - min_lag) / 2).max(1);
        let alignment_score = py_max(
            0.0,
            1.0 - ((median_start_samples - expected_center_samples as f64).abs()
                / half_width_samples as f64),
        );
        confidence = 0.88 * confidence + 0.12 * alignment_score;
    }

    // Short-burst normalized matches are more common in noise than full probes.
    let success = confidence >= 0.32
        && measured_route_ms > 0.0
        && peak_value >= 0.07
        && coarse.score >= 0.35
        && !unresolved_earlier_path
        && ambiguity_score < 0.90
        && estimates.len() >= offsets.len().min(2)
        && agreement_ms <= 6.0;
    let message = if success {
        "ok"
    } else if unresolved_earlier_path {
        "Echo ambiguity: an earlier path could not be resolved reliably."
    } else if agreement_ms > 6.0 && estimates.len() > 1 {
        "Repeated probes disagree; echoes or bleed make latency ambiguous."
    } else if ambiguity_score > 0.82 {
        "Echo ambiguity: competing correlation peaks are too close."
    } else {
        "Low confidence or ambiguous coded-probe correlation."
    };

    Ok(LatencyAnalysis {
        success,
        measured_round_trip_ms: measured_route_ms,
        confidence,
        peak_sample_offset: py_round(median_start_samples)?,
        message: message.to_string(),
        repetition_count: estimates.len(),
        agreement_ms,
        ambiguity_score,
        sub_sample_offset: median_start_samples,
        route_latency_ms: measured_route_ms,
        route_kind: route_kind.to_string(),
    })
}
