//! Offline analysis primitives shared by Auto-EQ and Voice Setup.
//!
//! These replaced NumPy implementations and repeat their arithmetic
//! operation for operation: NumPy's pairwise summation and the sequential
//! dot product its matmul uses for the strided polyphase operands. Values
//! NumPy computed with its own transcendental functions (the resampling
//! filter, the probe window) or its own sort (the peak priority order) come
//! from NumPy through the callers, so results are bit-identical on any CPU.

use std::ops::{AddAssign, Mul};

use realfft::num_complex::Complex;

pub mod headroom;
pub mod latency;
pub(crate) mod python_api;
#[cfg(test)]
mod tests;

const SILERO_WINDOW_SAMPLES: i64 = 512;
const SILERO_SAMPLE_RATE: i64 = 16_000;
const VAD_ANALYSIS_SAMPLE_RATE: i64 = 48_000;

/// Python's `max(low, min(high, value))`: a NaN value yields `high`.
pub fn clamp(value: f64, low: f64, high: f64) -> f64 {
    let upper = if value < high { value } else { high };
    if upper > low {
        upper
    } else {
        low
    }
}

/// NumPy's pairwise summation of a contiguous float64 array.
fn pairwise_sum(values: &[f64]) -> f64 {
    let count = values.len();
    if count < 8 {
        let mut sum = 0.0;
        for value in values {
            sum += value;
        }
        sum
    } else if count <= 128 {
        let mut lanes = [0.0; 8];
        lanes.copy_from_slice(&values[..8]);
        let blocked = count - count % 8;
        for block in values[8..blocked].chunks_exact(8) {
            for (lane, value) in lanes.iter_mut().zip(block) {
                *lane += value;
            }
        }
        let mut sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        for value in &values[blocked..] {
            sum += value;
        }
        sum
    } else {
        let mut half = count / 2;
        half -= half % 8;
        pairwise_sum(&values[..half]) + pairwise_sum(&values[half..])
    }
}

/// `np.sum`: the reduction starts from the identity `+0.0`.
fn numpy_sum(values: &[f64]) -> f64 {
    0.0 + pairwise_sum(values)
}

/// `np.mean`; NaN for an empty slice.
fn numpy_mean(values: &[f64]) -> f64 {
    numpy_sum(values) / values.len() as f64
}

/// Mean of `values` over `low..=high` Hz, or over everything when no bin falls in it.
pub fn band_mean(freqs: &[f64], values: &[f64], low: f64, high: f64) -> Result<f64, &'static str> {
    if freqs.len() != values.len() {
        return Err("freqs and values must have the same length");
    }
    let selected: Vec<f64> = freqs
        .iter()
        .zip(values)
        .filter(|(frequency, _)| **frequency >= low && **frequency <= high)
        .map(|(_, value)| *value)
        .collect();
    Ok(if selected.is_empty() {
        numpy_mean(values)
    } else {
        numpy_mean(&selected)
    })
}

/// A sample type the polyphase resampler runs in; it accumulates in its own precision.
///
/// Complex samples multiply the real taps as complex numbers with a zero
/// imaginary part, as NumPy did, so a non-finite imaginary part reaches the
/// real output.
pub trait ResampleSample: Copy + Default + AddAssign + Mul<Output = Self> {
    fn from_f64(value: f64) -> Self;
}

impl ResampleSample for f32 {
    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl ResampleSample for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }
}

impl ResampleSample for Complex<f32> {
    fn from_f64(value: f64) -> Self {
        Complex::new(value as f32, 0.0)
    }
}

impl ResampleSample for Complex<f64> {
    fn from_f64(value: f64) -> Self {
        Complex::new(value, 0.0)
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `up` and `down` reduced by their greatest common divisor.
pub fn reduced_rates(up: usize, down: usize) -> Result<(usize, usize), &'static str> {
    if up < 1 || down < 1 {
        return Err("up and down must be positive integers");
    }
    let divisor = gcd(up, down);
    Ok((up / divisor, down / divisor))
}

/// How many taps the default filter for reduced rates `up`/`down` has.
pub fn resample_tap_count(up: usize, down: usize) -> usize {
    2 * 10 * up.max(down) + 1
}

/// Zero-phase, zero-padded resampling by reduced rates `up`/`down` with
/// SciPy `resample_poly`'s centered output.
///
/// `taps` is the 10-lobe Kaiser-windowed sinc SciPy designs, normalized and
/// scaled by `up`, as float64; it is rounded to the sample type here.
pub fn resample_poly<T: ResampleSample>(
    samples: &[T],
    up: usize,
    down: usize,
    taps: &[f64],
) -> Result<Vec<T>, &'static str> {
    if up < 1 || down < 1 || gcd(up, down) != 1 {
        return Err("up and down must be positive, reduced rates");
    }
    if taps.len() != resample_tap_count(up, down) {
        return Err("the filter must have 20 * max(up, down) + 1 taps");
    }
    if up == 1 && down == 1 {
        return Ok(samples.to_vec());
    }
    if samples.is_empty() {
        return Ok(Vec::new());
    }

    let half_length = 10 * up.max(down);
    let pad_before = down - half_length % down;
    let trim_before = (half_length + pad_before) / down;
    let mut padded_taps = vec![T::default(); pad_before];
    padded_taps.extend(taps.iter().copied().map(T::from_f64));
    let taps = padded_taps;

    let output_size = (samples.len() * up).div_ceil(down);
    let terms = taps.len().div_ceil(up);
    let mut padded = vec![T::default(); samples.len() + 2 * (terms - 1)];
    padded[terms - 1..terms - 1 + samples.len()].copy_from_slice(samples);
    let mut output = vec![T::default(); output_size];
    for residue in 0..up.min(output_size) {
        let group_size = (output_size - 1 - residue) / up + 1;
        let first_high_rate_position = (residue + trim_before) * down;
        let phase = first_high_rate_position % up;
        let first_center = first_high_rate_position / up;
        let phase_taps: Vec<T> = taps[phase..].iter().step_by(up).copied().collect();
        let skip = terms - phase_taps.len();
        for row in 0..group_size {
            let start = first_center + row * down + skip;
            let mut sum = T::default();
            for (sample, tap) in padded[start..start + phase_taps.len()]
                .iter()
                .zip(phase_taps.iter().rev())
            {
                sum += *sample * *tap;
            }
            output[residue + row * up] = sum;
        }
    }
    Ok(output)
}

/// Local maxima of `values`, a plateau counting once at its middle (left of
/// center for an even plateau), as SciPy's `find_peaks` finds them.
pub fn peak_candidates(values: &[f64]) -> Vec<usize> {
    let mut candidates = Vec::new();
    let mut index = 1;
    while index + 1 < values.len() {
        if values[index - 1] < values[index] {
            let mut plateau_end = index;
            while plateau_end + 1 < values.len() && values[plateau_end + 1] == values[index] {
                plateau_end += 1;
            }
            if plateau_end + 1 < values.len() && values[plateau_end] > values[plateau_end + 1] {
                candidates.push((index + plateau_end) / 2);
            }
            index = plateau_end + 1;
        } else {
            index += 1;
        }
    }
    candidates
}

/// The `candidates` that SciPy `find_peaks`' distance and prominence rules keep.
///
/// `ascending_order` is `np.argsort` of the candidates' heights: higher
/// peaks claim their neighbourhood first, and among equal heights NumPy's
/// sort decides, so the caller passes NumPy's order rather than one sorted
/// here.
pub fn select_peaks(
    values: &[f64],
    candidates: &[usize],
    ascending_order: &[usize],
    distance: f64,
    prominence: f64,
) -> Result<Vec<usize>, &'static str> {
    if distance < 1.0 || prominence < 0.0 {
        return Err("distance must be positive and prominence nonnegative");
    }
    let mut seen = vec![false; candidates.len()];
    if ascending_order.len() != candidates.len()
        || ascending_order
            .iter()
            .any(|&index| index >= candidates.len() || std::mem::replace(&mut seen[index], true))
        || candidates.iter().any(|&peak| peak >= values.len())
    {
        return Err("the priority order must be a permutation of the candidates");
    }

    let priority: Vec<usize> = ascending_order.iter().rev().copied().collect();
    let mut keep = vec![true; candidates.len()];
    for (order_index, &candidate) in priority.iter().enumerate() {
        if !keep[candidate] {
            continue;
        }
        if priority[..order_index].iter().any(|&higher| {
            keep[higher] && (candidates[candidate].abs_diff(candidates[higher]) as f64) < distance
        }) {
            keep[candidate] = false;
        }
    }

    let slice_min = |slice: &[f64]| {
        slice.iter().copied().fold(
            f64::INFINITY,
            |low, value| if value < low { value } else { low },
        )
    };
    Ok(candidates
        .iter()
        .zip(keep)
        .filter(|(_, kept)| *kept)
        .map(|(&peak, _)| peak)
        .filter(|&peak| {
            let height = values[peak];
            let mut left = peak;
            while left > 0 && values[left - 1] <= height {
                left -= 1;
            }
            let mut right = peak;
            while right + 1 < values.len() && values[right + 1] <= height {
                right += 1;
            }
            let left_base = slice_min(&values[left..=peak]);
            let right_base = slice_min(&values[peak..=right]);
            let base = if right_base > left_base {
                right_base
            } else {
                left_base
            };
            height - base >= prominence
        })
        .collect())
}

/// Whether `map_causal_vad_probabilities` has posteriors to map.
fn usable_posteriors(probabilities: &[f64], positions: &[i64]) -> bool {
    !probabilities.is_empty()
        && !positions.is_empty()
        && probabilities
            .iter()
            .all(|probability| probability.is_finite())
}

/// Per position, the newest posterior whose 32 ms window ended by it, or
/// `None` without finite posteriors or positions.
///
/// To control processing, pass each block's first sample: the live loop
/// reads the VAD before the worker can see that block, so a window ending
/// inside a block cannot control it. Offline analysis that only labels
/// frames may pass frame ends.
///
/// `sample_rate` gives the whole rate (the removed Python's `int()` of a
/// positive rate) and is called only when there is something to map, where
/// the removed Python converted it. A zero rate maps every position to 0,
/// as NumPy's integer division by zero did.
pub fn map_causal_vad_probabilities<E>(
    probabilities: &[f64],
    positions: &[i64],
    sample_rate: impl FnOnce() -> Result<i64, E>,
) -> Result<Option<Vec<f32>>, E> {
    if !usable_posteriors(probabilities, positions) {
        return Ok(None);
    }
    let sample_rate = sample_rate()?;
    if sample_rate < 0 {
        return Ok(None);
    }
    // Integer time keeps the model's cadence: rounding a 44.1 kHz window to
    // 1412 samples accumulates a sample of drift every window.
    let window_span = i128::from(sample_rate) * i128::from(SILERO_WINDOW_SAMPLES);
    let last = probabilities.len() - 1;
    Ok(Some(
        positions
            .iter()
            .map(|&position| {
                // NumPy multiplied in int64, wrapping.
                let scaled = i128::from(position.wrapping_mul(SILERO_SAMPLE_RATE));
                let completed = if window_span == 0 {
                    -1
                } else {
                    scaled.div_euclid(window_span) - 1
                };
                match usize::try_from(completed) {
                    Ok(completed) => {
                        let probability = probabilities[completed.min(last)];
                        let probability = if probability > 0.0 { probability } else { 0.0 };
                        (if probability < 1.0 { probability } else { 1.0 }) as f32
                    }
                    Err(_) => 0.0,
                }
            })
            .collect(),
    ))
}

/// Silero posteriors for an offline capture, or `None` when the analysis
/// must fall back to energy.
///
/// Rates without whole 32 ms windows are resampled to 48 kHz first, with
/// the filter `design_taps` returns for the reduced rates, and `pre_gain`
/// (at least 0.1) is applied before inference, matching the live model's
/// input. `infer` runs the model on the prepared samples and rate. A
/// failure to design the filter is returned, not a fallback.
pub fn offline_vad_probabilities<E, F>(
    audio: &[f32],
    sample_rate: i64,
    pre_gain: f64,
    design_taps: impl FnOnce(usize, usize) -> Result<Vec<f64>, F>,
    infer: impl FnOnce(&[f32], u32) -> Result<Vec<f32>, E>,
) -> Result<Option<Vec<f64>>, F> {
    if !pre_gain.is_finite() || pre_gain <= 0.0 || audio.is_empty() || sample_rate <= 0 {
        return Ok(None);
    }
    let gain = pre_gain.max(0.1);
    let window_span = i128::from(sample_rate) * i128::from(SILERO_WINDOW_SAMPLES);
    let (mut samples, inference_rate) = if window_span % i128::from(SILERO_SAMPLE_RATE) != 0 {
        let rate = sample_rate as usize;
        let divisor = gcd(rate, VAD_ANALYSIS_SAMPLE_RATE as usize);
        let (up, down) = (VAD_ANALYSIS_SAMPLE_RATE as usize / divisor, rate / divisor);
        let taps = design_taps(up, down)?;
        let widened: Vec<f64> = audio.iter().copied().map(f64::from).collect();
        let resampled = resample_poly(&widened, up, down, &taps)
            .expect("design_taps returns the filter for the reduced rates");
        (
            resampled.into_iter().map(|sample| sample as f32).collect(),
            VAD_ANALYSIS_SAMPLE_RATE,
        )
    } else {
        (audio.to_vec(), sample_rate)
    };
    if gain != 1.0 {
        let gain = gain as f32;
        for sample in &mut samples {
            *sample *= gain;
        }
    }

    let Ok(inference_rate) = u32::try_from(inference_rate) else {
        return Ok(None);
    };
    let Ok(probabilities) = infer(&samples, inference_rate) else {
        return Ok(None);
    };
    if probabilities.is_empty() || probabilities.iter().any(|value| !value.is_finite()) {
        return Ok(None);
    }
    Ok(Some(
        probabilities
            .into_iter()
            .map(|probability| f64::from(probability).clamp(0.0, 1.0))
            .collect(),
    ))
}
