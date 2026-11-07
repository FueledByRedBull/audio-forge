//! Offline analysis primitives shared by Auto-EQ and Voice Setup.
//!
//! These replaced NumPy implementations and repeat their arithmetic
//! operation for operation: NumPy's pairwise summation, the sequential dot
//! product its matmul uses for the strided polyphase operands, and the
//! Cephes `i0` series behind `np.kaiser`. Results are bit-identical to the
//! NumPy versions wherever the platform `sin` and `exp` match NumPy's.

use std::f64::consts::PI;
use std::ops::{AddAssign, Mul};

pub mod headroom;
pub mod latency;
pub(crate) mod python_api;
#[cfg(test)]
mod tests;

const SILERO_WINDOW_SAMPLES: i64 = 512;
const SILERO_SAMPLE_RATE: i64 = 16_000;
const VAD_ANALYSIS_SAMPLE_RATE: i64 = 48_000;
const KAISER_BETA: f64 = 5.0;

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

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Cephes Chebyshev coefficients for `exp(-x) I0(x)` on `[0, 8]` (NumPy's `_i0A`).
const I0_A: [f64; 30] = [
    -4.4153416464793395e-18,
    3.3307945188222384e-17,
    -2.431279846547955e-16,
    1.715391285555133e-15,
    -1.1685332877993451e-14,
    7.676185498604936e-14,
    -4.856446783111929e-13,
    2.95505266312964e-12,
    -1.726826291441556e-11,
    9.675809035373237e-11,
    -5.189795601635263e-10,
    2.6598237246823866e-09,
    -1.300025009986248e-08,
    6.046995022541919e-08,
    -2.670793853940612e-07,
    1.1173875391201037e-06,
    -4.4167383584587505e-06,
    1.6448448070728896e-05,
    -5.754195010082104e-05,
    0.00018850288509584165,
    -0.0005763755745385824,
    0.0016394756169413357,
    -0.004324309995050576,
    0.010546460394594998,
    -0.02373741480589947,
    0.04930528423967071,
    -0.09490109704804764,
    0.17162090152220877,
    -0.3046826723431984,
    0.6767952744094761,
];

/// `np.i0` for `0 <= x <= 8`, the only range a beta-5 Kaiser window reaches.
fn bessel_i0_up_to_8(x: f64) -> f64 {
    let t = x / 2.0 - 2.0;
    let mut b0 = I0_A[0];
    let mut b1 = 0.0;
    let mut b2 = 0.0;
    for coefficient in &I0_A[1..] {
        b2 = b1;
        b1 = b0;
        b0 = t * b1 - b2 + coefficient;
    }
    x.exp() * (0.5 * (b0 - b2))
}

/// The 10-lobe Kaiser-windowed sinc SciPy's `resample_poly` designs, scaled by `up`.
fn kaiser_sinc_taps(half_length: usize, max_rate: usize, up: usize) -> Vec<f64> {
    let count = 2 * half_length + 1;
    let cutoff = 1.0 / max_rate as f64;
    let alpha = (count as f64 - 1.0) / 2.0;
    let window_norm = bessel_i0_up_to_8(KAISER_BETA);
    let mut taps: Vec<f64> = (0..count)
        .map(|index| {
            let x = PI * (cutoff * (index as f64 - half_length as f64));
            let y = if x != 0.0 { x } else { f64::EPSILON };
            let ratio = (index as f64 - alpha) / alpha;
            let window =
                bessel_i0_up_to_8(KAISER_BETA * (1.0 - ratio * ratio).sqrt()) / window_norm;
            cutoff * (y.sin() / y) * window
        })
        .collect();
    let sum = numpy_sum(&taps);
    for tap in &mut taps {
        *tap = *tap / sum * up as f64;
    }
    taps
}

/// Zero-phase, zero-padded Kaiser resampling with SciPy `resample_poly`'s
/// default filter and centered output.
pub fn resample_poly<T: ResampleSample>(
    samples: &[T],
    up: usize,
    down: usize,
) -> Result<Vec<T>, &'static str> {
    if up < 1 || down < 1 {
        return Err("up and down must be positive integers");
    }
    let divisor = gcd(up, down);
    let (up, down) = (up / divisor, down / divisor);
    if up == 1 && down == 1 {
        return Ok(samples.to_vec());
    }
    if samples.is_empty() {
        return Ok(Vec::new());
    }

    let max_rate = up.max(down);
    let half_length = 10 * max_rate;
    let pad_before = down - half_length % down;
    let trim_before = (half_length + pad_before) / down;
    let mut taps = vec![T::default(); pad_before];
    taps.extend(
        kaiser_sinc_taps(half_length, max_rate, up)
            .into_iter()
            .map(T::from_f64),
    );

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

/// NumPy's ascending sort order for floats: NaN last.
fn ascending_nan_last(a: f64, b: f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (false, false) => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
        (a_nan, b_nan) => a_nan.cmp(&b_nan),
    }
}

/// 1-D local maxima with SciPy `find_peaks`' distance and prominence semantics.
///
/// Peaks of equal height are ranked later-index first, the order NumPy's
/// argsort gives them for small inputs; NumPy's own order for larger tied
/// inputs depends on its SIMD sort.
pub fn find_peaks_distance_prominence(
    values: &[f64],
    distance: usize,
    prominence: f64,
) -> Result<Vec<usize>, &'static str> {
    if distance < 1 || prominence < 0.0 {
        return Err("distance must be positive and prominence nonnegative");
    }

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

    let mut priority: Vec<usize> = (0..candidates.len()).collect();
    priority.sort_by(|&a, &b| {
        ascending_nan_last(values[candidates[a]], values[candidates[b]]).then(a.cmp(&b))
    });
    priority.reverse();
    let mut keep = vec![true; candidates.len()];
    for (order_index, &candidate) in priority.iter().enumerate() {
        if !keep[candidate] {
            continue;
        }
        if priority[..order_index].iter().any(|&higher| {
            keep[higher] && candidates[candidate].abs_diff(candidates[higher]) < distance
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
        .into_iter()
        .zip(keep)
        .filter(|(_, kept)| *kept)
        .map(|(peak, _)| peak)
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

/// Per position, the newest posterior whose 32 ms window ended by it.
///
/// To control processing, pass each block's first sample: the live loop
/// reads the VAD before the worker can see that block, so a window ending
/// inside a block cannot control it. Offline analysis that only labels
/// frames may pass frame ends.
pub fn map_causal_vad_probabilities(
    probabilities: &[f64],
    positions: &[i64],
    sample_rate: i64,
) -> Option<Vec<f32>> {
    if sample_rate <= 0
        || probabilities.is_empty()
        || positions.is_empty()
        || probabilities
            .iter()
            .any(|probability| !probability.is_finite())
    {
        return None;
    }
    // Integer time keeps the model's cadence: rounding a 44.1 kHz window to
    // 1412 samples accumulates a sample of drift every window.
    let window_span = sample_rate * SILERO_WINDOW_SAMPLES;
    let last = probabilities.len() - 1;
    Some(
        positions
            .iter()
            .map(|&position| {
                let completed = position
                    .wrapping_mul(SILERO_SAMPLE_RATE)
                    .div_euclid(window_span)
                    - 1;
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
    )
}

/// Silero posteriors for an offline capture, or `None` when the analysis
/// must fall back to energy.
///
/// Rates without whole 32 ms windows are resampled to 48 kHz first, and
/// `pre_gain` (at least 0.1) is applied before inference, matching the live
/// model's input. `infer` runs the model on the prepared samples and rate.
pub fn offline_vad_probabilities<E>(
    audio: &[f32],
    sample_rate: i64,
    pre_gain: f64,
    infer: impl FnOnce(&[f32], u32) -> Result<Vec<f32>, E>,
) -> Option<Vec<f64>> {
    if !pre_gain.is_finite() || pre_gain <= 0.0 || audio.is_empty() || sample_rate <= 0 {
        return None;
    }
    let gain = pre_gain.max(0.1);
    let (mut samples, inference_rate) =
        if (sample_rate * SILERO_WINDOW_SAMPLES) % SILERO_SAMPLE_RATE != 0 {
            let divisor = gcd(sample_rate as usize, VAD_ANALYSIS_SAMPLE_RATE as usize);
            let widened: Vec<f64> = audio.iter().copied().map(f64::from).collect();
            let resampled = resample_poly(
                &widened,
                VAD_ANALYSIS_SAMPLE_RATE as usize / divisor,
                sample_rate as usize / divisor,
            )
            .ok()?;
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

    let probabilities = infer(&samples, u32::try_from(inference_rate).ok()?).ok()?;
    if probabilities.is_empty() || probabilities.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(
        probabilities
            .into_iter()
            .map(|probability| f64::from(probability).clamp(0.0, 1.0))
            .collect(),
    )
}
