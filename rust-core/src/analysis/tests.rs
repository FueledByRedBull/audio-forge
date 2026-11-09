use std::f64::consts::PI;

use super::*;

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

/// SciPy's default filter with the platform `sin` and `exp`; the bindings
/// pass NumPy's, whose transcendental functions differ on some CPUs.
fn kaiser_taps(up: usize, down: usize) -> Vec<f64> {
    let max_rate = up.max(down);
    let half_length = 10 * max_rate;
    let count = 2 * half_length + 1;
    let cutoff = 1.0 / max_rate as f64;
    let alpha = (count as f64 - 1.0) / 2.0;
    let window_norm = bessel_i0_up_to_8(5.0);
    let mut taps: Vec<f64> = (0..count)
        .map(|index| {
            let x = PI * (cutoff * (index as f64 - half_length as f64));
            let y = if x != 0.0 { x } else { f64::EPSILON };
            let ratio = (index as f64 - alpha) / alpha;
            let window = bessel_i0_up_to_8(5.0 * (1.0 - ratio * ratio).sqrt()) / window_norm;
            cutoff * (y.sin() / y) * window
        })
        .collect();
    let sum = numpy_sum(&taps);
    for tap in &mut taps {
        *tap = *tap / sum * up as f64;
    }
    taps
}

/// `resample_poly` with unreduced rates, as the bindings call it.
fn resample<T: ResampleSample>(samples: &[T], up: usize, down: usize) -> Vec<T> {
    let (up, down) = reduced_rates(up, down).unwrap();
    resample_poly(samples, up, down, &kaiser_taps(up, down)).unwrap()
}

/// SciPy `find_peaks` with a stable ascending order standing in for NumPy's argsort.
fn peaks(values: &[f64], distance: f64, prominence: f64) -> Vec<usize> {
    let candidates = peak_candidates(values);
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by(|&a, &b| values[candidates[a]].total_cmp(&values[candidates[b]]));
    select_peaks(values, &candidates, &order, distance, prominence).unwrap()
}

fn no_taps(_up: usize, _down: usize) -> Result<Vec<f64>, ()> {
    panic!("no resampling expected");
}

#[test]
fn clamp_follows_python_builtin_min_max() {
    assert_eq!(clamp(5.0, 0.0, 3.0), 3.0);
    assert_eq!(clamp(-1.0, 0.0, 3.0), 0.0);
    assert_eq!(clamp(2.0, 0.0, 3.0), 2.0);
    // max(low, min(high, nan)) is high; an inverted range yields low.
    assert_eq!(clamp(f64::NAN, 0.0, 3.0), 3.0);
    assert_eq!(clamp(2.0, 3.0, 1.0), 3.0);
}

#[test]
fn sums_group_like_numpy_pairwise_summation() {
    // A sequential sum loses every 1.0 against 1e16; NumPy's eight lanes keep them.
    let mut short = vec![1e16];
    short.extend([1.0; 8]);
    assert_eq!(numpy_sum(&short), 1.0000000000000008e16);
    let mut long = vec![1e16];
    long.extend([1.0; 130]);
    assert_eq!(numpy_sum(&long), 1.0000000000000124e16);
    assert_eq!(numpy_sum(&[-0.0; 200]).to_bits(), 0.0_f64.to_bits());
    assert!(numpy_mean(&[]).is_nan());
}

#[test]
fn band_mean_averages_the_band_or_the_whole_spectrum() {
    let freqs = [100.0, 200.0, 400.0, 800.0];
    let values = [1.0, 2.0, 4.0, 8.0];
    assert_eq!(band_mean(&freqs, &values, 150.0, 500.0), Ok(3.0));
    assert_eq!(band_mean(&freqs, &values, 1000.0, 2000.0), Ok(3.75));
    assert!(band_mean(&freqs, &values[..3], 150.0, 500.0).is_err());
}

#[test]
fn resample_poly_output_length_follows_the_rate_ratio() {
    for (up, down) in [(1, 3), (3, 2), (160, 147), (147, 160)] {
        for length in [1_usize, 2, 5, 31, 1024, 4410] {
            let samples = vec![0.25_f64; length];
            let output = resample(&samples, up, down);
            assert_eq!(
                output.len(),
                (length * up).div_ceil(down),
                "{up}/{down} {length}"
            );
        }
    }
}

#[test]
fn resample_poly_handles_identity_empty_and_invalid_rates() {
    let samples: Vec<f32> = (0..12).map(|index| index as f32).collect();
    assert_eq!(reduced_rates(2, 2), Ok((1, 1)));
    assert_eq!(reduced_rates(320, 294), Ok((160, 147)));
    assert_eq!(resample(&samples, 2, 2), samples);
    assert!(resample::<f64>(&[], 1, 3).is_empty());
    assert!(reduced_rates(0, 1).is_err());
    assert!(reduced_rates(1, 0).is_err());
    // Unreduced rates and a filter for other rates are refused.
    assert!(resample_poly(&samples, 2, 4, &kaiser_taps(2, 4)).is_err());
    assert!(resample_poly(&samples, 1, 3, &kaiser_taps(1, 2)).is_err());
    assert_eq!(resample_tap_count(160, 147), 3201);
}

#[test]
fn resample_poly_preserves_dc_and_rejects_above_nyquist_energy() {
    let dc = resample(&vec![1.0_f64; 48_000], 1, 3);
    assert!(dc[100..dc.len() - 100]
        .iter()
        .all(|sample| (sample - 1.0).abs() < 1e-12));

    let tone: Vec<f64> = (0..48_000)
        .map(|index| (2.0 * PI * 0.42 * index as f64).sin())
        .collect();
    let filtered = resample(&tone, 1, 4);
    let interior = &filtered[100..filtered.len() - 100];
    let rms =
        (interior.iter().map(|sample| sample * sample).sum::<f64>() / interior.len() as f64).sqrt();
    assert!(rms < 1e-4, "rms {rms}");
}

#[test]
fn resample_poly_float32_matches_float64_within_float32_precision() {
    let samples: Vec<f64> = (0..2048)
        .map(|index| (index as f64 * 0.013).sin() * 0.8)
        .collect();
    let narrow: Vec<f32> = samples.iter().map(|sample| *sample as f32).collect();
    let wide = resample(&samples, 160, 147);
    let single = resample(&narrow, 160, 147);
    assert_eq!(wide.len(), single.len());
    for (a, b) in wide.iter().zip(&single) {
        assert!((a - f64::from(*b)).abs() < 2e-6);
    }
}

#[test]
fn resample_poly_runs_complex_audio_through_the_real_filter() {
    let real: Vec<f64> = (0..600).map(|index| (index as f64 * 0.03).sin()).collect();
    let imaginary: Vec<f64> = (0..600).map(|index| (index as f64 * 0.07).cos()).collect();
    let complex: Vec<Complex<f64>> = real
        .iter()
        .zip(&imaginary)
        .map(|(re, im)| Complex::new(*re, *im))
        .collect();
    let output = resample(&complex, 3, 2);
    let real_output = resample(&real, 3, 2);
    let imaginary_output = resample(&imaginary, 3, 2);
    for ((value, re), im) in output.iter().zip(&real_output).zip(&imaginary_output) {
        assert_eq!(value.re, *re);
        assert_eq!(value.im, *im);
    }

    // NumPy's complex product carries a non-finite imaginary part into the real output.
    let mut with_inf = complex.clone();
    with_inf[300] = Complex::new(0.5, f64::INFINITY);
    let output = resample(&with_inf, 3, 2);
    assert!(output[450].re.is_nan());
}

#[test]
fn peaks_match_scipy_distance_and_prominence() {
    let first = [0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0];
    let second = [0.0, 3.0, 2.0, 3.0, 0.0];
    // Expected values are scipy.signal.find_peaks' output.
    assert_eq!(peaks(&first, 2.0, 0.75), vec![1, 4, 7, 9]);
    assert_eq!(peaks(&first, 3.0, 0.75), vec![1, 4, 9]);
    assert_eq!(peaks(&first, 8.0, 0.75), vec![1, 9]);
    assert_eq!(peaks(&first, 1.0, 1.5), vec![4, 9]);
    assert_eq!(peaks(&second, 2.0, 0.75), vec![1, 3]);
    // Equal heights: the later in the ascending order ranks first.
    assert_eq!(peaks(&second, 3.0, 0.75), vec![3]);
    let candidates = peak_candidates(&second);
    assert_eq!(
        select_peaks(&second, &candidates, &[1, 0], 3.0, 0.75),
        Ok(vec![1])
    );
}

#[test]
fn peak_selection_refuses_an_order_that_is_not_a_permutation() {
    let values = [0.0, 3.0, 2.0, 3.0, 0.0];
    let candidates = peak_candidates(&values);
    assert_eq!(candidates, vec![1, 3]);
    for order in [&[0][..], &[0, 0], &[0, 2], &[1, 0, 1]] {
        assert!(select_peaks(&values, &candidates, order, 1.0, 0.0).is_err());
    }
    assert!(select_peaks(&values, &[1, 9], &[0, 1], 1.0, 0.0).is_err());
}

#[test]
fn peaks_ignore_nan_neighbours_and_short_input() {
    let peaks = |values: &[f64]| peaks(values, 1.0, 0.0);
    assert!(peaks(&[]).is_empty());
    assert!(peaks(&[1.0, 2.0]).is_empty());
    assert!(peaks(&[0.0, 1.0, f64::NAN, 0.0, 2.0, 0.0]) == vec![4]);
    assert!(select_peaks(&[0.0], &[], &[], 0.0, 0.0).is_err());
    assert!(select_peaks(&[0.0], &[], &[], 1.0, -1.0).is_err());
    // A NaN distance suppresses nothing, as `abs(gap) < nan` is false.
    assert_eq!(
        super::select_peaks(&[0.0, 1.0, 0.0, 1.0, 0.0], &[1, 3], &[0, 1], f64::NAN, 0.0),
        Ok(vec![1, 3])
    );
}

fn map(probabilities: &[f64], positions: &[i64], sample_rate: i64) -> Option<Vec<f32>> {
    map_causal_vad_probabilities(probabilities, positions, || Ok::<_, ()>(sample_rate)).unwrap()
}

#[test]
fn causal_vad_mapping_keeps_exact_44k1_window_timing() {
    let probabilities: Vec<f64> = (0..700)
        .map(|index| f64::from(index as f32 / 700.0))
        .collect();
    let frame_ends: Vec<i64> = (1..2_001).map(|index| index * 441).collect();
    let mapped = map(&probabilities, &frame_ends, 44_100).unwrap();

    assert_eq!(&mapped[..3], &[0.0; 3]);
    // 1.6 s contains exactly 50 model windows; 20 s contains 625.
    assert_eq!(mapped[159], probabilities[49] as f32);
    assert_eq!(mapped[1999], probabilities[624] as f32);
}

#[test]
fn causal_vad_mapping_clips_and_rejects_unusable_input() {
    let mapped = map(&[1.5, -0.5], &[-512, 512, 1024, 99_999], 16_000);
    assert_eq!(mapped, Some(vec![0.0, 1.0, 0.0, 0.0]));
    // int() of a rate below 1 is 0; NumPy's integer division by zero gave 0.
    assert_eq!(map(&[0.5], &[512, 10_000], 0), Some(vec![0.0, 0.0]));
    assert_eq!(map(&[0.5], &[512], -1), None);
    assert_eq!(map(&[], &[512], 16_000), None);
    assert_eq!(map(&[0.5], &[], 16_000), None);
    assert_eq!(map(&[f64::NAN], &[512], 16_000), None);
    // A rate too large for 32-bit time still divides exactly.
    assert_eq!(
        map(&[0.25, 0.75], &[i64::MAX / 16_000], i64::MAX / 512),
        Some(vec![0.0])
    );

    // The rate is converted only when there is something to map.
    let unconverted = || -> Result<i64, &str> { panic!("rate converted") };
    assert_eq!(
        map_causal_vad_probabilities(&[], &[512], unconverted),
        Ok(None)
    );
    let failing = || Err("not a number");
    assert_eq!(
        map_causal_vad_probabilities(&[0.5], &[512], failing),
        Err("not a number")
    );
}

#[test]
fn offline_vad_applies_pre_gain_before_inference() {
    let mut observed = None;
    let probabilities =
        offline_vad_probabilities(&[0.5, -0.25], 48_000, 2.0, no_taps, |samples, rate| {
            observed = Some((samples.to_vec(), rate));
            Ok::<_, ()>(vec![0.7])
        });

    assert_eq!(observed, Some((vec![1.0, -0.5], 48_000)));
    assert_eq!(probabilities, Ok(Some(vec![f64::from(0.7_f32)])));
}

#[test]
fn offline_vad_resamples_rates_without_whole_model_windows() {
    let audio = vec![0.1_f32; 4_410];
    let mut observed = None;
    let mut designed = None;
    let design = |up, down| {
        designed = Some((up, down));
        Ok::<_, ()>(kaiser_taps(up, down))
    };
    let probabilities = offline_vad_probabilities(&audio, 44_100, 1.0, design, |samples, rate| {
        observed = Some((samples.len(), rate));
        Ok::<_, ()>(vec![1.2, -0.1])
    });

    assert_eq!(designed, Some((160, 147)));
    assert_eq!(observed, Some((4_800, 48_000)));
    assert_eq!(probabilities, Ok(Some(vec![1.0, 0.0])));

    // A filter design failure is the caller's error, not an energy fallback.
    let failing = offline_vad_probabilities(
        &audio,
        44_100,
        1.0,
        |_, _| Err("no filter"),
        |_: &[f32], _: u32| Ok::<_, ()>(vec![0.5]),
    );
    assert_eq!(failing, Err(VadResampleError::Design("no filter")));

    // A filter for other rates is refused rather than run.
    let mismatched = offline_vad_probabilities(
        &audio,
        44_100,
        1.0,
        |_, _| Ok::<_, ()>(kaiser_taps(1, 2)),
        |_: &[f32], _: u32| -> Result<Vec<f32>, ()> { panic!("inference must not run") },
    );
    assert!(matches!(mismatched, Err(VadResampleError::Filter(_))));
}

#[test]
fn offline_vad_falls_back_on_unusable_gain_capture_or_result() {
    let unreachable = |_: &[f32], _: u32| -> Result<Vec<f32>, ()> {
        panic!("inference must not run");
    };
    let fallback = Ok(None);
    for gain in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            offline_vad_probabilities(&[0.5], 48_000, gain, no_taps, unreachable),
            fallback
        );
    }
    assert_eq!(
        offline_vad_probabilities(&[], 48_000, 1.0, no_taps, unreachable),
        fallback
    );
    assert_eq!(
        offline_vad_probabilities(&[0.5], 0, 1.0, no_taps, unreachable),
        fallback
    );

    let mut floored = None;
    let _ = offline_vad_probabilities(&[1.0], 48_000, 0.01, no_taps, |samples, _| {
        floored = Some(samples[0]);
        Ok::<_, ()>(vec![0.5])
    });
    assert_eq!(floored, Some(0.1_f32));

    let failing = |_: &[f32], _: u32| Err::<Vec<f32>, _>("model missing");
    assert_eq!(
        offline_vad_probabilities(&[0.5], 48_000, 1.0, no_taps, failing),
        fallback
    );
    let empty = |_: &[f32], _: u32| Ok::<Vec<f32>, ()>(Vec::new());
    assert_eq!(
        offline_vad_probabilities(&[0.5], 48_000, 1.0, no_taps, empty),
        fallback
    );
    let non_finite = |_: &[f32], _: u32| Ok::<_, ()>(vec![f32::NAN]);
    assert_eq!(
        offline_vad_probabilities(&[0.5], 48_000, 1.0, no_taps, non_finite),
        fallback
    );
    // A rate past the model's u32 falls back rather than overflowing.
    assert_eq!(
        offline_vad_probabilities(&[0.5], 125 << 33, 1.0, no_taps, unreachable),
        fallback
    );
}
