use super::*;

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
fn kaiser_bessel_series_matches_i0() {
    assert!((bessel_i0_up_to_8(0.0) - 1.0).abs() < 1e-15);
    assert!((bessel_i0_up_to_8(5.0) / 27.239_871_823_604_442 - 1.0).abs() < 1e-14);
}

#[test]
fn resample_poly_output_length_follows_the_rate_ratio() {
    for (up, down) in [(1, 3), (3, 2), (160, 147), (147, 160)] {
        for length in [1_usize, 2, 5, 31, 1024, 4410] {
            let samples = vec![0.25_f64; length];
            let output = resample_poly(&samples, up, down).unwrap();
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
    assert_eq!(resample_poly(&samples, 2, 2).unwrap(), samples);
    assert!(resample_poly::<f64>(&[], 1, 3).unwrap().is_empty());
    assert!(resample_poly(&samples, 0, 1).is_err());
    assert!(resample_poly(&samples, 1, 0).is_err());
}

#[test]
fn resample_poly_preserves_dc_and_rejects_above_nyquist_energy() {
    let dc = resample_poly(&vec![1.0_f64; 48_000], 1, 3).unwrap();
    assert!(dc[100..dc.len() - 100]
        .iter()
        .all(|sample| (sample - 1.0).abs() < 1e-12));

    let tone: Vec<f64> = (0..48_000)
        .map(|index| (2.0 * PI * 0.42 * index as f64).sin())
        .collect();
    let filtered = resample_poly(&tone, 1, 4).unwrap();
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
    let wide = resample_poly(&samples, 160, 147).unwrap();
    let single = resample_poly(&narrow, 160, 147).unwrap();
    assert_eq!(wide.len(), single.len());
    for (a, b) in wide.iter().zip(&single) {
        assert!((a - f64::from(*b)).abs() < 2e-6);
    }
}

#[test]
fn peaks_match_scipy_distance_and_prominence() {
    let first = [0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0];
    let second = [0.0, 3.0, 2.0, 3.0, 0.0];
    let peaks = |values: &[f64], distance, prominence| {
        find_peaks_distance_prominence(values, distance, prominence).unwrap()
    };
    // Expected values are scipy.signal.find_peaks' output.
    assert_eq!(peaks(&first, 2, 0.75), vec![1, 4, 7, 9]);
    assert_eq!(peaks(&first, 3, 0.75), vec![1, 4, 9]);
    assert_eq!(peaks(&first, 8, 0.75), vec![1, 9]);
    assert_eq!(peaks(&first, 1, 1.5), vec![4, 9]);
    assert_eq!(peaks(&second, 2, 0.75), vec![1, 3]);
    // Equal heights: the later peak ranks first, as SciPy keeps it.
    assert_eq!(peaks(&second, 3, 0.75), vec![3]);
}

#[test]
fn peaks_ignore_nan_neighbours_and_short_input() {
    let peaks = |values: &[f64]| find_peaks_distance_prominence(values, 1, 0.0).unwrap();
    assert!(peaks(&[]).is_empty());
    assert!(peaks(&[1.0, 2.0]).is_empty());
    assert!(peaks(&[0.0, 1.0, f64::NAN, 0.0, 2.0, 0.0]) == vec![4]);
    assert!(find_peaks_distance_prominence(&[0.0], 0, 0.0).is_err());
    assert!(find_peaks_distance_prominence(&[0.0], 1, -1.0).is_err());
}

#[test]
fn causal_vad_mapping_keeps_exact_44k1_window_timing() {
    let probabilities: Vec<f64> = (0..700)
        .map(|index| f64::from(index as f32 / 700.0))
        .collect();
    let frame_ends: Vec<i64> = (1..2_001).map(|index| index * 441).collect();
    let mapped = map_causal_vad_probabilities(&probabilities, &frame_ends, 44_100).unwrap();

    assert_eq!(&mapped[..3], &[0.0; 3]);
    // 1.6 s contains exactly 50 model windows; 20 s contains 625.
    assert_eq!(mapped[159], probabilities[49] as f32);
    assert_eq!(mapped[1999], probabilities[624] as f32);
}

#[test]
fn causal_vad_mapping_clips_and_rejects_unusable_input() {
    let mapped = map_causal_vad_probabilities(&[1.5, -0.5], &[-512, 512, 1024, 99_999], 16_000);
    assert_eq!(mapped, Some(vec![0.0, 1.0, 0.0, 0.0]));
    assert_eq!(map_causal_vad_probabilities(&[0.5], &[512], 0), None);
    assert_eq!(map_causal_vad_probabilities(&[], &[512], 16_000), None);
    assert_eq!(map_causal_vad_probabilities(&[0.5], &[], 16_000), None);
    assert_eq!(
        map_causal_vad_probabilities(&[f64::NAN], &[512], 16_000),
        None
    );
}

#[test]
fn offline_vad_applies_pre_gain_before_inference() {
    let mut observed = None;
    let probabilities = offline_vad_probabilities(&[0.5, -0.25], 48_000, 2.0, |samples, rate| {
        observed = Some((samples.to_vec(), rate));
        Ok::<_, ()>(vec![0.7])
    });

    assert_eq!(observed, Some((vec![1.0, -0.5], 48_000)));
    assert_eq!(probabilities, Some(vec![f64::from(0.7_f32)]));
}

#[test]
fn offline_vad_resamples_rates_without_whole_model_windows() {
    let audio = vec![0.1_f32; 4_410];
    let mut observed = None;
    let probabilities = offline_vad_probabilities(&audio, 44_100, 1.0, |samples, rate| {
        observed = Some((samples.len(), rate));
        Ok::<_, ()>(vec![1.2, -0.1])
    });

    assert_eq!(observed, Some((4_800, 48_000)));
    assert_eq!(probabilities, Some(vec![1.0, 0.0]));
}

#[test]
fn offline_vad_falls_back_on_unusable_gain_capture_or_result() {
    let unreachable = |_: &[f32], _: u32| -> Result<Vec<f32>, ()> {
        panic!("inference must not run");
    };
    for gain in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            offline_vad_probabilities(&[0.5], 48_000, gain, unreachable),
            None
        );
    }
    assert_eq!(
        offline_vad_probabilities(&[], 48_000, 1.0, unreachable),
        None
    );
    assert_eq!(offline_vad_probabilities(&[0.5], 0, 1.0, unreachable), None);

    let mut floored = None;
    offline_vad_probabilities(&[1.0], 48_000, 0.01, |samples, _| {
        floored = Some(samples[0]);
        Ok::<_, ()>(vec![0.5])
    });
    assert_eq!(floored, Some(0.1_f32));

    let failing = |_: &[f32], _: u32| Err::<Vec<f32>, _>("model missing");
    assert_eq!(
        offline_vad_probabilities(&[0.5], 48_000, 1.0, failing),
        None
    );
    let empty = |_: &[f32], _: u32| Ok::<Vec<f32>, ()>(Vec::new());
    assert_eq!(offline_vad_probabilities(&[0.5], 48_000, 1.0, empty), None);
    let non_finite = |_: &[f32], _: u32| Ok::<_, ()>(vec![f32::NAN]);
    assert_eq!(
        offline_vad_probabilities(&[0.5], 48_000, 1.0, non_finite),
        None
    );
}
