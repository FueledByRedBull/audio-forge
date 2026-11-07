use super::*;

/// Deterministic standard-normal-ish noise (sum of uniforms) without a crate.
fn noise(seed: u64, length: usize) -> Vec<f64> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..length)
        .map(|_| {
            (0..12)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    (state >> 11) as f64 / (1_u64 << 53) as f64
                })
                .sum::<f64>()
                - 6.0
        })
        .collect()
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

#[test]
fn full_correlation_matches_direct_correlation() {
    for (recording_size, reference_size) in [(1, 1), (7, 4), (13, 16), (4097, 312)] {
        let recording = noise(810, recording_size);
        let reference = noise(811, reference_size);
        let actual = Correlator::new().full(&recording, &reference);
        assert_eq!(actual.len(), recording_size + reference_size - 1);
        let tolerance = 1e-11 * norm(&recording) * norm(&reference);
        for (index, value) in actual.iter().enumerate() {
            let lag = index as i64 - (reference_size as i64 - 1);
            let direct: f64 = reference
                .iter()
                .enumerate()
                .filter_map(|(position, sample)| {
                    let at = lag + position as i64;
                    (0..recording_size as i64)
                        .contains(&at)
                        .then(|| recording[at as usize] * sample)
                })
                .sum();
            assert!((value - direct).abs() <= tolerance, "lag {lag}");
        }
    }
}

#[test]
fn normalized_scores_match_direct_windows() {
    let recording = noise(812, 49);
    let reference = noise(813, 12);
    for (low, high) in [(-20, 50), (3, 9), (60, 70), (7, 2)] {
        let (lags, scores) = Correlator::new().normalized_scores(&recording, &reference, low, high);
        let expected_lags: Vec<i64> = (low.max(0)..=high.min(49 - 12)).collect();
        assert_eq!(lags, expected_lags);
        for (lag, score) in lags.iter().zip(&scores) {
            let window = &recording[*lag as usize..*lag as usize + 12];
            let dot: f64 = window.iter().zip(&reference).map(|(a, b)| a * b).sum();
            let window_energy: f64 = window.iter().map(|x| x * x).sum();
            let reference_energy: f64 = reference.iter().map(|x| x * x).sum::<f64>() + 1e-12;
            let expected = dot.abs() / (window_energy.max(1e-12) * reference_energy).sqrt();
            assert!((score - expected).abs() < 1e-10, "lag {lag}");
        }
    }
}

#[test]
fn numeric_helpers_follow_python_and_numpy() {
    assert_eq!(py_round(2.5), Ok(2));
    assert_eq!(py_round(3.5), Ok(4));
    assert_eq!(py_round(-2.5), Ok(-2));
    assert_eq!(py_int(-7.9), Ok(-7));
    assert!(py_int(f64::NAN).is_err());
    assert!(py_int(f64::INFINITY).is_err());
    assert_eq!(py_max(0.0, f64::NAN), 0.0);
    assert_eq!(argmax(&[1.0, 3.0, 3.0, f64::NAN]), 3);
    assert_eq!(argmax(&[1.0, 3.0, 3.0]), 1);
    assert_eq!(median(&[3.0, 1.0, 2.0, 4.0]), 2.5);
    assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
    assert!(median(&[1.0, f64::NAN]).is_nan());
    assert_eq!(percentile_75(&[0.0, 1.0, 2.0, 3.0, 4.0]), 3.0);
    assert_eq!(percentile_75(&[0.0, 1.0]), 0.75);
    assert_eq!(at(&[1.0, 2.0, 3.0], -1), Ok(3.0));
    assert!(at(&[1.0, 2.0, 3.0], 3).is_err());
    let window = hanning(5);
    assert_eq!(window[0], 0.0);
    assert_eq!(window[2], 1.0);
    assert_eq!(window[1], window[3]);
}

#[test]
fn probe_has_four_bursts_peaking_at_the_amplitude() {
    let probe = generate_probe_signal(48_000, 80.0, 0.8).unwrap();
    assert_eq!(probe.len(), 3840);
    let peak = probe
        .iter()
        .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
    assert!((peak - 0.8).abs() < 1e-6);
    let (burst, offsets) = coded_probe_components(48_000, 3840).unwrap();
    assert_eq!(offsets.len(), 4);
    assert!(burst.len() >= 16);
    assert!(generate_probe_signal(48_000, f64::NAN, 0.8).is_err());
    assert_eq!(generate_probe_signal(48_000, -10.0, 0.8).unwrap().len(), 1);
}

#[test]
fn clean_route_is_measured_without_halving() {
    let probe: Vec<f64> = generate_probe_signal(48_000, 80.0, 0.8)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect();
    let route = 6_864;
    let mut recording = vec![0.0; route + probe.len() + 2048];
    recording[route..route + probe.len()].copy_from_slice(&probe);

    let analysis = analyze_latency(
        Some(&probe),
        Some(&recording),
        48_000,
        SearchWindow::default(),
        DEFAULT_ROUTE_KIND,
    )
    .unwrap();

    assert!(analysis.success, "{}", analysis.message);
    assert!((analysis.peak_sample_offset - route as i64).abs() <= 96);
    assert_eq!(analysis.route_latency_ms, analysis.measured_round_trip_ms);
    assert!((analysis.route_latency_ms - route as f64 / 48.0).abs() < 2.0);
}

#[test]
fn refusals_name_their_reason() {
    let probe: Vec<f64> = generate_probe_signal(48_000, 80.0, 0.8)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect();
    let recording = vec![0.0; 48_000];
    let analyze = |reference: Option<&[f64]>, recorded: Option<&[f64]>, window, kind| {
        analyze_latency(reference, recorded, 48_000, window, kind).unwrap()
    };

    let unsupported = analyze(Some(&probe), Some(&recording), SearchWindow::default(), "x");
    assert_eq!(unsupported.route_kind, "x");
    assert!(unsupported.message.starts_with("Unsupported latency route"));
    let missing = analyze(
        None,
        Some(&recording),
        SearchWindow::default(),
        DEFAULT_ROUTE_KIND,
    );
    assert_eq!(missing.message, "Missing probe or recording.");
    let short = analyze(
        Some(&probe),
        Some(&recording[..100]),
        SearchWindow::default(),
        DEFAULT_ROUTE_KIND,
    );
    assert_eq!(
        short.message,
        "Recording too short for reliable correlation."
    );
    let empty_window = SearchWindow {
        min_search_ms: 300.0,
        max_search_ms: 300.0,
        ..SearchWindow::default()
    };
    let outside = analyze(
        Some(&probe),
        Some(&recording),
        empty_window,
        DEFAULT_ROUTE_KIND,
    );
    assert_eq!(outside.message, "Search window is outside valid lag range.");
    let late_window = SearchWindow {
        min_search_ms: 2000.0,
        max_search_ms: 3000.0,
        ..SearchWindow::default()
    };
    let past = analyze(
        Some(&probe),
        Some(&recording),
        late_window,
        DEFAULT_ROUTE_KIND,
    );
    assert_eq!(
        past.message,
        "Search window does not overlap captured audio."
    );
    assert!(!past.success && past.peak_sample_offset == 0);
}
