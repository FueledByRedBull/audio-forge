use std::f64::consts::PI;

use super::*;

impl Default for SearchWindow {
    fn default() -> Self {
        Self {
            min_search_ms: 5.0,
            max_search_ms: 500.0,
            expected_playback_start_ms: None,
            expected_playback_jitter_ms: None,
            expected_latency_min_ms: None,
            expected_latency_max_ms: None,
        }
    }
}

/// `np.hanning(size)` with the platform `cos`; the bindings pass NumPy's.
fn hanning(size: usize) -> Result<Vec<f64>, LatencyError> {
    if size == 1 {
        return Ok(vec![1.0]);
    }
    let span = size as f64 - 1.0;
    Ok((0..size)
        .map(|index| {
            let position = (1.0 - size as f64) + 2.0 * index as f64;
            0.5 + 0.5 * (PI * position / span).cos()
        })
        .collect())
}

fn probe(sample_rate: f64, duration_ms: f64, amplitude: f64) -> Result<Vec<f32>, LatencyError> {
    generate_probe_signal(sample_rate, duration_ms, amplitude, &mut hanning)
}

fn rate(sample_rate: f64) -> impl FnOnce() -> Result<f64, LatencyError> {
    move || Ok(sample_rate)
}

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
    assert_eq!(py_round(2.5).unwrap(), 2);
    assert_eq!(py_round(3.5).unwrap(), 4);
    assert_eq!(py_round(-2.5).unwrap(), -2);
    assert_eq!(py_int(-7.9).unwrap(), -7);
    // int() raises ValueError for NaN and OverflowError for infinity.
    assert!(matches!(py_int(f64::NAN), Err(LatencyError::Value(_))));
    assert!(matches!(
        py_int(f64::INFINITY),
        Err(LatencyError::Overflow(_))
    ));
    assert!(matches!(
        py_round(f64::NEG_INFINITY),
        Err(LatencyError::Overflow(_))
    ));
    assert_eq!(py_max(0.0, f64::NAN), 0.0);
    assert_eq!(argmax(&[1.0, 3.0, 3.0, f64::NAN]), 3);
    assert_eq!(argmax(&[1.0, 3.0, 3.0]), 1);
    assert_eq!(median(&[3.0, 1.0, 2.0, 4.0]), 2.5);
    assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
    assert!(median(&[1.0, f64::NAN]).is_nan());
    assert_eq!(percentile_75(&[0.0, 1.0, 2.0, 3.0, 4.0]), 3.0);
    assert_eq!(percentile_75(&[0.0, 1.0]), 0.75);
    assert_eq!(at(&[1.0, 2.0, 3.0], -1).unwrap(), 3.0);
    assert!(matches!(
        at(&[1.0, 2.0, 3.0], 3),
        Err(LatencyError::Index(_))
    ));
}

#[test]
fn probe_has_four_bursts_peaking_at_the_amplitude() {
    let signal = probe(48_000.0, 80.0, 0.8).unwrap();
    assert_eq!(signal.len(), 3840);
    let peak = signal
        .iter()
        .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
    assert!((peak - 0.8).abs() < 1e-6);
    let (burst, offsets) = coded_probe_components(48_000.0, 3840, &mut hanning).unwrap();
    assert_eq!(offsets.len(), 4);
    assert!(burst.len() >= 16);
    assert!(matches!(
        probe(48_000.0, f64::NAN, 0.8),
        Err(LatencyError::Value(_))
    ));
    assert!(matches!(
        probe(f64::INFINITY, 80.0, 0.8),
        Err(LatencyError::Overflow(_))
    ));
    assert_eq!(probe(48_000.0, -10.0, 0.8).unwrap().len(), 1);
}

#[test]
fn probe_window_comes_from_the_caller() {
    let mut sizes = Vec::new();
    let mut flat = |size: usize| {
        sizes.push(size);
        Ok(vec![1.0; size])
    };
    let flat_probe = generate_probe_signal(48_000.0, 80.0, 0.8, &mut flat).unwrap();
    assert_eq!(sizes, [13 * 24]);
    assert_ne!(flat_probe, probe(48_000.0, 80.0, 0.8).unwrap());

    let mut short = |size: usize| Ok(vec![1.0; size - 1]);
    assert!(matches!(
        generate_probe_signal(48_000.0, 80.0, 0.8, &mut short),
        Err(LatencyError::Value(_))
    ));
    let mut failing = |_size: usize| Err(LatencyError::Caller("no window".into()));
    let error = generate_probe_signal(48_000.0, 80.0, 0.8, &mut failing).unwrap_err();
    assert_eq!(error.to_string(), "no window");
}

#[test]
fn clean_route_is_measured_without_halving() {
    let probe: Vec<f64> = probe(48_000.0, 80.0, 0.8)
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
        rate(48_000.0),
        SearchWindow::default(),
        DEFAULT_ROUTE_KIND,
        &mut hanning,
    )
    .unwrap();

    assert!(analysis.success, "{}", analysis.message);
    assert!((analysis.peak_sample_offset - route as i64).abs() <= 96);
    assert_eq!(analysis.route_latency_ms, analysis.measured_round_trip_ms);
    assert!((analysis.route_latency_ms - route as f64 / 48.0).abs() < 2.0);
}

#[test]
fn refusals_name_their_reason() {
    let probe: Vec<f64> = probe(48_000.0, 80.0, 0.8)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect();
    let recording = vec![0.0; 48_000];
    // The rate is never used for these refusals, as in the removed Python.
    let analyze = |reference: Option<&[f64]>, recorded: Option<&[f64]>, window, kind| {
        let unusable_rate = || Err(LatencyError::Caller("rate used".into()));
        analyze_latency(
            reference,
            recorded,
            unusable_rate,
            window,
            kind,
            &mut hanning,
        )
        .unwrap()
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
    let analyze = |window| {
        analyze_latency(
            Some(&probe),
            Some(&recording),
            rate(48_000.0),
            window,
            DEFAULT_ROUTE_KIND,
            &mut hanning,
        )
        .unwrap()
    };
    let outside = analyze(empty_window);
    assert_eq!(outside.message, "Search window is outside valid lag range.");
    let late_window = SearchWindow {
        min_search_ms: 2000.0,
        max_search_ms: 3000.0,
        ..SearchWindow::default()
    };
    let past = analyze(late_window);
    assert_eq!(
        past.message,
        "Search window does not overlap captured audio."
    );
    assert!(!past.success && past.peak_sample_offset == 0);
}

#[test]
fn a_rate_conversion_failure_surfaces_where_the_rate_is_first_used() {
    let probe: Vec<f64> = probe(48_000.0, 80.0, 0.8)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect();
    let recording = vec![0.0; 48_000];
    let error = analyze_latency(
        Some(&probe),
        Some(&recording),
        || Err(LatencyError::Caller("not a number".into())),
        SearchWindow::default(),
        DEFAULT_ROUTE_KIND,
        &mut hanning,
    )
    .unwrap_err();
    assert_eq!(error.to_string(), "not a number");
}
