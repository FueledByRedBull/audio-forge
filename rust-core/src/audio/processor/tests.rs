#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;
    use pyo3::Python;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn test_duration_samples_for_44k1_output() {
        assert_eq!(duration_samples(44_100, OUTPUT_PRIME_MS), 882);
        assert_eq!(duration_samples(44_100, OUTPUT_TARGET_HIGH_MS), 1323);
        assert_eq!(duration_samples(44_100, OUTPUT_HARD_BACKLOG_MS), 2646);
    }

    #[test]
    fn test_samples_to_micros_uses_device_rate() {
        assert_eq!(samples_to_micros(441, 44_100), 10_000);
        assert_eq!(samples_to_micros(480, 48_000), 10_000);
    }

    #[test]
    fn test_total_reported_latency_includes_limiter_lookahead_across_sample_rates() {
        for (sample_rate, expected_us) in
            [(44_100_u32, 1_995_u64), (48_000, 2_000), (96_000, 2_000)]
        {
            let limiter = Limiter::default_settings(sample_rate as f64);
            let lookahead_samples = limiter.lookahead_samples() as u64;
            let total = total_reported_latency_us(
                0,
                sample_rate,
                0,
                lookahead_samples,
                true,
                sample_rate,
                0,
            );
            assert_eq!(total, expected_us);
        }
    }

    #[test]
    fn test_total_reported_latency_respects_output_vs_processing_rates() {
        let total = total_reported_latency_us(
            882,    // 20ms @ 44.1kHz output buffer
            44_100, // output sample rate
            480,    // 10ms suppressor latency @ 48kHz
            96,     // 2ms limiter lookahead @ 48kHz
            true, 48_000, // processing sample rate
            500,    // fixed compensation
        );
        assert_eq!(total, 20_000 + 10_000 + 2_000 + 500);
    }

    #[test]
    fn test_build_sinc_resampler_for_valid_rates() {
        let resampler = build_sinc_resampler(44_100, 48_000, 1024);
        assert!(resampler.is_ok());
    }

    #[test]
    fn test_active_device_names_only_report_when_running() {
        let mut processor = AudioProcessor::new();
        processor.input_device_name = Some("Mic A".to_string());
        processor.output_device_name = Some("Out B".to_string());

        assert_eq!(processor.active_input_device_name(), None);
        assert_eq!(processor.active_output_device_name(), None);

        processor.running.store(true, Ordering::SeqCst);
        assert_eq!(
            processor.active_input_device_name().as_deref(),
            Some("Mic A")
        );
        assert_eq!(
            processor.active_output_device_name().as_deref(),
            Some("Out B")
        );
    }

    #[test]
    fn test_raw_monitor_toggle_round_trip() {
        let processor = AudioProcessor::new();
        assert!(!processor.is_raw_monitor_enabled());
        processor.set_raw_monitor_enabled(true);
        assert!(processor.is_raw_monitor_enabled());
        processor.set_raw_monitor_enabled(false);
        assert!(!processor.is_raw_monitor_enabled());
    }

    fn install_raw_recording_consumer(processor: &AudioProcessor) {
        let rb = crate::audio::AudioRingBuffer::new(processor.sample_rate as usize);
        let (_producer, consumer) = rb.split();
        *processor.raw_recording_consumer.lock().unwrap() = Some(consumer);
    }

    #[test]
    fn test_raw_recording_rejects_invalid_durations_without_activating() {
        for duration in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut processor = AudioProcessor::new();
            install_raw_recording_consumer(&processor);

            let err = processor.start_raw_recording(duration).unwrap_err();

            assert!(err.contains("finite positive"));
            assert!(!processor.recording_active.load(Ordering::Acquire));
            assert_eq!(processor.raw_recording_target.load(Ordering::Acquire), 0);
        }
    }

    #[test]
    fn test_raw_recording_rejects_too_short_duration_without_activating() {
        let mut processor = AudioProcessor::new();
        install_raw_recording_consumer(&processor);

        let err = processor.start_raw_recording(0.25 / processor.sample_rate as f64);

        assert_eq!(err.unwrap_err(), "Recording duration is too short");
        assert!(!processor.recording_active.load(Ordering::Acquire));
        assert_eq!(processor.raw_recording_target.load(Ordering::Acquire), 0);
    }

    #[test]
    fn test_raw_recording_accepts_valid_duration_with_nonzero_target() {
        let mut processor = AudioProcessor::new();
        install_raw_recording_consumer(&processor);

        processor.start_raw_recording(0.01).unwrap();

        assert!(processor.recording_active.load(Ordering::Acquire));
        assert!(processor.raw_recording_target.load(Ordering::Acquire) > 0);
    }

    #[test]
    fn test_pending_suppressor_swap_keeps_current_when_model_already_matches() {
        let strength = Arc::new(AtomicU32::new(1.0_f32.to_bits()));
        let mut current = NoiseSuppressionEngine::new(NoiseModel::RNNoise, Arc::clone(&strength));
        let mut pending = Some(NoiseSuppressionEngine::new(NoiseModel::RNNoise, strength));
        let control = SuppressorControlState {
            enabled: true,
            model: NoiseModel::RNNoise,
        };

        assert!(swap_pending_suppressor_if_ready(
            &mut current,
            &control,
            &mut pending
        ));
        assert_eq!(current.model_type(), NoiseModel::RNNoise);
        assert!(pending.is_some());
    }

    #[cfg(feature = "deepfilter")]
    #[test]
    fn test_pending_suppressor_swap_without_candidate_leaves_current_backend() {
        let strength = Arc::new(AtomicU32::new(1.0_f32.to_bits()));
        let mut current = NoiseSuppressionEngine::new(NoiseModel::RNNoise, strength);
        let mut pending = None;
        let control = SuppressorControlState {
            enabled: true,
            model: NoiseModel::DeepFilterNetLL,
        };

        assert!(!swap_pending_suppressor_if_ready(
            &mut current,
            &control,
            &mut pending
        ));
        assert_eq!(current.model_type(), NoiseModel::RNNoise);
    }

    #[test]
    fn test_raw_monitor_path_selection_and_clean_write_mode() {
        let raw = select_processing_path(true, false);
        let bypass = select_processing_path(false, true);
        let full = select_processing_path(false, false);

        assert_eq!(raw, ProcessingPath::RawMonitor);
        assert_eq!(bypass, ProcessingPath::Bypass);
        assert_eq!(full, ProcessingPath::Full);

        assert!(uses_clean_write_path(raw));
        assert!(!uses_clean_write_path(bypass));
        assert!(!uses_clean_write_path(full));
    }

    #[test]
    fn test_raw_monitor_sanitizes_but_skips_prefilter_shaping() {
        let mut non_finite = vec![f32::NAN, f32::INFINITY, -f32::INFINITY];
        sanitize_non_finite_inplace(&mut non_finite);
        assert_eq!(non_finite, vec![0.0, 0.0, 0.0]);

        let mut raw_buffer = vec![0.25_f32; 256];
        let mut normal_buffer = raw_buffer.clone();

        sanitize_non_finite_inplace(&mut raw_buffer);
        let mut pre_filter_state = InputPreFilterState::default();
        let mut pre_filter = Biquad::new(
            BiquadType::HighPass,
            INPUT_PREFILTER_HZ,
            0.0,
            INPUT_PREFILTER_Q,
            48_000.0,
        );
        apply_input_pre_filter(&mut normal_buffer, &mut pre_filter_state, &mut pre_filter);

        let max_abs_diff = normal_buffer
            .iter()
            .zip(raw_buffer.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_abs_diff > 0.05);
    }

    #[test]
    fn test_final_output_sanitizer_rejects_non_finite_and_limits_ceiling() {
        let mut buffer = vec![f32::NAN, f32::INFINITY, -f32::INFINITY, 0.75, -0.75];
        sanitize_and_clamp_output_inplace(&mut buffer, 0.5);
        assert_eq!(buffer, vec![0.0, 0.0, 0.0, 0.5, -0.5]);
    }

    #[test]
    fn test_retime_audio_block_can_expand_and_compress() {
        let input = [0.0_f32, 0.25, 0.5, 0.75, 1.0, 0.5];
        let mut scratch = Vec::new();

        let expanded = retime_audio_block(&input, 0.5, 32, &mut scratch);
        assert!(expanded.len() > input.len());

        let compressed = retime_audio_block(&input, 2.0, 32, &mut scratch);
        assert!(compressed.len() < input.len());
    }

    #[test]
    fn test_retime_audio_block_linear_interpolation_does_not_overshoot_neighbors() {
        let input = [0.0_f32, 0.5, 1.0, 0.5, 0.0];
        let mut scratch = Vec::new();

        let expanded = retime_audio_block(&input, 0.7, 32, &mut scratch);

        for sample in expanded {
            assert!((*sample >= 0.0) && (*sample <= 1.0));
        }
    }

    #[test]
    fn test_eq_single_band_validation_rejects_invalid_values_and_preserves_state() {
        let processor = AudioProcessor::new();
        let original = processor.get_eq_band_params(0).unwrap();

        assert!(processor.set_eq_band_gain(0, f64::NAN).is_err());
        assert!(processor.set_eq_band_frequency(0, 50_000.0).is_err());
        assert!(processor.set_eq_band_q(0, 0.01).is_err());
        assert_eq!(processor.get_eq_band_params(0).unwrap(), original);
    }

    #[test]
    fn test_gate_control_validation_rejects_non_finite_and_clamps_ranges() {
        let processor = AudioProcessor::new();
        processor.set_gate_threshold(f64::NAN);
        processor.set_gate_attack(f64::INFINITY);
        processor.set_gate_release(f64::NEG_INFINITY);

        {
            let control = processor.gate_control.lock().unwrap();
            assert_eq!(control.threshold_db, -40.0);
            assert_eq!(control.attack_ms, 10.0);
            assert_eq!(control.release_ms, 100.0);
        }

        processor.set_gate_threshold(-120.0);
        processor.set_gate_attack(0.01);
        processor.set_gate_release(5000.0);

        let control = processor.gate_control.lock().unwrap();
        assert_eq!(control.threshold_db, GATE_THRESHOLD_MIN_DB);
        assert_eq!(control.attack_ms, GATE_ATTACK_MIN_MS);
        assert_eq!(control.release_ms, GATE_RELEASE_MAX_MS);
    }

    #[test]
    fn test_rnnoise_strength_validation_rejects_non_finite_and_clamps_ranges() {
        let processor = AudioProcessor::new();
        let original = processor.get_rnnoise_strength();

        processor.set_rnnoise_strength(f32::NAN);
        assert_eq!(processor.get_rnnoise_strength(), original);

        processor.set_rnnoise_strength(2.0);
        assert_eq!(processor.get_rnnoise_strength(), RNNOISE_STRENGTH_MAX);

        processor.set_rnnoise_strength(-1.0);
        assert_eq!(processor.get_rnnoise_strength(), RNNOISE_STRENGTH_MIN);
    }

    #[test]
    fn test_deesser_control_validation_rejects_non_finite_and_clamps_ranges() {
        let processor = AudioProcessor::new();

        processor.set_deesser_low_cut_hz(f64::NAN);
        processor.set_deesser_high_cut_hz(f64::INFINITY);
        processor.set_deesser_threshold_db(f64::NEG_INFINITY);
        processor.set_deesser_ratio(f64::NAN);
        processor.set_deesser_attack_ms(f64::NAN);
        processor.set_deesser_release_ms(f64::NAN);
        processor.set_deesser_max_reduction_db(f64::NAN);
        processor.set_deesser_auto_amount(f64::NAN);

        assert_eq!(processor.get_deesser_low_cut_hz(), 4000.0);
        assert_eq!(processor.get_deesser_high_cut_hz(), 9000.0);
        assert_eq!(processor.get_deesser_threshold_db(), -28.0);
        assert_eq!(processor.get_deesser_ratio(), 4.0);
        assert_eq!(processor.get_deesser_max_reduction_db(), 6.0);
        assert_eq!(processor.get_deesser_auto_amount(), 0.5);

        processor.set_deesser_low_cut_hz(100.0);
        processor.set_deesser_high_cut_hz(80_000.0);
        processor.set_deesser_threshold_db(-100.0);
        processor.set_deesser_ratio(100.0);
        processor.set_deesser_attack_ms(0.01);
        processor.set_deesser_release_ms(10_000.0);
        processor.set_deesser_max_reduction_db(100.0);
        processor.set_deesser_auto_amount(2.0);

        assert_eq!(processor.get_deesser_low_cut_hz(), DEESSER_LOW_CUT_MIN_HZ);
        assert_eq!(processor.get_deesser_high_cut_hz(), DEESSER_HIGH_CUT_MAX_HZ);
        assert_eq!(
            processor.get_deesser_threshold_db(),
            DEESSER_THRESHOLD_MIN_DB
        );
        assert_eq!(processor.get_deesser_ratio(), DEESSER_RATIO_MAX);
        assert_eq!(
            processor.get_deesser_max_reduction_db(),
            DEESSER_MAX_REDUCTION_MAX_DB
        );
        assert_eq!(processor.get_deesser_auto_amount(), DEESSER_AUTO_AMOUNT_MAX);
    }

    #[test]
    fn test_apply_eq_settings_rejects_above_nyquist() {
        let processor = AudioProcessor::new();
        let mut bands = vec![(100.0, 0.0, 1.0); NUM_BANDS];
        bands[NUM_BANDS - 1] = (processor.eq_nyquist_limit_hz() + 100.0, 0.0, 1.0);

        pyo3::prepare_freethreaded_python();
        let err = processor.apply_eq_settings(bands).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn test_lock_rt_counts_contention() {
        let mutex = Arc::new(Mutex::new(1_u32));
        let contention = Arc::new(AtomicU64::new(0));
        let (started_tx, started_rx) = mpsc::channel();
        let mutex_for_thread = Arc::clone(&mutex);

        let holder = std::thread::spawn(move || {
            let _guard = mutex_for_thread.lock().unwrap();
            started_tx.send(()).unwrap();
            std::thread::sleep(Duration::from_millis(30));
        });

        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let guard = lock_rt(mutex.as_ref(), contention.as_ref());
        assert!(guard.is_none());
        assert_eq!(contention.load(Ordering::Relaxed), 1);
        holder.join().unwrap();
    }

    #[test]
    fn test_release_ms_to_tenth_ms_rounds_expected_values() {
        assert_eq!(release_ms_to_tenth_ms(200.0), 2000);
        assert_eq!(release_ms_to_tenth_ms(12.34), 123);
        assert_eq!(release_ms_to_tenth_ms(12.35), 124);
        assert_eq!(release_ms_to_tenth_ms(-5.0), 0);
        assert_eq!(release_ms_to_tenth_ms(f64::NAN), 0);
    }

    #[test]
    fn test_smoothing_coeff_for_time_constant_matches_100ms_meter_target() {
        let coeff = smoothing_coeff_for_time_constant(48_000.0, 100.0);
        assert!((coeff - 0.999_791_7).abs() < 1e-6);

        let invalid = smoothing_coeff_for_time_constant(0.0, 100.0);
        assert_eq!(invalid, 0.0);
    }

    #[test]
    fn test_runtime_diagnostics_include_output_recovery_count() {
        let wrapper = PyAudioProcessor {
            processor: AudioProcessor::new(),
        };
        wrapper
            .processor
            .output_recovery_count
            .store(7, Ordering::Relaxed);

        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let diagnostics = wrapper.get_runtime_diagnostics(py).unwrap();
            let diagnostics = diagnostics.bind(py);
            let diagnostics = diagnostics.downcast::<PyDict>().unwrap();
            assert_eq!(
                diagnostics
                    .get_item("output_recovery_count")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                7
            );
        });
    }
}
