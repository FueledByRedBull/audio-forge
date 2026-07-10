#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;
    use pyo3::Python;

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
    fn test_next_process_idle_sleep_stays_fast_when_input_is_recent() {
        assert_eq!(next_process_idle_sleep_us(0, 500), PROCESS_IDLE_SLEEP_US);
        assert_eq!(next_process_idle_sleep_us(8, 1_500), PROCESS_IDLE_SLEEP_US);
    }

    #[test]
    fn test_next_process_idle_sleep_uses_bounded_backoff_when_idle() {
        assert_eq!(next_process_idle_sleep_us(0, u64::MAX), 100);
        assert_eq!(next_process_idle_sleep_us(1, u64::MAX), 200);
        assert_eq!(next_process_idle_sleep_us(2, u64::MAX), 400);
        assert_eq!(next_process_idle_sleep_us(3, u64::MAX), 800);
        assert_eq!(next_process_idle_sleep_us(10, u64::MAX), PROCESS_IDLE_MAX_SLEEP_US);
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

    #[test]
    fn test_same_noise_model_selection_is_noop_while_running() {
        let processor = AudioProcessor::new();
        processor.running.store(true, Ordering::SeqCst);

        assert!(processor.set_noise_model(NoiseModel::RNNoise));
        assert_eq!(processor.get_noise_model(), NoiseModel::RNNoise);
        assert!(processor
            .pending_suppressor_tx
            .lock()
            .map(|tx| tx.is_none())
            .unwrap_or(false));
    }

    #[test]
    fn test_start_publishes_processing_thread_before_starting_streams() {
        let source = include_str!("dsp_loop.rs");
        let start_fn = source_between(source, "    pub fn start(", "    /// Stop audio processing");
        let process_thread_published = start_fn
            .find("self.process_thread = Some(handle);")
            .expect("start must publish the processing thread handle");
        let before_thread_publish = &start_fn[..process_thread_published];

        assert!(
            !before_thread_publish.contains(".start()"),
            "audio streams must not be started before the DSP thread handle is published"
        );

        let ready_wait = start_fn[process_thread_published..]
            .find("dsp_ready_rx.recv_timeout")
            .map(|offset| process_thread_published + offset)
            .expect("start must wait for the DSP thread to finish initialization");
        let output_start = start_fn[process_thread_published..]
            .find("output.start()")
            .map(|offset| process_thread_published + offset)
            .expect("start must start the output stream after publishing the DSP thread");
        let input_start = start_fn[process_thread_published..]
            .find("input.start()")
            .map(|offset| process_thread_published + offset)
            .expect("start must start the input stream after publishing the DSP thread");

        assert!(
            ready_wait < output_start,
            "audio streams must not start before DSP pre-loop initialization is complete"
        );
        assert!(
            output_start < input_start,
            "input stream must start last to avoid accumulating startup backlog"
        );
    }

    #[cfg(feature = "deepfilter")]
    #[test]
    fn test_failed_running_noise_model_queue_preserves_backend_diagnostics() {
        let processor = AudioProcessor::new();
        processor.running.store(true, Ordering::SeqCst);
        processor
            .current_model
            .store(NoiseModel::DeepFilterNetLL as u8, Ordering::Release);
        processor
            .noise_backend_available
            .store(false, Ordering::Relaxed);
        processor.noise_backend_failed.store(true, Ordering::Relaxed);
        *processor.noise_backend_error.lock().unwrap() = Some("previous backend".to_string());

        let queue = RtCommandQueue::<NoiseSuppressionEngine, 1>::new();
        let (mut tx, _rx) = queue.split();
        let queued_engine = NoiseSuppressionEngine::new(
            NoiseModel::RNNoise,
            Arc::clone(&processor.suppressor_strength),
        );
        assert!(tx.push(queued_engine).is_ok());
        *processor.pending_suppressor_tx.lock().unwrap() = Some(tx);

        assert!(!processor.set_noise_model(NoiseModel::RNNoise));
        assert_eq!(processor.get_noise_model(), NoiseModel::DeepFilterNetLL);
        assert!(!processor.is_noise_backend_available());
        assert!(processor.noise_backend_failed());
        assert_eq!(
            processor.noise_backend_error().as_deref(),
            Some("previous backend")
        );
    }

    #[test]
    fn test_control_snapshot_reports_unstable_odd_sequence() {
        let control = AtomicSuppressorControlState::new();
        control.seq.store(1, Ordering::Release);

        let snapshot = control.snapshot();

        assert!(snapshot.is_none());
    }

    #[test]
    fn test_control_snapshot_reads_stable_sequence() {
        let control = AtomicSuppressorControlState::new();

        let snapshot = control.snapshot().expect("stable snapshot");

        assert_eq!(snapshot.model, NoiseModel::RNNoise);
        assert!(snapshot.enabled);
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
        apply_input_pre_filter(
            &mut normal_buffer,
            &mut pre_filter_state,
            &mut pre_filter,
            true,
        );

        let max_abs_diff = normal_buffer
            .iter()
            .zip(raw_buffer.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_abs_diff > 0.05);
    }

    fn process_fixed_input_prefilter(input: &[f32], sample_rate: f32) -> Vec<f32> {
        let mut output = input.to_vec();
        let mut pre_filter_state = InputPreFilterState::default();
        let mut pre_filter = Biquad::new(
            BiquadType::HighPass,
            INPUT_PREFILTER_HZ,
            0.0,
            INPUT_PREFILTER_Q,
            sample_rate as f64,
        );
        apply_input_pre_filter(&mut output, &mut pre_filter_state, &mut pre_filter, true);
        output
    }

    fn process_adaptive_input_cleanup(
        input: &[f32],
        mode: InputCleanupMode,
        sample_rate: f32,
        chunk_size: usize,
    ) -> (Vec<f32>, bool, bool, f32, f32) {
        let mut pre_filter_state = InputPreFilterState::default();
        let mut pre_filter = Biquad::new(
            BiquadType::HighPass,
            INPUT_PREFILTER_HZ,
            0.0,
            INPUT_PREFILTER_Q,
            sample_rate as f64,
        );
        let mut cleanup = AdaptiveInputCleanupState::new(sample_rate);
        cleanup.set_mode(mode);

        let mut output = Vec::with_capacity(input.len());
        let mut ever_hum = false;
        let mut ever_rumble = false;
        let mut max_high_pass_hz = INPUT_PREFILTER_HZ as f32;

        for chunk in input.chunks(chunk_size) {
            let mut block = chunk.to_vec();
            if mode.is_enabled() {
                cleanup.analyze_input(&block);
            }
            apply_input_pre_filter(
                &mut block,
                &mut pre_filter_state,
                &mut pre_filter,
                !mode.is_enabled(),
            );
            if mode.is_enabled() {
                cleanup.process_block(&mut block);
            }
            ever_hum |= cleanup.hum_detected;
            ever_rumble |= cleanup.rumble_detected;
            max_high_pass_hz = max_high_pass_hz.max(cleanup.selected_high_pass_hz);
            output.extend_from_slice(&block);
        }

        (
            output,
            ever_hum,
            ever_rumble,
            max_high_pass_hz,
            cleanup.hum_line_hz,
        )
    }

    #[test]
    fn test_input_cleanup_off_matches_existing_prefilter_bit_exactly() {
        let sample_rate = 48_000.0;
        let len = 4096;
        let input: Vec<f32> = (0..len)
            .map(|index| {
                let t = index as f32 / sample_rate;
                0.15 * (2.0 * std::f32::consts::PI * 60.0 * t).sin()
                    + 0.08 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.04 * (2.0 * std::f32::consts::PI * 1200.0 * t).sin()
            })
            .collect();

        let fixed = process_fixed_input_prefilter(&input, sample_rate);
        let (adaptive_off, hum, rumble, high_pass_hz, _) =
            process_adaptive_input_cleanup(&input, InputCleanupMode::Off, sample_rate, 480);

        assert_eq!(adaptive_off, fixed);
        assert!(!hum);
        assert!(!rumble);
        assert_eq!(high_pass_hz, INPUT_PREFILTER_HZ as f32);
    }

    #[test]
    fn test_adaptive_input_cleanup_reduces_synthetic_line_hum() {
        let sample_rate = 48_000.0;
        let len = sample_rate as usize;
        let input: Vec<f32> = (0..len)
            .map(|index| {
                let t = index as f32 / sample_rate;
                0.14 * (2.0 * std::f32::consts::PI * 60.0 * t).sin()
                    + 0.08 * (2.0 * std::f32::consts::PI * 120.0 * t).sin()
                    + 0.05 * (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
            })
            .collect();

        let fixed = process_fixed_input_prefilter(&input, sample_rate);
        let (cleaned, hum, _rumble, high_pass_hz, _) =
            process_adaptive_input_cleanup(&input, InputCleanupMode::Strong, sample_rate, 480);
        let tail = len / 2;
        let fixed_hum = tone_amplitude(&fixed[tail..], sample_rate, 60.0);
        let cleaned_hum = tone_amplitude(&cleaned[tail..], sample_rate, 60.0);
        let fixed_voice = tone_amplitude(&fixed[tail..], sample_rate, 1000.0);
        let cleaned_voice = tone_amplitude(&cleaned[tail..], sample_rate, 1000.0);

        assert!(hum);
        assert!(cleaned_hum < fixed_hum * 0.65);
        assert!(cleaned_voice > fixed_voice * 0.94);
        assert_eq!(high_pass_hz, INPUT_PREFILTER_HZ as f32);
    }

    #[test]
    fn test_adaptive_input_cleanup_raises_highpass_for_plosive_not_sustained_voice() {
        let sample_rate = 48_000.0;
        let len = sample_rate as usize;
        let input: Vec<f32> = (0..len)
            .map(|index| {
                let t = index as f32 / sample_rate;
                let voice = 0.08 * (2.0 * std::f32::consts::PI * 180.0 * t).sin()
                    + 0.05 * (2.0 * std::f32::consts::PI * 1200.0 * t).sin();
                let plosive = if t < 0.05 {
                    let env = (1.0 - t / 0.05).max(0.0);
                    0.65 * env * (2.0 * std::f32::consts::PI * 38.0 * t).sin()
                } else {
                    0.0
                };
                voice + plosive
            })
            .collect();

        let fixed = process_fixed_input_prefilter(&input, sample_rate);
        let (cleaned, _hum, rumble, high_pass_hz, _) =
            process_adaptive_input_cleanup(&input, InputCleanupMode::Gentle, sample_rate, 480);
        let tail = len * 3 / 4;
        let fixed_voice = tone_amplitude(&fixed[tail..], sample_rate, 180.0);
        let cleaned_voice = tone_amplitude(&cleaned[tail..], sample_rate, 180.0);

        assert!(rumble);
        assert!(high_pass_hz >= 100.0);
        assert!(cleaned_voice > fixed_voice * 0.94);
    }

    #[test]
    fn test_adaptive_cleanup_tracks_49_to_61_hz_drift_and_retunes_smoothly() {
        let sample_rate = 48_000.0_f32;
        let len = sample_rate as usize * 2;
        let mut phase = 0.0_f32;
        let mut input = Vec::with_capacity(len);
        let mut voice_only = Vec::with_capacity(len);
        for index in 0..len {
            let time = index as f32 / sample_rate;
            let frequency = 49.0 + 12.0 * index as f32 / (len - 1) as f32;
            phase += 2.0 * std::f32::consts::PI * frequency / sample_rate;
            let voice = 0.045 * (2.0 * std::f32::consts::PI * 1000.0 * time).sin();
            voice_only.push(voice);
            input.push(voice + 0.13 * phase.sin() + 0.065 * (2.0 * phase).sin());
        }

        let (cleaned, hum, _, _, tracked_hz) =
            process_adaptive_input_cleanup(&input, InputCleanupMode::Strong, sample_rate, 480);
        let (clean_voice, _, _, _, _) = process_adaptive_input_cleanup(
            &voice_only,
            InputCleanupMode::Strong,
            sample_rate,
            480,
        );
        let tail = len / 2;
        let input_residual = input[tail..]
            .iter()
            .zip(voice_only[tail..].iter())
            .map(|(mixed, voice)| (mixed - voice).powi(2))
            .sum::<f32>();
        let cleaned_residual = cleaned[tail..]
            .iter()
            .zip(clean_voice[tail..].iter())
            .map(|(mixed, voice)| (mixed - voice).powi(2))
            .sum::<f32>();
        let max_step = cleaned
            .windows(2)
            .map(|window| (window[1] - window[0]).abs())
            .fold(0.0_f32, f32::max);

        assert!(hum);
        assert!((57.0..=61.0).contains(&tracked_hz), "tracked_hz={tracked_hz}");
        assert!(
            cleaned_residual < input_residual * 0.72,
            "cleaned_residual={cleaned_residual} input_residual={input_residual}"
        );
        assert!(max_step < 0.20, "retune max_step={max_step}");
    }

    #[test]
    fn test_adaptive_cleanup_uses_harmonic_to_track_off_nominal_hum() {
        let sample_rate = 48_000.0_f32;
        let len = sample_rate as usize * 2;
        let fundamental = 51.5_f32;
        let input: Vec<f32> = (0..len)
            .map(|index| {
                let time = index as f32 / sample_rate;
                0.025 * (2.0 * std::f32::consts::PI * fundamental * time).sin()
                    + 0.14 * (2.0 * std::f32::consts::PI * fundamental * 2.0 * time).sin()
                    + 0.04 * (2.0 * std::f32::consts::PI * 1200.0 * time).sin()
            })
            .collect();
        let fixed = process_fixed_input_prefilter(&input, sample_rate);
        let (cleaned, hum, _, _, tracked_hz) =
            process_adaptive_input_cleanup(&input, InputCleanupMode::Strong, sample_rate, 480);
        let tail = len / 2;
        let fixed_harmonic = tone_amplitude(&fixed[tail..], sample_rate, fundamental * 2.0);
        let cleaned_harmonic = tone_amplitude(&cleaned[tail..], sample_rate, fundamental * 2.0);

        assert!(hum);
        assert!((tracked_hz - fundamental).abs() < 1.5, "tracked_hz={tracked_hz}");
        assert!(cleaned_harmonic < fixed_harmonic * 0.72);
    }

    #[test]
    fn test_adaptive_cleanup_does_not_classify_plosive_or_low_voice_as_hum() {
        let sample_rate = 48_000.0_f32;
        let len = sample_rate as usize;
        let plosive: Vec<f32> = (0..len)
            .map(|index| {
                let time = index as f32 / sample_rate;
                if time < 0.055 {
                    0.7 * (1.0 - time / 0.055)
                        * (2.0 * std::f32::consts::PI * 52.0 * time).sin()
                } else {
                    0.0
                }
            })
            .collect();
        let low_voice: Vec<f32> = (0..len)
            .map(|index| {
                let time = index as f32 / sample_rate;
                0.12 * (2.0 * std::f32::consts::PI * 90.0 * time).sin()
                    + 0.06 * (2.0 * std::f32::consts::PI * 180.0 * time).sin()
                    + 0.03 * (2.0 * std::f32::consts::PI * 270.0 * time).sin()
            })
            .collect();

        let (_, plosive_hum, plosive_rumble, _, _) = process_adaptive_input_cleanup(
            &plosive,
            InputCleanupMode::Strong,
            sample_rate,
            480,
        );
        let (_, voice_hum, _, voice_highpass, _) = process_adaptive_input_cleanup(
            &low_voice,
            InputCleanupMode::Strong,
            sample_rate,
            480,
        );

        assert!(!plosive_hum);
        assert!(plosive_rumble);
        assert!(!voice_hum);
        assert_eq!(voice_highpass, INPUT_PREFILTER_HZ as f32);
    }

    #[test]
    fn test_adaptive_cleanup_selects_one_highpass_instead_of_cascading() {
        let sample_rate = 48_000.0_f32;
        let input: Vec<f32> = (0..8192)
            .map(|index| {
                let time = index as f32 / sample_rate;
                0.05 * (2.0 * std::f32::consts::PI * 300.0 * time).sin()
                    + 0.03 * (2.0 * std::f32::consts::PI * 2000.0 * time).sin()
            })
            .collect();
        let fixed = process_fixed_input_prefilter(&input, sample_rate);
        let (adaptive, hum, rumble, highpass, _) = process_adaptive_input_cleanup(
            &input,
            InputCleanupMode::Gentle,
            sample_rate,
            480,
        );
        let max_difference = fixed
            .iter()
            .zip(adaptive.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);

        assert!(!hum);
        assert!(!rumble);
        assert_eq!(highpass, INPUT_PREFILTER_HZ as f32);
        assert!(max_difference < 1.0e-5, "max_difference={max_difference}");
    }

    #[test]
    fn test_final_output_sanitizer_rejects_non_finite_and_limits_ceiling() {
        let mut buffer = vec![f32::NAN, f32::INFINITY, -f32::INFINITY, 0.75, -0.75];
        sanitize_and_clamp_output_inplace(&mut buffer, 0.5);
        assert_eq!(buffer, vec![0.0, 0.0, 0.0, 0.5, -0.5]);
    }

    #[test]
    fn test_final_output_sanitizer_records_clip_events() {
        let output_clip_event_count = AtomicU64::new(0);
        let output_clip_peak_db = AtomicU32::new((-120.0_f32).to_bits());
        let mut buffer = vec![0.25, 0.75, -0.8, f32::NAN];

        sanitize_and_clamp_output_inplace_with_metrics(
            &mut buffer,
            0.5,
            &output_clip_event_count,
            &output_clip_peak_db,
        );

        assert_eq!(buffer, vec![0.25, 0.5, -0.5, 0.0]);
        assert_eq!(output_clip_event_count.load(Ordering::Relaxed), 2);
        let peak_db = f32::from_bits(output_clip_peak_db.load(Ordering::Relaxed));
        assert!((peak_db - (20.0_f32 * 0.8_f32.log10())).abs() < 1e-6);
    }

    #[test]
    fn test_careful_output_mode_lowers_effective_limiter_ceiling() {
        let processor = AudioProcessor::new();

        assert!(processor.is_limiter_careful_output_enabled());
        assert_eq!(
            processor.limiter_effective_ceiling_db(),
            CAREFUL_OUTPUT_CEILING_DB
        );

        processor.set_limiter_ceiling(-0.25);
        assert_eq!(
            processor.limiter_effective_ceiling_db(),
            CAREFUL_OUTPUT_CEILING_DB
        );

        processor.set_limiter_careful_output_enabled(false);
        assert_eq!(processor.limiter_effective_ceiling_db(), -0.25);

        processor.set_limiter_ceiling(-3.0);
        processor.set_limiter_careful_output_enabled(true);
        assert_eq!(processor.limiter_effective_ceiling_db(), -3.0);
    }

    #[test]
    fn test_retime_audio_block_can_expand_and_compress() {
        let input = [0.0_f32, 0.25, 0.5, 0.75, 1.0, 0.5];
        let mut scratch = FixedAudioBuffer::<f32, 32>::new();

        let expanded = retime_audio_block(&input, 0.5, 32, &mut scratch);
        assert!(expanded.len() > input.len());

        let compressed = retime_audio_block(&input, 2.0, 32, &mut scratch);
        assert!(compressed.len() < input.len());
    }

    #[test]
    fn test_retime_audio_block_linear_interpolation_does_not_overshoot_neighbors() {
        let input = [0.0_f32, 0.5, 1.0, 0.5, 0.0];
        let mut scratch = FixedAudioBuffer::<f32, 32>::new();

        let expanded = retime_audio_block(&input, 0.7, 32, &mut scratch);

        for sample in expanded {
            assert!((*sample >= 0.0) && (*sample <= 1.0));
        }
    }

    fn warmed_meter_stats(buffer: &[f32]) -> MeterBlockStats {
        let coeff = smoothing_coeff_for_time_constant(TARGET_SAMPLE_RATE as f32, 100.0);
        let mut rms_acc = 0.0;
        let mut stats = MeterBlockStats {
            peak_db: -120.0,
            rms_db: -120.0,
            crest_factor_db: 0.0,
            mean_power: 0.0,
        };
        for _ in 0..4 {
            stats = update_meter_block_stats(buffer, &mut rms_acc, coeff);
        }
        stats
    }

    #[test]
    fn test_meter_stats_are_stable_for_silence_sine_noise_and_speech_like_input() {
        let sample_rate = TARGET_SAMPLE_RATE as f32;
        let len = TARGET_SAMPLE_RATE as usize;

        let silence = vec![0.0_f32; len];
        let silence_stats = warmed_meter_stats(&silence);
        assert_eq!(silence_stats.peak_db, -120.0);
        assert_eq!(silence_stats.rms_db, -120.0);
        assert_eq!(silence_stats.crest_factor_db, 0.0);

        let sine = generate_sine(sample_rate, 1000.0, len);
        let sine_stats = warmed_meter_stats(&sine);
        assert!((-2.5..=-1.5).contains(&sine_stats.peak_db));
        assert!((-5.6..=-4.4).contains(&sine_stats.rms_db));
        assert!((2.5..=3.6).contains(&sine_stats.crest_factor_db));

        let noise_like: Vec<f32> = (0..len)
            .map(|index| {
                let n = ((index as u32).wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
                    & 0xffff) as f32
                    / 32768.0
                    - 1.0;
                0.16 * n
            })
            .collect();
        let noise_stats = warmed_meter_stats(&noise_like);
        assert!(noise_stats.rms_db < noise_stats.peak_db);
        assert!((4.0..=6.5).contains(&noise_stats.crest_factor_db));

        let speech_like: Vec<f32> = (0..len)
            .map(|index| {
                let t = index as f32 / sample_rate;
                let envelope = 0.35 + 0.65 * (2.0 * std::f32::consts::PI * 3.0 * t).sin().max(0.0);
                envelope
                    * (0.12 * (2.0 * std::f32::consts::PI * 140.0 * t).sin()
                        + 0.06 * (2.0 * std::f32::consts::PI * 280.0 * t).sin()
                        + 0.03 * (2.0 * std::f32::consts::PI * 1120.0 * t).sin())
            })
            .collect();
        let speech_stats = warmed_meter_stats(&speech_like);
        assert!(speech_stats.rms_db < -12.0);
        assert!((5.0..=14.0).contains(&speech_stats.crest_factor_db));
    }

    fn generate_sine(sample_rate: f32, frequency_hz: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let phase =
                    2.0 * std::f32::consts::PI * frequency_hz * index as f32 / sample_rate;
                0.8 * phase.sin()
            })
            .collect()
    }

    fn tone_components(signal: &[f32], sample_rate: f32, frequency_hz: f32) -> (f64, f64) {
        let omega = 2.0 * std::f64::consts::PI * frequency_hz as f64 / sample_rate as f64;
        let mut cos_sum = 0.0_f64;
        let mut sin_sum = 0.0_f64;
        for (index, sample) in signal.iter().copied().enumerate() {
            let phase = omega * index as f64;
            cos_sum += sample as f64 * phase.cos();
            sin_sum += sample as f64 * phase.sin();
        }
        let scale = 2.0 / signal.len().max(1) as f64;
        (cos_sum * scale, sin_sum * scale)
    }

    fn tone_amplitude(signal: &[f32], sample_rate: f32, frequency_hz: f32) -> f32 {
        let (cos_coeff, sin_coeff) = tone_components(signal, sample_rate, frequency_hz);
        (cos_coeff.hypot(sin_coeff)) as f32
    }

    fn max_harmonic_db(signal: &[f32], sample_rate: f32, frequency_hz: f32) -> f32 {
        let nyquist = sample_rate / 2.0;
        let mut max_harmonic = 0.0_f32;
        for harmonic in 2..=5 {
            let harmonic_hz = frequency_hz * harmonic as f32;
            if harmonic_hz >= nyquist {
                break;
            }
            max_harmonic = max_harmonic.max(tone_amplitude(signal, sample_rate, harmonic_hz));
        }
        dbfs(max_harmonic)
    }

    fn subtract_fitted_fundamental(signal: &[f32], sample_rate: f32, frequency_hz: f32) -> Vec<f32> {
        let (cos_coeff, sin_coeff) = tone_components(signal, sample_rate, frequency_hz);
        let omega = 2.0 * std::f64::consts::PI * frequency_hz as f64 / sample_rate as f64;
        signal
            .iter()
            .copied()
            .enumerate()
            .map(|(index, sample)| {
                let phase = omega * index as f64;
                let fitted = cos_coeff * phase.cos() + sin_coeff * phase.sin();
                sample - fitted as f32
            })
            .collect()
    }

    fn rms_db(signal: &[f32]) -> f32 {
        if signal.is_empty() {
            return -120.0;
        }
        let power = signal
            .iter()
            .map(|sample| {
                let sample = *sample as f64;
                sample * sample
            })
            .sum::<f64>()
            / signal.len() as f64;
        dbfs(power.sqrt() as f32)
    }

    fn dbfs(value: f32) -> f32 {
        (value.max(1e-12)).log10() * 20.0
    }

    fn rms_error_db(left: &[f32], right: &[f32]) -> f32 {
        let len = left.len().min(right.len());
        if len == 0 {
            return -120.0;
        }
        let power = left
            .iter()
            .zip(right.iter())
            .take(len)
            .map(|(l, r)| {
                let diff = *l as f64 - *r as f64;
                diff * diff
            })
            .sum::<f64>()
            / len as f64;
        dbfs(power.sqrt() as f32)
    }

    fn sinc_reference_retime(input: &[f32], speed_ratio: f32) -> Vec<f32> {
        let params = SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: calculate_cutoff(128, WindowFunction::BlackmanHarris2),
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler =
            SincFixedIn::<f64>::new(1.0 / speed_ratio as f64, 1.2, params, input.len(), 1)
                .unwrap();
        let input_f64: Vec<f64> = input.iter().map(|sample| *sample as f64).collect();
        let in_slices = [&input_f64[..]];
        let mut outbuf = resampler.output_buffer_allocate(true);
        let (_nbr_in, nbr_out) = resampler
            .process_into_buffer(&in_slices, &mut outbuf, None)
            .unwrap();
        outbuf[0]
            .iter()
            .take(nbr_out)
            .map(|sample| *sample as f32)
            .collect()
    }

    #[test]
    fn test_retime_audio_block_quality_stays_within_measured_reference_bounds() {
        let sample_rate = TARGET_SAMPLE_RATE as f32;
        let input = generate_sine(sample_rate, 10_000.0, TARGET_SAMPLE_RATE as usize);
        let cases = [
            (0.995_f32, -20.0_f32, 1.5_f32),
            (1.003_f32, -20.0_f32, 1.5_f32),
            (1.03_f32, -16.0_f32, 2.0_f32),
            (1.06_f32, -12.0_f32, 2.5_f32),
        ];
        let mut scratch = FixedAudioBuffer::<f32, 96_000>::new();

        for (speed_ratio, max_error_db, max_fundamental_delta_db) in cases {
            let linear = retime_audio_block(&input, speed_ratio, 96_000, &mut scratch).to_vec();
            let reference = sinc_reference_retime(&input, speed_ratio);
            let compare_len = linear.len().min(reference.len());
            let linear = &linear[..compare_len];
            let reference = &reference[..compare_len];
            let expected_fundamental_hz = 10_000.0 * speed_ratio;
            let linear_fundamental_db =
                dbfs(tone_amplitude(linear, sample_rate, expected_fundamental_hz));
            let reference_fundamental_db =
                dbfs(tone_amplitude(reference, sample_rate, expected_fundamental_hz));
            let linear_harmonic_db =
                max_harmonic_db(linear, sample_rate, expected_fundamental_hz);
            let linear_residual_db = rms_db(&subtract_fitted_fundamental(
                linear,
                sample_rate,
                expected_fundamental_hz,
            ));
            let reference_residual_db = rms_db(&subtract_fitted_fundamental(
                reference,
                sample_rate,
                expected_fundamental_hz,
            ));
            let error_db = rms_error_db(linear, reference);

            assert!(
                (linear_fundamental_db - reference_fundamental_db).abs()
                    <= max_fundamental_delta_db,
                "ratio={speed_ratio} linear_fundamental_db={linear_fundamental_db} reference_fundamental_db={reference_fundamental_db}",
            );
            assert!(
                error_db <= max_error_db,
                "ratio={speed_ratio} error_db={error_db} max_error_db={max_error_db}",
            );
            assert!(
                linear_harmonic_db <= linear_fundamental_db - 12.0,
                "ratio={speed_ratio} linear_harmonic_db={linear_harmonic_db} linear_fundamental_db={linear_fundamental_db}",
            );
            assert!(
                linear_residual_db <= reference_residual_db + 40.0,
                "ratio={speed_ratio} linear_residual_db={linear_residual_db} reference_residual_db={reference_residual_db}",
            );
        }
    }

    fn test_output_writer_limits(
        output_target_center_samples: usize,
        output_hard_backlog_samples: usize,
        discontinuity_fade_samples: usize,
    ) -> OutputWriteLimits {
        OutputWriteLimits {
            output_target_center_samples,
            output_hard_backlog_samples,
            discontinuity_fade_samples,
            max_catchup_ratio: 1.03,
            max_emergency_catchup_ratio: 1.06,
        }
    }

    fn output_writer_counters<'a>(
        output_counts: (&'a AtomicU64, &'a AtomicU64, &'a AtomicU64),
        output_short_write_dropped_samples: &'a AtomicU64,
        rt_buffer_overflow_count: &'a AtomicU64,
        rt_error_code: &'a AtomicU32,
        output_buffer_len: &'a AtomicU32,
        last_output_write_time: &'a AtomicU64,
    ) -> OutputWriteCounters<'a> {
        static OUTPUT_CLIP_EVENT_COUNT: AtomicU64 = AtomicU64::new(0);
        static OUTPUT_CLIP_PEAK_DB: AtomicU32 = AtomicU32::new((-120.0_f32).to_bits());
        static OUTPUT_TRUE_PEAK_EVENT_COUNT: AtomicU64 = AtomicU64::new(0);
        static OUTPUT_TRUE_PEAK_DB: AtomicU32 = AtomicU32::new((-120.0_f32).to_bits());
        static OUTPUT_TRUE_PEAK_INPUT_DB: AtomicU32 = AtomicU32::new((-120.0_f32).to_bits());
        static OUTPUT_TRUE_PEAK_GAIN_REDUCTION_DB: AtomicU32 =
            AtomicU32::new(0.0_f32.to_bits());
        static OUTPUT_TRUE_PEAK_GAIN_REDUCTION_HISTORY_DB: AtomicU32 =
            AtomicU32::new(0.0_f32.to_bits());
        static OUTPUT_TRUE_PEAK_HEADROOM_DB: AtomicU32 = AtomicU32::new(120.0_f32.to_bits());
        let (
            jitter_dropped_samples,
            output_retime_adjustment_count,
            output_recovery_event_count,
        ) = output_counts;
        OutputWriteCounters {
            jitter_dropped_samples,
            output_retime_adjustment_count,
            output_recovery_event_count,
            output_short_write_dropped_samples,
            rt_buffer_overflow_count,
            rt_error_code,
            output_buffer_len,
            last_output_write_time,
            output_clip_event_count: &OUTPUT_CLIP_EVENT_COUNT,
            output_clip_peak_db: &OUTPUT_CLIP_PEAK_DB,
            output_true_peak_event_count: &OUTPUT_TRUE_PEAK_EVENT_COUNT,
            output_true_peak_db: &OUTPUT_TRUE_PEAK_DB,
            output_true_peak_input_db: &OUTPUT_TRUE_PEAK_INPUT_DB,
            output_true_peak_gain_reduction_db: &OUTPUT_TRUE_PEAK_GAIN_REDUCTION_DB,
            output_true_peak_gain_reduction_history_db:
                &OUTPUT_TRUE_PEAK_GAIN_REDUCTION_HISTORY_DB,
            output_true_peak_headroom_db: &OUTPUT_TRUE_PEAK_HEADROOM_DB,
        }
    }

    #[test]
    fn test_output_writer_noop_write_returns_false() {
        let rb = AudioRingBuffer::new(32);
        let (mut producer, consumer) = rb.split();
        let mut control_scratch = FixedAudioBuffer::<f32, 64>::new();
        let mut fade_scratch = FixedAudioBuffer::<f32, 64>::new();
        let mut safety_scratch = FixedAudioBuffer::<f32, 64>::new();
        let mut drift_error_ema = 0.0_f32;
        let fade_remaining = Cell::new(0usize);
        let limiter_enabled = AtomicBool::new(true);
        let output_ceiling_linear = Cell::new(1.0_f32);
        let jitter_dropped_samples = AtomicU64::new(0);
        let output_retime_adjustment_count = AtomicU64::new(0);
        let output_recovery_event_count = AtomicU64::new(0);
        let output_short_write_dropped_samples = AtomicU64::new(0);
        let rt_buffer_overflow_count = AtomicU64::new(0);
        let rt_error_code = AtomicU32::new(RtErrorCode::None as u32);
        let output_buffer_len = AtomicU32::new(0);
        let last_output_write_time = AtomicU64::new(0);
        let mut true_peak_detector = TruePeakDetector::new();
        let mut true_peak_limiter = TruePeakLimiter::default();

        let mut writer = OutputWriteContext {
            output_producer: &mut producer,
            output_queue_control_scratch: &mut control_scratch,
            discontinuity_fade_scratch: &mut fade_scratch,
            output_safety_scratch: &mut safety_scratch,
            true_peak_detector: &mut true_peak_detector,
            true_peak_limiter: &mut true_peak_limiter,
            drift_error_ema: &mut drift_error_ema,
            discontinuity_fade_remaining: &fade_remaining,
            limiter_enabled: &limiter_enabled,
            output_ceiling_linear: &output_ceiling_linear,
            counters: output_writer_counters(
                (
                    &jitter_dropped_samples,
                    &output_retime_adjustment_count,
                    &output_recovery_event_count,
                ),
                &output_short_write_dropped_samples,
                &rt_buffer_overflow_count,
                &rt_error_code,
                &output_buffer_len,
                &last_output_write_time,
            ),
            limits: test_output_writer_limits(8, 16, 4),
        };

        assert!(!writer.write_chunk(&[], false));
        assert!(consumer.is_empty());
        assert_eq!(output_buffer_len.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_output_writer_accounts_for_queue_full_short_write() {
        let rb = AudioRingBuffer::new(4);
        let (mut producer, mut consumer) = rb.split();
        producer.write(&[0.0, 0.0, 0.0]);
        let mut control_scratch = FixedAudioBuffer::<f32, 64>::new();
        let mut fade_scratch = FixedAudioBuffer::<f32, 64>::new();
        let mut safety_scratch = FixedAudioBuffer::<f32, 64>::new();
        let mut drift_error_ema = 0.0_f32;
        let fade_remaining = Cell::new(0usize);
        let limiter_enabled = AtomicBool::new(true);
        let output_ceiling_linear = Cell::new(1.0_f32);
        let jitter_dropped_samples = AtomicU64::new(0);
        let output_retime_adjustment_count = AtomicU64::new(0);
        let output_recovery_event_count = AtomicU64::new(0);
        let output_short_write_dropped_samples = AtomicU64::new(0);
        let rt_buffer_overflow_count = AtomicU64::new(0);
        let rt_error_code = AtomicU32::new(RtErrorCode::None as u32);
        let output_buffer_len = AtomicU32::new(3);
        let last_output_write_time = AtomicU64::new(0);
        let mut true_peak_detector = TruePeakDetector::new();
        let mut true_peak_limiter = TruePeakLimiter::default();

        let mut writer = OutputWriteContext {
            output_producer: &mut producer,
            output_queue_control_scratch: &mut control_scratch,
            discontinuity_fade_scratch: &mut fade_scratch,
            output_safety_scratch: &mut safety_scratch,
            true_peak_detector: &mut true_peak_detector,
            true_peak_limiter: &mut true_peak_limiter,
            drift_error_ema: &mut drift_error_ema,
            discontinuity_fade_remaining: &fade_remaining,
            limiter_enabled: &limiter_enabled,
            output_ceiling_linear: &output_ceiling_linear,
            counters: output_writer_counters(
                (
                    &jitter_dropped_samples,
                    &output_retime_adjustment_count,
                    &output_recovery_event_count,
                ),
                &output_short_write_dropped_samples,
                &rt_buffer_overflow_count,
                &rt_error_code,
                &output_buffer_len,
                &last_output_write_time,
            ),
            limits: test_output_writer_limits(2, 4, 4),
        };

        assert!(writer.write_chunk(&[0.1, 0.2, 0.3, 0.4], false));

        assert_eq!(
            output_short_write_dropped_samples.load(Ordering::Relaxed),
            3
        );
        assert_eq!(output_retime_adjustment_count.load(Ordering::Relaxed), 0);
        assert_eq!(output_recovery_event_count.load(Ordering::Relaxed), 1);
        assert_eq!(fade_remaining.get(), 4);
        assert_eq!(output_buffer_len.load(Ordering::Relaxed), 4);

        let mut drained = [0.0_f32; 4];
        assert_eq!(consumer.read(&mut drained), 4);
    }

    #[test]
    fn test_output_writer_retime_can_expand_and_compress_output() {
        let input: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();

        let expand_rb = AudioRingBuffer::new(1024);
        let (mut expand_producer, mut expand_consumer) = expand_rb.split();
        let mut expand_control_scratch = FixedAudioBuffer::<f32, 512>::new();
        let mut expand_fade_scratch = FixedAudioBuffer::<f32, 512>::new();
        let mut expand_safety_scratch = FixedAudioBuffer::<f32, 512>::new();
        let mut expand_drift_error_ema = -10_000.0_f32;
        let expand_fade_remaining = Cell::new(0usize);
        let limiter_enabled = AtomicBool::new(false);
        let output_ceiling_linear = Cell::new(1.0_f32);
        let expand_jitter_dropped_samples = AtomicU64::new(0);
        let expand_output_retime_adjustment_count = AtomicU64::new(0);
        let expand_output_recovery_event_count = AtomicU64::new(0);
        let expand_output_short_write_dropped_samples = AtomicU64::new(0);
        let expand_rt_buffer_overflow_count = AtomicU64::new(0);
        let expand_rt_error_code = AtomicU32::new(RtErrorCode::None as u32);
        let expand_output_buffer_len = AtomicU32::new(0);
        let expand_last_output_write_time = AtomicU64::new(0);
        let mut expand_true_peak_detector = TruePeakDetector::new();
        let mut expand_true_peak_limiter = TruePeakLimiter::default();
        let mut expand_writer = OutputWriteContext {
            output_producer: &mut expand_producer,
            output_queue_control_scratch: &mut expand_control_scratch,
            discontinuity_fade_scratch: &mut expand_fade_scratch,
            output_safety_scratch: &mut expand_safety_scratch,
            true_peak_detector: &mut expand_true_peak_detector,
            true_peak_limiter: &mut expand_true_peak_limiter,
            drift_error_ema: &mut expand_drift_error_ema,
            discontinuity_fade_remaining: &expand_fade_remaining,
            limiter_enabled: &limiter_enabled,
            output_ceiling_linear: &output_ceiling_linear,
            counters: output_writer_counters(
                (
                    &expand_jitter_dropped_samples,
                    &expand_output_retime_adjustment_count,
                    &expand_output_recovery_event_count,
                ),
                &expand_output_short_write_dropped_samples,
                &expand_rt_buffer_overflow_count,
                &expand_rt_error_code,
                &expand_output_buffer_len,
                &expand_last_output_write_time,
            ),
            limits: test_output_writer_limits(128, 256, 4),
        };

        assert!(expand_writer.write_chunk(&input, false));
        let mut expanded = vec![0.0_f32; 512];
        let expanded_len = expand_consumer.read(&mut expanded);
        assert!(expanded_len > input.len());
        assert_eq!(
            expand_output_retime_adjustment_count.load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            expand_output_recovery_event_count.load(Ordering::Relaxed),
            0
        );

        let compress_rb = AudioRingBuffer::new(1024);
        let (mut compress_producer, mut compress_consumer) = compress_rb.split();
        compress_producer.write(&vec![0.0_f32; 256]);
        let mut compress_control_scratch = FixedAudioBuffer::<f32, 512>::new();
        let mut compress_fade_scratch = FixedAudioBuffer::<f32, 512>::new();
        let mut compress_safety_scratch = FixedAudioBuffer::<f32, 512>::new();
        let mut compress_drift_error_ema = 10_000.0_f32;
        let compress_fade_remaining = Cell::new(0usize);
        let compress_jitter_dropped_samples = AtomicU64::new(0);
        let compress_output_retime_adjustment_count = AtomicU64::new(0);
        let compress_output_recovery_event_count = AtomicU64::new(0);
        let compress_output_short_write_dropped_samples = AtomicU64::new(0);
        let compress_rt_buffer_overflow_count = AtomicU64::new(0);
        let compress_rt_error_code = AtomicU32::new(RtErrorCode::None as u32);
        let compress_output_buffer_len = AtomicU32::new(256);
        let compress_last_output_write_time = AtomicU64::new(0);
        let mut compress_true_peak_detector = TruePeakDetector::new();
        let mut compress_true_peak_limiter = TruePeakLimiter::default();
        let mut compress_writer = OutputWriteContext {
            output_producer: &mut compress_producer,
            output_queue_control_scratch: &mut compress_control_scratch,
            discontinuity_fade_scratch: &mut compress_fade_scratch,
            output_safety_scratch: &mut compress_safety_scratch,
            true_peak_detector: &mut compress_true_peak_detector,
            true_peak_limiter: &mut compress_true_peak_limiter,
            drift_error_ema: &mut compress_drift_error_ema,
            discontinuity_fade_remaining: &compress_fade_remaining,
            limiter_enabled: &limiter_enabled,
            output_ceiling_linear: &output_ceiling_linear,
            counters: output_writer_counters(
                (
                    &compress_jitter_dropped_samples,
                    &compress_output_retime_adjustment_count,
                    &compress_output_recovery_event_count,
                ),
                &compress_output_short_write_dropped_samples,
                &compress_rt_buffer_overflow_count,
                &compress_rt_error_code,
                &compress_output_buffer_len,
                &compress_last_output_write_time,
            ),
            limits: test_output_writer_limits(128, 256, 4),
        };

        assert!(compress_writer.write_chunk(&input, false));
        let mut compressed = vec![0.0_f32; 1024];
        let compressed_len = compress_consumer.read(&mut compressed);
        assert!(compressed_len < input.len() + 256);
        assert!(compressed_len > 256);
        assert!(compress_jitter_dropped_samples.load(Ordering::Relaxed) > 0);
        assert_eq!(
            compress_output_retime_adjustment_count.load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            compress_output_recovery_event_count.load(Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn test_output_writer_applies_discontinuity_fade_after_short_write_drop() {
        let rb = AudioRingBuffer::new(8);
        let (mut producer, mut consumer) = rb.split();
        producer.write(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut control_scratch = FixedAudioBuffer::<f32, 64>::new();
        let mut fade_scratch = FixedAudioBuffer::<f32, 64>::new();
        let mut safety_scratch = FixedAudioBuffer::<f32, 64>::new();
        let mut drift_error_ema = 0.0_f32;
        let fade_remaining = Cell::new(0usize);
        let limiter_enabled = AtomicBool::new(false);
        let output_ceiling_linear = Cell::new(1.0_f32);
        let jitter_dropped_samples = AtomicU64::new(0);
        let output_retime_adjustment_count = AtomicU64::new(0);
        let output_recovery_event_count = AtomicU64::new(0);
        let output_short_write_dropped_samples = AtomicU64::new(0);
        let rt_buffer_overflow_count = AtomicU64::new(0);
        let rt_error_code = AtomicU32::new(RtErrorCode::None as u32);
        let output_buffer_len = AtomicU32::new(6);
        let last_output_write_time = AtomicU64::new(0);
        let mut true_peak_detector = TruePeakDetector::new();
        let mut true_peak_limiter = TruePeakLimiter::default();
        let mut writer = OutputWriteContext {
            output_producer: &mut producer,
            output_queue_control_scratch: &mut control_scratch,
            discontinuity_fade_scratch: &mut fade_scratch,
            output_safety_scratch: &mut safety_scratch,
            true_peak_detector: &mut true_peak_detector,
            true_peak_limiter: &mut true_peak_limiter,
            drift_error_ema: &mut drift_error_ema,
            discontinuity_fade_remaining: &fade_remaining,
            limiter_enabled: &limiter_enabled,
            output_ceiling_linear: &output_ceiling_linear,
            counters: output_writer_counters(
                (
                    &jitter_dropped_samples,
                    &output_retime_adjustment_count,
                    &output_recovery_event_count,
                ),
                &output_short_write_dropped_samples,
                &rt_buffer_overflow_count,
                &rt_error_code,
                &output_buffer_len,
                &last_output_write_time,
            ),
            limits: test_output_writer_limits(4, 8, 4),
        };

        assert!(writer.write_chunk(&[1.0, 1.0, 1.0, 1.0], false));
        let mut first_drain = [0.0_f32; 8];
        assert_eq!(consumer.read(&mut first_drain), 8);
        assert_eq!(fade_remaining.get(), 4);

        assert!(writer.write_chunk(&[1.0, 1.0, 1.0, 1.0], false));
        let mut faded = [0.0_f32; 4];
        assert_eq!(consumer.read(&mut faded), 4);
        assert!(faded[0] > 0.0);
        assert!(faded[0] < faded[1]);
        assert!(faded[1] < faded[2]);
        assert!(faded[2] < faded[3]);
        assert!((faded[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_output_writer_still_applies_limiter_ceiling_clamp() {
        let rb = AudioRingBuffer::new(64);
        let (mut producer, mut consumer) = rb.split();
        let mut control_scratch = FixedAudioBuffer::<f32, 32>::new();
        let mut fade_scratch = FixedAudioBuffer::<f32, 32>::new();
        let mut safety_scratch = FixedAudioBuffer::<f32, 32>::new();
        let mut drift_error_ema = 0.0_f32;
        let fade_remaining = Cell::new(0usize);
        let limiter_enabled = AtomicBool::new(true);
        let output_ceiling_linear = Cell::new(0.5_f32);
        let jitter_dropped_samples = AtomicU64::new(0);
        let output_retime_adjustment_count = AtomicU64::new(0);
        let output_recovery_event_count = AtomicU64::new(0);
        let output_short_write_dropped_samples = AtomicU64::new(0);
        let rt_buffer_overflow_count = AtomicU64::new(0);
        let rt_error_code = AtomicU32::new(RtErrorCode::None as u32);
        let output_buffer_len = AtomicU32::new(0);
        let last_output_write_time = AtomicU64::new(0);
        let mut true_peak_detector = TruePeakDetector::new();
        let mut true_peak_limiter = TruePeakLimiter::default();
        let mut writer = OutputWriteContext {
            output_producer: &mut producer,
            output_queue_control_scratch: &mut control_scratch,
            discontinuity_fade_scratch: &mut fade_scratch,
            output_safety_scratch: &mut safety_scratch,
            true_peak_detector: &mut true_peak_detector,
            true_peak_limiter: &mut true_peak_limiter,
            drift_error_ema: &mut drift_error_ema,
            discontinuity_fade_remaining: &fade_remaining,
            limiter_enabled: &limiter_enabled,
            output_ceiling_linear: &output_ceiling_linear,
            counters: output_writer_counters(
                (
                    &jitter_dropped_samples,
                    &output_retime_adjustment_count,
                    &output_recovery_event_count,
                ),
                &output_short_write_dropped_samples,
                &rt_buffer_overflow_count,
                &rt_error_code,
                &output_buffer_len,
                &last_output_write_time,
            ),
            limits: test_output_writer_limits(4, 8, 4),
        };

        assert!(writer.write_chunk(&[2.0, -2.0, 0.5, 0.0, 0.0, 0.0, 0.0], true));
        assert!(writer.write_chunk(&[0.0; 24], true));
        let mut limited = [0.0_f32; 31];
        assert_eq!(consumer.read(&mut limited), limited.len());
        assert!(limited.iter().all(|sample| sample.abs() <= 0.5 + 1e-6));
        assert!(limited.iter().any(|sample| sample.abs() > 0.1));
    }

    #[test]
    fn test_output_writer_limits_true_peak_without_sample_clip() {
        let rb = AudioRingBuffer::new(64);
        let (mut producer, _consumer) = rb.split();
        let mut control_scratch = FixedAudioBuffer::<f32, 32>::new();
        let mut fade_scratch = FixedAudioBuffer::<f32, 32>::new();
        let mut safety_scratch = FixedAudioBuffer::<f32, 32>::new();
        let mut true_peak_detector = TruePeakDetector::new();
        let mut drift_error_ema = 0.0_f32;
        let fade_remaining = Cell::new(0usize);
        let limiter_enabled = AtomicBool::new(true);
        let output_ceiling_linear = Cell::new(1.0_f32);
        let jitter_dropped_samples = AtomicU64::new(0);
        let output_retime_adjustment_count = AtomicU64::new(0);
        let output_recovery_event_count = AtomicU64::new(0);
        let output_short_write_dropped_samples = AtomicU64::new(0);
        let rt_buffer_overflow_count = AtomicU64::new(0);
        let rt_error_code = AtomicU32::new(RtErrorCode::None as u32);
        let output_buffer_len = AtomicU32::new(0);
        let last_output_write_time = AtomicU64::new(0);
        let output_clip_event_count = AtomicU64::new(0);
        let output_clip_peak_db = AtomicU32::new((-120.0_f32).to_bits());
        let output_true_peak_event_count = AtomicU64::new(0);
        let output_true_peak_db = AtomicU32::new((-120.0_f32).to_bits());
        let output_true_peak_input_db = AtomicU32::new((-120.0_f32).to_bits());
        let output_true_peak_gain_reduction_db = AtomicU32::new(0.0_f32.to_bits());
        let output_true_peak_gain_reduction_history_db = AtomicU32::new(0.0_f32.to_bits());
        let output_true_peak_headroom_db = AtomicU32::new(120.0_f32.to_bits());
        let mut true_peak_limiter = TruePeakLimiter::default();
        let mut writer = OutputWriteContext {
            output_producer: &mut producer,
            output_queue_control_scratch: &mut control_scratch,
            discontinuity_fade_scratch: &mut fade_scratch,
            output_safety_scratch: &mut safety_scratch,
            true_peak_detector: &mut true_peak_detector,
            true_peak_limiter: &mut true_peak_limiter,
            drift_error_ema: &mut drift_error_ema,
            discontinuity_fade_remaining: &fade_remaining,
            limiter_enabled: &limiter_enabled,
            output_ceiling_linear: &output_ceiling_linear,
            counters: OutputWriteCounters {
                jitter_dropped_samples: &jitter_dropped_samples,
                output_retime_adjustment_count: &output_retime_adjustment_count,
                output_recovery_event_count: &output_recovery_event_count,
                output_short_write_dropped_samples: &output_short_write_dropped_samples,
                rt_buffer_overflow_count: &rt_buffer_overflow_count,
                rt_error_code: &rt_error_code,
                output_buffer_len: &output_buffer_len,
                last_output_write_time: &last_output_write_time,
                output_clip_event_count: &output_clip_event_count,
                output_clip_peak_db: &output_clip_peak_db,
                output_true_peak_event_count: &output_true_peak_event_count,
                output_true_peak_db: &output_true_peak_db,
                output_true_peak_input_db: &output_true_peak_input_db,
                output_true_peak_gain_reduction_db: &output_true_peak_gain_reduction_db,
                output_true_peak_gain_reduction_history_db:
                    &output_true_peak_gain_reduction_history_db,
                output_true_peak_headroom_db: &output_true_peak_headroom_db,
            },
            limits: test_output_writer_limits(4, 8, 4),
        };

        assert!(writer.write_chunk(&[0.0, 1.0, 1.0, 0.0, 0.0], true));
        assert!(writer.write_chunk(&[0.0; 32], true));

        assert_eq!(output_clip_event_count.load(Ordering::Relaxed), 0);
        assert_eq!(output_true_peak_event_count.load(Ordering::Relaxed), 1);
        assert!(f32::from_bits(output_true_peak_db.load(Ordering::Relaxed)) <= 0.01);
        assert!(f32::from_bits(output_true_peak_input_db.load(Ordering::Relaxed)) > 0.0);
        assert!(
            f32::from_bits(output_true_peak_gain_reduction_db.load(Ordering::Relaxed)) > 0.0
        );
    }

    fn fill_probe_block(buffer: &mut [f32]) {
        for (index, sample) in buffer.iter_mut().enumerate() {
            let phase = index as f32 * 0.037;
            *sample = (phase.sin() * 0.35).clamp(-0.8, 0.8);
        }
    }

    #[test]
    fn test_steady_state_dsp_blocks_do_not_allocate() {
        let mut block = [0.0_f32; 512];
        fill_probe_block(&mut block);

        let mut biquad = Biquad::new(
            BiquadType::Peaking,
            1_000.0,
            3.0,
            DEFAULT_Q,
            TARGET_SAMPLE_RATE as f64,
        );
        biquad.process_block_inplace(&mut block);
        crate::test_alloc::assert_no_allocations("biquad block", || {
            biquad.process_block_inplace(&mut block);
        });

        let mut eq = ParametricEQ::new(TARGET_SAMPLE_RATE as f64);
        eq.set_band_gain(4, 3.0);
        eq.process_block_inplace(&mut block);
        crate::test_alloc::assert_no_allocations("eq block", || {
            eq.process_block_inplace(&mut block);
        });

        let mut gate = NoiseGate::new(-45.0, 5.0, 80.0, TARGET_SAMPLE_RATE as f64);
        gate.process_block_inplace(&mut block);
        crate::test_alloc::assert_no_allocations("gate block", || {
            gate.process_block_inplace(&mut block);
        });

        let mut compressor = Compressor::default_voice(TARGET_SAMPLE_RATE as f64);
        compressor.process_block_inplace(&mut block);
        crate::test_alloc::assert_no_allocations("compressor block", || {
            compressor.process_block_inplace(&mut block);
        });

        let mut deesser = DeEsser::new(TARGET_SAMPLE_RATE as f64);
        deesser.set_enabled(true);
        deesser.process_block_inplace(&mut block);
        crate::test_alloc::assert_no_allocations("deesser block", || {
            deesser.process_block_inplace(&mut block);
        });

        let mut limiter = Limiter::default_settings(TARGET_SAMPLE_RATE as f64);
        limiter.process_block_inplace(&mut block);
        crate::test_alloc::assert_no_allocations("limiter block", || {
            limiter.process_block_inplace(&mut block);
        });

        let mut true_peak_limiter = TruePeakLimiter::default_settings(TARGET_SAMPLE_RATE as f32);
        true_peak_limiter.process_block_inplace(&mut block);
        crate::test_alloc::assert_no_allocations("true peak limiter block", || {
            true_peak_limiter.process_block_inplace(&mut block);
        });

        let mut scratch = FixedAudioBuffer::<f32, 1024>::new();
        let retime_input = [0.1_f32; 256];
        let _ = retime_audio_block(&retime_input, 0.98, 512, &mut scratch);
        crate::test_alloc::assert_no_allocations("retime block", || {
            let output = retime_audio_block(&retime_input, 0.98, 512, &mut scratch);
            assert!(!output.is_empty());
        });

        let strength = Arc::new(AtomicU32::new(1.0_f32.to_bits()));
        let mut suppressor = NoiseSuppressionEngine::new(NoiseModel::RNNoise, strength);
        let rnnoise_frame = [0.0_f32; RNNOISE_FRAME_SIZE];
        let mut suppressor_output = [0.0_f32; RNNOISE_FRAME_SIZE];
        suppressor.push_samples(&rnnoise_frame);
        suppressor.process_frames();
        assert_eq!(
            suppressor.pop_samples_into(&mut suppressor_output),
            RNNOISE_FRAME_SIZE
        );
        crate::test_alloc::assert_no_allocations("rnnoise wrapper block", || {
            suppressor.push_samples(&rnnoise_frame);
            suppressor.process_frames();
            assert_eq!(
                suppressor.pop_samples_into(&mut suppressor_output),
                RNNOISE_FRAME_SIZE
            );
        });

        #[cfg(feature = "deepfilter")]
        {
            use crate::dsp::deepfilter_ffi::{
                DeepFilterModel, DeepFilterProcessor, DEEPFILTER_FRAME_SIZE,
            };

            let strength = Arc::new(AtomicU32::new(1.0_f32.to_bits()));
            let mut deepfilter = DeepFilterProcessor::new(strength, DeepFilterModel::LowLatency);
            deepfilter.set_enabled(false);
            let deepfilter_frame = [0.0_f32; DEEPFILTER_FRAME_SIZE];
            let mut deepfilter_output = [0.0_f32; DEEPFILTER_FRAME_SIZE];
            deepfilter.push_samples(&deepfilter_frame);
            deepfilter.process_frames();
            assert_eq!(
                deepfilter.pop_samples_into(&mut deepfilter_output),
                DEEPFILTER_FRAME_SIZE
            );
            crate::test_alloc::assert_no_allocations("deepfilter passthrough block", || {
                deepfilter.push_samples(&deepfilter_frame);
                deepfilter.process_frames();
                assert_eq!(
                    deepfilter.pop_samples_into(&mut deepfilter_output),
                    DEEPFILTER_FRAME_SIZE
                );
            });
        }
    }

    #[test]
    fn test_offline_block_processor_processes_without_live_stream_state() {
        let mut processor = OfflineDspBlockProcessor::new(TARGET_SAMPLE_RATE as f64);
        processor.set_deesser_enabled(false);
        processor.set_eq_enabled(false);
        processor.set_compressor_enabled(false);
        processor.set_limiter_enabled(false);

        let mut input = [0.0_f32; 8];
        fill_probe_block(&mut input);
        let mut output = FixedAudioBuffer::<f32, 4>::new();

        processor.process_block(&mut input, &mut output);

        assert_eq!(output.len(), 4);
        assert_eq!(output.as_slice(), &input[..4]);
    }

    #[test]
    fn test_offline_block_processor_matches_deterministic_live_stages() {
        let sample_rate = TARGET_SAMPLE_RATE as f64;
        let mut offline = OfflineDspBlockProcessor::new(sample_rate);
        offline.set_deesser_enabled(false);
        offline.set_eq_enabled(true);
        offline.set_compressor_enabled(false);
        offline.set_limiter_enabled(true);
        offline.eq_mut().set_band_frequency(5, 2500.0);
        offline.eq_mut().set_band_gain(5, 4.0);
        offline.eq_mut().set_band_q(5, 1.8);
        offline.limiter_mut().set_ceiling(-1.5);

        let mut input = [0.0_f32; 512];
        for (index, sample) in input.iter_mut().enumerate() {
            let t = index as f64 / sample_rate;
            *sample = (0.38 * (2.0 * std::f64::consts::PI * 2500.0 * t).sin()
                + 0.22 * (2.0 * std::f64::consts::PI * 180.0 * t).sin())
                as f32;
        }
        let mut manual = input;
        let mut eq = ParametricEQ::new(sample_rate);
        eq.set_band_frequency(5, 2500.0);
        eq.set_band_gain(5, 4.0);
        eq.set_band_q(5, 1.8);
        eq.process_block_inplace(&mut manual);
        let mut limiter = Limiter::default_settings(sample_rate);
        limiter.set_ceiling(-1.5);
        limiter.process_block_inplace(&mut manual);
        let mut true_peak_limiter = TruePeakLimiter::default_settings(sample_rate as f32);
        true_peak_limiter.set_ceiling_linear(10.0_f32.powf(-1.5 / 20.0));
        true_peak_limiter.process_block_inplace(&mut manual);

        let mut offline_input = input;
        let mut output = FixedAudioBuffer::<f32, 1024>::new();
        let stats = offline.process_block_with_stats(&mut offline_input, &mut output);

        assert_eq!(output.len(), manual.len());
        assert!(rms_error_db(output.as_slice(), &manual) < -100.0);
        assert!(stats.output_true_peak.is_finite());
        assert!(stats.true_peak_limiter_input_peak.is_finite());
    }

    #[test]
    fn test_rnnoise_accepts_full_rt_block_without_short_write() {
        let strength = Arc::new(AtomicU32::new(1.0_f32.to_bits()));
        let mut suppressor = NoiseSuppressionEngine::new(NoiseModel::RNNoise, strength);
        let input = [0.0_f32; RT_PROCESS_BUFFER_CAPACITY];
        let accepted = suppressor.push_samples(&input);

        assert_eq!(accepted, input.len());

        suppressor.process_frames();
        let expected_frames = input.len() / RNNOISE_FRAME_SIZE;
        let expected_processed = expected_frames * RNNOISE_FRAME_SIZE;
        assert_eq!(suppressor.available_samples(), expected_processed);
        assert_eq!(suppressor.pending_input(), input.len() - expected_processed);

        let mut output = [0.0_f32; RT_SUPPRESSOR_OUTPUT_CAPACITY];
        assert_eq!(
            suppressor.pop_samples_into(&mut output[..expected_processed]),
            expected_processed
        );
    }

    #[test]
    fn test_output_scratch_covers_max_suppressor_output() {
        let scratch = FixedAudioBuffer::<f32, RT_OUTPUT_SCRATCH_CAPACITY>::new();
        assert!(scratch.capacity() >= RT_SUPPRESSOR_OUTPUT_CAPACITY);
    }

    #[test]
    fn test_disabled_rnnoise_accepts_full_rt_block_without_short_write() {
        let strength = Arc::new(AtomicU32::new(1.0_f32.to_bits()));
        let mut suppressor = NoiseSuppressionEngine::new(NoiseModel::RNNoise, strength);
        suppressor.set_enabled(false);
        let input = [0.0_f32; RT_PROCESS_BUFFER_CAPACITY];
        let accepted = suppressor.push_samples(&input);

        assert_eq!(accepted, input.len());

        suppressor.process_frames();
        assert_eq!(suppressor.available_samples(), input.len());
        assert_eq!(suppressor.pending_input(), 0);
    }

    #[cfg(feature = "deepfilter")]
    #[test]
    fn test_deepfilter_accepts_full_rt_block_without_short_write() {
        use crate::dsp::deepfilter_ffi::{DeepFilterModel, DeepFilterProcessor, DEEPFILTER_FRAME_SIZE};

        let strength = Arc::new(AtomicU32::new(1.0_f32.to_bits()));
        let mut suppressor = DeepFilterProcessor::new(strength, DeepFilterModel::LowLatency);
        suppressor.set_enabled(false);
        let input = [0.0_f32; RT_PROCESS_BUFFER_CAPACITY];
        let accepted = suppressor.push_samples(&input);

        assert_eq!(accepted, input.len());

        suppressor.process_frames();
        let expected_frames = input.len() / DEEPFILTER_FRAME_SIZE;
        let expected_processed = expected_frames * DEEPFILTER_FRAME_SIZE;
        assert_eq!(suppressor.available_samples(), expected_processed);
        assert_eq!(suppressor.pending_input(), input.len() - expected_processed);
    }

    #[test]
    fn test_input_resampler_keeps_pending_input_when_output_scratch_fills() {
        let mut resampler = build_sinc_resampler(44_100, TARGET_SAMPLE_RATE, 1024).unwrap();
        let mut resample_input =
            crate::audio::rt::FixedAudioRing::<f64, RT_RESAMPLE_QUEUE_CAPACITY>::new();
        let mut input_frame = [0.0_f64; 1024];
        let mut outbuf = resampler.output_buffer_allocate(true);
        let mut scratch = FixedAudioBuffer::<f32, 2048>::new();

        for i in 0..RT_PROCESS_BUFFER_CAPACITY {
            assert!(resample_input.push((i as f64 * 0.001).sin()));
        }

        let input_frames_needed = resampler.input_frames_next();
        while resample_input.len() >= input_frames_needed && input_frames_needed <= input_frame.len()
        {
            if !has_resampler_output_capacity(&scratch, &outbuf) {
                break;
            }
            assert_eq!(
                resample_input.pop_into(&mut input_frame[..input_frames_needed]),
                input_frames_needed
            );
            let in_slices = [&input_frame[..input_frames_needed]];
            let (_nbr_in, nbr_out) = resampler
                .process_into_buffer(&in_slices, &mut outbuf, None)
                .unwrap();
            for &sample in outbuf[0].iter().take(nbr_out) {
                assert!(scratch.push(sample as f32));
            }
        }

        assert!(scratch.remaining() < outbuf[0].len());
        assert!(resample_input.len() >= input_frames_needed);
    }

    #[test]
    fn test_output_resampler_drains_large_block_in_bounded_chunks() {
        let mut resampler = build_sinc_resampler(TARGET_SAMPLE_RATE, 96_000, 1024).unwrap();
        let mut resample_input =
            crate::audio::rt::FixedAudioRing::<f64, RT_RESAMPLE_QUEUE_CAPACITY>::new();
        let mut input_frame = [0.0_f64; 1024];
        let mut outbuf = resampler.output_buffer_allocate(true);
        let mut scratch = FixedAudioBuffer::<f32, RT_OUTPUT_SCRATCH_CAPACITY>::new();

        for i in 0..RT_PROCESS_BUFFER_CAPACITY {
            assert!(resample_input.push((i as f64 * 0.001).sin()));
        }

        let mut chunks = 0usize;
        let mut total_generated = 0usize;
        loop {
            scratch.clear();
            let input_frames_needed = resampler.input_frames_next();
            while resample_input.len() >= input_frames_needed
                && input_frames_needed <= input_frame.len()
            {
                if !has_resampler_output_capacity(&scratch, &outbuf) {
                    break;
                }
                assert_eq!(
                    resample_input.pop_into(&mut input_frame[..input_frames_needed]),
                    input_frames_needed
                );
                let in_slices = [&input_frame[..input_frames_needed]];
                let (_nbr_in, nbr_out) = resampler
                    .process_into_buffer(&in_slices, &mut outbuf, None)
                    .unwrap();
                for &sample in outbuf[0].iter().take(nbr_out) {
                    assert!(scratch.push(sample as f32));
                }
            }

            if scratch.is_empty() {
                break;
            }
            chunks += 1;
            total_generated += scratch.len();
            if resample_input.len() < input_frames_needed {
                break;
            }
        }

        assert!(chunks > 1);
        assert!(total_generated > RT_OUTPUT_SCRATCH_CAPACITY);
        assert!(resample_input.len() < resampler.input_frames_next());
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

        let control = processor.gate_rt_control.snapshot().expect("stable gate state");
        assert_eq!(control.threshold_db, -40.0);
        assert_eq!(control.attack_ms, 10.0);
        assert_eq!(control.release_ms, 100.0);

        processor.set_gate_threshold(-120.0);
        processor.set_gate_attack(0.01);
        processor.set_gate_release(5000.0);

        let control = processor.gate_rt_control.snapshot().expect("stable gate state");
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
        assert_eq!(processor.get_deesser_high_cut_hz(), 11_000.0);
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
    fn test_compressor_runtime_meter_getters_read_rt_atomically() {
        let processor = AudioProcessor::new();

        assert!(processor.get_compressor_sidechain_highpass_enabled());
        processor.set_compressor_sidechain_highpass_enabled(false);
        assert!(!processor.get_compressor_sidechain_highpass_enabled());
        processor.set_compressor_sidechain_highpass_enabled(true);
        assert!(processor.get_compressor_sidechain_highpass_enabled());

        processor
            .compressor_current_lufs
            .store((-23.5_f64).to_bits(), Ordering::Relaxed);
        processor
            .compressor_current_makeup_gain
            .store(4.25_f64.to_bits(), Ordering::Relaxed);

        assert_eq!(processor.get_compressor_current_lufs(), -23.5);
        assert_eq!(processor.get_compressor_current_makeup_gain(), 4.25);

        processor.set_compressor_makeup_gain(6.0);
        assert_eq!(processor.get_compressor_current_makeup_gain(), 6.0);
    }

    #[test]
    fn test_apply_eq_settings_rejects_above_nyquist() {
        let processor = AudioProcessor::new();
        let mut bands = vec![(100.0, 0.0, 1.0); NUM_BANDS];
        bands[NUM_BANDS - 1] = (processor.eq_nyquist_limit_hz() + 100.0, 0.0, 1.0);

        Python::initialize();
        let err = processor.apply_eq_settings(bands).unwrap_err();
        assert!(err.to_string().contains("out of range"));
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

    fn marked_region<'a>(source: &'a str, name: &str) -> &'a str {
        let start_marker = format!("RT_REGION_START: {name}");
        let end_marker = format!("RT_REGION_END: {name}");
        let start = source
            .find(&start_marker)
            .unwrap_or_else(|| panic!("missing {start_marker}"));
        let body_start = source[start..]
            .find('\n')
            .map(|offset| start + offset + 1)
            .unwrap_or(start);
        let end = source[body_start..]
            .find(&end_marker)
            .map(|offset| body_start + offset)
            .unwrap_or_else(|| panic!("missing {end_marker}"));
        &source[body_start..end]
    }

    fn source_between<'a>(source: &'a str, start_marker: &str, end_marker: &str) -> &'a str {
        let start = source
            .find(start_marker)
            .unwrap_or_else(|| panic!("missing {start_marker}"));
        let end = source[start..]
            .find(end_marker)
            .map(|offset| start + offset)
            .unwrap_or_else(|| panic!("missing {end_marker}"));
        &source[start..end]
    }

    fn assert_no_forbidden_rt_patterns(label: &str, region: &str) {
        let forbidden = [
            ".lock()",
            ".try_lock()",
            "try_lock",
            ".reserve(",
            ".resize(",
            ".to_vec(",
            ".drain(",
            "drop(",
            "format!",
            "println!",
            "eprintln!",
            "processor_debug_log!",
            "std::mem::forget",
        ];

        for pattern in forbidden {
            assert!(
                !region.contains(pattern),
                "{label} contains forbidden RT pattern {pattern}"
            );
        }
    }

    #[test]
    fn test_marked_rt_regions_reject_blocking_or_allocating_apis() {
        let files = [
            (
                "processor/dsp_loop.rs",
                include_str!("dsp_loop.rs"),
                &["dsp_processing_loop"][..],
            ),
            (
                "input.rs",
                include_str!("../input.rs"),
                &["cpal_input_callback"][..],
            ),
            (
                "output.rs",
                include_str!("../output.rs"),
                &["cpal_output_callback"][..],
            ),
        ];

        for (file, source, regions) in files {
            for region_name in regions {
                let region = marked_region(source, region_name);
                assert_no_forbidden_rt_patterns(&format!("{file}:{region_name}"), region);
            }
        }
    }

    #[test]
    fn test_downstream_rt_macro_rejects_blocking_or_allocating_apis() {
        let source = include_str!("dsp_loop.rs");
        let region = source_between(
            source,
            "macro_rules! apply_downstream_chain_rt",
            "// Time-based EMA smoothing",
        );

        assert_no_forbidden_rt_patterns("processor/dsp_loop.rs:apply_downstream_chain_rt", region);
    }

    #[test]
    fn test_dsp_loop_uses_caller_provided_suppressor_output_buffers() {
        let source = include_str!("dsp_loop.rs");
        let region = marked_region(source, "dsp_processing_loop");
        for pattern in ["pop_samples(", "pop_all_samples(", "drain_pending_input("] {
            assert!(
                !region.contains(pattern),
                "DSP RT loop must not call Vec-returning suppressor API {pattern}"
            );
        }
        assert!(region.contains("pop_samples_into("));
    }

    #[test]
    fn test_dsp_loop_diagnoses_suppressor_short_writes() {
        let source = include_str!("dsp_loop.rs");
        let region = marked_region(source, "dsp_processing_loop");

        assert!(region.contains("let accepted = suppressor_rt.push_samples(buffer);"));
        assert!(region.contains("if accepted < buffer.len()"));
        assert!(region.contains("RtErrorCode::FixedBufferOverflow"));
    }

    #[test]
    fn test_dsp_loop_uses_swap_dirty_flags_to_preserve_racing_updates() {
        let source = include_str!("dsp_loop.rs");

        for dirty_flag in [
            "gate_dirty",
            "suppressor_dirty",
            "eq_dirty",
            "deesser_dirty",
            "compressor_dirty",
            "limiter_dirty",
        ] {
            assert!(
                source.contains(&format!("{dirty_flag}.swap(false, Ordering::AcqRel)")),
                "{dirty_flag} must be consumed with swap(false) in the RT loop"
            );
            assert!(
                !source.contains(&format!("{dirty_flag}.load(Ordering::Acquire)")),
                "{dirty_flag} must not use load/apply/store(false) in the RT loop"
            );
        }
    }

    #[test]
    fn test_dsp_loop_rearms_dirty_flags_on_unstable_snapshots() {
        let source = include_str!("dsp_loop.rs");

        for dirty_flag in [
            "gate_dirty",
            "suppressor_dirty",
            "eq_dirty",
            "deesser_dirty",
            "compressor_dirty",
            "limiter_dirty",
        ] {
            assert!(
                source.contains(&format!("{dirty_flag}.store(true, Ordering::Release)")),
                "{dirty_flag} must be re-armed when a control snapshot is unstable"
            );
        }
    }

    #[cfg(feature = "deepfilter")]
    #[test]
    fn test_deepfilter_rt_failure_path_does_not_allocate_error_strings() {
        let source = include_str!("../../dsp/deepfilter_ffi.rs");
        let start = source
            .find("fn process_into(")
            .expect("DeepFilter process_into must exist");
        let end = source[start..]
            .find("    /// Set attenuation")
            .map(|offset| start + offset)
            .expect("DeepFilter process_into end marker must exist");
        let region = &source[start..end];

        for pattern in ["format!", ".to_string()", "String"] {
            assert!(
                !region.contains(pattern),
                "DeepFilter RT process error path must not allocate via {pattern}"
            );
        }
    }

    #[test]
    fn test_dsp_loop_defers_suppressor_drops_out_of_rt_region() {
        let source = include_str!("dsp_loop.rs");
        let region = marked_region(source, "dsp_processing_loop");

        assert!(region.contains("retired_suppressor_tx"));
        assert!(region.contains("deferred_suppressor_retire"));
        assert!(!region.contains("suppressor_rt = candidate"));
        assert!(!region.contains("std::mem::forget"));
    }

    #[cfg(feature = "vad")]
    #[test]
    fn test_dsp_loop_diagnoses_vad_worker_short_writes() {
        let source = include_str!("dsp_loop.rs");
        let region = marked_region(source, "dsp_processing_loop");

        assert!(region.contains("let written = vad_worker_producer.write(buffer);"));
        assert!(region.contains("if written < buffer.len()"));
        assert!(region.contains("store_rt_error("));
        assert!(region.contains("RtErrorCode::FixedBufferOverflow"));
    }

    #[test]
    fn test_runtime_diagnostics_split_retime_adjustments_from_recovery_events() {
        let wrapper = PyAudioProcessor {
            processor: AudioProcessor::new(),
        };
        wrapper
            .processor
            .output_retime_adjustment_count
            .store(11, Ordering::Relaxed);
        wrapper
            .processor
            .output_recovery_event_count
            .store(7, Ordering::Relaxed);
        wrapper
            .processor
            .output_underrun_streak
            .store(3, Ordering::Relaxed);
        wrapper
            .processor
            .output_clip_event_count
            .store(2, Ordering::Relaxed);
        wrapper
            .processor
            .output_clip_peak_db
            .store((-0.75_f32).to_bits(), Ordering::Relaxed);
        wrapper
            .processor
            .output_true_peak_event_count
            .store(3, Ordering::Relaxed);
        wrapper
            .processor
            .output_true_peak_db
            .store(0.5_f32.to_bits(), Ordering::Relaxed);
        wrapper
            .processor
            .gate_chatter_event_count
            .store(4, Ordering::Relaxed);
        wrapper
            .processor
            .deesser_detector_confidence
            .store(0.62_f32.to_bits(), Ordering::Relaxed);
        wrapper.processor.input_phase_rescue_strategy.store(
            crate::audio::input::PhaseRescueStrategy::FractionalDelay as u8,
            Ordering::Relaxed,
        );
        wrapper
            .processor
            .input_phase_estimated_delay_samples
            .store(2.25_f32.to_bits(), Ordering::Relaxed);
        wrapper
            .processor
            .input_phase_polarity_flipped
            .store(true, Ordering::Relaxed);
        wrapper
            .processor
            .set_limiter_careful_output_enabled(false);
        wrapper.processor.set_limiter_ceiling(-0.25);

        Python::initialize();
        Python::attach(|py| {
            let diagnostics = wrapper.get_runtime_diagnostics(py).unwrap();
            let diagnostics = diagnostics.bind(py);
            let diagnostics = diagnostics.cast::<PyDict>().unwrap();
            assert_eq!(
                diagnostics
                    .get_item("output_retime_adjustment_count")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                11
            );
            assert_eq!(
                diagnostics
                    .get_item("output_recovery_event_count")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                7
            );
            assert_eq!(
                diagnostics
                    .get_item("output_recovery_count")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                7
            );
            assert_eq!(
                diagnostics
                    .get_item("output_underrun_streak")
                    .unwrap()
                    .unwrap()
                    .extract::<u32>()
                    .unwrap(),
                3
            );
            assert_eq!(
                diagnostics
                    .get_item("output_clip_event_count")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                2
            );
            assert!(
                (diagnostics
                    .get_item("output_clip_peak_db")
                    .unwrap()
                    .unwrap()
                    .extract::<f32>()
                    .unwrap()
                    + 0.75)
                    .abs()
                    < 1e-6
            );
            assert_eq!(
                diagnostics
                    .get_item("output_true_peak_event_count")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                3
            );
            assert!(
                (diagnostics
                    .get_item("output_true_peak_db")
                    .unwrap()
                    .unwrap()
                    .extract::<f32>()
                    .unwrap()
                    - 0.5)
                    .abs()
                    < 1e-6
            );
            assert!(
                !diagnostics
                    .get_item("limiter_careful_output_enabled")
                    .unwrap()
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            );
            assert!(
                (diagnostics
                    .get_item("limiter_effective_ceiling_db")
                    .unwrap()
                    .unwrap()
                    .extract::<f64>()
                    .unwrap()
                    + 0.25)
                    .abs()
                    < 1e-6
            );
            assert_eq!(
                diagnostics
                    .get_item("gate_chatter_event_count")
                    .unwrap()
                    .unwrap()
                    .extract::<u64>()
                    .unwrap(),
                4
            );
            assert!(
                (diagnostics
                    .get_item("deesser_detector_confidence")
                    .unwrap()
                    .unwrap()
                    .extract::<f32>()
                    .unwrap()
                    - 0.62)
                    .abs()
                    < 1e-6
            );
            assert_eq!(
                diagnostics
                    .get_item("input_phase_rescue_strategy")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "fractional_delay"
            );
            assert!(
                (diagnostics
                    .get_item("input_phase_estimated_delay_samples")
                    .unwrap()
                    .unwrap()
                    .extract::<f32>()
                    .unwrap()
                    - 2.25)
                    .abs()
                    < 1e-6
            );
            assert!(
                diagnostics
                    .get_item("input_phase_polarity_flipped")
                    .unwrap()
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            );
        });
    }

    #[cfg(feature = "vad")]
    #[test]
    fn test_runtime_diagnostics_include_gate_fused_score() {
        let wrapper = PyAudioProcessor {
            processor: AudioProcessor::new(),
        };
        wrapper
            .processor
            .gate_fused_score
            .store(0.42_f32.to_bits(), Ordering::Relaxed);
        wrapper
            .processor
            .gate_auto_relax_active
            .store(true, Ordering::Relaxed);

        Python::initialize();
        Python::attach(|py| {
            let diagnostics = wrapper.get_runtime_diagnostics(py).unwrap();
            let diagnostics = diagnostics.bind(py);
            let diagnostics = diagnostics.cast::<PyDict>().unwrap();
            let score = diagnostics
                .get_item("gate_fused_score")
                .unwrap()
                .unwrap()
                .extract::<f32>()
                .unwrap();
            assert!((score - 0.42).abs() < 1e-6);
            assert!(
                diagnostics
                    .get_item("gate_auto_relax_active")
                    .unwrap()
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            );
        });
    }
}
