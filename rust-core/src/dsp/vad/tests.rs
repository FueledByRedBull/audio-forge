#[cfg(test)]
mod tests {
    #![allow(clippy::excessive_precision)] // SciPy golden vectors are copied verbatim.

    use super::*;

    #[test]
    fn test_rms_computation() {
        let silence = vec![0.0; 1000];
        assert!(compute_rms_db(&silence) < -100.0);

        let signal = vec![1.0; 1000];
        assert!((compute_rms_db(&signal) - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_gate_mode_enum() {
        assert_ne!(GateMode::ThresholdOnly, GateMode::VadAssisted);
        assert_ne!(GateMode::VadAssisted, GateMode::VadOnly);
    }

    #[test]
    fn test_hold_time_persists_gate_after_speech_drop() {
        let mut gate = VadAutoGate::new(48000, 0.5);
        gate.set_gate_mode(GateMode::VadOnly);
        gate.set_hold_time(100.0);

        let frame = vec![0.01f32; 480]; // 10ms at 48kHz

        // Open gate with strong speech probability.
        let (open, _) = gate.process_with_probability(&frame, 0.9);
        assert!(open);

        // Hold should keep gate open for ~100ms after raw close.
        for _ in 0..5 {
            let (held_open, _) = gate.process_with_probability(&frame, 0.0);
            assert!(held_open);
        }

        let mut final_state = true;
        for _ in 0..6 {
            let (state, _) = gate.process_with_probability(&frame, 0.0);
            final_state = state;
        }
        assert!(!final_state);
    }

    #[test]
    fn test_debounce_blocks_short_reopen_glitch() {
        let mut gate = VadAutoGate::new(48000, 0.5);
        gate.set_gate_mode(GateMode::VadOnly);
        gate.set_hold_time(0.0); // isolate debounce behavior

        let frame = vec![0.01f32; 480]; // 10ms

        // Open, then close.
        assert!(gate.process_with_probability(&frame, 0.9).0);
        assert!(!gate.process_with_probability(&frame, 0.0).0);

        // Short reopen should be blocked by 50ms debounce.
        assert!(!gate.process_with_probability(&frame, 0.9).0);

        // After enough closed time, reopen should be allowed.
        for _ in 0..5 {
            let _ = gate.process_with_probability(&frame, 0.0);
        }
        assert!(gate.process_with_probability(&frame, 0.9).0);
    }

    #[test]
    fn test_auto_threshold_adapts_toward_background_level() {
        let mut gate = VadAutoGate::new(48000, 0.5);
        gate.set_gate_mode(GateMode::VadAssisted);
        gate.set_auto_threshold(true);

        let initial_floor = gate.noise_floor();

        // Constant background around -45 dBFS.
        let amp = 10f32.powf(-45.0 / 20.0);
        let frame = vec![amp; 480]; // 10ms

        for _ in 0..250 {
            let _ = gate.process_with_probability(&frame, 0.1);
        }

        assert!(gate.noise_floor() > initial_floor + 4.0);
        assert!(gate.noise_floor() < -40.0);
    }

    #[test]
    fn test_auto_threshold_ignores_high_confidence_speech_frames() {
        let mut gate = VadAutoGate::new(48_000, 0.5);
        gate.set_gate_mode(GateMode::VadAssisted);
        gate.set_auto_threshold(true);
        let initial_floor = gate.noise_floor();

        // Loud speech-like frames should be ignored for floor adaptation.
        let amp = 10f32.powf(-25.0 / 20.0);
        let frame = vec![amp; 480];
        for _ in 0..300 {
            let _ = gate.process_with_probability(&frame, 0.9);
        }

        assert!(
            (gate.noise_floor() - initial_floor).abs() < 0.25,
            "high-confidence speech should not pollute noise floor"
        );
    }

    #[test]
    fn test_auto_threshold_slew_limits_per_frame() {
        let mut gate = VadAutoGate::new(48_000, 0.5);
        gate.set_gate_mode(GateMode::VadAssisted);
        gate.set_auto_threshold(true);

        let quiet_amp = 10f32.powf(-70.0 / 20.0);
        let loud_amp = 10f32.powf(-35.0 / 20.0);
        let quiet = vec![quiet_amp; 480];
        let loud = vec![loud_amp; 480];

        // Warm history with quiet non-speech.
        for _ in 0..NOISE_FLOOR_HISTORY_FRAMES {
            let _ = gate.process_with_probability(&quiet, 0.1);
        }
        let before_rise = gate.noise_floor();
        let _ = gate.process_with_probability(&loud, 0.1);
        let after_rise = gate.noise_floor();
        assert!(
            after_rise - before_rise <= NOISE_FLOOR_UP_SLEW_DB_PER_FRAME + 1e-6,
            "rise slew exceeded per-frame limit"
        );

        // Warm history with louder non-speech, then check downward slew.
        gate.reset();
        gate.set_auto_threshold(true);
        for _ in 0..NOISE_FLOOR_HISTORY_FRAMES {
            let _ = gate.process_with_probability(&loud, 0.1);
        }
        let before_fall = gate.noise_floor();
        let _ = gate.process_with_probability(&quiet, 0.1);
        let after_fall = gate.noise_floor();
        assert!(
            before_fall - after_fall <= NOISE_FLOOR_DOWN_SLEW_DB_PER_FRAME + 1e-6,
            "fall slew exceeded per-frame limit"
        );
    }

    #[test]
    fn test_anti_aliased_resample_into_identity_preserves_samples() {
        let input = vec![0.0, 0.25, -0.5, 1.0];
        let mut output = Vec::new();

        anti_aliased_resample_into(&input, 1.0, &mut output);

        assert_eq!(output, input);
    }

    #[test]
    fn test_anti_aliased_resample_into_downsamples_to_expected_length() {
        let input = vec![1.0f32; 1536];
        let mut output = Vec::new();

        anti_aliased_resample_into(&input, SILERO_SAMPLE_RATE as f32 / 48_000.0, &mut output);

        assert_eq!(output.len(), SILERO_WINDOW_SIZE);
        assert!(output.iter().all(|sample| (*sample - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_anti_aliased_resample_rejects_high_frequency_aliasing() {
        let sample_rate = 48_000.0_f32;
        let ratio = SILERO_SAMPLE_RATE as f32 / sample_rate;
        let mut high = vec![0.0_f32; 1536];
        let mut voice = vec![0.0_f32; 1536];
        for idx in 0..1536 {
            let t = idx as f32 / sample_rate;
            high[idx] = (2.0 * std::f32::consts::PI * 12_000.0 * t).sin();
            voice[idx] = (2.0 * std::f32::consts::PI * 1_000.0 * t).sin();
        }

        let mut high_out = Vec::new();
        let mut voice_out = Vec::new();
        anti_aliased_resample_into(&high, ratio, &mut high_out);
        anti_aliased_resample_into(&voice, ratio, &mut voice_out);

        let rms = |samples: &[f32]| -> f32 {
            (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32)
                .sqrt()
        };
        let high_rms = rms(&high_out[32..high_out.len() - 32]);
        let voice_rms = rms(&voice_out[32..voice_out.len() - 32]);

        assert_eq!(high_out.len(), SILERO_WINDOW_SIZE);
        assert_eq!(voice_out.len(), SILERO_WINDOW_SIZE);
        assert!(
            high_rms < voice_rms * 0.08,
            "high_rms={high_rms} voice_rms={voice_rms}"
        );
    }

    const GOLDEN_INDICES: [usize; 15] = [
        32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480,
    ];
    // Generated with scipy.signal.resample_poly(input, 16000, input_rate,
    // window=("kaiser", 5.0)). The sparse vectors stay reviewable while
    // covering the complete 32 ms Silero window.
    const SCIPY_SPEECH_48K: [f32; 15] = [
        0.230655119,
        -0.130217150,
        0.222004876,
        -0.501845241,
        0.438263059,
        -0.354755819,
        0.575726748,
        -0.572377145,
        0.344859362,
        -0.422269702,
        0.480481923,
        -0.196243078,
        0.101228043,
        -0.199756607,
        -0.031403121,
    ];
    const SCIPY_SPEECH_44K1: [f32; 15] = [
        0.230644554,
        -0.130219638,
        0.222025439,
        -0.501842320,
        0.438263506,
        -0.354771644,
        0.575753927,
        -0.572403729,
        0.344874859,
        -0.422270089,
        0.480479032,
        -0.196262598,
        0.101228848,
        -0.199745625,
        -0.031403158,
    ];
    const SCIPY_NOISE_48K: [f32; 15] = [
        -0.071705781,
        -0.088858202,
        -0.167572007,
        -0.222385511,
        0.008212528,
        -0.007762028,
        -0.401018500,
        -0.032284141,
        0.061968692,
        -0.021019392,
        0.256368130,
        0.136872485,
        0.286439091,
        0.330785215,
        -0.129022315,
    ];
    const SCIPY_NOISE_44K1: [f32; 15] = [
        -0.306934148,
        -0.399872720,
        0.231722042,
        0.042804722,
        -0.146513596,
        0.042475406,
        0.043586224,
        -0.038887408,
        0.097171865,
        -0.206554338,
        0.337191194,
        0.286520392,
        -0.111895755,
        0.178830177,
        -0.271840066,
    ];

    fn scipy_fixture(sample_rate: u32, high_frequency_noise: bool) -> Vec<f32> {
        let input_len = (SILERO_WINDOW_SIZE as f64 * sample_rate as f64
            / SILERO_SAMPLE_RATE as f64)
            .ceil() as usize;
        if high_frequency_noise {
            let mut state = 0xA17D_5EED_u32;
            return (0..input_len)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    if state & 0x8000_0000 != 0 { 0.4 } else { -0.4 }
                })
                .collect();
        }

        (0..input_len)
            .map(|index| {
                let time = index as f64 / sample_rate as f64;
                (0.5 * (2.0 * std::f64::consts::PI * 233.0 * time).sin()
                    + 0.2 * (2.0 * std::f64::consts::PI * 3100.0 * time).sin()
                    + 0.07 * (2.0 * std::f64::consts::PI * 6900.0 * time).sin())
                    as f32
            })
            .collect()
    }

    fn assert_matches_scipy_golden(
        sample_rate: u32,
        high_frequency_noise: bool,
        expected: &[f32; 15],
        max_error: f32,
    ) {
        let input = scipy_fixture(sample_rate, high_frequency_noise);
        let mut output = Vec::new();
        anti_aliased_resample_into(
            &input,
            SILERO_SAMPLE_RATE as f32 / sample_rate as f32,
            &mut output,
        );
        assert!(output.len() >= SILERO_WINDOW_SIZE);
        for (index, expected) in GOLDEN_INDICES.iter().zip(expected.iter()) {
            let error = (output[*index] - expected).abs();
            assert!(
                error <= max_error,
                "sample_rate={sample_rate} noise={high_frequency_noise} index={index} actual={} expected={expected} error={error}",
                output[*index]
            );
        }
    }

    #[test]
    fn test_resampler_matches_scipy_polyphase_speech_golden_vectors() {
        assert_matches_scipy_golden(48_000, false, &SCIPY_SPEECH_48K, 0.013);
        assert_matches_scipy_golden(44_100, false, &SCIPY_SPEECH_44K1, 0.013);
    }

    #[test]
    fn test_resampler_matches_scipy_polyphase_noise_golden_vectors() {
        assert_matches_scipy_golden(48_000, true, &SCIPY_NOISE_48K, 0.06);
        assert_matches_scipy_golden(44_100, true, &SCIPY_NOISE_44K1, 0.06);
    }

    #[test]
    fn test_model_optional_vad_buffer_compacts_after_threshold() {
        let Some(mut vad) = optional_silero_vad(48_000, 0.5) else {
            return;
        };

        let frame = vec![0.0f32; 480];
        for _ in 0..20 {
            let _ = vad.process(&frame);
        }

        assert!(vad.buffer_read_pos < VAD_INPUT_COMPACT_THRESHOLD);
        assert!(vad.buffer.len() >= vad.buffer_read_pos);
        assert!(vad.buffer.len() < vad.window_size() * 4);
    }

    #[test]
    fn test_model_optional_vad_resampled_path_reuses_fixed_output_window() {
        let Some(mut vad) = optional_silero_vad(48_000, 0.5) else {
            return;
        };

        let frame = vec![0.0f32; vad.window_size()];
        let _ = vad.process(&frame);

        assert_eq!(vad.audio_512.len(), SILERO_WINDOW_SIZE);
        assert_eq!(vad.resample_scratch.len(), SILERO_WINDOW_SIZE);
    }

    #[test]
    fn test_model_optional_vad_returns_neutral_probability_before_first_inference() {
        let Some(mut vad) = optional_silero_vad(48_000, 0.42) else {
            return;
        };

        let partial = vec![0.0_f32; vad.window_size().saturating_sub(1)];
        let prob = vad.process(&partial).unwrap();

        assert!((prob - 0.42).abs() < 1e-6);
        assert!((vad.probability() - 0.42).abs() < 1e-6);
    }

    fn optional_silero_vad(sample_rate: u32, threshold: f32) -> Option<SileroVAD> {
        match SileroVAD::new(sample_rate, threshold) {
            Ok(vad) => Some(vad),
            Err(error) => {
                eprintln!("Skipping model-optional Silero VAD coverage: {error}");
                None
            }
        }
    }
}
