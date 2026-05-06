#[cfg(test)]
mod tests {
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
    fn test_linear_resample_into_identity_preserves_samples() {
        let input = vec![0.0, 0.25, -0.5, 1.0];
        let mut output = Vec::new();

        linear_resample_into(&input, 1.0, &mut output);

        assert_eq!(output, input);
    }

    #[test]
    fn test_linear_resample_into_downsamples_to_expected_length() {
        let input = vec![1.0f32; 1536];
        let mut output = Vec::new();

        linear_resample_into(&input, SILERO_SAMPLE_RATE as f32 / 48_000.0, &mut output);

        assert_eq!(output.len(), SILERO_WINDOW_SIZE);
        assert!(output.iter().all(|sample| (*sample - 1.0).abs() < 1e-6));
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
