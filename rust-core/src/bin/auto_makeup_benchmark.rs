use mic_eq_core::dsp::{AutoMakeupActivityInput, Compressor};

const SAMPLE_RATE: f64 = 48_000.0;
const BLOCK_SIZE: usize = 480;

fn compressor() -> Compressor {
    let mut compressor = Compressor::new(-24.0, 3.0, 10.0, 180.0, 0.0, 6.0, SAMPLE_RATE);
    compressor.set_auto_makeup_enabled(true);
    compressor.set_target_lufs(-18.0);
    compressor.set_noise_reference_reliability(1.0);
    compressor
}

fn tone_block(amplitude: f32, frequency_hz: f64, phase: &mut f64) -> [f32; BLOCK_SIZE] {
    let phase_step = 2.0 * std::f64::consts::PI * frequency_hz / SAMPLE_RATE;
    std::array::from_fn(|_| {
        let sample = amplitude * phase.sin() as f32;
        *phase = (*phase + phase_step) % (2.0 * std::f64::consts::PI);
        sample
    })
}

fn noise_block(amplitude: f32, state: &mut u64) -> [f32; BLOCK_SIZE] {
    std::array::from_fn(|_| {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = ((*state >> 40) as u32) as f32 / ((1_u32 << 24) - 1) as f32;
        (2.0 * unit - 1.0) * amplitude
    })
}

fn speech_evidence() -> AutoMakeupActivityInput {
    AutoMakeupActivityInput {
        vad_probability: 0.96,
        vad_reliability: 1.0,
        noise_floor_db: -46.0,
        live_noise_reliability: 1.0,
    }
}

fn noise_evidence() -> AutoMakeupActivityInput {
    AutoMakeupActivityInput {
        vad_probability: 0.01,
        vad_reliability: 1.0,
        noise_floor_db: -38.0,
        live_noise_reliability: 1.0,
    }
}

fn stale_evidence() -> AutoMakeupActivityInput {
    AutoMakeupActivityInput {
        vad_probability: 0.96,
        vad_reliability: 0.0,
        noise_floor_db: -46.0,
        live_noise_reliability: 0.0,
    }
}

fn run_noise_false_activation(candidate: bool) -> f64 {
    let mut compressor = compressor();
    let mut noise_state = 0x6a09_e667_f3bc_c909;
    for _ in 0..800 {
        let mut block = noise_block(0.022, &mut noise_state);
        compressor.process_block_inplace_with_activity_control(
            &mut block,
            candidate.then_some(noise_evidence()),
        );
    }
    compressor.current_makeup_gain()
}

fn run_speech_convergence(candidate: bool) -> f64 {
    let mut compressor = compressor();
    let mut phase = 0.0;
    for _ in 0..800 {
        let mut block = tone_block(0.035, 187.0, &mut phase);
        compressor.process_block_inplace_with_activity_control(
            &mut block,
            candidate.then_some(speech_evidence()),
        );
    }
    compressor.current_makeup_gain()
}

fn run_silence_relaxation() -> (f64, f64) {
    let mut compressor = compressor();
    let mut phase = 0.0;
    for _ in 0..800 {
        let mut block = tone_block(0.035, 187.0, &mut phase);
        compressor.process_block_inplace_with_activity_control(&mut block, Some(speech_evidence()));
    }
    let speech_gain = compressor.current_makeup_gain();
    let mut maximum_silence_gain = 0.0_f64;
    for _ in 0..500 {
        let mut block = [0.0_f32; BLOCK_SIZE];
        compressor.process_block_inplace_with_activity_control(&mut block, Some(noise_evidence()));
        maximum_silence_gain = maximum_silence_gain.max(compressor.current_makeup_gain());
    }
    (
        maximum_silence_gain,
        speech_gain - compressor.current_makeup_gain(),
    )
}

fn run_stale_relaxation() -> (f64, f64) {
    let mut compressor = compressor();
    compressor.set_noise_reference_reliability(0.0);
    let mut phase = 0.0;
    for _ in 0..800 {
        let mut block = tone_block(0.035, 187.0, &mut phase);
        compressor.process_block_inplace_with_activity_control(&mut block, Some(speech_evidence()));
    }
    let fresh_gain = compressor.current_makeup_gain();
    for _ in 0..300 {
        let mut block = tone_block(0.008, 187.0, &mut phase);
        compressor.process_block_inplace_with_activity_control(&mut block, Some(stale_evidence()));
    }
    (fresh_gain, compressor.current_makeup_gain())
}

fn standard_deviation(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

fn run_pumping(candidate: bool) -> f64 {
    let mut compressor = compressor();
    let mut phase = 0.0;
    let mut noise_state = 0xbb67_ae85_84ca_a73b;
    let mut history = Vec::with_capacity(1600);
    for cycle in 0..20 {
        for _ in 0..40 {
            let mut block = tone_block(0.035, 187.0, &mut phase);
            compressor.process_block_inplace_with_activity_control(
                &mut block,
                candidate.then_some(speech_evidence()),
            );
            if cycle >= 4 {
                history.push(compressor.current_makeup_gain());
            }
        }
        for _ in 0..40 {
            let mut block = noise_block(0.012, &mut noise_state);
            compressor.process_block_inplace_with_activity_control(
                &mut block,
                candidate.then_some(noise_evidence()),
            );
            if cycle >= 4 {
                history.push(compressor.current_makeup_gain());
            }
        }
    }
    standard_deviation(&history)
}

fn run_transition_click() -> f64 {
    let mut compressor = compressor();
    let mut maximum_boundary_jump = 0.0_f64;
    let mut previous_last = 0.0_f32;
    for block_index in 0..600 {
        let mut block = [0.02_f32; BLOCK_SIZE];
        let evidence = if (block_index / 30) % 2 == 0 {
            speech_evidence()
        } else {
            noise_evidence()
        };
        compressor.process_block_inplace_with_activity_control(&mut block, Some(evidence));
        if block_index > 0 {
            maximum_boundary_jump =
                maximum_boundary_jump.max((block[0] - previous_last).abs() as f64);
        }
        previous_last = block[BLOCK_SIZE - 1];
    }
    maximum_boundary_jump
}

fn main() {
    let baseline_noise_gain = run_noise_false_activation(false);
    let candidate_noise_gain = run_noise_false_activation(true);
    let baseline_speech_gain = run_speech_convergence(false);
    let candidate_speech_gain = run_speech_convergence(true);
    let (maximum_silence_gain, silence_relaxation_db) = run_silence_relaxation();
    let (fresh_gain, stale_gain) = run_stale_relaxation();
    let baseline_pumping = run_pumping(false);
    let candidate_pumping = run_pumping(true);
    let maximum_transition_jump = run_transition_click();

    let false_activation_passed =
        candidate_noise_gain <= 0.25 && candidate_noise_gain + 1.0 < baseline_noise_gain;
    let speech_convergence_passed =
        candidate_speech_gain >= 0.5 && candidate_speech_gain + 1.0 >= baseline_speech_gain;
    let silence_passed = maximum_silence_gain.is_finite() && silence_relaxation_db >= 0.5;
    let stale_passed = stale_gain <= fresh_gain - 0.5;
    let pumping_passed = candidate_pumping <= baseline_pumping * 1.05;
    let click_passed = maximum_transition_jump <= 0.01;
    let retained = false_activation_passed
        && speech_convergence_passed
        && silence_passed
        && stale_passed
        && pumping_passed
        && click_passed;

    println!(
        concat!(
            "{{\n",
            "  \"schema_version\": 1,\n",
            "  \"experiment\": \"VAD/noise-reliability auto-makeup versus RMS-only fallback\",\n",
            "  \"retained\": {},\n",
            "  \"metrics\": {{\n",
            "    \"noise_false_activation\": {{\"baseline_gain_db\": {:.9}, \"candidate_gain_db\": {:.9}, \"passed\": {}}},\n",
            "    \"speech_convergence\": {{\"baseline_gain_db\": {:.9}, \"candidate_gain_db\": {:.9}, \"passed\": {}}},\n",
            "    \"silence_relaxation\": {{\"maximum_gain_db\": {:.9}, \"relaxation_db\": {:.9}, \"passed\": {}}},\n",
            "    \"stale_vad\": {{\"fresh_gain_db\": {:.9}, \"stale_gain_db\": {:.9}, \"passed\": {}}},\n",
            "    \"pumping\": {{\"baseline_makeup_std_db\": {:.9}, \"candidate_makeup_std_db\": {:.9}, \"passed\": {}}},\n",
            "    \"transition_click\": {{\"maximum_boundary_jump_linear\": {:.9}, \"passed\": {}}}\n",
            "  }},\n",
            "  \"limitations\": [\n",
            "    \"Deterministic signal/control benchmark; long-form perceptual listening remains required.\",\n",
            "    \"Realtime allocation safety is enforced by a separate allocator-instrumented Rust test.\"\n",
            "  ]\n",
            "}}"
        ),
        retained,
        baseline_noise_gain,
        candidate_noise_gain,
        false_activation_passed,
        baseline_speech_gain,
        candidate_speech_gain,
        speech_convergence_passed,
        maximum_silence_gain,
        silence_relaxation_db,
        silence_passed,
        fresh_gain,
        stale_gain,
        stale_passed,
        baseline_pumping,
        candidate_pumping,
        pumping_passed,
        maximum_transition_jump,
        click_passed,
    );
    if !retained {
        std::process::exit(1);
    }
}
