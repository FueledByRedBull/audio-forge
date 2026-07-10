/// Results from a deterministic control-thread/DSP-thread contention run.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct ControlDspStressReport {
    pub control_updates: usize,
    pub processed_blocks: usize,
    pub snapshot_rearms: usize,
    pub model_switches: usize,
    pub suppressor_resets: usize,
    pub max_output_abs: f32,
}

#[derive(Clone, Copy)]
struct StressRng(u64);

impl StressRng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }

    fn range_f64(&mut self, minimum: f64, maximum: f64) -> f64 {
        minimum + (maximum - minimum) * self.unit_f64()
    }

    fn boolean(&mut self) -> bool {
        (self.next_u64() & 1) != 0
    }
}

/// Exercise the live control handoff protocol concurrently with production DSP types.
///
/// Engine construction stays on the control thread. The DSP thread consumes
/// prebuilt engines from a bounded command queue and retires the old engine to a
/// second queue, matching the live model-switching ownership boundary.
#[doc(hidden)]
pub fn run_seeded_control_dsp_stress(
    seed: u64,
    iterations: usize,
) -> Result<ControlDspStressReport, String> {
    if iterations == 0 {
        return Err("iterations must be greater than zero".to_string());
    }

    const BLOCK_SIZE: usize = RNNOISE_FRAME_SIZE;
    const MAX_OUTPUT_ABS: f32 = 16.0;

    let gate_control = Arc::new(AtomicGateControlState::new());
    let suppressor_control = Arc::new(AtomicSuppressorControlState::new());
    let eq_control = Arc::new(EqControlState::new());
    let compressor_control = Arc::new(AtomicCompressorControlState::new());
    let deesser_control = Arc::new(AtomicDeesserControlState::new());
    let limiter_control = Arc::new(AtomicLimiterControlState::new());
    let gate_dirty = Arc::new(AtomicBool::new(false));
    let suppressor_dirty = Arc::new(AtomicBool::new(false));
    let eq_dirty = Arc::new(AtomicBool::new(true));
    let compressor_dirty = Arc::new(AtomicBool::new(false));
    let deesser_dirty = Arc::new(AtomicBool::new(false));
    let limiter_dirty = Arc::new(AtomicBool::new(false));
    let reset_requested = Arc::new(AtomicBool::new(false));
    let snapshot_rearms = Arc::new(AtomicUsize::new(0));
    let control_done = Arc::new(AtomicBool::new(false));

    // Force one unstable read so dirty-flag rearming is covered deterministically.
    eq_control.seq.store(1, Ordering::Release);

    let (mut command_tx, mut command_rx) =
        RtCommandQueue::<NoiseSuppressionEngine, 4>::new().split();
    let (mut retire_tx, mut retire_rx) =
        RtCommandQueue::<NoiseSuppressionEngine, 64>::new().split();
    let strength = Arc::new(AtomicU32::new(1.0_f32.to_bits()));

    let control_gate = Arc::clone(&gate_control);
    let control_suppressor = Arc::clone(&suppressor_control);
    let control_eq = Arc::clone(&eq_control);
    let control_compressor = Arc::clone(&compressor_control);
    let control_deesser = Arc::clone(&deesser_control);
    let control_limiter = Arc::clone(&limiter_control);
    let control_gate_dirty = Arc::clone(&gate_dirty);
    let control_suppressor_dirty = Arc::clone(&suppressor_dirty);
    let control_eq_dirty = Arc::clone(&eq_dirty);
    let control_compressor_dirty = Arc::clone(&compressor_dirty);
    let control_deesser_dirty = Arc::clone(&deesser_dirty);
    let control_limiter_dirty = Arc::clone(&limiter_dirty);
    let control_reset = Arc::clone(&reset_requested);
    let control_rearms = Arc::clone(&snapshot_rearms);
    let control_finished = Arc::clone(&control_done);
    let control_strength = Arc::clone(&strength);

    let control_thread = std::thread::spawn(move || -> Result<(usize, usize), String> {
        let deadline = Instant::now() + Duration::from_secs(2);
        while control_rearms.load(Ordering::Acquire) == 0 {
            if Instant::now() >= deadline {
                control_finished.store(true, Ordering::Release);
                return Err("DSP thread did not rearm an unstable snapshot".to_string());
            }
            std::thread::yield_now();
        }
        control_eq.seq.fetch_add(1, Ordering::Release);

        let mut rng = StressRng::new(seed);
        let mut requested_switches = 0;
        for index in 0..iterations {
            match rng.next_u64() % 6 {
                0 => {
                    control_gate.update(|state| {
                        state.enabled.store(rng.boolean(), Ordering::Relaxed);
                        state.threshold_db_bits.store(
                            rng.range_f64(-70.0, -15.0).to_bits(),
                            Ordering::Relaxed,
                        );
                        state.attack_ms_bits.store(
                            rng.range_f64(0.1, 80.0).to_bits(),
                            Ordering::Relaxed,
                        );
                        state.release_ms_bits.store(
                            rng.range_f64(10.0, 800.0).to_bits(),
                            Ordering::Relaxed,
                        );
                    });
                    control_gate_dirty.store(true, Ordering::Release);
                }
                1 => {
                    let band = (rng.next_u64() as usize) % NUM_BANDS;
                    control_eq.update(|state| {
                        state.enabled.store(rng.boolean(), Ordering::Relaxed);
                        state.frequency_bits[band].store(
                            rng.range_f64(30.0, 20_000.0).to_bits(),
                            Ordering::Relaxed,
                        );
                        state.gain_bits[band].store(
                            rng.range_f64(-6.0, 6.0).to_bits(),
                            Ordering::Relaxed,
                        );
                        state.q_bits[band].store(
                            rng.range_f64(0.2, 6.0).to_bits(),
                            Ordering::Relaxed,
                        );
                    });
                    control_eq_dirty.store(true, Ordering::Release);
                }
                2 => {
                    control_compressor.update(|state| {
                        state.enabled.store(rng.boolean(), Ordering::Relaxed);
                        state.threshold_db_bits.store(
                            rng.range_f64(-50.0, -6.0).to_bits(),
                            Ordering::Relaxed,
                        );
                        state.ratio_bits.store(
                            rng.range_f64(1.0, 12.0).to_bits(),
                            Ordering::Relaxed,
                        );
                        state.attack_ms_bits.store(
                            rng.range_f64(0.1, 80.0).to_bits(),
                            Ordering::Relaxed,
                        );
                        state.base_release_ms_bits.store(
                            rng.range_f64(20.0, 500.0).to_bits(),
                            Ordering::Relaxed,
                        );
                        state.makeup_gain_db_bits.store(
                            rng.range_f64(0.0, 6.0).to_bits(),
                            Ordering::Relaxed,
                        );
                    });
                    control_compressor_dirty.store(true, Ordering::Release);
                }
                3 => {
                    let low_cut = rng.range_f64(2_000.0, 10_000.0);
                    control_deesser.update(|state| {
                        state.enabled.store(rng.boolean(), Ordering::Relaxed);
                        state.low_cut_hz_bits.store(low_cut.to_bits(), Ordering::Relaxed);
                        state.high_cut_hz_bits.store(
                            rng.range_f64(low_cut + 200.0, 16_000.0).to_bits(),
                            Ordering::Relaxed,
                        );
                        state.threshold_db_bits.store(
                            rng.range_f64(-50.0, -8.0).to_bits(),
                            Ordering::Relaxed,
                        );
                        state.ratio_bits.store(
                            rng.range_f64(1.0, 12.0).to_bits(),
                            Ordering::Relaxed,
                        );
                    });
                    control_deesser_dirty.store(true, Ordering::Release);
                }
                4 => {
                    control_limiter.update(|state| {
                        state.enabled.store(rng.boolean(), Ordering::Relaxed);
                        state.ceiling_db_bits.store(
                            rng.range_f64(-9.0, -0.1).to_bits(),
                            Ordering::Relaxed,
                        );
                        state.release_ms_bits.store(
                            rng.range_f64(10.0, 300.0).to_bits(),
                            Ordering::Relaxed,
                        );
                    });
                    control_limiter_dirty.store(true, Ordering::Release);
                }
                _ => {
                    control_suppressor.set_enabled(rng.boolean());
                    control_suppressor_dirty.store(true, Ordering::Release);
                }
            }

            if index % 97 == 0 {
                #[cfg(feature = "deepfilter")]
                let model = if requested_switches % 2 == 0 {
                    NoiseModel::DeepFilterNetLL
                } else {
                    NoiseModel::RNNoise
                };
                #[cfg(not(feature = "deepfilter"))]
                let model = NoiseModel::RNNoise;

                let mut candidate =
                    NoiseSuppressionEngine::new(model, Arc::clone(&control_strength));
                loop {
                    match command_tx.push(candidate) {
                        Ok(()) => break,
                        Err(returned) => {
                            candidate = returned;
                            while retire_rx.pop().is_some() {}
                            std::thread::yield_now();
                        }
                    }
                }
                control_suppressor.set_model(model);
                control_suppressor_dirty.store(true, Ordering::Release);
                requested_switches += 1;
            }
            if index % 43 == 0 {
                control_reset.store(true, Ordering::Release);
            }
            while retire_rx.pop().is_some() {}
            std::thread::yield_now();
        }

        control_finished.store(true, Ordering::Release);
        while retire_rx.pop().is_some() {}
        Ok((iterations, requested_switches))
    });

    let dsp_gate = Arc::clone(&gate_control);
    let dsp_suppressor = Arc::clone(&suppressor_control);
    let dsp_eq = Arc::clone(&eq_control);
    let dsp_compressor = Arc::clone(&compressor_control);
    let dsp_deesser = Arc::clone(&deesser_control);
    let dsp_limiter = Arc::clone(&limiter_control);
    let dsp_rearms = Arc::clone(&snapshot_rearms);
    let dsp_finished = Arc::clone(&control_done);
    let dsp_strength = Arc::clone(&strength);

    let dsp_thread = std::thread::spawn(move || -> Result<ControlDspStressReport, String> {
        let mut gate = NoiseGate::new(-40.0, 10.0, 100.0, TARGET_SAMPLE_RATE as f64);
        let mut suppressor =
            NoiseSuppressionEngine::new(NoiseModel::RNNoise, Arc::clone(&dsp_strength));
        let mut chain = OfflineDspBlockProcessor::new(TARGET_SAMPLE_RATE as f64);
        let mut output = FixedAudioBuffer::<f32, BLOCK_SIZE>::new();
        let mut input = [0.0_f32; BLOCK_SIZE];
        let mut suppressed = [0.0_f32; BLOCK_SIZE];
        let mut rng = StressRng::new(seed ^ 0xa076_1d64_78bd_642f);
        let mut processed_blocks = 0;
        let mut resets = 0;
        let mut switches = 0;
        let mut max_output_abs = 0.0_f32;
        let mut deferred_retire = None;

        while processed_blocks < iterations || !dsp_finished.load(Ordering::Acquire) {
            macro_rules! apply_snapshot {
                ($dirty:expr, $control:expr, $apply:expr) => {
                    if $dirty.swap(false, Ordering::AcqRel) {
                        if let Some(snapshot) = $control.snapshot() {
                            $apply(snapshot);
                        } else {
                            $dirty.store(true, Ordering::Release);
                            dsp_rearms.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                };
            }

            apply_snapshot!(gate_dirty, dsp_gate, |snapshot| {
                apply_gate_control(&mut gate, &snapshot)
            });
            apply_snapshot!(eq_dirty, dsp_eq, |snapshot| {
                apply_eq_control(chain.eq_mut(), &snapshot)
            });
            apply_snapshot!(compressor_dirty, dsp_compressor, |snapshot| {
                apply_compressor_control(chain.compressor_mut(), &snapshot)
            });
            apply_snapshot!(deesser_dirty, dsp_deesser, |snapshot| {
                apply_deesser_control(chain.deesser_mut(), &snapshot)
            });
            apply_snapshot!(limiter_dirty, dsp_limiter, |snapshot| {
                apply_limiter_control(chain.limiter_mut(), &snapshot)
            });

            if let Some(retired) = deferred_retire.take() {
                if let Err(returned) = retire_tx.push(retired) {
                    deferred_retire = Some(returned);
                }
            }
            if suppressor_dirty.swap(false, Ordering::AcqRel) {
                if let Some(snapshot) = dsp_suppressor.snapshot() {
                    while deferred_retire.is_none() {
                        let Some(candidate) = command_rx.pop() else {
                            break;
                        };
                        if candidate.model_type() == snapshot.model
                            && suppressor.model_type() != snapshot.model
                        {
                            let retired = std::mem::replace(&mut suppressor, candidate);
                            switches += 1;
                            if let Err(returned) = retire_tx.push(retired) {
                                deferred_retire = Some(returned);
                            }
                            break;
                        }
                        if let Err(returned) = retire_tx.push(candidate) {
                            deferred_retire = Some(returned);
                        }
                    }
                    if suppressor.model_type() == snapshot.model {
                        apply_suppressor_control(&mut suppressor, &snapshot);
                    } else {
                        suppressor_dirty.store(true, Ordering::Release);
                    }
                } else {
                    suppressor_dirty.store(true, Ordering::Release);
                    dsp_rearms.fetch_add(1, Ordering::Relaxed);
                }
            }
            if reset_requested.swap(false, Ordering::AcqRel) {
                suppressor.soft_reset();
                resets += 1;
            }

            for (sample_index, sample) in input.iter_mut().enumerate() {
                let phase = (processed_blocks * BLOCK_SIZE + sample_index) as f32
                    * (2.0 * std::f32::consts::PI * 173.0 / TARGET_SAMPLE_RATE as f32);
                let noise = (rng.unit_f64() as f32 - 0.5) * 0.02;
                *sample = phase.sin() * 0.22 + noise;
            }
            gate.process_block_inplace(&mut input);

            let accepted = suppressor.push_samples(&input);
            if accepted != input.len() {
                return Err(format!(
                    "suppressor accepted {accepted} of {} samples",
                    input.len()
                ));
            }
            suppressor.process_frames();
            let written = suppressor.pop_samples_into(&mut suppressed);
            let chain_input = if written == BLOCK_SIZE {
                &mut suppressed[..]
            } else {
                &mut input[..]
            };
            let stats = chain.process_block_with_stats(chain_input, &mut output);
            if !stats.output_true_peak.is_finite() {
                return Err("non-finite true-peak metric".to_string());
            }
            for &sample in output.as_slice() {
                if !sample.is_finite() {
                    return Err("non-finite DSP output".to_string());
                }
                max_output_abs = max_output_abs.max(sample.abs());
                if max_output_abs > MAX_OUTPUT_ABS {
                    return Err(format!("DSP output exceeded bound: {max_output_abs}"));
                }
            }

            processed_blocks += 1;
            std::thread::yield_now();
        }

        if let Some(retired) = deferred_retire {
            let _ = retire_tx.push(retired);
        }
        Ok(ControlDspStressReport {
            control_updates: 0,
            processed_blocks,
            snapshot_rearms: dsp_rearms.load(Ordering::Relaxed),
            model_switches: switches,
            suppressor_resets: resets,
            max_output_abs,
        })
    });

    let (control_updates, requested_switches) = control_thread
        .join()
        .map_err(|_| "control stress thread panicked".to_string())??;
    let mut report = dsp_thread
        .join()
        .map_err(|_| "DSP stress thread panicked".to_string())??;
    report.control_updates = control_updates;
    if report.snapshot_rearms == 0 {
        return Err("snapshot dirty flag was never rearmed".to_string());
    }
    if requested_switches > 0 && report.model_switches == 0 {
        return Err("no queued suppressor model switch reached the DSP thread".to_string());
    }
    if report.suppressor_resets == 0 {
        return Err("no suppressor reset reached the DSP thread".to_string());
    }
    Ok(report)
}
