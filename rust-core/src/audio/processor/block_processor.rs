#[derive(Debug, Clone, Copy)]
pub struct OfflineDspBlockStats {
    pub input_sample_peak: f32,
    pub output_sample_peak: f32,
    pub pre_limiter_true_peak: f32,
    pub true_peak_limiter_input_peak: f32,
    pub output_true_peak: f32,
    pub limiter_peak_gain_reduction_db: f32,
    pub true_peak_limiter_gain_reduction_db: f32,
    pub true_peak_limited_events: u64,
    pub compressor_gain_reduction_db: f32,
    pub deesser_gain_reduction_db: f32,
}

impl Default for OfflineDspBlockStats {
    fn default() -> Self {
        Self {
            input_sample_peak: 0.0,
            output_sample_peak: 0.0,
            pre_limiter_true_peak: 0.0,
            true_peak_limiter_input_peak: 0.0,
            output_true_peak: 0.0,
            limiter_peak_gain_reduction_db: 0.0,
            true_peak_limiter_gain_reduction_db: 0.0,
            true_peak_limited_events: 0,
            compressor_gain_reduction_db: 0.0,
            deesser_gain_reduction_db: 0.0,
        }
    }
}

/// Offline DSP chain that does not depend on CPAL streams or live ring buffers.
pub struct OfflineDspBlockProcessor {
    deesser: DeEsser,
    correction_eq: ParametricEQ,
    tone_eq: ParametricEQ,
    compressor: Compressor,
    limiter: Limiter,
    true_peak_limiter: TruePeakLimiter,
    true_peak_detector: TruePeakDetector,
    pre_limiter_true_peak_detector: TruePeakDetector,
    deesser_enabled: bool,
    /// Whether any block has been processed; enable changes before the first
    /// block apply instantly, later ones ramp like the live path.
    started: bool,
    /// The normal lookahead limiter is separate from the final output safety
    /// limiter so bypass can match the live path.
    normal_limiter_enabled: bool,
    output_protection_enabled: bool,
    eq_before_deesser: bool,
    previous_true_peak_limiter_gain_reduction_db: f64,
}

impl OfflineDspBlockProcessor {
    pub fn new(sample_rate: f64) -> Self {
        let mut compressor = Compressor::new(-18.0, 3.0, 5.0, 100.0, 0.0, 6.0, sample_rate);
        compressor.set_enabled(false);
        compressor.finish_enable_transition();
        Self {
            deesser: DeEsser::new(sample_rate),
            correction_eq: ParametricEQ::new(sample_rate),
            tone_eq: ParametricEQ::new(sample_rate),
            compressor,
            limiter: Limiter::default_settings(sample_rate),
            true_peak_limiter: TruePeakLimiter::default_settings(sample_rate as f32),
            true_peak_detector: TruePeakDetector::new(),
            pre_limiter_true_peak_detector: TruePeakDetector::new(),
            deesser_enabled: false,
            started: false,
            normal_limiter_enabled: true,
            output_protection_enabled: true,
            eq_before_deesser: false,
            previous_true_peak_limiter_gain_reduction_db: 0.0,
        }
    }

    pub fn set_deesser_enabled(&mut self, enabled: bool) {
        self.deesser_enabled = enabled;
        self.deesser.set_enabled(enabled);
    }

    pub fn set_eq_enabled(&mut self, enabled: bool) {
        self.correction_eq.set_enabled(enabled);
        self.tone_eq.set_enabled(enabled);
    }

    pub fn set_compressor_enabled(&mut self, enabled: bool) {
        self.compressor.set_enabled(enabled);
        if !self.started {
            self.compressor.finish_enable_transition();
        }
    }

    /// Match the live limiter toggle: both stages stay in the path (and keep
    /// their delay) while limiting is disabled.
    pub fn set_limiter_enabled(&mut self, enabled: bool) {
        self.normal_limiter_enabled = true;
        self.output_protection_enabled = enabled;
        if !enabled {
            self.previous_true_peak_limiter_gain_reduction_db = 0.0;
        }
        self.limiter.set_enabled(enabled);
    }

    /// Include or skip the configured lookahead limiter stage, as the live
    /// Raw and Bypass paths do. Output safety remains controlled by
    /// `set_limiter_enabled`.
    pub fn set_normal_limiter_enabled(&mut self, enabled: bool) {
        self.normal_limiter_enabled = enabled;
    }

    pub fn set_eq_before_deesser(&mut self, enabled: bool) {
        self.eq_before_deesser = enabled;
    }

    pub fn eq_mut(&mut self) -> &mut ParametricEQ {
        &mut self.tone_eq
    }

    /// Access the independent microphone-correction EQ stage.
    pub fn correction_eq_mut(&mut self) -> &mut ParametricEQ {
        &mut self.correction_eq
    }

    /// Apply both EQ layers before the next block is rendered.
    pub fn set_eq_layers(
        &mut self,
        correction_bands: &[EqBandConfig; NUM_BANDS],
        tone_bands: &[EqBandConfig; NUM_BANDS],
    ) {
        for index in 0..NUM_BANDS {
            self.correction_eq
                .set_band_config(index, correction_bands[index]);
            self.tone_eq.set_band_config(index, tone_bands[index]);
        }
        self.correction_eq.reset();
        self.tone_eq.reset();
    }

    pub fn deesser_mut(&mut self) -> &mut DeEsser {
        &mut self.deesser
    }

    pub fn compressor_mut(&mut self) -> &mut Compressor {
        &mut self.compressor
    }

    pub fn limiter_mut(&mut self) -> &mut Limiter {
        &mut self.limiter
    }

    pub fn true_peak_limiter_mut(&mut self) -> &mut TruePeakLimiter {
        &mut self.true_peak_limiter
    }

    pub fn latency_samples(&self) -> usize {
        let normal_latency = if self.normal_limiter_enabled {
            self.limiter.lookahead_samples()
        } else {
            0
        };
        normal_latency + self.true_peak_limiter.lookahead_samples()
    }

    pub fn process_block_with_stats<const N: usize>(
        &mut self,
        input: &mut [f32],
        output: &mut FixedAudioBuffer<f32, N>,
    ) -> OfflineDspBlockStats {
        self.process_block_with_stats_and_activity_control(input, output, None)
    }

    /// Process a block with the causal frontend evidence used by live
    /// auto-makeup control. `None` preserves the downstream-only simulator's
    /// historical RMS fallback and limiter behavior.
    fn process_block_with_stats_and_activity_control<const N: usize>(
        &mut self,
        input: &mut [f32],
        output: &mut FixedAudioBuffer<f32, N>,
        activity: Option<AutoMakeupActivityInput>,
    ) -> OfflineDspBlockStats {
        let mut stats = OfflineDspBlockStats {
            input_sample_peak: input.iter().map(|sample| sample.abs()).fold(0.0_f32, f32::max),
            ..OfflineDspBlockStats::default()
        };
        self.started = true;

        output.clear();
        let count = input.len().min(output.capacity());
        if !output.set_len_zeroed(count) {
            return stats;
        }

        output.as_mut_slice().copy_from_slice(&input[..count]);
        let block = output.as_mut_slice();

        if self.eq_before_deesser {
            self.correction_eq.process_block_inplace(block);
            self.tone_eq.process_block_inplace(block);
            if self.deesser_enabled {
                self.deesser.process_block_inplace(block);
                stats.deesser_gain_reduction_db = self.deesser.current_gain_reduction_db();
            }
        } else {
            if self.deesser_enabled {
                self.deesser.process_block_inplace(block);
                stats.deesser_gain_reduction_db = self.deesser.current_gain_reduction_db();
            }
            self.correction_eq.process_block_inplace(block);
            self.tone_eq.process_block_inplace(block);
        }
        if self.compressor.is_active() {
            if let Some(activity) = activity {
                let limiter_feedback = if self.normal_limiter_enabled {
                    self.limiter.current_gain_reduction().abs()
                } else {
                    0.0
                };
                let true_peak_feedback = if self.output_protection_enabled {
                    self.previous_true_peak_limiter_gain_reduction_db
                } else {
                    0.0
                };
                self.compressor.set_limiter_feedback_gain_reduction_db(
                    limiter_feedback.max(true_peak_feedback),
                );
                self.compressor
                    .process_block_inplace_with_activity_control(block, Some(activity));
            } else {
                self.compressor.process_block_inplace(block);
            }
            stats.compressor_gain_reduction_db =
                self.compressor.block_peak_gain_reduction() as f32;
        }
        stats.pre_limiter_true_peak = self.pre_limiter_true_peak_detector.process_block(block);
        if self.normal_limiter_enabled {
            self.limiter.process_block_inplace(block);
            stats.limiter_peak_gain_reduction_db =
                self.limiter.peak_gain_reduction_and_reset() as f32;
        }
        sanitize_non_finite_inplace(block);
        let output_ceiling = if self.output_protection_enabled {
            10.0_f32.powf(self.limiter.ceiling_db() as f32 / 20.0)
        } else {
            1.0
        };
        // Same order as the live output writer: ceiling first, so a re-enable
        // plans queued audio against the configured ceiling.
        if self.output_protection_enabled {
            self.true_peak_limiter
                .set_ceiling_linear(output_ceiling);
        }
        self.true_peak_limiter
            .set_enabled(self.output_protection_enabled);
        let true_peak_stats = self.true_peak_limiter.process_block_inplace(block);
        if self.output_protection_enabled {
            stats.true_peak_limiter_input_peak = true_peak_stats.input_true_peak;
            stats.true_peak_limiter_gain_reduction_db = true_peak_stats.max_gain_reduction_db;
            stats.true_peak_limited_events = true_peak_stats.limited_events;
            self.previous_true_peak_limiter_gain_reduction_db =
                f64::from(true_peak_stats.max_gain_reduction_db);
        } else {
            self.previous_true_peak_limiter_gain_reduction_db = 0.0;
        }

        sanitize_and_clamp_output_inplace(block, output_ceiling);

        stats.output_sample_peak = block.iter().map(|sample| sample.abs()).fold(0.0_f32, f32::max);
        stats.output_true_peak = self.true_peak_detector.process_block(block);
        stats
    }

    pub fn process_block<const N: usize>(
        &mut self,
        input: &mut [f32],
        output: &mut FixedAudioBuffer<f32, N>,
    ) {
        let _stats = self.process_block_with_stats(input, output);
    }
}

#[cfg(test)]
mod block_processor_tests {
    use super::*;

    #[test]
    fn limiter_toggle_keeps_delayed_audio_in_the_timeline() {
        let mut processor = OfflineDspBlockProcessor::new(48_000.0);
        processor.set_eq_enabled(false);
        processor.eq_mut().reset();
        processor.correction_eq_mut().reset();
        let latency = processor.latency_samples();
        let level = 0.25_f32;
        let mut output = FixedAudioBuffer::<f32, 256>::new();
        let mut emitted = Vec::new();

        for enabled in [true, false, true, false] {
            processor.set_limiter_enabled(enabled);
            assert_eq!(processor.latency_samples(), latency);
            let mut block = [level; 256];
            processor.process_block_with_stats(&mut block, &mut output);
            emitted.extend_from_slice(output.as_slice());
        }

        assert!(emitted[..latency].iter().all(|sample| *sample == 0.0));
        assert!(emitted[latency..]
            .iter()
            .all(|sample| (*sample - level).abs() < 1e-5));
    }

    #[test]
    fn limiter_reenable_keeps_queued_intersample_overs_under_the_ceiling() {
        use crate::dsp::TruePeakDetector;

        let mut processor = OfflineDspBlockProcessor::new(48_000.0);
        processor.set_eq_enabled(false);
        processor.eq_mut().reset();
        processor.correction_eq_mut().reset();
        processor.limiter_mut().set_ceiling(-6.0);
        let ceiling = 10.0_f32.powf(-6.0 / 20.0);
        // 12 kHz tone: samples at 0.64, true peak 0.9.
        let block_at = |start: usize| -> Vec<f32> {
            (start..start + 480)
                .map(|n| {
                    0.9 * (std::f32::consts::PI * n as f32 / 2.0 + std::f32::consts::PI / 4.0)
                        .sin()
                })
                .collect()
        };
        let mut detector = TruePeakDetector::new();
        let mut output = FixedAudioBuffer::<f32, 480>::new();

        processor.set_limiter_enabled(false);
        for index in 0..10 {
            let mut block = block_at(index * 480);
            processor.process_block_with_stats(&mut block, &mut output);
            detector.process_block(output.as_slice());
        }
        processor.set_limiter_enabled(true);
        let mut peak = 0.0_f32;
        for index in 10..20 {
            let mut block = block_at(index * 480);
            processor.process_block_with_stats(&mut block, &mut output);
            if index == 10 {
                // The first block carries the queued audio. Skip only the
                // unlimited audio's 16-sample reconstruction tail plus the
                // detector's 128-sample latency.
                detector.process_block(&output.as_slice()[..144]);
                peak = peak.max(detector.process_block(&output.as_slice()[144..]));
            } else {
                peak = peak.max(detector.process_block(output.as_slice()));
            }
        }
        assert!(peak <= ceiling * 1.01, "true peak {peak} over ceiling {ceiling}");
    }

    #[test]
    fn offline_compressor_toggle_fades_like_live_after_audio_starts() {
        let mut processor = OfflineDspBlockProcessor::new(48_000.0);
        processor.set_eq_enabled(false);
        processor.eq_mut().reset();
        processor.correction_eq_mut().reset();
        processor.set_limiter_enabled(false);
        processor.set_compressor_enabled(true);
        processor.compressor_mut().set_makeup_gain(9.0);
        let mut output = FixedAudioBuffer::<f32, 480>::new();
        for _ in 0..100 {
            let mut block = [0.3_f32; 480];
            processor.process_block_with_stats(&mut block, &mut output);
        }
        let compressed = *output.as_slice().last().unwrap();
        assert!((compressed - 0.3).abs() > 0.05, "compressed={compressed}");

        processor.set_compressor_enabled(false);
        let mut emitted = vec![compressed];
        for _ in 0..3 {
            let mut block = [0.3_f32; 480];
            processor.process_block_with_stats(&mut block, &mut output);
            emitted.extend_from_slice(output.as_slice());
        }
        let max_step = emitted
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .fold(0.0_f32, f32::max);
        // A ramp moves a small fraction of the processed/dry difference per
        // sample; skipping the compressor would step by all of it.
        assert!(max_step < (compressed - 0.3).abs() / 20.0, "max_step={max_step}");
        assert!((emitted.last().unwrap() - 0.3).abs() < 1e-4);
    }

    #[test]
    fn activity_control_reaches_offline_compressor() {
        let mut with_evidence = OfflineDspBlockProcessor::new(48_000.0);
        with_evidence.set_eq_enabled(false);
        with_evidence.set_compressor_enabled(true);
        with_evidence.set_limiter_enabled(false);
        with_evidence.compressor_mut().set_auto_makeup_enabled(true);
        with_evidence.compressor_mut().set_target_lufs(-12.0);

        let mut without_evidence = OfflineDspBlockProcessor::new(48_000.0);
        without_evidence.set_eq_enabled(false);
        without_evidence.set_compressor_enabled(true);
        without_evidence.set_limiter_enabled(false);
        without_evidence
            .compressor_mut()
            .set_auto_makeup_enabled(true);
        without_evidence
            .compressor_mut()
            .set_target_lufs(-12.0);

        let evidence = AutoMakeupActivityInput {
            vad_probability: 0.95,
            vad_reliability: 1.0,
            noise_floor_db: -60.0,
            live_noise_reliability: 1.0,
        };
        let mut input_with_evidence = [0.001_f32; 480];
        let mut input_without_evidence = input_with_evidence;
        let mut output_with_evidence = FixedAudioBuffer::<f32, 480>::new();
        let mut output_without_evidence = FixedAudioBuffer::<f32, 480>::new();

        for _ in 0..60 {
            with_evidence.process_block_with_stats_and_activity_control(
                &mut input_with_evidence,
                &mut output_with_evidence,
                Some(evidence),
            );
            without_evidence.process_block_with_stats(
                &mut input_without_evidence,
                &mut output_without_evidence,
            );
        }

        assert!(
            with_evidence.compressor.current_makeup_gain()
                > without_evidence.compressor.current_makeup_gain() + 0.1
        );
    }
}
