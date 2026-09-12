#[derive(Debug, Clone, Copy)]
pub struct OfflineDspBlockStats {
    pub input_sample_peak: f32,
    pub output_sample_peak: f32,
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
    deesser_enabled: bool,
    compressor_enabled: bool,
    /// The normal lookahead limiter is separate from the final output safety
    /// limiter so bypass can match the live path.
    normal_limiter_enabled: bool,
    output_protection_enabled: bool,
    eq_before_deesser: bool,
}

impl OfflineDspBlockProcessor {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            deesser: DeEsser::new(sample_rate),
            correction_eq: ParametricEQ::new(sample_rate),
            tone_eq: ParametricEQ::new(sample_rate),
            compressor: Compressor::new(-18.0, 3.0, 5.0, 100.0, 0.0, 6.0, sample_rate),
            limiter: Limiter::default_settings(sample_rate),
            true_peak_limiter: TruePeakLimiter::default_settings(sample_rate as f32),
            true_peak_detector: TruePeakDetector::new(),
            deesser_enabled: false,
            compressor_enabled: false,
            normal_limiter_enabled: true,
            output_protection_enabled: true,
            eq_before_deesser: false,
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
        self.compressor_enabled = enabled;
        self.compressor.set_enabled(enabled);
    }

    pub fn set_limiter_enabled(&mut self, enabled: bool) {
        self.normal_limiter_enabled = enabled;
        self.output_protection_enabled = enabled;
        self.limiter.set_enabled(enabled);
    }

    /// Enable or disable only the configured lookahead limiter stage.
    /// Output safety remains controlled by `set_limiter_enabled`.
    pub fn set_normal_limiter_enabled(&mut self, enabled: bool) {
        self.normal_limiter_enabled = enabled;
        self.limiter.set_enabled(enabled);
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
        let output_safety_latency = if self.output_protection_enabled {
            self.true_peak_limiter.lookahead_samples()
        } else {
            0
        };
        normal_latency + output_safety_latency
    }

    pub fn process_block_with_stats<const N: usize>(
        &mut self,
        input: &mut [f32],
        output: &mut FixedAudioBuffer<f32, N>,
    ) -> OfflineDspBlockStats {
        let mut stats = OfflineDspBlockStats {
            input_sample_peak: input.iter().map(|sample| sample.abs()).fold(0.0_f32, f32::max),
            ..OfflineDspBlockStats::default()
        };

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
        if self.compressor_enabled {
            self.compressor.process_block_inplace(block);
            stats.compressor_gain_reduction_db =
                self.compressor.block_peak_gain_reduction() as f32;
        }
        if self.normal_limiter_enabled {
            self.limiter.process_block_inplace(block);
            stats.limiter_peak_gain_reduction_db =
                self.limiter.peak_gain_reduction_and_reset() as f32;
        }
        if self.output_protection_enabled {
            self.true_peak_limiter
                .set_ceiling_linear(10.0_f32.powf(self.limiter.ceiling_db() as f32 / 20.0));
            let true_peak_stats = self.true_peak_limiter.process_block_inplace(block);
            stats.true_peak_limiter_input_peak = true_peak_stats.input_true_peak;
            stats.true_peak_limiter_gain_reduction_db = true_peak_stats.max_gain_reduction_db;
            stats.true_peak_limited_events = true_peak_stats.limited_events;
        }

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
