/// Pure block-processing boundary for offline/file-based tests and tools.
pub trait AudioBlockProcessor<const N: usize> {
    fn process_block(&mut self, input: &mut [f32], output: &mut FixedAudioBuffer<f32, N>);
}

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
    eq: ParametricEQ,
    compressor: Compressor,
    limiter: Limiter,
    true_peak_limiter: TruePeakLimiter,
    true_peak_detector: TruePeakDetector,
    deesser_enabled: bool,
    eq_enabled: bool,
    compressor_enabled: bool,
    limiter_enabled: bool,
}

impl OfflineDspBlockProcessor {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            deesser: DeEsser::new(sample_rate),
            eq: ParametricEQ::new(sample_rate),
            compressor: Compressor::new(-18.0, 3.0, 5.0, 100.0, 0.0, 6.0, sample_rate),
            limiter: Limiter::default_settings(sample_rate),
            true_peak_limiter: TruePeakLimiter::default_settings(sample_rate as f32),
            true_peak_detector: TruePeakDetector::new(),
            deesser_enabled: false,
            eq_enabled: true,
            compressor_enabled: false,
            limiter_enabled: true,
        }
    }

    pub fn set_deesser_enabled(&mut self, enabled: bool) {
        self.deesser_enabled = enabled;
        self.deesser.set_enabled(enabled);
    }

    pub fn set_eq_enabled(&mut self, enabled: bool) {
        self.eq_enabled = enabled;
        self.eq.set_enabled(enabled);
    }

    pub fn set_compressor_enabled(&mut self, enabled: bool) {
        self.compressor_enabled = enabled;
        self.compressor.set_enabled(enabled);
    }

    pub fn set_limiter_enabled(&mut self, enabled: bool) {
        self.limiter_enabled = enabled;
        self.limiter.set_enabled(enabled);
    }

    pub fn eq_mut(&mut self) -> &mut ParametricEQ {
        &mut self.eq
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

        if self.deesser_enabled {
            self.deesser.process_block_inplace(block);
            stats.deesser_gain_reduction_db = self.deesser.current_gain_reduction_db();
        }
        if self.eq_enabled {
            self.eq.process_block_inplace(block);
        }
        if self.compressor_enabled {
            self.compressor.process_block_inplace(block);
            stats.compressor_gain_reduction_db = self.compressor.current_gain_reduction() as f32;
        }
        if self.limiter_enabled {
            self.limiter.process_block_inplace(block);
            stats.limiter_peak_gain_reduction_db =
                self.limiter.peak_gain_reduction_and_reset() as f32;
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
}

impl<const N: usize> AudioBlockProcessor<N> for OfflineDspBlockProcessor {
    fn process_block(&mut self, input: &mut [f32], output: &mut FixedAudioBuffer<f32, N>) {
        let _stats = self.process_block_with_stats(input, output);
    }
}
