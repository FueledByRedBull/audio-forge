/// Pure block-processing boundary for offline/file-based tests and tools.
pub trait AudioBlockProcessor<const N: usize> {
    fn process_block(&mut self, input: &mut [f32], output: &mut FixedAudioBuffer<f32, N>);
}

/// Offline DSP chain that does not depend on CPAL streams or live ring buffers.
pub struct OfflineDspBlockProcessor {
    deesser: DeEsser,
    eq: ParametricEQ,
    compressor: Compressor,
    limiter: Limiter,
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
}

impl<const N: usize> AudioBlockProcessor<N> for OfflineDspBlockProcessor {
    fn process_block(&mut self, input: &mut [f32], output: &mut FixedAudioBuffer<f32, N>) {
        output.clear();
        let count = input.len().min(output.capacity());
        if !output.set_len_zeroed(count) {
            return;
        }

        output.as_mut_slice().copy_from_slice(&input[..count]);
        let block = output.as_mut_slice();

        if self.deesser_enabled {
            self.deesser.process_block_inplace(block);
        }
        if self.eq_enabled {
            self.eq.process_block_inplace(block);
        }
        if self.compressor_enabled {
            self.compressor.process_block_inplace(block);
        }
        if self.limiter_enabled {
            self.limiter.process_block_inplace(block);
        }
    }
}
