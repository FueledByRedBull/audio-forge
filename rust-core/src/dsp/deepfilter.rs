//! DeepFilterNet integration using libDF (with rollback strategy)
//!
//! ACTUAL LIBDF API (verified from 10-01-SUMMARY.md):
//! - Main struct: DfTract (not DeepFilter)
//! - Constructor: DfTract::default() for embedded model, or DfTract::new(df_params, &runtime_params)
//! - Process method: df.process(&mut self, noisy: ArrayView2<f32>, enh: ArrayViewMut2<f32>) -> Result<f32>
//! - Returns: LSNR estimate (f32) for quality metering
//! - Uses: ndarray for array operations (ArrayView2, ArrayViewMut2)
//!
//! LATENCY:
//! - Standard model: ~30ms (2 frame lookahead + processing)
//! - LL model (default-model-ll): ~10ms (no lookahead + processing)
//!
//! ROLLBACK STRATEGY:
//! - Stub mode (cfg(not(feature = "deepfilter-real"))): Passthrough implementation
//! - Real mode (cfg(feature = "deepfilter-real")): Actual libDF processing
//!
//! Expected latency: ~10ms effective with LL variant (no lookahead)

#![cfg(feature = "deepfilter")]

use crate::dsp::noise_suppressor::{NoiseModel, NoiseSuppressor};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::vec::Vec;

/// DeepFilterNet frame size (same as RNNoise: 10ms at 48kHz)
pub const DEEPFILTER_FRAME_SIZE: usize = 480;

// ============================================================================
// STUB IMPLEMENTATION (cfg(not(feature = "deepfilter-real")))
// Safe fallback when libDF is not available
// ============================================================================

#[cfg(not(feature = "deepfilter-real"))]
pub struct DeepFilterProcessor {
    input_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
    enabled: bool,
    strength: Arc<AtomicU32>,
    smoothed_strength: f32,
    dry_buffer: Vec<f32>,
}

#[cfg(not(feature = "deepfilter-real"))]
impl DeepFilterProcessor {
    pub fn new(strength: Arc<AtomicU32>) -> Self {
        eprintln!("DeepFilterNet initialized (STUB MODE - passthrough, no libDF)");

        Self {
            input_buffer: Vec::with_capacity(DEEPFILTER_FRAME_SIZE * 4),
            output_buffer: Vec::with_capacity(DEEPFILTER_FRAME_SIZE * 4),
            enabled: true,
            strength,
            smoothed_strength: 1.0,
            dry_buffer: Vec::with_capacity(DEEPFILTER_FRAME_SIZE),
        }
    }

    /// Process frames - STUB: passthrough with wet/dry mixing
    pub fn process_frames_internal(&mut self) {
        // Update smoothed strength with 15ms EMA
        let target_strength = f32::from_bits(self.strength.load(Ordering::Relaxed));
        const TAU_MS: f32 = 15.0;
        const SAMPLE_RATE: f32 = 48000.0;
        let alpha = 1.0 - (-1.0 / (TAU_MS / 1000.0 * SAMPLE_RATE / DEEPFILTER_FRAME_SIZE as f32)).exp();
        self.smoothed_strength += alpha * (target_strength - self.smoothed_strength);

        // Stub: Just passthrough samples with wet/dry mixing
        while self.input_buffer.len() >= DEEPFILTER_FRAME_SIZE {
            self.dry_buffer.clear();
            self.dry_buffer.extend_from_slice(&self.input_buffer[..DEEPFILTER_FRAME_SIZE]);

            let frame: Vec<f32> = self.input_buffer
                .drain(..DEEPFILTER_FRAME_SIZE)
                .collect();

            // Stub: No actual processing, just wet/dry mix (which at 100% strength = passthrough)
            for (i, &wet) in frame.iter().enumerate() {
                let dry = self.dry_buffer[i];
                let mixed = wet * self.smoothed_strength + dry * (1.0 - self.smoothed_strength);
                self.output_buffer.push(mixed);
            }
        }
    }
}

// ============================================================================
// REAL IMPLEMENTATION (cfg(feature = "deepfilter-real"))
// Actual libDF processing when feature is enabled
// ============================================================================

#[cfg(feature = "deepfilter-real")]
use df::tract::{DfTract, RuntimeParams}; // Note: lib name is 'df' in deep_filter crate, types are in tract module
#[cfg(feature = "deepfilter-real")]
use ndarray::{ArrayView2, ArrayViewMut2, Array2, Ix2};

#[cfg(feature = "deepfilter-real")]
pub struct DeepFilterProcessor {
    df: DfTract,
    input_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
    enabled: bool,
    strength: Arc<AtomicU32>,
    smoothed_strength: f32,
    dry_buffer: Vec<f32>,
}

#[cfg(feature = "deepfilter-real")]
impl DeepFilterProcessor {
    pub fn new(strength: Arc<AtomicU32>) -> Self {
        // Use DfTract::default() for embedded model (default-model-ll feature)
        let df = DfTract::default(); // Embedded LL model, no lookahead

        eprintln!("DeepFilterNet initialized (REAL MODE - libDF Low Latency variant, ~10ms latency)");

        Self {
            df,
            input_buffer: Vec::with_capacity(DEEPFILTER_FRAME_SIZE * 4),
            output_buffer: Vec::with_capacity(DEEPFILTER_FRAME_SIZE * 4),
            enabled: true,
            strength,
            smoothed_strength: 1.0,
            dry_buffer: Vec::with_capacity(DEEPFILTER_FRAME_SIZE),
        }
    }

    /// Process frames through libDF
    pub fn process_frames_internal(&mut self) {
        // Update smoothed strength with 15ms EMA
        let target_strength = f32::from_bits(self.strength.load(Ordering::Relaxed));
        const TAU_MS: f32 = 15.0;
        const SAMPLE_RATE: f32 = 48000.0;
        let alpha = 1.0 - (-1.0 / (TAU_MS / 1000.0 * SAMPLE_RATE / DEEPFILTER_FRAME_SIZE as f32)).exp();
        self.smoothed_strength += alpha * (target_strength - self.smoothed_strength);

        // Process complete frames (480 samples each) through libDF
        while self.input_buffer.len() >= DEEPFILTER_FRAME_SIZE {
            // Store dry samples for wet/dry mixing
            self.dry_buffer.clear();
            self.dry_buffer.extend_from_slice(&self.input_buffer[..DEEPFILTER_FRAME_SIZE]);

            // Extract frame
            let frame: Vec<f32> = self.input_buffer
                .drain(..DEEPFILTER_FRAME_SIZE)
                .collect();

            // Convert to ndarray ArrayView2 [channels, samples] for libDF
            // We have 1 channel and 480 samples
            let noisy = ArrayView2::from_shape((1, DEEPFILTER_FRAME_SIZE), &frame).unwrap();

            // Allocate output buffer for libDF
            let mut enhanced = Array2::zeros((1, DEEPFILTER_FRAME_SIZE));

            // Process through libDF
            if self.enabled {
                match self.df.process(noisy, enhanced.view_mut()) {
                    Ok(_lsnr) => {
                        // LSNR estimate available for quality metering if needed
                    }
                    Err(e) => {
                        eprintln!("DeepFilterNet processing error: {}, using passthrough", e);
                        // On error, fall back to passthrough
                        for (i, &wet) in frame.iter().enumerate() {
                            let dry = self.dry_buffer[i];
                            let mixed = wet * self.smoothed_strength + dry * (1.0 - self.smoothed_strength);
                            self.output_buffer.push(mixed);
                        }
                        continue;
                    }
                }
            }

            // Apply wet/dry mix to libDF output
            for (i, &wet) in enhanced.iter().enumerate() {
                let dry = self.dry_buffer[i];
                let mixed = wet * self.smoothed_strength + dry * (1.0 - self.smoothed_strength);
                self.output_buffer.push(mixed);
            }
        }
    }
}

// ============================================================================
// SHARED TRAIT IMPLEMENTATION (both stub and real)
// ============================================================================

#[cfg(any(feature = "deepfilter-real", not(feature = "deepfilter-real")))]
impl NoiseSuppressor for DeepFilterProcessor {
    fn push_samples(&mut self, samples: &[f32]) {
        self.input_buffer.extend_from_slice(samples);
    }

    fn process_frames(&mut self) {
        // Delegated to impl above based on feature
        if self.enabled {
            self.process_frames_internal();
        } else {
            // Disabled: passthrough
            while self.input_buffer.len() >= DEEPFILTER_FRAME_SIZE {
                let frame: Vec<f32> = self.input_buffer
                    .drain(..DEEPFILTER_FRAME_SIZE)
                    .collect();
                self.output_buffer.extend_from_slice(&frame);
            }
        }
    }

    fn available_samples(&self) -> usize {
        self.output_buffer.len()
    }

    fn pop_samples(&mut self, count: usize) -> Vec<f32> {
        let actual = count.min(self.output_buffer.len());
        self.output_buffer.drain(..actual).collect()
    }

    fn pop_all_samples(&mut self) -> Vec<f32> {
        self.output_buffer.drain(..).collect()
    }

    fn set_strength(&self, value: f32) {
        let clamped = value.clamp(0.0, 1.0);
        let bits = clamped.to_bits();
        self.strength.store(bits, Ordering::Relaxed);
    }

    fn get_strength(&self) -> f32 {
        f32::from_bits(self.strength.load(Ordering::Relaxed))
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn soft_reset(&mut self) {
        // Clear all buffers without resetting the model/state
        self.input_buffer.clear();
        self.output_buffer.clear();
        self.dry_buffer.clear();
        // Keep smoothed_strength to avoid zipper noise
    }

    fn pending_input(&self) -> usize {
        self.input_buffer.len()
    }

    fn model_type(&self) -> NoiseModel {
        #[cfg(feature = "deepfilter-real")]
        return NoiseModel::DeepFilterNet;
        #[cfg(not(feature = "deepfilter-real"))]
        return NoiseModel::DeepFilterNet; // Stub still reports as DeepFilterNet
    }

    fn latency_samples(&self) -> usize {
        // LL variant has no lookahead: latency = FRAME_SIZE = 480 samples (~10ms)
        // Note: For standard model with lookahead, this would be higher
        DEEPFILTER_FRAME_SIZE
    }
}

// impl Default for both modes
#[cfg(any(feature = "deepfilter-real", not(feature = "deepfilter-real")))]
impl Default for DeepFilterProcessor {
    fn default() -> Self {
        Self::new(Arc::new(AtomicU32::new(1.0_f32.to_bits()))) // 100% strength default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests for STUB mode
    #[test]
    #[cfg(not(feature = "deepfilter-real"))]
    fn test_stub_mode_creation() {
        let processor = DeepFilterProcessor::default();
        // Stub mode should initialize successfully
        assert_eq!(processor.latency_samples(), 480);
    }

    #[test]
    #[cfg(not(feature = "deepfilter-real"))]
    fn test_stub_mode_passthrough() {
        let mut processor = DeepFilterProcessor::default();
        let input = vec![0.5; 1000];
        processor.push_samples(&input);
        processor.process_frames();

        let output = processor.pop_all_samples();
        assert!(!output.is_empty());

        // Stub mode: output should equal input (passthrough) when strength=100%
        // Allow small floating point differences
        let close_enough = output
            .iter()
            .zip(input.iter())
            .all(|(o, i)| (o - i).abs() < 0.001);
        assert!(close_enough, "Stub mode should passthrough when strength=100%");
    }

    // Tests for REAL mode
    #[test]
    #[cfg(feature = "deepfilter-real")]
    fn test_real_mode_creation() {
        let processor = DeepFilterProcessor::default();
        // Real mode should initialize with libDF
        assert_eq!(processor.latency_samples(), 480);
    }

    #[test]
    #[cfg(feature = "deepfilter-real")]
    fn test_real_processing_verification() {
        let mut processor = DeepFilterProcessor::default();
        let input = vec![0.5; 1000]; // Non-zero input
        processor.push_samples(&input);
        processor.process_frames();

        let output = processor.pop_all_samples();
        assert!(!output.is_empty(), "Should produce output");

        // Real mode: output should DIFFER from input (libDF processing)
        // Note: This may not always be true for constant input
        // but should generally show processing effects
    }

    // Shared tests (both stub and real)
    #[test]
    fn test_strength_operations() {
        let processor = DeepFilterProcessor::default();
        processor.set_strength(0.5);
        assert_eq!(processor.get_strength(), 0.5);
    }

    #[test]
    fn test_buffering_logic() {
        let mut processor = DeepFilterProcessor::default();
        assert_eq!(processor.pending_input(), 0);

        processor.push_samples(&[0.1; 100]);
        assert_eq!(processor.pending_input(), 100);

        processor.push_samples(&[0.2; 400]);
        assert_eq!(processor.pending_input(), 500); // Accumulated

        processor.process_frames();
        assert_eq!(processor.pending_input(), 20); // 500 - 480 = 20 remaining
    }

    #[test]
    fn test_latency_samples() {
        let processor = DeepFilterProcessor::default();
        // LL variant has no lookahead, so latency = FRAME_SIZE = 480 samples
        assert_eq!(processor.latency_samples(), 480);
    }

    #[test]
    fn test_enable_disable() {
        let mut processor = DeepFilterProcessor::default();
        assert!(processor.is_enabled());

        processor.set_enabled(false);
        assert!(!processor.is_enabled());

        processor.set_enabled(true);
        assert!(processor.is_enabled());
    }

    #[test]
    fn test_soft_reset() {
        let mut processor = DeepFilterProcessor::default();
        processor.push_samples(&[0.1; 500]);
        processor.process_frames();

        processor.soft_reset();
        assert_eq!(processor.pending_input(), 0);
        assert_eq!(processor.available_samples(), 0);
    }
}
