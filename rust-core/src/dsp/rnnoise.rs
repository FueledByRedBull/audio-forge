//! RNNoise integration with proper scaling and 480-sample frame buffering

use nnnoiseless::DenoiseState;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// RNNoise frame size (10ms at 48kHz)
pub const RNNOISE_FRAME_SIZE: usize = 480;

/// Scaling factor to map [-1.0, 1.0] to 16-bit range for RNNoise
/// RNNoise expects audio in the range of ~[-32768, 32767]
const PCM_SCALE: f32 = 32768.0;

/// RNNoise processor with frame buffering and wet/dry mix control
///
/// RNNoise requires exactly 480 samples per call. This processor
/// buffers input samples and processes them in valid frame sizes.
pub struct RNNoiseProcessor {
    denoiser: Box<DenoiseState<'static>>,
    input_buffer: Vec<f32>,
    output_buffer: Vec<f32>,
    enabled: bool,
    /// Strength parameter (f32 bits stored as u32 for atomic access)
    /// 0.0 = fully dry (original), 1.0 = fully wet (processed)
    strength: Arc<AtomicU32>,
    /// Current smoothed strength value (DSP thread only)
    /// Updated via exponential moving average to prevent zipper noise
    smoothed_strength: f32,
}

impl RNNoiseProcessor {
    /// Create a new RNNoise processor
    pub fn new(strength: Arc<AtomicU32>) -> Self {
        Self {
            denoiser: DenoiseState::new(),
            input_buffer: Vec::with_capacity(RNNOISE_FRAME_SIZE * 2),
            output_buffer: Vec::with_capacity(RNNOISE_FRAME_SIZE * 2),
            enabled: true,
            strength,
            smoothed_strength: 1.0, // Default: full processing
        }
    }

    /// Set the wet/dry mix strength
    /// 0.0 = fully dry (original signal)
    /// 1.0 = fully wet (processed signal)
    pub fn set_strength(&self, value: f32) {
        let clamped = value.clamp(0.0, 1.0);
        let bits = clamped.to_bits();
        self.strength.store(bits, Ordering::Relaxed);
    }

    /// Get the current wet/dry mix strength
    pub fn get_strength(&self) -> f32 {
        f32::from_bits(self.strength.load(Ordering::Relaxed))
    }

    /// Update smoothed strength using exponential moving average
    /// Returns the current smoothed value
    fn update_smoothing(&mut self) -> f32 {
        let target = f32::from_bits(self.strength.load(Ordering::Relaxed));
        let smoothing_ms: f32 = 15.0;
        let sample_rate: f32 = 48000.0;
        // EMA coefficient: 1 - exp(-1 / (time_constant * sample_rate))
        let coeff = 1.0 - (-1.0_f32 / (smoothing_ms / 1000.0 * sample_rate)).exp();
        self.smoothed_strength = target * coeff + self.smoothed_strength * (1.0 - coeff);
        self.smoothed_strength
    }

    /// Push samples into the input buffer
    pub fn push_samples(&mut self, samples: &[f32]) {
        self.input_buffer.extend_from_slice(samples);
    }

    /// Process any complete frames in the input buffer
    ///
    /// Call this after pushing samples. It will process as many
    /// complete 480-sample frames as possible.
    pub fn process_frames(&mut self) {
        if !self.enabled {
            self.output_buffer.append(&mut self.input_buffer);
            return;
        }

        while self.input_buffer.len() >= RNNOISE_FRAME_SIZE {
            // Store dry samples BEFORE processing (for wet/dry mix)
            let dry_samples: Vec<f32> = self.input_buffer[..RNNOISE_FRAME_SIZE].to_vec();

            // 1. Extract frame and SCALE UP for RNNoise
            let frame: Vec<f32> = self
                .input_buffer
                .drain(..RNNOISE_FRAME_SIZE)
                .map(|s| {
                    // Scale to [-32768, 32767] and CLAMP to prevent wrapping clicks
                    (s * PCM_SCALE).clamp(-32760.0, 32760.0)
                })
                .collect();

            // 2. Process through RNNoise
            let mut output_frame = [0.0f32; RNNOISE_FRAME_SIZE];
            self.denoiser.process_frame(&mut output_frame, &frame);

            // 3. Scale DOWN back to [-1.0, 1.0]
            for sample in output_frame.iter_mut() {
                *sample /= PCM_SCALE;
            }

            // 4. Get smoothed strength and apply wet/dry mix
            let strength = self.update_smoothing();

            if strength < 1.0 {
                // Linear interpolation: output = strength * wet + (1.0 - strength) * dry
                for i in 0..RNNOISE_FRAME_SIZE {
                    let wet = output_frame[i];
                    let dry = dry_samples[i];
                    output_frame[i] = (strength * wet) + ((1.0 - strength) * dry);
                }
            }

            self.output_buffer.extend_from_slice(&output_frame);
        }
    }

    /// Get available output samples
    pub fn available_samples(&self) -> usize {
        self.output_buffer.len()
    }

    /// Pop samples from the output buffer
    pub fn pop_samples(&mut self, count: usize) -> Vec<f32> {
        let actual_count = count.min(self.output_buffer.len());
        self.output_buffer.drain(..actual_count).collect()
    }

    /// Pop all available samples from the output buffer
    pub fn pop_all_samples(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.output_buffer)
    }

    /// Read samples into provided buffer, returns actual count read
    pub fn read_samples(&mut self, buffer: &mut [f32]) -> usize {
        let count = buffer.len().min(self.output_buffer.len());
        for (i, sample) in self.output_buffer.drain(..count).enumerate() {
            buffer[i] = sample;
        }
        count
    }

    /// Enable or disable RNNoise processing
    ///
    /// Note: Disabling does not reset state - audio passes through
    /// but the model state is preserved for quick re-enabling.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if RNNoise is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Reset the processor state
    ///
    /// Warning: This causes ~200ms of convergence time when
    /// processing resumes. Prefer using set_enabled(false) for
    /// temporary bypass.
    pub fn reset(&mut self) {
        self.denoiser = DenoiseState::new();
        self.input_buffer.clear();
        self.output_buffer.clear();
    }

    /// Flush internal buffers without resetting DenoiseState
    ///
    /// This clears input and output buffers to prevent stale audio data
    /// from being processed when re-enabling, while preserving the
    /// RNNoise model state (avoids 200ms convergence time).
    pub fn flush_buffers(&mut self) {
        self.input_buffer.clear();
        self.output_buffer.clear();
    }

    /// Soft reset: clear buffers without resetting model state
    ///
    /// This is the same as flush_buffers() and is preferred over reset()
    /// when stopping/restarting processing, as it preserves the RNNoise
    /// model's learned background noise profile (avoids 200ms convergence).
    pub fn soft_reset(&mut self) {
        self.flush_buffers();
    }

    /// Get pending input samples count
    pub fn pending_input(&self) -> usize {
        self.input_buffer.len()
    }

    /// Drain and return pending input samples without processing
    ///
    /// This is used when we want to bypass RNNoise and output raw audio.
    /// Returns the pending samples that haven't been processed yet.
    pub fn drain_pending_input(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.input_buffer)
    }
}

impl Default for RNNoiseProcessor {
    fn default() -> Self {
        Self::new(Arc::new(AtomicU32::new(1.0_f32.to_bits())))
    }
}

// Implement NoiseSuppressor trait for runtime model selection
impl super::noise_suppressor::NoiseSuppressor for RNNoiseProcessor {
    fn push_samples(&mut self, samples: &[f32]) {
        self.input_buffer.extend_from_slice(samples);
    }

    fn process_frames(&mut self) {
        RNNoiseProcessor::process_frames(self);
    }

    fn available_samples(&self) -> usize {
        self.output_buffer.len()
    }

    fn pop_samples(&mut self, count: usize) -> Vec<f32> {
        let actual_count = count.min(self.output_buffer.len());
        self.output_buffer.drain(..actual_count).collect()
    }

    fn pop_all_samples(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.output_buffer)
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
        self.input_buffer.clear();
        self.output_buffer.clear();
    }

    fn pending_input(&self) -> usize {
        self.input_buffer.len()
    }

    fn drain_pending_input(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.input_buffer)
    }

    fn model_type(&self) -> super::noise_suppressor::NoiseModel {
        super::noise_suppressor::NoiseModel::RNNoise
    }

    fn latency_samples(&self) -> usize {
        RNNOISE_FRAME_SIZE // 480 samples = 10ms at 48kHz
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rnnoise_frame_buffering() {
        let strength = Arc::new(AtomicU32::new(1.0_f32.to_bits()));
        let mut processor = RNNoiseProcessor::new(strength);

        // Push less than a frame
        processor.push_samples(&[0.0; 400]);
        processor.process_frames();
        assert_eq!(processor.available_samples(), 0);
        assert_eq!(processor.pending_input(), 400);

        // Push more to complete a frame
        processor.push_samples(&[0.0; 100]);
        processor.process_frames();
        assert_eq!(processor.available_samples(), 480);
        assert_eq!(processor.pending_input(), 20);
    }

    #[test]
    fn test_rnnoise_bypass() {
        let strength = Arc::new(AtomicU32::new(1.0_f32.to_bits()));
        let mut processor = RNNoiseProcessor::new(strength);
        processor.set_enabled(false);

        processor.push_samples(&[1.0; 100]);
        processor.process_frames();

        // Should pass through immediately when disabled
        assert_eq!(processor.available_samples(), 100);
    }

    #[test]
    fn test_rnnoise_strength_getter_setter() {
        let strength = Arc::new(AtomicU32::new(1.0_f32.to_bits()));
        let processor = RNNoiseProcessor::new(strength);

        // Test default strength
        assert_eq!(processor.get_strength(), 1.0);

        // Test setting strength
        processor.set_strength(0.5);
        assert_eq!(processor.get_strength(), 0.5);

        // Test clamping
        processor.set_strength(1.5);
        assert_eq!(processor.get_strength(), 1.0);

        processor.set_strength(-0.5);
        assert_eq!(processor.get_strength(), 0.0);
    }

    #[test]
    fn test_rnnoise_wet_dry_mix() {
        let strength = Arc::new(AtomicU32::new(0.5_f32.to_bits()));
        let mut processor = RNNoiseProcessor::new(strength);

        // Push exactly one frame
        processor.push_samples(&[0.5; 480]);
        processor.process_frames();

        // Should have 480 samples available (processed with mix)
        assert_eq!(processor.available_samples(), 480);
    }
}
