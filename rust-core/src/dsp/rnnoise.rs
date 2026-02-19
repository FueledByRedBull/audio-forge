//! RNNoise integration with proper scaling and 480-sample frame buffering

use crate::audio::input::TARGET_SAMPLE_RATE;
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
    input_read_pos: usize,
    output_buffer: Vec<f32>,
    output_read_pos: usize,
    dry_scratch: [f32; RNNOISE_FRAME_SIZE],
    frame_scratch: [f32; RNNOISE_FRAME_SIZE],
    output_frame: [f32; RNNOISE_FRAME_SIZE],
    enabled: bool,
    /// Strength parameter (f32 bits stored as u32 for atomic access)
    /// 0.0 = fully dry (original), 1.0 = fully wet (processed)
    strength: Arc<AtomicU32>,
    /// Current smoothed strength value (DSP thread only)
    /// Updated via exponential moving average to prevent zipper noise
    smoothed_strength: f32,
    /// Precomputed smoothing coefficient to avoid per-frame `exp`.
    smoothing_coeff: f32,
}

impl RNNoiseProcessor {
    /// Create a new RNNoise processor
    pub fn new(strength: Arc<AtomicU32>) -> Self {
        let sample_rate = TARGET_SAMPLE_RATE as f32;
        let smoothing_ms = 15.0_f32;
        let smoothing_coeff = 1.0 - (-1.0_f32 / (smoothing_ms / 1000.0 * sample_rate)).exp();
        Self {
            denoiser: DenoiseState::new(),
            input_buffer: Vec::with_capacity(RNNOISE_FRAME_SIZE * 2),
            input_read_pos: 0,
            output_buffer: Vec::with_capacity(RNNOISE_FRAME_SIZE * 2),
            output_read_pos: 0,
            dry_scratch: [0.0; RNNOISE_FRAME_SIZE],
            frame_scratch: [0.0; RNNOISE_FRAME_SIZE],
            output_frame: [0.0; RNNOISE_FRAME_SIZE],
            enabled: true,
            strength,
            smoothed_strength: 1.0, // Default: full processing
            smoothing_coeff,
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
        self.smoothed_strength =
            target * self.smoothing_coeff + self.smoothed_strength * (1.0 - self.smoothing_coeff);
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
            if self.input_read_pos > 0 {
                self.output_buffer
                    .extend_from_slice(&self.input_buffer[self.input_read_pos..]);
                self.input_buffer.clear();
                self.input_read_pos = 0;
            } else {
                self.output_buffer.append(&mut self.input_buffer);
            }
            return;
        }

        while self.input_buffer.len().saturating_sub(self.input_read_pos) >= RNNOISE_FRAME_SIZE {
            let start = self.input_read_pos;
            let end = start + RNNOISE_FRAME_SIZE;
            let input_frame = &self.input_buffer[start..end];
            self.dry_scratch.copy_from_slice(input_frame);

            // Scale to RNNoise PCM-like range.
            for (dst, &src) in self.frame_scratch.iter_mut().zip(input_frame.iter()) {
                *dst = (src * PCM_SCALE).clamp(-32760.0, 32760.0);
            }

            // 2. Process through RNNoise
            self.denoiser
                .process_frame(&mut self.output_frame, &self.frame_scratch);

            // 3. Scale DOWN back to [-1.0, 1.0]
            for sample in &mut self.output_frame {
                *sample /= PCM_SCALE;
            }

            // 4. Get smoothed strength and apply wet/dry mix
            let strength = self.update_smoothing();

            if strength < 1.0 {
                // Linear interpolation: output = strength * wet + (1.0 - strength) * dry
                for i in 0..RNNOISE_FRAME_SIZE {
                    let wet = self.output_frame[i];
                    let dry = self.dry_scratch[i];
                    self.output_frame[i] = (strength * wet) + ((1.0 - strength) * dry);
                }
            }

            self.output_buffer.extend_from_slice(&self.output_frame);
            self.input_read_pos = end;
        }

        // Compact occasionally to keep the active window contiguous.
        if self.input_read_pos >= RNNOISE_FRAME_SIZE
            && self.input_read_pos.saturating_mul(2) >= self.input_buffer.len()
        {
            self.input_buffer.drain(..self.input_read_pos);
            self.input_read_pos = 0;
        }
    }

    /// Get available output samples
    pub fn available_samples(&self) -> usize {
        self.output_buffer.len().saturating_sub(self.output_read_pos)
    }

    /// Pop samples from the output buffer
    pub fn pop_samples(&mut self, count: usize) -> Vec<f32> {
        let actual_count = count.min(self.available_samples());
        let start = self.output_read_pos;
        let end = start + actual_count;
        let out = self.output_buffer[start..end].to_vec();
        self.output_read_pos = end;
        if self.output_read_pos >= RNNOISE_FRAME_SIZE
            && self.output_read_pos.saturating_mul(2) >= self.output_buffer.len()
        {
            self.output_buffer.drain(..self.output_read_pos);
            self.output_read_pos = 0;
        }
        out
    }

    /// Pop all available samples from the output buffer
    pub fn pop_all_samples(&mut self) -> Vec<f32> {
        let out = self.output_buffer[self.output_read_pos..].to_vec();
        self.output_buffer.clear();
        self.output_read_pos = 0;
        out
    }

    /// Read samples into provided buffer, returns actual count read
    pub fn read_samples(&mut self, buffer: &mut [f32]) -> usize {
        let count = buffer.len().min(self.available_samples());
        let start = self.output_read_pos;
        let end = start + count;
        buffer[..count].copy_from_slice(&self.output_buffer[start..end]);
        self.output_read_pos = end;
        if self.output_read_pos >= RNNOISE_FRAME_SIZE
            && self.output_read_pos.saturating_mul(2) >= self.output_buffer.len()
        {
            self.output_buffer.drain(..self.output_read_pos);
            self.output_read_pos = 0;
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
        self.input_read_pos = 0;
        self.output_read_pos = 0;
    }

    /// Flush internal buffers without resetting DenoiseState
    ///
    /// This clears input and output buffers to prevent stale audio data
    /// from being processed when re-enabling, while preserving the
    /// RNNoise model state (avoids 200ms convergence time).
    pub fn flush_buffers(&mut self) {
        self.input_buffer.clear();
        self.output_buffer.clear();
        self.input_read_pos = 0;
        self.output_read_pos = 0;
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
        self.input_buffer.len().saturating_sub(self.input_read_pos)
    }

    /// Drain and return pending input samples without processing
    ///
    /// This is used when we want to bypass RNNoise and output raw audio.
    /// Returns the pending samples that haven't been processed yet.
    pub fn drain_pending_input(&mut self) -> Vec<f32> {
        let pending = self.input_buffer[self.input_read_pos..].to_vec();
        self.input_buffer.clear();
        self.input_read_pos = 0;
        pending
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
        RNNoiseProcessor::push_samples(self, samples);
    }

    fn process_frames(&mut self) {
        RNNoiseProcessor::process_frames(self);
    }

    fn available_samples(&self) -> usize {
        RNNoiseProcessor::available_samples(self)
    }

    fn pop_samples(&mut self, count: usize) -> Vec<f32> {
        RNNoiseProcessor::pop_samples(self, count)
    }

    fn pop_samples_into(&mut self, buffer: &mut [f32]) -> usize {
        self.read_samples(buffer)
    }

    fn pop_all_samples(&mut self) -> Vec<f32> {
        RNNoiseProcessor::pop_all_samples(self)
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
        RNNoiseProcessor::set_enabled(self, enabled);
    }

    fn is_enabled(&self) -> bool {
        RNNoiseProcessor::is_enabled(self)
    }

    fn soft_reset(&mut self) {
        RNNoiseProcessor::soft_reset(self);
    }

    fn pending_input(&self) -> usize {
        RNNoiseProcessor::pending_input(self)
    }

    fn drain_pending_input(&mut self) -> Vec<f32> {
        RNNoiseProcessor::drain_pending_input(self)
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
