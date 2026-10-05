//! RNNoise integration with proper scaling and 480-sample frame buffering

use super::noise_suppressor::{ControlledSample, GateCompensation, GateControl};
use crate::audio::input::TARGET_SAMPLE_RATE;
use crate::audio::rt::FixedAudioRing;
use nnnoiseless::DenoiseState;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// RNNoise frame size (10ms at 48kHz)
pub const RNNOISE_FRAME_SIZE: usize = 480;
const RNNOISE_BUFFER_CAPACITY: usize = 8192 + RNNOISE_FRAME_SIZE;

/// Scaling factor to map [-1.0, 1.0] to 16-bit range for RNNoise
/// RNNoise expects audio in the range of ~[-32768, 32767]
const PCM_SCALE: f32 = 32768.0;
const PCM_MODEL_LIMIT: f32 = 32760.0;
const PCM_MODEL_LIMIT_UNIT: f32 = PCM_MODEL_LIMIT / PCM_SCALE;
const MODEL_SOFT_CLIP_THRESHOLD: f32 = 0.98;
const MODEL_SOFT_CLIP_KNEE: f32 = 1.0 - MODEL_SOFT_CLIP_THRESHOLD;

/// RNNoise processor with frame buffering and wet/dry mix control
///
/// RNNoise requires exactly 480 samples per call. This processor
/// buffers input samples and processes them in valid frame sizes.
pub struct RNNoiseProcessor {
    denoiser: Box<DenoiseState<'static>>,
    input_buffer: FixedAudioRing<ControlledSample, RNNOISE_BUFFER_CAPACITY>,
    output_buffer: FixedAudioRing<f32, RNNOISE_BUFFER_CAPACITY>,
    input_frame: [ControlledSample; RNNOISE_FRAME_SIZE],
    gate_compensation: GateCompensation,
    dry_scratch: [f32; RNNOISE_FRAME_SIZE],
    dry_delay_frame: [f32; RNNOISE_FRAME_SIZE],
    invalid_wet_samples: usize,
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
        let smoothing_tau_s = smoothing_ms / 1000.0;
        let frame_dt_s = RNNOISE_FRAME_SIZE as f32 / sample_rate;
        let smoothing_coeff = 1.0 - (-(frame_dt_s / smoothing_tau_s)).exp();
        let requested_strength = f32::from_bits(strength.load(Ordering::Relaxed));
        let initial_strength = if requested_strength.is_finite() {
            requested_strength.clamp(0.0, 1.0)
        } else {
            1.0
        };
        Self {
            denoiser: DenoiseState::new(),
            input_buffer: FixedAudioRing::new(),
            output_buffer: FixedAudioRing::new(),
            input_frame: [ControlledSample::default(); RNNOISE_FRAME_SIZE],
            gate_compensation: GateCompensation::new(RNNOISE_FRAME_SIZE),
            dry_scratch: [0.0; RNNOISE_FRAME_SIZE],
            dry_delay_frame: [0.0; RNNOISE_FRAME_SIZE],
            invalid_wet_samples: 0,
            frame_scratch: [0.0; RNNOISE_FRAME_SIZE],
            output_frame: [0.0; RNNOISE_FRAME_SIZE],
            enabled: true,
            strength,
            smoothed_strength: initial_strength,
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

    #[inline]
    fn soft_clip_model_input_sample(sample: f32) -> f32 {
        if !sample.is_finite() {
            return 0.0;
        }

        let sign = sample.signum();
        let magnitude = sample.abs();
        if magnitude <= MODEL_SOFT_CLIP_THRESHOLD {
            return sample;
        }

        let over = magnitude - MODEL_SOFT_CLIP_THRESHOLD;
        let compressed = over / (over + MODEL_SOFT_CLIP_KNEE);
        let softened = MODEL_SOFT_CLIP_THRESHOLD
            + (PCM_MODEL_LIMIT_UNIT - MODEL_SOFT_CLIP_THRESHOLD) * compressed;
        sign * softened.min(PCM_MODEL_LIMIT_UNIT)
    }

    #[inline]
    fn scale_sample_for_model(sample: f32) -> f32 {
        (Self::soft_clip_model_input_sample(sample) * PCM_SCALE)
            .clamp(-PCM_MODEL_LIMIT, PCM_MODEL_LIMIT)
    }

    /// Push samples into the input buffer
    pub fn push_samples(&mut self, samples: &[f32]) -> usize {
        self.push_samples_with_controls(samples, None)
    }

    fn push_samples_with_controls(
        &mut self,
        samples: &[f32],
        controls: Option<&[GateControl]>,
    ) -> usize {
        if let Some(controls) = controls {
            assert_eq!(samples.len(), controls.len());
        }
        let accepted = samples.len().min(self.input_buffer.remaining());
        for (index, &sample) in samples[..accepted].iter().enumerate() {
            let gate = controls.map_or(GateControl::BYPASS, |controls| controls[index]);
            self.input_buffer.push(ControlledSample { sample, gate });
        }
        accepted
    }

    /// Process any complete frames in the input buffer
    ///
    /// Call this after pushing samples. It will process as many
    /// complete 480-sample frames as possible.
    pub fn process_frames(&mut self) {
        if !self.enabled {
            let count = self.input_buffer.len().min(self.output_buffer.remaining());
            let mut sample = [ControlledSample::default(); 1];
            for _ in 0..count {
                self.input_buffer.pop_into(&mut sample);
                self.output_buffer.push(sample[0].sample);
            }
            return;
        }

        while self.input_buffer.len() >= RNNOISE_FRAME_SIZE
            && self.output_buffer.remaining() >= RNNOISE_FRAME_SIZE
        {
            let read = self.input_buffer.pop_into(&mut self.input_frame);
            if read != RNNOISE_FRAME_SIZE {
                break;
            }

            for (dry, input) in self.dry_scratch.iter_mut().zip(&self.input_frame) {
                *dry = input.sample;
            }
            // Scale to RNNoise PCM-like range.
            for (dst, &src) in self.frame_scratch.iter_mut().zip(self.dry_scratch.iter()) {
                *dst = Self::scale_sample_for_model(src);
            }

            // 2. Process through RNNoise
            self.denoiser
                .process_frame(&mut self.output_frame, &self.frame_scratch);

            // 3. Scale DOWN back to [-1.0, 1.0]
            for sample in &mut self.output_frame {
                *sample /= PCM_SCALE;
            }

            // A soft reset keeps the model's overlap, which belongs to the old
            // stream until its already-reported latency has elapsed.
            let invalid = self.invalid_wet_samples.min(RNNOISE_FRAME_SIZE);
            self.output_frame[..invalid].fill(0.0);
            self.invalid_wet_samples -= invalid;

            for (wet, input) in self.output_frame.iter_mut().zip(&self.input_frame) {
                let ratio = self.gate_compensation.next_ratio(input.gate);
                if ratio != 1.0 {
                    *wet *= ratio;
                }
            }

            // 4. Get smoothed strength and apply wet/dry mix
            let strength = self.update_smoothing();

            if strength < 1.0 {
                // RNNoise emits one frame late, so mix against the matching dry frame.
                for i in 0..RNNOISE_FRAME_SIZE {
                    let wet = self.output_frame[i];
                    let dry = self.dry_delay_frame[i];
                    self.output_frame[i] = (strength * wet) + ((1.0 - strength) * dry);
                }
            }
            self.dry_delay_frame.copy_from_slice(&self.dry_scratch);

            self.output_buffer.push_slice(&self.output_frame);
        }
    }

    /// Get available output samples
    pub fn available_samples(&self) -> usize {
        self.output_buffer.len()
    }

    /// Read samples into provided buffer, returns actual count read
    pub fn read_samples(&mut self, buffer: &mut [f32]) -> usize {
        let count = buffer.len().min(self.available_samples());
        self.output_buffer.pop_into(&mut buffer[..count])
    }

    /// Enable or disable RNNoise processing
    ///
    /// Note: Disabling does not reset state - audio passes through
    /// but the model state is preserved for quick re-enabling.
    pub fn set_enabled(&mut self, enabled: bool) {
        if self.enabled != enabled {
            self.gate_compensation.reset();
        }
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
        self.dry_delay_frame.fill(0.0);
        self.invalid_wet_samples = 0;
        self.gate_compensation.reset();
    }

    /// Flush internal buffers without resetting DenoiseState
    ///
    /// This clears input and output buffers to prevent stale audio data
    /// from being processed when re-enabling, while preserving the
    /// RNNoise model state (avoids 200ms convergence time).
    pub fn flush_buffers(&mut self) {
        self.input_buffer.clear();
        self.output_buffer.clear();
        self.dry_delay_frame.fill(0.0);
        self.invalid_wet_samples = RNNOISE_FRAME_SIZE;
        self.gate_compensation.reset();
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
}

impl Default for RNNoiseProcessor {
    fn default() -> Self {
        Self::new(Arc::new(AtomicU32::new(1.0_f32.to_bits())))
    }
}

impl super::noise_suppressor::ControlledNoiseSuppressor for RNNoiseProcessor {
    fn push_controlled_samples(&mut self, samples: &[f32], controls: &[GateControl]) -> usize {
        self.push_samples_with_controls(samples, Some(controls))
    }
}

// Implement NoiseSuppressor trait for runtime model selection
impl super::noise_suppressor::NoiseSuppressor for RNNoiseProcessor {
    fn push_samples(&mut self, samples: &[f32]) -> usize {
        RNNoiseProcessor::push_samples(self, samples)
    }

    fn process_frames(&mut self) {
        RNNoiseProcessor::process_frames(self);
    }

    fn available_samples(&self) -> usize {
        RNNoiseProcessor::available_samples(self)
    }

    fn pop_samples_into(&mut self, buffer: &mut [f32]) -> usize {
        self.read_samples(buffer)
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

    fn model_type(&self) -> super::noise_suppressor::NoiseModel {
        super::noise_suppressor::NoiseModel::RNNoise
    }

    fn latency_samples(&self) -> usize {
        RNNOISE_FRAME_SIZE // 480 samples = 10ms at 48kHz
    }

    fn backend_available(&self) -> bool {
        true
    }

    fn backend_error(&self) -> Option<&str> {
        None
    }

    fn backend_failed(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn controlled_short_writes_keep_audio_and_gate_records_together() {
        use super::super::noise_suppressor::ControlledNoiseSuppressor;
        let mut processor = RNNoiseProcessor::default();
        let length = RNNOISE_BUFFER_CAPACITY + 317;
        let samples: Vec<_> = (0..length).map(|index| index as f32 * 0.00001).collect();
        let controls: Vec<_> = (0..length)
            .map(|index| GateControl {
                gain: (index % 701) as f32 / 701.0,
                open: index % 13 < 7,
            })
            .collect();
        assert_eq!(
            processor.push_controlled_samples(&samples, &controls),
            RNNOISE_BUFFER_CAPACITY
        );
        assert_eq!(
            processor.push_controlled_samples(&samples[..1], &controls[..1]),
            0
        );
        let mut prefix = [ControlledSample::default(); 317];
        assert_eq!(processor.input_buffer.pop_into(&mut prefix), 317);
        assert_eq!(
            processor.push_controlled_samples(
                &samples[RNNOISE_BUFFER_CAPACITY..],
                &controls[RNNOISE_BUFFER_CAPACITY..]
            ),
            317
        );
        let mut remaining = vec![ControlledSample::default(); RNNOISE_BUFFER_CAPACITY];
        assert_eq!(
            processor.input_buffer.pop_into(&mut remaining),
            RNNOISE_BUFFER_CAPACITY
        );
        for (index, record) in prefix.iter().chain(&remaining).enumerate() {
            assert_eq!(record.sample, samples[index]);
            assert_eq!(record.gate, controls[index]);
        }
    }

    #[test]
    fn gated_input_mix_keeps_frame_strength_and_partition_contract() {
        for partition in [1, 127, RNNOISE_FRAME_SIZE] {
            let mut mixed = RNNoiseProcessor::new(Arc::new(AtomicU32::new(0.0_f32.to_bits())));
            let mut wet = RNNoiseProcessor::new(Arc::new(AtomicU32::new(1.0_f32.to_bits())));
            let mut previous_dry = [0.0; RNNOISE_FRAME_SIZE];
            let mut expected_strength = 0.0_f32;
            let alpha = 1.0 - (-10.0_f32 / 15.0).exp();
            for (frame, target) in [0.0_f32, 1.0, 1.0, 0.37, 0.0, 0.0].into_iter().enumerate() {
                let raw = std::array::from_fn::<_, RNNOISE_FRAME_SIZE, _>(|index| {
                    ((index + frame * RNNOISE_FRAME_SIZE) as f32 * 0.031).sin() * 0.15
                });
                let gains = std::array::from_fn::<_, RNNOISE_FRAME_SIZE, _>(|index| {
                    if (index / 73 + frame) % 2 == 0 {
                        0.08_f32
                    } else {
                        0.9
                    }
                });
                let dry = std::array::from_fn::<_, RNNOISE_FRAME_SIZE, _>(|index| {
                    raw[index] * gains[index]
                });
                mixed.set_strength(target);
                for start in (0..RNNOISE_FRAME_SIZE).step_by(partition) {
                    let end = (start + partition).min(RNNOISE_FRAME_SIZE);
                    assert_eq!(mixed.push_samples(&dry[start..end]), end - start);
                    mixed.process_frames();
                }
                wet.push_samples(&dry);
                wet.process_frames();
                let mut actual = [0.0; RNNOISE_FRAME_SIZE];
                let mut full_wet = [0.0; RNNOISE_FRAME_SIZE];
                assert_eq!(mixed.read_samples(&mut actual), RNNOISE_FRAME_SIZE);
                assert_eq!(wet.read_samples(&mut full_wet), RNNOISE_FRAME_SIZE);
                expected_strength = target * alpha + expected_strength * (1.0 - alpha);
                assert_eq!(mixed.smoothed_strength, expected_strength);
                for index in 0..RNNOISE_FRAME_SIZE {
                    let expected = expected_strength * full_wet[index]
                        + (1.0 - expected_strength) * previous_dry[index];
                    assert!((actual[index] - expected).abs() <= 1.0e-7);
                }
                previous_dry = dry;
            }
        }
    }

    #[test]
    fn soft_reset_clears_partial_and_output_without_allocating() {
        let mut processor = RNNoiseProcessor::new(Arc::new(AtomicU32::new(0.37_f32.to_bits())));
        processor.push_samples(&[0.1; 600]);
        processor.process_frames();
        assert_eq!(processor.pending_input(), 120);
        assert_eq!(processor.available_samples(), 480);
        let strength = processor.smoothed_strength;
        crate::test_alloc::assert_no_allocations("RNNoise soft reset and processing", || {
            processor.soft_reset();
            assert_eq!(processor.pending_input(), 0);
            assert_eq!(processor.available_samples(), 0);
            assert_eq!(processor.smoothed_strength, strength);
            processor.push_samples(&[0.0; 480]);
            processor.process_frames();
            let mut output = [1.0; 480];
            assert_eq!(processor.read_samples(&mut output), 480);
            assert!(output.iter().all(|sample| *sample == 0.0));
        });
        processor.soft_reset();
        processor.set_enabled(false);
        processor.push_samples(&[0.1, 0.4]);
        processor.process_frames();
        let mut output = [0.0; 2];
        assert_eq!(processor.read_samples(&mut output), 2);
        assert_eq!(output, [0.1, 0.4]);
    }

    #[test]
    fn soft_reset_does_not_emit_model_overlap_from_before_gap() {
        let before = std::array::from_fn::<_, RNNOISE_FRAME_SIZE, _>(|index| {
            (index as f32 * 0.047).sin() * 0.25
        });
        let after = std::array::from_fn::<_, RNNOISE_FRAME_SIZE, _>(|index| {
            (index as f32 * 0.091).sin() * -0.17
        });
        for gain in [0.3, 1.0] {
            for strength in [0.0_f32, 0.37, 1.0] {
                let mut processor =
                    RNNoiseProcessor::new(Arc::new(AtomicU32::new(strength.to_bits())));
                let mut output = [0.0; RNNOISE_FRAME_SIZE];
                for _ in 0..20 {
                    processor.push_samples(&before);
                    processor.process_frames();
                    processor.read_samples(&mut output);
                }
                assert!(output.iter().any(|sample| *sample != 0.0));
                processor.push_samples(&before[..123]);
                processor.soft_reset();
                assert_eq!(processor.pending_input(), 0);
                let dry = after.map(|sample| sample * gain);
                for frame in 0..2 {
                    processor.push_samples(&dry);
                    processor.process_frames();
                    assert_eq!(processor.read_samples(&mut output), 480);
                    if frame == 0 {
                        assert!(output.iter().all(|sample| *sample == 0.0));
                    } else if strength == 0.0 {
                        assert_eq!(output, dry);
                    } else {
                        assert!(output.iter().any(|sample| *sample != 0.0));
                    }
                }
            }
        }
    }

    #[test]
    fn test_rnnoise_model_input_soft_clip_transfer() {
        let below = RNNoiseProcessor::scale_sample_for_model(0.5);
        assert!((below - 0.5 * PCM_SCALE).abs() < 1e-3);

        let near_full_scale = RNNoiseProcessor::scale_sample_for_model(1.0);
        assert!(near_full_scale > MODEL_SOFT_CLIP_THRESHOLD * PCM_SCALE);
        assert!(near_full_scale < PCM_MODEL_LIMIT);

        let louder = RNNoiseProcessor::scale_sample_for_model(1.5);
        assert!(louder > near_full_scale);
        assert!(louder <= PCM_MODEL_LIMIT);

        let negative = RNNoiseProcessor::scale_sample_for_model(-1.0);
        assert!((negative + near_full_scale).abs() < 1e-3);

        let non_finite = RNNoiseProcessor::scale_sample_for_model(f32::NAN);
        assert_eq!(non_finite, 0.0);
    }

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
    fn test_disabled_rnnoise_preserves_hot_input_samples() {
        let strength = Arc::new(AtomicU32::new(1.0_f32.to_bits()));
        let mut processor = RNNoiseProcessor::new(strength);
        processor.set_enabled(false);

        let input = [1.25, -1.5, 0.25, -0.75];
        processor.push_samples(&input);
        processor.process_frames();

        let mut output = [0.0; 4];
        assert_eq!(processor.read_samples(&mut output), input.len());
        assert_eq!(output, input);
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

    #[test]
    fn test_rnnoise_dry_mix_matches_reported_frame_latency() {
        let strength = Arc::new(AtomicU32::new(0.0_f32.to_bits()));
        let mut processor = RNNoiseProcessor::new(strength);
        let input: Vec<f32> = (0..RNNOISE_FRAME_SIZE)
            .map(|index| (index as f32 * 0.017).sin() * 0.25)
            .collect();

        processor.push_samples(&input);
        processor.process_frames();

        let mut output = [0.0; RNNOISE_FRAME_SIZE];
        assert_eq!(processor.read_samples(&mut output), RNNOISE_FRAME_SIZE);
        assert_eq!(output, [0.0; RNNOISE_FRAME_SIZE]);

        processor.push_samples(&[0.0; RNNOISE_FRAME_SIZE]);
        processor.process_frames();
        assert_eq!(processor.read_samples(&mut output), RNNOISE_FRAME_SIZE);
        assert_eq!(output.as_slice(), input.as_slice());
    }

    #[test]
    fn test_rnnoise_partial_strength_uses_the_aligned_dry_frame() {
        let input: Vec<f32> = (0..RNNOISE_FRAME_SIZE)
            .map(|index| {
                let phase = 2.0 * std::f32::consts::PI * index as f32 / 73.0;
                0.18 * phase.sin() + 0.07 * (phase * 2.37).sin()
            })
            .collect();
        let render = |strength: f32| {
            let mut processor = RNNoiseProcessor::new(Arc::new(AtomicU32::new(strength.to_bits())));
            let mut output = [0.0; RNNOISE_FRAME_SIZE];
            processor.push_samples(&input);
            processor.process_frames();
            assert_eq!(processor.read_samples(&mut output), RNNOISE_FRAME_SIZE);
            processor.push_samples(&[0.0; RNNOISE_FRAME_SIZE]);
            processor.process_frames();
            assert_eq!(processor.read_samples(&mut output), RNNOISE_FRAME_SIZE);
            output
        };

        let dry = render(0.0);
        let wet = render(1.0);
        for strength in [0.25_f32, 0.5, 0.75] {
            let mixed = render(strength);
            for index in 0..RNNOISE_FRAME_SIZE {
                let expected = dry[index] * (1.0 - strength) + wet[index] * strength;
                assert!((mixed[index] - expected).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn test_rnnoise_output_stays_finite_for_clipped_input() {
        let strength = Arc::new(AtomicU32::new(1.0_f32.to_bits()));
        let mut processor = RNNoiseProcessor::new(strength);

        for n in 0..RNNOISE_FRAME_SIZE {
            let sample = if n % 2 == 0 { 1.0 } else { -1.0 };
            processor.push_samples(&[sample]);
        }
        processor.process_frames();

        let mut output = [0.0; RNNOISE_FRAME_SIZE];
        assert_eq!(processor.read_samples(&mut output), RNNOISE_FRAME_SIZE);
        for sample in output {
            assert!(sample.is_finite());
            assert!(sample.abs() <= 2.0);
        }
    }
}
