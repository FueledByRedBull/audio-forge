//! IIR Biquad filter implementation using Direct Form II Transposed
//!
//! Uses f64 internally for coefficient precision, f32 for audio samples, and a
//! sample-rate-scaled 1.5 ms parallel-state crossfade for live coefficient edits.

use std::f64::consts::PI;

const MIN_BIQUAD_Q: f64 = 1e-6;
const COEFF_CROSSFADE_MS: f64 = 1.5;
const MAX_COEFF_CROSSFADE_SAMPLES: usize = 4096;

fn coefficient_crossfade_samples(sample_rate: f64) -> usize {
    let samples = (sample_rate * COEFF_CROSSFADE_MS / 1000.0).round();
    if samples.is_finite() {
        (samples as usize).clamp(1, MAX_COEFF_CROSSFADE_SAMPLES)
    } else {
        1
    }
}

/// Biquad filter types
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BiquadType {
    LowShelf,
    HighShelf,
    Peaking,
    HighPass,
    LowPass,
}

/// IIR Biquad filter using Direct Form II Transposed
///
/// Transfer function: H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
///
/// Uses f64 internally for coefficient precision to avoid numerical issues,
/// but processes f32 audio samples for compatibility with audio drivers.
#[derive(Clone, Debug)]
pub struct Biquad {
    // Coefficients (normalized so a0 = 1)
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    pending_b0: f64,
    pending_b1: f64,
    pending_b2: f64,
    pending_a1: f64,
    pending_a2: f64,
    pending_z1: f64,
    pending_z2: f64,
    crossfade_total: usize,
    crossfade_remaining: usize,

    // State variables for Direct Form II Transposed
    z1: f64,
    z2: f64,

    // Parameters for recalculation
    filter_type: BiquadType,
    frequency: f64,
    gain_db: f64,
    q: f64,
    sample_rate: f64,
    enabled: bool,
}

impl Biquad {
    /// Create a new biquad filter
    pub fn new(
        filter_type: BiquadType,
        frequency: f64,
        gain_db: f64,
        q: f64,
        sample_rate: f64,
    ) -> Self {
        let mut filter = Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            pending_b0: 1.0,
            pending_b1: 0.0,
            pending_b2: 0.0,
            pending_a1: 0.0,
            pending_a2: 0.0,
            pending_z1: 0.0,
            pending_z2: 0.0,
            crossfade_total: 0,
            crossfade_remaining: 0,
            z1: 0.0,
            z2: 0.0,
            filter_type,
            frequency,
            gain_db,
            q,
            sample_rate,
            enabled: true,
        };
        let coeffs = filter.calculate_coefficients_values();
        filter.set_coefficients_immediate(coeffs);
        filter
    }

    /// Calculate filter coefficients based on current parameters
    /// Uses Robert Bristow-Johnson's Audio EQ Cookbook formulas
    fn calculate_coefficients_values(&self) -> (f64, f64, f64, f64, f64) {
        let omega = 2.0 * PI * self.frequency / self.sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let q = self.q.max(MIN_BIQUAD_Q);
        let alpha = sin_omega / (2.0 * q);
        let a = 10.0_f64.powf(self.gain_db / 40.0); // sqrt(10^(dB/20))

        let (b0, b1, b2, a0, a1, a2) = match self.filter_type {
            BiquadType::Peaking => {
                let b0 = 1.0 + alpha * a;
                let b1 = -2.0 * cos_omega;
                let b2 = 1.0 - alpha * a;
                let a0 = 1.0 + alpha / a;
                let a1 = -2.0 * cos_omega;
                let a2 = 1.0 - alpha / a;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::LowShelf => {
                let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
                let b0 = a * ((a + 1.0) - (a - 1.0) * cos_omega + two_sqrt_a_alpha);
                let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_omega);
                let b2 = a * ((a + 1.0) - (a - 1.0) * cos_omega - two_sqrt_a_alpha);
                let a0 = (a + 1.0) + (a - 1.0) * cos_omega + two_sqrt_a_alpha;
                let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_omega);
                let a2 = (a + 1.0) + (a - 1.0) * cos_omega - two_sqrt_a_alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::HighShelf => {
                let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
                let b0 = a * ((a + 1.0) + (a - 1.0) * cos_omega + two_sqrt_a_alpha);
                let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_omega);
                let b2 = a * ((a + 1.0) + (a - 1.0) * cos_omega - two_sqrt_a_alpha);
                let a0 = (a + 1.0) - (a - 1.0) * cos_omega + two_sqrt_a_alpha;
                let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_omega);
                let a2 = (a + 1.0) - (a - 1.0) * cos_omega - two_sqrt_a_alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::HighPass => {
                // High-pass filter - cuts frequencies below the cutoff
                let b0 = (1.0 + cos_omega) / 2.0;
                let b1 = -(1.0 + cos_omega);
                let b2 = (1.0 + cos_omega) / 2.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_omega;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
            BiquadType::LowPass => {
                // Low-pass filter - cuts frequencies above the cutoff
                let b0 = (1.0 - cos_omega) / 2.0;
                let b1 = 1.0 - cos_omega;
                let b2 = (1.0 - cos_omega) / 2.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_omega;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            }
        };

        // Normalize coefficients
        (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    fn set_coefficients_immediate(&mut self, coeffs: (f64, f64, f64, f64, f64)) {
        let (b0, b1, b2, a1, a2) = coeffs;
        self.b0 = b0;
        self.b1 = b1;
        self.b2 = b2;
        self.a1 = a1;
        self.a2 = a2;
        self.pending_b0 = b0;
        self.pending_b1 = b1;
        self.pending_b2 = b2;
        self.pending_a1 = a1;
        self.pending_a2 = a2;
        self.pending_z1 = 0.0;
        self.pending_z2 = 0.0;
        self.crossfade_total = 0;
        self.crossfade_remaining = 0;
    }

    fn schedule_coefficients_crossfade(&mut self, coeffs: (f64, f64, f64, f64, f64)) {
        let (b0, b1, b2, a1, a2) = coeffs;
        self.pending_b0 = b0;
        self.pending_b1 = b1;
        self.pending_b2 = b2;
        self.pending_a1 = a1;
        self.pending_a2 = a2;
        self.pending_z1 = self.z1;
        self.pending_z2 = self.z2;
        self.crossfade_total = coefficient_crossfade_samples(self.sample_rate);
        self.crossfade_remaining = self.crossfade_total;
    }

    #[inline]
    fn process_direct(
        input: f64,
        coeffs: (f64, f64, f64, f64, f64),
        z1: &mut f64,
        z2: &mut f64,
    ) -> f64 {
        let (b0, b1, b2, a1, a2) = coeffs;
        let output = b0 * input + *z1;
        *z1 = b1 * input - a1 * output + *z2;
        *z2 = b2 * input - a2 * output;
        output
    }

    fn promote_pending_coefficients(&mut self) {
        self.b0 = self.pending_b0;
        self.b1 = self.pending_b1;
        self.b2 = self.pending_b2;
        self.a1 = self.pending_a1;
        self.a2 = self.pending_a2;
        self.z1 = self.pending_z1;
        self.z2 = self.pending_z2;
        self.crossfade_total = 0;
        self.crossfade_remaining = 0;
    }

    /// Process a single sample using Direct Form II Transposed
    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        if !self.enabled {
            return input;
        }

        let input_f64 = input as f64;
        let active_output = Self::process_direct(
            input_f64,
            (self.b0, self.b1, self.b2, self.a1, self.a2),
            &mut self.z1,
            &mut self.z2,
        );

        if self.crossfade_remaining == 0 {
            return active_output as f32;
        }

        let pending_output = Self::process_direct(
            input_f64,
            (
                self.pending_b0,
                self.pending_b1,
                self.pending_b2,
                self.pending_a1,
                self.pending_a2,
            ),
            &mut self.pending_z1,
            &mut self.pending_z2,
        );
        let fade_pos = self.crossfade_total - self.crossfade_remaining + 1;
        let fade = fade_pos as f64 / self.crossfade_total as f64;
        let output = active_output * (1.0 - fade) + pending_output * fade;
        self.crossfade_remaining -= 1;
        if self.crossfade_remaining == 0 {
            self.promote_pending_coefficients();
        }
        output as f32
    }

    /// Process a block of samples in-place
    pub fn process_block_inplace(&mut self, buffer: &mut [f32]) {
        if !self.enabled {
            return;
        }

        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
        self.pending_z1 = 0.0;
        self.pending_z2 = 0.0;
        self.crossfade_total = 0;
        self.crossfade_remaining = 0;
    }

    /// Set filter frequency and recalculate coefficients
    pub fn set_frequency(&mut self, frequency: f64) {
        self.frequency = frequency;
        self.schedule_coefficients_crossfade(self.calculate_coefficients_values());
    }

    /// Set filter gain and recalculate coefficients
    pub fn set_gain_db(&mut self, gain_db: f64) {
        self.gain_db = gain_db;
        self.schedule_coefficients_crossfade(self.calculate_coefficients_values());
    }

    /// Set filter gain immediately.
    ///
    /// This is intended for already-smoothed modulation sources. UI parameter
    /// changes should use `set_gain_db` so they crossfade the audio output.
    pub fn set_gain_db_immediate(&mut self, gain_db: f64) {
        self.gain_db = gain_db;
        self.set_coefficients_immediate(self.calculate_coefficients_values());
    }

    /// Set filter Q factor and recalculate coefficients
    pub fn set_q(&mut self, q: f64) {
        self.q = q.max(MIN_BIQUAD_Q);
        self.schedule_coefficients_crossfade(self.calculate_coefficients_values());
    }

    /// Enable or disable the filter
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if filter is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get current frequency
    pub fn frequency(&self) -> f64 {
        self.frequency
    }

    /// Get current gain in dB
    pub fn gain_db(&self) -> f64 {
        self.gain_db
    }

    /// Get current Q factor
    pub fn q(&self) -> f64 {
        self.q
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biquad_passthrough_at_zero_gain() {
        let mut filter = Biquad::new(BiquadType::Peaking, 1000.0, 0.0, 1.0, 48000.0);

        // With 0 dB gain, a peaking filter should pass through
        let input = 0.5f32;
        let output = filter.process_sample(input);

        // Allow for small numerical error
        assert!((output - input).abs() < 0.01);
    }

    #[test]
    fn test_biquad_disabled() {
        let mut filter = Biquad::new(BiquadType::Peaking, 1000.0, 12.0, 1.0, 48000.0);
        filter.set_enabled(false);

        let input = 0.5f32;
        let output = filter.process_sample(input);

        assert_eq!(output, input);
    }

    #[test]
    fn test_biquad_boost() {
        let mut filter = Biquad::new(BiquadType::Peaking, 1000.0, 12.0, 1.0, 48000.0);

        // Feed a 1kHz sine wave and check for gain
        let sample_rate = 48000.0;
        let freq = 1000.0;
        let mut max_output = 0.0f32;

        for i in 0..4800 {
            let t = i as f64 / sample_rate;
            let input = (2.0 * PI * freq * t).sin() as f32 * 0.1;
            let output = filter.process_sample(input);
            max_output = max_output.max(output.abs());
        }

        // With 12 dB boost, output should be significantly higher than 0.1
        assert!(
            max_output > 0.2,
            "Expected boost, got max_output = {}",
            max_output
        );
    }

    #[test]
    fn test_biquad_q_zero_guard() {
        let mut filter = Biquad::new(BiquadType::Peaking, 1000.0, 0.0, 0.0, 48000.0);
        let output = filter.process_sample(0.25);
        assert!(output.is_finite());
    }

    #[test]
    fn test_biquad_crossfade_promotes_pending_coefficients() {
        let mut filter = Biquad::new(BiquadType::Peaking, 1000.0, 0.0, 1.0, 48000.0);
        filter.set_gain_db(12.0);
        let expected = filter.calculate_coefficients_values();
        let crossfade_samples = coefficient_crossfade_samples(48_000.0);
        assert_eq!(filter.crossfade_remaining, crossfade_samples);
        for _ in 0..crossfade_samples {
            let _ = filter.process_sample(0.2);
        }
        assert_eq!(filter.crossfade_remaining, 0);
        assert!((filter.b0 - expected.0).abs() < 1e-12);
        assert!((filter.a2 - expected.4).abs() < 1e-12);
    }

    #[test]
    fn test_biquad_crossfade_keeps_silence_silent() {
        let mut filter = Biquad::new(BiquadType::Peaking, 1000.0, 0.0, 8.0, 48000.0);
        filter.set_gain_db(12.0);
        filter.set_frequency(80.0);

        for _ in 0..(coefficient_crossfade_samples(48_000.0) * 2) {
            let output = filter.process_sample(0.0);
            assert!(output.abs() < 1e-9);
        }
    }

    #[test]
    fn test_biquad_reset_clears_pending_crossfade() {
        let mut filter = Biquad::new(BiquadType::Peaking, 1000.0, 0.0, 1.0, 48000.0);
        filter.set_gain_db(12.0);
        assert!(filter.crossfade_remaining > 0);

        filter.reset();

        assert_eq!(filter.crossfade_remaining, 0);
    }

    #[test]
    fn test_biquad_crossfade_duration_is_sample_rate_independent() {
        for sample_rate in [44_100.0, 48_000.0, 96_000.0, 192_000.0] {
            let samples = coefficient_crossfade_samples(sample_rate);
            let duration_ms = samples as f64 * 1000.0 / sample_rate;
            assert!(
                (duration_ms - COEFF_CROSSFADE_MS).abs() <= 1000.0 / sample_rate,
                "sample_rate={sample_rate} duration_ms={duration_ms}"
            );
        }
    }

    #[test]
    fn test_rapid_biquad_automation_stays_finite_and_click_bounded() {
        let sample_rate = 192_000.0;
        let mut filter = Biquad::new(BiquadType::Peaking, 3000.0, 0.0, 2.0, sample_rate);
        let mut previous = 0.0_f32;
        let mut max_step = 0.0_f32;
        for sample_index in 0..(sample_rate as usize / 2) {
            if sample_index % (sample_rate as usize / 200) == 0 {
                let update = (sample_index / (sample_rate as usize / 200)) as f64;
                filter.set_frequency(300.0 + (update * 173.0) % 12_000.0);
                filter.set_gain_db(if update as usize & 1 == 0 {
                    12.0
                } else {
                    -12.0
                });
            }
            let phase = 2.0 * PI * 1000.0 * sample_index as f64 / sample_rate;
            let output = filter.process_sample((0.25 * phase.sin()) as f32);
            assert!(output.is_finite());
            max_step = max_step.max((output - previous).abs());
            previous = output;
        }
        assert!(max_step < 0.35, "rapid automation max step was {max_step}");
    }

    #[test]
    #[ignore = "release-mode callback cost measurement"]
    fn benchmark_biquad_morph_cost() {
        use std::hint::black_box;
        use std::time::Instant;

        const SAMPLES: usize = 2_000_000;
        let mut steady = Biquad::new(BiquadType::Peaking, 1000.0, 6.0, 1.0, 192_000.0);
        let started = Instant::now();
        for index in 0..SAMPLES {
            black_box(steady.process_sample(black_box((index as f32 * 0.013).sin())));
        }
        let steady_elapsed = started.elapsed();

        let mut automated = Biquad::new(BiquadType::Peaking, 1000.0, 6.0, 1.0, 192_000.0);
        let started = Instant::now();
        for index in 0..SAMPLES {
            if index % 960 == 0 {
                automated.set_gain_db(if (index / 960) % 2 == 0 { 12.0 } else { -12.0 });
            }
            black_box(automated.process_sample(black_box((index as f32 * 0.013).sin())));
        }
        let automated_elapsed = started.elapsed();

        println!(
            "biquad steady={:.2} ns/sample automated={:.2} ns/sample ratio={:.2}",
            steady_elapsed.as_nanos() as f64 / SAMPLES as f64,
            automated_elapsed.as_nanos() as f64 / SAMPLES as f64,
            automated_elapsed.as_secs_f64() / steady_elapsed.as_secs_f64()
        );
    }
}
