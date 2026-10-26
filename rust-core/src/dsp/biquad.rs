//! IIR Biquad filter implementation using Direct Form II Transposed
//!
//! Uses f64 internally for coefficient precision, f32 for audio samples, and a
//! sample-rate-scaled 1.5 ms parallel-state crossfade for live coefficient edits.

use std::f64::consts::PI;

const MIN_BIQUAD_Q: f64 = 1e-6;
/// Far above any EQ or notch in the chain (EQ_Q_MAX is 10, the hum notch 36);
/// beyond it the bandwidth is below a hertz at audio rates.
const MAX_BIQUAD_Q: f64 = 1e3;
const COEFF_CROSSFADE_MS: f64 = 1.5;
const MAX_COEFF_CROSSFADE_SAMPLES: usize = 4096;
/// Lowest tunable frequency; 0 Hz and below put the poles on or outside the unit circle.
const MIN_BIQUAD_FREQUENCY_HZ: f64 = 1e-3;
/// Keep the centre frequency just below Nyquist, where `sin(omega)` turns negative.
const NYQUIST_FRACTION_MAX: f64 = 0.5 * (1.0 - 1e-9);
/// Beyond this a gain is not a level change but an overflow (`10^(dB/40)`).
const MAX_BIQUAD_GAIN_DB: f64 = 120.0;

pub(super) fn coefficient_crossfade_samples(sample_rate: f64) -> usize {
    let samples = (sample_rate * COEFF_CROSSFADE_MS / 1000.0).round();
    if samples.is_finite() {
        (samples as usize).clamp(1, MAX_COEFF_CROSSFADE_SAMPLES)
    } else {
        1
    }
}

/// Clamp coefficient-driving parameters to where the RBJ filters stay stable
/// (0 < f < Nyquist, finite gain, Q at most `MAX_BIQUAD_Q`). A NaN, or an
/// infinite Q, keeps its `previous` value, because it would otherwise poison
/// the coefficients and the filter state for good.
/// Values already in range pass through unchanged.
fn sanitized_parameters(
    sample_rate: f64,
    frequency: f64,
    gain_db: f64,
    q: f64,
    previous: (f64, f64, f64),
) -> (f64, f64, f64) {
    // `max`/`min` rather than `clamp`: a bad sample rate must not panic here.
    let bounded = |value: f64, fallback: f64, low: f64, high: f64| {
        if value.is_nan() {
            fallback
        } else {
            value.max(low).min(high)
        }
    };
    (
        bounded(
            frequency,
            previous.0,
            MIN_BIQUAD_FREQUENCY_HZ,
            sample_rate * NYQUIST_FRACTION_MAX,
        ),
        bounded(gain_db, previous.1, -MAX_BIQUAD_GAIN_DB, MAX_BIQUAD_GAIN_DB),
        // An infinite Q is no more meaningful than a NaN one.
        if q.is_finite() {
            q.min(MAX_BIQUAD_Q)
        } else {
            previous.2
        },
    )
}

/// Biquad filter types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiquadType {
    LowShelf,
    HighShelf,
    Peaking,
    Notch,
    HighPass,
    LowPass,
    Bypass,
}

/// IIR Biquad filter using Direct Form II Transposed
///
/// Transfer function: `H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)`
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
    /// A retarget arrived mid-fade; fade on to the current target afterwards.
    retarget_queued: bool,

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
    ///
    /// Out-of-range parameters are clamped to the stable range; a NaN falls
    /// back to 1 kHz, 0 dB and Q 1.
    pub fn new(
        filter_type: BiquadType,
        frequency: f64,
        gain_db: f64,
        q: f64,
        sample_rate: f64,
    ) -> Self {
        let (frequency, gain_db, q) =
            sanitized_parameters(sample_rate, frequency, gain_db, q, (1_000.0, 0.0, 1.0));
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
            retarget_queued: false,
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

    /// Clamp new parameters for this filter; a NaN keeps the current value.
    fn sanitized(&self, frequency: f64, gain_db: f64, q: f64) -> (f64, f64, f64) {
        sanitized_parameters(
            self.sample_rate,
            frequency,
            gain_db,
            q,
            (self.frequency, self.gain_db, self.q),
        )
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
            BiquadType::Notch => {
                let b0 = 1.0;
                let b1 = -2.0 * cos_omega;
                let b2 = 1.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_omega;
                let a2 = 1.0 - alpha;
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
            BiquadType::Bypass => (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        };

        // Normalize coefficients
        (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    fn coefficient_magnitude_response_db(
        coefficients: (f64, f64, f64, f64, f64),
        frequency_hz: f64,
        sample_rate: f64,
    ) -> f64 {
        let (b0, b1, b2, a1, a2) = coefficients;
        let omega = 2.0 * PI * frequency_hz / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let cos_two_omega = (2.0 * omega).cos();
        let sin_two_omega = (2.0 * omega).sin();

        let numerator_real = b0 + b1 * cos_omega + b2 * cos_two_omega;
        let numerator_imag = -b1 * sin_omega - b2 * sin_two_omega;
        let denominator_real = 1.0 + a1 * cos_omega + a2 * cos_two_omega;
        let denominator_imag = -a1 * sin_omega - a2 * sin_two_omega;
        let numerator_power = numerator_real * numerator_real + numerator_imag * numerator_imag;
        let denominator_power =
            denominator_real * denominator_real + denominator_imag * denominator_imag;
        let magnitude = (numerator_power / denominator_power.max(1.0e-30)).sqrt();
        20.0 * magnitude.max(1.0e-10).log10()
    }

    /// Magnitude response of the active coefficients at one frequency.
    pub fn magnitude_response_db(&self, frequency_hz: f64) -> f64 {
        if !self.enabled {
            return 0.0;
        }
        Self::coefficient_magnitude_response_db(
            (self.b0, self.b1, self.b2, self.a1, self.a2),
            frequency_hz,
            self.sample_rate,
        )
    }

    /// Magnitude response of the configured target after any live crossfade.
    pub fn target_magnitude_response_db(&self, frequency_hz: f64) -> f64 {
        if !self.enabled {
            return 0.0;
        }
        Self::coefficient_magnitude_response_db(
            self.calculate_coefficients_values(),
            frequency_hz,
            self.sample_rate,
        )
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
        self.retarget_queued = false;
    }

    /// A fade that has emitted samples blends two filters; restarting it
    /// from the active filter would step the output back to that filter.
    fn crossfade_has_started(&self) -> bool {
        self.crossfade_remaining != 0 && self.crossfade_remaining != self.crossfade_total
    }

    /// Drop the filter memory and any crossfade, keeping the coefficients.
    pub(crate) fn clear_state(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
        self.pending_z1 = 0.0;
        self.pending_z2 = 0.0;
        self.crossfade_total = 0;
        self.crossfade_remaining = 0;
        self.retarget_queued = false;
    }

    /// Fade to the current parameters; mid-fade, queue the retarget and leave
    /// the coefficients to be computed when it starts.
    fn schedule_coefficients_crossfade(&mut self) {
        if self.crossfade_has_started() {
            self.retarget_queued = true;
            return;
        }
        let (b0, b1, b2, a1, a2) = self.calculate_coefficients_values();
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
            if self.retarget_queued {
                self.retarget_queued = false;
                self.schedule_coefficients_crossfade();
            }
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
        // A stream discontinuity must start from the configured target, not
        // whichever coefficients happened to be active partway through a UI
        // crossfade.
        let target = self.calculate_coefficients_values();
        self.set_coefficients_immediate(target);
        self.clear_state();
    }

    /// Set filter frequency and recalculate coefficients
    pub fn set_frequency(&mut self, frequency: f64) {
        self.frequency = self.sanitized(frequency, self.gain_db, self.q).0;
        self.schedule_coefficients_crossfade();
    }

    /// Set filter gain and recalculate coefficients
    pub fn set_gain_db(&mut self, gain_db: f64) {
        self.gain_db = self.sanitized(self.frequency, gain_db, self.q).1;
        self.schedule_coefficients_crossfade();
    }

    /// Set filter gain immediately.
    ///
    /// This is intended for already-smoothed modulation sources. UI parameter
    /// changes should use `set_gain_db` so they crossfade the audio output.
    /// During a crossfade the new gain goes to the incoming filter only, so the
    /// fade (e.g. a de-esser band moving) still completes.
    pub fn set_gain_db_immediate(&mut self, gain_db: f64) {
        self.gain_db = self.sanitized(self.frequency, gain_db, self.q).1;
        if self.crossfade_remaining == 0 {
            self.set_coefficients_immediate(self.calculate_coefficients_values());
        } else if !self.retarget_queued {
            // A queued retarget already fades to this gain when it starts.
            (
                self.pending_b0,
                self.pending_b1,
                self.pending_b2,
                self.pending_a1,
                self.pending_a2,
            ) = self.calculate_coefficients_values();
        }
    }

    /// Set every coefficient-driving parameter in one click-safe update.
    pub fn set_parameters(
        &mut self,
        filter_type: BiquadType,
        frequency: f64,
        gain_db: f64,
        q: f64,
    ) {
        let (frequency, gain_db, q) = self.sanitized(frequency, gain_db, q);
        self.filter_type = filter_type;
        self.frequency = frequency;
        self.gain_db = gain_db;
        self.q = q.max(MIN_BIQUAD_Q);
        self.schedule_coefficients_crossfade();
    }

    /// Set every coefficient-driving parameter without a transition.
    ///
    /// Intended for stream reset and construction paths where no prior audio
    /// should be blended into the next stream.
    pub fn set_parameters_immediate(
        &mut self,
        filter_type: BiquadType,
        frequency: f64,
        gain_db: f64,
        q: f64,
    ) {
        let (frequency, gain_db, q) = self.sanitized(frequency, gain_db, q);
        self.filter_type = filter_type;
        self.frequency = frequency;
        self.gain_db = gain_db;
        self.q = q.max(MIN_BIQUAD_Q);
        self.set_coefficients_immediate(self.calculate_coefficients_values());
    }

    /// Enable or disable the filter
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if filter is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get current gain in dB
    pub fn gain_db(&self) -> f64 {
        self.gain_db
    }

    /// Whether a live coefficient transition is still in progress.
    pub fn is_crossfading(&self) -> bool {
        self.crossfade_remaining != 0
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
    fn test_notch_nulls_its_center_frequency() {
        let filter = Biquad::new(BiquadType::Notch, 1_000.0, 0.0, 4.0, 48_000.0);
        assert!(filter.magnitude_response_db(1_000.0) < -150.0);
        assert!(filter.magnitude_response_db(100.0).abs() < 0.1);
        assert!(filter.magnitude_response_db(10_000.0).abs() < 0.1);
    }

    #[test]
    fn test_bypass_is_exactly_flat() {
        let mut filter = Biquad::new(BiquadType::Bypass, 1_000.0, 12.0, 10.0, 48_000.0);
        for sample in [-1.0_f32, -0.25, 0.0, 0.25, 1.0] {
            assert_eq!(filter.process_sample(sample), sample);
        }
        assert_eq!(filter.magnitude_response_db(1_000.0), 0.0);
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

        for _ in 0..256 {
            let _ = filter.process_sample(1.0);
        }
        assert!(filter.z1.abs() > 1.0e-9 || filter.z2.abs() > 1.0e-9);

        filter.reset();

        assert_eq!(filter.crossfade_remaining, 0);
        assert!((filter.magnitude_response_db(1_000.0) - 12.0).abs() < 1.0e-9);
        assert_eq!(filter.z1, 0.0);
        assert_eq!(filter.z2, 0.0);
        for _ in 0..256 {
            assert_eq!(filter.process_sample(0.0), 0.0);
        }
    }

    #[test]
    fn test_magnitude_response_matches_peaking_center_gain() {
        let filter = Biquad::new(BiquadType::Peaking, 1_000.0, 6.0, 2.0, 48_000.0);
        assert!((filter.magnitude_response_db(1_000.0) - 6.0).abs() < 1.0e-9);
    }

    #[test]
    fn test_disabled_biquad_reports_flat_response() {
        let mut filter = Biquad::new(BiquadType::LowShelf, 100.0, 12.0, 0.7, 48_000.0);
        filter.set_enabled(false);
        assert_eq!(filter.magnitude_response_db(50.0), 0.0);
        assert_eq!(filter.target_magnitude_response_db(50.0), 0.0);
    }

    #[test]
    fn test_target_response_updates_before_live_crossfade_completes() {
        let mut filter = Biquad::new(BiquadType::Peaking, 1_000.0, 0.0, 2.0, 48_000.0);
        filter.set_gain_db(6.0);

        assert!(filter.magnitude_response_db(1_000.0).abs() < 1.0e-9);
        assert!((filter.target_magnitude_response_db(1_000.0) - 6.0).abs() < 1.0e-9);
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

    /// Render a 200 Hz tone through a 200 Hz peaking filter, boost it to 12 dB at
    /// sample 4800 and run `retarget` 36 samples (half the fade) later.
    fn render_retarget(retarget: impl Fn(&mut Biquad)) -> Vec<f32> {
        use crate::test_signal::sine;
        let mut filter = Biquad::new(BiquadType::Peaking, 200.0, 0.0, 1.0, 48_000.0);
        (0..9_600)
            .map(|n| {
                if n == 4_800 {
                    filter.set_gain_db(12.0);
                }
                if n == 4_836 {
                    retarget(&mut filter);
                }
                filter.process_sample(sine(200.0, 0.2, 48_000.0, n))
            })
            .collect()
    }

    #[test]
    fn test_retarget_during_crossfade_continues_from_the_blend() {
        use crate::test_signal::{max_residual, sine};
        let tones = [200.0];
        let single = render_retarget(|_| {});
        let retargeted = render_retarget(|filter| filter.set_gain_db(-12.0));
        // Restarting the fade from the old filter left a residual of 0.0501
        // here; one clean fade leaves 0.0028.
        let click = max_residual(&retargeted, &tones, 48_000.0, 4_752, 5_280);
        let clean = max_residual(&single, &tones, 48_000.0, 4_752, 5_280);
        assert!(
            click < 1.1 * clean,
            "retarget residual {click}, one fade {clean}"
        );

        // Steady state: untouched before the retarget, and 50 ms after it the
        // filter is the -12 dB target within 1e-6 (measured 6e-15 before and after).
        assert_eq!(retargeted[..4_836], single[..4_836]);
        let mut settled = Biquad::new(BiquadType::Peaking, 200.0, -12.0, 1.0, 48_000.0);
        for (n, &actual) in retargeted.iter().enumerate() {
            let expected = settled.process_sample(sine(200.0, 0.2, 48_000.0, n));
            if n >= 7_200 {
                assert!((actual - expected).abs() < 1e-6, "sample {n}");
            }
        }
    }

    #[test]
    fn test_immediate_gain_during_crossfade_keeps_the_fade() {
        let mut filter = Biquad::new(BiquadType::Peaking, 300.0, -6.0, 2.0, 48_000.0);
        filter.set_parameters(BiquadType::Peaking, 600.0, -6.0, 2.0);
        let _ = filter.process_sample(0.1);
        filter.set_gain_db_immediate(-6.5);
        assert!(filter.is_crossfading(), "a gain update cancelled the move");
        for _ in 0..coefficient_crossfade_samples(48_000.0) {
            let _ = filter.process_sample(0.1);
        }
        let target = Biquad::new(BiquadType::Peaking, 600.0, -6.5, 2.0, 48_000.0);
        assert!(!filter.is_crossfading());
        assert!(
            (filter.magnitude_response_db(600.0) - target.magnitude_response_db(600.0)).abs()
                < 1e-9
        );
    }

    #[test]
    fn test_immediate_gain_during_a_queued_retarget_leaves_the_running_fade_alone() {
        use crate::test_signal::{max_residual, sine};
        // Boost at sample 4800, retune one sample into that fade (queued) and
        // send an immediate gain 50 samples later (70% through), still queued.
        let render = |immediate_gain: Option<f64>| {
            let mut filter = Biquad::new(BiquadType::Peaking, 200.0, 0.0, 1.0, 48_000.0);
            let output: Vec<f32> = (0..9_600)
                .map(|n| {
                    if n == 4_800 {
                        filter.set_gain_db(12.0);
                    }
                    if n == 4_801 {
                        filter.set_frequency(400.0);
                    }
                    if let (Some(gain_db), 4_850) = (immediate_gain, n) {
                        filter.set_gain_db_immediate(gain_db);
                    }
                    filter.process_sample(sine(200.0, 0.2, 48_000.0, n))
                })
                .collect();
            (output, filter)
        };
        let (queued_only, _) = render(None);
        let (with_immediate, filter) = render(Some(-12.0));
        let tones = [200.0];
        let clean = max_residual(&queued_only, &tones, 48_000.0, 4_752, 5_280);
        let click = max_residual(&with_immediate, &tones, 48_000.0, 4_752, 5_280);
        // Measured 0.0040 against 0.0027; rewriting the pending coefficients
        // under the queued retarget left 0.0171.
        assert!(
            click < 2.0 * clean,
            "immediate gain residual {click}, queued retarget alone {clean}"
        );

        assert!(!filter.is_crossfading());
        let target = Biquad::new(BiquadType::Peaking, 400.0, -12.0, 1.0, 48_000.0);
        for frequency in [100.0, 200.0, 400.0, 2_000.0] {
            assert!(
                (filter.magnitude_response_db(frequency) - target.magnitude_response_db(frequency))
                    .abs()
                    < 1e-9,
                "{frequency} Hz"
            );
        }
    }

    #[test]
    fn test_non_finite_and_huge_q_are_bounded() {
        let reference = Biquad::new(BiquadType::Peaking, 1_000.0, 6.0, 1.4, 48_000.0);
        for q in [f64::INFINITY, f64::NEG_INFINITY] {
            let mut filter = reference.clone();
            filter.set_parameters_immediate(BiquadType::Peaking, 1_000.0, 6.0, q);
            assert_eq!(filter.q, 1.4, "Q {q} must keep the previous value");
            filter.set_parameters(BiquadType::Peaking, 1_000.0, 6.0, q);
            assert_eq!(filter.q, 1.4, "Q {q} must keep the previous value");
            let constructed = Biquad::new(BiquadType::Peaking, 1_000.0, 6.0, q, 48_000.0);
            assert_eq!(constructed.q, 1.0, "Q {q} must fall back like NaN");
        }
        let mut huge = reference.clone();
        huge.set_parameters_immediate(BiquadType::Notch, 1_000.0, 0.0, 1e12);
        assert_eq!(huge.q, MAX_BIQUAD_Q);
        assert_bounded_response(&mut huge, 2.0);
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

    fn assert_bounded_response(filter: &mut Biquad, limit: f32) {
        let mut peak = 0.0_f32;
        for index in 0..20_000 {
            let output = filter.process_sample((index as f32 * 0.05).sin());
            assert!(output.is_finite(), "non-finite output at sample {index}");
            peak = peak.max(output.abs());
        }
        assert!(peak < limit, "filter diverged: peak {peak}");
    }

    #[test]
    fn test_out_of_range_frequency_cannot_destabilize_the_filter() {
        for frequency in [
            -1_000.0,
            0.0,
            30_000.0,
            48_000.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            let mut constructed = Biquad::new(BiquadType::Peaking, frequency, 12.0, 1.0, 48_000.0);
            assert_bounded_response(&mut constructed, 100.0);

            let mut retuned = Biquad::new(BiquadType::HighShelf, 1_000.0, 12.0, 1.0, 48_000.0);
            retuned.set_frequency(frequency);
            assert_bounded_response(&mut retuned, 100.0);
            retuned.set_parameters_immediate(BiquadType::LowPass, frequency, 0.0, 0.7);
            assert_bounded_response(&mut retuned, 100.0);
        }
    }

    #[test]
    fn test_non_finite_gain_is_bounded_and_nan_is_rejected() {
        let mut filter = Biquad::new(BiquadType::Peaking, 1_000.0, 6.0, 1.0, 48_000.0);
        filter.set_gain_db(f64::INFINITY);
        assert_bounded_response(&mut filter, 2.0e6);
        filter.set_gain_db(6.0);
        filter.set_gain_db(f64::NAN);
        assert_eq!(filter.gain_db(), 6.0);
    }

    #[test]
    fn test_nan_parameters_keep_the_previous_filter() {
        let mut filter = Biquad::new(BiquadType::Peaking, 1_000.0, 6.0, 1.4, 48_000.0);
        let mut reference = filter.clone();
        filter.set_frequency(f64::NAN);
        filter.set_gain_db(f64::NAN);
        filter.set_gain_db_immediate(f64::NAN);
        filter.set_parameters(BiquadType::Peaking, f64::NAN, f64::NAN, f64::NAN);
        filter.set_parameters_immediate(BiquadType::Peaking, f64::NAN, f64::NAN, f64::NAN);
        for index in 0..2_000 {
            let input = (index as f32 * 0.05).sin();
            let expected = reference.process_sample(input);
            let actual = filter.process_sample(input);
            assert!(
                (actual - expected).abs() < 1e-6,
                "sample {index}: {actual} vs {expected}"
            );
        }

        let mut constructed =
            Biquad::new(BiquadType::Peaking, f64::NAN, f64::NAN, f64::NAN, 48_000.0);
        assert_bounded_response(&mut constructed, 100.0);
    }

    #[test]
    fn test_valid_parameters_are_not_altered_by_sanitizing() {
        let mut filter = Biquad::new(BiquadType::Peaking, 1_000.0, 6.0, 1.4, 48_000.0);
        filter.set_parameters(BiquadType::LowShelf, 80.0, -12.0, 0.707);
        assert_eq!(
            (filter.frequency, filter.gain_db(), filter.q),
            (80.0, -12.0, 0.707)
        );
        filter.set_frequency(23_999.0);
        assert_eq!(filter.frequency, 23_999.0);
    }
}
