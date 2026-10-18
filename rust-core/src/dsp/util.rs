//! Shared DSP utility math helpers.

/// Convert time constant in milliseconds to a single-pole smoothing coefficient.
#[inline]
pub fn time_constant_to_coeff(time_ms: f64, sample_rate: f64) -> f64 {
    let tau = (time_ms.max(0.001)) / 1000.0;
    (-1.0 / (tau * sample_rate)).exp()
}

/// Convert dBFS to linear amplitude.
#[inline]
pub fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Convert linear amplitude to dBFS with a configurable floor.
#[inline]
pub fn linear_to_db(linear: f64, min_linear: f64) -> f64 {
    20.0 * linear.abs().max(min_linear).log10()
}

/// A finite `value` clamped to 0..=1; NaN and infinities keep `fallback`.
#[inline]
pub(crate) fn unit_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        fallback
    }
}

/// Length of the dry/processed ramp a stage runs when it is toggled.
const ENABLE_RAMP_MS: f64 = 10.0;

/// Linear ramp between dry (0) and processed (1) output that the compressor
/// and the de-esser run when they are enabled or disabled, instead of stepping.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EnableRamp {
    /// 0 is bypassed, `samples` is fully processed.
    position: usize,
    samples: usize,
}

impl EnableRamp {
    /// A ramp at `sample_rate` that starts fully processed or fully bypassed.
    pub(crate) fn new(sample_rate: f64, processed: bool) -> Self {
        let samples = if sample_rate.is_finite() && sample_rate > 0.0 {
            (sample_rate * ENABLE_RAMP_MS / 1_000.0).round().max(1.0) as usize
        } else {
            1
        };
        Self {
            position: if processed { samples } else { 0 },
            samples,
        }
    }

    /// Fully faded out: a re-enable starts from fresh state.
    pub(crate) fn is_bypassed(&self) -> bool {
        self.position == 0
    }

    pub(crate) fn is_fully_processed(&self) -> bool {
        self.position == self.samples
    }

    /// Jump to the end of the ramp, for state set before audio starts.
    pub(crate) fn finish(&mut self, enabled: bool) {
        self.position = if enabled { self.samples } else { 0 };
    }

    /// Advance one sample toward `enabled`. Returns the processed share of the
    /// output while ramping, `None` once the ramp has settled.
    #[inline]
    pub(crate) fn advance(&mut self, enabled: bool) -> Option<f64> {
        let target = if enabled { self.samples } else { 0 };
        if self.position == target {
            return None;
        }
        if enabled {
            self.position += 1;
        } else {
            self.position -= 1;
        }
        Some(self.position as f64 / self.samples as f64)
    }
}
