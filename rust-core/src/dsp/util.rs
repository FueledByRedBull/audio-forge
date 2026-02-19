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
