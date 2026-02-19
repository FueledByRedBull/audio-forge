//! Monotonic clock helpers for real-time-safe telemetry timestamps.

use std::sync::OnceLock;
use std::time::Instant;

static START_INSTANT: OnceLock<Instant> = OnceLock::new();

#[inline]
fn start_instant() -> Instant {
    *START_INSTANT.get_or_init(Instant::now)
}

/// Monotonic microseconds since process-local start.
#[inline]
pub fn now_micros() -> u64 {
    start_instant().elapsed().as_micros() as u64
}
