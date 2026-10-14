//! Monotonic clock helpers for real-time-safe telemetry timestamps.

use std::sync::OnceLock;
use std::time::Instant;

static START_INSTANT: OnceLock<Instant> = OnceLock::new();

#[inline]
fn start_instant() -> Instant {
    *START_INSTANT.get_or_init(Instant::now)
}

/// Monotonic microseconds since process-local start, never 0: callers store
/// these in atomics where 0 means "never happened".
#[inline]
pub fn now_micros() -> u64 {
    (start_instant().elapsed().as_micros() as u64).max(1)
}

#[cfg(test)]
mod tests {
    use super::now_micros;

    #[test]
    fn first_reading_is_never_the_zero_never_written_sentinel() {
        // Run alone, this is the process's first reading, taken microseconds after the origin.
        assert!(now_micros() >= 1);
    }
}
