//! Realtime scheduling helpers shared by the DSP thread and the DeepFilter
//! worker: the bounded RTKit request, SIGXCPU, and how long a realtime thread
//! may run without blocking.

use crate::audio::bounded::{self, Unanswered};
use std::time::Duration;

/// Longest stretch a realtime thread ever runs without blocking: half of
/// RTKit's default 200 ms RLIMIT_RTTIME budget.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) const REALTIME_BUSY_LIMIT: Duration = Duration::from_millis(100);
/// Shortest busy limit. cpal sets the soft limit to one stream quantum, which can
/// be a millisecond; half of that would put a thread to sleep after nearly every
/// pass, and a pass can take several milliseconds. The soft limit is only a
/// hint (SIGXCPU is ignored), so this floor trades nothing for safety: it is far
/// below the hard limit, which is RTKit's maximum, 200 ms by default.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) const REALTIME_BUSY_FLOOR: Duration = Duration::from_millis(10);

/// `bounded::run_bounded` for an RTKit request, with the answer's failures in words.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn run_bounded(
    timeout: Duration,
    request: impl FnOnce() -> Result<(), String> + Send + 'static,
) -> Result<(), String> {
    match bounded::run_bounded("audioforge-rtkit", timeout, request) {
        Ok(answer) => answer,
        Err(Unanswered::NotStarted(error)) => {
            Err(format!("could not start the RTKit request thread: {error}"))
        }
        Err(Unanswered::TimedOut) => Err("RTKit did not answer in time".to_string()),
        Err(Unanswered::Ended) => Err("the RTKit request ended without an answer".to_string()),
    }
}

/// Ignore SIGXCPU for the whole process. A realtime thread that uses up the
/// soft RLIMIT_RTTIME without blocking is sent SIGXCPU, which ends the process
/// by default, and cpal sets that soft limit to one stream quantum when it
/// promotes its callback thread, a millisecond or so at a low quantum. Ignored,
/// the kernel keeps re-sending it once a second and the hard limit (RTKit's
/// maximum for every promotion) still ends a runaway thread with SIGKILL; the
/// busy guard stays far below both. A handler someone else installed already
/// makes the signal harmless, so it is put back.
#[cfg(target_os = "linux")]
pub(crate) fn ignore_sigxcpu() -> std::io::Result<()> {
    // SAFETY: signal() with SIG_IGN installs no code; it only changes the
    // process's disposition of one signal, from any thread.
    // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
    let previous = unsafe { libc::signal(libc::SIGXCPU, libc::SIG_IGN) };
    if previous == libc::SIG_ERR {
        return Err(std::io::Error::last_os_error());
    }
    if previous != libc::SIG_DFL && previous != libc::SIG_IGN {
        // SAFETY: `previous` is the handler signal() just returned for this signal.
        // nosemgrep: rust.lang.security.unsafe-usage.unsafe-usage
        unsafe { libc::signal(libc::SIGXCPU, previous) };
    }
    Ok(())
}

/// Longest stretch a realtime thread may run without blocking, for a process whose soft
/// and hard RLIMIT_RTTIME are `soft_us` and `hard_us` (None: unlimited): half
/// the soft limit, within `REALTIME_BUSY_FLOOR` and `REALTIME_BUSY_LIMIT`, and
/// never more than half the hard limit, which ends the process.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn busy_limit_for(soft_us: Option<u64>, hard_us: Option<u64>) -> Duration {
    let half = |limit_us: u64| Duration::from_micros(limit_us / 2);
    let from_soft = soft_us.map_or(REALTIME_BUSY_LIMIT, |limit_us| {
        half(limit_us)
            .max(REALTIME_BUSY_FLOOR)
            .min(REALTIME_BUSY_LIMIT)
    });
    hard_us.map_or(from_soft, |limit_us| from_soft.min(half(limit_us)))
}

/// `busy_limit_for` the limits the process has now. They are read for every
/// decision: each thread that asks RTKit for realtime (cpal's PipeWire callback
/// thread asks for one stream quantum) sets the soft limit again.
#[cfg(target_os = "linux")]
pub(crate) fn realtime_busy_limit() -> Duration {
    let limit = rustix::process::getrlimit(rustix::process::Resource::Rttime);
    busy_limit_for(limit.current, limit.maximum)
}

/// Ask RTKit, from a helper thread and for at most `timeout`, to make the
/// calling thread realtime, after ignoring SIGXCPU. Ask for a second of budget
/// at `sample_rate`, which RTKit caps at its maximum, as the DSP thread does.
#[cfg(target_os = "linux")]
pub(crate) fn promote_current_thread(sample_rate: u32, timeout: Duration) -> Result<(), String> {
    ignore_sigxcpu().map_err(|error| format!("could not ignore SIGXCPU: {error}"))?;
    let thread =
        audio_thread_priority::get_current_thread_info().map_err(|error| error.to_string())?;
    run_bounded(timeout, move || {
        audio_thread_priority::promote_thread_to_real_time(thread, sample_rate, sample_rate)
            .map(|_| ())
            .map_err(|error| error.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn busy_limit_follows_the_soft_limit_within_its_floor_and_cap() {
        let hard = Some(200_000);
        for (soft_us, expected) in [
            (None, REALTIME_BUSY_LIMIT),
            (Some(1_000_000), REALTIME_BUSY_LIMIT),
            (Some(200_000), REALTIME_BUSY_LIMIT),
            (Some(100_000), Duration::from_millis(50)),
            (Some(30_000), Duration::from_millis(15)),
            // A low quantum cannot push the limit under the floor.
            (
                Some(21_333),
                Duration::from_micros(10_666).max(REALTIME_BUSY_FLOOR),
            ),
            (Some(1_333), REALTIME_BUSY_FLOOR),
            (Some(0), REALTIME_BUSY_FLOOR),
        ] {
            assert_eq!(busy_limit_for(soft_us, hard), expected, "{soft_us:?}");
        }
    }

    #[test]
    fn busy_limit_stays_under_half_the_hard_limit() {
        // The hard limit ends the process, so it outranks the floor.
        for (soft_us, hard_us, expected) in [
            (None, Some(200_000), REALTIME_BUSY_LIMIT),
            (Some(200_000), Some(100_000), Duration::from_millis(50)),
            (Some(200_000), Some(8_000), Duration::from_millis(4)),
            (None, Some(8_000), Duration::from_millis(4)),
            (Some(1), Some(1), Duration::ZERO),
            (Some(200_000), None, REALTIME_BUSY_LIMIT),
        ] {
            assert_eq!(
                busy_limit_for(soft_us, hard_us),
                expected,
                "{soft_us:?} {hard_us:?}"
            );
        }
    }

    #[test]
    fn the_floor_is_well_inside_rtkits_default_hard_limit() {
        assert!(REALTIME_BUSY_FLOOR * 10 <= Duration::from_millis(200));
        assert!(REALTIME_BUSY_FLOOR <= REALTIME_BUSY_LIMIT);
    }

    #[test]
    fn a_bounded_request_returns_its_answer() {
        assert_eq!(run_bounded(Duration::from_secs(5), || Ok(())), Ok(()));
        assert_eq!(
            run_bounded(Duration::from_secs(5), || Err("refused".to_string())),
            Err("refused".to_string())
        );
    }

    #[test]
    fn a_bounded_request_that_never_answers_times_out() {
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let error = run_bounded(Duration::from_millis(50), move || {
            let _ = release_rx.recv();
            Ok(())
        })
        .unwrap_err();
        assert!(error.contains("did not answer"), "{error}");
        let _ = release_tx.send(());
    }

    #[test]
    fn a_bounded_request_that_panics_reports_it() {
        let error = run_bounded(Duration::from_secs(5), || panic!("RTKit exploded")).unwrap_err();
        assert!(error.contains("without an answer"), "{error}");
    }
}
