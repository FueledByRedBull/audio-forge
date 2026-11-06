//! A call that may block on another process, run so that it cannot hold the caller up.

use std::sync::mpsc;
use std::time::Duration;

/// Why `run_bounded` has no answer.
#[derive(Debug)]
pub(crate) enum Unanswered {
    NotStarted(std::io::Error),
    TimedOut,
    /// The helper thread ended without answering, as it does when `request` panics.
    Ended,
}

/// Run `request` on a helper thread named `name` and wait at most `timeout` for
/// its answer. A request that outlasts the timeout is left to finish on its own.
pub(crate) fn run_bounded<T: Send + 'static>(
    name: &str,
    timeout: Duration,
    request: impl FnOnce() -> T + Send + 'static,
) -> Result<T, Unanswered> {
    let (answer_tx, answer_rx) = mpsc::channel();
    std::thread::Builder::new()
        .name(name.to_string())
        .spawn(move || {
            let _ = answer_tx.send(request());
        })
        .map_err(Unanswered::NotStarted)?;
    answer_rx
        .recv_timeout(timeout)
        .map_err(|error| match error {
            mpsc::RecvTimeoutError::Timeout => Unanswered::TimedOut,
            mpsc::RecvTimeoutError::Disconnected => Unanswered::Ended,
        })
}
