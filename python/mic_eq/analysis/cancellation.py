"""Cooperative cancellation primitives for background analysis."""

from __future__ import annotations

from collections.abc import Callable


class AnalysisCancelled(RuntimeError):
    """Raised when a caller cancels a non-realtime analysis job."""


def check_analysis_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise AnalysisCancelled("analysis cancelled")


__all__ = ["AnalysisCancelled", "check_analysis_cancelled"]
