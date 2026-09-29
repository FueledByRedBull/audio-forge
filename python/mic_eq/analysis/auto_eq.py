"""Auto-EQ calculation using constrained least-squares fitting.

Public entry points for the split Auto-EQ implementation.
"""

from .auto_eq_parts.constants import (
    GAIN_MAX_DB,
    GAIN_MIN_DB,
    NUM_EQ_BANDS,
    SAMPLE_RATE,
)
from .auto_eq_parts.optimizer import calculate_eq_bands
from .auto_eq_parts.pipeline import analyze_auto_eq
from .auto_eq_parts.target import get_target_curve
from .auto_eq_parts.headroom import apply_headroom_validation, simulate_candidate_chain
from .cancellation import AnalysisCancelled
from .eq_quality import evaluate_eq_quality, weighted_target_error

__all__ = [
    "AnalysisCancelled",
    "GAIN_MAX_DB",
    "GAIN_MIN_DB",
    "NUM_EQ_BANDS",
    "SAMPLE_RATE",
    "analyze_auto_eq",
    "apply_headroom_validation",
    "calculate_eq_bands",
    "evaluate_eq_quality",
    "get_target_curve",
    "simulate_candidate_chain",
    "weighted_target_error",
]
