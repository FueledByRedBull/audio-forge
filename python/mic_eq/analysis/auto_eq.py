"""Auto-EQ calculation using constrained least-squares fitting.

Public compatibility facade for the split Auto-EQ implementation.
"""

from .auto_eq_parts.constants import (
    GAIN_MAX_DB,
    GAIN_MIN_DB,
    NUM_EQ_BANDS,
    SAMPLE_RATE,
)
from .auto_eq_parts.dynamic_bands import (
    _build_dense_log_grid,
    _center_bounds,
    _enforce_adjacent_gain_limit,
    _estimate_band_snr_db,
    _q_bounds,
    _remove_spectral_tilt,
    _select_dynamic_band_layout,
    _snr_aware_gain_upper_bounds,
    _snr_weight_scale_dense,
    _voice_weights,
)
from .auto_eq_parts.optimizer import calculate_eq_bands
from .auto_eq_parts.pipeline import analyze_auto_eq
from .auto_eq_parts.response import _predict_eq_response
from .auto_eq_parts.target import get_target_curve

__all__ = [
    "GAIN_MAX_DB",
    "GAIN_MIN_DB",
    "NUM_EQ_BANDS",
    "SAMPLE_RATE",
    "_build_dense_log_grid",
    "_center_bounds",
    "_enforce_adjacent_gain_limit",
    "_estimate_band_snr_db",
    "_predict_eq_response",
    "_q_bounds",
    "_remove_spectral_tilt",
    "_select_dynamic_band_layout",
    "_snr_aware_gain_upper_bounds",
    "_snr_weight_scale_dense",
    "_voice_weights",
    "analyze_auto_eq",
    "calculate_eq_bands",
    "get_target_curve",
]
