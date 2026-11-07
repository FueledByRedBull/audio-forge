"""The Python headroom helpers the native core replaced, kept verbatim as the parity reference.

test_headroom_native_parity.py compares the native versions with this copy on fixed cases.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np

NUM_EQ_BANDS = 10
HEADROOM_TARGET_DB = 1.0
LIMITER_GAIN_REDUCTION_WARN_DB = 1.0
TRUE_PEAK_GAIN_REDUCTION_WARN_DB = 0.5
_NATIVE_SAFETY_METRICS = (
    "pre_limiter_true_peak_headroom_db",
    "limiter_gain_reduction_db",
    "true_peak_limiter_gain_reduction_db",
)


def _finite_float(value: Any) -> float:
    if not isinstance(value, bool):
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            parsed = float("nan")
        if np.isfinite(parsed):
            return parsed
    raise ValueError(f"chain setting is not a finite number: {value!r}")


def _as_float(value: Any, default: float) -> float:
    """``default`` for a setting that is absent; a value that is no finite number is an error.

    A substituted default would be simulated and certified as if it were the candidate's.
    """
    return default if value is None else _finite_float(value)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"chain setting is not a boolean: {value!r}")


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, bool):
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            pass
    raise ValueError(f"chain setting is not an integer: {value!r}")


def _flatten_chain_settings(chain_settings: dict[str, Any] | None) -> dict[str, Any]:
    chain_settings = chain_settings or {}
    deesser = chain_settings.get("deesser") or {}
    compressor = chain_settings.get("compressor") or {}
    limiter = chain_settings.get("limiter") or {}
    gate = (
        chain_settings.get("gate")
        or chain_settings.get("gate_settings")
        or {}
    )
    suppressor = (
        chain_settings.get("suppressor")
        or chain_settings.get("suppression")
        or chain_settings.get("rnnoise")
        or chain_settings.get("suppressor_settings")
        or {}
    )

    flattened = {
        "return_output_audio": _as_bool(
            chain_settings.get("return_output_audio"),
            False,
        ),
        "deesser_enabled": _as_bool(deesser.get("enabled"), False),
        "deesser_auto_enabled": _as_bool(deesser.get("auto_enabled"), True),
        "deesser_auto_amount": _as_float(deesser.get("auto_amount"), 0.5),
        "deesser_low_cut_hz": _as_float(deesser.get("low_cut_hz"), 4000.0),
        "deesser_high_cut_hz": _as_float(deesser.get("high_cut_hz"), 11000.0),
        "deesser_threshold_db": _as_float(deesser.get("threshold_db"), -28.0),
        "deesser_ratio": _as_float(deesser.get("ratio"), 4.0),
        "deesser_attack_ms": _as_float(deesser.get("attack_ms"), 2.0),
        "deesser_release_ms": _as_float(deesser.get("release_ms"), 80.0),
        "deesser_max_reduction_db": _as_float(deesser.get("max_reduction_db"), 6.0),
        "compressor_enabled": _as_bool(compressor.get("enabled"), True),
        "compressor_threshold_db": _as_float(compressor.get("threshold_db"), -20.0),
        "compressor_ratio": _as_float(compressor.get("ratio"), 4.0),
        "compressor_attack_ms": _as_float(compressor.get("attack_ms"), 10.0),
        "compressor_release_ms": _as_float(compressor.get("release_ms"), 200.0),
        "compressor_makeup_gain_db": _as_float(compressor.get("makeup_gain_db"), 0.0),
        "compressor_adaptive_release": _as_bool(compressor.get("adaptive_release"), False),
        "compressor_base_release_ms": _as_float(compressor.get("base_release_ms"), 50.0),
        "compressor_auto_makeup_enabled": _as_bool(compressor.get("auto_makeup_enabled"), False),
        "compressor_target_lufs": _as_float(compressor.get("target_lufs"), -18.0),
        "compressor_noise_reference_reliability": _as_float(
            compressor.get("noise_reference_reliability"), 0.0
        ),
        "compressor_sidechain_highpass_enabled": _as_bool(
            compressor.get("sidechain_highpass_enabled"),
            True,
        ),
        "limiter_enabled": _as_bool(limiter.get("enabled"), True),
        "limiter_ceiling_db": _as_float(limiter.get("ceiling_db"), -0.5),
        "limiter_release_ms": _as_float(limiter.get("release_ms"), 50.0),
        "limiter_careful_output_enabled": _as_bool(
            limiter.get("careful_output_enabled"),
            True,
        ),
    }
    if isinstance(gate, Mapping):
        flattened.update(
            {
                "gate_enabled": _as_bool(gate.get("enabled"), True),
                "gate_threshold_db": _as_float(
                    gate.get("threshold_db"), -40.0
                ),
                "gate_attack_ms": _as_float(gate.get("attack_ms"), 10.0),
                "gate_release_ms": _as_float(
                    gate.get("release_ms"), 100.0
                ),
                "gate_mode": _as_int(gate.get("gate_mode"), 0),
                "gate_vad_threshold": _as_float(
                    gate.get("vad_threshold"), 0.48
                ),
                "gate_vad_hold_time_ms": _as_float(
                    gate.get("vad_hold_time_ms"), 200.0
                ),
                "gate_vad_pre_gain": _as_float(
                    gate.get("vad_pre_gain"), 1.0
                ),
                "gate_auto_threshold_enabled": _as_bool(
                    gate.get("auto_threshold_enabled"), True
                ),
                "gate_margin_db": _as_float(
                    gate.get("gate_margin_db"), 10.0
                ),
            }
        )
    if isinstance(suppressor, Mapping):
        flattened.update(
            {
                "suppressor_enabled": _as_bool(
                    suppressor.get("enabled"), True
                ),
                "suppressor_strength": _as_float(
                    suppressor.get("strength"), 1.0
                ),
                "noise_model": str(suppressor.get("model", "rnnoise")),
            }
        )
    # Explicit native-boundary values override defaults derived from missing
    # nested settings. Accept only the controls this adapter already knows.
    for key in flattened.keys() & chain_settings.keys():
        flattened[key] = deepcopy(chain_settings[key])
    for key in (
        "full_chain",
        "input_pre_filtered",
        "input_cleanup_mode",
        "processing_mode",
        "vad_probabilities",
        "vad_available",
        "auto_makeup_activity",
        "return_auto_makeup_activity",
    ):
        if key in chain_settings:
            flattened[key] = deepcopy(chain_settings[key])
    return flattened


def _bands_from_settings(eq_settings: dict[str, Any]) -> list[tuple[float, float, float]]:
    freqs = list(eq_settings.get("band_freqs") or [])
    gains = list(eq_settings.get("band_gains") or [])
    qs = list(eq_settings.get("band_qs") or [])
    if not (len(freqs) == len(gains) == len(qs) == NUM_EQ_BANDS):
        raise ValueError("Auto-EQ settings must contain 10 frequencies, gains, and Q values")
    return [
        (_finite_float(freq), _finite_float(gain), _finite_float(q))
        for freq, gain, q in zip(freqs, gains, qs)
    ]

def _legacy_eq_inputs(eq_settings: dict[str, Any]) -> tuple[list[tuple[float, float, float]], bool]:
    """The legacy branch of simulate_candidate_chain: the bands and the enabled flag."""
    bands = _bands_from_settings(eq_settings)
    return bands, _as_bool(eq_settings.get("enabled", eq_settings.get("eq_enabled")), True)


def _safety_metric_problems(normalized: Mapping[str, Any]) -> list[str]:
    """The metric loop of _native_simulate."""
    invalid_metrics = []
    for key in _NATIVE_SAFETY_METRICS:
        value = normalized.get(key)
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.number)
        ):
            invalid_metrics.append(f"{key} is missing or non-numeric")
            continue
        if not np.isfinite(float(value)):
            invalid_metrics.append(f"{key} is not finite")
    return invalid_metrics


def _headroom_excess_db(simulation: Mapping[str, Any]) -> float:
    """Return how far a render passes the headroom rule, whichever term fails most.

    It fits at <= 0; a reading that is missing or not finite is infinite.
    """
    metrics: list[float] = []
    for key in _NATIVE_SAFETY_METRICS:
        value = simulation.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            return float("inf")
        parsed = float(value)
        if not np.isfinite(parsed):
            return float("inf")
        metrics.append(parsed)
    pre_true_peak_headroom, limiter_gr, true_peak_gr = metrics
    return max(
        HEADROOM_TARGET_DB - pre_true_peak_headroom,
        limiter_gr - LIMITER_GAIN_REDUCTION_WARN_DB,
        true_peak_gr - TRUE_PEAK_GAIN_REDUCTION_WARN_DB,
    )


def _is_headroom_safe(simulation: Mapping[str, Any]) -> bool:
    return _headroom_excess_db(simulation) <= 0.0
