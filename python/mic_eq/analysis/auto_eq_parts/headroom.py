"""Headroom-aware Auto-EQ validation and offline chain simulation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

import numpy as np

from ..eq_quality import evaluate_eq_quality, weighted_target_error
from .constants import NUM_EQ_BANDS, REDUCED_RECOMMENDATION_CONFIDENCE_THRESHOLD, SAMPLE_RATE
from .dynamic_bands import _voice_weights
from .optimizer import _build_fit_context, _overall_confidence, _validation_confidence
from ..cancellation import check_analysis_cancelled
from ...mic_eq_core import (
    flatten_chain_settings,
    headroom_excess_db,
    is_headroom_safe,
    legacy_eq_inputs,
    safety_metric_problems,
    simulate_auto_eq_chain,
)

HEADROOM_SCALES = (1.0, 0.85, 0.70, 0.55, 0.40, 0.25, 0.0)


def _result_float(value: Any, default: float) -> float:
    """Lenient read of a field of the Auto-EQ result: ``default`` for anything that is no finite number."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _native_failure(kind: str, *, message: str) -> dict[str, str]:
    """Return a bounded, user-safe reason for an unavailable native path."""
    return {
        "kind": kind,
        "message": str(message)[:256],
    }


def _native_simulate(
    audio_data: np.ndarray,
    sample_rate: int,
    bands: list[tuple[float, float, float]],
    flat_settings: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    try:
        audio = np.ascontiguousarray(audio_data, dtype=np.float32)
        result: Any = simulate_auto_eq_chain(
            audio, float(sample_rate), bands, flat_settings
        )
        if not isinstance(result, Mapping):
            return None, _native_failure(
                "invalid_result",
                message=(
                    "native simulator returned "
                    f"{type(result).__name__}, expected a mapping"
                ),
            )
        normalized = {str(key): value for key, value in result.items()}
        invalid_metrics = safety_metric_problems(normalized)
        if invalid_metrics:
            return None, _native_failure(
                "invalid_result",
                message="native simulator safety metrics invalid: "
                + "; ".join(invalid_metrics),
            )
        return normalized, None
    except Exception as exc:
        return None, _native_failure(
            "runtime_error",
            message=f"native simulator failed: {type(exc).__name__}: {exc}",
        )


def _typed_eq_payload(eq_settings: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the typed EQ fields; a payload from ``EQState.get_settings()`` also carries legacy arrays."""
    if "bands" not in eq_settings:  # a legacy-only payload is read as such
        return dict(eq_settings)
    return {
        key: eq_settings[key]
        for key in ("schema_version", "enabled", "bands", "layers")
        if key in eq_settings
    }


def simulate_candidate_chain(
    audio_data: np.ndarray,
    sample_rate: int,
    eq_settings: dict[str, Any],
    chain_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate the deterministic downstream chain for a candidate EQ."""

    flat_settings = flatten_chain_settings(chain_settings)
    if "bands" in eq_settings:
        from ...config import EQSettings

        typed = EQSettings.from_dict(_typed_eq_payload(eq_settings))
        bands = [(band.frequency_hz, band.gain_db, band.q) for band in typed.bands]
        flat_settings.update(
            eq_enabled=typed.enabled,
            eq_bands=[band.to_native() for band in typed.bands],
        )
        if typed.correction_bands:
            flat_settings["correction_bands"] = [
                band.to_native() for band in typed.correction_bands
            ]
    else:
        bands, flat_settings["eq_enabled"] = legacy_eq_inputs(eq_settings)
    native, native_failure = _native_simulate(
        audio_data, sample_rate, bands, flat_settings
    )
    if native is not None:
        native["simulation_backend"] = "rust"
        native["safety_authority"] = "authoritative"
        return native

    reason = (native_failure or {}).get("message", "native simulator unavailable")
    raise RuntimeError(f"This configuration requires native DSP simulation: {reason}")


def headroom_shortfall_db(validation: Mapping[str, Any]) -> int | None:
    """Whole decibels of level that let the full correction meet the rule, or None."""
    before = validation.get("before")
    excess = headroom_excess_db(before) if isinstance(before, Mapping) else float("inf")
    return int(np.ceil(excess)) if 0.0 < excess < float("inf") else None


def headroom_excess_split_db(validation: Mapping[str, Any]) -> tuple[float, float] | None:
    """Split the full correction's excess into (the recording's part, the part the correction adds).

    The recording's part is the excess with no correction at all (``flat_excess_db``, at
    most 0 when the recording alone fits). None when that was not rendered or the
    correction adds nothing.
    """
    before = validation.get("before")
    full = headroom_excess_db(before) if isinstance(before, Mapping) else float("inf")
    flat = validation.get("flat_excess_db")
    if isinstance(flat, bool) or not isinstance(flat, (int, float)) or not np.isfinite(flat):
        return None
    if not 0.0 < full < float("inf") or flat >= full:
        return None
    recording = max(float(flat), 0.0)
    return recording, full - recording


def _add_abstention(result: dict[str, Any], reason: str) -> None:
    reasons = list(result.get("abstention_reasons") or [])
    if reason not in reasons:
        reasons.append(reason)
    result["abstention_reasons"] = reasons
    result["recommendation_status"] = "abstain"
    result["apply_recommended"] = False


def _refresh_scaled_eq_metadata(
    result: dict[str, Any],
    analysis_freqs: np.ndarray | None,
    measured_db: np.ndarray | None,
    target_db: np.ndarray | None,
    fit_context: Mapping[str, Any] | None = None,
    sample_rate: float = SAMPLE_RATE,
) -> None:
    gains = np.asarray(result["band_gains"], dtype=float)
    centers = np.asarray(result.get("band_freqs", []), dtype=float)
    qs = np.asarray(result.get("band_qs", []), dtype=float)
    if not (gains.size == centers.size == qs.size == NUM_EQ_BANDS):
        return

    result["active_band_count"] = int(np.count_nonzero(np.abs(gains) >= 0.25))
    result["eq_quality"] = evaluate_eq_quality(
        centers,
        gains,
        qs,
        sample_rate=sample_rate,
    ).to_dict()
    result["max_adjacent_gain_difference_db"] = float(np.max(np.abs(np.diff(gains))))
    octave_gaps = np.maximum(np.diff(np.log2(np.clip(centers, 1e-6, None))), 1e-6)
    result["max_adjacent_gain_slope_db_per_octave"] = float(
        np.max(np.abs(np.diff(gains)) / octave_gaps)
    )

    if fit_context is None and all(
        value is not None for value in (analysis_freqs, measured_db, target_db)
    ):
        fit_context = _build_fit_context(
            np.asarray(analysis_freqs, dtype=float),
            np.asarray(measured_db, dtype=float),
            np.asarray(target_db, dtype=float),
            tilt_policy=str(result.get("spectral_tilt_policy", "voice_safe")),
            smoothing_strength=str(result.get("smoothing_strength", "conservative")),
        )
    if fit_context is not None:
        freqs = np.asarray(fit_context["freqs"], dtype=float)
        measured = np.asarray(fit_context["measured_db"], dtype=float)
        target = np.asarray(fit_context["target_db"], dtype=float)
        if freqs.shape == measured.shape == target.shape and freqs.size:
            weights = fit_context.get("weights")
            weights = _voice_weights(freqs) if weights is None else np.asarray(weights, dtype=float)
            before_error = weighted_target_error(
                freqs,
                measured,
                target,
                np.zeros_like(gains),
                qs,
                centers,
                weights,
                sample_rate=sample_rate,
            )
            after_error = weighted_target_error(
                freqs,
                measured,
                target,
                gains,
                qs,
                centers,
                weights,
                sample_rate=sample_rate,
            )
            result["validation_before_error_db"] = before_error
            result["validation_after_error_db"] = after_error
            validation_confidence = _validation_confidence(
                before_error,
                after_error,
                _result_float(result.get("validation_gain_scale"), 1.0),
            )
            result["validation_confidence"] = validation_confidence
            band_confidences = np.asarray(result.get("band_confidences", []), dtype=float)
            if band_confidences.size == NUM_EQ_BANDS:
                overall, eq_confidence, capture_confidence = _overall_confidence(
                    band_confidences,
                    gains,
                    _result_float(result.get("capture_confidence"), 1.0),
                    validation_confidence,
                )
                result["analysis_confidence"] = overall
                result["eq_confidence"] = eq_confidence
                result["capture_confidence"] = capture_confidence

    status = str(result.get("recommendation_status", "apply"))
    reasons = list(result.get("recommendation_reasons") or [])
    abstention_reasons = list(result.get("abstention_reasons") or [])
    # ``validation_gain_scale`` belongs to the optimizer's spectral-fit
    # attenuation.  Only the scale selected by this headroom simulation can
    # make a neutral candidate lose its usable correction.
    scale = _result_float(result.get("headroom_gain_scale"), 1.0)
    if scale < 1.0 and not np.any(np.abs(gains) >= 0.25):
        status = "abstain"
        reason = "headroom validation removed the usable correction"
        if reason not in abstention_reasons:
            abstention_reasons.append(reason)
    elif status == "apply" and (
        scale < 0.70
        or _result_float(result.get("analysis_confidence"), 1.0)
        < REDUCED_RECOMMENDATION_CONFIDENCE_THRESHOLD
    ):
        status = "reduced"
        reason = "headroom validation reduced the fitted correction"
        if reason not in reasons:
            reasons.append(reason)
    result["recommendation_status"] = status
    result["apply_recommended"] = status != "abstain"
    result["recommendation_reasons"] = reasons
    result["abstention_reasons"] = abstention_reasons


def apply_headroom_validation(
    audio_data: np.ndarray,
    sample_rate: int,
    eq_settings: dict[str, Any],
    chain_settings: dict[str, Any] | None = None,
    *,
    analysis_freqs: np.ndarray | None = None,
    measured_db: np.ndarray | None = None,
    target_db: np.ndarray | None = None,
    fit_context: Mapping[str, Any] | None = None,
    candidate_base_eq_settings: Mapping[str, Any] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Scale Auto-EQ boosts/cuts when offline chain simulation predicts headroom risk."""

    audio = np.asarray(audio_data, dtype=np.float32)
    result = deepcopy(eq_settings)
    original_gains = np.asarray(result.get("band_gains", []), dtype=float)
    if original_gains.size != NUM_EQ_BANDS:
        return result

    typed_base_eq = None
    if candidate_base_eq_settings is not None:
        from ...config import EQSettings

        try:
            typed_base_eq = EQSettings.from_dict(_typed_eq_payload(candidate_base_eq_settings))
        except (TypeError, ValueError) as exc:
            result["validated_candidate_eq"] = None
            result["headroom_validation"] = {
                "safe": False,
                "authoritative": False,
                "advisory": False,
                "status": "unavailable",
                "reason": f"incumbent EQ settings are invalid: {exc}"[:256],
            }
            result["headroom_safe"] = False
            result["headroom_advisory"] = False
            _add_abstention(result, "incumbent EQ is unavailable for full-chain validation")
            return result
        if (
            chain_settings is None
            or chain_settings.get("full_chain") is not True
            or not isinstance(chain_settings.get("input_pre_filtered"), bool)
        ):
            reason = "complete-chain settings are unavailable"
            result["validated_candidate_eq"] = None
            result["headroom_validation"] = {
                "safe": False,
                "authoritative": False,
                "advisory": False,
                "status": "unavailable",
                "reason": reason,
            }
            result["headroom_safe"] = False
            result["headroom_advisory"] = False
            _add_abstention(result, reason)
            return result

    def simulate_scale(scale: float) -> tuple[dict[str, Any], Any | None]:
        candidate_gains = original_gains * scale
        if typed_base_eq is None:
            simulation_settings = deepcopy(result)
            simulation_settings["band_gains"] = candidate_gains.tolist()
            return (
                simulate_candidate_chain(
                    audio, sample_rate, simulation_settings, chain_settings
                ),
                None,
            )

        from ...config import build_eq_candidate_settings

        candidate_eq = build_eq_candidate_settings(
            typed_base_eq,
            result.get("band_freqs", []),
            candidate_gains.tolist(),
            result.get("band_qs", []),
            layer="correction",
            enabled=bool(result.get("enabled", True)),
        )
        return (
            simulate_candidate_chain(
                audio,
                sample_rate,
                candidate_eq.to_dict(),
                chain_settings,
            ),
            candidate_eq,
        )

    def unavailable(exc: Exception) -> dict[str, Any]:
        reason = (
            "native full-chain headroom validation failed: "
            f"{type(exc).__name__}: {exc}"
        )[:256]
        result["validated_candidate_eq"] = None
        result["headroom_validation"] = {
            "safe": False,
            "authoritative": False,
            "advisory": False,
            "status": "unavailable",
            "reason": reason,
        }
        result["headroom_safe"] = False
        result["headroom_advisory"] = False
        _add_abstention(result, reason)
        return result

    check_analysis_cancelled(cancel_check)
    try:
        before, before_candidate = simulate_scale(1.0)
    except Exception as exc:
        if typed_base_eq is None:
            raise
        return unavailable(exc)
    selected = before
    selected_candidate = before_candidate
    selected_scale = 1.0
    selected_gains = original_gains.copy()
    rendered = [(1.0, before, before_candidate)]
    flat_excess = None
    if not is_headroom_safe(before):
        for scale in HEADROOM_SCALES[1:]:
            check_analysis_cancelled(cancel_check)
            try:
                simulation, candidate_eq = simulate_scale(scale)
            except Exception as exc:
                if typed_base_eq is None:
                    raise
                return unavailable(exc)
            rendered.append((scale, simulation, candidate_eq))
            selected = simulation
            selected_candidate = candidate_eq
            selected_scale = scale
            selected_gains = original_gains * scale
            if scale == 0.0:
                flat_excess = headroom_excess_db(simulation)
            if is_headroom_safe(simulation):
                break

    # No scale meets the rule, so ``selected`` is the render with no correction. A full-chain
    # candidate then gets in only if the user's current EQ, rendered as it is, is over the
    # rule too and the candidate leaves no more excess than that EQ does: the largest such
    # scale, or nothing, and the current EQ stays. (No correction at all is not the user's
    # current EQ: it drops the correction layer that EQ carries, and may be hotter.)
    reference_excess = None
    over_rule: dict[str, float] = {}
    accepted_over_rule = False
    declined = False
    excess = headroom_excess_db(selected)
    if excess > 0.0 and np.isfinite(excess):
        reference_excess = excess
        if typed_base_eq is not None and all(
            sim.get("simulation_backend") == "rust" for _, sim, _ in rendered
        ):
            check_analysis_cancelled(cancel_check)
            try:
                incumbent = simulate_candidate_chain(
                    audio, sample_rate, typed_base_eq.to_dict(), chain_settings
                )
            except Exception as exc:
                return unavailable(exc)
            incumbent_excess = headroom_excess_db(incumbent)
            if np.isfinite(incumbent_excess) and incumbent.get("simulation_backend") == "rust":
                reference_excess = incumbent_excess
                declined = True
                over_rule = {"flat_excess_db": excess, "incumbent_excess_db": incumbent_excess}
                peak = incumbent.get("output_true_peak_db")
                if isinstance(peak, (int, float, np.number)) and np.isfinite(peak):
                    over_rule["incumbent_output_true_peak_db"] = float(peak)
                # Every rendered scale is over the rule, so none is at or below an
                # incumbent that meets it.
                for scale, simulation, candidate_eq in rendered:
                    if headroom_excess_db(simulation) <= incumbent_excess:
                        selected = simulation
                        selected_candidate = candidate_eq
                        selected_scale = scale
                        selected_gains = original_gains * scale
                        accepted_over_rule, declined = True, False
                        break

    if flat_excess is None and selected_scale < 0.70:
        # The advice tells the recording's excess from what the correction adds. A failed
        # render only costs that split; the validated decision stands.
        check_analysis_cancelled(cancel_check)
        try:
            flat_excess = headroom_excess_db(simulate_scale(0.0)[0])
        except Exception:
            flat_excess = None

    result["band_gains"] = selected_gains.tolist()
    existing_scale = _result_float(result.get("validation_gain_scale"), 1.0)
    result["validation_gain_scale"] = float(existing_scale * selected_scale)
    result["headroom_gain_scale"] = float(selected_scale)
    _refresh_scaled_eq_metadata(
        result,
        analysis_freqs,
        measured_db,
        target_db,
        fit_context,
        sample_rate=sample_rate,
    )

    meets_thresholds = is_headroom_safe(selected)
    authoritative = selected.get("simulation_backend") == "rust"
    safe = bool(meets_thresholds and authoritative)
    if not safe and not accepted_over_rule:
        result["validation_confidence"] = float(
            min(_result_float(result.get("validation_confidence"), 1.0), 0.42)
        )
        result["analysis_confidence"] = float(
            min(_result_float(result.get("analysis_confidence"), 1.0), 0.58)
        )
    elif selected_scale < 1.0:
        result["validation_confidence"] = float(
            min(_result_float(result.get("validation_confidence"), 1.0), 0.72)
        )

    result["headroom_validation"] = {
        "safe": safe,
        "authoritative": authoritative,
        "advisory": not authoritative,
        "meets_advisory_thresholds": meets_thresholds,
        "gain_scale": selected_scale,
        "before": before,
        "after": selected,
        "status": (
            "safe"
            if safe
            else "risk_not_worsened"
            if accepted_over_rule
            else "risk_worse"
            if declined
            else "risk"
            if authoritative
            else "advisory"
        ),
    }
    if reference_excess is not None:
        result["headroom_validation"]["reference_excess_db"] = reference_excess
        result["headroom_validation"].update(over_rule)
    if flat_excess is not None and np.isfinite(flat_excess):
        result["headroom_validation"]["flat_excess_db"] = flat_excess
    native_failure = selected.get("native_simulation_failure")
    if native_failure is not None:
        result["headroom_validation"]["native_simulation_failure"] = native_failure
    result["headroom_safe"] = safe
    result["headroom_advisory"] = not authoritative
    result["headroom_gain_scale"] = selected_scale
    if typed_base_eq is not None:
        if (safe or accepted_over_rule) and selected_candidate is not None:
            result["validated_candidate_eq"] = selected_candidate.to_dict()
        else:
            result["validated_candidate_eq"] = None
            _add_abstention(
                result,
                "no safe candidate remained after full-chain headroom validation",
            )
    return result
