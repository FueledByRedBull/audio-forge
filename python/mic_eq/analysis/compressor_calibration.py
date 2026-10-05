"""Shared candidate evaluation and threshold-only compressor calibration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import time
from typing import Any

import numpy as np

from .auto_eq import simulate_candidate_chain
from .auto_eq_parts.headroom import _is_headroom_safe
from .cancellation import check_analysis_cancelled
from .vad import map_causal_vad_probabilities
from ..config import LimiterSettings

_DEFAULT_LIMITER_SETTINGS: dict[str, Any] = asdict(LimiterSettings())
_LIMITER_SETTING_KEYS = frozenset(_DEFAULT_LIMITER_SETTINGS)


def _normalise_limiter_settings(
    settings: Mapping[str, Any] | None = None,
    *,
    require_complete: bool = False,
) -> dict[str, Any] | None:
    if settings is None:
        return None if require_complete else dict(_DEFAULT_LIMITER_SETTINGS)
    if not isinstance(settings, Mapping):
        raise ValueError("limiter settings must be a mapping")
    missing = _LIMITER_SETTING_KEYS.difference(settings)
    if missing:
        if require_complete:
            return None
        raise ValueError(
            "limiter settings are missing: " + ", ".join(sorted(missing))
        )
    return dict(settings)


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


_COMPRESSOR_SEARCH_BUDGET = 68
_COMPRESSOR_P95_TARGET_TOLERANCE_DB = 0.75
_COMPRESSOR_SEARCH_BOUNDS = {
    "threshold_db": (-55.0, -6.0),
    "ratio": (1.5, 6.0),
    "attack_ms": (3.0, 25.0),
    "release_ms": (60.0, 320.0),
}
_COMPRESSOR_OBJECTIVE_NORMALIZERS = {
    "loudness_error_db": 2.0,
    "median_gr_error_db": 1.0,
    "p95_gr_error_db": 1.0,
    "headroom_shortfall_db": 1.0,
    "pumping_score_db": 1.0,
    "silence_gain_excess_db": 1.0,
    "activity_ratio_deficit": 0.20,
}
_COMPRESSOR_OBJECTIVE_WEIGHTS = {
    "loudness": 1.00,
    "median_gr": 0.35,
    "p95_gr": 0.90,
    "headroom": 0.45,
    "pumping": 0.30,
    "silence_gain": 1.50,
    "activity": 0.25,
    "prior": 0.08,
}


def _compressor_p95_target_status(
    calibration: Mapping[str, Any],
    target_p95_db: float,
) -> tuple[bool, str | None]:
    measured = calibration.get(
        "measured_p95_gain_reduction_db",
        calibration.get("measured_gain_reduction_db"),
    )
    if (
        calibration.get("backend") != "rust"
        or not isinstance(measured, (int, float))
        or isinstance(measured, bool)
        or not np.isfinite(float(measured))
        or not np.isfinite(float(target_p95_db))
    ):
        return False, "compressor p95 gain reduction target is unavailable"

    measured_db = float(measured)
    if abs(measured_db - float(target_p95_db)) <= _COMPRESSOR_P95_TARGET_TOLERANCE_DB:
        return True, None
    return (
        False,
        "compressor p95 gain reduction target missed: "
        f"measured {measured_db:.2f} dB, target {target_p95_db:.2f} dB "
        f"(±{_COMPRESSOR_P95_TARGET_TOLERANCE_DB:.2f} dB)",
    )


def _huber(value: float) -> float:
    magnitude = abs(float(value))
    return 0.5 * magnitude * magnitude if magnitude <= 1.0 else magnitude - 0.5


class _CompressorCalibration:
    """One capture's bounded candidate cache and native winner verification."""

    def __init__(
        self,
        *,
        speech_audio: np.ndarray,
        sample_rate: int,
        eq_settings: dict[str, Any],
        deesser_settings: dict[str, Any],
        compressor_settings: dict[str, Any],
        target_p95_db: float,
        target_median_db: float,
        peak_cap_db: float,
        limiter_settings: Mapping[str, Any] | None = None,
        vad_probabilities: np.ndarray | None = None,
        auto_makeup_activity: list[tuple[float, float, float, float]] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> None:
        self.speech_audio = speech_audio
        self.sample_rate = sample_rate
        self.eq_settings = eq_settings
        self.deesser_settings = deesser_settings
        self.target_p95_db = target_p95_db
        self.target_median_db = target_median_db
        self.peak_cap_db = peak_cap_db
        self.auto_makeup_activity = auto_makeup_activity
        self.cancel_check = cancel_check
        self.progress_callback = progress_callback
        self.calibrated = dict(compressor_settings)
        effective_limiter_settings = _normalise_limiter_settings(limiter_settings)
        if effective_limiter_settings is None:
            raise ValueError("limiter settings are incomplete")
        self.effective_limiter_settings = effective_limiter_settings
        self.diagnostics: dict[str, Any] = {
            "backend": "unavailable",
            "objective": "bounded_multi_objective_compressor_search_v2",
            "target_p95_gain_reduction_db": target_p95_db,
            "target_median_gain_reduction_db": target_median_db,
            "target_p95_tolerance_db": _COMPRESSOR_P95_TARGET_TOLERANCE_DB,
            "target_p95_met": False,
            "peak_gain_reduction_cap_db": peak_cap_db,
            "measured_p95_gain_reduction_db": None,
            "measured_median_gain_reduction_db": None,
            "measured_peak_gain_reduction_db": None,
            "silence_level_delta_db": None,
            "silence_analysis_block_count": 0,
            "iterations": 0,
            "candidate_budget": _COMPRESSOR_SEARCH_BUDGET,
            "objective_normalizers": dict(_COMPRESSOR_OBJECTIVE_NORMALIZERS),
            "objective_weights": dict(_COMPRESSOR_OBJECTIVE_WEIGHTS),
            "expanded_search_status": "disabled_unqualified",
            "expanded_search_selected": False,
            "expanded_candidate_objective": None,
        }
        self.started = time.perf_counter()
        self.mapped_vad_probabilities: np.ndarray | None = None
        if vad_probabilities is not None and auto_makeup_activity is None:
            control_block_size = max(1, int(round(sample_rate * 0.010)))
            block_count = (
                speech_audio.size + control_block_size - 1
            ) // control_block_size
            self.mapped_vad_probabilities = map_causal_vad_probabilities(
                vad_probabilities,
                np.arange(block_count, dtype=np.int64) * control_block_size,
                sample_rate,
            )
        self.incumbent = {
            key: _clamp(
                float(self.calibrated[key]),
                *_COMPRESSOR_SEARCH_BOUNDS[key],
            )
            for key in _COMPRESSOR_SEARCH_BOUNDS
        }
        self.evaluated: dict[
            tuple[float, ...], tuple[float, dict[str, Any], dict[str, float]]
        ] = {}

    @staticmethod
    def key_for(candidate: Mapping[str, float]) -> tuple[float, ...]:
        return tuple(round(float(candidate[key]), 6) for key in _COMPRESSOR_SEARCH_BOUNDS)

    def evaluate(
        self,
        candidate_values: Mapping[str, float],
    ) -> tuple[float, dict[str, Any], dict[str, float]]:
        check_analysis_cancelled(self.cancel_check)
        candidate = dict(self.calibrated)
        candidate.update(
            {
                key: _clamp(
                    float(candidate_values[key]),
                    *_COMPRESSOR_SEARCH_BOUNDS[key],
                )
                for key in _COMPRESSOR_SEARCH_BOUNDS
            }
        )
        # Evaluate the same compressor controls that will be applied.  In
        # particular, auto makeup changes both loudness and downstream
        # headroom, so disabling it here would calibrate a different chain.
        simulation_compressor = dict(candidate)
        simulation_chain: dict[str, Any] = {
            "deesser": self.deesser_settings,
            "compressor": simulation_compressor,
            "limiter": dict(self.effective_limiter_settings),
        }
        if self.auto_makeup_activity is not None:
            simulation_chain["auto_makeup_activity"] = self.auto_makeup_activity
        elif self.mapped_vad_probabilities is not None:
            simulation_chain["vad_probabilities"] = self.mapped_vad_probabilities
        simulation = simulate_candidate_chain(
            self.speech_audio.astype(np.float32, copy=False),
            self.sample_rate,
            self.eq_settings,
            simulation_chain,
        )
        if simulation.get("simulation_backend") != "rust":
            return (
                float("inf"),
                simulation,
                dict(candidate_values),
            )
        peak = float(simulation.get("compressor_gain_reduction_db", 0.0))
        median = float(
            simulation.get("compressor_gain_reduction_median_db", peak)
        )
        p95 = float(simulation.get("compressor_gain_reduction_p95_db", peak))
        active_ratio = float(
            simulation.get("compressor_gain_reduction_active_ratio", 0.0)
        )
        active_gain = float(simulation.get("active_output_gain_db", 0.0))
        target_lufs = float(self.calibrated.get("target_lufs", -18.0))
        measured_lufs = float(self.calibrated.get("measured_short_term_lufs", -18.0))
        input_rms_db = float(simulation.get("input_rms_db", np.nan))
        output_rms_db = float(simulation.get("output_rms_db", np.nan))
        if np.isfinite(input_rms_db) and np.isfinite(output_rms_db):
            # The native simulator reports RMS rather than LUFS.  Preserve the
            # measured capture's LUFS/RMS offset while scoring the actual
            # downstream gain delivered by this candidate.
            output_lufs = measured_lufs + output_rms_db - input_rms_db
        else:
            output_lufs = measured_lufs + active_gain
        output_true_peak = float(simulation.get("output_true_peak_db", 120.0))
        ceiling = float(
            simulation.get(
                "limiter_effective_ceiling_db",
                self.effective_limiter_settings["ceiling_db"],
            )
        )
        pre_limiter_headroom = float(
            simulation.get("pre_limiter_true_peak_headroom_db", -120.0)
        )
        pumping = float(simulation.get("compressor_pumping_score_db", 120.0))
        silence_delta = simulation.get("silence_level_delta_db", float("inf"))
        silence_gain = float("inf") if silence_delta is None else float(silence_delta)
        non_finite = bool(simulation.get("non_finite_output", True))
        finite_values = np.asarray(
            [
                peak,
                median,
                p95,
                active_ratio,
                output_lufs,
                output_true_peak,
                pre_limiter_headroom,
                pumping,
                silence_gain,
            ],
            dtype=float,
        )
        hard_rejected = bool(
            non_finite
            or not np.isfinite(finite_values).all()
            or output_true_peak > ceiling + 0.10
            or peak > self.peak_cap_db + 1.0e-6
        )
        prior_terms = []
        for key, (lower, upper) in _COMPRESSOR_SEARCH_BOUNDS.items():
            span = upper - lower
            prior_terms.append(
                ((float(candidate[key]) - self.incumbent[key]) / span) ** 2
            )
        terms = {
            "loudness": _huber(
                (output_lufs - target_lufs)
                / _COMPRESSOR_OBJECTIVE_NORMALIZERS["loudness_error_db"]
            ),
            "median_gr": _huber(
                (median - self.target_median_db)
                / _COMPRESSOR_OBJECTIVE_NORMALIZERS["median_gr_error_db"]
            ),
            "p95_gr": _huber(
                (p95 - self.target_p95_db)
                / _COMPRESSOR_OBJECTIVE_NORMALIZERS["p95_gr_error_db"]
            ),
            "headroom": _huber(
                max(0.0, 1.0 - pre_limiter_headroom)
                / _COMPRESSOR_OBJECTIVE_NORMALIZERS["headroom_shortfall_db"]
            ),
            "pumping": _huber(
                pumping / _COMPRESSOR_OBJECTIVE_NORMALIZERS["pumping_score_db"]
            ),
            "silence_gain": _huber(
                max(0.0, silence_gain - 0.25)
                / _COMPRESSOR_OBJECTIVE_NORMALIZERS["silence_gain_excess_db"]
            ),
            "activity": _huber(
                max(0.0, 0.20 - active_ratio)
                / _COMPRESSOR_OBJECTIVE_NORMALIZERS["activity_ratio_deficit"]
            ),
            "prior": float(np.mean(prior_terms)),
        }
        score = sum(
            _COMPRESSOR_OBJECTIVE_WEIGHTS[name] * value
            for name, value in terms.items()
        )
        if hard_rejected:
            score = float("inf")
        return (
            float(score),
            simulation,
            {key: float(candidate[key]) for key in _COMPRESSOR_SEARCH_BOUNDS},
        )

    def evaluate_batch(self, candidates: list[dict[str, float]]) -> None:
        pending = {}
        for candidate in candidates:
            key = self.key_for(candidate)
            if key not in self.evaluated and key not in pending:
                pending[key] = candidate
        items = list(pending.items())[: _COMPRESSOR_SEARCH_BUDGET - 1 - len(self.evaluated)]
        # Native simulations release the GIL and own independent DSP state.
        # Small batches bound cancellation latency and concurrent CPU use.
        with ThreadPoolExecutor(max_workers=4) as pool:
            for start in range(0, len(items), 4):
                check_analysis_cancelled(self.cancel_check)
                batch = items[start : start + 4]
                for (key, _), result in zip(
                    batch, pool.map(self.evaluate, [v for _, v in batch])
                ):
                    self.evaluated[key] = result
                    if self.progress_callback is not None:
                        self.progress_callback(
                            f"Calibrating compression ({len(self.evaluated)} candidates checked)...",
                            60 + int(30 * len(self.evaluated) / _COMPRESSOR_SEARCH_BUDGET),
                        )

    def threshold_candidates(self) -> list[dict[str, float]]:
        initial_candidates = [self.incumbent]
        for threshold in np.linspace(-55.0, -6.0, 33):
            check_analysis_cancelled(self.cancel_check)
            threshold_candidate = dict(self.incumbent)
            threshold_candidate["threshold_db"] = float(threshold)
            initial_candidates.append(threshold_candidate)
        return initial_candidates

    def ranked_results(self) -> list[tuple[float, dict[str, Any], dict[str, float]]]:
        return sorted(
            (item for item in self.evaluated.values() if np.isfinite(item[0])),
            key=lambda item: (item[0], self.key_for(item[2])),
        )

    def threshold_only(
        self,
        feasible: list[tuple[float, dict[str, Any], dict[str, float]]],
    ) -> tuple[float, dict[str, Any], dict[str, float]] | None:
        return min(
            (
                item
                for item in feasible
                if all(
                    abs(item[2][key] - self.incumbent[key]) <= 1.0e-6
                    for key in ("ratio", "attack_ms", "release_ms")
                )
            ),
            key=lambda item: (item[0], self.key_for(item[2])),
            default=None,
        )

    def finish(
        self,
        chosen: tuple[float, dict[str, Any], dict[str, float]] | None,
        *,
        expanded: tuple[float, dict[str, Any], dict[str, float]] | None = None,
        expanded_selected: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if chosen is None:
            self.diagnostics["iterations"] = len(self.evaluated)
            self.diagnostics["search_runtime_ms"] = (
                time.perf_counter() - self.started
            ) * 1000.0
            return self.calibrated, self.diagnostics

        best_score, best_simulation, best_values = chosen
        self.calibrated.update(best_values)
        check_analysis_cancelled(self.cancel_check)
        # Recheck the winner with the exact controls that will be applied.
        winner_chain: dict[str, Any] = {
            "deesser": self.deesser_settings,
            "compressor": dict(self.calibrated),
            "limiter": dict(self.effective_limiter_settings),
        }
        if self.auto_makeup_activity is not None:
            winner_chain["auto_makeup_activity"] = self.auto_makeup_activity
        elif self.mapped_vad_probabilities is not None:
            winner_chain["vad_probabilities"] = self.mapped_vad_probabilities
        winner_verification = simulate_candidate_chain(
            self.speech_audio.astype(np.float32, copy=False),
            self.sample_rate,
            self.eq_settings,
            winner_chain,
        )
        if winner_verification.get("simulation_backend") == "rust":
            best_simulation = winner_verification
        median = float(best_simulation["compressor_gain_reduction_median_db"])
        p95 = float(best_simulation["compressor_gain_reduction_p95_db"])
        peak = float(best_simulation["compressor_gain_reduction_db"])
        active_ratio = float(best_simulation["compressor_gain_reduction_active_ratio"])
        target_p95_met, _target_status_reason = _compressor_p95_target_status(
            {
                "backend": "rust",
                "measured_p95_gain_reduction_db": p95,
            },
            self.target_p95_db,
        )
        threshold_only_scores = [
            score
            for score, _, values in self.evaluated.values()
            if all(
                abs(values[key] - self.incumbent[key]) <= 1.0e-6
                for key in ("ratio", "attack_ms", "release_ms")
            )
        ]
        incumbent_entry = self.evaluated.get(self.key_for(self.incumbent))
        self.diagnostics.update(
            {
                "backend": "rust",
                "target_p95_met": target_p95_met,
                "target_p95_tolerance_db": _COMPRESSOR_P95_TARGET_TOLERANCE_DB,
                "measured_median_gain_reduction_db": median,
                "measured_p95_gain_reduction_db": p95,
                "measured_peak_gain_reduction_db": peak,
                "active_reduction_ratio": active_ratio,
                "peak_cap_passed": peak <= self.peak_cap_db + 1.0e-6,
                "total_objective": best_score,
                "incumbent_objective": (
                    incumbent_entry[0] if incumbent_entry is not None else float("inf")
                ),
                "threshold_only_objective": min(
                    threshold_only_scores, default=float("inf")
                ),
                "expanded_candidate_objective": (
                    expanded[0] if expanded is not None else None
                ),
                "expanded_search_selected": expanded_selected,
                "active_output_gain_db": float(
                    best_simulation.get("active_output_gain_db", 0.0)
                ),
                "silence_level_delta_db": best_simulation.get("silence_level_delta_db"),
                "silence_analysis_block_count": int(
                    best_simulation.get("silence_analysis_block_count", 0)
                ),
                "compressor_pumping_score_db": float(
                    best_simulation.get("compressor_pumping_score_db", 0.0)
                ),
                "output_true_peak_db": float(
                    best_simulation.get("output_true_peak_db", -120.0)
                ),
                "headroom_safe": _is_headroom_safe(best_simulation),
                "limiter_gain_reduction_db": float(
                    best_simulation.get("limiter_gain_reduction_db", 0.0)
                ),
                "true_peak_limiter_gain_reduction_db": float(
                    best_simulation.get("true_peak_limiter_gain_reduction_db", 0.0)
                ),
                "auto_makeup_simulated": bool(
                    self.calibrated.get("auto_makeup_enabled", False)
                ),
                "pre_limiter_true_peak_headroom_db": float(
                    best_simulation.get("pre_limiter_true_peak_headroom_db", 0.0)
                ),
                "search_runtime_ms": (time.perf_counter() - self.started) * 1000.0,
                "candidate_count": len(self.evaluated) + 1,
                "iterations": len(self.evaluated) + 1,
                # Compatibility aliases for older diagnostic consumers.
                "target_gain_reduction_db": self.target_p95_db,
                "measured_gain_reduction_db": p95,
                "threshold_db": self.calibrated["threshold_db"],
                "ratio": self.calibrated["ratio"],
                "attack_ms": self.calibrated["attack_ms"],
                "release_ms": self.calibrated["release_ms"],
            }
        )
        return self.calibrated, self.diagnostics


def _calibrate_compressor_threshold(
    *,
    speech_audio: np.ndarray,
    sample_rate: int,
    eq_settings: dict[str, Any],
    deesser_settings: dict[str, Any],
    compressor_settings: dict[str, Any],
    target_p95_db: float,
    target_median_db: float,
    peak_cap_db: float,
    limiter_settings: Mapping[str, Any] | None = None,
    vad_probabilities: np.ndarray | None = None,
    auto_makeup_activity: list[tuple[float, float, float, float]] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[str, int], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Calibrate only threshold while preserving the selected compressor controls."""
    run = _CompressorCalibration(
        speech_audio=speech_audio,
        sample_rate=sample_rate,
        eq_settings=eq_settings,
        deesser_settings=deesser_settings,
        compressor_settings=compressor_settings,
        target_p95_db=target_p95_db,
        target_median_db=target_median_db,
        peak_cap_db=peak_cap_db,
        limiter_settings=limiter_settings,
        vad_probabilities=vad_probabilities,
        auto_makeup_activity=auto_makeup_activity,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )
    run.evaluate_batch(run.threshold_candidates())
    return run.finish(run.threshold_only(run.ranked_results()))
