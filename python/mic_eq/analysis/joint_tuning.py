"""Bounded joint tuning for noise suppression, gating, and dynamics.

The live gate and suppressor are evaluated on the two captures separately.
The resulting speech/noise renders are then sent through the existing native
downstream simulator so the recommendation cannot claim that an EQ-only
render verified the input stages.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import time
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from .auto_eq import simulate_candidate_chain
from .cancellation import AnalysisCancelled, check_analysis_cancelled

_SIMULATION_RATE = 48_000
_FRAME_SAMPLES = 480
_MIN_SPEECH_RETAINED = 0.72
_MAX_FALSE_CLOSURE = 0.08
_MIN_TAIL_RETAINED = 0.45
_MIN_NOISE_ATTENUATION_DB = 2.0
_MAX_CHATTER_EVENTS = 8
_MAX_RUNTIME_FACTOR = 1.0
_IMPROVEMENT_MARGIN = 0.03
_MAX_CANDIDATES_PER_MODEL = 9
_SUPPORTED_NOISE_MODELS = ("rnnoise", "deepfilter-ll", "deepfilter")
_DEEPFILTER_MODELS = ("deepfilter-ll", "deepfilter")
_DEEPFILTER_SELECTION_POLICY = "retain_deepfilter_family_v1"
# Native contracts are 10 ms for RNNoise/DeepFilterNet LL and 30 ms for the
# standard DeepFilterNet model. Keep a small 5 ms allowance for the reported
# frame-boundary measurement while rejecting a future high-latency backend.
_DEFAULT_MAX_SUPPRESSOR_LATENCY_MS = 35.0
_MAX_CONFIGURED_SUPPRESSOR_LATENCY_MS = 100.0


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), 1.0e-12)))


def _rms(audio: np.ndarray) -> float:
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def _frame_rms(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.empty(0, dtype=float)
    return np.asarray(
        [
            _rms(values[start : start + _FRAME_SAMPLES])
            for start in range(0, values.size, _FRAME_SAMPLES)
        ],
        dtype=float,
    )


def _to_simulation_rate(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if int(sample_rate) == _SIMULATION_RATE:
        return np.ascontiguousarray(values)
    divisor = int(np.gcd(int(sample_rate), _SIMULATION_RATE))
    return np.ascontiguousarray(
        resample_poly(
            values.astype(np.float64, copy=False),
            _SIMULATION_RATE // divisor,
            int(sample_rate) // divisor,
        ),
        dtype=np.float32,
    )


def _align_probabilities(
    probabilities: np.ndarray | None,
    source_sample_count: int,
    target_sample_count: int,
) -> np.ndarray:
    target_count = max(1, (int(target_sample_count) + _FRAME_SAMPLES - 1) // _FRAME_SAMPLES)
    if probabilities is None:
        return np.zeros(target_count, dtype=np.float32)
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        return np.zeros(target_count, dtype=np.float32)
    source_duration = max(float(target_sample_count) / _SIMULATION_RATE, 1.0e-9)
    source_times = (np.arange(values.size, dtype=float) + 0.5) * source_duration / values.size
    target_times = (np.arange(target_count, dtype=float) + 0.5) * _FRAME_SAMPLES / _SIMULATION_RATE
    return np.asarray(
        np.interp(
            target_times,
            source_times,
            np.clip(values, 0.0, 1.0),
            left=float(np.clip(values[0], 0.0, 1.0)),
            right=float(np.clip(values[-1], 0.0, 1.0)),
        ),
        dtype=np.float32,
    )


def _energy_probabilities(audio: np.ndarray, noise_floor_db: float) -> np.ndarray:
    frames = _frame_rms(audio)
    if frames.size == 0:
        return np.zeros(1, dtype=np.float32)
    levels = np.asarray([_db(value) for value in frames], dtype=float)
    # This is only a simulator control signal when Silero was unavailable;
    # the live threshold-only gate still remains the authoritative behavior.
    return np.asarray(
        np.clip((levels - float(noise_floor_db) - 4.0) / 18.0, 0.0, 1.0),
        dtype=np.float32,
    )


def _tail_mask(active: np.ndarray) -> np.ndarray:
    values = np.asarray(active, dtype=bool).reshape(-1)
    tail = np.zeros(values.size, dtype=bool)
    tail_frames = max(1, int(round(0.20 * _SIMULATION_RATE / _FRAME_SAMPLES)))
    for index in np.flatnonzero(values[:-1] & ~values[1:]):
        tail[index + 1 : min(values.size, index + 1 + tail_frames)] = True
    return tail


def _retained_ratio(
    output_rms: np.ndarray,
    input_rms: np.ndarray,
    mask: np.ndarray,
) -> float:
    count = min(output_rms.size, input_rms.size, mask.size)
    if count == 0 or not np.any(mask[:count]):
        return 1.0
    ratios = output_rms[:count][mask[:count]] / np.maximum(input_rms[:count][mask[:count]], 1.0e-9)
    return float(np.clip(np.median(ratios), 0.0, 4.0))


def _candidate_settings(
    gate_settings: Mapping[str, Any],
    *,
    noise_floor_db: float,
    strength: float,
    threshold_delta_db: float,
    release_delta_ms: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gate = dict(gate_settings)
    auto_threshold = bool(gate.get("auto_threshold_enabled", False)) and int(
        gate.get("gate_mode", 0)
    ) > 0
    if auto_threshold:
        base_margin = float(gate.get("gate_margin_db", 10.0))
        margin = _clamp(base_margin + threshold_delta_db, 0.0, 20.0)
        threshold = _clamp(noise_floor_db + margin, -80.0, -10.0)
        gate["gate_margin_db"] = margin
    else:
        threshold = _clamp(
            float(gate.get("threshold_db", -40.0)) + threshold_delta_db,
            -80.0,
            -10.0,
        )
        gate["threshold_db"] = threshold
    gate["release_ms"] = _clamp(
        float(gate.get("release_ms", 120.0)) + release_delta_ms,
        5.0,
        1_000.0,
    )
    # The native offline gate accepts the manual threshold. For auto-threshold
    # candidates, use the capture's measured floor plus the selected margin.
    simulator_gate = {
        "gate_enabled": bool(gate.get("enabled", True)),
        "gate_vad_threshold": float(gate.get("vad_threshold", 0.48)),
        "gate_vad_hold_time_ms": float(gate.get("vad_hold_time_ms", 200.0)),
        "gate_vad_pre_gain": float(gate.get("vad_pre_gain", 1.0)),
        "gate_auto_threshold_enabled": bool(gate.get("auto_threshold_enabled", True)),
        "gate_margin_db": float(gate.get("gate_margin_db", 10.0)),
        "gate_threshold_db": threshold,
        "gate_attack_ms": _clamp(float(gate.get("attack_ms", 5.0)), 1.0, 100.0),
        "gate_release_ms": gate["release_ms"],
        "gate_mode": int(_clamp(float(gate.get("gate_mode", 0)), 0.0, 2.0)),
    }
    return gate, simulator_gate


def _load_native_simulator() -> Callable[..., Any] | None:
    try:
        from mic_eq import CORE_AVAILABLE
        from mic_eq.mic_eq_core import simulate_gate_suppressor_order
    except (ImportError, OSError):
        return None
    return simulate_gate_suppressor_order if CORE_AVAILABLE else None


def _run_gate_suppressor(
    simulator: Callable[..., Any],
    audio: np.ndarray,
    probabilities: np.ndarray,
    settings: Mapping[str, Any],
    *,
    suppressor_strength: float,
    noise_model: str,
) -> dict[str, Any]:
    result = simulator(
        np.ascontiguousarray(audio, dtype=np.float32),
        probabilities.tolist(),
        False,
        float(suppressor_strength),
        {
            **dict(settings),
            "noise_model": str(noise_model),
        },
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("native gate/suppressor simulator returned a non-mapping result")
    normalized = {str(key): value for key, value in result.items()}
    output = np.asarray(normalized.get("output_audio", []), dtype=np.float32).reshape(-1)
    if output.size != audio.size or not np.isfinite(output).all():
        raise RuntimeError("native gate/suppressor simulator returned invalid audio")
    normalized["output_audio"] = output
    return normalized


def _metric_float(result: Mapping[str, Any], key: str, default: float) -> float:
    value = result.get(key, default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if np.isfinite(parsed) else float(default)


def _score(metrics: Mapping[str, float], compressor: Mapping[str, Any]) -> float:
    target_p95 = float(compressor.get("target_p95_reduction_db", 3.5))
    target_median = float(
        compressor.get("target_median_reduction_db", target_p95 * 0.42)
    )
    noise_attenuation = float(metrics["noise_attenuation_db"])
    noise_quality = _clamp(
        (noise_attenuation - _MIN_NOISE_ATTENUATION_DB) / 12.0,
        0.0,
        1.0,
    )
    dynamics_error = (
        abs(float(metrics["compressor_p95_db"]) - target_p95) / 2.0
        + 0.45 * abs(float(metrics["compressor_median_db"]) - target_median)
    )
    return float(
        1.9 * max(0.0, _MIN_SPEECH_RETAINED - metrics["speech_retained_ratio"])
        + 1.4 * metrics["false_closure_rate"]
        + 1.2 * max(0.0, _MIN_TAIL_RETAINED - metrics["tail_retained_ratio"])
        + 0.85 * (1.0 - noise_quality)
        + 0.35 * dynamics_error
        + 0.20 * max(0.0, metrics["compressor_pumping_db"]) / 5.0
        + 0.50 * max(0.0, -metrics["pre_limiter_headroom_db"])
        + 0.08 * metrics["chatter_events"]
        + 0.06 * metrics["runtime_factor"]
        + 0.06
        * metrics["suppressor_latency_ms"]
        / max(metrics["max_suppressor_latency_ms"], 1.0e-9)
    )


def _gates(metrics: Mapping[str, float]) -> dict[str, bool]:
    return {
        "active_speech_retention": metrics["speech_retained_ratio"] >= _MIN_SPEECH_RETAINED,
        "false_closure": metrics["false_closure_rate"] <= _MAX_FALSE_CLOSURE,
        "gate_tail": metrics["tail_retained_ratio"] >= _MIN_TAIL_RETAINED,
        "noise_reduction": metrics["noise_attenuation_db"] >= _MIN_NOISE_ATTENUATION_DB,
        "chatter": metrics["chatter_events"] <= _MAX_CHATTER_EVENTS,
        "dynamics": bool(
            metrics["compressor_p95_db"]
            <= metrics["compressor_target_p95_db"] + 1.25
            and metrics["compressor_peak_db"]
            <= metrics["compressor_peak_cap_db"] + 0.25
        ),
        "headroom": bool(
            metrics["output_true_peak_db"]
            <= metrics["limiter_ceiling_db"] + 0.15
            and metrics["pre_limiter_headroom_db"] >= 0.0
        ),
        "finite": metrics["non_finite_output"] < 0.5,
        "runtime": metrics["runtime_factor"] <= _MAX_RUNTIME_FACTOR,
        "suppressor_latency": metrics["suppressor_latency_ms"]
        <= metrics["max_suppressor_latency_ms"],
    }


def _canonical_noise_model(value: Any) -> str:
    model = str(value).strip().lower()
    if model in {"deepfilterll", "deepfilternet-ll"}:
        return "deepfilter-ll"
    if model == "deepfilternet":
        return "deepfilter"
    return model


def _model_selection_metadata(incumbent_model: str) -> dict[str, Any]:
    if incumbent_model in _DEEPFILTER_MODELS:
        candidate_models = list(_DEEPFILTER_MODELS)
        excluded_models = ["rnnoise"]
        reason = (
            "retain the DeepFilter family after unrestricted DeepFilterNet LL "
            "to RNNoise selections failed the independent clean-speech gate"
        )
        policy = _DEEPFILTER_SELECTION_POLICY
    else:
        candidate_models = list(_SUPPORTED_NOISE_MODELS)
        excluded_models = []
        reason = "evaluate every supported suppressor model"
        policy = "all_supported_models_v1"
    return {
        "id": policy,
        "incumbent_model": incumbent_model,
        "candidate_models": candidate_models,
        "excluded_models": excluded_models,
        "reason": reason,
    }


def _model_order(incumbent_model: str) -> list[str]:
    metadata = _model_selection_metadata(incumbent_model)
    return [
        incumbent_model,
        *(
            model
            for model in metadata["candidate_models"]
            if model != incumbent_model
        ),
    ]


def _latency_samples(result: Mapping[str, Any]) -> int:
    value = result.get("suppressor_latency_samples")
    if value is None or isinstance(value, bool):
        raise RuntimeError("native simulator returned an invalid suppressor latency")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise RuntimeError("native simulator did not report suppressor latency") from None
    if not np.isfinite(parsed) or parsed < 0.0 or parsed != round(parsed):
        raise RuntimeError("native simulator returned an invalid suppressor latency")
    return int(parsed)


def _validate_latency_limit(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("max_suppressor_latency_ms must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError("max_suppressor_latency_ms must be finite") from None
    if (
        not np.isfinite(parsed)
        or parsed <= 0.0
        or parsed > _MAX_CONFIGURED_SUPPRESSOR_LATENCY_MS
    ):
        raise ValueError(
            "max_suppressor_latency_ms must be greater than 0 and at most "
            f"{_MAX_CONFIGURED_SUPPRESSOR_LATENCY_MS:g}"
        )
    return parsed


def _model_evaluation(
    model: str,
    candidates: list[dict[str, Any]],
    *,
    status: str = "available",
    reason: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": model,
        "status": status,
        "candidate_count": len(candidates),
        "passing_candidate_count": sum(
            bool(candidate.get("passes", False)) for candidate in candidates
        ),
    }
    if candidates:
        best = min(candidates, key=lambda candidate: float(candidate["score"]))
        result["best_score"] = float(best["score"])
        result["best_latency_ms"] = float(
            best["metrics"]["suppressor_latency_ms"]
        )
    if reason:
        result["reason"] = str(reason)[:256]
    return result


def _unavailable_result(
    gate_settings: Mapping[str, Any],
    *,
    noise_model: str,
    suppressor_strength: float,
    suppressor_enabled: bool,
    incumbent_suppressor: Mapping[str, Any] | None,
    reason: str,
    runtime_ms: float,
    max_suppressor_latency_ms: float = _DEFAULT_MAX_SUPPRESSOR_LATENCY_MS,
    model_evaluations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    retained_suppressor = dict(incumbent_suppressor or {})
    retained_suppressor.setdefault("enabled", bool(suppressor_enabled))
    retained_suppressor.setdefault("strength", _clamp(float(suppressor_strength), 0.0, 1.0))
    retained_suppressor.setdefault("model", str(noise_model))
    retained_suppressor["auto_tuned"] = False
    model_selection = _model_selection_metadata(str(noise_model))
    return {
        "status": "unavailable",
        "decision": "retain_incumbent",
        "apply_recommended": False,
        "evidence_scope": "joint_noise_speech_gate_suppressor_downstream",
        "objective": "speech_retention_noise_reduction_dynamics_safety_v1",
        "reason": str(reason)[:256],
        "reasons": [str(reason)[:256]],
        "candidate_count": 0,
        "selected_candidate": None,
        "suppressor_settings": retained_suppressor,
        "gate_settings": dict(gate_settings),
        "runtime_ms": float(runtime_ms),
        "gates": {},
        "candidates": [],
        "model_evaluations": list(model_evaluations or []),
        "model_selection_policy": model_selection,
        "max_suppressor_latency_ms": float(max_suppressor_latency_ms),
        "latency_tradeoff": (
            "RNNoise and DeepFilterNet LL target 10 ms; standard DeepFilterNet "
            "target is 30 ms and is accepted only within the configured bound"
        ),
    }


def tune_gate_suppression_dynamics(
    noise_audio: np.ndarray,
    speech_audio: np.ndarray,
    sample_rate: int,
    gate_settings: Mapping[str, Any],
    compressor_settings: Mapping[str, Any],
    eq_settings: Mapping[str, Any],
    deesser_settings: Mapping[str, Any],
    limiter_settings: Mapping[str, Any],
    *,
    vad_probabilities: np.ndarray | None = None,
    noise_vad_probabilities: np.ndarray | None = None,
    noise_model: str = "rnnoise",
    suppressor_strength: float = 1.0,
    suppressor_enabled: bool = True,
    incumbent_settings: Mapping[str, Any] | None = None,
    noise_floor_db: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
    max_suppressor_latency_ms: float = _DEFAULT_MAX_SUPPRESSOR_LATENCY_MS,
) -> dict[str, Any]:
    """Select a safe joint gate/suppressor candidate with downstream evidence.

    The search is deliberately small and deterministic. It never returns a
    candidate for application unless every predefined gate passes; this keeps
    weak captures and unavailable native assets on the incumbent settings.
    """
    started = time.perf_counter()
    check_analysis_cancelled(cancel_check)
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 192000:
        raise ValueError("Unsupported capture sample rate")
    latency_limit_ms = _validate_latency_limit(max_suppressor_latency_ms)
    base_gate = dict(gate_settings)
    strength = _clamp(float(suppressor_strength), 0.0, 1.0)
    incumbent_payload = dict(incumbent_settings or {})
    incumbent_gate = dict(incumbent_payload.get("gate") or base_gate)
    incumbent_suppressor_payload = dict(
        incumbent_payload.get("suppressor")
        or {
            "enabled": bool(suppressor_enabled),
            "strength": strength,
            "model": str(noise_model),
        }
    )
    incumbent_enabled = bool(
        incumbent_suppressor_payload.get("enabled", suppressor_enabled)
    )
    incumbent_strength = _clamp(
        float(incumbent_suppressor_payload.get("strength", strength)), 0.0, 1.0
    )
    requested_model = _canonical_noise_model(noise_model)
    incumbent_model = _canonical_noise_model(
        incumbent_suppressor_payload.get("model", requested_model)
    )
    model_order = _model_order(incumbent_model)
    if not incumbent_enabled:
        return _unavailable_result(
            incumbent_gate,
            noise_model=incumbent_model,
            suppressor_strength=incumbent_strength,
            suppressor_enabled=False,
            incumbent_suppressor=incumbent_suppressor_payload,
            reason="noise suppression is disabled in the incumbent settings",
            runtime_ms=(time.perf_counter() - started) * 1000.0,
            max_suppressor_latency_ms=latency_limit_ms,
            model_evaluations=[
                _model_evaluation(
                    model,
                    [],
                    status="skipped" if model != incumbent_model else "disabled",
                    reason="incumbent suppression is disabled",
                )
                for model in model_order
            ],
        )
    noise = np.asarray(noise_audio, dtype=np.float32).reshape(-1)
    speech = np.asarray(speech_audio, dtype=np.float32).reshape(-1)
    if (
        noise.size == 0
        or speech.size == 0
        or not np.isfinite(noise).all()
        or not np.isfinite(speech).all()
    ):
        return _unavailable_result(
            incumbent_gate,
            noise_model=incumbent_model,
            suppressor_strength=strength,
            suppressor_enabled=incumbent_enabled,
            incumbent_suppressor=incumbent_suppressor_payload,
            reason="joint tuning captures were empty or non-finite",
            runtime_ms=(time.perf_counter() - started) * 1000.0,
            max_suppressor_latency_ms=latency_limit_ms,
            model_evaluations=[
                _model_evaluation(
                    model,
                    [],
                    status="skipped",
                    reason="captures were empty or non-finite",
                )
                for model in model_order
            ],
        )

    simulator = _load_native_simulator()
    if simulator is None:
        return _unavailable_result(
            incumbent_gate,
            noise_model=incumbent_model,
            suppressor_strength=strength,
            suppressor_enabled=incumbent_enabled,
            incumbent_suppressor=incumbent_suppressor_payload,
            reason="native gate/suppressor simulator is unavailable",
            runtime_ms=(time.perf_counter() - started) * 1000.0,
            max_suppressor_latency_ms=latency_limit_ms,
            model_evaluations=[
                _model_evaluation(
                    model,
                    [],
                    status="unavailable",
                    reason="native gate/suppressor simulator is unavailable",
                )
                for model in model_order
            ],
        )

    simulation_noise = _to_simulation_rate(noise, sample_rate)
    simulation_speech = _to_simulation_rate(speech, sample_rate)
    # Split before estimating fallback context so held-out noise cannot affect
    # the candidate threshold or the energy-VAD baseline.
    noise_split = (simulation_noise.size // (2 * _FRAME_SAMPLES)) * _FRAME_SAMPLES
    speech_split = (simulation_speech.size // (2 * _FRAME_SAMPLES)) * _FRAME_SAMPLES
    noise_floor_source = (
        "explicit_precomputed_context"
        if noise_floor_db is not None
        else "training_noise_split"
    )
    measured_noise_floor = (
        float(noise_floor_db)
        if noise_floor_db is not None
        else _db(_rms(simulation_noise[:noise_split]))
    )
    speech_probabilities = _align_probabilities(
        vad_probabilities,
        speech.size,
        simulation_speech.size,
    )
    noise_probabilities = _align_probabilities(
        noise_vad_probabilities,
        noise.size,
        simulation_noise.size,
    )
    if vad_probabilities is None:
        speech_probabilities = _energy_probabilities(simulation_speech, measured_noise_floor)
    if noise_vad_probabilities is None:
        noise_probabilities = np.zeros(
            max(1, (simulation_noise.size + _FRAME_SAMPLES - 1) // _FRAME_SAMPLES),
            dtype=np.float32,
        )
    active = np.asarray(speech_probabilities >= 0.40, dtype=bool)
    if float(np.mean(active)) < 0.05:
        active = _energy_probabilities(simulation_speech, measured_noise_floor) >= 0.40

    base_release = float(base_gate.get("release_ms", 120.0))
    deltas = (
        (0.0, 0.0, 0.0),
        (-0.25, -3.0, 0.0),
        (-0.25, 3.0, 0.0),
        (0.15, -3.0, 0.0),
        (0.15, 3.0, 0.0),
        (0.0, -3.0, -35.0),
        (0.0, 3.0, 35.0),
        (0.0, 0.0, -70.0),
        (0.0, 0.0, 70.0),
    )
    candidates: list[dict[str, Any]] = []
    failure_reasons: list[str] = []
    model_failure_reasons: dict[str, list[str]] = {}

    def evaluate(
        candidate_model: str,
        candidate_base: Mapping[str, Any],
        requested_strength: float,
        threshold_delta: float,
        release_delta: float,
        is_incumbent: bool,
        evaluation_noise: np.ndarray,
        evaluation_speech: np.ndarray,
        evaluation_noise_probabilities: np.ndarray,
        evaluation_speech_probabilities: np.ndarray,
    ) -> dict[str, Any] | None:
        active = np.asarray(evaluation_speech_probabilities >= 0.40, dtype=bool)
        tails = _tail_mask(active)
        check_analysis_cancelled(cancel_check)
        candidate_strength = _clamp(float(requested_strength), 0.0, 1.0)
        candidate_gate, simulator_gate = _candidate_settings(
            candidate_base,
            noise_floor_db=measured_noise_floor,
            strength=candidate_strength,
            threshold_delta_db=threshold_delta,
            release_delta_ms=release_delta,
        )
        try:
            rendered_noise = _run_gate_suppressor(
                simulator,
                evaluation_noise,
                evaluation_noise_probabilities,
                simulator_gate,
                suppressor_strength=candidate_strength,
                noise_model=candidate_model,
            )
            rendered_speech = _run_gate_suppressor(
                simulator,
                evaluation_speech,
                evaluation_speech_probabilities,
                simulator_gate,
                suppressor_strength=candidate_strength,
                noise_model=candidate_model,
            )
            noise_latency_samples = _latency_samples(rendered_noise)
            speech_latency_samples = _latency_samples(rendered_speech)
            if noise_latency_samples != speech_latency_samples:
                raise RuntimeError(
                    "native simulator reported inconsistent suppressor latency"
                )
            suppressor_latency_ms = (
                speech_latency_samples * 1000.0 / _SIMULATION_RATE
            )
            check_analysis_cancelled(cancel_check)
            downstream_chain = {
                "deesser": dict(deesser_settings),
                "compressor": dict(compressor_settings),
                "limiter": dict(limiter_settings),
            }
            speech_downstream = simulate_candidate_chain(
                rendered_speech["output_audio"],
                _SIMULATION_RATE,
                dict(eq_settings),
                downstream_chain,
            )
            noise_downstream = simulate_candidate_chain(
                rendered_noise["output_audio"],
                _SIMULATION_RATE,
                dict(eq_settings),
                downstream_chain,
            )
            check_analysis_cancelled(cancel_check)
            if (
                speech_downstream.get("simulation_backend") != "rust"
                or noise_downstream.get("simulation_backend") != "rust"
            ):
                raise RuntimeError("native downstream simulator is unavailable")
            noise_in_rms = _rms(evaluation_noise)
            noise_out_rms = _rms(
                np.asarray(rendered_noise["output_audio"], dtype=float)
            )
            speech_in_rms = _frame_rms(evaluation_speech)
            speech_out_rms = _frame_rms(
                np.asarray(rendered_speech["output_audio"], dtype=float)
            )
            active_count = min(active.size, speech_in_rms.size, speech_out_rms.size)
            active_slice = active[:active_count]
            speech_ratios = speech_out_rms[:active_count] / np.maximum(
                speech_in_rms[:active_count], 1.0e-9
            )
            false_closure_rate = (
                float(np.mean(speech_ratios[active_slice] < 0.25))
                if np.any(active_slice)
                else 1.0
            )
            metrics = {
                "speech_retained_ratio": _retained_ratio(
                    speech_out_rms, speech_in_rms, active
                ),
                "false_closure_rate": false_closure_rate,
                "tail_retained_ratio": _retained_ratio(
                    speech_out_rms, speech_in_rms, tails
                ),
                "noise_attenuation_db": _db(noise_in_rms) - _db(noise_out_rms),
                "chatter_events": float(
                    int(rendered_speech.get("gate_chatter_event_count", 0))
                    + int(rendered_noise.get("gate_chatter_event_count", 0))
                ),
                "compressor_p95_db": _metric_float(
                    speech_downstream, "compressor_gain_reduction_p95_db", 120.0
                ),
                "compressor_median_db": _metric_float(
                    speech_downstream, "compressor_gain_reduction_median_db", 120.0
                ),
                "compressor_peak_db": _metric_float(
                    speech_downstream, "compressor_gain_reduction_db", 120.0
                ),
                "compressor_pumping_db": _metric_float(
                    speech_downstream, "compressor_pumping_score_db", 120.0
                ),
                "compressor_target_p95_db": float(
                    compressor_settings.get("target_p95_reduction_db", 3.5)
                ),
                "compressor_peak_cap_db": float(
                    compressor_settings.get("peak_reduction_cap_db", 8.0)
                ),
                "output_true_peak_db": _metric_float(
                    speech_downstream, "output_true_peak_db", 120.0
                ),
                "limiter_ceiling_db": _metric_float(
                    speech_downstream,
                    "limiter_effective_ceiling_db",
                    float(limiter_settings.get("ceiling_db", -0.5)),
                ),
                "pre_limiter_headroom_db": _metric_float(
                    speech_downstream, "pre_limiter_true_peak_headroom_db", -120.0
                ),
                "non_finite_output": float(
                    bool(speech_downstream.get("non_finite_output", True))
                ),
                "runtime_factor": (
                    _metric_float(rendered_noise, "runtime_ms", 0.0)
                    + _metric_float(rendered_speech, "runtime_ms", 0.0)
                    + _metric_float(speech_downstream, "candidate_runtime_ms", 0.0)
                    + _metric_float(noise_downstream, "candidate_runtime_ms", 0.0)
                )
                / max(
                    1.0,
                    1000.0
                    * (evaluation_noise.size + evaluation_speech.size)
                    / _SIMULATION_RATE,
                ),
                "suppressor_latency_samples": float(speech_latency_samples),
                "suppressor_latency_ms": suppressor_latency_ms,
                "max_suppressor_latency_ms": latency_limit_ms,
            }
            candidate = {
                "model": candidate_model,
                "gate": candidate_gate,
                "suppressor_strength": candidate_strength,
                "simulator_gate": simulator_gate,
                "metrics": metrics,
                "gates": _gates(metrics),
                "is_incumbent": is_incumbent,
            }
            candidate["score"] = _score(metrics, compressor_settings)
            candidate["passes"] = bool(all(candidate["gates"].values()))
            return candidate
        except AnalysisCancelled:
            raise
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            failure_reasons.append(f"{candidate_model} unavailable: {reason}")
            model_failure_reasons.setdefault(candidate_model, []).append(reason)
            return None

    # Split at complete 10 ms frames; only the first half selects parameters.
    if min(noise_split, speech_split) < _FRAME_SAMPLES:
        return _unavailable_result(
            incumbent_gate,
            noise_model=incumbent_model,
            suppressor_strength=incumbent_strength,
            suppressor_enabled=incumbent_enabled,
            incumbent_suppressor=incumbent_suppressor_payload,
            reason="joint tuning captures are too short for a held-out split",
            runtime_ms=(time.perf_counter() - started) * 1000.0,
            max_suppressor_latency_ms=latency_limit_ms,
            model_evaluations=[
                _model_evaluation(
                    model,
                    [],
                    status="skipped",
                    reason="captures are too short for a held-out split",
                )
                for model in model_order
            ],
        )

    model_evaluations: list[dict[str, Any]] = []
    for candidate_model in model_order:
        check_analysis_cancelled(cancel_check)
        is_incumbent_model = candidate_model == incumbent_model
        candidate_specs: list[
            tuple[Mapping[str, Any], float, float, float, float, bool]
        ] = []
        if is_incumbent_model:
            candidate_specs.append(
                (incumbent_gate, incumbent_strength, 0.0, 0.0, 0.0, True)
            )
        else:
            candidate_specs.append((base_gate, strength, 0.0, 0.0, 0.0, False))
        for strength_delta, threshold_delta, release_delta in deltas:
            if len(candidate_specs) >= _MAX_CANDIDATES_PER_MODEL:
                break
            candidate_specs.append(
                (
                    base_gate,
                    strength * (1.0 + strength_delta),
                    strength_delta,
                    threshold_delta,
                    release_delta,
                    False,
                )
            )

        evaluated_for_model: list[dict[str, Any]] = []
        for (
            candidate_base,
            requested_strength,
            _delta,
            threshold_delta,
            release_delta,
            candidate_is_incumbent,
        ) in candidate_specs:
            candidate = evaluate(
                candidate_model,
                candidate_base,
                requested_strength,
                threshold_delta,
                release_delta,
                candidate_is_incumbent,
                simulation_noise[:noise_split],
                simulation_speech[:speech_split],
                noise_probabilities[: noise_split // _FRAME_SAMPLES],
                speech_probabilities[: speech_split // _FRAME_SAMPLES],
            )
            if candidate is None:
                # A model's first attempt is its backend availability probe.
                # Do not spend the remaining bounded budget repeating a known
                # unavailable backend; retain the reason in model_evaluations.
                if not evaluated_for_model:
                    break
                continue
            candidates.append(candidate)
            evaluated_for_model.append(candidate)
        model_evaluations.append(
            _model_evaluation(
                candidate_model,
                evaluated_for_model,
                status="available" if evaluated_for_model else "unavailable",
                reason=(
                    model_failure_reasons.get(candidate_model, ["model unavailable"])[0]
                    if not evaluated_for_model
                    else None
                ),
            )
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    incumbent = next(
        (candidate for candidate in candidates if candidate["is_incumbent"]),
        None,
    )
    if incumbent is None:
        return _unavailable_result(
            incumbent_gate,
            noise_model=incumbent_model,
            suppressor_strength=incumbent_strength,
            suppressor_enabled=incumbent_enabled,
            incumbent_suppressor=incumbent_suppressor_payload,
            reason=failure_reasons[0] if failure_reasons else "joint candidate evaluation failed",
            runtime_ms=elapsed_ms,
            max_suppressor_latency_ms=latency_limit_ms,
            model_evaluations=model_evaluations,
        )

    passing = [candidate for candidate in candidates if candidate["passes"]]
    best = (
        min(
            passing,
            key=lambda candidate: (
                float(candidate["score"]),
                float(candidate["metrics"]["suppressor_latency_ms"]),
                float(candidate["suppressor_strength"]),
                float(candidate["gate"].get("release_ms", base_release)),
                model_order.index(candidate["model"]),
            ),
        )
        if passing
        else incumbent
    )
    selected = best
    decision = "apply_candidate"
    if best is incumbent or (
        incumbent.get("passes", False)
        and float(incumbent["score"]) - float(best["score"]) <= _IMPROVEMENT_MARGIN
    ):
        selected = incumbent
        decision = "retain_incumbent"
    holdout = None
    holdout_improved = False
    if decision == "apply_candidate":
        holdout = evaluate(
            selected["model"],
            selected["gate"],
            selected["suppressor_strength"],
            0.0,
            0.0,
            False,
            simulation_noise[noise_split:],
            simulation_speech[speech_split:],
            noise_probabilities[noise_split // _FRAME_SAMPLES :],
            speech_probabilities[speech_split // _FRAME_SAMPLES :],
        )
        holdout_incumbent = evaluate(
            incumbent_model,
            incumbent_gate,
            incumbent_strength,
            0.0,
            0.0,
            True,
            simulation_noise[noise_split:],
            simulation_speech[speech_split:],
            noise_probabilities[noise_split // _FRAME_SAMPLES :],
            speech_probabilities[speech_split // _FRAME_SAMPLES :],
        )
        holdout_improved = bool(
            holdout is not None and holdout["passes"] and holdout_incumbent is not None
            and (not holdout_incumbent["passes"]
                 or float(holdout_incumbent["score"]) - float(holdout["score"]) > _IMPROVEMENT_MARGIN)
        )
        if not holdout_improved:
            decision = "retain_incumbent"
            selected = incumbent
    if decision == "retain_incumbent":
        selected_gate = dict(incumbent_gate)
        selected_strength = incumbent_strength
        selected_model = incumbent_model
    elif passing:
        selected_gate = dict(selected["gate"])
        selected_strength = float(selected["suppressor_strength"])
        selected_model = str(selected["model"])
    else:
        selected_model = incumbent_model
    reasons: list[str] = []
    model_selection = _model_selection_metadata(incumbent_model)
    if decision == "retain_incumbent":
        reasons.append("incumbent retained unless a bounded candidate clearly improves all gates")
    else:
        reasons.append("bounded candidate passed speech, noise, dynamics, safety, and runtime gates")
    if not passing:
        reasons.append("no candidate passed every predefined gate")
    if failure_reasons:
        reasons.append(f"{len(failure_reasons)} candidate evaluations were unavailable")
    return {
        "status": "accepted" if decision == "apply_candidate" else "retained",
        "decision": decision,
        "apply_recommended": decision == "apply_candidate",
        "holdout_passed": holdout_improved,
        "holdout_gates": dict(holdout["gates"]) if holdout is not None else {},
        "evidence_scope": "joint_noise_speech_gate_suppressor_downstream",
        "objective": "speech_retention_noise_reduction_dynamics_safety_v1",
        "reasons": reasons,
        "candidate_count": len(candidates),
        "selected_candidate": {
            "noise_model": selected_model,
            "suppressor_strength": selected_strength,
            "gate_settings": selected_gate,
            "score": float(selected["score"]),
            "passes": bool(selected["passes"]),
            "metrics": dict(selected["metrics"]),
            "gates": dict(selected["gates"]),
        },
        "suppressor_settings": {
            "enabled": bool(incumbent_enabled),
            "strength": selected_strength,
            "model": selected_model,
            "auto_tuned": decision == "apply_candidate",
        },
        "gate_settings": selected_gate,
        "runtime_ms": (time.perf_counter() - started) * 1000.0,
        "noise_floor_db": measured_noise_floor,
        "noise_floor_source": noise_floor_source,
        "gates": dict(selected["gates"]),
        "model_evaluations": model_evaluations,
        "model_selection_policy": model_selection,
        "max_suppressor_latency_ms": latency_limit_ms,
        "latency_tradeoff": (
            "RNNoise and DeepFilterNet LL target 10 ms; standard DeepFilterNet "
            "target is 30 ms and is accepted only within the configured bound"
        ),
        "candidates": [
            {
                "noise_model": candidate["model"],
                "suppressor_strength": float(candidate["suppressor_strength"]),
                "gate_settings": dict(candidate["gate"]),
                "score": float(candidate["score"]),
                "passes": bool(candidate["passes"]),
                "metrics": dict(candidate["metrics"]),
                "gates": dict(candidate["gates"]),
            }
            for candidate in candidates
        ],
        "failure_reasons": failure_reasons[:4],
    }


__all__ = ["tune_gate_suppression_dynamics"]
