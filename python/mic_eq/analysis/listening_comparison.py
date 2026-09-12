"""Offline, same-capture before/after rendering for candidate review."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import logging
from typing import Any

import numpy as np

from .auto_eq import simulate_candidate_chain
from .cancellation import AnalysisCancelled


logger = logging.getLogger(__name__)

MAX_COMPARISON_SECONDS = 60.0
MAX_LEVEL_MATCH_DB = 12.0
PLAYBACK_PEAK_DB = -1.0
_MIN_DB = -120.0
_VAD_THRESHOLD = 0.48
_VAD_FRAME_SECONDS = 512.0 / 16_000.0

RENDERED_STAGES = (
    "input cleanup",
    "gate/VAD",
    "noise suppression",
    "correction EQ",
    "tone EQ",
    "deesser",
    "compressor",
    "limiter",
)
EXCLUDED_STAGES: tuple[str, ...] = ()


def _db_from_rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return _MIN_DB
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if not np.isfinite(rms) or rms <= 1.0e-12:
        return _MIN_DB
    return float(20.0 * np.log10(rms))


def _db_from_peak(samples: np.ndarray) -> float:
    if samples.size == 0:
        return _MIN_DB
    peak = float(np.max(np.abs(samples)))
    if not np.isfinite(peak) or peak <= 1.0e-12:
        return _MIN_DB
    return float(20.0 * np.log10(peak))


def _linear_from_db(value: float) -> float:
    return float(10.0 ** (float(value) / 20.0))


def _check_cancel(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise AnalysisCancelled()


def _validate_capture(audio_data: Any, sample_rate: int) -> np.ndarray:
    try:
        rate = int(sample_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_rate must be an integer") from exc
    if not 8_000 <= rate <= 768_000:
        raise ValueError("sample_rate must be between 8000 and 768000 Hz")

    audio = np.asarray(audio_data, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("comparison capture must be a mono one-dimensional array")
    if audio.size == 0:
        raise ValueError("comparison capture is empty")
    if audio.size > int(rate * MAX_COMPARISON_SECONDS):
        raise ValueError(
            f"comparison capture is longer than {MAX_COMPARISON_SECONDS:.0f} seconds"
        )
    if not np.isfinite(audio).all():
        raise ValueError("comparison capture contains non-finite samples")
    return np.ascontiguousarray(audio, dtype=np.float32).copy()


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _eq_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Accept EqPanel, Auto-EQ, or full Voice Setup settings."""
    nested = _as_mapping(settings.get("eq_settings")) or _as_mapping(
        settings.get("eq")
    )
    source = nested or settings

    # Keep the typed schema intact.  The native simulator accepts the same
    # six-field tuples as the live processor, so custom filter types and
    # slopes can be rendered without a second Python filter implementation.
    if "bands" in source:
        from ..config import EQSettings

        typed = EQSettings.from_dict(
            {
                key: source[key]
                for key in ("schema_version", "enabled", "bands", "layers")
                if key in source
            }
        )
        tone = typed.bands
        correction = typed.correction_bands or ()
        payload = typed.to_dict()
        payload.update(
            {
                "band_freqs": [band.frequency_hz for band in tone],
                "band_gains": [
                    band.gain_db if band.enabled else 0.0 for band in tone
                ],
                "band_qs": [band.q for band in tone],
                # These flat aliases are retained for callers that display
                # legacy metrics; simulate_candidate_chain uses the typed
                # layers above for native rendering.
                "eq_enabled": typed.enabled,
                "eq_bands": [band.to_native() for band in tone],
            }
        )
        if correction:
            payload["correction_bands"] = [
                band.to_native() for band in correction
            ]
        return payload

    # Keep this import local so importing the comparison helper does not load
    # the entire UI configuration surface in headless analysis tools.
    from ..config import EQ_FREQUENCIES

    frequencies = list(source.get("band_freqs") or EQ_FREQUENCIES)
    gains = list(source.get("band_gains") or [0.0] * 10)
    qs = list(source.get("band_qs") or [1.41] * 10)
    if not (len(frequencies) == len(gains) == len(qs) == 10):
        raise ValueError("comparison EQ settings must contain 10 complete bands")

    enabled = bool(source.get("enabled", source.get("eq_enabled", True)))
    if not enabled:
        gains = [0.0] * 10
    return {
        "band_freqs": [float(value) for value in frequencies],
        "band_gains": [float(value) for value in gains],
        "band_qs": [float(value) for value in qs],
    }


def _chain_settings(
    settings: Mapping[str, Any], override: Mapping[str, Any] | None
) -> dict[str, Any]:
    if override is not None:
        chain: dict[str, Any] = deepcopy(dict(override))
    else:
        nested = _as_mapping(settings.get("chain_settings"))
        chain = deepcopy(dict(nested)) if nested is not None else {}

    aliases = {
        "deesser": ("deesser", "deesser_settings"),
        "compressor": ("compressor", "compressor_settings"),
        "limiter": ("limiter", "limiter_settings"),
    }
    for component, keys in aliases.items():
        if component in chain:
            continue
        for key in keys:
            candidate = _as_mapping(settings.get(key))
            if candidate is not None:
                chain[component] = deepcopy(dict(candidate))
                break

    gate = _as_mapping(chain.get("gate")) or _as_mapping(
        chain.get("gate_settings")
    ) or _as_mapping(settings.get("gate")) or _as_mapping(
        settings.get("gate_settings")
    )
    if gate is not None:
        chain.update(
            {
                "gate_enabled": bool(gate.get("enabled", True)),
                "gate_threshold_db": gate.get("threshold_db", -40.0),
                "gate_attack_ms": gate.get("attack_ms", 10.0),
                "gate_release_ms": gate.get("release_ms", 100.0),
                "gate_mode": gate.get("gate_mode", 0),
                "gate_vad_threshold": gate.get("vad_threshold", 0.48),
                "gate_vad_hold_time_ms": gate.get("vad_hold_time_ms", 200.0),
                "gate_vad_pre_gain": gate.get("vad_pre_gain", 1.0),
                "gate_auto_threshold_enabled": gate.get(
                    "auto_threshold_enabled", True
                ),
                "gate_margin_db": gate.get("gate_margin_db", 10.0),
            }
        )

    suppressor = (
        _as_mapping(chain.get("suppressor"))
        or _as_mapping(chain.get("suppression"))
        or _as_mapping(chain.get("rnnoise"))
        or _as_mapping(chain.get("suppressor_settings"))
        or _as_mapping(settings.get("rnnoise"))
        or _as_mapping(settings.get("suppressor_settings"))
    )
    if suppressor is not None:
        chain.update(
            {
                "suppressor_enabled": bool(suppressor.get("enabled", True)),
                "suppressor_strength": suppressor.get("strength", 1.0),
                "noise_model": suppressor.get("model", "rnnoise"),
            }
        )

    for key in ("full_chain", "input_pre_filtered", "input_cleanup_mode", "processing_mode"):
        if key not in chain and key in settings:
            chain[key] = deepcopy(settings[key])
    if chain.get("full_chain"):
        # Full-chain audition captures use the optional pre-filter tap. Older
        # callers can still mark an already filtered capture explicitly.
        chain.setdefault("input_pre_filtered", False)

    # The native simulator only honors these deterministic stages. Retain any
    # other metadata for callers, but always request the rendered samples.
    chain["return_output_audio"] = True
    return chain


def _speech_mask(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, str]:
    """Return a capture-wide mask, preferring the existing VAD when available."""
    try:
        from mic_eq import CORE_AVAILABLE
        from mic_eq.mic_eq_core import analyze_vad_probabilities

        if CORE_AVAILABLE:
            probabilities = np.asarray(
                analyze_vad_probabilities(audio, int(sample_rate), _VAD_THRESHOLD),
                dtype=np.float32,
            )
            if probabilities.size:
                frame = max(1, int(round(sample_rate * _VAD_FRAME_SECONDS)))
                mask = np.repeat(probabilities >= _VAD_THRESHOLD, frame)
                mask = mask[: audio.size]
                if mask.size < audio.size:
                    mask = np.pad(mask, (0, audio.size - mask.size), constant_values=False)
                if np.any(mask):
                    return mask, "silero_vad"
    except Exception:
        logger.debug("VAD unavailable for listening comparison", exc_info=True)

    # Advisory fallback for reduced/native-less builds. The threshold is
    # capture-relative and therefore does not claim to identify speech.
    magnitude = np.abs(audio.astype(np.float64, copy=False))
    threshold = max(1.0e-4, float(np.percentile(magnitude, 35.0)))
    mask = magnitude >= threshold
    if not np.any(mask):
        mask = np.ones(audio.size, dtype=bool)
    return mask, "level_gate_fallback"


def _speech_level(samples: np.ndarray, mask: np.ndarray) -> float:
    selected = samples[mask] if mask.size == samples.size and np.any(mask) else samples
    return _db_from_rms(selected)


def _simulation_audio(simulation: Mapping[str, Any], expected_size: int) -> np.ndarray:
    raw = simulation.get("output_audio")
    if raw is None:
        raise RuntimeError("offline simulator did not return rendered audio")
    output = np.asarray(raw, dtype=np.float32)
    if output.ndim != 1 or output.size != expected_size:
        raise RuntimeError("offline simulator returned audio with the wrong length")
    if not np.isfinite(output).all():
        raise RuntimeError("offline simulator returned non-finite audio")
    return np.ascontiguousarray(output, dtype=np.float32).copy()


def _playback_safe(
    samples: np.ndarray,
    *,
    level_match_gain_db: float,
) -> tuple[np.ndarray, float]:
    matched = samples.astype(np.float32, copy=True)
    if level_match_gain_db:
        matched *= np.float32(_linear_from_db(level_match_gain_db))

    peak = float(np.max(np.abs(matched))) if matched.size else 0.0
    safe_peak = _linear_from_db(PLAYBACK_PEAK_DB)
    safety_gain_db = 0.0
    if np.isfinite(peak) and peak > safe_peak:
        safety_gain_db = float(20.0 * np.log10(safe_peak / peak))
        matched *= np.float32(_linear_from_db(safety_gain_db))
    return np.ascontiguousarray(matched, dtype=np.float32), safety_gain_db


@dataclass(frozen=True)
class RenderedComparisonClip:
    """One playback clip and its un-matched level evidence."""

    label: str
    samples: np.ndarray
    actual_level_db: float
    actual_level_delta_db: float
    playback_level_db: float
    peak_db: float
    level_match_gain_db: float
    safety_gain_db: float
    simulation_backend: str
    simulation: Mapping[str, Any] | None = None

    @property
    def duration_seconds(self) -> float:
        sample_rate = self.simulation.get("sample_rate") if self.simulation else None
        if isinstance(sample_rate, (int, float)) and sample_rate > 0:
            return float(self.samples.size / sample_rate)
        return 0.0


@dataclass(frozen=True)
class ComparisonRenderResult:
    """All three renderings from one immutable capture."""

    sample_rate: int
    original: RenderedComparisonClip
    current: RenderedComparisonClip
    proposed: RenderedComparisonClip
    level_match: bool
    level_match_reference_db: float
    speech_detection: str
    alignment_samples: int
    alignment_ms: float
    rendered_stages: tuple[str, ...] = RENDERED_STAGES
    excluded_stages: tuple[str, ...] = EXCLUDED_STAGES
    native_authoritative: bool = False

    @property
    def scope_label(self) -> str:
        return "Offline full-chain input, EQ, suppression, and dynamics preview"

    @property
    def clips(self) -> tuple[RenderedComparisonClip, ...]:
        return (self.original, self.current, self.proposed)

    def clip(self, label: str) -> RenderedComparisonClip:
        for clip in self.clips:
            if clip.label == label:
                return clip
        raise KeyError(label)


def _render_processed(
    audio: np.ndarray,
    sample_rate: int,
    settings: Mapping[str, Any],
    chain_override: Mapping[str, Any] | None,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    chain = _chain_settings(settings, chain_override)
    simulation = simulate_candidate_chain(
        audio,
        sample_rate,
        _eq_settings(settings),
        chain,
    )
    if not isinstance(simulation, Mapping):
        raise RuntimeError("offline simulator returned an invalid result")
    if chain.get("full_chain") and simulation.get("simulation_backend") != "rust":
        failure = simulation.get("native_simulation_failure")
        detail = "native full-chain renderer is unavailable"
        if isinstance(failure, Mapping) and failure.get("message"):
            detail += f": {failure['message']}"
        raise RuntimeError(detail)
    return _simulation_audio(simulation, audio.size), simulation


def render_comparison(
    audio_data: Any,
    sample_rate: int,
    current_settings: Mapping[str, Any],
    proposed_settings: Mapping[str, Any],
    *,
    current_chain_settings: Mapping[str, Any] | None = None,
    proposed_chain_settings: Mapping[str, Any] | None = None,
    level_match: bool = False,
    cancel_check: Callable[[], bool] | None = None,
) -> ComparisonRenderResult:
    """Render unaltered, current, and proposed audio from the same capture.

    The existing Rust simulation removes its deterministic chain latency before
    returning ``output_audio``. ``alignment_samples`` reports that native
    offset so callers can show the evidence while the three clips share the
    capture timeline. Python fallback output is same-length and marked
    advisory by ``native_authoritative``.
    """
    audio = _validate_capture(audio_data, sample_rate)
    rate = int(sample_rate)
    if not isinstance(current_settings, Mapping) or not isinstance(
        proposed_settings, Mapping
    ):
        raise ValueError("current and proposed settings must be mappings")

    current_chain = _chain_settings(current_settings, current_chain_settings)
    proposed_chain = _chain_settings(proposed_settings, proposed_chain_settings)
    full_chain = bool(
        current_chain.get("full_chain") or proposed_chain.get("full_chain")
    )
    _check_cancel(cancel_check)
    current_audio, current_simulation = _render_processed(
        audio, rate, current_settings, current_chain
    )
    _check_cancel(cancel_check)
    proposed_audio, proposed_simulation = _render_processed(
        audio, rate, proposed_settings, proposed_chain
    )
    _check_cancel(cancel_check)

    mask, detection = _speech_mask(audio, rate)
    reference_level_db = _speech_level(audio, mask)
    current_level_db = _speech_level(current_audio, mask)
    proposed_level_db = _speech_level(proposed_audio, mask)

    def prepare(
        label: str,
        rendered: np.ndarray,
        actual_level_db: float,
        simulation: Mapping[str, Any] | None,
    ) -> RenderedComparisonClip:
        delta = float(actual_level_db - reference_level_db)
        match_gain = 0.0
        if level_match and label != "original" and np.isfinite(delta):
            match_gain = float(np.clip(-delta, -MAX_LEVEL_MATCH_DB, MAX_LEVEL_MATCH_DB))
        playback, safety_gain = _playback_safe(
            rendered,
            level_match_gain_db=match_gain,
        )
        simulation_backend = (
            str(simulation.get("simulation_backend", "unknown"))
            if simulation is not None
            else "capture"
        )
        simulation_copy = dict(simulation) if simulation is not None else None
        if simulation_copy is not None:
            simulation_copy["sample_rate"] = rate
        return RenderedComparisonClip(
            label=label,
            samples=playback,
            actual_level_db=float(actual_level_db),
            actual_level_delta_db=delta,
            playback_level_db=_speech_level(playback, mask),
            peak_db=_db_from_peak(playback),
            level_match_gain_db=match_gain,
            safety_gain_db=float(safety_gain),
            simulation_backend=simulation_backend,
            simulation=simulation_copy,
        )

    original = prepare("original", audio, reference_level_db, None)
    current = prepare("current", current_audio, current_level_db, current_simulation)
    proposed = prepare("proposed", proposed_audio, proposed_level_db, proposed_simulation)
    latencies = [
        int(value)
        for value in (
            current_simulation.get("chain_latency_samples", 0),
            proposed_simulation.get("chain_latency_samples", 0),
        )
        if isinstance(value, (int, float)) and np.isfinite(float(value)) and value >= 0
    ]
    alignment_samples = max(latencies, default=0)
    native_authoritative = all(
        clip.simulation_backend == "rust" for clip in (current, proposed)
    )
    return ComparisonRenderResult(
        sample_rate=rate,
        original=original,
        current=current,
        proposed=proposed,
        level_match=bool(level_match),
        level_match_reference_db=float(reference_level_db),
        speech_detection=detection,
        alignment_samples=alignment_samples,
        alignment_ms=float(alignment_samples * 1000.0 / rate),
        rendered_stages=RENDERED_STAGES if full_chain else ("tone EQ", "deesser", "compressor", "limiter"),
        excluded_stages=EXCLUDED_STAGES if full_chain else (
            "input cleanup",
            "gate/VAD",
            "noise suppression",
        ),
        native_authoritative=native_authoritative,
    )


__all__ = [
    "ComparisonRenderResult",
    "EXCLUDED_STAGES",
    "MAX_COMPARISON_SECONDS",
    "MAX_LEVEL_MATCH_DB",
    "PLAYBACK_PEAK_DB",
    "RENDERED_STAGES",
    "RenderedComparisonClip",
    "render_comparison",
]
