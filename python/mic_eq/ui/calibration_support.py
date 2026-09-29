"""Shared analysis and status helpers for calibration dialogs."""

from __future__ import annotations

import logging
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import CORE_AVAILABLE
from ..config import EQSettings, Preset

logger = logging.getLogger(__name__)

RAINBOW_PASSAGE = """The Rainbow Passage
The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end. People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries, men have been fascinated by this spectacle of the sky. They have tried to find explanations for it. Scientists, however, have found that it is caused by the reflection and refraction of sunlight on drops of water in the air."""

TOO_QUIET_DB = -40.0
TOO_LOUD_DB = -3.0


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class AutoEqCandidate:
    """Immutable snapshot of the incumbent and its merged typed EQ proposal."""

    incumbent_preset_json: str
    proposed_preset_json: str
    chain_settings_json: str

    @classmethod
    def create(
        cls,
        incumbent: Preset,
        candidate_eq: EQSettings,
        chain: dict[str, Any],
    ) -> "AutoEqCandidate":
        incumbent_json = _canonical_json(incumbent.to_dict())
        proposed_payload = incumbent.to_dict()
        proposed_payload["eq"] = candidate_eq.to_dict()
        proposed = Preset.from_dict(proposed_payload)
        return cls(
            incumbent_json,
            _canonical_json(proposed.to_dict()),
            _canonical_json(chain),
        )

    @property
    def incumbent_signature(self) -> str:
        return hashlib.sha256(self.incumbent_preset_json.encode("utf-8")).hexdigest()

    @property
    def proposed_signature(self) -> str:
        return hashlib.sha256(self.proposed_preset_json.encode("utf-8")).hexdigest()

    @property
    def incumbent_preset(self) -> Preset:
        return Preset.from_dict(json.loads(self.incumbent_preset_json))

    @property
    def proposed_preset(self) -> Preset:
        return Preset.from_dict(json.loads(self.proposed_preset_json))

    @property
    def chain(self) -> dict[str, Any]:
        return json.loads(self.chain_settings_json)

    def matches(self, incumbent: Preset, chain: dict[str, Any]) -> bool:
        return (
            self.incumbent_preset_json == _canonical_json(incumbent.to_dict())
            and self.chain_settings_json == _canonical_json(chain)
        )


def candidate_metadata(
    scope: str,
    *,
    target: dict[str, Any],
    capture: dict[str, Any],
    options: dict[str, Any],
    allowed_scope: tuple[str, ...],
    verified_stages: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Describe a captured candidate without retaining its audio buffers."""
    return {
        "scope": scope,
        "allowed_scope": list(allowed_scope),
        "target": deepcopy(target),
        "capture_identity": deepcopy(capture),
        "options": deepcopy(options),
        "verified_stages": list(verified_stages),
    }


def chain_settings(
    owner: Any,
    *,
    full_chain: bool = False,
    input_pre_filtered: bool = False,
    preset: Preset | None = None,
) -> dict[str, Any]:
    if full_chain:
        current = preset if preset is not None else owner._get_current_preset()
        if not isinstance(current, Preset):
            raise TypeError("current processing configuration is unavailable")
        preset_payload = current.to_dict()
        if hasattr(owner, "compressor_panel"):
            calibration = owner.compressor_panel.get_compressor_settings(
                include_calibration=True
            )
            preset_payload["compressor"]["noise_reference_reliability"] = calibration.get(
                "noise_reference_reliability", 0.0
            )
        return {
            **{
                name: preset_payload[name]
                for name in ("gate", "rnnoise", "deesser", "compressor", "limiter")
            },
            "full_chain": True,
            "input_pre_filtered": bool(input_pre_filtered),
            "input_cleanup_mode": owner.processor.get_input_cleanup_mode(),
            "processing_mode": owner._processing_mode(),
        }
    settings: dict[str, Any] = {}
    if owner is None:
        return settings
    if hasattr(owner, "deesser_panel"):
        try:
            settings["deesser"] = owner.deesser_panel.get_settings()
        except Exception:
            logger.debug(
                "Failed to collect de-esser settings for Auto-EQ simulation",
                exc_info=True,
            )
    if hasattr(owner, "compressor_panel"):
        try:
            settings["compressor"] = owner.compressor_panel.get_compressor_settings()
            settings["limiter"] = owner.compressor_panel.get_limiter_settings()
        except Exception:
            logger.debug(
                "Failed to collect dynamics settings for Auto-EQ simulation",
                exc_info=True,
            )
    return settings


def filtered_capture_for_analysis(
    audio_data: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Apply only the live fixed pre-filter used by the analysis paths."""
    audio = np.ascontiguousarray(np.asarray(audio_data, dtype=np.float32).reshape(-1))
    if audio.size == 0:
        return audio.copy()
    if not CORE_AVAILABLE:
        raise RuntimeError("Native input conditioning is unavailable")
    from ..mic_eq_core import simulate_auto_eq_chain

    result = simulate_auto_eq_chain(
        audio,
        float(sample_rate),
        [(1000.0, 0.0, 1.41)] * 10,
        {
            "full_chain": True,
            "processing_mode": "bypass",
            "limiter_enabled": False,
            "return_output_audio": True,
        },
    )
    output = np.asarray(result.get("output_audio"), dtype=np.float32).reshape(-1)
    if output.size != audio.size or not np.isfinite(output).all():
        raise ValueError("fixed pre-filter returned invalid audio")
    return np.ascontiguousarray(output, dtype=np.float32)


def format_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.0f}%"
    except (TypeError, ValueError):
        return "--"


def format_db(value: Any) -> str:
    try:
        return f"{float(value):.1f} dB"
    except (TypeError, ValueError):
        return "--"


def diagnostic_state(confidence: float) -> str:
    if confidence >= 0.72:
        return "ok"
    if confidence >= 0.45:
        return "warn"
    return "bad"
