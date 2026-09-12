"""Bounded, audio-free calibration evidence tied to its exact processing context."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from .shared import CURRENT_VERSION


def calibration_fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CalibrationResult:
    scope: str
    context_hash: str
    settings_hash: str
    created_at: str
    target_curve: str
    verified_stages: tuple[str, ...]
    noise_reference_reliability: float = 0.0
    version: int = 1
    app_version: str = CURRENT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: object) -> CalibrationResult | None:
        if (
            not isinstance(value, dict)
            or type(value.get("version")) is not int
            or value.get("version") != 1
        ):
            return None
        if value.get("scope") not in ("eq_only", "full_voice_setup"):
            return None
        for name in ("context_hash", "settings_hash"):
            digest = value.get(name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                return None
        stages = value.get("verified_stages")
        if not isinstance(stages, (list, tuple)) or not stages or len(stages) > 6:
            return None
        if any(
            not isinstance(stage, str)
            or stage
            not in {"eq", "gate", "suppression", "deesser", "compressor", "limiter"}
            for stage in stages
        ):
            return None
        curve = value.get("target_curve")
        if not isinstance(curve, str) or not 1 <= len(curve) <= 80:
            return None
        app_version = value.get("app_version")
        if not isinstance(app_version, str) or not 1 <= len(app_version) <= 40:
            return None
        try:
            stamp = value["created_at"]
            if not isinstance(stamp, str) or len(stamp) > 40:
                return None
            if datetime.fromisoformat(stamp).tzinfo is None:
                return None
            reliability = value.get("noise_reference_reliability", 0.0)
            if isinstance(reliability, bool) or not isinstance(
                reliability, (int, float)
            ):
                return None
            if not math.isfinite(reliability) or not 0.0 <= reliability <= 1.0:
                return None
        except (KeyError, TypeError, ValueError):
            return None
        return cls(
            value["scope"],
            value["context_hash"],
            value["settings_hash"],
            stamp,
            curve,
            tuple(stages),
            float(reliability),
            app_version=app_version,
        )

    @classmethod
    def capture(
        cls,
        scope: str,
        context: str,
        settings: dict,
        target_curve: str,
        verified_stages: tuple[str, ...],
        reliability: float = 0.0,
    ) -> CalibrationResult:
        result = cls(
            scope,
            calibration_fingerprint(context),
            calibration_fingerprint(settings),
            datetime.now(timezone.utc).isoformat(),
            target_curve,
            verified_stages,
            reliability,
        )
        if cls.from_value(result.to_dict()) is None:
            raise ValueError("Invalid calibration evidence")
        return result

    def matches(self, context: str | None, settings: dict) -> bool:
        return (
            self.app_version == CURRENT_VERSION
            and bool(context)
            and self.context_hash == calibration_fingerprint(context)
            and (self.settings_hash == calibration_fingerprint(settings))
        )


def parse_calibration_results(value: object) -> list[CalibrationResult]:
    if not isinstance(value, list):
        return []
    return [
        result
        for item in value[-16:]
        if (result := CalibrationResult.from_value(item)) is not None
    ]
