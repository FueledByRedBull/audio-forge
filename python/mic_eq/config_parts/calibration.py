"""Bounded, audio-free calibration evidence tied to its exact processing context."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .shared import CURRENT_VERSION

CALIBRATION_SCHEMA_VERSION = 2
CALIBRATION_COMPATIBILITY = "capture-eq-v1"


def calibration_fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def calibration_settings_parts(
    settings: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split correction identity from settings that affect output verification.

    Schema-3 EQ records have independent correction and tone layers. Older
    combined EQ records stay conservative: the whole EQ remains part of both
    identities because the two stages cannot be separated truthfully.
    """
    if not isinstance(settings, Mapping):
        raise TypeError("calibration settings must be an object")
    payload = dict(settings)
    raw_eq = payload.get("eq")
    if not isinstance(raw_eq, Mapping):
        raise ValueError("calibration settings must contain an EQ object")
    eq = dict(raw_eq)
    layers = eq.get("layers")
    if (
        isinstance(layers, Mapping)
        and set(layers) == {"correction", "tone"}
    ):
        common_eq = {
            "schema_version": eq.get("schema_version"),
            "enabled": eq.get("enabled"),
        }
        correction_eq = {
            **common_eq,
            "bands": layers["correction"],
        }
        downstream_eq = {
            **common_eq,
            "bands": layers["tone"],
        }
        correction = {"eq": correction_eq}
        verification = dict(payload)
        verification["eq"] = downstream_eq
        return correction, verification

    # A combined/legacy EQ has no honest correction-vs-tone boundary.
    return {"eq": eq}, payload


@dataclass(frozen=True)
class CalibrationResult:
    scope: str
    context_hash: str
    settings_hash: str
    created_at: str
    target_curve: str
    verified_stages: tuple[str, ...]
    noise_reference_reliability: float = 0.0
    version: int = CALIBRATION_SCHEMA_VERSION
    app_version: str = CURRENT_VERSION
    correction_hash: str | None = None
    verification_hash: str | None = None
    compatibility: str | None = CALIBRATION_COMPATIBILITY

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.version == 1:
            payload.pop("correction_hash", None)
            payload.pop("verification_hash", None)
            payload.pop("compatibility", None)
        return payload

    @classmethod
    def from_value(cls, value: object) -> CalibrationResult | None:
        if (
            not isinstance(value, dict)
            or type(value.get("version")) is not int
            or value.get("version") not in (1, CALIBRATION_SCHEMA_VERSION)
        ):
            return None
        if value.get("scope") not in ("eq_only", "full_voice_setup"):
            return None
        for name in ("context_hash", "settings_hash"):
            if not _is_digest(value.get(name)):
                return None
        version = value["version"]
        correction_hash = value.get("correction_hash")
        verification_hash = value.get("verification_hash")
        compatibility = value.get("compatibility")
        if version == CALIBRATION_SCHEMA_VERSION:
            if not _is_digest(correction_hash) or not _is_digest(verification_hash):
                return None
            if (
                not isinstance(compatibility, str)
                or not 1 <= len(compatibility) <= 80
            ):
                return None
        else:
            correction_hash = None
            verification_hash = None
            compatibility = None
        stages = value.get("verified_stages")
        if not isinstance(stages, (list, tuple)) or not stages or len(stages) > 7:
            return None
        if any(
            not isinstance(stage, str)
            or stage
            not in {
                "input_cleanup",
                "eq",
                "gate",
                "suppression",
                "deesser",
                "compressor",
                "limiter",
            }
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
            scope=value["scope"],
            context_hash=value["context_hash"],
            settings_hash=value["settings_hash"],
            created_at=stamp,
            target_curve=curve,
            verified_stages=tuple(stages),
            noise_reference_reliability=float(reliability),
            version=version,
            app_version=app_version,
            correction_hash=correction_hash,
            verification_hash=verification_hash,
            compatibility=compatibility,
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
        *,
        correction_settings: Mapping[str, Any] | None = None,
        verification_settings: Mapping[str, Any] | None = None,
    ) -> CalibrationResult:
        correction = settings if correction_settings is None else correction_settings
        verification = (
            settings if verification_settings is None else verification_settings
        )
        result = cls(
            scope=scope,
            context_hash=calibration_fingerprint(context),
            settings_hash=calibration_fingerprint(settings),
            created_at=datetime.now(timezone.utc).isoformat(),
            target_curve=target_curve,
            verified_stages=verified_stages,
            noise_reference_reliability=reliability,
            version=CALIBRATION_SCHEMA_VERSION,
            app_version=CURRENT_VERSION,
            correction_hash=calibration_fingerprint(correction),
            verification_hash=calibration_fingerprint(verification),
            compatibility=CALIBRATION_COMPATIBILITY,
        )
        if cls.from_value(result.to_dict()) is None:
            raise ValueError("Invalid calibration evidence")
        return result

    def matches(self, context: str | None, settings: dict) -> bool:
        """Compatibility match for callers that only have one settings hash."""
        return (
            self.matches_capture(context)
            and self.matches_correction(settings)
        )

    def matches_capture(self, context: str | None) -> bool:
        return (
            bool(context)
            and self.context_hash == calibration_fingerprint(context)
            and (
                self.version == CALIBRATION_SCHEMA_VERSION
                or self.app_version == CURRENT_VERSION
            )
            and (
                self.version != CALIBRATION_SCHEMA_VERSION
                or self.compatibility == CALIBRATION_COMPATIBILITY
            )
        )

    def matches_correction(self, settings: Mapping[str, Any]) -> bool:
        digest = (
            self.settings_hash
            if self.version == 1
            else self.correction_hash
        )
        return digest == calibration_fingerprint(settings)

    def matches_verification(self, settings: Mapping[str, Any]) -> bool:
        digest = (
            self.settings_hash
            if self.version == 1
            else self.verification_hash
        )
        return (
            (
                self.app_version == CURRENT_VERSION
                if self.version == 1
                else self.compatibility == CALIBRATION_COMPATIBILITY
            )
            and digest == calibration_fingerprint(settings)
        )

    def matches_capture_and_correction(
        self,
        context: str | None,
        correction_settings: Mapping[str, Any],
    ) -> bool:
        return self.matches_capture(context) and self.matches_correction(
            correction_settings
        )


def parse_calibration_results(value: object) -> list[CalibrationResult]:
    if not isinstance(value, list):
        return []
    return [
        result
        for item in value[-16:]
        if (result := CalibrationResult.from_value(item)) is not None
    ]
