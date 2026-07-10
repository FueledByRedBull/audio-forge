"""Preset schema, migration, and preset-file persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .settings import (
    CompressorSettings,
    DeEsserSettings,
    EQSettings,
    EQ_FREQUENCIES,
    GateSettings,
    LimiterSettings,
    RNNoiseSettings,
)
from .shared import (
    CURRENT_VERSION,
    PresetValidationError,
    _reject_json_constant,
    _version_tuple,
    get_preset_imports_dir,
    get_presets_dir,
)
from .validation import (
    VALIDATION_RANGES,
    _validate_bool,
    _validate_fixed_float_list,
    _validate_range,
)


@dataclass
class Preset:
    """Complete preset with all processing settings."""

    name: str = "Default"
    description: str = ""
    version: str = CURRENT_VERSION
    gate: GateSettings = field(default_factory=GateSettings)
    eq: EQSettings = field(default_factory=EQSettings)
    rnnoise: RNNoiseSettings = field(default_factory=RNNoiseSettings)
    deesser: DeEsserSettings = field(default_factory=DeEsserSettings)
    compressor: CompressorSettings = field(default_factory=CompressorSettings)
    limiter: LimiterSettings = field(default_factory=LimiterSettings)
    bypass: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "gate": asdict(self.gate),
            "eq": asdict(self.eq),
            "rnnoise": asdict(self.rnnoise),
            "deesser": asdict(self.deesser),
            "compressor": asdict(self.compressor),
            "limiter": asdict(self.limiter),
            "bypass": self.bypass,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Preset":
        try:
            version_tuple = _version_tuple(data.get("version", "1.0.0"))

            if version_tuple < _version_tuple("1.1.0"):
                if "rnnoise" in data:
                    data["rnnoise"].setdefault("strength", 1.0)
                else:
                    data["rnnoise"] = {"enabled": True, "strength": 1.0}
                data["version"] = "1.1.0"
                version_tuple = _version_tuple("1.1.0")

            if version_tuple < _version_tuple("1.2.0"):
                if "rnnoise" in data:
                    data["rnnoise"].setdefault("model", "rnnoise")
                else:
                    data["rnnoise"] = {"enabled": True, "strength": 1.0, "model": "rnnoise"}
                data["version"] = "1.2.0"
                version_tuple = _version_tuple("1.2.0")

            if version_tuple < _version_tuple("1.3.0"):
                if "compressor" in data:
                    data["compressor"].setdefault("auto_makeup_enabled", False)
                    data["compressor"].setdefault("target_lufs", -18.0)
                    data["compressor"].setdefault("sidechain_highpass_enabled", True)
                else:
                    data["compressor"] = {
                        "enabled": True,
                        "threshold_db": -20.0,
                        "ratio": 4.0,
                        "attack_ms": 10.0,
                        "release_ms": 200.0,
                        "makeup_gain_db": 0.0,
                        "adaptive_release": False,
                        "base_release_ms": 50.0,
                        "auto_makeup_enabled": False,
                        "target_lufs": -18.0,
                        "sidechain_highpass_enabled": True,
                    }
                data["version"] = "1.3.0"
                version_tuple = _version_tuple("1.3.0")

            if version_tuple < _version_tuple("1.4.0"):
                data["version"] = "1.4.0"
                version_tuple = _version_tuple("1.4.0")

            if version_tuple < _version_tuple("1.5.0"):
                data["version"] = "1.5.0"
                version_tuple = _version_tuple("1.5.0")

            if version_tuple < _version_tuple("1.6.0"):
                if "gate" in data:
                    data["gate"].setdefault("auto_threshold_enabled", True)
                    data["gate"].setdefault("gate_margin_db", 10.0)
                    if data["gate"].get("vad_threshold", 0.5) == 0.5:
                        data["gate"]["vad_threshold"] = 0.4
                else:
                    data["gate"] = {
                        "auto_threshold_enabled": True,
                        "gate_margin_db": 10.0,
                        "vad_threshold": 0.4,
                    }
                data["version"] = "1.6.0"
                version_tuple = _version_tuple("1.6.0")

            if version_tuple < _version_tuple("1.7.0"):
                if "deesser" in data:
                    data["deesser"].setdefault("enabled", False)
                    data["deesser"].setdefault("auto_enabled", True)
                    data["deesser"].setdefault("auto_amount", 0.5)
                    data["deesser"].setdefault("low_cut_hz", 4000.0)
                    data["deesser"].setdefault("high_cut_hz", 11000.0)
                    data["deesser"].setdefault("threshold_db", -28.0)
                    data["deesser"].setdefault("ratio", 4.0)
                    data["deesser"].setdefault("attack_ms", 2.0)
                    data["deesser"].setdefault("release_ms", 80.0)
                    data["deesser"].setdefault("max_reduction_db", 6.0)
                else:
                    data["deesser"] = asdict(DeEsserSettings())
                data["version"] = "1.7.0"
                version_tuple = _version_tuple("1.7.0")

            for version in (
                "1.7.1",
                "1.7.2",
                "1.7.3",
                "1.7.4",
                "1.8.0",
                "1.8.1",
                "1.8.2",
                "1.8.3",
                "1.8.4",
                "1.8.5",
                "1.8.6",
            ):
                if version_tuple < _version_tuple(version):
                    data["version"] = version
                    version_tuple = _version_tuple(version)

            gate_data = data.get("gate", {})
            gate_ranges = VALIDATION_RANGES["gate"]
            validated_gate = GateSettings(
                enabled=_validate_bool(gate_data.get("enabled", True), "enabled", "gate"),
                threshold_db=_validate_range(
                    gate_data.get("threshold_db", -40.0),
                    *gate_ranges["threshold_db"],
                    "threshold_db",
                    "gate",
                ),
                attack_ms=_validate_range(
                    gate_data.get("attack_ms", 10.0),
                    *gate_ranges["attack_ms"],
                    "attack_ms",
                    "gate",
                ),
                release_ms=_validate_range(
                    gate_data.get("release_ms", 100.0),
                    *gate_ranges["release_ms"],
                    "release_ms",
                    "gate",
                ),
                gate_mode=int(
                    _validate_range(
                        gate_data.get("gate_mode", 0),
                        *gate_ranges["gate_mode"],
                        "gate_mode",
                        "gate",
                    )
                ),
                vad_threshold=_validate_range(
                    gate_data.get("vad_threshold", 0.4),
                    *gate_ranges["vad_threshold"],
                    "vad_threshold",
                    "gate",
                ),
                vad_hold_time_ms=_validate_range(
                    gate_data.get("vad_hold_time_ms", 200.0),
                    *gate_ranges["vad_hold_time_ms"],
                    "vad_hold_time_ms",
                    "gate",
                ),
                vad_pre_gain=_validate_range(
                    gate_data.get("vad_pre_gain", 1.0),
                    *gate_ranges["vad_pre_gain"],
                    "vad_pre_gain",
                    "gate",
                ),
                auto_threshold_enabled=_validate_bool(
                    gate_data.get("auto_threshold_enabled", True),
                    "auto_threshold_enabled",
                    "gate",
                ),
                gate_margin_db=_validate_range(
                    gate_data.get("gate_margin_db", 10.0),
                    *gate_ranges["gate_margin_db"],
                    "gate_margin_db",
                    "gate",
                ),
            )

            eq_data = data.get("eq", {})
            eq_ranges = VALIDATION_RANGES["eq"]
            validated_eq = EQSettings(
                enabled=_validate_bool(eq_data.get("enabled", True), "enabled", "eq"),
                band_freqs=_validate_fixed_float_list(
                    eq_data.get("band_freqs", list(EQ_FREQUENCIES)),
                    10,
                    *eq_ranges["band_freq"],
                    "band_freqs",
                    "eq",
                ),
                band_gains=_validate_fixed_float_list(
                    eq_data.get("band_gains", [0.0] * 10),
                    10,
                    *eq_ranges["band_gain"],
                    "band_gains",
                    "eq",
                ),
                band_qs=_validate_fixed_float_list(
                    eq_data.get("band_qs", [1.41] * 10),
                    10,
                    *eq_ranges["band_q"],
                    "band_qs",
                    "eq",
                ),
            )

            comp_data = data.get("compressor", {})
            comp_ranges = VALIDATION_RANGES["compressor"]
            validated_comp = CompressorSettings(
                enabled=_validate_bool(comp_data.get("enabled", True), "enabled", "compressor"),
                threshold_db=_validate_range(
                    comp_data.get("threshold_db", -20.0),
                    *comp_ranges["threshold_db"],
                    "threshold_db",
                    "compressor",
                ),
                ratio=_validate_range(
                    comp_data.get("ratio", 4.0),
                    *comp_ranges["ratio"],
                    "ratio",
                    "compressor",
                ),
                attack_ms=_validate_range(
                    comp_data.get("attack_ms", 10.0),
                    *comp_ranges["attack_ms"],
                    "attack_ms",
                    "compressor",
                ),
                release_ms=_validate_range(
                    comp_data.get("release_ms", 200.0),
                    *comp_ranges["release_ms"],
                    "release_ms",
                    "compressor",
                ),
                makeup_gain_db=_validate_range(
                    comp_data.get("makeup_gain_db", 0.0),
                    *comp_ranges["makeup_gain_db"],
                    "makeup_gain_db",
                    "compressor",
                ),
                adaptive_release=_validate_bool(
                    comp_data.get("adaptive_release", False),
                    "adaptive_release",
                    "compressor",
                ),
                base_release_ms=_validate_range(
                    comp_data.get("base_release_ms", 50.0),
                    20.0,
                    200.0,
                    "base_release_ms",
                    "compressor",
                ),
                auto_makeup_enabled=_validate_bool(
                    comp_data.get("auto_makeup_enabled", False),
                    "auto_makeup_enabled",
                    "compressor",
                ),
                target_lufs=_validate_range(
                    comp_data.get("target_lufs", -18.0),
                    *comp_ranges["target_lufs"],
                    "target_lufs",
                    "compressor",
                ),
                sidechain_highpass_enabled=_validate_bool(
                    comp_data.get("sidechain_highpass_enabled", True),
                    "sidechain_highpass_enabled",
                    "compressor",
                ),
            )

            lim_data = data.get("limiter", {})
            lim_ranges = VALIDATION_RANGES["limiter"]
            validated_lim = LimiterSettings(
                enabled=_validate_bool(lim_data.get("enabled", True), "enabled", "limiter"),
                ceiling_db=_validate_range(
                    lim_data.get("ceiling_db", -0.5),
                    *lim_ranges["ceiling_db"],
                    "ceiling_db",
                    "limiter",
                ),
                release_ms=_validate_range(
                    lim_data.get("release_ms", 50.0),
                    *lim_ranges["release_ms"],
                    "release_ms",
                    "limiter",
                ),
                careful_output_enabled=_validate_bool(
                    lim_data.get("careful_output_enabled", True),
                    "careful_output_enabled",
                    "limiter",
                ),
            )

            rnnoise_data = data.get("rnnoise", {})
            rnnoise_ranges = VALIDATION_RANGES["rnnoise"]
            model = rnnoise_data.get("model", "rnnoise")
            valid_models = rnnoise_ranges.get("model", ["rnnoise", "deepfilter-ll", "deepfilter"])
            if model not in valid_models:
                model = "rnnoise"
            validated_rnnoise = RNNoiseSettings(
                enabled=_validate_bool(rnnoise_data.get("enabled", True), "enabled", "rnnoise"),
                strength=_validate_range(
                    rnnoise_data.get("strength", 1.0),
                    *rnnoise_ranges.get("strength", (0.0, 1.0)),
                    "strength",
                    "rnnoise",
                ),
                model=model,
            )

            deesser_data = data.get("deesser", {})
            deesser_ranges = VALIDATION_RANGES["deesser"]
            low_cut_hz = _validate_range(
                deesser_data.get("low_cut_hz", 4000.0),
                *deesser_ranges["low_cut_hz"],
                "low_cut_hz",
                "deesser",
            )
            high_cut_hz = _validate_range(
                deesser_data.get("high_cut_hz", 11000.0),
                *deesser_ranges["high_cut_hz"],
                "high_cut_hz",
                "deesser",
            )
            if high_cut_hz <= low_cut_hz + 200.0:
                high_cut_hz = min(16000.0, low_cut_hz + 200.0)
                low_cut_hz = min(low_cut_hz, high_cut_hz - 200.0)
            validated_deesser = DeEsserSettings(
                enabled=_validate_bool(deesser_data.get("enabled", False), "enabled", "deesser"),
                auto_enabled=_validate_bool(
                    deesser_data.get("auto_enabled", True),
                    "auto_enabled",
                    "deesser",
                ),
                auto_amount=_validate_range(
                    deesser_data.get("auto_amount", 0.5),
                    *deesser_ranges["auto_amount"],
                    "auto_amount",
                    "deesser",
                ),
                low_cut_hz=low_cut_hz,
                high_cut_hz=high_cut_hz,
                threshold_db=_validate_range(
                    deesser_data.get("threshold_db", -28.0),
                    *deesser_ranges["threshold_db"],
                    "threshold_db",
                    "deesser",
                ),
                ratio=_validate_range(
                    deesser_data.get("ratio", 4.0),
                    *deesser_ranges["ratio"],
                    "ratio",
                    "deesser",
                ),
                attack_ms=_validate_range(
                    deesser_data.get("attack_ms", 2.0),
                    *deesser_ranges["attack_ms"],
                    "attack_ms",
                    "deesser",
                ),
                release_ms=_validate_range(
                    deesser_data.get("release_ms", 80.0),
                    *deesser_ranges["release_ms"],
                    "release_ms",
                    "deesser",
                ),
                max_reduction_db=_validate_range(
                    deesser_data.get("max_reduction_db", 6.0),
                    *deesser_ranges["max_reduction_db"],
                    "max_reduction_db",
                    "deesser",
                ),
            )

            return cls(
                name=data.get("name", "Unnamed"),
                description=data.get("description", ""),
                version=data.get("version", CURRENT_VERSION),
                gate=validated_gate,
                eq=validated_eq,
                rnnoise=validated_rnnoise,
                deesser=validated_deesser,
                compressor=validated_comp,
                limiter=validated_lim,
                bypass=_validate_bool(data.get("bypass", False), "bypass", "preset"),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise PresetValidationError(f"Preset data is invalid or corrupted: {exc}") from exc


def save_preset(preset: Preset, filepath: Optional[Path] = None) -> Path:
    if filepath is None:
        safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in preset.name)
        safe_name = safe_name.strip().replace(" ", "_")
        if not safe_name:
            safe_name = "preset"
        filepath = get_presets_dir() / f"{safe_name}.json"

    filepath = Path(filepath)
    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(preset.to_dict(), handle, indent=2)
    return filepath


def load_preset(filepath: Path) -> Preset:
    requested_path = Path(filepath)
    if requested_path.suffix.lower() != ".json":
        raise PresetValidationError(
            f"Invalid preset file: '{requested_path.name}' - must be a .json file"
        )
    if not requested_path.exists():
        raise PresetValidationError(f"Preset file not found: '{requested_path.name}'")

    try:
        resolved_path = requested_path.resolve(strict=True)
    except OSError as exc:
        raise PresetValidationError(
            f"Invalid preset path: '{requested_path.name}' - {exc}"
        ) from exc

    if not resolved_path.is_file():
        raise PresetValidationError(
            f"Invalid preset path: '{requested_path.name}' - not a file"
        )

    allowed_roots = [
        get_presets_dir().resolve(),
        get_preset_imports_dir().resolve(),
    ]
    within_allowed_root = any(
        root == resolved_path or root in resolved_path.parents for root in allowed_roots
    )
    if not within_allowed_root:
        allowed_display = ", ".join(str(root) for root in allowed_roots)
        raise PresetValidationError(
            f"Invalid preset path: '{requested_path.name}' - "
            f"path must be inside allowed preset roots: {allowed_display}"
        )

    with open(resolved_path, "r", encoding="utf-8") as handle:
        data = json.load(handle, parse_constant=_reject_json_constant)
    return Preset.from_dict(data)


def list_presets() -> list[tuple[str, Path]]:
    presets_dir = get_presets_dir()
    presets: list[tuple[str, Path]] = []
    for filepath in presets_dir.glob("*.json"):
        try:
            preset = load_preset(filepath)
            presets.append((preset.name, filepath))
        except (json.JSONDecodeError, KeyError, PresetValidationError, TypeError, ValueError):
            continue
    return sorted(presets, key=lambda item: item[0].lower())


def generate_auto_eq_preset_name(target_curve: str) -> str:
    curve_display_names = {
        "broadcast": "Broadcast",
        "podcast": "Podcast",
        "streaming": "Streaming",
        "flat": "Flat",
    }
    curve_name = curve_display_names.get(target_curve.lower(), target_curve.title())
    return f"Auto-EQ {curve_name}"


__all__ = [
    "Preset",
    "generate_auto_eq_preset_name",
    "list_presets",
    "load_preset",
    "save_preset",
]
