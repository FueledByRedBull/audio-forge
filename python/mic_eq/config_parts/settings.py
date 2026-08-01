"""Typed config dataclasses and public constants."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Protocol


EQ_FREQUENCIES = [
    80.0,
    160.0,
    320.0,
    640.0,
    1280.0,
    2500.0,
    5000.0,
    8000.0,
    12000.0,
    16000.0,
]
AUTO_EQ_DEFAULT_Q = 4.33
EQ_SCHEMA_VERSION = 2
EQ_BAND_COUNT = 10
EQ_FILTER_TYPES = frozenset(
    {
        "bell",
        "notch",
        "low_shelf",
        "high_shelf",
        "high_pass",
        "low_pass",
    }
)
EQ_BANDWIDTH_MODES = frozenset({"q", "octaves"})
# Schema v2 reserves the field but the retained runtime has one combined
# stage. Reject unimplemented ownership semantics instead of silently ignoring
# them; a future multi-stage design requires a new measured schema revision.
EQ_STAGES = frozenset({"combined"})
EQ_SLOPES_DB_PER_OCTAVE = frozenset({12, 24, 36, 48})
EQ_RUNTIME_SAMPLE_RATE = 48_000.0

ANALYSIS_MIN_PEAK_COUNT = 3
ANALYSIS_MIN_DYNAMIC_RANGE = 20
ANALYSIS_MIN_SNR = 12
ANALYSIS_MAX_SPECTRAL_FLATNESS = 0.8


class _SizedBandValues(Protocol):
    """Structural input accepted by legacy list and NumPy-array callers."""

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[Any]: ...


def q_from_bandwidth_octaves(
    frequency_hz: float,
    bandwidth_octaves: float,
    sample_rate: float = EQ_RUNTIME_SAMPLE_RATE,
) -> float:
    """Return the exact RBJ digital-Q equivalent of an octave bandwidth."""
    values = (frequency_hz, bandwidth_octaves, sample_rate)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("frequency, bandwidth, and sample rate must be finite")
    frequency = float(frequency_hz)
    bandwidth = float(bandwidth_octaves)
    rate = float(sample_rate)
    if frequency <= 0.0 or bandwidth <= 0.0 or rate <= 0.0:
        raise ValueError("frequency, bandwidth, and sample rate must be positive")
    if frequency >= rate / 2.0:
        raise ValueError("frequency must be below Nyquist")
    omega = 2.0 * math.pi * frequency / rate
    argument = (
        math.log(2.0)
        * 0.5
        * bandwidth
        * omega
        / math.sin(omega)
    )
    if argument > 700.0:
        return 0.0
    return 1.0 / (2.0 * math.sinh(argument))


@dataclass
class GateSettings:
    """Noise gate settings."""

    enabled: bool = True
    threshold_db: float = -40.0
    attack_ms: float = 10.0
    release_ms: float = 100.0
    gate_mode: int = 0
    vad_threshold: float = 0.48
    vad_hold_time_ms: float = 200.0
    vad_pre_gain: float = 1.0
    auto_threshold_enabled: bool = True
    gate_margin_db: float = 10.0


def _finite_float(
    value: object,
    *,
    name: str,
    low: float,
    high: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    if not low <= parsed <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return parsed


def _strict_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


@dataclass(frozen=True, slots=True)
class EQBandSettings:
    """One immutable serialized EQ band."""

    filter_type: str
    frequency_hz: float
    gain_db: float
    q: float
    bandwidth_mode: str = "q"
    bandwidth_octaves: float | None = None
    slope_db_per_octave: int = 12
    stage: str = "combined"
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.filter_type not in EQ_FILTER_TYPES:
            raise ValueError(f"unsupported EQ filter type: {self.filter_type}")
        _finite_float(
            self.frequency_hz,
            name="frequency_hz",
            low=20.0,
            high=20_000.0,
        )
        _finite_float(self.gain_db, name="gain_db", low=-12.0, high=12.0)
        _finite_float(self.q, name="q", low=0.1, high=10.0)
        if self.bandwidth_mode not in EQ_BANDWIDTH_MODES:
            raise ValueError(
                f"unsupported EQ bandwidth mode: {self.bandwidth_mode}"
            )
        if self.bandwidth_octaves is not None:
            _finite_float(
                self.bandwidth_octaves,
                name="bandwidth_octaves",
                low=0.1,
                high=8.0,
            )
        if (
            self.bandwidth_mode == "octaves"
            and self.bandwidth_octaves is None
        ):
            raise ValueError(
                "bandwidth_octaves is required when bandwidth_mode is octaves"
            )
        if self.bandwidth_mode == "q" and self.bandwidth_octaves is not None:
            raise ValueError(
                "bandwidth_octaves must be null when bandwidth_mode is q"
            )
        if self.bandwidth_mode == "octaves":
            if self.filter_type not in {"bell", "notch"}:
                raise ValueError(
                    "octave bandwidth is supported only for bell and notch"
                )
            bandwidth_octaves = self.bandwidth_octaves
            if bandwidth_octaves is None:
                raise ValueError(
                    "bandwidth_octaves is required when bandwidth_mode is octaves"
                )
            equivalent_q = q_from_bandwidth_octaves(
                self.frequency_hz,
                bandwidth_octaves,
            )
            if not 0.1 <= equivalent_q <= 10.0:
                raise ValueError(
                    "octave bandwidth resolves outside the supported Q range"
                )
            if not math.isclose(
                float(self.q),
                equivalent_q,
                rel_tol=1.0e-6,
                abs_tol=1.0e-8,
            ):
                raise ValueError(
                    "q must match the octave-bandwidth equivalent at 48 kHz"
                )
        if self.slope_db_per_octave not in EQ_SLOPES_DB_PER_OCTAVE:
            raise ValueError(
                f"unsupported EQ slope: {self.slope_db_per_octave}"
            )
        if self.stage not in EQ_STAGES:
            raise ValueError(f"unsupported EQ stage: {self.stage}")
        _strict_bool(self.enabled, name="enabled")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: object, *, index: int) -> "EQBandSettings":
        if not isinstance(data, Mapping):
            raise ValueError(f"eq.bands.{index} must be an object")
        allowed = {
            "filter_type",
            "frequency_hz",
            "gain_db",
            "q",
            "bandwidth_mode",
            "bandwidth_octaves",
            "slope_db_per_octave",
            "stage",
            "enabled",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(
                f"eq.bands.{index} contains unknown fields: "
                f"{', '.join(sorted(str(key) for key in unknown))}"
            )
        missing = allowed - set(data)
        if missing:
            raise ValueError(
                f"eq.bands.{index} is missing fields: "
                f"{', '.join(sorted(missing))}"
            )
        filter_type = data["filter_type"]
        bandwidth_mode = data["bandwidth_mode"]
        stage = data["stage"]
        slope = data["slope_db_per_octave"]
        if not isinstance(filter_type, str):
            raise ValueError(f"eq.bands.{index}.filter_type must be a string")
        if not isinstance(bandwidth_mode, str):
            raise ValueError(
                f"eq.bands.{index}.bandwidth_mode must be a string"
            )
        if not isinstance(stage, str):
            raise ValueError(f"eq.bands.{index}.stage must be a string")
        if isinstance(slope, bool) or not isinstance(slope, int):
            raise ValueError(
                f"eq.bands.{index}.slope_db_per_octave must be an integer"
            )
        bandwidth = data["bandwidth_octaves"]
        return cls(
            filter_type=filter_type,
            frequency_hz=_finite_float(
                data["frequency_hz"],
                name=f"eq.bands.{index}.frequency_hz",
                low=20.0,
                high=20_000.0,
            ),
            gain_db=_finite_float(
                data["gain_db"],
                name=f"eq.bands.{index}.gain_db",
                low=-12.0,
                high=12.0,
            ),
            q=_finite_float(
                data["q"],
                name=f"eq.bands.{index}.q",
                low=0.1,
                high=10.0,
            ),
            bandwidth_mode=bandwidth_mode,
            bandwidth_octaves=(
                None
                if bandwidth is None
                else _finite_float(
                    bandwidth,
                    name=f"eq.bands.{index}.bandwidth_octaves",
                    low=0.1,
                    high=8.0,
                )
            ),
            slope_db_per_octave=slope,
            stage=stage,
            enabled=_strict_bool(
                data["enabled"],
                name=f"eq.bands.{index}.enabled",
            ),
        )


def _default_filter_type(index: int) -> str:
    if index == 0:
        return "low_shelf"
    if index == EQ_BAND_COUNT - 1:
        return "high_shelf"
    return "bell"


def _legacy_bands(
    frequencies: _SizedBandValues,
    gains: _SizedBandValues,
    qs: _SizedBandValues,
) -> tuple[EQBandSettings, ...]:
    if not (
        len(frequencies) == len(gains) == len(qs) == EQ_BAND_COUNT
    ):
        raise ValueError(f"EQ settings must contain {EQ_BAND_COUNT} bands")
    return tuple(
        EQBandSettings(
            filter_type=_default_filter_type(index),
            frequency_hz=_finite_float(
                frequency,
                name=f"band_freqs.{index}",
                low=20.0,
                high=20_000.0,
            ),
            gain_db=_finite_float(
                gain,
                name=f"band_gains.{index}",
                low=-12.0,
                high=12.0,
            ),
            q=_finite_float(
                q,
                name=f"band_qs.{index}",
                low=0.1,
                high=10.0,
            ),
        )
        for index, (frequency, gain, q) in enumerate(
            zip(frequencies, gains, qs)
        )
    )


@dataclass(init=False)
class EQSettings:
    """Versioned EQ schema with immutable bands and legacy list views."""

    enabled: bool
    schema_version: int
    bands: tuple[EQBandSettings, ...]

    def __init__(
        self,
        enabled: bool = True,
        band_freqs: _SizedBandValues | None = None,
        band_gains: _SizedBandValues | None = None,
        band_qs: _SizedBandValues | None = None,
        *,
        schema_version: int = EQ_SCHEMA_VERSION,
        bands: Sequence[EQBandSettings] | None = None,
    ) -> None:
        self.enabled = _strict_bool(enabled, name="eq.enabled")
        if schema_version != EQ_SCHEMA_VERSION:
            raise ValueError(f"unsupported EQ schema version: {schema_version}")
        self.schema_version = schema_version
        if bands is not None:
            if any(
                value is not None
                for value in (band_freqs, band_gains, band_qs)
            ):
                raise ValueError(
                    "EQ bands cannot be combined with legacy band arrays"
                )
            parsed_bands = tuple(bands)
            if len(parsed_bands) != EQ_BAND_COUNT or not all(
                isinstance(band, EQBandSettings) for band in parsed_bands
            ):
                raise ValueError(
                    f"EQ settings must contain {EQ_BAND_COUNT} typed bands"
                )
            self.bands = parsed_bands
        else:
            self.bands = _legacy_bands(
                EQ_FREQUENCIES if band_freqs is None else band_freqs,
                [0.0] * EQ_BAND_COUNT if band_gains is None else band_gains,
                [1.41] * EQ_BAND_COUNT if band_qs is None else band_qs,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "bands": [band.to_dict() for band in self.bands],
        }

    @classmethod
    def from_dict(cls, data: object) -> "EQSettings":
        if not isinstance(data, Mapping):
            raise ValueError("eq must be an object")
        new_fields = {"schema_version", "enabled", "bands"}
        legacy_fields = {"enabled", "band_freqs", "band_gains", "band_qs"}
        fields = set(data)
        if "bands" in data or "schema_version" in data:
            unknown = fields - new_fields
            if unknown:
                raise ValueError(
                    "eq contains unknown fields: "
                    + ", ".join(sorted(str(key) for key in unknown))
                )
            if fields != new_fields:
                missing = new_fields - fields
                raise ValueError(
                    "eq is missing fields: "
                    + ", ".join(sorted(str(key) for key in missing))
                )
            schema_version = data["schema_version"]
            if isinstance(schema_version, bool) or not isinstance(
                schema_version,
                int,
            ):
                raise ValueError("eq.schema_version must be an integer")
            raw_bands = data["bands"]
            if not isinstance(raw_bands, list):
                raise ValueError("eq.bands must be an array")
            return cls(
                enabled=_strict_bool(data["enabled"], name="eq.enabled"),
                schema_version=schema_version,
                bands=[
                    EQBandSettings.from_dict(band, index=index)
                    for index, band in enumerate(raw_bands)
                ],
            )

        unknown = fields - legacy_fields
        if unknown:
            raise ValueError(
                "legacy eq contains unknown fields: "
                + ", ".join(sorted(str(key) for key in unknown))
            )
        return cls(
            enabled=_strict_bool(
                data.get("enabled", True),
                name="eq.enabled",
            ),
            band_freqs=_validated_legacy_array(
                data.get("band_freqs", EQ_FREQUENCIES),
                name="eq.band_freqs",
                low=20.0,
                high=20_000.0,
            ),
            band_gains=_validated_legacy_array(
                data.get("band_gains", [0.0] * EQ_BAND_COUNT),
                name="eq.band_gains",
                low=-12.0,
                high=12.0,
            ),
            band_qs=_validated_legacy_array(
                data.get("band_qs", [1.41] * EQ_BAND_COUNT),
                name="eq.band_qs",
                low=0.1,
                high=10.0,
            ),
        )

    def _replace_values(self, field_name: str, values: Sequence[object]) -> None:
        if len(values) != EQ_BAND_COUNT:
            raise ValueError(f"{field_name} must contain {EQ_BAND_COUNT} values")
        replacements: list[EQBandSettings] = []
        for index, (band, value) in enumerate(zip(self.bands, values)):
            if field_name == "frequency_hz":
                parsed: object = _finite_float(
                    value,
                    name=f"band_freqs.{index}",
                    low=20.0,
                    high=20_000.0,
                )
            elif field_name == "gain_db":
                parsed = _finite_float(
                    value,
                    name=f"band_gains.{index}",
                    low=-12.0,
                    high=12.0,
                )
            elif field_name == "q":
                parsed = _finite_float(
                    value,
                    name=f"band_qs.{index}",
                    low=0.1,
                    high=10.0,
                )
            else:
                parsed = value
            replacements.append(replace(band, **{field_name: parsed}))
        self.bands = tuple(replacements)

    @property
    def band_freqs(self) -> list[float]:
        return [band.frequency_hz for band in self.bands]

    @band_freqs.setter
    def band_freqs(self, values: Sequence[float]) -> None:
        self._replace_values("frequency_hz", values)

    @property
    def band_gains(self) -> list[float]:
        return [band.gain_db for band in self.bands]

    @band_gains.setter
    def band_gains(self, values: Sequence[float]) -> None:
        self._replace_values("gain_db", values)

    @property
    def band_qs(self) -> list[float]:
        return [band.q for band in self.bands]

    @band_qs.setter
    def band_qs(self, values: Sequence[float]) -> None:
        self._replace_values("q", values)


def _validated_legacy_array(
    value: object,
    *,
    name: str,
    low: float,
    high: float,
) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != EQ_BAND_COUNT:
        raise ValueError(f"{name} must contain {EQ_BAND_COUNT} values")
    return [
        _finite_float(
            item,
            name=f"{name}.{index}",
            low=low,
            high=high,
        )
        for index, item in enumerate(value)
    ]


@dataclass
class RNNoiseSettings:
    """Noise suppression settings (RNNoise or DeepFilterNet)."""

    enabled: bool = True
    strength: float = 1.0
    model: str = "rnnoise"


@dataclass
class DeEsserSettings:
    """De-esser settings."""

    enabled: bool = False
    auto_enabled: bool = True
    auto_amount: float = 0.5
    low_cut_hz: float = 4000.0
    high_cut_hz: float = 11000.0
    threshold_db: float = -28.0
    ratio: float = 4.0
    attack_ms: float = 2.0
    release_ms: float = 80.0
    max_reduction_db: float = 6.0


@dataclass
class CompressorSettings:
    """Compressor settings."""

    enabled: bool = True
    threshold_db: float = -20.0
    ratio: float = 4.0
    attack_ms: float = 10.0
    release_ms: float = 200.0
    makeup_gain_db: float = 0.0
    adaptive_release: bool = False
    base_release_ms: float = 50.0
    auto_makeup_enabled: bool = False
    target_lufs: float = -18.0
    sidechain_highpass_enabled: bool = True


@dataclass
class LimiterSettings:
    """Hard limiter settings."""

    enabled: bool = True
    ceiling_db: float = -0.5
    release_ms: float = 50.0
    careful_output_enabled: bool = True


@dataclass
class TargetCurve:
    """Target frequency response curve for Auto-EQ calibration."""

    name: str
    description: str
    band_targets: list[float]


@dataclass
class LatencyCalibrationProfile:
    """Measured latency calibration result for one input/output pair."""

    measured_round_trip_ms: float
    estimated_one_way_ms: float
    applied_compensation_ms: float
    confidence: float
    agreement_ms: float = 0.0
    ambiguity_score: float = 0.0
    repetition_count: int = 0
    sample_rate: int = 48000
    timestamp_utc: str = ""
    route_latency_ms: float = 0.0
    directional_latency_ms: float | None = None
    route_kind: str = "output_to_input"
    compensation_basis: str = "measured_output_to_input_route"
    engine_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    engine_config_signature: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LatencyCalibrationProfile":
        if not isinstance(data, Mapping):
            raise ValueError("latency profile must be an object")

        def finite_latency(name: str, default: float = 0.0) -> float:
            return _finite_float(
                data.get(name, default),
                name=f"latency profile {name}",
                low=0.0,
                high=60_000.0,
            )

        def bounded_integer(
            name: str,
            default: int,
            low: int,
            high: int,
        ) -> int:
            value = data.get(name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ValueError(f"latency profile {name} must be an integer")
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"latency profile {name} must be an integer"
                ) from error
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(f"latency profile {name} must be an integer")
            parsed = int(numeric)
            if not low <= parsed <= high:
                raise ValueError(
                    f"latency profile {name} must be between {low} and {high}"
                )
            return parsed

        measured_route = finite_latency("measured_round_trip_ms")
        applied_compensation = finite_latency("applied_compensation_ms")
        # Profiles saved before route-aware calibration only had the legacy
        # round-trip field. Migrate them to the measured route delay instead
        # of preserving the unjustified half-latency compensation.
        route_latency = data.get("route_latency_ms")
        if route_latency is None:
            route_latency = (
                measured_route
                if measured_route > 0.0
                else applied_compensation
            )
        else:
            route_latency = finite_latency("route_latency_ms")
            if route_latency <= 0.0:
                route_latency = (
                    measured_route
                    if measured_route > 0.0
                    else applied_compensation
                )
        directional = data.get("directional_latency_ms")
        directional_latency = (
            finite_latency("directional_latency_ms")
            if directional is not None
            else None
        )
        engine_latency = finite_latency("engine_latency_ms")
        total_latency = float(route_latency) + engine_latency
        route_kind = data.get("route_kind", "output_to_input")
        compensation_basis = data.get(
            "compensation_basis", "measured_output_to_input_route"
        )
        if route_kind != "output_to_input":
            raise ValueError("latency profile route_kind is unsupported")
        if compensation_basis != "measured_output_to_input_route":
            raise ValueError("latency profile compensation_basis is unsupported")
        timestamp = data.get("timestamp_utc", "")
        signature = data.get("engine_config_signature", "")
        if not isinstance(timestamp, str) or len(timestamp) > 128:
            raise ValueError("latency profile timestamp_utc must be bounded text")
        if not isinstance(signature, str) or len(signature) > 4096:
            raise ValueError(
                "latency profile engine_config_signature must be bounded text"
            )
        return cls(
            measured_round_trip_ms=measured_route,
            estimated_one_way_ms=finite_latency("estimated_one_way_ms"),
            applied_compensation_ms=applied_compensation,
            confidence=_finite_float(
                data.get("confidence", 0.0),
                name="latency profile confidence",
                low=0.0,
                high=1.0,
            ),
            agreement_ms=finite_latency("agreement_ms"),
            ambiguity_score=_finite_float(
                data.get("ambiguity_score", 0.0),
                name="latency profile ambiguity_score",
                low=0.0,
                high=1.0,
            ),
            repetition_count=bounded_integer("repetition_count", 0, 0, 10_000),
            sample_rate=bounded_integer("sample_rate", 48_000, 8_000, 768_000),
            timestamp_utc=timestamp,
            route_latency_ms=float(route_latency),
            directional_latency_ms=directional_latency,
            route_kind=route_kind,
            compensation_basis=compensation_basis,
            engine_latency_ms=engine_latency,
            total_latency_ms=total_latency,
            engine_config_signature=signature,
        )


__all__ = [
    "ANALYSIS_MAX_SPECTRAL_FLATNESS",
    "ANALYSIS_MIN_DYNAMIC_RANGE",
    "ANALYSIS_MIN_PEAK_COUNT",
    "ANALYSIS_MIN_SNR",
    "AUTO_EQ_DEFAULT_Q",
    "EQ_BANDWIDTH_MODES",
    "EQ_BAND_COUNT",
    "EQ_FILTER_TYPES",
    "EQ_RUNTIME_SAMPLE_RATE",
    "EQ_SCHEMA_VERSION",
    "EQ_SLOPES_DB_PER_OCTAVE",
    "EQ_STAGES",
    "EQBandSettings",
    "CompressorSettings",
    "DeEsserSettings",
    "EQSettings",
    "EQ_FREQUENCIES",
    "q_from_bandwidth_octaves",
    "GateSettings",
    "LatencyCalibrationProfile",
    "LimiterSettings",
    "RNNoiseSettings",
    "TargetCurve",
]
