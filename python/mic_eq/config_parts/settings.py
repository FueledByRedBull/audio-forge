"""Typed config dataclasses and public constants."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


EQ_FREQUENCIES = [80.0, 160.0, 320.0, 640.0, 1280.0, 2500.0, 5000.0, 8000.0, 12000.0, 16000.0]
AUTO_EQ_DEFAULT_Q = 4.33

ANALYSIS_MIN_PEAK_COUNT = 3
ANALYSIS_MIN_DYNAMIC_RANGE = 20
ANALYSIS_MIN_SNR = 12
ANALYSIS_MAX_SPECTRAL_FLATNESS = 0.8


@dataclass
class GateSettings:
    """Noise gate settings."""

    enabled: bool = True
    threshold_db: float = -40.0
    attack_ms: float = 10.0
    release_ms: float = 100.0
    gate_mode: int = 0
    vad_threshold: float = 0.4
    vad_hold_time_ms: float = 200.0
    vad_pre_gain: float = 1.0
    auto_threshold_enabled: bool = True
    gate_margin_db: float = 10.0


@dataclass
class EQSettings:
    """10-band parametric EQ settings."""

    enabled: bool = True
    band_freqs: list[float] = field(default_factory=lambda: list(EQ_FREQUENCIES))
    band_gains: list[float] = field(default_factory=lambda: [0.0] * 10)
    band_qs: list[float] = field(default_factory=lambda: [1.41] * 10)


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
    high_cut_hz: float = 9000.0
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
    sample_rate: int = 48000
    timestamp_utc: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LatencyCalibrationProfile":
        return cls(
            measured_round_trip_ms=float(data.get("measured_round_trip_ms", 0.0)),
            estimated_one_way_ms=float(data.get("estimated_one_way_ms", 0.0)),
            applied_compensation_ms=float(data.get("applied_compensation_ms", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            sample_rate=int(data.get("sample_rate", 48000)),
            timestamp_utc=str(data.get("timestamp_utc", "")),
        )


__all__ = [
    "ANALYSIS_MAX_SPECTRAL_FLATNESS",
    "ANALYSIS_MIN_DYNAMIC_RANGE",
    "ANALYSIS_MIN_PEAK_COUNT",
    "ANALYSIS_MIN_SNR",
    "AUTO_EQ_DEFAULT_Q",
    "CompressorSettings",
    "DeEsserSettings",
    "EQSettings",
    "EQ_FREQUENCIES",
    "GateSettings",
    "LatencyCalibrationProfile",
    "LimiterSettings",
    "RNNoiseSettings",
    "TargetCurve",
]
