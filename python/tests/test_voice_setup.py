"""Tests for auto voice setup analysis."""

from __future__ import annotations

import numpy as np

from mic_eq.analysis.voice_setup import analyze_voice_setup


def _make_noise(sample_rate: int, seconds: float = 2.0, amplitude: float = 0.0012) -> np.ndarray:
    rng = np.random.default_rng(1200)
    return (amplitude * rng.normal(size=int(sample_rate * seconds))).astype(np.float32)


def _make_voice(
    sample_rate: int,
    *,
    seconds: float = 10.0,
    sibilant: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(2400 if sibilant else 2401)
    t = np.arange(int(sample_rate * seconds), dtype=np.float64) / sample_rate
    gate = ((np.floor(t * 3.0) % 4.0) != 3.0).astype(np.float64)
    envelope = gate * (0.55 + 0.25 * np.sin(2.0 * np.pi * 2.1 * t) ** 2)
    voiced = (
        0.11 * np.sin(2.0 * np.pi * 140.0 * t)
        + 0.07 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.05 * np.sin(2.0 * np.pi * 440.0 * t)
        + (0.035 if sibilant else 0.004) * np.sin(2.0 * np.pi * 6500.0 * t)
    )
    noise = 0.0025 * rng.normal(size=t.size)
    return (envelope * voiced + noise).astype(np.float32)


def test_voice_setup_uses_vad_assisted_when_available():
    sample_rate = 48_000
    result = analyze_voice_setup(
        _make_noise(sample_rate),
        _make_voice(sample_rate),
        sample_rate,
        "streaming",
        vad_available=True,
    )

    assert result["gate_settings"]["gate_mode"] == 1
    assert result["gate_settings"]["auto_threshold_enabled"] is True
    assert result["compressor_settings"]["adaptive_release"] is True
    assert result["compressor_settings"]["enabled"] is True
    assert result["diagnostics"]["setup_confidence"] > 0.0


def test_voice_setup_falls_back_without_vad_and_can_enable_deesser():
    sample_rate = 48_000
    result = analyze_voice_setup(
        _make_noise(sample_rate),
        _make_voice(sample_rate, sibilant=True),
        sample_rate,
        "broadcast",
        vad_available=False,
    )

    assert result["gate_settings"]["gate_mode"] == 0
    assert result["gate_settings"]["auto_threshold_enabled"] is False
    assert result["deesser_settings"]["enabled"] is True
    assert result["deesser_settings"]["high_cut_hz"] > result["deesser_settings"]["low_cut_hz"]
