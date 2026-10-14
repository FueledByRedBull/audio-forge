"""Transient compressor calibration state stays out of persisted presets."""

from __future__ import annotations

import pytest
from unittest.mock import Mock

from mic_eq.config import CompressorSettings
from mic_eq.ui.compressor_state import CompressorState


def _state() -> tuple[CompressorState, Mock]:
    processor = Mock()
    return CompressorState(processor), processor


def test_calibration_reliability_round_trips_without_entering_presets(qapp):
    state, processor = _state()

    state.set_settings({"noise_reference_reliability": 0.7})
    snapshot = state.get_settings(include_calibration=True)
    state.set_settings({"noise_reference_reliability": 0.2})
    state.set_settings(snapshot)

    assert snapshot["noise_reference_reliability"] == pytest.approx(0.7)
    assert state.get_settings(include_calibration=True)[
        "noise_reference_reliability"
    ] == pytest.approx(0.7)
    calls = processor.set_compressor_noise_reference_reliability.call_args_list
    assert [call.args[0] for call in calls] == pytest.approx([0.7, 0.2, 0.7])
    assert "noise_reference_reliability" not in state.get_settings()
    CompressorSettings(**state.get_settings())

    state.set_settings({})
    calls = processor.set_compressor_noise_reference_reliability.call_args_list
    assert calls[-1].args[0] == 0.0
    assert state.get_settings(include_calibration=True)[
        "noise_reference_reliability"
    ] == 0.0


def test_calibration_reliability_matches_native_finite_unit_clamp(qapp):
    state, processor = _state()

    state.set_settings({"noise_reference_reliability": 2.0})
    calls = processor.set_compressor_noise_reference_reliability.call_args_list
    assert calls[-1].args[0] == 1.0
    assert state.get_settings(include_calibration=True)[
        "noise_reference_reliability"
    ] == 1.0

    state.set_settings({"noise_reference_reliability": float("nan")})
    calls = processor.set_compressor_noise_reference_reliability.call_args_list
    assert calls[-1].args[0] == 1.0
    assert state.get_settings(include_calibration=True)[
        "noise_reference_reliability"
    ] == 1.0
