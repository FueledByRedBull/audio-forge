"""Transient compressor calibration state stays out of persisted presets."""

from __future__ import annotations

import pytest
from unittest.mock import Mock

from mic_eq.config import CompressorSettings
from mic_eq.ui.compressor_panel import CompressorPanel


def _panel(qapp) -> tuple[CompressorPanel, Mock]:
    processor = Mock()
    panel = CompressorPanel(processor)
    qapp.processEvents()
    processor.set_compressor_noise_reference_reliability.reset_mock()
    return panel, processor


def _close_panel(panel: CompressorPanel, qapp) -> None:
    panel.close()
    panel.deleteLater()
    qapp.processEvents()


def test_calibration_reliability_round_trips_without_entering_presets(qapp):
    panel, processor = _panel(qapp)
    try:
        panel.set_compressor_settings({"noise_reference_reliability": 0.7})
        snapshot = panel.get_compressor_settings(include_calibration=True)
        panel.set_compressor_settings({"noise_reference_reliability": 0.2})
        panel.set_compressor_settings(snapshot)

        assert snapshot["noise_reference_reliability"] == pytest.approx(0.7)
        assert panel.get_compressor_settings(include_calibration=True)[
            "noise_reference_reliability"
        ] == pytest.approx(0.7)
        calls = processor.set_compressor_noise_reference_reliability.call_args_list
        assert [call.args[0] for call in calls] == pytest.approx([0.7, 0.2, 0.7])
        assert "noise_reference_reliability" not in panel.get_compressor_settings()
        CompressorSettings(**panel.get_compressor_settings())

        panel.set_compressor_settings({})
        calls = processor.set_compressor_noise_reference_reliability.call_args_list
        assert calls[-1].args[0] == 0.0
        assert panel.get_compressor_settings(include_calibration=True)[
            "noise_reference_reliability"
        ] == 0.0
    finally:
        _close_panel(panel, qapp)


def test_calibration_reliability_matches_native_finite_unit_clamp(qapp):
    panel, processor = _panel(qapp)
    try:
        panel.set_compressor_settings({"noise_reference_reliability": 2.0})
        calls = processor.set_compressor_noise_reference_reliability.call_args_list
        assert calls[-1].args[0] == 1.0
        assert panel.get_compressor_settings(include_calibration=True)[
            "noise_reference_reliability"
        ] == 1.0

        panel.set_compressor_settings({"noise_reference_reliability": float("nan")})
        calls = processor.set_compressor_noise_reference_reliability.call_args_list
        assert calls[-1].args[0] == 1.0
        assert panel.get_compressor_settings(include_calibration=True)[
            "noise_reference_reliability"
        ] == 1.0
    finally:
        _close_panel(panel, qapp)
