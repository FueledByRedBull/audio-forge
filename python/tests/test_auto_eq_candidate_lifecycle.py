"""Focused Auto-EQ candidate lifecycle tests."""

from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtWidgets import QMessageBox, QWidget

from mic_eq.ui.calibration_dialog import CalibrationDialog


class _SignalStub:
    def connect(self, _slot: Any) -> None:
        return None


class _AnalysisWorkerStub:
    def __init__(self, _audio, _sample_rate, target_preset, **kwargs):
        self.target_preset = target_preset
        self.target_mode = kwargs["target_mode"]
        self.smoothing_strength = kwargs["smoothing_strength"]
        self.step_progress = _SignalStub()
        self.result_ready = _SignalStub()
        self.failed = _SignalStub()
        self.finished = _SignalStub()

    def start(self) -> None:
        return None

    def isRunning(self) -> bool:
        return False

    def stop(self) -> None:
        return None

    def deleteLater(self) -> None:
        return None


class _ProcessorStub:
    def sample_rate(self) -> int:
        return 48_000

    def is_running(self) -> bool:
        return False

    def stop_raw_recording(self) -> None:
        return None

    def set_output_mute(self, _muted: bool) -> None:
        return None

    def set_recovery_suppressed(self, _suppressed: bool) -> None:
        return None


class _EqPanelStub:
    def __init__(self) -> None:
        self.enabled = False
        self.operations: list[str] = []
        self.applied_bands: list[tuple[float, float, float]] | None = None
        self.diagnostics: dict | None = None
        self.fail_apply = False

    def apply_auto_eq_results(self, bands, diagnostics=None) -> None:
        self.operations.append("apply")
        if self.fail_apply:
            raise RuntimeError("native apply failed")
        self.applied_bands = list(bands)
        self.diagnostics = diagnostics

    def set_settings(self, settings: dict) -> None:
        self.operations.append("enabled")
        self.enabled = bool(settings["enabled"])

    def set_auto_eq_diagnostics(self, diagnostics: dict | None) -> None:
        self.operations.append("diagnostics")
        self.diagnostics = diagnostics


class _Owner(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.processor = _ProcessorStub()
        self.eq_panel = _EqPanelStub()


def _candidate(gain: float = 1.0) -> dict:
    return {
        "band_freqs": [80.0, 160.0, 320.0, 640.0, 1280.0, 2500.0, 5000.0, 8000.0, 12000.0, 16000.0],
        "band_gains": [gain] * 10,
        "band_qs": [1.41] * 10,
        "apply_recommended": True,
    }


def _capture(dialog: CalibrationDialog, monkeypatch, curve: str) -> None:
    monkeypatch.setattr(
        "mic_eq.ui.calibration_dialog.AnalysisWorker", _AnalysisWorkerStub
    )
    dialog.curve_combo.setCurrentIndex(dialog.curve_combo.findData(curve))
    dialog.recording_state = "recording"
    dialog._on_recording_complete(np.ones(128, dtype=np.float32))


def test_retake_and_failed_next_analysis_cannot_apply_old_candidate(qapp, monkeypatch):
    owner = _Owner()
    dialog = CalibrationDialog(owner)
    try:
        _capture(dialog, monkeypatch, "broadcast")
        generation_a = dialog._analysis_generation
        assert dialog.recording_state == "analyzing"
        assert dialog.start_button.isEnabled() is False
        assert dialog.curve_combo.isEnabled() is False

        dialog._on_analysis_complete(_candidate(1.0), generation=generation_a)
        assert dialog.recording_state == "ready"
        assert dialog.eq_settings is not None
        assert dialog.curve_combo.isEnabled() is False

        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
        )
        dialog._on_retake_clicked()
        assert dialog.recording_state == "idle"
        assert dialog.eq_settings is None
        assert dialog._candidate_target_metadata is None

        _capture(dialog, monkeypatch, "podcast")
        generation_b = dialog._analysis_generation
        assert dialog.recording_state == "analyzing"
        assert dialog.eq_settings is None
        assert dialog.curve_combo.isEnabled() is False

        dialog._on_analysis_failed("B failed", generation=generation_b)
        assert dialog.recording_state == "ready"
        assert dialog.eq_settings is None
        assert dialog.start_button.text() == "Record Again"
        dialog._apply_eq_settings()
        assert owner.eq_panel.applied_bands is None
    finally:
        dialog.reject()
        owner.close()


def test_apply_uses_captured_target_and_enables_eq_after_bands(qapp, monkeypatch):
    owner = _Owner()
    dialog = CalibrationDialog(owner)
    applied_targets: list[str] = []
    dialog.auto_eq_applied.connect(applied_targets.append)
    try:
        _capture(dialog, monkeypatch, "broadcast")
        generation = dialog._analysis_generation
        dialog._on_analysis_complete(_candidate(2.0), generation=generation)

        dialog.curve_combo.setCurrentIndex(dialog.curve_combo.findData("podcast"))
        assert owner.eq_panel.enabled is False
        dialog._on_start_clicked()

        assert owner.eq_panel.operations == ["apply", "enabled", "diagnostics"]
        assert owner.eq_panel.enabled is True
        assert owner.eq_panel.applied_bands is not None
        assert applied_targets == ["broadcast"]
    finally:
        if not dialog._close_requested:
            dialog.reject()
        owner.close()


def test_failed_native_apply_does_not_enable_eq(qapp, monkeypatch):
    owner = _Owner()
    owner.eq_panel.fail_apply = True
    dialog = CalibrationDialog(owner)
    try:
        _capture(dialog, monkeypatch, "broadcast")
        generation = dialog._analysis_generation
        dialog._on_analysis_complete(_candidate(), generation=generation)
        monkeypatch.setattr(QMessageBox, "critical", lambda *_args, **_kwargs: None)

        dialog._on_start_clicked()

        assert owner.eq_panel.enabled is False
        assert owner.eq_panel.operations == ["apply"]
        assert dialog.recording_state == "ready"
    finally:
        if not dialog._close_requested:
            dialog.reject()
        owner.close()
