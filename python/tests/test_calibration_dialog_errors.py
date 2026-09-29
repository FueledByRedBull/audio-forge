from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, call

import pytest
from PyQt6.QtWidgets import QWidget

from mic_eq.ui import capture_session
from mic_eq.ui.calibration_dialog import CalibrationDialog
from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog


class _WarningLabel:
    def __init__(self):
        self.text = ""
        self.stylesheet = ""

    def setText(self, text):
        self.text = text

    def setStyleSheet(self, stylesheet):
        self.stylesheet = stylesheet


def test_recording_failure_remains_visible_after_reset():
    warning_label = _WarningLabel()

    def reset_recording_ui():
        warning_label.setText("Ready to record")
        warning_label.setStyleSheet("idle")

    dialog = SimpleNamespace(
        recording_timer=Mock(),
        warning_label=warning_label,
        _reset_recording_ui=reset_recording_ui,
    )

    CalibrationDialog._on_recording_failed(cast(Any, dialog), "device disconnected")

    assert warning_label.text == "❌ Recording failed: device disconnected"
    assert warning_label.stylesheet


@pytest.mark.parametrize(
    ("dialog_type", "capture_state"),
    [(CalibrationDialog, "recording"), (VoiceSetupDialog, "noise_recording")],
)
def test_stalled_capture_fails_and_releases_only_capture_owned_state(
    qapp, monkeypatch, dialog_type, capture_state
):
    processor = Mock()
    processor.is_recovery_suppressed.return_value = False
    processor.sample_rate.return_value = 48_000
    processor.recording_progress.return_value = 0.0
    processor.recording_level_db.return_value = -60.0
    processor.get_input_peak_db.return_value = -60.0
    processor.is_recording_complete.return_value = False
    owner = QWidget()
    setattr(owner, "processor", processor)
    setattr(owner, "user_muted", True)
    muted_reasons: set[str] = set()
    setattr(owner, "_calibration_context_key", lambda: "selected-route")

    def set_temporary_output_mute(muted: bool, reason: str) -> None:
        if muted:
            muted_reasons.add(reason)
        else:
            muted_reasons.discard(reason)

    setattr(owner, "set_temporary_output_mute", set_temporary_output_mute)
    dialog = dialog_type(parent=owner)
    if dialog_type is CalibrationDialog:
        dialog.recording_state = capture_state
    else:
        dialog.setup_state = capture_state

    try:
        dialog._begin_recording_capture()
        assert dialog._capture_session is not None
        started_at = dialog._capture_session.deadline.started_at
        assert started_at is not None
        monkeypatch.setattr(
            capture_session.time,
            "monotonic",
            lambda: started_at + 4.0,
        )

        dialog._poll_recording_progress()

        assert "No audio capture progress" in dialog.warning_label.text()
        assert dialog.recording_timer.isActive() is False
        assert dialog._capture_session is None
        assert not muted_reasons
        assert getattr(owner, "user_muted") is True
        assert processor.set_recovery_suppressed.call_args_list == [
            call(True),
            call(False),
        ]
    finally:
        dialog.reject()
        dialog.deleteLater()
        owner.close()
        qapp.processEvents()
