"""Checks for the two-clip Raw/Processed dialog and its capture entry point."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtWidgets import QWidget

from mic_eq.analysis import listening_comparison as comparison
from mic_eq.config import DeviceIdentity
from mic_eq.ui import listening_comparison_dialog as module
from mic_eq.ui.accessibility import audit_widget_tree
from mic_eq.ui.listening_comparison_dialog import (
    ListeningComparisonDialog,
    RawProcessedDialog,
    open_test_my_sound,
)


def _settings() -> dict:
    return {
        "enabled": True,
        "band_freqs": [100.0, 200.0, 400.0, 800.0, 1200.0, 2400.0, 4800.0, 8000.0, 12000.0, 16000.0],
        "band_gains": [0.0] * 10,
        "band_qs": [1.41] * 10,
    }


@pytest.fixture
def dialog(qapp, monkeypatch):
    # No render thread and no playback device: nothing here opens real audio.
    monkeypatch.setattr(QMediaDevices, "audioOutputs", lambda: [])
    monkeypatch.setattr(ListeningComparisonDialog, "_start_render", lambda self: None)
    dialog = RawProcessedDialog(
        audio_data=np.full(1024, 0.05, dtype=np.float32),
        sample_rate=48_000,
        settings=_settings(),
    )
    yield dialog
    dialog.close()


def test_two_clip_dialog_offers_only_raw_and_processed(dialog):
    assert dialog.windowTitle() == "Test my sound"
    assert list(dialog._play_buttons) == ["original", "current"]
    assert [button.text() for button in dialog._play_buttons.values()] == [
        "Play Raw",
        "Play Processed",
    ]
    assert dialog.keep_button.isHidden()
    assert dialog.reject_button.text() == "Close"
    assert audit_widget_tree(dialog) == ()


def test_two_clip_dialog_marks_the_playing_clip_and_toggles_it_off(dialog):
    capture = np.full(1024, 0.05, dtype=np.float32)
    result = comparison.render_comparison(capture, 48_000, _settings(), _settings())
    dialog._on_render_ready(result)
    assert "Stages applied to the processed clip" in dialog.scope_warning.text()
    assert "Proposed" not in dialog.scope_warning.text()

    # Stand in for a started sink; the state logic does not touch the device.
    dialog._audio_sink = Mock()
    dialog._playing_clip_key = "current"
    dialog._update_playback_buttons()
    assert dialog._play_buttons["current"].text() == "Stop Processed"
    assert dialog._play_buttons["current"].accessibleName() == "Stop Processed"
    assert dialog._play_buttons["original"].text() == "Play Raw"
    assert dialog.status_label.text() == "Playing: Processed"

    dialog._play_clip("current")
    assert dialog._audio_sink is None
    assert dialog._playing_clip_key is None
    assert dialog._play_buttons["current"].text() == "Play Processed"
    assert dialog.status_label.text() == "Playback stopped."


def test_two_clip_dialog_shows_a_render_failure(dialog):
    dialog._on_render_failed("RuntimeError: native renderer is unavailable")
    assert dialog.status_label.text() == (
        "Could not process the recording: RuntimeError: native renderer is unavailable"
    )
    assert not any(button.isEnabled() for button in dialog._play_buttons.values())


class _Processor:
    def __init__(self, *, running: bool = True) -> None:
        self.running = running
        self.audio = np.full(480, 0.1, dtype=np.float32)
        self.start_error: Exception | None = None
        self.record_error: Exception | None = None
        self.progress = 1.0
        self.calls: list[str] = []

    def is_running(self) -> bool:
        return self.running

    def start(self, *_args, **_kwargs) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.calls.append("start")
        self.running = True

    def stop(self) -> None:
        self.calls.append("stop")
        self.running = False

    def get_active_input_device(self) -> str:
        return "Mic"

    def get_active_output_device(self) -> str:
        return "Out"

    def set_recovery_suppressed(self, _value: bool) -> None:
        pass

    def start_raw_recording(self, duration_s: float, before_cleanup: bool = False) -> None:
        if self.record_error is not None:
            raise self.record_error
        self.calls.append(f"record {duration_s:g} before_cleanup={before_cleanup}")

    def stop_raw_recording(self) -> np.ndarray:
        return self.audio

    def recording_progress(self) -> float:
        return self.progress

    def is_recording_complete(self) -> bool:
        return self.progress >= 1.0

    def sample_rate(self) -> int:
        return 48_000

    def get_input_cleanup_mode(self) -> str:
        return "off"


class _Window(QWidget):
    def __init__(self, processor: _Processor | None) -> None:
        super().__init__()
        if processor is not None:
            self.processor = processor
        self.input_combo = Mock()
        self.input_combo.currentData.return_value = DeviceIdentity(name="Mic")
        self.output_combo = Mock()
        self.output_combo.currentData.return_value = DeviceIdentity(name="Out")
        self.eq_panel = Mock()
        self.eq_panel.get_settings.return_value = _settings()
        self.mutes: list[tuple[bool, str]] = []

    def set_temporary_output_mute(self, muted: bool, reason: str) -> None:
        self.mutes.append((muted, reason))


@pytest.fixture
def harness(qapp, monkeypatch):
    """Record warnings and opened dialogs instead of showing either."""
    warnings: list[str] = []
    opened: list[dict] = []

    class _Dialog:
        def __init__(self, parent=None, **kwargs) -> None:
            opened.append(kwargs)

        def exec(self) -> int:
            return 0

        def deleteLater(self) -> None:
            pass

    monkeypatch.setattr(
        module.QMessageBox,
        "warning",
        lambda _parent, _title, text: warnings.append(text),
    )
    monkeypatch.setattr(module, "RawProcessedDialog", _Dialog)
    monkeypatch.setattr(
        module, "chain_settings", lambda _window, **kwargs: {"marker": kwargs}
    )
    return warnings, opened


def test_entry_reports_missing_processor(harness):
    warnings, opened = harness
    open_test_my_sound(_Window(None))
    assert warnings == ["Audio processing is not available."]
    assert opened == []


def test_entry_reports_missing_input_device(harness):
    warnings, opened = harness
    window = _Window(_Processor())
    window.input_combo.currentData.return_value = None
    open_test_my_sound(window)
    assert len(warnings) == 1 and "No microphone is selected" in warnings[0]
    assert opened == [] and window.processor.calls == []


def test_entry_reports_processing_on_another_route(harness):
    warnings, opened = harness
    window = _Window(_Processor())
    window.input_combo.currentData.return_value = DeviceIdentity(name="Other mic")
    open_test_my_sound(window)
    assert len(warnings) == 1 and "different devices" in warnings[0]
    assert opened == [] and window.processor.calls == []


def test_entry_reports_a_stream_that_will_not_start(harness):
    warnings, opened = harness
    processor = _Processor(running=False)
    processor.start_error = RuntimeError("device busy")
    open_test_my_sound(_Window(processor))
    assert len(warnings) == 1
    assert "Could not start the microphone" in warnings[0] and "device busy" in warnings[0]
    assert opened == []


def test_entry_reports_capture_failure_and_releases_what_it_took(harness):
    warnings, opened = harness
    processor = _Processor(running=False)
    processor.record_error = RuntimeError("capture unavailable")
    window = _Window(processor)
    open_test_my_sound(window)
    assert len(warnings) == 1
    assert "Recording did not work" in warnings[0] and "capture unavailable" in warnings[0]
    assert opened == []
    # The stream this call started is stopped again and its mute is released.
    assert processor.calls == ["start", "stop"]
    assert window.mutes == [(True, "test_my_sound"), (False, "test_my_sound")]


def test_entry_reports_a_stalled_capture(harness, monkeypatch):
    warnings, opened = harness
    processor = _Processor()
    processor.progress = 0.0
    monkeypatch.setattr(
        module.CaptureSession,
        "failure_reason",
        lambda self, _progress: "No audio capture progress was received.",
    )
    open_test_my_sound(_Window(processor))
    assert len(warnings) == 1
    assert "No audio capture progress was received." in warnings[0]
    assert opened == []


def test_entry_reports_an_empty_recording(harness):
    warnings, opened = harness
    processor = _Processor()
    processor.audio = np.zeros(0, dtype=np.float32)
    open_test_my_sound(_Window(processor))
    assert len(warnings) == 1 and "recording was empty" in warnings[0]
    assert opened == []


def test_entry_reports_settings_that_cannot_be_rendered(harness, monkeypatch):
    warnings, opened = harness

    def unavailable(_window, **_kwargs):
        raise TypeError("current processing configuration is unavailable")

    monkeypatch.setattr(module, "chain_settings", unavailable)
    window = _Window(_Processor())
    open_test_my_sound(window)
    assert len(warnings) == 1
    assert "Could not prepare the processed clip" in warnings[0]
    assert opened == []
    assert window.mutes == [(True, "test_my_sound"), (False, "test_my_sound")]


def test_entry_opens_the_two_clip_dialog_with_the_raw_capture(harness):
    warnings, opened = harness
    processor = _Processor()
    window = _Window(processor)
    open_test_my_sound(window)
    assert warnings == []
    assert processor.calls == ["record 5 before_cleanup=True"]
    assert len(opened) == 1
    np.testing.assert_array_equal(opened[0]["audio_data"], processor.audio)
    assert opened[0]["sample_rate"] == 48_000
    assert opened[0]["settings"] == _settings()
    assert opened[0]["chain_settings"] == {
        "marker": {"full_chain": True, "input_pre_filtered": False}
    }
    # One mute for the capture, one while the preview dialog is open.
    assert window.mutes == [(True, "test_my_sound"), (False, "test_my_sound")] * 2
