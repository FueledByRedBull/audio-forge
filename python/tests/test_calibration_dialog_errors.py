from unittest.mock import Mock, call

import pytest
from PySide6.QtWidgets import QWidget

from mic_eq.ui import capture_session
from mic_eq.ui.calibration_dialog import CalibrationDialog
from mic_eq.ui import calibration_dialog, voice_setup_dialog
from mic_eq.ui.engine_session import AudioLease
from mic_eq.ui.theme import message_text_style
from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog
from engine_fakes import fake_session


def test_recording_failure_remains_visible_after_reset(qapp):
    dialog = CalibrationDialog()
    dialog._report = Mock()  # type: ignore[method-assign]
    try:
        dialog.view_model.set_warning("Checking the recording...", state="info", strong=True)

        dialog._on_recording_failed("device disconnected")

        assert dialog.view_model.warning_text == "device disconnected"
        dialog._report.assert_called_once_with("failed", ["recording_failed"], ["device disconnected"])
        assert (dialog.view_model.warning_state, dialog.view_model.warning_strong) == ("bad", True)
    finally:
        dialog.reject()
        dialog.deleteLater()
        qapp.processEvents()


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
    setattr(owner, "_calibration_context_key", lambda: "selected-route")
    session = fake_session(processor)
    session.set_user_muted(True)
    setattr(owner, "engine_session", session)
    dialog = dialog_type(parent=owner)
    # The take borrowed the audio already; a capture mutes through that lease.
    dialog._audio_lease = AudioLease(session, "capture", started=False)
    report = Mock(wraps=dialog._report)
    monkeypatch.setattr(dialog, "_report", report)
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

        if dialog_type is CalibrationDialog:
            text = dialog.view_model.warning_text
            assert (dialog.view_model.warning_state, dialog.view_model.warning_strong) == ("bad", True)
        else:
            text = dialog.warning_label.text()
            assert dialog.warning_label.styleSheet() == message_text_style("bad", strong=True)
        assert "The microphone stopped sending sound" in text and "record again" in text
        assert report.call_args.args[:2] == ("failed", ["recording_failed"])
        assert dialog.recording_timer.isActive() is False
        assert dialog._capture_session is None
        assert session.view.mute_reasons == frozenset()
        assert session.view.user_muted is True
        assert processor.set_recovery_suppressed.call_args_list == [
            call(True),
            call(False),
        ]
    finally:
        dialog.reject()
        dialog.deleteLater()
        owner.close()
        qapp.processEvents()


@pytest.mark.parametrize("dialog_type", [CalibrationDialog, VoiceSetupDialog])
def test_a_take_refused_while_the_engine_is_busy_says_to_wait_not_to_check_devices(
    qapp, monkeypatch, dialog_type
):
    owner = QWidget()
    session = fake_session()
    setattr(owner, "processor", session.processor)
    setattr(owner, "engine_session", session)
    session.request_start("button")  # the start stays out: the fake tasks never run it
    messages: list[str] = []
    module = calibration_dialog if dialog_type is CalibrationDialog else voice_setup_dialog
    monkeypatch.setattr(
        module.QMessageBox, "critical", lambda _parent, _title, text: messages.append(text)
    )
    dialog = dialog_type(parent=owner)
    try:
        if isinstance(dialog, CalibrationDialog):
            assert dialog._borrow_audio(session) is None
        else:
            assert dialog._ensure_processor_ready() is False
        assert len(messages) == 1
        assert "Wait a moment and try again" in messages[0]
        assert "connected" not in messages[0]
    finally:
        dialog.reject()
        dialog.deleteLater()
        owner.close()
        qapp.processEvents()


def test_latency_calibration_refused_while_the_engine_is_busy_says_to_wait(qapp, monkeypatch):
    from mic_eq.ui import latency_calibration_dialog as latency_ui

    owner = QWidget()
    session = fake_session()
    setattr(owner, "processor", session.processor)
    setattr(owner, "engine_session", session)
    session.request_start("button")  # the start stays out: the fake tasks never run it
    messages: list[str] = []
    monkeypatch.setattr(
        latency_ui.QMessageBox, "critical", lambda _parent, _title, text: messages.append(text)
    )
    dialog = latency_ui.LatencyCalibrationDialog(owner)
    try:
        dialog._on_run_clicked()
        assert messages == ["Audio is starting, stopping or recovering. Wait a moment and try again."]
    finally:
        dialog.reject()
        dialog.deleteLater()
        owner.close()
        qapp.processEvents()


@pytest.mark.parametrize("dialog_type", [CalibrationDialog, VoiceSetupDialog])
def test_a_switch_refused_because_the_engine_went_busy_says_to_wait(qapp, monkeypatch, dialog_type):
    from engine_fakes import FakeEngine

    engine = FakeEngine()
    engine.start("Headset", "Destination")  # running on another microphone than the selected one
    owner = QWidget()
    session = fake_session(engine)
    setattr(owner, "processor", engine)
    setattr(owner, "engine_session", session)
    module = calibration_dialog if dialog_type is CalibrationDialog else voice_setup_dialog
    messages: list[str] = []
    monkeypatch.setattr(
        module.QMessageBox, "critical", lambda _parent, _title, text: messages.append(text)
    )

    def answer_yes_after_a_stop_went_out(*_args):
        # The question runs its own event loop: a Stop can go out before the user answers.
        session.request_stop("button")
        return module.QMessageBox.StandardButton.Yes

    monkeypatch.setattr(module.QMessageBox, "question", answer_yes_after_a_stop_went_out)
    dialog = dialog_type(parent=owner)
    try:
        if isinstance(dialog, CalibrationDialog):
            assert dialog._borrow_audio(session) is None
        else:
            assert dialog._ensure_processor_ready() is False
        assert messages == ["Audio is starting, stopping or recovering. Wait a moment and try again."]
    finally:
        dialog.reject()
        dialog.deleteLater()
        owner.close()
        qapp.processEvents()
