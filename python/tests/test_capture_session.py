from __future__ import annotations

from unittest.mock import Mock, call

import pytest

from mic_eq.ui.capture_session import CaptureProgressDeadline, CaptureSession


def test_capture_deadline_fails_when_recording_progress_stops() -> None:
    deadline = CaptureProgressDeadline(
        duration_s=10.0,
        no_progress_timeout_s=2.0,
        start_grace_s=1.0,
        completion_grace_s=5.0,
        started_at=100.0,
    )

    assert deadline.failure_reason(0.0, now=101.9) is None
    assert deadline.failure_reason(0.0, now=102.1) == (
        "No audio capture progress was received."
    )


def test_capture_deadline_bounds_total_elapsed_time_while_progress_advances() -> None:
    deadline = CaptureProgressDeadline(
        duration_s=2.0,
        no_progress_timeout_s=2.0,
        start_grace_s=0.0,
        completion_grace_s=1.0,
        started_at=10.0,
    )

    assert deadline.failure_reason(0.1, now=11.0) is None
    assert deadline.failure_reason(0.2, now=12.0) is None
    assert deadline.failure_reason(0.3, now=13.1) == (
        "Audio capture exceeded its 3-second time limit."
    )


def test_capture_session_restores_prior_recovery_and_owned_mute() -> None:
    processor = Mock()
    processor.is_recovery_suppressed.return_value = True
    audio = object()
    processor.stop_raw_recording.return_value = audio
    set_mute = Mock()
    owner = Mock(processor=processor, set_temporary_output_mute=set_mute)
    session = CaptureSession(owner, "test_capture")

    session.start(4.0, before_cleanup=True)
    captured = session.stop_recording()

    assert captured is audio
    set_mute.assert_has_calls([call(True, "test_capture"), call(False, "test_capture")])
    processor.set_recovery_suppressed.assert_has_calls([call(True), call(True)])
    processor.stop_raw_recording.assert_called_once_with()


def test_capture_session_cleans_up_when_native_start_raises() -> None:
    processor = Mock()
    processor.is_recovery_suppressed.return_value = False
    processor.start_raw_recording.side_effect = RuntimeError("capture unavailable")
    set_mute = Mock()
    owner = Mock(processor=processor, set_temporary_output_mute=set_mute)
    session = CaptureSession(owner, "test_capture")

    with pytest.raises(RuntimeError, match="capture unavailable"):
        session.start(4.0, before_cleanup=True)

    set_mute.assert_has_calls([call(True, "test_capture"), call(False, "test_capture")])
    processor.set_recovery_suppressed.assert_has_calls([call(True), call(False)])


def test_capture_session_aborts_when_prior_recovery_state_cannot_be_read() -> None:
    recovery_state = True

    def get_recovery_suppressed() -> bool:
        raise RuntimeError("recovery state unavailable")

    def set_recovery_suppressed(value: bool) -> None:
        nonlocal recovery_state
        recovery_state = value

    processor = Mock()
    processor.is_recovery_suppressed.side_effect = get_recovery_suppressed
    processor.set_recovery_suppressed.side_effect = set_recovery_suppressed
    set_mute = Mock()
    owner = Mock(processor=processor, set_temporary_output_mute=set_mute)
    session = CaptureSession(owner, "test_capture")

    with pytest.raises(RuntimeError, match="recovery state unavailable"):
        session.start(4.0, before_cleanup=True)

    assert recovery_state is True
    assert session.active is False
    processor.start_raw_recording.assert_not_called()
    processor.set_recovery_suppressed.assert_not_called()
    set_mute.assert_has_calls([call(True, "test_capture"), call(False, "test_capture")])


def test_capture_session_retries_failed_state_restoration_on_cleanup() -> None:
    processor = Mock()
    processor.is_recovery_suppressed.return_value = True
    processor.set_recovery_suppressed.side_effect = [
        None,
        RuntimeError("recovery restore failed"),
        None,
    ]
    processor.stop_raw_recording.return_value = object()
    mute_active = False
    mute_releases = 0

    def set_mute(muted: bool, _reason: str) -> None:
        nonlocal mute_active, mute_releases
        if muted:
            mute_active = True
        else:
            mute_releases += 1
            if mute_releases == 1:
                raise RuntimeError("mute restore failed")
            mute_active = False

    owner = Mock(processor=processor, set_temporary_output_mute=set_mute)
    session = CaptureSession(owner, "test_capture")

    session.start(4.0, before_cleanup=True)
    session.stop_recording()
    session.cleanup()

    processor.set_recovery_suppressed.assert_has_calls([call(True), call(True), call(True)])
    assert mute_active is False
    assert mute_releases == 2
