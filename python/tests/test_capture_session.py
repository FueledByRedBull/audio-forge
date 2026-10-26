from __future__ import annotations

from unittest.mock import Mock, call

import pytest
from PySide6.QtWidgets import QWidget

from mic_eq.config import DeviceIdentity
from mic_eq.ui.capture_session import (
    CaptureProgressDeadline,
    CaptureSession,
    active_device_identities as _active_device_identities,
    restart_processor_for_route as _restart_processor_for_route,
    route_identities_match as _route_identities_match,
    set_temporary_mute as _set_temporary_mute,
)


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


def test_calibration_owner_discovery_and_chain_snapshot_require_no_processing_panels(qapp):
    from types import SimpleNamespace

    from mic_eq.ui.calibration_support import chain_settings
    from mic_eq.ui.capture_session import find_eq_state_owner

    owner = QWidget()
    vars(owner)["eq_state"] = SimpleNamespace(get_settings=lambda: {})
    vars(owner)["deesser_state"] = SimpleNamespace(get_settings=lambda: {"enabled": True})
    vars(owner)["compressor_state"] = SimpleNamespace(get_settings=lambda: {"ratio": 3.5})
    vars(owner)["limiter_state"] = SimpleNamespace(get_settings=lambda: {"ceiling_db": -2.0})
    nested = QWidget(owner)
    assert find_eq_state_owner(nested) is owner
    assert chain_settings(owner) == {
        "deesser": {"enabled": True},
        "compressor": {"ratio": 3.5},
        "limiter": {"ceiling_db": -2.0},
    }


class _MuteOwner:
    user_muted = True

    def __init__(self) -> None:
        self.calls: list[tuple[bool, str]] = []

    def set_temporary_output_mute(self, muted: bool, reason: str) -> None:
        self.calls.append((muted, reason))


class _ProcessorOnlyOwner:
    user_muted = True

    def __init__(self) -> None:
        self.processor = Mock()


def test_temporary_mute_keeps_owner_reason_and_fallback_user_mute() -> None:
    owner = _MuteOwner()
    _set_temporary_mute(owner, "auto_eq_calibration", True)
    _set_temporary_mute(owner, "auto_eq_calibration", False)
    assert owner.calls == [
        (True, "auto_eq_calibration"),
        (False, "auto_eq_calibration"),
    ]

    fallback_owner = _ProcessorOnlyOwner()
    _set_temporary_mute(fallback_owner, "auto_voice_setup", False)
    fallback_owner.processor.set_output_mute.assert_called_once_with(True)


def test_route_identity_uses_endpoint_ids_and_default_semantics() -> None:
    selected = (
        DeviceIdentity(name="Mic", endpoint_id="new-mic", direction="input"),
        DeviceIdentity(name="Speaker", endpoint_id="new-speaker", direction="output"),
    )
    active = (
        DeviceIdentity(name="Mic", endpoint_id="old-mic", direction="input"),
        DeviceIdentity(name="Speaker", endpoint_id="old-speaker", direction="output"),
    )
    assert _route_identities_match(selected, active) is False
    assert _route_identities_match((None, None), (None, None)) is True


def test_active_route_reads_endpoint_ids_and_ordinals() -> None:
    processor = Mock()
    processor.get_active_input_device.return_value = "Mic"
    processor.get_active_input_device_endpoint_id.return_value = "mic-id"
    processor.get_active_input_device_name_ordinal.return_value = 1
    processor.get_active_output_device.return_value = "Speaker"
    processor.get_active_output_device_endpoint_id.return_value = "speaker-id"
    processor.get_active_output_device_name_ordinal.return_value = 0

    active_input, active_output = _active_device_identities(processor)

    assert active_input == DeviceIdentity(
        name="Mic", endpoint_id="mic-id", direction="input", name_ordinal=1
    )
    assert active_output == DeviceIdentity(
        name="Speaker", endpoint_id="speaker-id", direction="output", name_ordinal=0
    )


class _RouteProcessor:
    def __init__(self, *, fail_endpoint_ids: set[str] | None = None) -> None:
        self.fail_endpoint_ids = fail_endpoint_ids or set()
        self.running = True
        self.stop_calls = 0
        self.starts: list[tuple[object, ...]] = []

    def stop(self) -> None:
        self.stop_calls += 1
        self.running = False

    def start(self, *args, **kwargs) -> str:
        endpoint_id = kwargs.get("input_device_endpoint_id")
        self.starts.append((*args, endpoint_id, kwargs.get("output_device_endpoint_id")))
        if endpoint_id in self.fail_endpoint_ids:
            raise RuntimeError(f"cannot open {endpoint_id}")
        self.running = True
        return "started"


def test_route_handoff_restores_previous_route_after_selected_start_failure() -> None:
    old_route = (
        DeviceIdentity(name="Mic", endpoint_id="old-mic", direction="input"),
        DeviceIdentity(name="Speaker", endpoint_id="old-speaker", direction="output"),
    )
    selected_route = (
        DeviceIdentity(name="Mic", endpoint_id="new-mic", direction="input"),
        DeviceIdentity(name="Speaker", endpoint_id="new-speaker", direction="output"),
    )
    processor = _RouteProcessor(fail_endpoint_ids={"new-mic"})

    try:
        _restart_processor_for_route(processor, selected_route, old_route)
    except RuntimeError as exc:
        assert "previous route restored" in str(exc)
    else:
        raise AssertionError("selected route failure was not reported")

    assert processor.stop_calls == 1
    assert processor.running is True
    assert processor.starts[-1][4] == "old-mic"
    assert processor.starts[-1][5] == "old-speaker"


def test_route_handoff_reports_restore_failure() -> None:
    old_route = (
        DeviceIdentity(name="Mic", endpoint_id="old-mic", direction="input"),
        DeviceIdentity(name="Speaker", endpoint_id="old-speaker", direction="output"),
    )
    selected_route = (
        DeviceIdentity(name="Mic", endpoint_id="new-mic", direction="input"),
        DeviceIdentity(name="Speaker", endpoint_id="new-speaker", direction="output"),
    )
    processor = _RouteProcessor(fail_endpoint_ids={"new-mic", "old-mic"})

    try:
        _restart_processor_for_route(processor, selected_route, old_route)
    except RuntimeError as exc:
        assert "failed to restore previous route" in str(exc)
    else:
        raise AssertionError("selected route failure was not reported")

    assert processor.running is False


def test_route_handoff_starts_selected_route_normally() -> None:
    old_route = (
        DeviceIdentity(name="Mic", endpoint_id="old-mic", direction="input"),
        DeviceIdentity(name="Speaker", endpoint_id="old-speaker", direction="output"),
    )
    selected_route = (
        DeviceIdentity(name="Mic", endpoint_id="new-mic", direction="input"),
        DeviceIdentity(name="Speaker", endpoint_id="new-speaker", direction="output"),
    )
    processor = _RouteProcessor()

    assert _restart_processor_for_route(processor, selected_route, old_route) == "started"
    assert processor.running is True
    assert processor.starts[-1][4] == "new-mic"
    assert processor.starts[-1][5] == "new-speaker"


def test_route_handoff_keeps_stream_when_previous_identity_is_unknown() -> None:
    processor = _RouteProcessor()
    selected = (DeviceIdentity(name="Mic", endpoint_id="mic"), None)
    try:
        _restart_processor_for_route(processor, selected, (None, None))
    except RuntimeError as exc:
        assert "Cannot identify the current route" in str(exc)
    else:
        raise AssertionError("handoff must preserve the unidentified stream")
    assert processor.stop_calls == 0
    assert processor.running is True
    assert processor.starts == []
