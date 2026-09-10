"""Latency evidence stays tied to the negotiated endpoint format."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from PyQt6.QtWidgets import QComboBox
from PyQt6.QtWidgets import QWidget

from mic_eq.config import (
    AppConfig,
    DeviceIdentity,
    LatencyCalibrationProfile,
    build_device_route_key,
)
from mic_eq.ui.main_window import MainWindow
from mic_eq.ui.latency_calibration_dialog import LatencyCalibrationDialog


def _profile(context: tuple[int | None, ...] | None) -> LatencyCalibrationProfile:
    return LatencyCalibrationProfile(
        measured_round_trip_ms=42.0,
        estimated_one_way_ms=21.0,
        applied_compensation_ms=42.0,
        confidence=0.9,
        route_latency_ms=42.0,
        capture_format_context=context,
    )


def _owner(
    input_identity: DeviceIdentity,
    output_identity: DeviceIdentity,
    profile: LatencyCalibrationProfile,
    *,
    config: Any | None = None,
) -> Any:
    owner = cast(Any, MainWindow.__new__(MainWindow))
    owner.input_combo = QComboBox()
    owner.output_combo = QComboBox()
    owner.input_combo.addItem(input_identity.name, input_identity)
    owner.output_combo.addItem(output_identity.name, output_identity)
    route_key = build_device_route_key(input_identity, output_identity)
    owner.config = config or SimpleNamespace(
        use_measured_latency=True,
        latency_calibration_profiles={route_key: profile},
    )
    owner.processor = Mock()
    owner.processor.is_running.return_value = False
    return owner


def _close_owner(owner: Any, qapp) -> None:
    owner.input_combo.deleteLater()
    owner.output_combo.deleteLater()
    qapp.processEvents()


@pytest.mark.parametrize("operation", ["save", "reset"])
@pytest.mark.parametrize("write_error", [False, True])
def test_latency_profile_write_failure_preserves_previous_profile_and_allows_retry(
    qapp, monkeypatch, operation, write_error
):
    input_identity = DeviceIdentity(name="Mic", endpoint_id="in", direction="input", sample_rate=48_000, channels=1)
    output_identity = DeviceIdentity(name="Cable", endpoint_id="out", direction="output", sample_rate=48_000, channels=2)
    previous = _profile((48_000, 1, 48_000, 2))
    owner = _owner(input_identity, output_identity, previous)
    owner.status_bar = Mock()
    owner._apply_latency_compensation_for_current_devices = Mock()
    replacement = _profile((48_000, 1, 48_000, 2))
    replacement.route_latency_ms = 65.0
    action = (
        lambda: owner._on_latency_calibration_saved(replacement.to_dict())
    ) if operation == "save" else owner._on_latency_calibration_reset
    save = Mock(side_effect=OSError("disk full")) if write_error else Mock(return_value=False)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", save)
    try:
        assert action() is False
        assert list(owner.config.latency_calibration_profiles.values()) == [previous]
        owner._apply_latency_compensation_for_current_devices.assert_not_called()
        assert "could not be saved" in owner.status_bar.showMessage.call_args.args[0]

        monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
        assert action() is True
        owner._apply_latency_compensation_for_current_devices.assert_called_once()
        profiles = list(owner.config.latency_calibration_profiles.values())
        assert [profile.route_latency_ms for profile in profiles] == (
            [65.0] if operation == "save" else []
        )
    finally:
        _close_owner(owner, qapp)


def test_latency_dialog_waits_for_successful_save_before_accepting(qapp):
    owner = cast(Any, QWidget())
    owner.processor = Mock()
    owner._current_device_route_key = lambda: "route"
    owner._current_capture_format_context = lambda: (48_000, 1, 48_000, 2)
    owner._calibration_context_key = lambda: "context"
    owner._on_latency_calibration_saved = Mock(return_value=False)
    profile = _profile((48_000, 1, 48_000, 2)).to_dict()
    dialog = LatencyCalibrationDialog(owner, existing_profile=profile)
    saved = Mock()
    dialog.calibration_saved.connect(saved)
    try:
        dialog._on_accept_clicked()
        assert dialog.result() == dialog.DialogCode.Rejected
        assert dialog._latest_profile == profile
        assert "could not be saved" in dialog.status_label.text()
        saved.assert_not_called()

        owner._on_latency_calibration_saved.return_value = True
        dialog._on_accept_clicked()
        assert dialog.result() == dialog.DialogCode.Accepted
        saved.assert_called_once_with(profile)
    finally:
        dialog.close()
        owner.deleteLater()
        qapp.processEvents()


def test_cancelled_latency_dialog_does_not_report_an_old_profile_as_saved(monkeypatch):
    owner = SimpleNamespace(
        _latency_profile_key=lambda: "route",
        _current_latency_profile=lambda: _profile((48_000, 1, 48_000, 2)),
        _refresh_latency_profile_engine=lambda _profile: False,
        _calibration_dialog_open=False,
    )
    dialog = Mock()
    dialog.exec.return_value = 0
    monkeypatch.setattr("mic_eq.ui.main_window.LatencyCalibrationDialog", lambda *args, **kwargs: dialog)
    assert MainWindow._on_latency_calibration_clicked(cast(Any, owner)) is False
    assert owner._calibration_dialog_open is False


def test_same_route_with_changed_sample_rate_or_channels_drops_old_compensation(qapp):
    input_identity = DeviceIdentity(
        name="Mic",
        endpoint_id="input-route",
        host_api="WASAPI",
        direction="input",
        sample_rate=48_000,
        channels=1,
    )
    output_identity = DeviceIdentity(
        name="Cable",
        endpoint_id="output-route",
        host_api="WASAPI",
        direction="output",
        sample_rate=48_000,
        channels=2,
    )
    profile = _profile((48_000, 1, 48_000, 2))
    owner = _owner(input_identity, output_identity, profile)
    try:
        changed_input = DeviceIdentity(
            name="Mic",
            endpoint_id="input-route",
            host_api="WASAPI",
            direction="input",
            sample_rate=44_100,
            channels=2,
        )
        owner.input_combo.setItemData(0, changed_input)

        assert MainWindow._current_device_route_key(owner) == build_device_route_key(
            input_identity, output_identity
        )
        MainWindow._apply_latency_compensation_for_current_devices(owner)

        owner.processor.set_latency_compensation_ms.assert_called_once_with(0.0)
    finally:
        _close_owner(owner, qapp)


def test_matching_format_reuses_profile_without_relabeling_context(qapp):
    input_identity = DeviceIdentity(
        name="Mic",
        endpoint_id="input-route",
        host_api="WASAPI",
        direction="input",
        sample_rate=48_000,
        channels=1,
    )
    output_identity = DeviceIdentity(
        name="Cable",
        endpoint_id="output-route",
        host_api="WASAPI",
        direction="output",
        sample_rate=48_000,
        channels=2,
    )
    profile = _profile((48_000, 1, 48_000, 2))
    config = SimpleNamespace(
        use_measured_latency=True,
        latency_calibration_profiles={},
    )
    owner = _owner(input_identity, output_identity, profile, config=config)
    try:
        MainWindow._sync_latency_profile_for_current_devices(owner, profile)
        MainWindow._apply_latency_compensation_for_current_devices(owner)

        assert profile.capture_format_context == (48_000, 1, 48_000, 2)
        owner.processor.set_latency_compensation_ms.assert_called_once_with(42.0)
    finally:
        _close_owner(owner, qapp)


def test_running_processor_context_uses_negotiated_sample_rates(qapp):
    input_identity = DeviceIdentity(
        name="Mic",
        endpoint_id="input-route",
        host_api="WASAPI",
        direction="input",
        sample_rate=44_100,
        channels=1,
    )
    output_identity = DeviceIdentity(
        name="Cable",
        endpoint_id="output-route",
        host_api="WASAPI",
        direction="output",
        sample_rate=44_100,
        channels=2,
    )
    owner = _owner(input_identity, output_identity, _profile(None))
    owner.processor.is_running.return_value = True
    owner.processor.get_runtime_diagnostics.return_value = {
        "input_sample_rate": 48_000,
        "output_sample_rate": 48_000,
    }
    owner.processor.output_sample_rate.return_value = 48_000
    try:
        assert MainWindow._current_capture_format_context(owner) == (
            48_000,
            1,
            48_000,
            2,
        )
    finally:
        _close_owner(owner, qapp)


def test_restart_preserves_context_but_legacy_profile_is_not_applied(qapp):
    input_identity = DeviceIdentity(
        name="Mic",
        endpoint_id="input-route",
        host_api="WASAPI",
        direction="input",
        sample_rate=48_000,
        channels=1,
    )
    output_identity = DeviceIdentity(
        name="Cable",
        endpoint_id="output-route",
        host_api="WASAPI",
        direction="output",
        sample_rate=48_000,
        channels=2,
    )
    route_key = build_device_route_key(input_identity, output_identity)
    config = AppConfig(
        use_measured_latency=True,
        last_input_device_identity=input_identity,
        last_output_device_identity=output_identity,
        latency_calibration_profiles={
            route_key: _profile((48_000, 1, 48_000, 2))
        },
    )
    restored = AppConfig.from_dict(config.to_dict())
    owner = _owner(
        input_identity,
        output_identity,
        restored.latency_calibration_profiles[route_key],
        config=restored,
    )
    try:
        assert MainWindow._current_latency_profile(owner) is not None

        legacy_data = config.to_dict()
        legacy_data["latency_calibration_profiles"][route_key].pop(
            "capture_format_context"
        )
        legacy = AppConfig.from_dict(legacy_data)
        owner.config = legacy
        assert MainWindow._current_latency_profile(owner) is None
    finally:
        _close_owner(owner, qapp)


def test_profile_context_parser_requires_positive_known_values():
    raw = _profile((48_000, 1, 48_000, 2)).to_dict()
    raw["capture_format_context[0]"] = 0
    restored = LatencyCalibrationProfile.from_dict(raw)
    assert restored.capture_format_context == (48_000, 1, 48_000, 2)

    raw["capture_format_context"] = [0, 1, 48_000, 2]
    with pytest.raises(ValueError, match=r"capture_format_context\[0\]"):
        LatencyCalibrationProfile.from_dict(raw)


def test_dialog_rejects_result_after_route_context_changes(qapp):
    class Owner(QWidget):
        def __init__(self):
            super().__init__()
            self.processor = Mock()
            self.route = "route-a"
            self.context = "context-a"

        def _current_device_route_key(self):
            return self.route

        def _current_capture_format_context(self):
            return (48_000, 1, 48_000, 2)

        def _calibration_context_key(self):
            return self.context

    owner = Owner()
    dialog = LatencyCalibrationDialog(
        owner,
        existing_profile={
            "capture_format_context": [48_000, 1, 48_000, 2],
            "measured_round_trip_ms": 42.0,
        },
    )
    saved: list[dict] = []
    dialog.calibration_saved.connect(saved.append)
    owner.route = "route-b"
    dialog._on_accept_clicked()

    assert saved == []
    assert "stale" in dialog.status_label.text().lower()
    dialog.close()
    owner.deleteLater()
    qapp.processEvents()


def test_failed_retake_clears_old_profile_before_accept(qapp, monkeypatch):
    from mic_eq.ui import latency_calibration_dialog as latency_module

    class Owner(QWidget):
        def __init__(self):
            super().__init__()
            self.processor = Mock()
            self.processor.is_running.return_value = False
            self.route = "route-a"

        def _current_device_route_key(self):
            return self.route

        def _current_capture_format_context(self):
            return (48_000, 1, 48_000, 2)

        def _calibration_context_key(self):
            return self.route

    owner = Owner()
    dialog = LatencyCalibrationDialog(
        owner,
        existing_profile={
            "capture_format_context": [48_000, 1, 48_000, 2],
            "measured_round_trip_ms": 42.0,
        },
    )
    critical = Mock()
    information = Mock()
    monkeypatch.setattr(latency_module.QMessageBox, "critical", critical)
    monkeypatch.setattr(latency_module.QMessageBox, "information", information)
    monkeypatch.setattr(
        latency_module,
        "start_processor_for_route",
        Mock(side_effect=RuntimeError("start failed")),
    )
    owner.route = "route-b"

    dialog._on_run_clicked()
    assert dialog._latest_profile is None

    dialog._on_accept_clicked()
    assert information.called
    dialog.close()
    owner.deleteLater()
    qapp.processEvents()


def test_running_stream_on_different_route_is_rejected(qapp, monkeypatch):
    from mic_eq.ui import latency_calibration_dialog as latency_module

    class Owner(QWidget):
        def __init__(self):
            super().__init__()
            self.processor = Mock()
            self.processor.is_running.return_value = True
            self.processor.get_active_input_device.return_value = "Mic A"
            self.processor.get_active_input_device_endpoint_id.return_value = "in-a"
            self.processor.get_active_input_device_name_ordinal.return_value = 0
            self.processor.get_active_output_device.return_value = "Cable A"
            self.processor.get_active_output_device_endpoint_id.return_value = "out-a"
            self.processor.get_active_output_device_name_ordinal.return_value = 0
            self.input_combo = QComboBox()
            self.output_combo = QComboBox()
            self.input_combo.addItem(
                "Mic B",
                DeviceIdentity(
                    name="Mic B",
                    endpoint_id="in-b",
                    direction="input",
                    sample_rate=48_000,
                    channels=1,
                ),
            )
            self.output_combo.addItem(
                "Cable B",
                DeviceIdentity(
                    name="Cable B",
                    endpoint_id="out-b",
                    direction="output",
                    sample_rate=48_000,
                    channels=2,
                ),
            )

        def _current_device_route_key(self):
            return "selected-route"

        def _current_capture_format_context(self):
            return (48_000, 1, 48_000, 2)

        def _calibration_context_key(self):
            return "selected-context"

    owner = Owner()
    dialog = LatencyCalibrationDialog(owner)
    start = Mock()
    monkeypatch.setattr(latency_module, "start_processor_for_route", start)

    dialog._on_run_clicked()

    start.assert_not_called()
    assert "selected route" in dialog.status_label.text().lower()
    dialog.close()
    owner.deleteLater()
    qapp.processEvents()
