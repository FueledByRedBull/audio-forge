"""Focused contracts for product-state persistence and route ownership."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

from PyQt6.QtWidgets import QComboBox, QWidget

from mic_eq.config import (
    AppConfig,
    DeviceIdentity,
    EQSettings,
    InputDevicePreference,
    Preset,
    build_device_route_key,
    build_input_device_preference_key,
)
from mic_eq.ui.first_run_setup_dialog import FirstRunSetupDialog
from mic_eq.ui.main_window import MainWindow


def test_active_preset_identity_tracks_edit_history_and_restart(qapp, monkeypatch, tmp_path) -> None:
    from mic_eq.config_parts import shared
    from mic_eq.ui import main_window

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [])

    window = MainWindow()
    try:
        preset = window._get_current_preset()
        preset.name = "Lifecycle"
        filepath = window._save_preset_file(preset)
        assert filepath is not None and filepath.is_file()
        assert window.current_preset_name == "Lifecycle"
        assert window.current_preset_path == filepath
        assert window.current_preset_modified is False

        window.gate_panel.threshold_spinbox.setValue(-33.0)
        assert window._commit_pending_configuration_snapshot(source="ui")
        assert window.current_preset_modified is True

        window.undo_configuration()
        assert window.current_preset_modified is False
        window.redo_configuration()
        assert window.current_preset_modified is True
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()

    restored = MainWindow()
    try:
        assert restored.current_preset_name == "Lifecycle"
        assert restored.current_preset_path == filepath
        assert restored.current_preset_modified is False
    finally:
        restored.close()
        restored.deleteLater()
        qapp.processEvents()


def test_start_applies_persisted_mute_before_and_after_native_start(
    qapp, monkeypatch, tmp_path
) -> None:
    from mic_eq.config_parts import shared
    from mic_eq.ui import main_window

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [])

    window = MainWindow()
    events: list[tuple[str, bool | None]] = []

    class Processor:
        running = False

        def is_running(self) -> bool:
            return self.running

        def set_output_mute(self, muted: bool) -> None:
            events.append(("mute", muted))

        def stop(self) -> None:
            self.running = False

    processor = Processor()
    cast(Any, window).processor = processor
    window.user_muted = True
    window._temporary_mute_reasons = set()
    window._apply_input_preferences_for_current_route = lambda: None

    def start(_processor: Processor, *_devices: object) -> str:
        events.append(("start", None))
        processor.running = True
        return "started"

    monkeypatch.setattr(main_window, "start_processor_for_route", start)
    try:
        window._start_processing()
        assert events == [("mute", True), ("start", None), ("mute", True)]
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_legacy_input_modes_migrate_only_to_the_known_stable_device() -> None:
    input_identity = DeviceIdentity(
        name="Old mic",
        endpoint_id="input-old",
        host_api="WASAPI",
        direction="input",
    )
    config = AppConfig.from_dict(
        {
            "last_input_device": input_identity.name,
            "last_input_device_identity": input_identity.to_dict(),
            "input_channel_mode": "left",
            "input_cleanup_mode": "gentle",
        }
    )

    key = build_input_device_preference_key(input_identity)
    assert config.input_device_preferences[key] == InputDevicePreference(
        "left", "gentle", "legacy_migration"
    )
    assert AppConfig.from_dict(config.to_dict()).input_device_preferences == config.input_device_preferences


def test_legacy_global_modes_without_stable_identity_use_safe_defaults() -> None:
    config = AppConfig.from_dict(
        {
            "last_input_device": "Old mic",
            "input_channel_mode": "left",
            "input_cleanup_mode": "gentle",
        }
    )

    assert config.input_device_preferences == {}
    owner = cast(Any, MainWindow.__new__(MainWindow))
    owner.config = config
    owner._current_device_route_key = lambda: None
    owner._current_input_preference_key = lambda: None
    assert MainWindow._input_preference_for_current_route(owner) == InputDevicePreference()


def test_missing_route_override_does_not_fall_back_to_global_modes() -> None:
    owner = cast(Any, MainWindow.__new__(MainWindow))
    owner.config = AppConfig(
        input_channel_mode="left",
        input_cleanup_mode="gentle",
        route_input_preferences={"another-route": InputDevicePreference("right", "strong")},
    )
    owner._current_device_route_key = lambda: "missing-route"
    owner._current_input_preference_key = lambda: None

    assert MainWindow._input_preference_for_current_route(owner) == InputDevicePreference()


def test_route_input_preference_round_trip_keeps_exact_route_override() -> None:
    input_identity = DeviceIdentity(name="Mic", endpoint_id="in", direction="input")
    output_identity = DeviceIdentity(name="Cable", endpoint_id="out", direction="output")
    route_key = build_device_route_key(input_identity, output_identity)
    config = AppConfig(
        input_device_preferences={
            build_input_device_preference_key(input_identity): InputDevicePreference(
                "left", "gentle"
            )
        },
        route_input_preferences={
            route_key: InputDevicePreference("right", "strong")
        },
    )

    restored = AppConfig.from_dict(config.to_dict())
    assert restored.route_input_preferences[route_key] == InputDevicePreference(
        "right", "strong"
    )


def test_route_input_preference_action_saves_visible_controls_for_current_route(
    qapp, monkeypatch, tmp_path
) -> None:
    from mic_eq.config_parts import shared
    from mic_eq.ui import main_window

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [
        SimpleNamespace(
            name="Mic",
            is_default=True,
            endpoint_id="input-route",
            host_api="WASAPI",
            sample_rate=48_000,
            channels=1,
            name_ordinal=0,
        )
    ])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [
        SimpleNamespace(
            name="Cable",
            is_default=True,
            endpoint_id="output-route",
            host_api="WASAPI",
            sample_rate=48_000,
            channels=2,
            name_ordinal=0,
        )
    ])

    window = MainWindow()
    try:
        window.input_channel_mode_combo.setCurrentIndex(
            window.input_channel_mode_combo.findData("left")
        )
        window.input_cleanup_mode_combo.setCurrentIndex(
            window.input_cleanup_mode_combo.findData("gentle")
        )
        window.save_route_input_preference_action.trigger()

        route_key = window._current_device_route_key()
        assert route_key is not None
        assert window.config.route_input_preferences[route_key] == InputDevicePreference(
            "left", "gentle", "explicit_user"
        )
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_calibration_context_tracks_endpoint_format_without_changing_route_key(qapp) -> None:
    owner = cast(Any, MainWindow.__new__(MainWindow))
    owner.input_combo = QComboBox()
    owner.output_combo = QComboBox()
    owner.config = SimpleNamespace(
        input_channel_mode="phase_safe_mono", input_cleanup_mode="off"
    )
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
    owner.input_combo.addItem("Mic", input_identity)
    owner.output_combo.addItem("Cable", output_identity)
    try:
        route_key = MainWindow._current_device_route_key(owner)
        context_before = MainWindow._calibration_context_key(owner)
        owner.input_combo.setItemData(
            0,
            DeviceIdentity(
                name="Mic",
                endpoint_id="input-route",
                host_api="WASAPI",
                direction="input",
                sample_rate=44_100,
                channels=2,
            ),
        )

        assert MainWindow._current_device_route_key(owner) == route_key
        assert MainWindow._calibration_context_key(owner) != context_before
    finally:
        owner.input_combo.deleteLater()
        owner.output_combo.deleteLater()
        qapp.processEvents()


def _fake_audio_device(name: str, endpoint_id: str, direction: str, **formats: int):
    return SimpleNamespace(
        name=name,
        is_default=True,
        endpoint_id=endpoint_id,
        host_api="WASAPI",
        sample_rate=formats.get("sample_rate", 48_000),
        channels=formats.get("channels", 1 if direction == "input" else 2),
        name_ordinal=0,
    )


def test_refresh_reapplies_latency_and_route_preset_after_reconnect(
    qapp, monkeypatch, tmp_path
) -> None:
    from mic_eq.config_parts import shared
    from mic_eq.ui import main_window

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    devices = {
        "input": [_fake_audio_device("Mic A", "input-a", "input")],
        "output": [_fake_audio_device("Cable A", "output-a", "output")],
    }
    monkeypatch.setattr(main_window, "list_input_devices", lambda: devices["input"])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: devices["output"])

    window = MainWindow()
    try:
        window._on_device_changed()
        window._apply_latency_compensation_for_current_devices = Mock()
        window._apply_bound_preset_for_current_route = Mock(return_value=False)
        devices["input"] = [_fake_audio_device("Mic B", "input-b", "input")]
        window._refresh_devices()

        window._apply_latency_compensation_for_current_devices.assert_called_once()
        window._apply_bound_preset_for_current_route.assert_called_once()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_refresh_reapplies_latency_when_endpoint_format_changes(
    qapp, monkeypatch, tmp_path
) -> None:
    from mic_eq.config_parts import shared
    from mic_eq.ui import main_window

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    devices = {
        "input": [_fake_audio_device("Mic", "input-route", "input")],
        "output": [_fake_audio_device("Cable", "output-route", "output")],
    }
    monkeypatch.setattr(main_window, "list_input_devices", lambda: devices["input"])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: devices["output"])

    window = MainWindow()
    try:
        window._on_device_changed()
        window._apply_latency_compensation_for_current_devices = Mock()
        window._apply_bound_preset_for_current_route = Mock(return_value=False)
        devices["input"] = [
            _fake_audio_device(
                "Mic", "input-route", "input", sample_rate=44_100, channels=2
            )
        ]
        window._refresh_devices()

        window._apply_latency_compensation_for_current_devices.assert_called_once()
        window._apply_bound_preset_for_current_route.assert_not_called()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_device_and_mute_controls_do_not_create_processing_history_entries(
    qapp, monkeypatch, tmp_path
) -> None:
    from mic_eq.config_parts import shared
    from mic_eq.ui import main_window

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [])

    window = MainWindow()
    try:
        window.preset_modified = False
        mode_index = window.input_channel_mode_combo.findData("left")
        window.input_channel_mode_combo.setCurrentIndex(mode_index)
        qapp.processEvents()
        assert window.preset_modified is False

        window.user_mute_checkbox.setChecked(True)
        qapp.processEvents()
        assert window.preset_modified is False
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_user_mute_survives_temporary_owner_release() -> None:
    calls: list[bool] = []
    owner = cast(Any, MainWindow.__new__(MainWindow))
    owner.processor = SimpleNamespace(set_output_mute=calls.append)
    owner.user_muted = True
    owner._temporary_mute_reasons = set()
    owner._update_session_summary = lambda: None

    MainWindow.set_temporary_output_mute(owner, True, "calibration")
    MainWindow.set_temporary_output_mute(owner, False, "calibration")

    assert calls == [True, True]


def test_failed_mute_application_stays_pending() -> None:
    owner = cast(Any, MainWindow.__new__(MainWindow))

    def fail(_muted: bool) -> None:
        raise RuntimeError("processor unavailable")

    owner.processor = SimpleNamespace(set_output_mute=fail)
    owner.user_muted = True
    owner._temporary_mute_reasons = set()
    owner._update_session_summary = lambda: None

    MainWindow._apply_output_mute(owner)

    assert owner._output_mute_error == "unavailable"


def test_eq_only_scope_preserves_the_rest_of_the_processing_chain() -> None:
    owner = cast(Any, MainWindow.__new__(MainWindow))
    applied_eq: dict[str, object] = {}
    owner.eq_panel = SimpleNamespace(set_settings=applied_eq.update)
    owner.status_bar = SimpleNamespace(showMessage=lambda *args: None)
    owner._history_ready = False
    owner._history_replaying = False
    owner.preset_modified = False
    owner._update_session_summary = lambda: None
    owner._set_preset_modified = lambda modified=None: setattr(owner, "preset_modified", modified)

    before = {"threshold_db": -31.0, "model": "rnnoise"}
    owner.gate_panel = SimpleNamespace(get_settings=lambda: dict(before))
    owner.rnnoise_checkbox = SimpleNamespace(isChecked=lambda: True)
    owner.strength_slider = SimpleNamespace(value=lambda: 100)
    owner.model_combo = SimpleNamespace(currentData=lambda: "rnnoise")
    owner.deesser_panel = SimpleNamespace(get_settings=lambda: {})
    owner.compressor_panel = SimpleNamespace(
        get_compressor_settings=lambda: {}, get_limiter_settings=lambda: {}
    )
    owner.bypass_checkbox = SimpleNamespace(isChecked=lambda: False)

    replacement = Preset()
    replacement.eq.enabled = False
    MainWindow._apply_preset(owner, replacement, scope="eq")

    assert owner.gate_panel.get_settings() == before
    assert applied_eq == replacement.eq.to_dict()
    assert owner.preset_modified is True


def test_new_input_endpoint_does_not_inherit_another_device_preference() -> None:
    old_identity = DeviceIdentity(
        name="Mic",
        endpoint_id="old-endpoint",
        direction="input",
    )
    new_identity = DeviceIdentity(
        name="Mic",
        endpoint_id="new-endpoint",
        direction="input",
    )
    owner = cast(Any, MainWindow.__new__(MainWindow))
    owner.config = AppConfig(
        input_device_preferences={
            build_input_device_preference_key(old_identity): InputDevicePreference(
                "left", "gentle"
            )
        }
    )
    owner._current_device_route_key = lambda: None
    owner._current_input_preference_key = lambda: build_input_device_preference_key(
        new_identity
    )

    assert MainWindow._input_preference_for_current_route(owner) == InputDevicePreference()


def test_onboarding_route_check_stops_for_user_mute(qapp, monkeypatch) -> None:
    monkeypatch.setattr(
        "mic_eq.ui.first_run_setup_dialog.save_config", lambda _config: None
    )

    class Owner(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.config = AppConfig(
                first_run_setup_state="in_progress",
                first_run_setup_step="route",
                first_run_setup_steps={
                    "devices": "completed",
                    "route": "pending",
                    "latency": "pending",
                    "voice": "pending",
                },
            )
            self.user_muted = True
            self.start_calls = 0
            self.input_combo = QComboBox()
            self.input_combo.addItem(
                "Mic", DeviceIdentity(name="Mic", endpoint_id="in", direction="input")
            )
            self.output_combo = QComboBox()
            self.output_combo.addItem(
                "Cable",
                DeviceIdentity(name="Cable", endpoint_id="out", direction="output"),
            )
            self.processor = SimpleNamespace(is_running=lambda: False)

        def _start_processing(self) -> None:
            self.start_calls += 1

    owner = Owner()
    dialog = FirstRunSetupDialog(owner)
    try:
        dialog._run_current_step()
        assert owner.start_calls == 0
        assert "Output is muted" in dialog.status_label.text()
    finally:
        dialog.close()
        owner.deleteLater()
        qapp.processEvents()


def test_real_eq_only_apply_changes_eq_and_preserves_other_sections(
    qapp, monkeypatch, tmp_path
) -> None:
    from mic_eq.config_parts import shared
    from mic_eq.ui import main_window

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [])

    window = MainWindow()
    try:
        before = window._get_current_preset()
        replacement = Preset()
        eq_payload = replacement.eq.to_dict()
        eq_payload["enabled"] = not before.eq.enabled
        eq_payload["bands"][0]["gain_db"] = before.eq.bands[0].gain_db + 2.0
        replacement.eq = EQSettings.from_dict(eq_payload)

        MainWindow._apply_preset(window, replacement, scope="eq")

        after = window._get_current_preset()
        assert after.eq.enabled is replacement.eq.enabled
        assert after.eq.bands[0].gain_db == replacement.eq.bands[0].gain_db
        assert after.gate == before.gate
        assert after.rnnoise == before.rnnoise
        assert after.deesser == before.deesser
        assert after.compressor == before.compressor
        assert after.limiter == before.limiter
        assert after.bypass == before.bypass
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_complete_catalogues_have_scope_descriptions_and_resolved_sections() -> None:
    from mic_eq.config import BUILTIN_PRESETS, TARGET_CURVES

    assert all(
        preset.description.startswith("Complete sound preset:")
        for preset in BUILTIN_PRESETS.values()
    )
    assert all(
        curve.description.startswith("Calibration target")
        for curve in TARGET_CURVES.values()
    )
    assert all(
        set(preset.to_dict())
        >= {"gate", "eq", "rnnoise", "deesser", "compressor", "limiter"}
        for preset in BUILTIN_PRESETS.values()
    )
