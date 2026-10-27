"""Bounded processing-configuration undo/redo contracts."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

import json

import numpy as np
import pytest

from mic_eq.config import AppConfig, Preset, build_eq_candidate_settings
from mic_eq.config_parts.shared import PresetValidationError
from mic_eq.ui.config_history import (
    BoundedConfigurationHistory,
    ConfigurationSnapshot,
    changed_configuration_paths,
    explicit_provenance_after_edit,
)
from mic_eq.ui.main_window import MainWindow
from mic_eq.ui.calibration_dialog import CalibrationDialog, _candidate_metadata
from mic_eq.ui.calibration_support import chain_settings as _chain_settings
import test_quick_shell as quick_shell
from test_eq_editing_state import eq_control, tone_action
from test_gate_controls_state import window_gate_control

quick_window = quick_shell.quick_window


def _snapshot(
    threshold_db: float,
    *,
    label: str,
    source: str = "test",
) -> ConfigurationSnapshot:
    preset = Preset()
    preset.gate.threshold_db = threshold_db
    return ConfigurationSnapshot.from_preset(
        preset,
        label=label,
        source=source,
    )


def test_undo_redo_presets_auto_eq_and_branching_are_transactional() -> None:
    history = BoundedConfigurationHistory(limit=8)
    baseline = _snapshot(-40.0, label="Baseline")
    manual = _snapshot(-38.0, label="Manual edit", source="ui")
    preset = _snapshot(-34.0, label="Loaded preset", source="preset")
    auto_eq = _snapshot(-31.0, label="Auto-EQ", source="auto_eq")
    history.initialize(baseline)
    assert history.record(manual)
    assert history.record(preset)
    assert history.record(auto_eq)

    restored: list[float] = []

    def restore(snapshot: ConfigurationSnapshot) -> None:
        restored.append(snapshot.to_preset().gate.threshold_db)

    assert history.undo(restore) == preset
    assert history.undo(restore) == manual
    assert history.redo(restore) == preset
    assert restored == [-34.0, -38.0, -34.0]

    branch = _snapshot(-29.0, label="Branch", source="ui")
    assert history.record(branch)
    assert history.can_redo is False
    assert history.current == branch


def test_history_is_bounded_and_deduplicates_identical_payloads() -> None:
    history = BoundedConfigurationHistory(limit=3)
    history.initialize(_snapshot(-40.0, label="0"))
    assert history.record(_snapshot(-40.0, label="duplicate")) is False
    for index, threshold in enumerate((-39.0, -38.0, -37.0), start=1):
        history.record(_snapshot(threshold, label=str(index)))

    assert history.size == 3
    assert history.cursor == 2
    assert history.current is not None
    assert history.current.to_preset().gate.threshold_db == -37.0


def test_history_keeps_transient_calibration_outside_preset_payload():
    history = BoundedConfigurationHistory()
    before = ConfigurationSnapshot.from_preset(
        Preset(), label="Before", source="test",
        noise_reference_reliability=0.7, calibration_context_key="route-a",
    )
    after = ConfigurationSnapshot.from_preset(
        Preset(), label="After", source="test",
        noise_reference_reliability=0.2, calibration_context_key="route-a",
    )
    assert before.payload_json == after.payload_json
    assert "noise_reference_reliability" not in before.to_preset().to_dict()["compressor"]
    history.initialize(before)
    assert history.record(after)
    restored = []
    history.undo(lambda snapshot: restored.append(snapshot.noise_reference_reliability))
    assert restored == [0.7]
    with pytest.raises(ValueError, match="reliability"):
        ConfigurationSnapshot.from_preset(
            Preset(), label="Invalid", source="test",
            noise_reference_reliability=float("nan"),
        )


def _on_route(threshold_db: float, route: str | None, reliability: float = 0.0, label: str = "edit"):
    """A snapshot of the gate threshold as seen on ``route`` with the live ``reliability``."""
    preset = Preset()
    preset.gate.threshold_db = threshold_db
    return ConfigurationSnapshot.from_preset(
        preset, label=label, source="test",
        noise_reference_reliability=reliability, calibration_context_key=route,
    )


def _restored_reliability(history: BoundedConfigurationHistory, route: str) -> float:
    """What undo would put back live on ``route``, as the window restores it."""
    seen: list[float] = []
    history.undo(lambda target: seen.append(
        target.noise_reference_reliability
        if target.calibration_context_key == route else 0.0
    ))
    return seen[0]


def test_routing_back_to_a_calibrated_route_is_not_an_entry_and_undo_restores_it() -> None:
    history = BoundedConfigurationHistory()
    history.initialize(_on_route(-40.0, "K1", 0.7))
    assert history.record(_on_route(-38.0, "K1", 0.7, "E1"))
    assert not history.record(_on_route(-38.0, "K2", 0.0))  # away, observed
    assert history.record(_on_route(-36.0, "K2", 0.0, "E2"))
    # Back on K1 the live reliability returns without any edit.
    assert not history.record(_on_route(-36.0, "K1", 0.7))
    assert history.size == 3

    # One press reaches the earlier edit, with its calibration back on the route it belongs to.
    assert _restored_reliability(history, "K1") == 0.7
    assert history.size == 3 and history.current is not None
    assert history.current.label == "E1"


def test_a_route_that_drops_out_and_comes_back_is_not_an_entry() -> None:
    history = BoundedConfigurationHistory()
    history.initialize(_on_route(-40.0, "K1", 0.7))
    # A device briefly gone leaves no route key; coming back returns the saved calibration.
    assert not history.record(_on_route(-40.0, None, 0.0))
    assert not history.record(_on_route(-40.0, "K1", 0.7))
    assert history.size == 1


def test_clearing_a_calibration_on_the_same_route_is_one_entry() -> None:
    history = BoundedConfigurationHistory()
    history.initialize(_on_route(-40.0, "K1", 0.7))
    assert history.record(_on_route(-40.0, "K1", 0.0))
    assert history.size == 2


def test_a_recalibration_matching_the_old_routes_reliability_is_one_entry() -> None:
    history = BoundedConfigurationHistory()
    history.initialize(_on_route(-40.0, "K1", 0.7))
    assert not history.record(_on_route(-40.0, "K2", 0.0))
    assert history.record(_on_route(-40.0, "K2", 0.7))
    assert history.size == 2


def test_the_redo_branch_survives_a_route_change() -> None:
    history = BoundedConfigurationHistory()
    history.initialize(_on_route(-40.0, "K1"))
    assert history.record(_on_route(-38.0, "K1"))
    history.undo(lambda _target: None)
    assert history.can_redo

    assert not history.record(_on_route(-40.0, "K2"))

    assert history.can_redo and history.size == 2


def test_failed_restore_does_not_move_history_cursor() -> None:
    history = BoundedConfigurationHistory()
    history.initialize(_snapshot(-40.0, label="baseline"))
    history.record(_snapshot(-35.0, label="change"))
    cursor = history.cursor

    def fail(_snapshot: ConfigurationSnapshot) -> None:
        raise RuntimeError("restore failed")

    with pytest.raises(RuntimeError, match="restore failed"):
        history.undo(fail)

    assert history.cursor == cursor
    assert history.current is not None
    assert history.current.to_preset().gate.threshold_db == -35.0


def test_malformed_snapshot_is_rejected_without_corrupting_history() -> None:
    history = BoundedConfigurationHistory()
    baseline = _snapshot(-40.0, label="baseline")
    history.initialize(baseline)
    malformed = ConfigurationSnapshot(
        '{"gate":{"threshold_db":"bad"}}',
        "malformed",
        "test",
    )

    with pytest.raises(PresetValidationError, match="threshold_db"):
        history.record(malformed)

    assert history.size == 1
    assert history.current == baseline


def test_snapshot_stores_validated_canonical_legacy_values():
    preset = Preset()
    preset.version = "1.12.1"
    preset.gate.vad_threshold = 0.73

    snapshot = ConfigurationSnapshot.from_preset(
        preset, label="migrated", source="preset"
    )

    assert snapshot.to_preset().gate.vad_threshold == pytest.approx(0.7)
    assert snapshot.payload()["gate"]["vad_threshold"] == pytest.approx(0.7)


def test_migration_provenance_survives_and_only_changed_path_becomes_explicit() -> None:
    preset = Preset()
    payload = preset.to_dict()
    payload["value_provenance"]["gate.vad_threshold"] = "migration_default"
    migrated = Preset.from_dict(payload)
    previous = ConfigurationSnapshot.from_preset(
        migrated,
        label="migrated",
        source="preset",
    )
    edited = previous.to_preset()
    edited.gate.threshold_db = -33.0

    provenance = explicit_provenance_after_edit(previous, edited)

    assert provenance["gate.vad_threshold"] == "migration_default"
    assert provenance["gate.threshold_db"] == "explicit"


def test_changed_paths_and_snapshot_schema_exclude_audio_and_runtime_state() -> None:
    previous = _snapshot(-40.0, label="before")
    current = _snapshot(-35.0, label="after")
    paths = changed_configuration_paths(previous.payload(), current.payload())
    payload = json.loads(current.payload_json)

    assert paths == {"gate.threshold_db"}
    assert set(payload) == {
        "name",
        "description",
        "version",
        "gate",
        "eq",
        "rnnoise",
        "deesser",
        "compressor",
        "limiter",
        "bypass",
        "value_provenance",
    }


@pytest.fixture
def window(quick_window):
    """The real Quick window."""
    return quick_window


def test_main_window_wires_manual_preset_auto_eq_undo_and_redo(
    window,
    qapp,
    monkeypatch,
) -> None:
    window._prompt_save_current_preset = lambda **_kwargs: None
    monkeypatch.setattr(window, "_calibration_context_key", lambda: "test-route/48000/mono")
    monkeypatch.setattr("mic_eq.ui.calibration_history.save_config", lambda _config: True)
    threshold = window_gate_control(window, "Threshold")
    eq = window.eq_state
    baseline = threshold.value

    threshold.setValue(baseline + 2.0)
    qapp.processEvents()

    assert window._commit_pending_configuration_snapshot()
    assert window._configuration_history.can_undo

    preset = window._get_current_preset()
    preset.gate.threshold_db = baseline + 5.0
    preset.eq.enabled = False
    window._apply_preset(preset)
    assert threshold.value == pytest.approx(baseline + 5.0)

    auto_eq_bands = [
        (80.0 * (1.6**index), 1.0 if index == 4 else 0.0, 1.41)
        for index in range(10)
    ]
    dialog = CalibrationDialog(window)
    dialog.recording_state = "analyzing"
    dialog._candidate_target_metadata = ("broadcast", "adaptive", "conservative")
    dialog.audio_data = np.zeros(16, dtype=np.float32)
    dialog.preview_audio_data = dialog.audio_data.copy()
    dialog._analysis_generation = 1
    incumbent_preset = window._get_current_preset()
    dialog._analysis_incumbent_preset = incumbent_preset
    dialog._analysis_headroom_chain = _chain_settings(
        window,
        full_chain=True,
        input_pre_filtered=False,
        preset=incumbent_preset,
    )
    dialog._candidate_metadata = _candidate_metadata(
        "eq_only",
        target={"curve": "broadcast", "mode": "adaptive", "smoothing": "conservative"},
        capture={
            "generation": 1,
            "sample_rate": int(window.processor.sample_rate()),
            "sample_count": 16,
            "context_key": window._calibration_context_key(),
        },
        options={"target_mode": "adaptive", "smoothing_strength": "conservative"},
        allowed_scope=("eq",),
    )
    monkeypatch.setattr(
        "mic_eq.ui.calibration_dialog.QMessageBox.critical",
        lambda _parent, _title, message: pytest.fail(message),
    )
    dialog.auto_eq_applied.connect(window.on_auto_eq_applied)
    candidate_eq = build_eq_candidate_settings(
        incumbent_preset.eq,
        [band[0] for band in auto_eq_bands],
        [band[1] for band in auto_eq_bands],
        [band[2] for band in auto_eq_bands],
        enabled=True,
    )
    dialog._on_analysis_complete({
        "band_freqs": [band[0] for band in auto_eq_bands],
        "band_gains": [band[1] for band in auto_eq_bands],
        "band_qs": [band[2] for band in auto_eq_bands],
        "enabled": True,
        "apply_recommended": True,
        "validated_candidate_eq": candidate_eq.to_dict(),
        "headroom_validation": {
            "safe": True,
            "authoritative": True,
            "advisory": False,
            "status": "safe",
        },
    })
    dialog._on_start_clicked()
    correction = eq.get_eq_settings().correction_bands
    assert correction is not None and correction[4].gain_db == 1.0
    assert eq_control(window, "gain", 4).value == 0
    assert window.processor.is_eq_enabled()
    assert eq.auto_eq_diagnostics is not None

    window.undo_configuration()
    assert eq_control(window, "gain", 4).value == 0
    assert eq.get_eq_settings().correction_bands is None
    assert not window.processor.is_eq_enabled()
    assert threshold.value == pytest.approx(baseline + 5.0)
    assert window.status_message.text == "Undid: Auto-EQ (Broadcast)"
    window.redo_configuration()
    assert eq.get_eq_settings().correction_bands == correction
    assert eq_control(window, "gain", 4).value == 0
    assert window.processor.is_eq_enabled()
    assert window.status_message.text == "Redid: Auto-EQ (Broadcast)"

    history_size = window._configuration_history.size
    previous_frequency = eq.get_band(4).frequency_hz
    eq.begin_curve_edit(4)
    eq.edit_curve_band(4, 3000.0, 2.2)
    eq.finish_curve_edit(4, 3000.0, 2.2)
    assert window._configuration_history.size == history_size + 1
    assert eq.get_band(4).frequency_hz == 3000.0
    window.undo_configuration()
    assert eq.get_band(4).frequency_hz == previous_frequency

    # A new edit must invalidate the redo branch even when its debounce timer
    # has not fired yet.
    assert window._configuration_history.can_redo
    fresh_threshold = baseline + 7.0
    threshold.setValue(fresh_threshold)
    window.redo_configuration()
    assert threshold.value == pytest.approx(fresh_threshold)
    assert not window._configuration_history.can_redo


def test_domain_edits_update_history_and_ignore_presentation_controls(window, qapp):
    baseline = window._preset_payload(window._get_current_preset())

    tone_action(window, "Voice").trigger()
    assert window.preset_modified is True
    window._commit_pending_configuration_snapshot()
    assert window._configuration_history.can_undo
    window.undo_configuration()
    assert window._preset_payload(window._get_current_preset()) == baseline

    window.processing_mode_choice.setCurrentIndex(
        window.processing_mode_choice.findData("raw")
    )
    assert window.preset_modified is True
    window._commit_pending_configuration_snapshot()
    assert window._configuration_history.can_undo
    assert window.preset_modified is True
    window.undo_configuration()
    assert window._processing_mode() == "normal"
    assert window.preset_modified is False

    history_size = window._configuration_history.size
    window.shell_page.setIndex(window.HEALTH_PAGE_INDEX)
    qapp.processEvents()
    assert not window._commit_pending_configuration_snapshot()
    assert window._configuration_history.size == history_size


def test_main_window_failed_history_restore_rolls_back_partial_state(
    window,
    monkeypatch,
) -> None:
    window.compressor_state.set_settings({"noise_reference_reliability": 0.7})
    original_threshold = window.gate_state.get_settings()["threshold_db"]
    target = window._get_current_preset()
    target.gate.threshold_db = original_threshold + 6.0
    snapshot = ConfigurationSnapshot.from_preset(
        target,
        label="failing restore",
        source="test",
    )
    real_apply = window._write_processing_configuration
    call_count = 0

    def fail_after_partial_apply(preset: Preset, *args, **kwargs) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            window.gate_state.set_value("threshold_db", preset.gate.threshold_db)
            raise RuntimeError("simulated native restore failure")
        real_apply(preset, *args, **kwargs)

    monkeypatch.setattr(window, "_write_processing_configuration", fail_after_partial_apply)
    with pytest.raises(RuntimeError, match="simulated native restore failure"):
        window._restore_configuration_snapshot(snapshot)

    assert call_count == 2
    assert window.gate_state.get_settings()["threshold_db"] == pytest.approx(original_threshold)
    assert window._history_replaying is False
    assert window.compressor_state.get_settings(include_calibration=True)[
        "noise_reference_reliability"
    ] == pytest.approx(0.7)


def test_history_restores_calibration_only_in_the_same_capture_context(window, monkeypatch):
    route = "route-a"
    monkeypatch.setattr(window, "_current_device_route_key", lambda: route)

    def reliability() -> float:
        return window.compressor_state.get_settings(include_calibration=True)[
            "noise_reference_reliability"
        ]

    # The route resolving is context, not an edit; the calibration landing on it is one.
    assert not window._commit_pending_configuration_snapshot()
    window.compressor_state.set_settings({"noise_reference_reliability": 0.7})
    assert window._commit_pending_configuration_snapshot()
    snapshot = window._configuration_history.current
    assert snapshot is not None
    window.compressor_state.set_settings({"noise_reference_reliability": 0.2})
    window._restore_configuration_snapshot(snapshot)
    assert reliability() == pytest.approx(0.7)
    for change in ("channel", "cleanup", "route"):
        window.config.input_channel_mode = "phase_safe_mono"
        window.config.input_cleanup_mode = "off"
        if change == "channel":
            window.config.input_channel_mode = "left"
        elif change == "cleanup":
            window.config.input_cleanup_mode = "gentle"
        else:
            route = "route-b"
        window._restore_configuration_snapshot(snapshot)
        assert reliability() == 0.0


def _without_noise_backend(window, model: str) -> None:
    """List the backends the processor offers without ``model``."""
    state = window.noise_suppression_state
    state.models = tuple(entry for entry in state.models if entry[0] != model)


def test_history_restore_refuses_an_unavailable_noise_backend(window) -> None:
    previous = window._get_current_preset()
    target = window._get_current_preset()
    target.rnnoise.model = "deepfilter"
    snapshot = ConfigurationSnapshot.from_preset(
        target,
        label="DeepFilter target",
        source="test",
    )
    _without_noise_backend(window, "deepfilter")

    with pytest.raises(RuntimeError, match="unavailable"):
        window._restore_configuration_snapshot(snapshot)

    assert window.noise_suppression_state.get_settings()["model"] == previous.rnnoise.model
    assert window._history_replaying is False


def test_normal_preset_load_retains_sound_when_noise_backend_is_absent(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "critical", lambda *_: None)
    previous = window._get_current_preset()
    preset = window._get_current_preset()
    preset.rnnoise.model = "deepfilter"
    _without_noise_backend(window, "deepfilter")
    assert window._apply_preset(preset) is False
    assert window._preset_payload(window._get_current_preset()) == window._preset_payload(previous)
    assert window.current_preset_name == "Default"


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_failed_configuration_restores_sound_or_stops_muted(window, monkeypatch, rollback_fails):
    window.user_mute_checkbox_action.setChecked(True)
    window.apply_processing_configuration(
        window._get_current_preset(),
        noise_reference_reliability=0.7,
        processing_mode="raw",
        compressor_metadata={"dynamics_profile": "dense"},
    )
    metadata = window.compressor_state.get_settings(include_calibration=True)
    assert metadata["noise_reference_reliability"] == 0.7
    assert metadata["dynamics_profile"] == "dense"
    before = window._preset_payload(window._get_current_preset())
    identity = (window.current_preset_name, window.current_preset_path, window.config.last_preset)
    cursor = window._configuration_history.cursor
    target = window._get_current_preset()
    target.gate.threshold_db = -25.0
    original_set = window.eq_state.set_settings
    calls = 0

    def fail_after_eq(settings):
        nonlocal calls
        calls += 1
        original_set(settings)
        if calls == 1 or rollback_fails:
            raise RuntimeError("injected failure after EQ")

    monkeypatch.setattr(window.eq_state, "set_settings", fail_after_eq)
    with pytest.raises(RuntimeError, match="injected failure after EQ"):
        window.apply_processing_configuration(target)
    assert (window.current_preset_name, window.current_preset_path, window.config.last_preset) == identity
    assert window._configuration_history.cursor == cursor
    assert window.user_muted
    if rollback_fails:
        assert not window.processor.is_running()
        assert "configuration" in window._temporary_mute_reasons
        assert "restoration failed" in str(window.status_message.text)
    else:
        assert window._preset_payload(window._get_current_preset()) == before
        assert "configuration" not in window._temporary_mute_reasons
        assert window._processing_mode() == "raw"
        assert window.compressor_state.get_settings(include_calibration=True) == metadata


def test_configuration_rollback_stop_failure_keeps_mute_and_reports_stop_failure(
    qapp, monkeypatch
):
    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
    for name in ("list_presets", "list_input_devices", "list_output_devices"):
        monkeypatch.setattr(f"mic_eq.ui.main_window.{name}", lambda: [])
    window = MainWindow()
    original_processor = window.processor

    class FailingStopProcessor:
        def is_running(self) -> bool:
            return True

        def set_output_mute(self, _muted: bool) -> None:
            return None

        def stop(self) -> None:
            raise RuntimeError("stop unavailable")

    try:
        monkeypatch.setattr(window, "processor", FailingStopProcessor())

        def fail_configuration_write(_preset):
            raise RuntimeError("configuration write unavailable")

        monkeypatch.setattr(window, "_write_processing_configuration", fail_configuration_write)
        with pytest.raises(RuntimeError, match="restoration failed"):
            window.apply_processing_configuration(window._get_current_preset())
        assert "configuration" in window._temporary_mute_reasons
        assert "Stopping audio failed: stop unavailable" in str(window.status_message.text)
    finally:
        window.processor = original_processor
        window.meter_timer.stop()
        window.diagnostics_timer.stop()
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
        mode_index = window.input_channel_mode_choice.findData("left")
        window.input_channel_mode_choice.setCurrentIndex(mode_index)
        qapp.processEvents()
        assert window.preset_modified is False

        window.user_mute_checkbox_action.setChecked(True)
        qapp.processEvents()
        assert window.preset_modified is False
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def _load(window, name: str, path, *, gate_change: float):
    preset = window._get_current_preset()
    preset.name = name
    preset.gate.threshold_db += gate_change
    assert window._apply_preset(preset, preset_path=path)


def _identity(window):
    return (
        window.current_preset_name, window.current_preset_path, window.config.last_preset,
        window.preset_modified, window.shell_texts["preset_status_label"].text,
    )


def test_undoing_and_redoing_a_preset_load_brings_back_the_preset_that_went_with_it(
    window, tmp_path
) -> None:
    before = _identity(window)
    assert before[0] == "Default" and before[1] is None
    loud_path = tmp_path / "Loud.json"
    _load(window, "Loud", loud_path, gate_change=5.0)
    loaded = _identity(window)
    assert loaded[:4] == ("Loud", loud_path, str(loud_path), False)

    window.undo_configuration()
    assert _identity(window) == before

    window.redo_configuration()
    assert _identity(window) == loaded


def test_a_load_that_changes_no_sound_is_still_one_undo_step(window, tmp_path) -> None:
    before = _identity(window)
    _load(window, "Same Sound", tmp_path / "Same.json", gate_change=0.0)
    assert window._configuration_history.can_undo

    window.undo_configuration()

    assert _identity(window) == before


def test_undoing_an_edit_keeps_the_loaded_preset(window, tmp_path) -> None:
    _load(window, "Loud", tmp_path / "Loud.json", gate_change=5.0)
    window.gate_state.set_settings({"threshold_db": window.gate_state.get_settings()["threshold_db"] + 3})
    assert window._commit_pending_configuration_snapshot()
    assert window.preset_modified

    window.undo_configuration()

    assert window.current_preset_name == "Loud" and not window.preset_modified


def test_saving_a_loaded_preset_under_a_new_name_survives_undo_and_redo(window, tmp_path) -> None:
    loud_path = tmp_path / "Loud.json"
    _load(window, "Loud", loud_path, gate_change=5.0)
    renamed = window._get_current_preset()
    renamed.name = "Renamed"
    saved_path = window._save_preset_file(renamed)
    assert saved_path is not None and window.current_preset_name == "Renamed"

    window.undo_configuration()
    assert window.current_preset_name == "Default"
    window.redo_configuration()

    # The load was saved as Renamed; going back to it must not hand the next Save the old file.
    assert window.current_preset_name == "Renamed"
    assert window.current_preset_path == saved_path != loud_path
    assert not window.preset_modified
