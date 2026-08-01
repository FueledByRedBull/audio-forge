"""Bounded processing-configuration undo/redo contracts."""

from __future__ import annotations

import json

import pytest

from mic_eq.config import AppConfig, Preset
from mic_eq.ui.config_history import (
    BoundedConfigurationHistory,
    ConfigurationSnapshot,
    changed_configuration_paths,
    explicit_provenance_after_edit,
)
from mic_eq.ui.main_window import MainWindow


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

    with pytest.raises(Exception):
        history.record(malformed)

    assert history.size == 1
    assert history.current == baseline


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


def test_main_window_wires_manual_preset_auto_eq_undo_and_redo(
    qapp,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "mic_eq.ui.main_window.load_config",
        AppConfig,
    )
    monkeypatch.setattr(
        "mic_eq.ui.main_window.save_config",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "mic_eq.ui.main_window.list_presets",
        lambda: [],
    )
    monkeypatch.setattr(
        "mic_eq.ui.main_window.list_input_devices",
        lambda: [],
    )
    monkeypatch.setattr(
        "mic_eq.ui.main_window.list_output_devices",
        lambda: [],
    )
    window = MainWindow()
    window._prompt_save_current_preset = lambda **_kwargs: None
    baseline = window.gate_panel.threshold_spinbox.value()

    window.gate_panel.threshold_spinbox.setValue(baseline + 2.0)
    qapp.processEvents()
    assert window._commit_pending_configuration_snapshot()
    assert window._configuration_history.can_undo

    preset = window._get_current_preset()
    preset.gate.threshold_db = baseline + 5.0
    window._apply_preset(preset)
    assert window.gate_panel.threshold_spinbox.value() == pytest.approx(
        baseline + 5.0
    )

    auto_eq_bands = [
        (80.0 * (1.6**index), 1.0 if index == 4 else 0.0, 1.41)
        for index in range(10)
    ]
    window.eq_panel.apply_auto_eq_results(auto_eq_bands)
    window.on_auto_eq_applied("broadcast")
    assert window.eq_panel.band_sliders[4].slider.value() == 10

    window.undo_configuration()
    assert window.eq_panel.band_sliders[4].slider.value() == 0
    assert window.gate_panel.threshold_spinbox.value() == pytest.approx(
        baseline + 5.0
    )
    assert window.status_bar.currentMessage() == "Undid: Auto-EQ (Broadcast)"
    window.redo_configuration()
    assert window.eq_panel.band_sliders[4].slider.value() == 10
    assert window.status_bar.currentMessage() == "Redid: Auto-EQ (Broadcast)"

    history_size = window._configuration_history.size
    previous_frequency = window.eq_panel.band_sliders[4].frequency_hz()
    window.eq_panel.configurationEditStarted.emit()
    window.eq_panel._apply_curve_band_edit(4, 3000.0, 2.2)
    window.eq_panel.configurationEditFinished.emit("EQ graph edit")
    assert window._configuration_history.size == history_size + 1
    assert window.eq_panel.band_sliders[4].frequency_hz() == 3000.0
    window.undo_configuration()
    assert window.eq_panel.band_sliders[4].frequency_hz() == previous_frequency

    # A new edit must invalidate the redo branch even when its debounce timer
    # has not fired yet.
    assert window._configuration_history.can_redo
    fresh_threshold = baseline + 7.0
    window.gate_panel.threshold_spinbox.setValue(fresh_threshold)
    window.redo_configuration()
    assert window.gate_panel.threshold_spinbox.value() == pytest.approx(
        fresh_threshold
    )
    assert not window._configuration_history.can_redo

    window.meter_timer.stop()
    window.diagnostics_timer.stop()
    window.close()
    window.deleteLater()
    qapp.processEvents()


def test_main_window_failed_history_restore_rolls_back_partial_state(
    qapp,
    monkeypatch,
) -> None:
    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: None)
    monkeypatch.setattr("mic_eq.ui.main_window.list_presets", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_input_devices", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_output_devices", lambda: [])
    window = MainWindow()
    original_threshold = window.gate_panel.threshold_spinbox.value()
    target = window._get_current_preset()
    target.gate.threshold_db = original_threshold + 6.0
    snapshot = ConfigurationSnapshot.from_preset(
        target,
        label="failing restore",
        source="test",
    )
    real_apply = window._apply_preset
    call_count = 0

    def fail_after_partial_apply(preset: Preset, *args, **kwargs) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            window.gate_panel.threshold_spinbox.setValue(
                preset.gate.threshold_db
            )
            raise RuntimeError("simulated native restore failure")
        real_apply(preset, *args, **kwargs)

    monkeypatch.setattr(window, "_apply_preset", fail_after_partial_apply)
    with pytest.raises(RuntimeError, match="simulated native restore failure"):
        window._restore_configuration_snapshot(snapshot)

    assert call_count == 2
    assert window.gate_panel.threshold_spinbox.value() == pytest.approx(
        original_threshold
    )
    assert window._history_replaying is False

    window.meter_timer.stop()
    window.diagnostics_timer.stop()
    window.close()
    window.deleteLater()
    qapp.processEvents()


def test_history_restore_refuses_an_unavailable_noise_backend(
    qapp,
    monkeypatch,
) -> None:
    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: None)
    monkeypatch.setattr("mic_eq.ui.main_window.list_presets", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_input_devices", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_output_devices", lambda: [])
    window = MainWindow()
    previous = window._get_current_preset()
    target = window._get_current_preset()
    target.rnnoise.model = "deepfilter"
    snapshot = ConfigurationSnapshot.from_preset(
        target,
        label="DeepFilter target",
        source="test",
    )
    while (model_index := window.model_combo.findData("deepfilter")) >= 0:
        window.model_combo.removeItem(model_index)

    with pytest.raises(RuntimeError, match="not present"):
        window._restore_configuration_snapshot(snapshot)

    assert window.model_combo.currentData() == previous.rnnoise.model
    assert window._history_replaying is False
    window.meter_timer.stop()
    window.diagnostics_timer.stop()
    window.close()
    window.deleteLater()
    qapp.processEvents()


def test_normal_preset_load_falls_back_when_noise_backend_is_absent(
    qapp,
    monkeypatch,
) -> None:
    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: None)
    monkeypatch.setattr("mic_eq.ui.main_window.list_presets", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_input_devices", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_output_devices", lambda: [])
    window = MainWindow()
    preset = window._get_current_preset()
    preset.rnnoise.model = "removed-backend"
    window.model_combo.addItem("Removed backend", "different-backend")
    window.model_combo.blockSignals(True)
    window.model_combo.setCurrentIndex(window.model_combo.count() - 1)
    window.model_combo.blockSignals(False)

    window._apply_preset(preset)

    assert window.model_combo.currentData() == "rnnoise"
    assert "using RNNoise" in window.status_bar.currentMessage()
    window.meter_timer.stop()
    window.diagnostics_timer.stop()
    window.close()
    window.deleteLater()
    qapp.processEvents()
