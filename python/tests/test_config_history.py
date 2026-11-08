"""Bounded processing-configuration undo/redo contracts."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

import json

import numpy as np
import pytest

from mic_eq.config import Preset, build_eq_candidate_settings
from mic_eq.config_parts.shared import PresetValidationError
from mic_eq.ui.config_history import (
    BoundedConfigurationHistory,
    ConfigurationSnapshot,
    changed_configuration_paths,
    explicit_provenance_after_edit,
)
from mic_eq.ui.main_window import MainWindow
from mic_eq.ui.processing_configuration import PresetOrigin, preset_payload
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


@pytest.mark.parametrize(("records", "size"), [
    # A device briefly gone leaves no route key; coming back returns the saved calibration.
    ([(None, 0.0, False), ("K1", 0.7, False)], 1),
    ([("K1", 0.0, True)], 2),
    ([("K2", 0.0, False), ("K2", 0.7, True)], 2),
], ids=["route-drops-out-and-returns", "calibration-cleared-on-same-route", "recalibration-matches-old-route"])
def test_a_route_gap_is_not_an_entry_but_a_calibration_change_is(records, size) -> None:
    history = BoundedConfigurationHistory()
    history.initialize(_on_route(-40.0, "K1", 0.7))
    for route, reliability, recorded in records:
        assert history.record(_on_route(-40.0, route, reliability)) is recorded
    assert history.size == size


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
    configuration = window.configuration
    threshold = window_gate_control(window, "Threshold")
    eq = window.eq_state
    baseline = threshold.value

    threshold.setValue(baseline + 2.0)
    qapp.processEvents()

    assert configuration.commit_edit()
    assert configuration.history.can_undo

    preset = configuration.current
    preset.gate.threshold_db = baseline + 5.0
    preset.eq.enabled = False
    window._report_load(configuration.load(preset, PresetOrigin(), remember=True))
    assert threshold.value == pytest.approx(baseline + 5.0)

    auto_eq_bands = [
        (80.0 * (1.6**index), 1.0 if index == 4 else 0.0, 1.41)
        for index in range(10)
    ]
    dialog = CalibrationDialog(window, configuration=configuration)
    dialog.recording_state = "analyzing"
    dialog._candidate_target_metadata = ("broadcast", "adaptive", "conservative")
    dialog.audio_data = np.zeros(16, dtype=np.float32)
    dialog.preview_audio_data = dialog.audio_data.copy()
    dialog._analysis_generation = 1
    incumbent_preset = configuration.current
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


def test_domain_edits_update_history_and_ignore_presentation_controls(window, qapp):
    configuration = window.configuration
    baseline = preset_payload(configuration.current)

    tone_action(window, "Voice").trigger()
    assert configuration.identity.modified is True
    configuration.commit_edit()
    assert configuration.history.can_undo
    window.undo_configuration()
    assert preset_payload(configuration.current) == baseline

    window.processing_mode_choice.setCurrentIndex(
        window.processing_mode_choice.findData("raw")
    )
    assert configuration.identity.modified is True
    configuration.commit_edit()
    assert configuration.history.can_undo
    window.undo_configuration()
    assert window._processing_mode() == "normal"
    assert configuration.identity.modified is False

    history_size = configuration.history.size
    window.shell_page.setIndex(window.HEALTH_PAGE_INDEX)
    qapp.processEvents()
    assert not configuration.commit_edit()
    assert configuration.history.size == history_size


def test_the_calibration_context_follows_route_channel_and_cleanup(window, monkeypatch):
    route = ["route-a"]
    monkeypatch.setattr(window, "_current_device_route_key", lambda: route[0])
    baseline = window._calibration_context_key()
    assert baseline is not None
    for change in ("channel", "cleanup", "route"):
        window.config.input_channel_mode = "left" if change == "channel" else "phase_safe_mono"
        window.config.input_cleanup_mode = "gentle" if change == "cleanup" else "off"
        route[0] = "route-b" if change == "route" else "route-a"
        assert window._calibration_context_key() != baseline


def test_a_preset_the_engine_refuses_is_reported_and_the_sound_kept(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    shown = []
    monkeypatch.setattr(QMessageBox, "critical", lambda _parent, title, text: shown.append((title, text)))
    previous = preset_payload(window.configuration.current)
    preset = window.configuration.current
    preset.rnnoise.model = "deepfilter"
    window.noise_suppression_state.models = tuple(
        entry for entry in window.noise_suppression_state.models if entry[0] != "deepfilter"
    )
    assert not window._report_load(window.configuration.load(preset, PresetOrigin(), remember=True))
    assert shown and shown[0][0] == "Preset Not Applied" and "unavailable" in shown[0][1]
    assert preset_payload(window.configuration.current) == previous


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
        assert window.configuration.identity.modified is False
        mode_index = window.input_channel_mode_choice.findData("left")
        window.input_channel_mode_choice.setCurrentIndex(mode_index)
        qapp.processEvents()
        assert window.configuration.identity.modified is False

        window.user_mute_checkbox_action.setChecked(True)
        qapp.processEvents()
        assert window.configuration.identity.modified is False
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_the_preset_label_follows_a_load_and_its_undo_and_redo(window, tmp_path) -> None:
    label = window.shell_texts["preset_status_label"]
    before = label.text
    assert before == "Preset: Default (saved)"
    preset = window.configuration.current
    preset.name = "Loud"
    preset.gate.threshold_db += 5.0
    assert window._report_load(
        window.configuration.load(preset, PresetOrigin(path=tmp_path / "Loud.json"), remember=True)
    )
    assert label.text == "Preset: Loud (saved)"

    window.undo_configuration()
    assert label.text == before

    window.redo_configuration()
    assert label.text == "Preset: Loud (saved)"
