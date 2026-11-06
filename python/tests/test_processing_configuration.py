"""The processing configuration module at its interface: real stage states, fake ports."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from mic_eq import AudioProcessor
from mic_eq.config import BUILTIN_PRESETS, AppConfig, CompressorSettings, Preset, PresetValidationError
from mic_eq.ui import rate_limiter
from mic_eq.ui.compressor_state import CompressorState
from mic_eq.ui.deesser_state import DeEsserState
from mic_eq.ui.eq_state import EQState
from mic_eq.ui.gate_state import GateState
from mic_eq.ui.limiter_state import LimiterState
from mic_eq.ui.noise_suppression_state import NoiseSuppressionState
from mic_eq.ui.processing_configuration import (
    ConfigurationNotApplied,
    ConfigurationRestoreFailed,
    LoadStatus,
    PresetOrigin,
    ProcessingConfiguration,
    StageStates,
    preset_payload,
)


class FakeHost:
    def __init__(self) -> None:
        self.mode = "normal"
        self.held: set[str] = set()
        self.mute_works = True
        self.running = False
        self.context: str | None = "route-a"
        self.events: list[object] = []
        self.statuses: list[str] = []
        self.changes = 0

    def hold_mute(self, reason, on):
        (self.held.add if on else self.held.discard)(reason)
        return self.mute_works

    def mute_held(self, reason):
        return reason in self.held

    def audio_running(self):
        return self.running

    def stop_audio(self):
        self.events.append("stop")

    def processing_mode(self):
        return self.mode

    def set_processing_mode(self, mode):
        self.mode = mode

    def calibration_context_key(self):
        return self.context

    def sync_calibration_evidence(self, *, force_reset=False):
        self.events.append(("sync", force_reset))

    def show_status(self, text, timeout_ms):
        self.statuses.append(text)

    def configuration_changed(self):
        self.changes += 1


class FakeLibrary:
    def __init__(self) -> None:
        self.custom_presets: list[tuple[str, Path]] = []
        self.files: dict[Path, Preset | Exception] = {}
        self.reads: list[Path] = []

    def builtin(self, key):
        return BUILTIN_PRESETS.get(key)

    def custom(self):
        return list(self.custom_presets)

    def read(self, path):
        self.reads.append(path)
        content = self.files[path]
        if isinstance(content, Exception):
            raise content
        return Preset.from_dict(content.to_dict())

    def exists(self, path):
        return path in self.files


class FakeDebouncer:
    def __init__(self) -> None:
        self.pending = False

    def start(self):
        self.pending = True

    def stop(self):
        self.pending = False


class Harness:
    def __init__(self, config: AppConfig) -> None:
        processor = AudioProcessor()
        # Answers to "save your sound changes?", popped in order; an unexpected question pops
        # from an empty list and fails the test.
        self.states = StageStates(
            gate=GateState(processor),
            eq=EQState(processor),
            noise_suppression=NoiseSuppressionState(processor),
            deesser=DeEsserState(processor),
            compressor=CompressorState(processor),
            limiter=LimiterState(processor),
        )
        self.config = config
        self.host = FakeHost()
        self.library = FakeLibrary()
        self.debouncer = FakeDebouncer()
        self.prompts: list[bool] = []
        self.saves: list[str] = []
        self.save_works = True
        self.configuration = ProcessingConfiguration(
            self.states,
            config,
            host=self.host,
            library=self.library,
            confirm_unsaved=self.prompts.pop,
            debouncer=self.debouncer,
            save_config=self._save,
        )

    def _save(self) -> bool:
        self.saves.append(self.config.last_preset)
        return self.save_works

    def preset(self, name: str = "Custom", *, threshold_db: float | None = None) -> Preset:
        preset = self.configuration.current
        preset.name = name
        if threshold_db is not None:
            preset.gate.threshold_db = threshold_db
        return preset

    def threshold(self) -> float:
        return self.states.gate.get_settings()["threshold_db"]

    def reliability(self) -> float:
        return self.states.compressor.get_settings(include_calibration=True)["noise_reference_reliability"]


@pytest.fixture
def harness(qapp) -> Harness:
    return Harness(AppConfig())


@pytest.fixture
def started(harness) -> Harness:
    harness.configuration.start_history()
    return harness


def _fail_once_after_writing(monkeypatch, state, *, times: int = 1) -> None:
    """Make ``state`` accept a write and then report it refused, ``times`` times."""
    original = state.set_settings
    left = [times]

    def fail(settings):
        original(settings)
        if left[0]:
            left[0] -= 1
            raise RuntimeError("injected failure")

    monkeypatch.setattr(state, "set_settings", fail)


# Writing a configuration


def test_apply_writes_every_stage_under_the_configuration_mute(harness) -> None:
    preset = harness.preset()
    preset.gate.threshold_db = -31.237
    preset.deesser.auto_amount = 0.33789
    preset.compressor = CompressorSettings(
        threshold_db=-18.0, ratio=3.14159, attack_ms=8.0, release_ms=150.0, makeup_gain_db=2.0,
        adaptive_release=True, base_release_ms=75.0, auto_makeup_enabled=True, target_lufs=-16.0,
        sidechain_highpass_enabled=False,
    )
    preset.limiter.ceiling_db = -1.23456
    preset.limiter.careful_output_enabled = False

    harness.configuration.apply(
        preset, noise_reference_reliability=0.7, processing_mode="raw",
        compressor_metadata={"dynamics_profile": "dense"},
    )

    states = harness.states
    for name in ("gate", "deesser", "compressor", "limiter"):
        assert getattr(states, name).get_settings() == pytest.approx(asdict(getattr(preset, name)))
    assert states.eq.get_eq_settings().to_dict() == preset.eq.to_dict()
    calibration = harness.states.compressor.get_settings(include_calibration=True)
    assert (calibration["noise_reference_reliability"], calibration["dynamics_profile"]) == (0.7, "dense")
    assert harness.host.mode == "raw" and not harness.host.held
    assert harness.states.deesser.processor.get_deesser_auto_amount() == pytest.approx(0.33789)
    # Off-grid values survive into the history baseline too.
    harness.configuration.start_history()
    baseline = harness.configuration.history.current
    assert baseline is not None and baseline.to_preset().compressor.ratio == pytest.approx(3.14159)


@pytest.mark.parametrize("stage", ["eq", "noise_suppression"])
def test_a_refused_apply_restores_the_previous_sound_and_releases_the_mute(harness, monkeypatch, stage) -> None:
    configuration = harness.configuration
    configuration.apply(
        harness.preset(), noise_reference_reliability=0.7, processing_mode="raw",
        compressor_metadata={"dynamics_profile": "dense"},
    )
    before = preset_payload(configuration.current)
    calibration = harness.states.compressor.get_settings(include_calibration=True)
    _fail_once_after_writing(monkeypatch, getattr(harness.states, stage))

    with pytest.raises(ConfigurationNotApplied, match="previous sound restored: injected failure"):
        configuration.apply(harness.preset(threshold_db=-25.0))

    assert preset_payload(configuration.current) == before
    assert harness.host.mode == "raw" and not harness.host.held
    assert harness.states.compressor.get_settings(include_calibration=True) == calibration


def test_a_failed_restore_keeps_the_output_muted_and_stops_audio(harness, monkeypatch) -> None:
    _fail_once_after_writing(monkeypatch, harness.states.eq, times=2)

    with pytest.raises(ConfigurationRestoreFailed, match="restoration failed .*Audio is muted and stopping"):
        harness.configuration.apply(harness.preset(threshold_db=-25.0))

    assert harness.host.held == {"configuration"}
    assert harness.host.events == ["stop"]
    assert harness.host.statuses == ["Audio muted and stopping: configuration restoration failed"]


def test_a_refused_mute_on_running_audio_stops_it_and_writes_nothing(harness) -> None:
    harness.host.mute_works = False
    harness.host.running = True
    before = harness.threshold()

    with pytest.raises(ConfigurationNotApplied, match="configuration mute failed"):
        harness.configuration.apply(harness.preset(threshold_db=before + 4))

    assert harness.threshold() == before
    assert harness.host.events == ["stop"]


def test_an_unavailable_noise_backend_is_refused_before_any_other_stage_is_written(harness, monkeypatch) -> None:
    writes: list[str] = []
    for name in StageStates._fields:
        state = getattr(harness.states, name)
        monkeypatch.setattr(
            state, "set_settings", lambda settings, name=name, write=state.set_settings: (
                writes.append(name), write(settings),
            ),
        )
    preset = harness.preset()
    preset.rnnoise.model = "deepfilter"  # the test processor offers RNNoise only

    with pytest.raises(ConfigurationNotApplied, match="unavailable"):
        harness.configuration.apply(preset)

    # The refused model is the first write; the rollback that follows starts again from it.
    assert writes[:2] == ["noise_suppression", "noise_suppression"]
    assert harness.states.noise_suppression.get_settings()["model"] == "rnnoise"
    assert not harness.configuration.replaying


def test_checkpoint_and_restore_put_back_sound_mode_and_chosen_calibration(harness) -> None:
    configuration = harness.configuration
    configuration.apply(
        harness.preset(threshold_db=-35.0), noise_reference_reliability=0.8,
        compressor_metadata={"dynamics_profile": "dense"},
    )
    checkpoint = configuration.checkpoint()
    assert (checkpoint.context_key, checkpoint.noise_reference_reliability) == ("route-a", 0.8)

    # Without a calibration choice the host re-derives the evidence, which this fake host does not.
    for calibration_valid, reliability in ((None, 0.0), (True, 0.8), (False, 0.0)):
        configuration.apply(
            harness.preset(threshold_db=-20.0), noise_reference_reliability=0.2, processing_mode="bypass",
            compressor_metadata={"dynamics_profile": "balanced"},
        )
        configuration.restore(checkpoint, calibration_valid=calibration_valid)
        assert (harness.threshold(), harness.host.mode, harness.reliability()) == (-35.0, "normal", reliability)
        if calibration_valid is not None:
            assert harness.states.compressor.get_settings(include_calibration=True)["dynamics_profile"] == "dense"
    assert harness.host.events == [("sync", True)]


@pytest.fixture
def logged(harness, monkeypatch) -> tuple[Harness, Mock]:
    """The stages write to a Mock engine that also logs the configuration mute, in one call order."""
    engine = Mock()
    engine.get_deesser_high_cut_hz.return_value = 0.0
    for state in harness.states:
        state.processor = engine
    hold = harness.host.hold_mute
    monkeypatch.setattr(harness.host, "hold_mute", lambda reason, on: engine.set_output_mute(on) or hold(reason, on))
    frozen = time.monotonic() + 1.0  # "applied at once, then throttled" must not depend on pauses
    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: frozen)
    return harness, engine


def _writes_between_mutes(calls) -> list:
    """The engine writes made after the configuration mute and before its release, and none after."""
    mute, unmute = call.set_output_mute(True), call.set_output_mute(False)
    start, end = calls.index(mute), len(calls) - 1 - calls[::-1].index(unmute)
    assert not [c for c in calls[end + 1:] if c[0].startswith(("set_", "apply_"))]
    return [c for c in calls[start + 1:end] if c[0].startswith(("set_", "apply_"))]


def test_queued_edits_are_superseded_and_every_write_lands_while_muted(logged) -> None:
    harness, engine = logged
    states, defaults = harness.states, Preset()
    stages = (
        (states.gate, "threshold_db", engine.apply_gate_settings, defaults.gate.threshold_db),
        (states.deesser, "threshold_db", engine.apply_deesser_settings, defaults.deesser.threshold_db),
        (states.compressor, "threshold_db", engine.apply_compressor_settings, defaults.compressor.threshold_db),
        (states.limiter, "ceiling_db", engine.set_limiter_ceiling, defaults.limiter.ceiling_db),
    )
    for state, name, _write, _default in stages:
        state.set_value(name, -11.0)  # applied at once
        state.set_value(name, -12.0)  # waits behind the throttle
    states.eq.set_band_value(0, "gain_db", 3.0)
    states.eq.set_band_value(0, "gain_db", 4.0)
    queued = len(engine.mock_calls)

    harness.configuration.apply(defaults)

    writes = _writes_between_mutes(engine.mock_calls[queued:])
    assert [name for name, _args, _kwargs in writes].count("apply_eq_layers") == 1
    assert states.eq.get_eq_settings().bands[0].gain_db == 0.0
    for state, name, write, default in (*stages, (states.eq, None, None, None)):
        assert not state._rate_limiter._timer.isActive()
        if write is not None:
            # The last write carries the configured value, not the queued -12.0.
            last = write.call_args.args[0]
            assert (last[name] if isinstance(last, dict) else last) == default


def test_a_refused_setter_is_rolled_back_before_the_mute_is_released(logged) -> None:
    harness, engine = logged
    def refuse_once(_settings):
        if engine.apply_deesser_settings.call_count == 1:
            raise RuntimeError("de-esser setter failed")

    engine.apply_deesser_settings.side_effect = refuse_once
    candidate = Preset()
    candidate.deesser.threshold_db = -11.0

    with pytest.raises(ConfigurationNotApplied, match="de-esser setter failed"):
        harness.configuration.apply(candidate)

    writes = _writes_between_mutes(engine.mock_calls)
    thresholds = [c.args[0]["threshold_db"] for c in writes if c[0] == "apply_deesser_settings"]
    # The rejected write comes first, and the last write restores the old value.
    assert (thresholds[0], thresholds[-1]) == (-11.0, -28.0)


# Loading presets


# A load that changes no sound is still one undo step.
@pytest.mark.parametrize("threshold_db", [-30.0, None], ids=["new-sound", "same-sound"])
def test_a_load_shows_and_remembers_the_preset_and_is_one_undo_step(started, threshold_db) -> None:
    configuration = started.configuration
    before = configuration.identity
    path = Path("Loud.json")

    outcome = configuration.load(started.preset("Loud", threshold_db=threshold_db), PresetOrigin(path=path), remember=True)

    assert outcome.loaded and outcome.remembered
    identity = configuration.identity
    assert (identity.name, identity.path, identity.modified) == ("Loud", path, False)
    assert started.config.last_preset == str(path) and started.saves == [str(path)]
    assert configuration.undo() == "Loaded preset (Loud)"
    assert configuration.identity == before and started.config.last_preset == ""
    assert configuration.redo() == "Loaded preset (Loud)"
    assert configuration.identity == identity and started.config.last_preset == str(path)


def test_cancelling_the_unsaved_changes_question_is_a_cancel(started) -> None:
    configuration = started.configuration
    started.states.gate.set_settings({"threshold_db": started.threshold() + 3})
    started.prompts.append(False)
    edited = started.threshold()

    outcome = configuration.load(started.preset("Voice", threshold_db=-50.0), PresetOrigin(key="voice"), remember=False)

    assert outcome.status is LoadStatus.CANCELLED
    assert started.threshold() == edited and configuration.identity.name == "Default"
    started.prompts.append(True)
    assert configuration.load_id("builtin:voice", remember=False).loaded


def test_a_refused_load_keeps_the_sound_and_the_identity(started) -> None:
    preset = started.preset("Deep")
    preset.rnnoise.model = "deepfilter"
    before = (preset_payload(started.configuration.current), started.configuration.identity)

    outcome = started.configuration.load(preset, PresetOrigin(key="deep"), remember=True)

    assert outcome.status is LoadStatus.NOT_APPLIED and not outcome.read_failed
    assert isinstance(outcome.error, ConfigurationNotApplied)
    assert (preset_payload(started.configuration.current), started.configuration.identity) == before


@pytest.mark.parametrize("operation", ["load", "save"])
def test_a_preset_that_cannot_be_remembered_is_still_shown(harness, operation) -> None:
    harness.config.last_preset = "previous.json"
    harness.save_works = False
    preset, path = Preset(name="Fresh"), Path("fresh.json")

    if operation == "load":
        outcome = harness.configuration.load(preset, PresetOrigin(path=path), remember=True)
        assert outcome.loaded and not outcome.remembered
    else:
        assert not harness.configuration.mark_saved(preset, path)

    identity = harness.configuration.identity
    assert (identity.name, identity.path, identity.remembered) == ("Fresh", path, False)
    assert harness.config.last_preset == "previous.json"


@pytest.mark.parametrize(
    ("custom", "preset_id", "loaded_file"),
    [
        ((("Target", "target.json"), ("target.json", "other.json")), "custom-file:target.json", "target.json"),
        ((("Warm Voice", "custom-voice.json"),), "custom:custom-voice.json", "custom-voice.json"),
        ((("Duplicate", "first.json"), ("Duplicate", "second.json")), "custom:Duplicate", None),
        ((("First", "Shared.json"), ("Shared.json", "other.json")), "custom:Shared.json", None),
        ((("missing.json", "other.json"),), "custom-file:missing.json", None),
        ((), "builtin:no-such-preset", None),
        ((), "Voice", None),
    ],
    ids=["file-id-wins", "legacy-filename", "duplicate-name", "filename-is-another-name",
         "missing-file-id", "missing-builtin", "unprefixed"],
)
def test_load_id_resolves_only_unambiguous_identities(harness, custom, preset_id, loaded_file) -> None:
    library = harness.library
    library.custom_presets = [(name, Path(filename)) for name, filename in custom]
    library.files = {path: Preset(name=name) for name, path in library.custom_presets}

    outcome = harness.configuration.load_id(preset_id, remember=False)

    assert outcome.loaded is (loaded_file is not None)
    assert library.reads == ([] if loaded_file is None else [Path(loaded_file)])
    if loaded_file is None:
        assert outcome.status is LoadStatus.MISSING


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (PresetValidationError("bad gate"), LoadStatus.INVALID),
        (ValueError("bad json"), LoadStatus.INVALID),
        (OSError("locked"), LoadStatus.NOT_APPLIED),
        (PresetValidationError("cannot read"), LoadStatus.NOT_APPLIED),
    ],
    ids=["invalid", "malformed", "unreadable", "unreadable-wrapped"],
)
def test_a_file_that_cannot_be_read_is_invalid_or_not_applied_now(harness, error, status) -> None:
    if str(error) == "cannot read":
        error.__cause__ = OSError("locked")
    harness.library.custom_presets = [("Broken", Path("broken.json"))]
    harness.library.files = {Path("broken.json"): error}

    outcome = harness.configuration.load_id("custom-file:broken.json", remember=False)

    assert (outcome.status, outcome.read_failed, outcome.error) == (status, True, error)


# Restoring the saved preset at startup


@pytest.mark.parametrize(
    ("startup_id", "loaded_file", "migrated_id"),
    [
        ("custom-file:second.json", "second.json", "custom-file:second.json"),
        ("custom:Unique", "unique.json", "custom-file:unique.json"),
        ("custom:Duplicate", None, "custom:Duplicate"),
        ("custom-file:missing.json", None, "custom-file:missing.json"),
        ("Unique", "unique.json", "custom-file:unique.json"),
    ],
)
def test_the_startup_preset_resolves_its_file_and_migrates_legacy_ids(harness, startup_id, loaded_file, migrated_id):
    harness.config.startup_preset = startup_id
    harness.library.custom_presets = [
        ("Duplicate", Path("first.json")), ("Duplicate", Path("second.json")),
        ("Unique", Path("unique.json")), ("missing.json", Path("other.json")),
    ]
    harness.library.files = {path: Preset(name=name) for name, path in harness.library.custom_presets}

    restore = harness.configuration.restore_saved_preset()

    assert restore.startup is not None and restore.startup.loaded is (loaded_file is not None)
    assert harness.library.reads == ([] if loaded_file is None else [Path(loaded_file)])
    assert harness.config.startup_preset == migrated_id
    assert restore.config_changed is (startup_id != migrated_id)
    assert harness.config.last_preset == "" and harness.saves == []


def test_a_startup_preset_is_not_remembered_as_last_used(harness) -> None:
    harness.config.startup_preset = "builtin:voice"
    harness.config.last_preset = "builtin:flat"

    restore = harness.configuration.restore_saved_preset()

    assert restore.loaded and restore.startup_name == BUILTIN_PRESETS["voice"].name
    assert restore.last_used is None and harness.config.last_preset == "builtin:flat"


def test_a_failed_startup_preset_falls_back_to_the_last_used_one(harness) -> None:
    harness.config.startup_preset = "builtin:no-such-preset"
    harness.config.last_preset = "builtin:flat"

    restore = harness.configuration.restore_saved_preset()

    assert restore.startup is not None and restore.startup.status is LoadStatus.MISSING
    assert restore.last_used is not None and restore.last_used.loaded
    assert harness.configuration.identity.name == BUILTIN_PRESETS["flat"].name
    assert harness.saves == ["builtin:flat"]


@pytest.mark.parametrize(
    ("last_preset", "content", "status"),
    [
        ("builtin:retired", None, LoadStatus.MISSING),
        ("gone.json", None, LoadStatus.MISSING),
        ("broken.json", PresetValidationError("bad gate"), LoadStatus.INVALID),
        ("locked.json", OSError("locked"), LoadStatus.NOT_APPLIED),
        ("refused.json", "deepfilter", LoadStatus.NOT_APPLIED),
        ("builtin:voice", "engine busy", LoadStatus.NOT_APPLIED),
    ],
)
def test_only_a_missing_or_invalid_last_used_preset_is_forgotten(
    harness, monkeypatch, last_preset, content, status
) -> None:
    harness.config.last_preset = last_preset
    path = Path(last_preset)
    if content == "deepfilter":
        refused = Preset(name="Refused")
        refused.rnnoise.model = "deepfilter"
        harness.library.files[path] = refused
    elif content == "engine busy":
        _fail_once_after_writing(monkeypatch, harness.states.gate)
    elif content is not None:
        harness.library.files[path] = content

    restore = harness.configuration.restore_saved_preset()

    assert restore.last_used is not None and restore.last_used.status is status
    assert restore.last_used_id == last_preset
    forgotten = status in (LoadStatus.MISSING, LoadStatus.INVALID)
    assert harness.config.last_preset == ("" if forgotten else last_preset)
    assert restore.config_changed is (status is LoadStatus.MISSING)


# Preset identity and history


def test_the_preset_is_modified_until_its_sound_and_mode_return(started) -> None:
    configuration = started.configuration
    path = Path("Lifecycle.json")
    assert configuration.mark_saved(started.preset("Lifecycle"), path)
    assert started.config.last_preset == str(path)
    threshold = started.threshold()

    started.states.gate.set_value("threshold_db", threshold + 1)
    assert configuration.identity.modified and started.debouncer.pending
    started.states.gate.set_value("threshold_db", threshold)
    assert not configuration.commit_edit()
    assert not configuration.identity.modified and not started.debouncer.pending

    started.host.mode = "raw"
    configuration.note_edit()
    assert configuration.commit_edit()
    assert configuration.identity.modified
    configuration.undo()
    assert not configuration.identity.modified and started.host.mode == "normal"
    configuration.redo()
    assert configuration.identity.modified


def test_undo_keeps_the_loaded_preset_and_a_later_save_under_a_new_name(started) -> None:
    configuration = started.configuration
    configuration.load(started.preset("Loud", threshold_db=-30.0), PresetOrigin(path=Path("Loud.json")), remember=True)
    started.states.gate.set_settings({"threshold_db": -27.0})
    assert configuration.commit_edit() and configuration.identity.modified

    assert configuration.undo() == "Manual processing edit"
    assert configuration.identity.name == "Loud" and not configuration.identity.modified

    saved = Path("Renamed.json")
    configuration.mark_saved(started.preset("Renamed"), saved)
    configuration.undo()
    assert configuration.identity.name == "Default"
    configuration.redo()
    # The load was saved as Renamed; going back to it must not hand the next Save the old file.
    identity = configuration.identity
    assert (identity.name, identity.path, identity.modified) == ("Renamed", saved, False)


def test_a_manual_edit_clears_auto_eq_diagnostics_only_when_recorded(started) -> None:
    eq = started.states.eq
    eq.set_auto_eq_diagnostics({"band_gains": [0.0] * 10})
    assert not started.configuration.commit_edit()
    assert eq.auto_eq_diagnostics is not None

    started.states.gate.set_settings({"threshold_db": -35.0})
    assert started.configuration.commit_edit()
    assert eq.auto_eq_diagnostics is None


def test_history_restores_calibration_only_in_the_same_capture_context(started) -> None:
    configuration = started.configuration
    started.states.compressor.set_settings({"noise_reference_reliability": 0.7})
    assert configuration.commit_edit()
    started.states.gate.set_settings({"threshold_db": -35.0})
    started.states.compressor.set_settings({"noise_reference_reliability": 0.2})
    assert configuration.commit_edit()

    configuration.undo()
    assert started.reliability() == 0.7
    configuration.redo()
    started.host.context = "route-b"
    configuration.undo()
    assert started.reliability() == 0.0


def test_a_failed_undo_leaves_the_history_and_sound_where_they_were(started, monkeypatch) -> None:
    configuration = started.configuration
    started.states.gate.set_settings({"threshold_db": -35.0})
    assert configuration.commit_edit()
    cursor = configuration.history.cursor
    _fail_once_after_writing(monkeypatch, started.states.eq)

    with pytest.raises(ConfigurationNotApplied, match="injected failure"):
        configuration.undo()

    assert configuration.history.cursor == cursor and started.threshold() == -35.0
    assert not configuration.replaying


def test_restarting_the_history_keeps_one_subscription_to_the_stages(started) -> None:
    configuration = started.configuration
    started.states.gate.set_settings({"threshold_db": -35.0})
    assert configuration.commit_edit()

    configuration.start_history()
    assert not configuration.history.can_undo
    changes = started.host.changes
    started.states.gate.set_value("threshold_db", -31.0)
    started.states.gate.flush()

    assert started.host.changes == changes + 1
    assert configuration.commit_edit() and configuration.history.size == 2


def test_an_edit_not_yet_committed_is_not_overwritten_by_redo(started) -> None:
    configuration = started.configuration
    started.states.gate.set_settings({"threshold_db": -35.0})
    assert configuration.commit_edit()
    configuration.undo()
    assert configuration.history.can_redo

    started.states.gate.set_settings({"threshold_db": -31.0})
    configuration.note_edit()
    assert started.debouncer.pending
    assert configuration.redo() is None

    assert started.threshold() == -31.0 and not configuration.history.can_redo


def test_edits_inside_a_gesture_or_dialog_session_are_one_entry(started) -> None:
    configuration = started.configuration
    eq = started.states.eq
    size = configuration.history.size
    eq.begin_curve_edit(4)
    eq.edit_curve_band(4, 2500.0, 2.0)
    eq.edit_curve_band(4, 3000.0, 2.2)
    assert not started.debouncer.pending
    eq.finish_curve_edit(4, 3000.0, 2.2)
    assert configuration.history.size == size + 1

    with configuration.dialog_session():
        started.states.gate.set_settings({"threshold_db": -35.0})
        started.states.gate.set_settings({"threshold_db": -33.0})
        assert not started.debouncer.pending
        assert configuration.commit_edit(label="Auto-EQ (Broadcast)", source="auto_eq")
    assert configuration.history.size == size + 2
    assert configuration.undo() == "Auto-EQ (Broadcast)"
