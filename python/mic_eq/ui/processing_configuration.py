"""The processing configuration: the six stage settings, the preset they came from, and their history.

One module owns how a configuration is written (output muted, all stages or none,
the previous sound restored on failure), which preset the window shows and
whether it is modified, the undo history, and how a stored preset is loaded at
startup or for a route. Callers get statuses back and do the telling: message
boxes, status texts and the warning banner stay with them.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, Protocol

from ..config import (
    CompressorSettings,
    DeEsserSettings,
    GateSettings,
    LimiterSettings,
    Preset,
    PresetValidationError,
    RNNoiseSettings,
)
from .config_history import (
    BoundedConfigurationHistory,
    ConfigurationSnapshot,
    explicit_provenance_after_edit,
)
from .shell_specs import PROCESSING_MODE_OPTIONS
from .startup_presets import (
    STARTUP_BUILTIN_PREFIX,
    STARTUP_CUSTOM_FILE_PREFIX,
    STARTUP_CUSTOM_PREFIX,
    route_preset_selection,
    startup_preset_display_name,
    startup_preset_selection,
)

logger = logging.getLogger(__name__)

# The reason this module holds the output mute under while it writes.
CONFIGURATION_MUTE = "configuration"
PROCESSING_MODES = tuple(mode for _label, mode in PROCESSING_MODE_OPTIONS)
# Calibration metadata a dialog may write with a configuration.
COMPRESSOR_METADATA_KEYS = ("dynamics_intensity", "dynamics_profile", "dynamics_customized")
# How config.last_preset names a built-in preset.
LAST_USED_BUILTIN_PREFIX = "builtin:"


class ConfigurationNotApplied(RuntimeError):
    """Nothing changed: the configuration was refused and the previous sound is back."""


class ConfigurationRestoreFailed(RuntimeError):
    """The previous sound could not be put back; the output stays muted and audio stops."""


class LoadStatus(Enum):
    LOADED = "loaded"
    # The user kept the unsaved changes.
    CANCELLED = "cancelled"
    # No preset has that identity.
    MISSING = "missing"
    # The preset file is not a valid preset.
    INVALID = "invalid"
    # The engine refused it, or its file cannot be read right now; it may load next time.
    NOT_APPLIED = "not_applied"


@dataclass(frozen=True)
class PresetOrigin:
    """Where a preset comes from: a built-in key, a file, or neither."""

    key: str | None = None
    path: Path | None = None

    @property
    def builtin_id(self) -> str | None:
        return f"{LAST_USED_BUILTIN_PREFIX}{self.key}" if self.key else None


@dataclass(frozen=True)
class LoadOutcome:
    status: LoadStatus
    origin: PresetOrigin = PresetOrigin()
    # The preset, once it was read.
    preset: Preset | None = None
    # Why it is not loaded: the read failure, or the apply failure.
    error: Exception | None = None
    # The failure was reading the file (INVALID, or NOT_APPLIED for a file unreadable right now).
    read_failed: bool = False
    # False when the load was to be remembered for the next launch and saving that failed.
    remembered: bool = True

    @property
    def loaded(self) -> bool:
        return self.status is LoadStatus.LOADED


@dataclass(frozen=True)
class SavedPresetRestore:
    """What restore_saved_preset tried: the startup preset, then the last-used one."""

    startup: LoadOutcome | None = None
    startup_name: str = ""
    last_used: LoadOutcome | None = None
    # config.last_preset as it was before a missing or invalid reference was cleared.
    last_used_id: str = ""
    # The startup ID was migrated or a missing last-used reference cleared; the config needs a save.
    config_changed: bool = False

    @property
    def loaded(self) -> bool:
        return any(o is not None and o.loaded for o in (self.startup, self.last_used))


@dataclass(frozen=True)
class PresetIdentity:
    name: str
    description: str
    path: Path | None
    modified: bool
    # Whether the last load or save was remembered for the next launch.
    remembered: bool


@dataclass(frozen=True)
class Checkpoint:
    """A configuration a dialog can put back, with the calibration that went with it."""

    preset: Preset
    processing_mode: str
    context_key: str | None
    noise_reference_reliability: float | None
    compressor_metadata: dict[str, Any] = field(default_factory=dict)


class StageStates(NamedTuple):
    gate: Any
    eq: Any
    noise_suppression: Any
    deesser: Any
    compressor: Any
    limiter: Any


class ConfigurationHost(Protocol):
    """What the window lends: output mute, the processing mode and calibration evidence."""

    def hold_mute(self, reason: str, on: bool) -> bool:
        """Hold or release one temporary output mute; False when the engine refused the mute."""
        ...

    def mute_held(self, reason: str) -> bool: ...

    def audio_running(self) -> bool: ...

    def stop_audio(self) -> None: ...

    def processing_mode(self) -> str: ...

    def set_processing_mode(self, mode: str) -> None:
        """Raise RuntimeError when the engine refuses the mode."""
        ...

    def calibration_context_key(self) -> str | None: ...

    def sync_calibration_evidence(self, *, force_reset: bool = False) -> None: ...

    def show_status(self, text: str, timeout_ms: int) -> None: ...

    def configuration_changed(self) -> None:
        """The preset identity, its modified flag or the history changed; redraw them."""
        ...


class PresetLibrary(Protocol):
    def builtin(self, key: str) -> Preset | None: ...

    def custom(self) -> list[tuple[str, Path]]: ...

    def read(self, path: Path) -> Preset: ...

    def exists(self, path: Path) -> bool: ...


class Debouncer(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...


def preset_payload(preset: Preset) -> str:
    """The sound of a preset, without its name, description, version or provenance."""
    payload = preset.to_dict()
    for key in ("name", "description", "version", "value_provenance"):
        payload.pop(key, None)
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _finite_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


class ProcessingConfiguration:
    """Apply, roll back, remember and undo the processing configuration.

    Ordering: construct, call ``restore_saved_preset`` while startup restores the
    route, then ``start_history``. Before ``start_history`` a load never asks
    about unsaved changes and records no history.
    """

    def __init__(
        self,
        states: StageStates,
        config: Any,
        *,
        host: ConfigurationHost,
        library: PresetLibrary,
        confirm_unsaved: Callable[[], bool],
        debouncer: Debouncer,
        save_config: Callable[[], bool],
        history_limit: int = 50,
    ) -> None:
        self.states = states
        self._config = config
        self._host = host
        self._library = library
        self._confirm_unsaved = confirm_unsaved
        self._debouncer = debouncer
        self._save_config = save_config
        self.history = BoundedConfigurationHistory(limit=history_limit)
        self._history_started = False
        self._replaying = False
        self._transaction_depth = 0
        self._provenance: dict[str, str] = {}
        self._name = "Default"
        self._description = ""
        self._path: Path | None = None
        self._saved_payload: str | None = None
        self._saved_mode = "normal"
        self._modified = False
        self._remembered = True
        # The preset the window showed with a sound; history snapshots of one load share it.
        self._record: dict[str, Any] = {}

    # Reading the configuration

    @property
    def current(self) -> Preset:
        """The configuration the stages hold now, as a preset."""
        states = self.states
        return Preset(
            name="Custom",
            description="User-defined preset",
            gate=GateSettings(**states.gate.get_settings()),
            eq=states.eq.get_eq_settings(),
            rnnoise=RNNoiseSettings(**states.noise_suppression.get_settings()),
            deesser=DeEsserSettings(**states.deesser.get_settings()),
            compressor=CompressorSettings(**states.compressor.get_settings()),
            limiter=LimiterSettings(**states.limiter.get_settings()),
            bypass=self._host.processing_mode() == "bypass",
            value_provenance=dict(self._provenance),
        )

    @property
    def identity(self) -> PresetIdentity:
        return PresetIdentity(self._name, self._description, self._path, self._modified, self._remembered)

    @property
    def processing_mode(self) -> str:
        return self._host.processing_mode()

    @property
    def calibration_context_key(self) -> str | None:
        return self._host.calibration_context_key()

    @property
    def history_started(self) -> bool:
        return self._history_started

    @property
    def replaying(self) -> bool:
        """A configuration is being written; edits and calibration syncs it causes are not the user's."""
        return self._replaying

    # Writing a configuration (dialogs use these directly)

    def apply(
        self,
        preset: Preset,
        *,
        noise_reference_reliability: float | None = None,
        processing_mode: str | None = None,
        compressor_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Write all settings or none, without changing preset identity or history.

        Raises ValueError for a bad mode or metadata (nothing written),
        ConfigurationNotApplied when the write was refused and the previous sound
        restored, and ConfigurationRestoreFailed when restoring failed too: then
        the output stays muted and audio stops.
        """
        if processing_mode is not None and processing_mode not in PROCESSING_MODES:
            raise ValueError(f"Unknown processing mode: {processing_mode!r}")
        if compressor_metadata is not None and compressor_metadata.keys() - set(COMPRESSOR_METADATA_KEYS):
            raise ValueError("Unknown compressor calibration metadata")
        candidate = ConfigurationSnapshot.from_preset(
            preset, label="Apply", source="configuration",
            noise_reference_reliability=noise_reference_reliability or 0.0,
        ).to_preset()
        # A bulk replacement supersedes queued gestures. Capture the accepted
        # state after cancellation so a failed replacement cannot apply them.
        self._cancel_writes()
        previous = self.current
        previous_mode = self._host.processing_mode()
        previous_compressor = self.states.compressor.get_settings(include_calibration=True)
        replaying = self._replaying
        self._replaying = True
        self._debouncer.stop()
        release_mute = True
        try:
            if not self._host.hold_mute(CONFIGURATION_MUTE, True) and self._host.audio_running():
                self._host.stop_audio()
                raise ConfigurationNotApplied("Audio is muted and stopping because configuration mute failed")
            try:
                self._write(candidate)
                if processing_mode is not None:
                    self._host.set_processing_mode(processing_mode)
                if noise_reference_reliability is not None or compressor_metadata:
                    self.states.compressor.set_settings({
                        **(compressor_metadata or {}),
                        "noise_reference_reliability": noise_reference_reliability or 0.0,
                    })
                if noise_reference_reliability is None:
                    self._replaying = False
                    try:
                        self._host.sync_calibration_evidence(force_reset=True)
                    finally:
                        self._replaying = True
                self.flush_writes()
            except Exception as error:
                try:
                    self._cancel_writes()
                    self._write(previous)
                    self._host.set_processing_mode(previous_mode)
                    self.states.compressor.set_settings(previous_compressor)
                    self.flush_writes()
                except Exception as restore_error:
                    release_mute = False
                    # Before the stop is asked for: its failure message must be the one left showing.
                    self._host.show_status("Audio muted and stopping: configuration restoration failed", 7000)
                    self._host.stop_audio()
                    raise ConfigurationRestoreFailed(
                        f"Configuration failed ({error}); restoration failed ({restore_error}). "
                        "Audio is muted and stopping."
                    ) from restore_error
                raise ConfigurationNotApplied(
                    f"Configuration was not applied; previous sound restored: {error}"
                ) from error
        finally:
            self._replaying = replaying
            if release_mute:
                self._host.hold_mute(CONFIGURATION_MUTE, False)

    def flush_writes(self) -> None:
        """Finish queued interactive writes; raises RuntimeError when a stage refuses one."""
        for state in self._queued_states():
            state.flush()

    def checkpoint(self) -> Checkpoint:
        """The configuration now, for ``restore`` to put back."""
        compressor = self.states.compressor.get_settings(include_calibration=True)
        return Checkpoint(
            preset=self.current,
            processing_mode=self._host.processing_mode(),
            context_key=self._host.calibration_context_key(),
            noise_reference_reliability=_finite_or_none(compressor.get("noise_reference_reliability")),
            compressor_metadata={
                key: deepcopy(compressor[key]) for key in COMPRESSOR_METADATA_KEYS if key in compressor
            },
        )

    def restore(self, checkpoint: Checkpoint, *, calibration_valid: bool | None = None) -> None:
        """Put a checkpoint back; raises as ``apply`` does.

        ``calibration_valid`` None writes no calibration, so the host re-derives
        it as after any plain apply; True restores the checkpoint's
        noise-reference reliability and dynamics metadata; False restores the
        metadata with the reliability reset to 0.
        """
        if calibration_valid is None:
            self.apply(checkpoint.preset, processing_mode=checkpoint.processing_mode)
            return
        self.apply(
            checkpoint.preset,
            noise_reference_reliability=checkpoint.noise_reference_reliability if calibration_valid else 0.0,
            processing_mode=checkpoint.processing_mode,
            compressor_metadata=dict(checkpoint.compressor_metadata) or None,
        )

    # Loading presets

    def confirm_discard(self) -> bool:
        """Whether unsaved sound changes may be replaced; asks only when there are some."""
        if not self._history_started:
            return True
        self._refresh_modified()
        if not self._modified:
            return True
        return self._confirm_unsaved()

    def load(self, preset: Preset, origin: PresetOrigin, *, remember: bool) -> LoadOutcome:
        """Apply a preset and show it as the current one; ``remember`` keeps it for the next launch."""
        if self._history_started and not self._replaying and not self.confirm_discard():
            return LoadOutcome(LoadStatus.CANCELLED, origin, preset)
        try:
            self.apply(preset)
        except Exception as error:
            return LoadOutcome(LoadStatus.NOT_APPLIED, origin, preset, error)
        replaying = self._replaying
        preset_id = origin.builtin_id
        persist_identity = remember and not replaying and (preset_id is not None or origin.path is not None)
        previous_last_preset = self._config.last_preset
        if not replaying:
            self._record = {}  # a load starts a record of its own; a save keeps the loaded one
            self._set_identity(preset, path=origin.path, preset_id=preset_id, remember=remember)
        if self._history_started and not replaying:
            self.commit_edit(
                label=f"Loaded preset ({preset.name})",
                source="preset",
                provenance=self._provenance,
            )
        remembered = True
        if persist_identity:
            remembered = self._save_config()
            if not remembered:
                logger.warning("Could not remember loaded preset %s", preset.name)
                self._config.last_preset = previous_last_preset
                self._mirror_record()
            self._remembered = remembered
        return LoadOutcome(LoadStatus.LOADED, origin, preset, remembered=remembered)

    def load_id(self, preset_id: str, *, remember: bool) -> LoadOutcome:
        """Load a preset by its stable ID (``builtin:key``, ``custom-file:name.json``, ``custom:name``)."""
        if preset_id.startswith(STARTUP_BUILTIN_PREFIX):
            return self._load_builtin(preset_id[len(STARTUP_BUILTIN_PREFIX):], remember=remember)
        if preset_id.startswith((STARTUP_CUSTOM_PREFIX, STARTUP_CUSTOM_FILE_PREFIX)):
            selection = route_preset_selection(preset_id, self._library.custom())
            if selection is not None:
                return self._load_file(selection[1][1], remember=remember)
        return LoadOutcome(LoadStatus.MISSING)

    def restore_saved_preset(self) -> SavedPresetRestore:
        """Load the startup preset, or failing that the last-used one.

        Migrates a legacy startup ID, and clears a last-used reference only
        when that preset is missing or invalid; one the engine refused, or a
        file unreadable right now, stays for the next launch.
        """
        config = self._config
        config_changed = False
        startup: LoadOutcome | None = None
        startup_name = ""
        if config.startup_preset:
            custom = self._library.custom()
            preset_id, custom_startup = startup_preset_selection(config.startup_preset, custom)
            if preset_id != config.startup_preset:
                config.startup_preset = preset_id
                config_changed = True
            startup_name = (
                custom_startup[0] if custom_startup is not None else startup_preset_display_name(preset_id)
            )
            if preset_id.startswith(STARTUP_BUILTIN_PREFIX):
                startup = self._load_builtin(preset_id[len(STARTUP_BUILTIN_PREFIX):], remember=False)
            elif preset_id.startswith((STARTUP_CUSTOM_PREFIX, STARTUP_CUSTOM_FILE_PREFIX)):
                startup = (
                    self._load_file(custom_startup[1], remember=False)
                    if custom_startup is not None
                    else LoadOutcome(LoadStatus.MISSING)
                )
            else:  # a legacy display name stored before IDs
                path = next((path for name, path in custom if name == preset_id), None)
                startup = (
                    self._load_file(path, remember=False) if path is not None else LoadOutcome(LoadStatus.MISSING)
                )

        last_used: LoadOutcome | None = None
        last_used_id = ""
        if (startup is None or not startup.loaded) and config.last_preset:
            last_used_id = config.last_preset
            if last_used_id.startswith(LAST_USED_BUILTIN_PREFIX):
                last_used = self._load_builtin(last_used_id[len(LAST_USED_BUILTIN_PREFIX):], remember=True)
            else:
                path = Path(last_used_id)
                last_used = (
                    self._load_file(path, remember=True)
                    if self._library.exists(path)
                    else LoadOutcome(LoadStatus.MISSING, PresetOrigin(path=path))
                )
            if last_used.status in (LoadStatus.MISSING, LoadStatus.INVALID):
                config.last_preset = ""
                config_changed = config_changed or last_used.status is LoadStatus.MISSING
        return SavedPresetRestore(startup, startup_name, last_used, last_used_id, config_changed)

    def mark_saved(self, preset: Preset, path: Path) -> bool:
        """Show a just-saved file as the current preset and remember it; False when remembering failed."""
        previous_last_preset = self._config.last_preset
        self._config.last_preset = str(path)
        remembered = self._save_config()
        if not remembered:
            self._config.last_preset = previous_last_preset
        self._remembered = remembered
        self._set_identity(preset, path=path, remember=remembered)
        return remembered

    # History

    def start_history(self) -> None:
        """Record the current configuration as the history's baseline and follow the stages' edits.

        Calling it again restarts the history from the configuration then current.
        """
        preset = self.current
        if self._saved_payload is None:
            self._name = "Default"
            self._path = None
            self._saved_payload = preset_payload(preset)
            self._saved_mode = self._host.processing_mode()
            self._modified = False
        snapshot = ConfigurationSnapshot.from_preset(
            preset,
            label="Startup configuration",
            source="startup",
            noise_reference_reliability=self._reliability(),
            calibration_context_key=self._host.calibration_context_key(),
            processing_mode=self._host.processing_mode(),
            preset_record=self._mirror_record(),
        )
        self.history.initialize(snapshot)
        self._provenance = dict(snapshot.to_preset().value_provenance)
        if not self._history_started:
            eq = self.states.eq
            eq.configurationEditStarted.connect(self._begin_gesture)
            eq.configurationEditFinished.connect(self._end_gesture)
            for state in self.states:
                state.configurationEdited.connect(self.note_edit)
        self._history_started = True
        self._host.sync_calibration_evidence()
        self._host.configuration_changed()

    def note_edit(self, *_args: object) -> None:
        """An edit landed; mark the preset modified and commit it once the edits pause."""
        if not self._history_started or self._replaying or self._transaction_depth > 0:
            return
        self._modified = True
        self._host.sync_calibration_evidence()
        self._host.configuration_changed()
        self._debouncer.start()

    def commit_edit(
        self,
        *,
        label: str = "Manual processing edit",
        source: str = "ui",
        provenance: dict[str, str] | None = None,
    ) -> bool:
        """Record the current configuration as one history entry; False when nothing new was recorded."""
        if not self._history_started or self._replaying:
            return False
        states = self.states
        try:
            for state in (states.limiter, states.deesser, states.compressor, states.gate, states.eq):
                state.flush()
        except RuntimeError:
            return False
        finally:
            self._debouncer.stop()
        if self._host.mute_held(CONFIGURATION_MUTE):
            return False
        preset = self.current
        current = self.history.current
        if provenance is not None:
            preset.value_provenance = dict(provenance)
        elif current is not None:
            preset.value_provenance = explicit_provenance_after_edit(current, preset)
        try:
            snapshot = ConfigurationSnapshot.from_preset(
                preset,
                label=label,
                source=source,
                noise_reference_reliability=self._reliability(),
                calibration_context_key=self._host.calibration_context_key(),
                processing_mode=self._host.processing_mode(),
                preset_record=self._record,
            )
            recorded = self.history.record(snapshot)
        except (PresetValidationError, TypeError, ValueError) as error:
            logger.warning("Configuration history snapshot rejected: %s", error)
            self._host.show_status("Could not record this configuration edit", 5000)
            return False
        self._host.sync_calibration_evidence()
        if recorded:
            self._provenance = dict(snapshot.to_preset().value_provenance)
            if source == "ui":
                self.states.eq.set_auto_eq_diagnostics(None)
        self._refresh_modified()
        return recorded

    def undo(self) -> str | None:
        """Undo one entry; returns its label, None when there is nothing to undo, raises when restoring fails."""
        self.commit_edit()
        undone_label = self.history.undo_label
        restored = self.history.undo(self._restore_snapshot)
        if restored is None:
            return None
        return undone_label or restored.label

    def redo(self) -> str | None:
        """Redo one entry; returns its label, None when there is nothing to redo, raises when restoring fails."""
        # A fresh edit invalidates the redo branch even if its debounce timer
        # has not fired yet. Commit it first so redo cannot overwrite it.
        self.commit_edit()
        restored = self.history.redo(self._restore_snapshot)
        return None if restored is None else restored.label

    @contextmanager
    def dialog_session(self) -> Iterator[None]:
        """Let a dialog change the configuration as one history entry, which it commits itself."""
        self.commit_edit()
        self._transaction_depth += 1
        try:
            yield
        finally:
            self._transaction_depth -= 1

    # Internals

    def _begin_gesture(self) -> None:
        """Suppress intermediate history entries for a compound gesture."""
        if self._transaction_depth == 0:
            self.commit_edit()
        self._transaction_depth += 1

    def _end_gesture(self, label: str) -> None:
        """Commit one final entry after a compound gesture."""
        if self._transaction_depth <= 0:
            logger.warning("Unbalanced configuration-history transaction")
            return
        self._transaction_depth -= 1
        if self._transaction_depth == 0:
            self.commit_edit(label=label, source="eq_graph")

    def _load_builtin(self, key: str, *, remember: bool) -> LoadOutcome:
        preset = self._library.builtin(key)
        if preset is None:
            return LoadOutcome(LoadStatus.MISSING, PresetOrigin(key=key))
        return self.load(preset, PresetOrigin(key=key), remember=remember)

    def _load_file(self, path: Path, *, remember: bool) -> LoadOutcome:
        origin = PresetOrigin(path=path)
        try:
            preset = self._library.read(path)
        except Exception as error:
            # load_preset wraps OS errors in PresetValidationError; such a file may read next time.
            # Anything else a malformed file raises makes it invalid, never a failed startup.
            unreadable = isinstance(error, OSError) or isinstance(error.__cause__, OSError)
            status = LoadStatus.NOT_APPLIED if unreadable else LoadStatus.INVALID
            return LoadOutcome(status, origin, error=error, read_failed=True)
        return self.load(preset, origin, remember=remember)

    def _restore_snapshot(self, snapshot: ConfigurationSnapshot) -> None:
        """Restore one validated snapshot without creating a new entry."""
        preset = snapshot.to_preset()
        self._replaying = True
        try:
            same_context = (
                snapshot.calibration_context_key is not None
                and snapshot.calibration_context_key == self._host.calibration_context_key()
            )
            self.apply(
                preset,
                noise_reference_reliability=snapshot.noise_reference_reliability if same_context else 0.0,
                processing_mode=snapshot.processing_mode,
            )
            if snapshot.preset_record is not None:
                self._restore_record(snapshot.preset_record)
            self._refresh_modified()
        finally:
            self._replaying = False

    def _write(self, preset: Preset) -> None:
        states = self.states
        states.noise_suppression.set_settings(asdict(preset.rnnoise))
        states.gate.set_settings(asdict(preset.gate))
        states.eq.set_settings(preset.eq.to_dict())
        states.deesser.set_settings(asdict(preset.deesser))
        states.compressor.set_settings(asdict(preset.compressor))
        states.limiter.set_settings(asdict(preset.limiter))
        self._host.set_processing_mode("bypass" if preset.bypass else "normal")
        self._provenance = dict(preset.value_provenance)

    def _queued_states(self) -> tuple[Any, ...]:
        states = self.states
        return (states.eq, states.gate, states.deesser, states.compressor, states.limiter)

    def _cancel_writes(self) -> None:
        """Discard interactive writes superseded by the bulk configuration."""
        for state in self._queued_states():
            state.cancel()

    def _reliability(self) -> float:
        return self.states.compressor.get_settings(include_calibration=True)["noise_reference_reliability"]

    def _set_identity(
        self,
        preset: Preset,
        *,
        path: Path | None = None,
        preset_id: str | None = None,
        remember: bool = True,
    ) -> None:
        self._name = str(preset.name or "Default")
        self._description = preset.description
        self._path = path
        self._saved_payload = preset_payload(preset)
        self._saved_mode = "bypass" if preset.bypass else "normal"
        if remember:
            if preset_id is not None:
                self._config.last_preset = preset_id
            elif path is not None:
                self._config.last_preset = str(path)
        self._mirror_record()
        self._refresh_modified()

    def _mirror_record(self) -> dict[str, Any]:
        """Copy the preset shown into the record that history snapshots share."""
        self._record.update(
            name=self._name,
            description=self._description,
            path=self._path,
            saved_payload=self._saved_payload,
            saved_mode=self._saved_mode,
            last_preset=self._config.last_preset,
        )
        return self._record

    def _restore_record(self, record: dict[str, Any]) -> None:
        """Show the preset that went with a history snapshot, and remember it for the next launch."""
        self._record = record
        self._name = record["name"]
        self._description = record["description"]
        self._path = record["path"]
        self._saved_payload = record["saved_payload"]
        self._saved_mode = record["saved_mode"]
        if self._config.last_preset != record["last_preset"]:
            self._config.last_preset = record["last_preset"]
            self._save_config()

    def _refresh_modified(self) -> None:
        if self._saved_payload is None:
            modified = False
        else:
            try:
                modified = (
                    preset_payload(self.current) != self._saved_payload
                    or self._host.processing_mode() != self._saved_mode
                )
            except (PresetValidationError, TypeError, ValueError):
                modified = True
        self._modified = modified
        self._host.configuration_changed()
