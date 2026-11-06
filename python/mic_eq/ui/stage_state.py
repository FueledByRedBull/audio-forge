"""The one write protocol behind every processing stage's settings.

A stage keeps two values: ``_settings``, what the views show (the desired
values, including an edit still queued for the processor), and
``_applied_settings``, what the processor last accepted. A rejected write
restores the applied values and reports the error. When restoring them fails
too, the processor's state is unknown, so audio stops.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from copy import copy

from PySide6.QtCore import QObject, Signal

from ..config import PresetValidationError
from ..config_parts.validation import _validate_bool, _validate_range
from .rate_limiter import RateLimiter


def validated_settings(
    current: dict,
    changes: Mapping[str, object],
    *,
    section: str,
    noun: str,
    booleans: Collection[str],
    ranges: Mapping[str, tuple],
    special: Mapping[str, Callable[[object], object]] | None = None,
) -> dict:
    """Return a copy of ``current`` with ``changes`` strictly checked; raise ValueError."""
    special = special or {}
    candidate = current.copy()
    for name, value in changes.items():
        try:
            if name in booleans:
                candidate[name] = _validate_bool(value, name, section)
            elif name in special:
                candidate[name] = special[name](value)
            elif name in ranges:
                candidate[name] = _validate_range(
                    value, *ranges[name], name, section, strict=True,
                )
            else:
                raise ValueError(f"Unknown {noun} setting: {name}")
        except PresetValidationError as error:
            raise ValueError(str(error)) from error
    return candidate


class StageState[T](QObject):
    """Write one stage's settings synchronously; subclasses supply ``_write``.

    ``changed`` tells views to re-read, ``configurationEdited`` names an
    accepted edit that changed something, ``writeFailed`` carries any rejected
    write and ``recoveryFailed`` precedes the audio stop after a failed restore.
    """

    changed = Signal()
    configurationEdited = Signal(str)
    writeFailed = Signal(str)
    recoveryFailed = Signal(str)

    # Names the stage in the failed-restore message.
    _stage_name: str

    def __init__(
        self,
        processor,
        settings: T,
        parent: QObject | None = None,
        *,
        stop_audio: Callable[[], object] | None = None,
    ):
        super().__init__(parent)
        self.processor = processor
        # Where the app routes a failure stop off the GUI thread; alone, the processor is stopped here.
        self._stop_audio = stop_audio
        self._settings = settings
        self._applied_settings = copy(settings)

    def _write(self, settings: T) -> None:
        """Send ``settings`` to the processor, raising when it rejects them."""
        raise NotImplementedError

    def _write_rejected(self) -> None:
        """Run before the restore: the view may already show the rejected value."""
        self.changed.emit()

    def _after_apply(self) -> None:
        """Refresh processor-derived state once views know the accepted values."""

    def _apply(
        self,
        candidate: T,
        *,
        label: str | None = None,
        notify: bool = False,
        write: Callable[[T], object] | None = None,
        on_accepted: Callable[[], object] | None = None,
    ) -> None:
        """Write ``candidate`` now; restore the applied values if it is rejected.

        ``label`` names the edit when it changes the applied values; ``notify``
        emits ``changed`` even if the views already show ``candidate``;
        ``write`` replaces ``_write`` for this call; ``on_accepted`` runs once
        the write is recorded, before views hear about it.
        """
        write = write or self._write
        previous = self._applied_settings
        try:
            write(candidate)
        except Exception as error:
            self._write_rejected()
            try:
                write(previous)
            except Exception as rollback_error:
                message = (
                    f"{self._stage_name} update failed: {error}; "
                    f"restoring its previous settings also failed: {rollback_error}"
                )
                self.recoveryFailed.emit(message)
                try:
                    (self._stop_audio or self.processor.stop)()
                except Exception as stop_error:
                    message += f"; stopping audio also failed: {stop_error}"
                self.writeFailed.emit(message)
                raise RuntimeError(message) from error
            self.writeFailed.emit(str(error))
            raise RuntimeError(str(error)) from error
        self._applied_settings = copy(candidate)
        if on_accepted is not None:
            on_accepted()
        if self._settings != candidate:
            self._settings = candidate
            notify = True
        if notify:
            self.changed.emit()
        self._after_apply()
        if label and candidate != previous:
            self.configurationEdited.emit(label)


class RateLimitedStageState[T](StageState[T]):
    """Show interactive edits at once and coalesce their writes to one per 33 ms."""

    def __init__(
        self,
        processor,
        settings: T,
        parent: QObject | None = None,
        *,
        stop_audio: Callable[[], object] | None = None,
    ):
        super().__init__(processor, settings, parent, stop_audio=stop_audio)
        self._rate_limiter = RateLimiter(interval_ms=33)
        self._rate_limiter._timer.setParent(self)

    def flush(self) -> None:
        self._rate_limiter.flush()

    def cancel(self) -> None:
        """Drop a queued edit and show the applied values again."""
        self._rate_limiter.cancel()
        if self._settings != self._applied_settings:
            self._settings = copy(self._applied_settings)
            self.changed.emit()

    def _write_rejected(self) -> None:
        self.cancel()

    def _queue(
        self,
        candidate: T,
        *,
        label: str | None = None,
        notify: bool = True,
        immediate: bool = False,
        on_accepted: Callable[[], object] | None = None,
    ) -> None:
        """Make ``candidate`` the desired values and write it now or within the limit.

        With ``notify`` false the view already shows ``candidate`` and hears
        about it when the write is accepted; ``immediate`` writes before
        returning, superseding any queued edit.
        """
        self._settings = candidate
        if notify:
            self.changed.emit()

        def apply() -> None:
            self._apply(candidate, label=label, notify=not notify, on_accepted=on_accepted)

        if immediate:
            self._rate_limiter.call_now(apply)
        else:
            self._rate_limiter.call(apply)

    def _apply_inline(
        self,
        candidate: T,
        *,
        label: str | None = None,
        on_accepted: Callable[[], object] | None = None,
    ) -> None:
        """Write a bulk configuration before returning, superseding any queued edit."""
        self._rate_limiter.call_now(
            lambda: self._apply(candidate, label=label, on_accepted=on_accepted)
        )
