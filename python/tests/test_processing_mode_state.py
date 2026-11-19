from __future__ import annotations

from typing import Any

import pytest

from mic_eq.ui.processing_mode_state import ProcessingModeState

_MODES = ("normal", "bypass", "raw")


class _Choice:
    def __init__(self, mode: object = "normal") -> None:
        self.mode = mode
        self.blocked: list[bool] = []

    def currentData(self) -> object:
        return self.mode

    def currentIndex(self) -> int:
        return _MODES.index(self.mode) if self.mode in _MODES else -1

    def findData(self, data: object) -> int:
        return _MODES.index(data) if data in _MODES else -1

    def setCurrentIndex(self, index: int) -> None:
        self.mode = _MODES[index]

    def blockSignals(self, blocked: bool) -> bool:
        self.blocked.append(blocked)
        return False


class _Processor:
    """Refuses to switch a flag on when its mode is in ``fail_on``; switching off always works."""

    def __init__(self, *, fail_on: tuple[str, ...] = ()) -> None:
        self.raw: bool | None = None
        self.bypass: bool | None = None
        self.fail_on = fail_on

    def set_raw_monitor_enabled(self, enabled: bool) -> None:
        if enabled and "raw" in self.fail_on:
            raise OSError("raw refused")
        self.raw = enabled

    def set_bypass(self, enabled: bool) -> None:
        if enabled and "bypass" in self.fail_on:
            raise OSError("bypass refused")
        self.bypass = enabled


def _refuse(_enabled: bool) -> None:
    raise OSError("bypass refused")


class _Recorder:
    def __init__(self) -> None:
        self.mutes: list[bool] = []
        self.statuses: list[tuple[str, int]] = []
        self.changes = 0

    def hold_mute(self, muted: bool) -> None:
        self.mutes.append(muted)

    def show_status(self, text: str, timeout_ms: int) -> None:
        self.statuses.append((text, timeout_ms))

    def on_changed(self) -> None:
        self.changes += 1


def _state(processor: Any = None, choice: _Choice | None = None):
    recorder = _Recorder()
    choice = choice or _Choice()
    state = ProcessingModeState(
        processor if processor is not None else _Processor(),
        choice,
        hold_mute=recorder.hold_mute,
        show_status=recorder.show_status,
        on_changed=recorder.on_changed,
    )
    return state, choice, recorder


def test_selected_reads_the_choice_and_falls_back_to_the_applied_mode() -> None:
    state, choice, _ = _state()
    choice.mode = "raw"
    assert state.selected() == "raw"

    choice.mode = None
    assert state.selected() == "normal"
    state.applied = "bypass"
    choice.mode = "unknown"
    assert state.selected() == "bypass"


@pytest.mark.parametrize(
    ("mode", "raw", "bypass"),
    [("normal", False, False), ("bypass", False, True), ("raw", True, False)],
)
def test_set_applies_the_mode_to_both_engine_flags(mode, raw, bypass) -> None:
    processor = _Processor()
    state, choice, recorder = _state(processor)

    state.set(mode)

    assert (processor.raw, processor.bypass) == (raw, bypass)
    assert state.applied == mode
    assert choice.mode == mode
    assert recorder.changes == 1
    assert recorder.statuses == []


def test_set_shows_the_mode_without_signalling_and_treats_an_unknown_mode_as_normal() -> None:
    state, choice, _ = _state()

    state.set("raw")
    assert choice.blocked == [True, False]

    state.set("invalid")
    assert state.applied == "normal"
    assert choice.mode == "normal"


def test_set_announces_the_mode_only_when_asked() -> None:
    state, _, recorder = _state()

    state.set("raw", notify=True)

    assert recorder.statuses == [("Raw monitor enabled for this session; it is not stored in presets", 0)]


def test_set_raises_and_keeps_the_applied_mode_when_the_engine_refuses() -> None:
    state, _, recorder = _state(_Processor(fail_on=("bypass",)))

    with pytest.raises(RuntimeError, match="Processing mode could not be applied: bypass refused"):
        state.set("bypass")

    assert state.applied == "normal"
    assert recorder.changes == 0


def test_choose_applies_the_picked_mode_and_releases_the_error_mute() -> None:
    processor = _Processor()
    state, choice, recorder = _state(processor)
    choice.mode = "bypass"

    assert state.choose() is True

    assert processor.bypass is True
    assert state.applied == "bypass"
    assert recorder.mutes == [False]
    assert recorder.statuses[0][0].startswith("Voice effects bypassed")


def test_choose_restores_the_previous_mode_when_the_new_one_is_refused() -> None:
    processor = _Processor()
    state, choice, recorder = _state(processor)
    state.set("raw")
    processor.fail_on = ("bypass",)
    choice.mode = "bypass"

    assert state.choose() is False

    assert state.applied == "raw"
    assert choice.mode == "raw"
    assert recorder.mutes == []
    assert recorder.statuses == [
        ("Processing mode could not be applied: bypass refused; previous mode restored", 6000)
    ]


def test_choose_mutes_the_output_when_restoring_fails_too() -> None:
    processor = _Processor()
    state, choice, recorder = _state(processor)
    processor.fail_on = ("raw",)
    processor.set_bypass = _refuse  # type: ignore[method-assign]
    choice.mode = "raw"

    assert state.choose() is False

    assert recorder.mutes == [True]
    assert recorder.statuses == [
        ("Processing mode could not be applied: raw refused; output muted because restoration failed", 0)
    ]
