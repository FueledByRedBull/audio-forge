"""The main window turns the processor's held peaks into meters and Health lines."""

from __future__ import annotations

from typing import Any

import ast
from pathlib import Path

import pytest

from mic_eq.ui.health import HEALTH_PEAK_WINDOW_SECONDS
from mic_eq.ui.level_meter import CLIP_FLAT_TOP_SAMPLES


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Engine:
    """The native processor with the level getters replaced by scripted readings."""

    def __init__(self, native: Any) -> None:
        self.native = native
        self.input_peak = 0.0
        self.output_peak = 0.0
        self.pre_limiter_peak = 0.0
        self.input_count = 0
        self.output_count = 0
        self.diagnostics: dict[str, Any] = {}
        self.calls: dict[str, int] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native, name)

    def _count(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def is_running(self) -> bool:
        return True

    def get_runtime_diagnostics(self) -> dict[str, Any]:
        # A stopped native engine reports a zero crest factor, which reads as "already compressed".
        return {**self.native.get_runtime_diagnostics(), "input_crest_factor_db": 14.0, **self.diagnostics}

    def take_input_peak_since_read(self) -> float:
        self._count("take_input_peak_since_read")
        peak, self.input_peak = self.input_peak, 0.0
        return peak

    def take_output_peak_since_read(self) -> float:
        self._count("take_output_peak_since_read")
        peak, self.output_peak = self.output_peak, 0.0
        return peak

    def take_output_pre_limiter_peak_since_read(self) -> float:
        self._count("take_output_pre_limiter_peak_since_read")
        peak, self.pre_limiter_peak = self.pre_limiter_peak, 0.0
        return peak

    def get_input_full_scale_count(self) -> int:
        return self.input_count

    def get_output_full_scale_count(self) -> int:
        return self.output_count

    def get_input_rms_db(self) -> float:
        return -20.0

    def get_output_rms_db(self) -> float:
        return -20.0

    def get_input_peak_db(self) -> float:
        return -12.0

    def get_output_peak_db(self) -> float:
        return -12.0


class _OldEngine(_Engine):
    """An engine from an older extension module, before the peak and count getters."""

    @staticmethod
    def _missing(name: str) -> bool:
        return name.startswith("take_") or name.endswith("_full_scale_count")

    def __getattribute__(self, name: str) -> Any:
        if _OldEngine._missing(name):
            raise AttributeError(name)
        return super().__getattribute__(name)

    def __getattr__(self, name: str) -> Any:
        if _OldEngine._missing(name):
            raise AttributeError(name)
        return super().__getattr__(name)


@pytest.fixture
def running_window(isolated_main_window):
    window = isolated_main_window
    engine = _Engine(window.processor)
    window.processor = engine
    clock = _Clock()
    window.input_meter.clock = clock
    window.output_meter.clock = clock
    window.engine, window.clock = engine, clock
    window._update_meters()  # the first full-scale count only sets the baseline
    engine.calls.clear()
    return window


def _clip_input(engine: _Engine, samples: int = CLIP_FLAT_TOP_SAMPLES) -> None:
    engine.input_peak = 1.0
    engine.input_count += samples


def _tick_diagnostics(window) -> None:
    window._update_diagnostics()


def test_the_meter_tick_is_the_only_caller_of_the_destructive_peak_getters(running_window):
    window, engine = running_window, running_window.engine
    window._update_meters()
    assert engine.calls == {
        "take_input_peak_since_read": 1,
        "take_output_peak_since_read": 1,
        "take_output_pre_limiter_peak_since_read": 1,
    }
    for _ in range(3):
        _tick_diagnostics(window)
    assert sum(engine.calls.values()) == 3


def test_a_transient_between_ticks_lights_the_input_marker_and_holds_the_line(running_window):
    window, engine = running_window, running_window.engine
    _clip_input(engine)
    window._update_meters()
    assert window.input_meter.clip_marker
    assert window.input_meter.peak_hold_db == window.input_meter.DB_MAX
    assert not window.output_meter.clip_marker


def test_a_saturated_converter_is_seen_through_the_full_scale_count_alone(running_window):
    window, engine = running_window, running_window.engine
    engine.input_count = CLIP_FLAT_TOP_SAMPLES
    window._update_meters()
    assert window.input_meter.clip_marker


def test_the_out_bar_keeps_the_pre_limiter_level_and_the_marker_follows_the_final_output(running_window):
    window, engine = running_window, running_window.engine
    engine.pre_limiter_peak, engine.output_peak = 1.5, 0.89
    window._update_meters()
    assert window.output_meter.peak_hold_db == window.output_meter.DB_MAX
    assert not window.output_meter.clip_marker
    window.limiter_state.set_value("enabled", False)  # a limiter at its ceiling is not a clip
    engine.output_count = CLIP_FLAT_TOP_SAMPLES
    window._update_meters()
    assert window.output_meter.clip_marker


def test_the_marker_clears_by_itself(running_window):
    window, engine, clock = running_window, running_window.engine, running_window.clock
    _clip_input(engine)
    window._update_meters()
    clock.now += 3.2
    window.input_meter._decay_peak_hold()
    assert not window.input_meter.clip_marker


def test_stopping_clears_both_markers(running_window):
    window, engine = running_window, running_window.engine
    window.limiter_state.set_value("enabled", False)
    _clip_input(engine)
    engine.output_peak, engine.output_count = 1.0, CLIP_FLAT_TOP_SAMPLES
    window._update_meters()
    assert window.input_meter.clip_marker and window.output_meter.clip_marker
    engine.is_running = lambda: False
    window._update_meters()
    assert not window.input_meter.clip_marker and not window.output_meter.clip_marker


def test_input_peaks_near_full_scale_become_a_held_health_line_with_the_number(running_window):
    window, engine, clock = running_window, running_window.engine, running_window.clock
    engine.input_peak = 0.966  # -0.3 dB
    window._update_meters()
    _tick_diagnostics(window)
    chip = window.shell_texts["input_health_label"]
    assert (chip.text, chip.state) == ("Input: PEAKS TOO HIGH (-0.3 dB)", "warn")
    assert window.shell_texts["health_summary_label"].text == chip.text
    advice = window.shell_texts["health_advice_label"].text
    assert "Input peaks at -0.3 dB" in advice
    assert "Lower the microphone level by about 6 dB" in advice

    # The warning is held: a later quiet tick does not erase it.
    clock.now += HEALTH_PEAK_WINDOW_SECONDS - 1
    window._update_meters()
    _tick_diagnostics(window)
    assert window.shell_texts["input_health_label"].state == "warn"
    clock.now += 2
    window._update_meters()
    _tick_diagnostics(window)
    assert window.shell_texts["input_health_label"].state == "ok"
    assert "Input: OK" in window.shell_texts["input_health_label"].text


def test_clipping_is_held_for_the_window_and_says_so_more_strongly(running_window):
    window, engine, clock = running_window, running_window.engine, running_window.clock
    _clip_input(engine)
    window._update_meters()
    _tick_diagnostics(window)
    chip = window.shell_texts["input_health_label"]
    assert (chip.text, chip.state) == ("Input: CLIPPED (peak 0.0 dB, full scale 6 times)", "bad")
    assert window.shell_texts["health_advice_label"].text.startswith("The microphone clipped")
    clock.now += 5
    window._update_meters()
    _tick_diagnostics(window)
    assert window.shell_texts["input_health_label"].state == "bad"


def test_the_old_input_clip_counter_no_longer_drives_the_input_line(running_window):
    window = running_window
    window.engine.diagnostics = {"clip_event_count": 40}
    window._update_meters()
    _tick_diagnostics(window)
    assert window.shell_texts["input_health_label"].state == "ok"


def test_the_old_clip_counters_do_not_raise_the_drops_row_or_the_summary(running_window):
    window = running_window
    window.engine.diagnostics = {"clip_event_count": 40, "output_clip_event_count": 3}
    window._update_meters()
    _tick_diagnostics(window)
    drops = window.shell_texts["dropped_label"]
    assert (drops.text, drops.state) == ("Drops: 0 samples dropped, 0 dropouts, 0 restarts", "ok")
    assert window.shell_texts["input_health_label"].state == "ok"
    assert window.shell_texts["health_summary_label"].text == "Health: OK"


def test_separate_touches_of_full_scale_are_reported_as_a_count_of_times(running_window):
    window, engine, clock = running_window, running_window.engine, running_window.clock
    for _ in range(CLIP_FLAT_TOP_SAMPLES + 2):
        clock.now += 0.3
        _clip_input(engine, 1)
        window._update_meters()
    _tick_diagnostics(window)
    chip = window.shell_texts["input_health_label"]
    assert (chip.text, chip.state) == (
        f"Input: CLIPPED (peak 0.0 dB, full scale {CLIP_FLAT_TOP_SAMPLES + 2} times)", "bad",
    )
    advice = window.shell_texts["health_advice_label"].text
    assert f"reached full scale {CLIP_FLAT_TOP_SAMPLES + 2} times in the last few seconds" in advice


def test_full_scale_output_with_the_limiter_off_warns_and_with_it_on_does_not(running_window):
    window, engine = running_window, running_window.engine
    engine.pre_limiter_peak, engine.output_peak = 1.41, 1.0  # +3.0 dB asked for, clamped
    engine.output_count = CLIP_FLAT_TOP_SAMPLES + 34
    window._update_meters()
    _tick_diagnostics(window)
    assert window.shell_texts["output_health_label"].state == "ok"

    window.limiter_state.set_value("enabled", False)
    window._update_meters()  # the limiter was on when those samples arrived: not a clip
    _tick_diagnostics(window)
    assert window.shell_texts["output_health_label"].state == "ok"

    engine.output_count += CLIP_FLAT_TOP_SAMPLES
    window._update_meters()
    _tick_diagnostics(window)
    chip = window.shell_texts["output_health_label"]
    assert (chip.text, chip.state) == ("Output: AT FULL SCALE (peak 3.0 dB)", "bad")
    advice = window.shell_texts["health_advice_label"].text
    assert "limiter is off" in advice
    assert "Turn the limiter on or lower the makeup gain" in advice
    assert "about 4 dB" in advice


@pytest.mark.parametrize("mode", ["normal", "bypass", "raw"])
@pytest.mark.parametrize("limiter", [True, False])
def test_the_out_marker_and_health_follow_the_limiter_in_every_processing_mode(
    running_window, mode, limiter
):
    # Bypass and Raw skip the lookahead limiter, but the final true-peak limiter
    # and the ceiling still apply while the limiter is on.
    window, engine = running_window, running_window.engine
    window.limiter_state.set_value("enabled", limiter)
    window._set_processing_mode(mode)
    engine.pre_limiter_peak, engine.output_peak = 1.41, 1.0
    engine.output_count += CLIP_FLAT_TOP_SAMPLES
    window._update_meters()
    _tick_diagnostics(window)
    assert window.output_meter.clip_marker is (not limiter)
    assert window.shell_texts["output_health_label"].state == ("ok" if limiter else "bad")


def test_a_limiter_holding_the_output_at_a_ceiling_of_zero_is_neither_a_marker_nor_a_warning(running_window):
    window, engine = running_window, running_window.engine
    window.limiter_state.set_value("ceiling_db", 0.0)
    engine.pre_limiter_peak, engine.output_peak = 1.41, 1.0
    engine.output_count += 400  # the limiter keeps landing on the ceiling
    window._update_meters()
    _tick_diagnostics(window)
    assert not window.output_meter.clip_marker
    assert window.shell_texts["output_health_label"].state == "ok"
    assert window.shell_texts["health_summary_label"].text == "Health: OK"


def test_the_out_marker_and_health_agree_when_the_limiter_is_switched(running_window):
    window, engine = running_window, running_window.engine
    window.limiter_state.set_value("enabled", False)
    engine.pre_limiter_peak, engine.output_peak = 1.41, 1.0
    engine.output_count += CLIP_FLAT_TOP_SAMPLES
    window._update_meters()
    _tick_diagnostics(window)
    assert window.output_meter.clip_marker
    assert window.shell_texts["output_health_label"].state == "bad"

    window.limiter_state.set_value("enabled", True)  # Health stops warning, so the marker goes
    window.limiter_state.flush()
    window._update_meters()
    _tick_diagnostics(window)
    assert not window.output_meter.clip_marker
    assert window.shell_texts["output_health_label"].state == "ok"


def test_one_full_scale_sample_is_a_peak_warning_and_a_flat_top_is_a_clip(running_window):
    window, engine = running_window, running_window.engine
    _clip_input(engine, 1)
    window._update_meters()
    _tick_diagnostics(window)
    chip = window.shell_texts["input_health_label"]
    assert (chip.text, chip.state) == ("Input: PEAKS TOO HIGH (0.0 dB)", "warn")
    assert not window.input_meter.clip_marker

    _clip_input(engine, CLIP_FLAT_TOP_SAMPLES)
    window._update_meters()
    _tick_diagnostics(window)
    chip = window.shell_texts["input_health_label"]
    # The first touch is still inside the window, so the clip counts seven.
    assert (chip.text, chip.state) == ("Input: CLIPPED (peak 0.0 dB, full scale 7 times)", "bad")
    assert window.input_meter.clip_marker


def test_a_gap_while_the_meter_timer_was_paused_is_not_reported_as_clipping(running_window):
    window, engine, clock = running_window, running_window.engine, running_window.clock
    window.show()
    window.hide()  # the window is minimised or hidden: the meter timer stops
    assert not window.meter_timer.isActive()
    clock.now += 60.0
    engine.input_peak, engine.pre_limiter_peak, engine.output_peak = 1.0, 1.5, 1.0
    engine.input_count += 500
    engine.output_count += 500

    window.show()  # the first tick afterwards must not read the whole gap
    assert window.meter_timer.isActive()
    assert engine.input_peak == engine.output_peak == engine.pre_limiter_peak == 0.0
    window._update_meters()
    _tick_diagnostics(window)
    assert not window.input_meter.clip_marker and not window.output_meter.clip_marker
    assert window.shell_texts["input_health_label"].state == "ok"
    assert window.input_meter.recent_peak_db(HEALTH_PEAK_WINDOW_SECONDS) is None

    _clip_input(engine)  # a clip after the resume still counts
    window._update_meters()
    assert window.input_meter.clip_marker


def test_missing_level_getters_make_health_unavailable_instead_of_ok(running_window):
    window = running_window
    window.processor = _OldEngine(window.engine.native)
    window.input_meter.set_levels(-20.0, -12.0)
    window._update_meters()
    _tick_diagnostics(window)
    summary = window.shell_texts["health_summary_label"]
    assert "unavailable" in summary.text
    assert "could not read" in window.shell_texts["health_advice_label"].text
    assert window.shell_texts["input_health_label"].state == "idle"


def test_the_destructive_peak_getters_are_called_from_exactly_one_function():
    source_root = Path(__file__).resolve().parents[1] / "mic_eq"
    callers: set[tuple[str, str]] = set()

    def visit(node: ast.AST, path: Path, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else scope
            if isinstance(child, ast.Call):
                callee = child.func.attr if isinstance(child.func, ast.Attribute) else ""
                names = [callee] + [
                    arg.value for arg in child.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                ]
                if any(name.startswith("take_") and name.endswith("_since_read") for name in names):
                    callers.add((path.name, scope))
            visit(child, path, inner)

    for path in source_root.rglob("*.py"):
        visit(ast.parse(path.read_text(encoding="utf-8")), path, "<module>")
    assert callers == {("main_window.py", "_note_meter_peaks")}
