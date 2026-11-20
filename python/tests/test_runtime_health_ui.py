"""Live health expires recovered faults and stops presenting stale measurements."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QObject
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QWidget

from mic_eq.ui.first_run_setup_dialog import route_health_reason
from mic_eq.ui.health import (
    HEALTH_PEAK_WINDOW_SECONDS,
    METER_GETTERS,
    NOISE_SELECTION_DROPPED_WARNING,
    READINGS_UNAVAILABLE,
    Chip,
    HealthMemory,
    HealthMonitor,
    HealthReadings,
    HealthReport,
    advice_for,
)
from mic_eq.ui.main_window import HEALTH_NOT_READ, MainWindow
from mic_eq.ui.shell_specs import (
    LATENCY_LOWER_BOUND_NOTE,
    LATENCY_LOWER_BOUND_NOTE_NO_BUFFER,
    TEXT_SPECS,
)
from mic_eq.ui.shell_state import TextState
from mic_eq.ui.theme import PALETTE
from engine_fakes import fake_session


def test_presentation_timer_pauses_without_stopping_diagnostics(qapp):
    from PySide6.QtCore import QTimer

    state = {"running": False, "visible": True, "minimized": False}
    meter_timer, diagnostics_timer = QTimer(), QTimer()
    meter_timer.setInterval(16)
    diagnostics_timer.start(250)
    window: Any = SimpleNamespace(
        processor=SimpleNamespace(is_running=lambda: state["running"]),
        engine_session=fake_session(),
        isVisible=lambda: state["visible"],
        isMinimized=lambda: state["minimized"],
        meter_timer=meter_timer,
        _invalidate_live_meters=Mock(),
        _clear_input_analyser=Mock(),
        _note_meter_peaks=Mock(),
    )
    try:
        cases = (
            (False, True, False),
            (True, True, False),
            (True, False, False),
            (True, True, True),
            (True, True, False),
        )
        for running, visible, minimized in cases:
            state.update(running=running, visible=visible, minimized=minimized)
            MainWindow._sync_meter_timer(window)
            assert meter_timer.isActive() == (running and visible and not minimized)
            assert diagnostics_timer.isActive()
        # Each restart first drops what piled up while the timer was stopped.
        assert [call.kwargs for call in window._note_meter_peaks.call_args_list] == [{"resumed": True}] * 2
    finally:
        meter_timer.stop()
        diagnostics_timer.stop()


def test_window_state_change_pauses_and_restarts_fast_meter_timer(qapp):
    from PySide6.QtCore import QEvent, QTimer
    from PySide6.QtWidgets import QMainWindow

    window = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(window)
    state = {"minimized": False}
    cast(Any, window).processor = SimpleNamespace(is_running=lambda: True)
    window.engine_session = Mock(view=SimpleNamespace(in_use=False))
    window.meter_timer = QTimer(window)
    window.diagnostics_timer = QTimer(window)
    window.diagnostics_timer.start(250)
    window._invalidate_live_meters = Mock()
    window._note_meter_peaks = Mock()
    window.isVisible = lambda: True
    window.isMinimized = lambda: state["minimized"]
    try:
        MainWindow._sync_meter_timer(window)
        assert window.meter_timer.isActive()
        assert window.diagnostics_timer.isActive()

        state["minimized"] = True
        window.changeEvent(QEvent(QEvent.Type.WindowStateChange))
        assert not window.meter_timer.isActive()
        assert window.diagnostics_timer.isActive()

        window._note_meter_peaks.reset_mock()
        state["minimized"] = False
        window.changeEvent(QEvent(QEvent.Type.WindowStateChange))
        assert window.meter_timer.isActive()
        assert window.diagnostics_timer.isActive()
        window._note_meter_peaks.assert_called_once()
        assert window._note_meter_peaks.call_args.kwargs == {"resumed": True}
    finally:
        window.meter_timer.stop()
        window.diagnostics_timer.stop()
        window.deleteLater()
        qapp.processEvents()


def test_unavailable_vad_does_not_present_default_noise_floor(qapp):
    from mic_eq import AudioProcessor
    from mic_eq.ui.gate_state import GateState
    from mic_eq.ui.processing_meters import ProcessingMeters

    native = AudioProcessor()

    class Processor:
        def __getattr__(self, name):
            return getattr(native, name)

        def is_vad_available(self):
            return False

        def get_noise_floor(self):
            return -60.0

    processor = Processor()
    state = GateState(processor)
    meters = ProcessingMeters(processor, state)
    try:
        state.update_vad_confidence(0.0)
        assert state.presentation()["noise_floor"] == "Noise Floor: --"
        assert not meters.confidence.measurement_available
    finally:
        native.stop()


def _diagnostic_window(diagnostics: dict[str, Any]):
    """A MainWindow stand-in that shows what its real health monitor judges of ``diagnostics``."""

    processor = SimpleNamespace(
        is_running=lambda: True,
        get_runtime_diagnostics=lambda: diagnostics,
        get_input_callback_age_ms=lambda: 0,
        get_output_callback_age_ms=lambda: 0,
    )
    state_owner = QObject()
    window: Any = SimpleNamespace(
        health_monitor=HealthMonitor(),
        _processing=False,
        processor=processor,
        engine_session=fake_session(),
        _update_session_summary=Mock(),
        _shell_state_owner=state_owner,
        shell_texts={},
    )
    names = (
        "latency_label",
        "latency_readout_label",
        "buffer_label",
        "input_health_label",
        "output_health_label",
        "gate_health_label",
        "callback_health_label",
        "underrun_health_label",
        "dropped_label",
        "backend_diag_label",
        "recovery_diag_label",
        "health_summary_label",
    )
    for name in (*names, "health_advice_label"):
        window.shell_texts[name] = TextState(state_owner, name=name)
    window._set_health_chip = lambda key, text, state: MainWindow._set_health_chip(
        window, key, text, state
    )
    window._update_health_summary = lambda: MainWindow._update_health_summary(window)
    args: dict[str, Any] = dict(
        diagnostics=diagnostics,
        latency_ms=10.0,
        dsp_time_ms=0.2,
        input_buf=0,
        rnnoise_buf=0,
        input_rms_db=-20.0,
        output_rms_db=-20.0,
        input_callback_age_ms=0,
        output_callback_age_ms=0,
    )
    return window, processor, args


def _judge(window, *, diagnostics, now: float = 0.0, **readings) -> None:
    """One diagnostics tick at monotonic time ``now``, judged by the window's monitor and shown by the window code."""
    view = window.health_monitor.judge(
        diagnostics,
        HealthReadings(
            output_muted=window.engine_session.view.output_muted, processing=window._processing, **readings
        ),
        now=now,
    )
    MainWindow._show_health(window, view)


def test_recovered_stream_clears_live_health_but_keeps_lifetime_counts(
    qapp, monkeypatch
):
    clock = [10.0]
    # The setup check reads the stream window on the module clock.
    monkeypatch.setattr("mic_eq.ui.health.time.monotonic", lambda: clock[0])
    diagnostics = {
        "input_callback_error_count": 1,
        "stream_restart_count": 1,
        "input_dropped_samples": 256,
        "last_restart_reason": "input callback error",
    }
    window, processor, args = _diagnostic_window(diagnostics)
    texts = window.shell_texts
    _judge(window, now=clock[0], **args)
    assert texts["health_summary_label"].state == "warn"
    assert not route_health_reason(processor, health_window=window.health_monitor.stream_health)[0]
    clock[0] += 5.1
    _judge(window, now=clock[0], **args)
    assert texts["health_summary_label"].state == "ok"
    assert texts["recovery_diag_label"].state == "ok"
    assert "1 restart" in texts["recovery_diag_label"].text
    assert "ICE:1" in texts["dropped_label"].toolTip
    assert route_health_reason(processor, health_window=window.health_monitor.stream_health)[0]
    assert diagnostics["input_callback_error_count"] == 1
    assert diagnostics["last_restart_reason"] == "input callback error"
    diagnostics["input_callback_error_count"] += 1
    _judge(window, now=clock[0], **args)
    assert texts["health_summary_label"].state == "warn"


def test_the_window_renders_what_the_health_module_decides(isolated_main_window, monkeypatch):
    window = isolated_main_window
    report = HealthReport(
        chips={
            "gate_health_label": Chip("Gate: CHATTERING (3 times)", "warn"),
            "dropped_label": Chip("Drops: nothing dropped, but the gate is chattering", "warn", "counters"),
        },
        messages=(("first", 8000), ("second", 6000)),
        memory=HealthMemory(noise_drops_seen=7),
    )
    asked: dict[str, Any] = {}

    def decide(snapshot, readings, **context):
        asked.update(snapshot=snapshot, readings=readings, **context)
        return report

    monkeypatch.setattr("mic_eq.ui.health.runtime_health", decide)
    shown: list[tuple[str, int]] = []
    monkeypatch.setattr(window.status_message, "showMessage", lambda text, timeout: shown.append((text, timeout)))

    window._show_health(window.health_monitor.judge(
        {"output_underrun_total": 2},
        HealthReadings(latency_ms=10.0, dsp_time_ms=0.2, input_buf=0, rnnoise_buf=0),
        now=0.0,
    ))

    assert asked["snapshot"].output_underrun_total == 2
    assert (asked["readings"].latency_ms, asked["memory"]) == (10.0, HealthMemory())
    texts = window.shell_texts
    assert (texts["gate_health_label"].text, texts["gate_health_label"].state) == ("Gate: CHATTERING (3 times)", "warn")
    assert texts["gate_health_label"].toolTip == TEXT_SPECS["gate_health_label"].tip  # no tip: it keeps its own
    assert texts["dropped_label"].toolTip == "counters"
    assert shown == [("first", 8000), ("second", 6000)]
    assert window.health_monitor.memory == HealthMemory(noise_drops_seen=7)
    summary = texts["health_summary_label"]
    assert (summary.text, summary.state) == ("Gate: CHATTERING (3 times)", "warn")
    assert texts["health_advice_label"].text == advice_for("Gate: CHATTERING (3 times)")


def test_latency_chip_is_a_lower_bound_until_the_route_is_measured(qapp):
    spec_tip = TEXT_SPECS["latency_label"].tip
    window, _processor, args = _diagnostic_window({})
    chip = window.shell_texts["latency_label"]

    _judge(window, **args)

    assert chip.text == "Latency: at least 10 ms (processing 0.2 ms)"
    assert chip.state == "info"
    assert chip.toolTip == spec_tip + LATENCY_LOWER_BOUND_NOTE_NO_BUFFER
    assert "lower bound" in chip.toolTip
    assert "Measure Route Latency" in chip.toolTip

    # With the output buffer read, the note says it is counted.
    _judge(window, **{**args, "diagnostics": {"output_device_buffer_known": True}})

    assert chip.text == "Latency: at least 10 ms (processing 0.2 ms)"
    assert chip.toolTip == spec_tip + LATENCY_LOWER_BOUND_NOTE
    assert "Windows" not in chip.toolTip

    # A measured route holds the whole delay, so the figure reads as an estimate again.
    _judge(window, **{**args, "diagnostics": {"latency_compensation_ms": 48.0}})

    assert chip.text == "Latency: about 10 ms (processing 0.2 ms)"
    assert chip.toolTip == spec_tip


class _ReadableEngine:
    """Every getter the monitor reads, with values a test can override."""

    def __init__(self, **values: Any) -> None:
        self.values: dict[str, Any] = {
            "get_runtime_diagnostics": {"output_underrun_total": 2},
            "get_input_rms_db": -20.0, "get_output_rms_db": -18.0, "get_output_buffer_samples": 480,
            "get_latency_ms": 24.0, "get_dsp_time_smoothed_ms": 0.5,
            "get_input_buffer_smoothed_samples": 128, "get_buffer_smoothed_samples": 64,
            "get_input_callback_age_ms": 5, "get_output_callback_age_ms": 7,
            **{name: 0 for name in METER_GETTERS},
            **values,
        }

    def __getattr__(self, name: str):
        if name not in self.values:
            raise AttributeError(name)
        value = self.values[name]

        def get():
            if isinstance(value, Exception):
                raise value
            return value

        return get


def _meter(peak_db: float | None = None, full_scale: int = 0, clipped: bool = False) -> Any:
    return SimpleNamespace(
        recent_peak_db=Mock(return_value=peak_db),
        full_scale_samples_within=Mock(return_value=full_scale),
        clipped_within=Mock(return_value=clipped),
    )


def _read(engine: Any, input_meter: Any = None, output_meter: Any = None, **flags: bool):
    flags = {"limiter_enabled": True, "output_muted": False, "processing": True, **flags}
    return HealthMonitor.read(engine, input_meter or _meter(), output_meter or _meter(), **flags)


def test_the_monitor_reads_the_engine_and_the_meters_held_peaks() -> None:
    input_meter, output_meter = _meter(-3.5, 2, False), _meter(0.5, 0, True)

    reading = _read(
        _ReadableEngine(), input_meter, output_meter, limiter_enabled=False, output_muted=True,
    )

    assert reading == ({"output_underrun_total": 2}, HealthReadings(
        latency_ms=24.0, dsp_time_ms=0.5, input_buf=128, rnnoise_buf=64,
        input_rms_db=-20.0, output_rms_db=-18.0, input_callback_age_ms=5, output_callback_age_ms=7,
        input_peak_db=-3.5, input_full_scale_samples=2, output_peak_db=0.5, output_at_full_scale=True,
        limiter_enabled=False, output_muted=True, recovery_pending=False, processing=True,
    ))
    input_meter.recent_peak_db.assert_called_once_with(HEALTH_PEAK_WINDOW_SECONDS)
    input_meter.full_scale_samples_within.assert_called_once_with(HEALTH_PEAK_WINDOW_SECONDS)
    output_meter.recent_peak_db.assert_called_once_with(HEALTH_PEAK_WINDOW_SECONDS)
    output_meter.clipped_within.assert_called_once_with(HEALTH_PEAK_WINDOW_SECONDS)


@pytest.mark.parametrize(
    "pending",
    [{"is_recovering": True}, {"is_recovery_requested": True}, {"is_recovering": RuntimeError("busy")}],
)
def test_a_recovery_in_flight_or_unreadable_counts_as_a_pending_restart(pending) -> None:
    reading = _read(_ReadableEngine(**{"is_recovering": False, "is_recovery_requested": False, **pending}))
    assert reading is not None and reading[1].recovery_pending


@pytest.mark.parametrize(
    "broken",
    [
        {"get_runtime_diagnostics": RuntimeError("no telemetry")},
        {"get_runtime_diagnostics": [("output_underrun_total", 2)]},
        {"get_latency_ms": float("nan")},
        {"get_output_buffer_samples": float("inf")},
        {"get_input_callback_age_ms": None},
        {"get_dsp_time_smoothed_ms": OverflowError("too big")},
        {"get_output_callback_age_ms": OSError("device gone")},
    ],
)
def test_the_monitor_refuses_a_tick_it_cannot_fully_read(broken) -> None:
    input_meter = _meter(-3.0)
    assert _read(_ReadableEngine(**broken), input_meter) is None
    input_meter.recent_peak_db.assert_not_called()


@pytest.mark.parametrize("missing", [*METER_GETTERS, "get_output_rms_db"])
def test_an_engine_without_a_getter_cannot_be_read(missing) -> None:
    engine = _ReadableEngine()
    del engine.values[missing]
    assert _read(engine) is None


def test_the_unreadable_view_resets_the_page_and_says_what_to_do() -> None:
    assert READINGS_UNAVAILABLE.reset
    assert READINGS_UNAVAILABLE.chips == {"health_summary_label": Chip("Health: readings unavailable", "warn")}
    assert READINGS_UNAVAILABLE.messages == ()
    advice = READINGS_UNAVAILABLE.advice or ""
    assert "could not read" in advice and "Stop and start processing" in advice


_JUDGED = HealthReadings(latency_ms=24.0, dsp_time_ms=0.5, input_buf=128, rnnoise_buf=0)


def test_judge_remembers_each_tick_for_the_next() -> None:
    monitor = HealthMonitor()

    first = monitor.judge({"output_underrun_total": 2}, _JUDGED, now=10.0)

    assert not first.reset and first.advice is None
    assert first.chips["underrun_health_label"] == Chip("Dropouts: 2", "warn")
    assert "health_summary_label" not in first.chips  # the window reads it back from the chips
    assert monitor.memory.output_underrun_total == 2
    assert monitor.stream_health.recent_events == {"output_underrun_total"}

    second = monitor.judge({"output_underrun_total": 2}, _JUDGED, now=15.0)
    assert second.chips["underrun_health_label"] == Chip("Dropouts: 2", "ok")
    assert not monitor.stream_health.recent_events


def test_judge_announces_a_dropped_selection_once() -> None:
    monitor = HealthMonitor()
    assert monitor.judge({"noise_selection_dropped_count": 1}, _JUDGED, now=0.0).messages == (
        (NOISE_SELECTION_DROPPED_WARNING, 8000),
    )
    assert monitor.judge({"noise_selection_dropped_count": 1}, _JUDGED, now=1.0).messages == ()


def test_judge_resets_the_page_when_the_counters_are_invalid_and_forgets_nothing() -> None:
    monitor = HealthMonitor(memory=HealthMemory(output_underrun_total=3, noise_drops_seen=1))

    view = monitor.judge({"input_dropped_samples": -1}, _JUDGED, now=0.0)

    assert view.reset and view.advice is None and view.messages == ()
    assert view.chips == {"recovery_diag_label": Chip("Recovery: counters could not be read", "warn")}
    assert monitor.memory == HealthMemory(output_underrun_total=3, noise_drops_seen=1)


def test_the_window_shows_a_reset_view_without_reading_the_summary_back(isolated_main_window) -> None:
    window = isolated_main_window
    texts = window.shell_texts
    window._set_health_chip("input_health_label", "Input: CLIPPED (peak 0.0 dB, full scale 8 times)", "bad")
    window._update_health_summary()

    window._show_health(window.health_monitor.judge({"input_dropped_samples": -1}, _JUDGED, now=0.0))

    assert (texts["recovery_diag_label"].text, texts["recovery_diag_label"].state) == (
        "Recovery: counters could not be read", "warn",
    )
    assert (texts["input_health_label"].text, texts["input_health_label"].state) == (
        TEXT_SPECS["input_health_label"].text, "idle",
    )
    summary = texts["health_summary_label"]
    # Still running, the summary waits for the next reading rather than saying it stopped.
    assert (summary.text, summary.state) == (HEALTH_NOT_READ, "idle")
    assert texts["health_advice_label"].text == TEXT_SPECS["health_advice_label"].text


def test_resetting_the_health_labels_drops_the_lower_bound_note(isolated_main_window):
    window = isolated_main_window
    chip = window.shell_texts["latency_label"]
    chip.update(tip=chip.toolTip + LATENCY_LOWER_BOUND_NOTE)

    window._reset_health_labels()

    assert chip.toolTip == TEXT_SPECS["latency_label"].tip
    assert chip.text == TEXT_SPECS["latency_label"].text



def test_resetting_the_health_labels_puts_the_toolbar_latency_back(isolated_main_window):
    window = isolated_main_window
    readout = window.shell_texts["latency_readout_label"]
    readout.update(text="~24 ms", status="info", tip="A measured route.")

    window._reset_health_labels()

    assert (readout.text, readout.state) == (TEXT_SPECS["latency_readout_label"].text, "idle")
    assert readout.toolTip == TEXT_SPECS["latency_readout_label"].tip

def test_recovery_chip_says_the_restarts_are_paused_during_setup(qapp):
    window, _processor, args = _diagnostic_window({"recovery_suppressed": True})

    _judge(window, **args)

    chip = window.shell_texts["recovery_diag_label"]
    assert chip.text == "Recovery: 0 restarts, 0 output recoveries, paused during setup"
    assert chip.state == "info"


def test_output_drift_retiming_is_not_an_engine_fault(qapp, monkeypatch):
    # Two devices on separate clocks are kept in step by trimming samples; that is not a dropout.
    clock = [10.0]
    # The setup check reads the stream window on the module clock.
    monkeypatch.setattr("mic_eq.ui.health.time.monotonic", lambda: clock[0])
    diagnostics: dict[str, Any] = {"jitter_dropped_samples": 0}
    window, processor, args = _diagnostic_window(diagnostics)
    _judge(window, now=clock[0], **args)
    diagnostics["jitter_dropped_samples"] = 3_845
    _judge(window, now=clock[0], **args)
    dropped = window.shell_texts["dropped_label"]
    assert "reported a fault" not in dropped.text
    assert dropped.state == "ok"
    assert route_health_reason(processor, health_window=window.health_monitor.stream_health)[0]
    diagnostics["input_dropped_samples"] = 256
    _judge(window, now=clock[0], **args)
    assert "reported a fault" in dropped.text


@pytest.mark.parametrize(
    ("chips", "mute_error", "state", "summary", "advice"),
    [
        ({}, False, "idle", "Health: IDLE", "Start processing to see audio health."),
        (
            {"gate_health_label": ("Gate: OK", "ok")},
            False, "ok", "Health: OK", "Everything looks fine.",
        ),
        # Informational chips do not count towards the summary.
        (
            {"latency_label": ("Latency: about 10 ms (processing 0.2 ms)", "info")},
            False, "idle", "Health: IDLE", "Start processing to see audio health.",
        ),
        (
            {"gate_health_label": ("Gate: OK", "ok")},
            True, "warn", "Health: WARN", "Check the items marked below.",
        ),
        # The worst level wins, and the first chip in the reporting order
        # (decision chips, then the stream chips) names the issue.
        (
            {
                "recovery_diag_label": ("Recovery: 2 restarts, 0 output recoveries, restarted just now", "warn"),
                "buffer_label": ("Buffer: FILLING UP (1200 samples)", "warn"),
                "input_health_label": ("Input: CLIPPED (peak 0.0 dB, full scale 8 times)", "bad"),
            },
            False, "bad", "Input: CLIPPED (peak 0.0 dB, full scale 8 times)",
            advice_for("Input: CLIPPED (peak 0.0 dB, full scale 8 times)"),
        ),
        (
            {
                "recovery_diag_label": ("Recovery: 2 restarts, 0 output recoveries, restarted just now", "warn"),
                "buffer_label": ("Buffer: FILLING UP (1200 samples)", "warn"),
            },
            False, "warn", "Buffer: FILLING UP (1200 samples)",
            advice_for("Buffer: FILLING UP (1200 samples)"),
        ),
        (
            {"input_health_label": ("Input: STRANGE", "warn")},
            False, "warn", "Input: STRANGE", "Check the items marked below.",
        ),
        # The chip is as red as its worst row, so it names that row and its advice,
        # not an earlier row that only warns.
        (
            {
                "input_health_label": ("Input: PEAKS TOO HIGH (-2.0 dB)", "warn"),
                "output_health_label": ("Output: AT FULL SCALE (peak 1.5 dB)", "bad"),
            },
            False, "bad", "Output: AT FULL SCALE (peak 1.5 dB)",
            advice_for("Output: AT FULL SCALE (peak 1.5 dB)"),
        ),
    ],
)
def test_health_summary_and_advice_come_from_chip_states_alone(
    isolated_main_window, chips, mute_error, state, summary, advice
):
    window = isolated_main_window
    for key, (text, status) in chips.items():
        window._set_health_chip(key, text, status)
    if mute_error:  # the engine refuses the mute; the session keeps it pending
        window.engine_session.processor = SimpleNamespace(
            set_output_mute=Mock(side_effect=RuntimeError("unavailable"))
        )
        window.engine_session.set_user_muted(False)
        window.engine_session.processor = window.processor

    window._update_health_summary()

    summary_state = window.shell_texts["health_summary_label"]
    assert (summary_state.text, summary_state.state) == (summary, state)
    assert window.shell_texts["health_advice_label"].text == advice
    assert summary_state.toolTip == advice
    # The session summary was refreshed from the same call.
    assert ("Mute pending" in window.shell_texts["transmission_status_label"].text) is mute_error


def test_a_reset_drops_the_advice_and_tooltips_of_the_last_reading(isolated_main_window):
    window = isolated_main_window
    texts = window.shell_texts
    window._set_health_chip("input_health_label", "Input: CLIPPED (peak 0.0 dB, full scale 8 times)", "bad")
    window._update_health_summary()
    texts["recovery_diag_label"].update(tip="Last recovery: input callback error")
    assert texts["health_advice_label"].text != TEXT_SPECS["health_advice_label"].text
    assert texts["health_summary_label"].toolTip

    window._reset_health_labels()

    assert texts["health_advice_label"].text == TEXT_SPECS["health_advice_label"].text
    for key in ("health_summary_label", "recovery_diag_label", "dropped_label"):
        assert texts[key].toolTip == TEXT_SPECS[key].tip


def test_unreadable_telemetry_does_not_keep_the_tooltip_of_the_last_reading(isolated_main_window):
    window = isolated_main_window
    native = window.processor

    class Engine:
        def __getattr__(self, name):
            return getattr(native, name)

        def is_running(self) -> bool:
            return True

        def get_runtime_diagnostics(self):
            raise RuntimeError("no telemetry")

    window._set_health_chip("gate_health_label", "Gate: OK", "ok")
    window._update_health_summary()
    assert "fine" in window.shell_texts["health_summary_label"].toolTip
    window.processor = window.engine_session.processor = cast(Any, Engine())

    window._update_diagnostics()

    chip = window.shell_texts["health_summary_label"]
    assert "unavailable" in chip.text and chip.state == "warn"
    assert chip.toolTip == TEXT_SPECS["health_summary_label"].tip
    advice = window.shell_texts["health_advice_label"].text
    assert advice != TEXT_SPECS["health_advice_label"].text
    assert "could not read" in advice and "Stop and start processing" in advice


def test_the_login_start_up_message_uses_a_state_the_theme_knows(isolated_main_window):
    window = isolated_main_window
    window._login_startup_message = "Login waiting for saved audio endpoints"

    window._reset_health_labels()

    chip = window.shell_texts["health_summary_label"]
    assert chip.text == "Login waiting for saved audio endpoints"
    assert hasattr(PALETTE, f"status_{chip.state}_text")


def test_unchanged_health_chip_does_not_notify_the_view(qapp):
    state = TextState(qapp, text="", name="health_summary_label")
    notified = QSignalSpy(state.changed)
    window: Any = SimpleNamespace(shell_texts={"health_summary_label": state})
    for _ in range(30):
        MainWindow._set_health_chip(window, "health_summary_label", "Health: --", "idle")
    assert notified.count() == 1
    MainWindow._set_health_chip(window, "health_summary_label", "Input: -20dB", "idle")
    assert notified.count() == 2


def test_main_window_invalidates_all_live_meters_on_stop_and_getter_failure(qapp):
    from mic_eq import AudioProcessor
    from mic_eq.ui.gate_state import GateState
    from mic_eq.ui.level_meter import LevelMeter
    from mic_eq.ui.processing_meters import ProcessingMeters

    native = AudioProcessor()

    class Processor:
        running = True
        fail = False

        def __getattr__(self, name):
            return getattr(native, name)

        def is_running(self):
            return self.running

        def get_input_rms_db(self):
            if self.fail:
                raise RuntimeError("telemetry unavailable")
            return -8.0

        def get_input_peak_db(self):
            return -1.0

        def get_output_rms_db(self):
            return -10.0

        def get_output_peak_db(self):
            return -3.0

        def get_vad_probability(self):
            return 0.8

        def is_vad_available(self):
            return True

        def get_compressor_auto_makeup_enabled(self):
            return True

    processor = Processor()
    meter_host = QWidget()
    gate_state = GateState(processor, meter_host)
    window: Any = SimpleNamespace(
        processor=processor,
        engine_session=fake_session(),
        input_meter=LevelMeter(),
        output_meter=LevelMeter(),
        limiter_state=SimpleNamespace(get_applied_settings=lambda: {"enabled": True}),
        _processing_mode=lambda: "normal",
        gate_state=gate_state,
        processing_meters=ProcessingMeters(processor, gate_state, meter_host),
        _reset_health_labels=Mock(),
        health_monitor=HealthMonitor(),
        isHidden=lambda: True,
    )
    window._invalidate_live_meters = lambda: MainWindow._invalidate_live_meters(window)
    window._note_meter_peaks = lambda reading: MainWindow._note_meter_peaks(window, reading)
    window._update_input_analyser = lambda: MainWindow._update_input_analyser(window)
    window._clear_input_analyser = lambda: MainWindow._clear_input_analyser(window)
    try:
        MainWindow._update_meters(window)
        assert window.input_meter.measurement_available
        assert window.processing_meters.confidence.measurement_available
        processor.fail = True
        MainWindow._update_meters(window)
        assert not window.input_meter.measurement_available
        processor.fail = False
        MainWindow._update_meters(window)
        processor.running = False
        window._hidden_to_tray = True
        MainWindow._update_meters(window)
        assert not window.input_meter.measurement_available
        assert not window.output_meter.measurement_available
        assert not window.processing_meters.compressor_gr.measurement_available
        assert not window.processing_meters.deesser_gr.measurement_available
        assert not window.processing_meters.confidence.measurement_available
        assert window.processing_meters.text_states["current_lufs"].text == "--"
        assert window.processing_meters.text_states["current_makeup_gain"].text == "--"
        assert "--" in window.gate_state.presentation()["noise_floor"]
    finally:
        native.stop()
        for item in (
            window.input_meter,
            window.output_meter,
            meter_host,
        ):
            item.deleteLater()
        qapp.processEvents()


def test_failed_diagnostics_still_service_recovery_when_presentation_is_hidden(qapp):
    window: Any = SimpleNamespace(
        processor=SimpleNamespace(
            is_running=lambda: True,
            get_runtime_diagnostics=Mock(side_effect=RuntimeError("no telemetry")),
        ),
        _hidden_to_tray=True,
        _reset_health_labels=Mock(),
        _set_health_chip=Mock(),
        engine_session=Mock(view=SimpleNamespace(in_use=False, output_muted=False)),
        _stop_for_own_microphone=Mock(return_value=False),
        shell_texts={"health_advice_label": Mock()},
        health_monitor=HealthMonitor(),
        input_meter=Mock(),
        output_meter=Mock(),
        processing_meters=Mock(),
        limiter_state=SimpleNamespace(get_settings=lambda: {"enabled": True}),
        _processing=False,
    )
    window._show_health = lambda view: MainWindow._show_health(window, view)
    MainWindow._update_diagnostics(window)
    window._reset_health_labels.assert_called_once()
    window.engine_session.tick.assert_called_once()
    assert "unavailable" in window._set_health_chip.call_args.args[1]
    advice = window.shell_texts["health_advice_label"].update.call_args.kwargs["text"]
    assert "could not read" in advice and "Stop and start processing" in advice


def test_a_stopped_tick_redoes_the_stopped_work_only_when_the_stopped_state_changes(qapp):
    state: dict[str, Any] = {"running": False, "restart_pending": False, "start_enabled": True}
    window: Any = SimpleNamespace(
        processor=SimpleNamespace(is_running=lambda: state["running"]),
        start_btn_action=SimpleNamespace(isEnabled=lambda: state["start_enabled"]),
        _restart_pending=lambda: state["restart_pending"],
        _stop_for_own_microphone=Mock(return_value=False),
        _sync_meter_timer=Mock(),
        _reset_health_labels=Mock(),
        _sync_processing_controls=Mock(),
        _update_session_summary=Mock(),
        engine_session=Mock(view=SimpleNamespace(in_use=False, output_muted=False)),
        _set_health_chip=Mock(),
        shell_texts={"health_advice_label": Mock()},
        _processing=False,
        health_monitor=HealthMonitor(
            memory=HealthMemory(device_wait_since=1.0, backend_warning="Noise suppression stopped")
        ),
        input_meter=Mock(),
        output_meter=Mock(),
        processing_meters=Mock(),
        limiter_state=SimpleNamespace(get_settings=lambda: {"enabled": True}),
    )
    window._show_health = lambda view: MainWindow._show_health(window, view)
    work =(window._sync_meter_timer, window._reset_health_labels, window._sync_processing_controls)

    def ticks(count: int = 1) -> list[int]:
        for _ in range(count):
            MainWindow._update_diagnostics(window)
        return [method.call_count for method in work]

    assert ticks() == [1, 1, 1]
    assert window.health_monitor.memory == HealthMemory()  # a stop ends any device wait and backend warning
    assert ticks(3) == [1, 1, 1]  # still stopped, nothing changed
    # A pending restart is serviced on every tick all the same, and edits made
    # while stopped (route, preset, mode, mute) still reach the summary.
    assert window.engine_session.tick.call_count == 4
    assert window._update_session_summary.call_count == 3

    state["restart_pending"] = True  # the Stop control now has a retry to cancel
    assert ticks(2) == [2, 2, 2]
    window._login_startup_message = "Login waiting for a microphone"  # the summary shows it
    assert ticks(2) == [3, 3, 3]
    # A start whose engine stopped again before any running tick leaves Start
    # disabled; the next stopped tick resyncs the controls.
    state["start_enabled"] = False
    assert ticks() == [4, 4, 4]
    state["start_enabled"] = True
    assert ticks(2) == [5, 5, 5]

    # A running tick (here one whose readings fail) ends the stopped state; the next stop redoes the work.
    state["running"] = True
    assert ticks() == [6, 6, 5]
    state["running"] = False
    assert ticks(2) == [7, 7, 6]
    assert window.engine_session.tick.call_count == 14  # every tick, the running one included


def test_measurements_begin_unavailable_and_failed_noise_floor_does_not_invent_default(
    qapp,
):
    from mic_eq import AudioProcessor
    from mic_eq.ui.gate_state import GateState
    from mic_eq.ui.processing_meters import ProcessingMeters

    native = AudioProcessor()

    class Processor:
        def __getattr__(self, name):
            return getattr(native, name)

        # The noise floor is only read while a VAD backend is present.
        def is_vad_available(self):
            return True

        def get_noise_floor(self):
            raise RuntimeError("missing noise floor")

    processor = Processor()
    gate = GateState(processor)
    meters = ProcessingMeters(processor, gate)
    try:
        assert meters.text_states["current_lufs"].text == "--"
        assert meters.text_states["current_makeup_gain"].text == "--"
        meters.update_current_release()
        assert meters.text_states["current_release"].text == "--"
        assert gate.presentation()["noise_floor"] == "Noise Floor: --"
        gate.update_vad_confidence(0.5)
        assert gate.confidence == 0.5
        assert gate.presentation()["noise_floor"] == "Noise Floor: --"
    finally:
        native.stop()
