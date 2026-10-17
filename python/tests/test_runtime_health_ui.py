"""Live health expires recovered faults and stops presenting stale measurements."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QObject
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QWidget

from mic_eq.ui.first_run_setup_dialog import route_health_reason
from mic_eq.ui.health import RecentStreamHealth, advice_for
from mic_eq.ui.main_window import MainWindow
from mic_eq.ui.shell_specs import TEXT_SPECS
from mic_eq.ui.shell_state import TextState
from mic_eq.ui.stream_recovery import StreamRecoveryManager
from mic_eq.ui.theme import PALETTE


def test_presentation_timer_pauses_without_stopping_diagnostics(qapp):
    from PySide6.QtCore import QEventLoop, QTimer

    state = {"running": False, "visible": True, "minimized": False}
    ticks = []
    meter_timer, diagnostics_timer = QTimer(), QTimer()
    meter_timer.setInterval(16)
    meter_timer.timeout.connect(lambda: ticks.append(1))
    diagnostics_timer.start(250)
    window: Any = SimpleNamespace(
        processor=SimpleNamespace(is_running=lambda: state["running"]),
        isVisible=lambda: state["visible"],
        isMinimized=lambda: state["minimized"],
        meter_timer=meter_timer,
        _invalidate_live_meters=Mock(),
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
            ticks.clear()
            MainWindow._sync_meter_timer(window)
            loop = QEventLoop()
            QTimer.singleShot(80, loop.quit)
            loop.exec()
            should_tick = running and visible and not minimized
            assert bool(ticks) == should_tick
            assert meter_timer.isActive() == should_tick
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


def test_recovered_stream_clears_live_health_but_keeps_lifetime_counts(
    qapp, monkeypatch
):
    clock = [10.0]
    monkeypatch.setattr("mic_eq.ui.health.time.monotonic", lambda: clock[0])
    diagnostics = {
        "input_callback_error_count": 1,
        "stream_restart_count": 1,
        "input_dropped_samples": 256,
        "last_restart_reason": "input callback error",
    }
    processor = SimpleNamespace(
        is_running=lambda: True,
        get_runtime_diagnostics=lambda: diagnostics,
        get_input_callback_age_ms=lambda: 0,
        get_output_callback_age_ms=lambda: 0,
    )
    state_owner = QObject()
    window: Any = SimpleNamespace(
        _stream_health=RecentStreamHealth(),
        _stream_recovery=StreamRecoveryManager(),
        _last_backend_warning=None,
        processor=processor,
        _update_session_summary=Mock(),
        _shell_state_owner=state_owner,
        shell_texts={},
    )
    names = (
        "latency_label",
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
    texts = window.shell_texts
    window._set_health_chip = lambda key, text, state: MainWindow._set_health_chip(
        window, key, text, state
    )
    window._update_health_summary = lambda: MainWindow._update_health_summary(window)
    window._extend_diag_tokens = MainWindow._extend_diag_tokens
    args: dict[str, Any] = dict(
        diagnostics=diagnostics,
        latency_ms=10.0,
        dsp_time_ms=0.2,
        input_buf=0,
        output_buf=128,
        rnnoise_buf=0,
        input_rms_db=-20.0,
        output_rms_db=-20.0,
        input_callback_age_ms=0,
        output_callback_age_ms=0,
    )
    MainWindow._update_diagnostic_labels(window, **args)
    assert texts["health_summary_label"].state == "warn"
    assert not route_health_reason(processor, health_window=window._stream_health)[0]
    clock[0] += 5.1
    MainWindow._update_diagnostic_labels(window, **args)
    assert texts["health_summary_label"].state == "ok"
    assert texts["recovery_diag_label"].state == "ok"
    assert "1 restart" in texts["recovery_diag_label"].text
    assert "ICE:1" in texts["dropped_label"].toolTip
    assert route_health_reason(processor, health_window=window._stream_health)[0]
    assert diagnostics["input_callback_error_count"] == 1
    assert diagnostics["last_restart_reason"] == "input callback error"
    diagnostics["input_callback_error_count"] += 1
    MainWindow._update_diagnostic_labels(window, **args)
    assert texts["health_summary_label"].state == "warn"


@pytest.mark.parametrize(
    ("chips", "mute_error", "state", "summary", "advice"),
    [
        ({}, False, "idle", "Health: IDLE", TEXT_SPECS["health_advice_label"].text),
        (
            {"gate_health_label": ("Gate: OK", "ok")},
            False, "ok", "Health: OK", "Everything looks fine.",
        ),
        # Informational chips do not count towards the summary.
        (
            {"latency_label": ("Latency: about 10 ms (processing 0.2 ms)", "info")},
            False, "idle", "Health: IDLE", TEXT_SPECS["health_advice_label"].text,
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
    window._output_mute_error = "unavailable" if mute_error else None

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
    window.processor = cast(Any, Engine())

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
        input_meter=LevelMeter(),
        output_meter=LevelMeter(),
        limiter_state=SimpleNamespace(get_applied_settings=lambda: {"enabled": True}),
        _processing_mode=lambda: "normal",
        gate_state=gate_state,
        processing_meters=ProcessingMeters(processor, gate_state, meter_host),
        _reset_health_labels=Mock(),
        isHidden=lambda: True,
    )
    window._invalidate_live_meters = lambda: MainWindow._invalidate_live_meters(window)
    window._note_meter_peaks = lambda reading: MainWindow._note_meter_peaks(window, reading)
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
        _service_stream_recovery=Mock(),
        shell_texts={"health_advice_label": Mock()},
    )
    MainWindow._update_diagnostics(window)
    window._reset_health_labels.assert_called_once()
    window._service_stream_recovery.assert_called_once()
    assert "unavailable" in window._set_health_chip.call_args.args[1]
    advice = window.shell_texts["health_advice_label"].update.call_args.kwargs["text"]
    assert "could not read" in advice and "Stop and start processing" in advice


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
