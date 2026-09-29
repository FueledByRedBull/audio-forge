"""Live health expires recovered faults and stops presenting stale measurements."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

from PyQt6.QtWidgets import QLabel

from mic_eq.ui.first_run_setup_dialog import route_health_reason
from mic_eq.ui.health import RecentStreamHealth
from mic_eq.ui.main_window import MainWindow


def test_presentation_timer_pauses_without_stopping_diagnostics(qapp):
    from PyQt6.QtCore import QEventLoop, QTimer

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
    finally:
        meter_timer.stop()
        diagnostics_timer.stop()


def test_window_state_change_pauses_and_restarts_fast_meter_timer(qapp):
    from PyQt6.QtCore import QEvent, QTimer
    from PyQt6.QtWidgets import QMainWindow

    window = MainWindow.__new__(MainWindow)
    QMainWindow.__init__(window)
    state = {"minimized": False}
    cast(Any, window).processor = SimpleNamespace(is_running=lambda: True)
    window.meter_timer = QTimer(window)
    window.diagnostics_timer = QTimer(window)
    window.diagnostics_timer.start(250)
    window._invalidate_live_meters = Mock()
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

        state["minimized"] = False
        window.changeEvent(QEvent(QEvent.Type.WindowStateChange))
        assert window.meter_timer.isActive()
        assert window.diagnostics_timer.isActive()
    finally:
        window.meter_timer.stop()
        window.diagnostics_timer.stop()
        window.deleteLater()
        qapp.processEvents()


def test_unavailable_vad_does_not_present_default_noise_floor(qapp):
    from mic_eq import AudioProcessor
    from mic_eq.ui.gate_panel import GatePanel

    native = AudioProcessor()

    class Processor:
        def __getattr__(self, name):
            return getattr(native, name)

        def is_vad_available(self):
            return False

        def get_noise_floor(self):
            return -60.0

    panel = GatePanel(Processor())
    try:
        panel.update_vad_confidence(0.0)
        assert panel.noise_floor_label.text() == "Noise Floor: --"
        assert not panel.confidence_meter.measurement_available
    finally:
        native.stop()
        panel.deleteLater()
        qapp.processEvents()


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
    window: Any = SimpleNamespace(
        _stream_health=RecentStreamHealth(),
        _last_backend_warning=None,
        processor=processor,
        _update_session_summary=Mock(),
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
    for name in names:
        setattr(window, name, QLabel())
    window._health_decision_widgets = [getattr(window, name) for name in names[:-1]]
    window._health_layout_widgets = []
    window._set_health_chip = lambda label, text, state: MainWindow._set_health_chip(
        window, label, text, state
    )
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
    assert window.health_summary_label.property("health_state") == "warn"
    assert not route_health_reason(processor, health_window=window._stream_health)[0]
    clock[0] += 5.1
    MainWindow._update_diagnostic_labels(window, **args)
    assert window.health_summary_label.property("health_state") == "ok"
    assert window.recovery_diag_label.property("health_state") == "ok"
    assert "R:1" in window.recovery_diag_label.text()
    assert "ICE:1" in window.dropped_label.toolTip()
    assert route_health_reason(processor, health_window=window._stream_health)[0]
    assert diagnostics["input_callback_error_count"] == 1
    assert diagnostics["last_restart_reason"] == "input callback error"
    diagnostics["input_callback_error_count"] += 1
    MainWindow._update_diagnostic_labels(window, **args)
    assert window.health_summary_label.property("health_state") == "warn"


def test_unchanged_health_chip_does_not_reapply_qt_styles(qapp, monkeypatch):
    label = QLabel()
    calls = dict(text=0, style=0, property=0)
    original_text, original_style, original_property = (
        label.setText,
        label.setStyleSheet,
        label.setProperty,
    )

    def text(value):
        calls["text"] += 1
        original_text(value)

    def style(value):
        calls["style"] += 1
        original_style(value)

    def prop(name, value):
        calls["property"] += 1
        return original_property(name, value)

    monkeypatch.setattr(label, "setText", text)
    monkeypatch.setattr(label, "setStyleSheet", style)
    monkeypatch.setattr(label, "setProperty", prop)
    window: Any = None
    for _ in range(30):
        MainWindow._set_health_chip(window, label, "Health: --", "idle")
    assert calls == dict(text=1, style=1, property=1)
    MainWindow._set_health_chip(window, label, "Input: -20dB", "idle")
    assert calls == dict(text=2, style=1, property=1)


def test_main_window_invalidates_all_live_meters_on_stop_and_getter_failure(qapp):
    from mic_eq import AudioProcessor
    from mic_eq.ui.compressor_panel import CompressorPanel
    from mic_eq.ui.deesser_panel import DeEsserPanel
    from mic_eq.ui.gate_panel import GatePanel
    from mic_eq.ui.level_meter import LevelMeter

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
    window: Any = SimpleNamespace(
        processor=processor,
        input_meter=LevelMeter(),
        output_meter=LevelMeter(),
        compressor_panel=CompressorPanel(processor),
        gate_panel=GatePanel(processor),
        deesser_panel=DeEsserPanel(processor),
        _reset_health_labels=Mock(),
        isHidden=lambda: True,
    )
    window._invalidate_live_meters = lambda: MainWindow._invalidate_live_meters(window)
    try:
        MainWindow._update_meters(window)
        assert window.input_meter.measurement_available
        assert window.gate_panel.confidence_meter.measurement_available
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
        assert not window.compressor_panel.gr_meter.measurement_available
        assert not window.deesser_panel.gr_meter.measurement_available
        assert not window.gate_panel.confidence_meter.measurement_available
        assert window.compressor_panel.current_lufs_label.text() == "--"
        assert window.compressor_panel.current_makeup_gain_label.text() == "--"
        assert "--" in window.gate_panel.noise_floor_label.text()
    finally:
        native.stop()
        for item in (
            window.input_meter,
            window.output_meter,
            window.compressor_panel,
            window.gate_panel,
            window.deesser_panel,
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
        health_summary_label=QLabel(),
        _service_stream_recovery=Mock(),
    )
    MainWindow._update_diagnostics(window)
    window._reset_health_labels.assert_called_once()
    window._service_stream_recovery.assert_called_once()
    assert "unavailable" in window._set_health_chip.call_args.args[1]


def test_panel_measurements_begin_unavailable_and_failed_noise_floor_does_not_invent_default(
    qapp,
):
    from mic_eq import AudioProcessor
    from mic_eq.ui.compressor_panel import CompressorPanel
    from mic_eq.ui.gate_panel import GatePanel

    native = AudioProcessor()

    class Processor:
        def __getattr__(self, name):
            return getattr(native, name)

        def get_noise_floor(self):
            raise RuntimeError("missing noise floor")

    processor = Processor()
    compressor, gate = CompressorPanel(processor), GatePanel(processor)
    try:
        assert compressor.current_lufs_label.text() == "--"
        assert compressor.current_makeup_gain_label.text() == "--"
        compressor._update_current_release()
        assert compressor.current_release_label.text() == "--"
        assert gate.noise_floor_label.text() == "Noise Floor: --"
        gate.update_vad_confidence(0.5)
        assert gate.noise_floor_label.text() == "Noise Floor: --"
    finally:
        native.stop()
        compressor.deleteLater()
        gate.deleteLater()
        qapp.processEvents()
