"""The Qt Quick view renders and edits the shared settings."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from typing import Any

import pytest
from PySide6.QtCore import (
    QCoreApplication,
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRectF,
    QSignalBlocker,
    Qt,
    QTimer,
    qInstallMessageHandler,
)
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QKeyEvent, QMouseEvent, QPainter
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtQml import QQmlProperty
from PySide6.QtQuick import QSGRendererInterface
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QMessageBox

from mic_eq.config import AppConfig, DeviceIdentity
from mic_eq.config_parts.settings import q_from_bandwidth_octaves
from mic_eq.ui import main_window
from mic_eq.ui.main_window import PAGE_NAMES, MainWindow
from mic_eq.ui.eq_quick_graph import EQGraphItem
from mic_eq.ui.eq_graph import EQGraphGeometry
from mic_eq.ui.level_meter import CLIP_FLAT_TOP_SAMPLES
from mic_eq.ui.quick_meter import QuickMeterItem
from mic_eq.ui.quick_shell import SettingControl
from mic_eq.ui.theme import PALETTE


@pytest.fixture
def quick_window(qapp, monkeypatch):
    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
    monkeypatch.setattr("mic_eq.ui.main_window.list_presets", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_input_devices", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_output_devices", lambda: [])
    scene_messages: list[str] = []
    previous = qInstallMessageHandler(
        lambda _mode, _context, message: scene_messages.append(message)
        if "Main.qml" in message
        else None
    )
    window = MainWindow()
    window.show()
    qapp.processEvents()
    qInstallMessageHandler(previous)
    vars(window)["scene_messages"] = scene_messages
    return window


def _rows(card: dict) -> dict:
    return {row["label"]: row for row in card["rows"] + card["advanced"]}


@pytest.mark.parametrize("name", ["input", "output", "compressor", "deesser", "eq"])
def test_live_data_changes_repaint_existing_qml_items(quick_window, name) -> None:
    window = quick_window
    window.resize(1500, 1000)
    # Supply telemetry without opening audio devices or running stopped-stream refreshes.
    for timer in window.findChildren(QTimer):
        timer.stop()
    target = {
        "input": window.input_meter,
        "output": window.output_meter,
        "compressor": window.processing_meters.compressor_gr,
        "deesser": window.processing_meters.deesser_gr,
        "eq": window.eq_presentation.graph_model,
    }[name]
    view = window.centralWidget()
    pending = [view.rootObject()]
    item = None
    while pending:
        child = pending.pop()
        if isinstance(child, QuickMeterItem) and child.property("source") is target:
            item = child
            break
        if isinstance(child, EQGraphItem) and child.graph is target:
            item = child
            break
        pending.extend(child.childItems())
    assert item is not None
    item.ensurePolished()
    item.forceActiveFocus()

    def snapshot():
        QTest.qWait(75)
        frame = view.grabFramebuffer()
        assert not frame.isNull()
        rect = item.mapRectToScene(QRectF(0, 0, item.width(), item.height()))
        assert view.rect().contains(rect.toAlignedRect())
        scale = frame.width() / view.width()
        return frame.copy(QRectF(rect.x() * scale, rect.y() * scale,
                                rect.width() * scale, rect.height() * scale).toAlignedRect())

    previous = snapshot()
    for level in (-30.0, -6.0, -45.0):
        if name in {"input", "output"}:
            target.set_levels(level, level + 3.0)
        elif name == "eq":
            target.set_all_params([("bell", 1000.0, level / 5.0, 1.41, 12, True)])
        else:
            target.set_gain_reduction(-level / 3.0)
        current = snapshot()
        assert current != previous, f"{name} did not repaint after its data changed"
        previous = current


@pytest.mark.parametrize(
    ("item", "slot"),
    [(QuickMeterItem, "_source_changed()"), (EQGraphItem, "_update_graph()")],
)
def test_qml_item_callbacks_are_registered_slots(item, slot) -> None:
    # A callback PySide registers at connect time is never called once the
    # scene declares a property on the element.
    assert item.staticMetaObject.indexOfSlot(slot) >= 0


def test_transport_follows_the_stop_action(quick_window) -> None:
    window = quick_window
    root = window.centralWidget().rootObject()
    assert root.property("running") is False
    # The action states _start_processing and _stop_processing set.
    window.start_btn_action.setEnabled(False)
    window.stop_btn_action.setEnabled(True)
    assert root.property("running") is True
    window.start_btn_action.setEnabled(True)
    window.stop_btn_action.setEnabled(False)
    assert root.property("running") is False


def test_a_quick_session_keeps_the_scene_from_start_to_close(quick_window, monkeypatch) -> None:
    """Start, run, change pages and close with the scene as the window's content."""

    window = quick_window
    refreshes: list[int] = []
    monkeypatch.setattr(
        window, "_refresh_login_startup_action", lambda: refreshes.append(window.shell_page.currentIndex())
    )

    def assert_scene_shown() -> None:
        assert isinstance(window.centralWidget(), QQuickWidget)
        assert window.quick_view_failure is None

    def text(key: str) -> str:
        return window.shell_texts[key].text

    assert_scene_shown()
    before = window._get_current_preset()
    window.apply_processing_configuration(before)
    window._commit_pending_configuration_snapshot()
    window.input_meter.set_levels(-30.0, -20.0)
    window.output_meter.set_levels(-24.0, -12.0)
    window._update_meters()
    window._update_diagnostics()
    window._update_diagnostic_labels(
        diagnostics={}, latency_ms=10.0, dsp_time_ms=0.2, input_buf=0, output_buf=0, rnnoise_buf=0,
    )
    window._set_health_chip("input_health_label", "Input: hot", "warn")
    window.quick_bridge.stepBand(1)
    window.quick_bridge._sample_levels()
    window.status_message.showMessage("Probe")
    window.shell_page.setIndex(window.SETTINGS_PAGE_INDEX)
    window.shell_page.setIndex(0)
    window.processing_mode_choice.setIndex(1)
    assert text("transmission_status_label") == "Transmission: Stopped / Bypass"
    window.processing_mode_choice.setIndex(0)
    assert text("transmission_status_label") == "Transmission: Stopped / Normal"
    window.shell_texts["config_warning_banner"].update(text="Warning", shown=True)

    # A transport cycle with the audio engine faked: no device is opened.
    native = window.processor

    class Engine:
        running = False

        def __getattr__(self, name):
            return getattr(native, name)

        def is_running(self) -> bool:
            return self.running

        def stop(self) -> None:
            self.running = False

        def set_noise_model(self, _model: str) -> bool:
            return True

    engine: Any = Engine()

    def start(processor: Engine, *_devices: object) -> str:
        processor.running = True
        return "started"

    monkeypatch.setattr("mic_eq.ui.main_window.start_processor_for_route", start)
    window.processor = engine
    noise = window.noise_suppression_state
    monkeypatch.setattr(noise, "processor", engine)
    monkeypatch.setattr(noise, "models", (*noise.models, ("deepfilter-ll", "DeepFilterNet LL")))

    # A device change updates the route text.
    window.input_choice.addItem("Microphone", DeviceIdentity(name="Microphone", direction="input"))
    window.output_choice.addItem("Destination", DeviceIdentity(name="Destination", direction="output"))
    window.output_choice.setCurrentIndex(window.output_choice.count() - 1)
    assert text("route_status_label") == "Route: Microphone -> Destination"
    window.input_choice.addItem("Headset", DeviceIdentity(name="Headset", direction="input"))
    window.input_choice.setCurrentIndex(window.input_choice.count() - 1)
    assert text("route_status_label") == "Route: Headset -> Destination"
    monkeypatch.setattr(window, "_apply_input_preferences_for_current_route", lambda: None)

    window.user_mute_checkbox_action.setChecked(True)
    assert text("transmission_status_label") == "Transmission: Stopped / Normal / Muted"
    window.user_mute_checkbox_action.setChecked(False)
    assert text("transmission_status_label") == "Transmission: Stopped / Normal"

    # Applying a preset makes it the saved one; a noise-model edit then modifies it.
    preset = window._get_current_preset()
    preset.name = "Probe"
    assert window._apply_preset(preset)
    assert text("preset_status_label") == "Preset: Probe (saved)"
    noise.set_value("model", "deepfilter-ll")
    assert text("preset_status_label") == "Preset: Probe (unsaved changes)"

    window.start_btn_action.trigger()
    assert engine.running
    assert text("transmission_status_label") == "Transmission: Running / Normal"
    assert window.stop_btn_action.isEnabled() and not window.start_btn_action.isEnabled()
    window._update_meters()
    window._update_diagnostics()
    window.stop_btn_action.trigger()
    assert not engine.running
    assert text("transmission_status_label") == "Transmission: Stopped / Normal"
    assert window.start_btn_action.isEnabled() and not window.stop_btn_action.isEnabled()
    window._update_meters()

    assert refreshes == [window.SETTINGS_PAGE_INDEX]
    assert window._get_current_preset().eq.to_dict() == before.eq.to_dict()
    assert_scene_shown()
    window._quitting = True
    window.close()
    assert_scene_shown()


def test_failed_gate_initialization_retains_configuration_recovery_barrier(qapp, monkeypatch) -> None:
    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
    for name in ("list_presets", "list_input_devices", "list_output_devices"):
        monkeypatch.setattr(f"mic_eq.ui.main_window.{name}", lambda: [])

    def fail_gate(*_args):
        raise RuntimeError("gate unavailable during initialization and recovery")

    monkeypatch.setattr("mic_eq.ui.gate_state.GateState._write", fail_gate)
    window = MainWindow()
    assert "configuration" in window._temporary_mute_reasons
    assert not window.processor.is_running()


def test_deesser_controls_share_exact_settings_and_derived_availability(quick_window) -> None:
    bridge = quick_window.quick_bridge
    rows = _rows(bridge.model["stages"][2])
    state = quick_window.deesser_state
    state.set_settings({"auto_enabled": False, "attack_ms": 2.125})
    attack = rows["Attack"]["proxy"]
    assert attack.value == 2.13
    rows["Auto"]["proxy"].click()
    rows["Auto"]["proxy"].released()
    assert not rows["Threshold"]["proxy"].enabled
    assert not rows["Ratio"]["proxy"].enabled
    rows["Low cut"]["proxy"].setValue(12000)
    rows["Low cut"]["proxy"].released()
    assert rows["High cut"]["proxy"].value == 12200
    assert state.get_settings()["attack_ms"] == 2.125
    assert quick_window._get_current_preset().deesser.high_cut_hz == 12200


def test_limiter_reads_shared_state_without_polling(quick_window) -> None:
    bridge = quick_window.quick_bridge
    control = _rows(bridge.model["stages"][-1])["Ceiling"]["proxy"]
    assert isinstance(control, SettingControl)
    assert not hasattr(bridge, "_timer")
    seen = []
    control.changed.connect(lambda: seen.append(control.value))

    quick_window.limiter_state.set_settings({"ceiling_db": -3.257})
    assert seen[-1] == -3.26
    # The control shows a rounded value; the exact one stays the product state.
    assert quick_window._get_current_preset().limiter.ceiling_db == -3.257


def test_limiter_quick_edits_preserve_exact_preset_and_undo(quick_window) -> None:
    preset = quick_window._get_current_preset()
    preset.limiter.ceiling_db = -3.257
    quick_window.apply_processing_configuration(preset)
    quick_window._commit_pending_configuration_snapshot(label="Exact limiter preset")
    control = _rows(quick_window.quick_bridge.model["stages"][-1])["Ceiling"]["proxy"]
    control.setText("-2.25 dB")
    assert quick_window._get_current_preset().limiter.ceiling_db == -2.25
    quick_window.undo_configuration()
    assert quick_window._get_current_preset().limiter.ceiling_db == -3.257
    assert control.value == -3.26
    quick_window.redo_configuration()
    assert control.value == -2.25
    assert quick_window._get_current_preset().limiter.ceiling_db == -2.25


@pytest.mark.parametrize(("value", "rounded"), [(-1.125, -1.13), (-2.125, -2.13), (-2.005, -2.0)])
def test_limiter_uses_qt_rounding(quick_window, value, rounded) -> None:
    state = quick_window.limiter_state
    control = _rows(quick_window.quick_bridge.model["stages"][-1])["Ceiling"]["proxy"]
    state.set_settings({"ceiling_db": value})
    assert control.value == rounded
    assert state.get_settings()["ceiling_db"] == value
    control.setValue(value)
    control.released()
    assert state.get_settings()["ceiling_db"] == rounded


def test_limiter_quick_rejected_write_restores_state_and_reports_failure(quick_window) -> None:
    state = quick_window.limiter_state
    native = state.processor

    class RejectCeiling:
        def __getattr__(self, name):
            return getattr(native, name)

        def set_limiter_ceiling(self, value):
            if value == -2.0:
                raise RuntimeError("test limiter write refused")
            native.set_limiter_ceiling(value)

    before = state.get_settings()
    state.processor = RejectCeiling()
    control = _rows(quick_window.quick_bridge.model["stages"][-1])["Ceiling"]["proxy"]
    control.setText("-2 dB")
    assert state.get_settings() == before
    assert control.value == round(float(before["ceiling_db"]), 2)
    assert "test limiter write refused" in quick_window.status_message.text
    control.setText("not a number")
    assert state.get_settings() == before


def test_pending_limiter_failure_cannot_enter_history(quick_window) -> None:
    state = quick_window.limiter_state
    native = state.processor

    class RejectCeiling:
        def __getattr__(self, name):
            return getattr(native, name)

        def set_limiter_ceiling(self, value):
            if value == -2.0:
                raise RuntimeError("pending write refused")
            native.set_limiter_ceiling(value)

    state.set_settings({})
    state._rate_limiter.interval_ms = 5000
    state.processor = RejectCeiling()
    before = quick_window._configuration_history.current
    state.set_value("ceiling_db", -2.0)
    assert state.get_settings()["ceiling_db"] == -2.0
    assert not quick_window._commit_pending_configuration_snapshot()
    assert quick_window._configuration_history.current == before
    assert state.get_settings()["ceiling_db"] != -2.0


def test_failed_limiter_recovery_blocks_restart_until_configuration_reapplied(quick_window) -> None:
    state = quick_window.limiter_state
    native = state.processor
    preset = quick_window._get_current_preset()

    class RejectWrites:
        def __getattr__(self, name):
            return getattr(native, name)

        def set_limiter_ceiling(self, _value):
            raise RuntimeError("ceiling unavailable during rollback too")

    state.processor = RejectWrites()
    control = _rows(quick_window.quick_bridge.model["stages"][-1])["Ceiling"]["proxy"]
    control.setText("-2 dB")
    assert "configuration" in quick_window._temporary_mute_reasons
    assert not quick_window._start_processing(interactive=False)
    assert "recovery failed" in quick_window.status_message.text
    assert not quick_window._commit_pending_configuration_snapshot()
    state.processor = native
    quick_window.apply_processing_configuration(preset)
    assert "configuration" not in quick_window._temporary_mute_reasons


def test_scene_loads_cleanly(quick_window) -> None:
    view = quick_window.centralWidget()
    assert isinstance(view, QQuickWidget)
    assert view.status() == QQuickWidget.Status.Ready
    assert view.errors() == []
    # Binding errors are only warnings to Qt; here they fail the test.
    assert vars(quick_window)["scene_messages"] == []
    # Meters age their own peak holds, so nothing needs a visible widget to run.
    meter = quick_window.input_meter
    meter.set_levels(-30.0, -0.2)
    assert meter.clip_flash_counter == 10 and meter.decay_timer.isActive()


def test_the_menu_bar_stays_hidden_and_the_window_can_shrink_to_900_by_640(quick_window) -> None:
    assert quick_window.menuBar().isHidden()
    assert quick_window.minimumWidth() <= 900
    assert quick_window.minimumHeight() <= 640


def test_the_page_state_ignores_an_index_outside_the_pages(quick_window) -> None:
    page = quick_window.shell_page
    page.setIndex(len(PAGE_NAMES))
    page.setIndex(-1)
    assert page.currentIndex() == 0
    page.setIndex(quick_window.HEALTH_PAGE_INDEX)
    page.setIndex(len(PAGE_NAMES))
    assert page.currentIndex() == quick_window.HEALTH_PAGE_INDEX


def test_field_proxy_writes_through_to_the_gate_state(quick_window) -> None:
    gate = _rows(quick_window.quick_bridge.model["stages"][1])
    threshold = gate["Threshold"]["proxy"]
    state = quick_window.gate_state

    threshold.setValue(-33.0)
    assert state.get_settings()["threshold_db"] == pytest.approx(-33.0)
    assert threshold.value == pytest.approx(-33.0)

    threshold.setText("-28 dB")
    assert state.get_settings()["threshold_db"] == pytest.approx(-28.0)
    threshold.setText("not a number")
    assert state.get_settings()["threshold_db"] == pytest.approx(-28.0)

    threshold.setValue(1000.0)
    assert state.get_settings()["threshold_db"] == threshold.maximum


def test_toggle_combo_and_page_proxies(quick_window) -> None:
    model = quick_window.quick_bridge.model
    gate_card = model["stages"][1]
    state = quick_window.gate_state
    before = state.get_settings()["enabled"]
    gate_card["toggle"].click()
    assert state.get_settings()["enabled"] is not before
    assert gate_card["toggle"].checked is state.get_settings()["enabled"]

    mode = model["top"]["mode"]
    choice = quick_window.processing_mode_choice
    assert mode.items == [choice.itemText(i) for i in range(choice.count())]
    mode.setIndex(1)
    assert choice.currentData() == "bypass" and quick_window.processor.is_bypass()

    model["page"].setIndex(MainWindow.HEALTH_PAGE_INDEX)
    assert quick_window.shell_page.currentIndex() == MainWindow.HEALTH_PAGE_INDEX


def test_health_notifies_without_widget_polling(quick_window) -> None:
    bridge = quick_window.quick_bridge
    status = bridge.model["health"]["signal"][0]["proxy"]
    seen: list[str] = []
    status.changed.connect(lambda: seen.append(status.text))
    quick_window._set_health_chip("input_health_label", "Input: hot", "warn")
    assert seen == ["Input: hot"]
    assert status.state == "warn"


def test_opening_settings_refreshes_the_login_shortcut(quick_window, monkeypatch) -> None:
    window = quick_window
    opened: list[int] = []
    monkeypatch.setattr(
        window, "_refresh_login_startup_action", lambda: opened.append(window.shell_page.currentIndex())
    )
    page = window.quick_bridge.model["page"]
    page.setIndex(window.HEALTH_PAGE_INDEX)
    assert opened == []
    page.setIndex(window.SETTINGS_PAGE_INDEX)
    page.setIndex(0)
    page.setIndex(window.SETTINGS_PAGE_INDEX)
    assert opened == [window.SETTINGS_PAGE_INDEX] * 2


def test_every_stage_is_in_the_model(quick_window) -> None:
    model = quick_window.quick_bridge.model
    assert [card["title"] for card in model["stages"]] == [
        "NOISE SUPPRESSION",
        "NOISE GATE",
        "DE-ESSER",
        "COMPRESSOR",
        "LIMITER",
    ]
    for card in model["stages"]:
        assert card["toggle"] is not None
        assert card["rows"]
    assert len(model["eq"]["bands"]) == 10
    assert [card["title"] for card in model["settings"]["cards"]] == [
        "OPTIONS",
        "TRAY AND BACKGROUND",
        "HELP",
    ]


def test_native_quick_graph_paints_and_selects_shared_state(quick_window) -> None:
    eq = quick_window.quick_bridge.model["eq"]
    curve = quick_window.eq_presentation.graph_model
    item = EQGraphItem()
    item.setWidth(900)
    item.setHeight(300)
    item.graph = eq["curve"]

    image = QImage(900, 300, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    item.paint(painter)
    painter.end()
    assert image.pixelColor(450, 150).alpha() > 0

    x, y = EQGraphGeometry.band_handle_position(curve.bands, 3, 900, 300)
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(x, y),
        QPointF(x, y),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    item.mousePressEvent(press)
    assert eq["band"].index == 3


def _find(item, kind: str, proxy):
    for child in item.childItems():
        if kind in child.metaObject().className() and child.property("proxy") is proxy:
            return child
        found = _find(child, kind, proxy)
        if found is not None:
            return found
    return None


def test_scene_controls_drive_the_shared_state(quick_window, qapp) -> None:
    """Click and drag in the scene itself, not on the proxies."""

    from PySide6.QtTest import QTest

    quick_window.resize(1400, 900)
    qapp.processEvents()
    view = quick_window.centralWidget()
    model = quick_window.quick_bridge.model

    mute = _find(view.rootObject(), "Toggle", model["top"]["mute"])
    assert mute is not None
    assert not quick_window.user_mute_checkbox_action.isChecked()
    QTest.mouseClick(
        view,
        Qt.MouseButton.LeftButton,
        pos=mute.mapToScene(QPointF(12, mute.height() / 2)).toPoint(),
    )
    assert quick_window.user_mute_checkbox_action.isChecked()
    assert quick_window.user_muted

    def band_gain() -> float:
        return quick_window.eq_state.get_eq_settings().bands[0].gain_db

    fader = _find(view.rootObject(), "Fader", model["eq"]["bands"][0]["gain"])
    assert fader is not None
    start = fader.mapToScene(QPointF(fader.width() / 2, fader.height() / 2)).toPoint()
    end = fader.mapToScene(QPointF(fader.width() * 0.8, fader.height() / 2)).toPoint()
    assert band_gain() == 0
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view, end)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=end)
    assert band_gain() > 3


def _graph_item(item):
    for child in item.childItems():
        if isinstance(child, EQGraphItem):
            return child
        found = _graph_item(child)
        if found is not None:
            return found
    return None


@pytest.fixture
def mic_page(quick_window, qapp):
    """The EQ graph and the page holding it, in a window the page overflows."""
    quick_window.resize(900, 600)
    qapp.processEvents()
    graph = _graph_item(quick_window.centralWidget().rootObject())
    assert graph is not None
    page = graph
    while page.property("contentY") is None:
        page = page.parentItem()
    assert page.property("contentHeight") > page.height()
    return graph, page


def _handle_point(window, item, band: int) -> QPoint:
    x, y = EQGraphGeometry.band_handle_position(
        window.eq_presentation.graph_model.bands, band, item.width(), item.height()
    )
    return item.mapToScene(QPointF(x, y)).toPoint()


def _drag_distance() -> int:
    return QGuiApplication.styleHints().startDragDistance()


def _drag_up(view, start: QPoint, steps: int = 8, step: int = 8):
    """Press at ``start``, yield after each move up, release when exhausted."""
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=start)
    pos = start
    for _ in range(steps):
        pos = pos - QPoint(0, step)
        QTest.mouseMove(view, pos, delay=5)
        yield pos
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=pos)


def _wait_until(condition, timeout_ms: int = 3000) -> bool:
    """Run the event loop in short slices until ``condition()`` holds or time runs out."""
    deadline = time.monotonic() + timeout_ms / 1000
    while not condition() and time.monotonic() < deadline:
        QTest.qWait(5)
    return condition()


def _touch_drag_up(view, start: QPoint, steps: int, step: int):
    """Like :func:`_drag_up` with a finger.

    Qt Quick delivers a touch move with the next frame and merges moves that
    arrive within one frame, so each move waits for two frames to be rendered:
    the second begins after the move was queued.
    """
    frames = QSignalSpy(view.quickWindow().afterRendering)
    touch = QTest.touchEvent(view, QTest.createTouchDevice())
    touch.press(0, start, view)
    touch.commit()
    pos = start
    for _ in range(steps):
        pos = pos - QPoint(0, step)
        touch.move(0, pos, view)
        touch.commit()
        rendered = frames.count() + 2
        _wait_until(lambda: frames.count() >= rendered)
        yield pos
    touch.release(0, pos, view)
    touch.commit()


def test_a_band_drag_is_not_taken_over_by_the_page(quick_window, mic_page) -> None:
    item, page = mic_page
    graph = quick_window.eq_presentation.graph_model
    band = 4
    finished = QSignalSpy(graph.bandDragFinished)

    gains = []
    for pos in _drag_up(
        quick_window.centralWidget(),
        _handle_point(quick_window, item, band),
        step=_drag_distance() + 2,
    ):
        assert graph.dragging_band_index == band
        assert page.property("contentY") == 0
        gains.append(graph.bands[band][2])

    assert all(before < after for before, after in zip(gains, gains[1:]))
    assert gains[-1] == EQGraphGeometry.y_to_gain(
        item.mapFromScene(QPointF(pos)).y(), item.width(), item.height()
    )
    # An ordinary release finishes the drag once, and Qt's later ungrab adds nothing.
    assert graph.dragging_band_index is None
    assert finished.count() == 1
    assert finished.at(0)[0] == band and finished.at(0)[2] == gains[-1]
    assert not item.keepMouseGrab()
    committed = quick_window.eq_state.get_eq_settings().bands[band].gain_db
    assert committed == pytest.approx(gains[-1])


def test_a_mouse_drag_does_not_scroll_the_page_but_the_wheel_does(
    quick_window, mic_page
) -> None:
    item, page = mic_page
    view = quick_window.centralWidget()
    margin = QPointF(3, 200)
    for owner, spot in ((page, margin), (item, QPointF(10, 10))):
        for _ in _drag_up(view, owner.mapToScene(spot).toPoint()):
            assert page.property("contentY") == 0

    QTest.wheelEvent(view.quickWindow(), page.mapToScene(margin), QPoint(0, -120))
    assert _wait_until(lambda: page.property("contentY") > 0)


def test_a_lost_mouse_grab_finishes_the_band_drag(quick_window, mic_page) -> None:
    item, _page = mic_page
    view = quick_window.centralWidget()
    graph = quick_window.eq_presentation.graph_model
    band = 4
    start = _handle_point(quick_window, item, band)
    finished = QSignalSpy(graph.bandDragFinished)

    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view, start - QPoint(0, 24), delay=5)
    assert graph.dragging_band_index == band
    gain = graph.bands[band][2]
    assert gain > 0

    item.ungrabMouse()

    assert graph.dragging_band_index is None
    assert not item.keepMouseGrab()
    assert finished.count() == 1
    assert finished.at(0)[0] == band and finished.at(0)[2] == gain
    committed = quick_window.eq_state.get_eq_settings().bands[band].gain_db
    assert committed == pytest.approx(gain)
    # No stale origin is left for Escape to jump back to.
    QTest.keyClick(view, Qt.Key.Key_Escape)
    assert graph.bands[band][2] == gain
    # The button coming up afterwards finishes nothing twice.
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=start - QPoint(0, 24))
    assert finished.count() == 1 and graph.bands[band][2] == gain


def test_touch_still_flicks_the_page(quick_window, mic_page) -> None:
    _item, page = mic_page
    start = page.mapToScene(QPointF(200, page.height() - 30)).toPoint()
    for _ in _touch_drag_up(quick_window.centralWidget(), start, steps=12, step=10):
        pass
    assert page.property("contentY") > 0


def test_a_touch_band_drag_is_not_taken_over_by_the_page(
    quick_window, mic_page
) -> None:
    item, page = mic_page
    graph = quick_window.eq_presentation.graph_model
    band = 4
    start = _handle_point(quick_window, item, band)

    gains = []
    for _ in _touch_drag_up(
        quick_window.centralWidget(), start, steps=8, step=_drag_distance() + 2
    ):
        assert graph.dragging_band_index == band
        assert page.property("contentY") == 0
        gains.append(graph.bands[band][2])

    assert all(before < after for before, after in zip(gains, gains[1:]))
    assert graph.dragging_band_index is None
    committed = quick_window.eq_state.get_eq_settings().bands[band].gain_db
    assert committed == pytest.approx(gains[-1])


def test_a_click_does_not_scroll_its_control_away(quick_window, mic_page, qapp) -> None:
    item, page = mic_page
    view = quick_window.centralWidget()
    graph = quick_window.eq_presentation.graph_model
    band = 4
    quick_window.activateWindow()
    view.setFocus()
    page.setProperty("contentY", 150)
    qapp.processEvents()
    assert not item.hasActiveFocus()
    start = _handle_point(quick_window, item, band)
    assert page.contains(page.mapFromScene(QPointF(start)))
    gain = graph.bands[band][2]

    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=start)
    assert item.hasActiveFocus()
    assert page.property("contentY") == 150
    QTest.mouseMove(view, start + QPoint(1, 0), delay=5)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=start + QPoint(1, 0))

    assert page.property("contentY") == 150
    assert graph.bands[band][2] == gain


def _rebaseline(window) -> None:
    """Make the current state the saved preset and the start of the history."""
    window._set_preset_identity(window._get_current_preset())
    window._initialize_configuration_history()


@pytest.mark.parametrize("offset", [(0, 0), (5, 5)])
def test_a_click_on_a_knob_leaves_the_band_alone(
    quick_window, mic_page, qapp, offset
) -> None:
    item, _page = mic_page
    state = quick_window.eq_state
    graph = quick_window.eq_presentation.graph_model
    band = 4
    # Off the graph's whole-Hz, 0.1 dB grid, so snapping to it would show.
    state.set_band(
        band, replace(state.get_band(band), frequency_hz=437.5, gain_db=2.35), edited=False
    )
    state.flush()
    qapp.processEvents()
    _rebaseline(quick_window)
    before = state.get_band(band)
    assert graph.bands[band][1:3] == (437.5, 2.35)

    QTest.mouseClick(
        quick_window.centralWidget(),
        Qt.MouseButton.LeftButton,
        pos=_handle_point(quick_window, item, band) + QPoint(*offset),
    )
    qapp.processEvents()

    assert graph.bands[band][1:3] == (437.5, 2.35)
    assert state.get_band(band) == before
    assert not quick_window.quick_bridge.model["presets"]["status"].data["modified"]
    assert not quick_window._undo_action.isEnabled()


def test_a_click_on_an_octave_bandwidth_band_keeps_its_bandwidth(
    quick_window, mic_page, qapp
) -> None:
    item, _page = mic_page
    view = quick_window.centralWidget()
    state = quick_window.eq_state
    band = 4
    octave = replace(
        state.get_band(band),
        frequency_hz=437.5,
        gain_db=2.35,
        q=q_from_bandwidth_octaves(437.5, 1.0),
        bandwidth_mode="octaves",
        bandwidth_octaves=1.0,
    )
    state.set_band(band, octave, edited=False)
    state.flush()
    qapp.processEvents()
    _rebaseline(quick_window)

    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=_handle_point(quick_window, item, band))
    qapp.processEvents()

    assert state.get_band(band) == octave
    assert not quick_window.quick_bridge.model["presets"]["status"].data["modified"]
    assert not quick_window._undo_action.isEnabled()

    start = _handle_point(quick_window, item, band)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view, start + QPoint(30, 0), delay=5)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=start + QPoint(30, 0))

    dragged = state.get_band(band)
    assert dragged.frequency_hz != octave.frequency_hz
    assert (dragged.bandwidth_mode, dragged.bandwidth_octaves) == ("q", None)


def test_escape_puts_an_octave_bandwidth_band_back_whole(
    quick_window, mic_page, qapp
) -> None:
    item, _page = mic_page
    view = quick_window.centralWidget()
    state = quick_window.eq_state
    band = 4
    octave = replace(
        state.get_band(band),
        frequency_hz=1280.0,
        gain_db=0.0,
        q=q_from_bandwidth_octaves(1280.0, 1.0),
        bandwidth_mode="octaves",
        bandwidth_octaves=1.0,
    )
    state.set_band(band, octave, edited=False)
    state.flush()
    qapp.processEvents()
    _rebaseline(quick_window)

    start = _handle_point(quick_window, item, band)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view, start + QPoint(30, -20), delay=5)
    assert state.get_band(band).bandwidth_mode == "q"
    QTest.keyClick(view, Qt.Key.Key_Escape)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=start + QPoint(30, -20))
    qapp.processEvents()

    assert state.get_band(band) == octave
    assert not quick_window.quick_bridge.model["presets"]["status"].data["modified"]
    assert not quick_window._undo_action.isEnabled()


def test_a_drag_moves_the_band_by_the_pointers_travel(quick_window, mic_page) -> None:
    item, page = mic_page
    view = quick_window.centralWidget()
    graph = quick_window.eq_presentation.graph_model
    band = 4
    width, height = item.width(), item.height()
    x, y = EQGraphGeometry.band_handle_position(graph.bands, band, width, height)
    # Pressed 10 px from the handle's centre, still inside its hit radius.
    start = _handle_point(quick_window, item, band) + QPoint(7, 7)

    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=start)
    # The first move already passes the drag distance, and the band takes all of it.
    for dx, dy in ((_drag_distance() + 2, 0), (30, -16)):
        QTest.mouseMove(view, start + QPoint(dx, dy), delay=5)
        assert page.property("contentY") == 0
        assert graph.bands[band][1] == pytest.approx(
            EQGraphGeometry.x_to_frequency(x + dx, width, height), abs=1.0
        )
        assert graph.bands[band][2] == pytest.approx(
            EQGraphGeometry.y_to_gain(y + dy, width, height), abs=0.11
        )
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=start + QPoint(30, -16))

    committed = quick_window.eq_state.get_band(band)
    assert committed.frequency_hz == pytest.approx(graph.bands[band][1])
    assert committed.gain_db == pytest.approx(graph.bands[band][2])


@pytest.fixture
def wide_page(quick_window, mic_page, qapp):
    """The mic page in a window wide enough for the whole EQ card to show."""
    page = mic_page[1]
    quick_window.resize(1300, 700)
    qapp.processEvents()
    assert page.property("contentHeight") > page.height()
    return page


def _reveal(page, item) -> QPoint:
    """Scroll ``item`` to the middle of the page and return its centre in the scene."""
    top = item.mapToItem(page.property("contentItem"), QPointF(0, 0)).y()
    limit = page.property("contentHeight") - page.height()
    page.setProperty("contentY", min(max(0, top - page.height() / 2), limit))
    centre = item.mapToScene(QPointF(item.width() / 2, item.height() / 2))
    assert page.contains(page.mapFromScene(centre))
    return centre.toPoint()


def _band_box(window, name: str):
    """The selected band's control ``name``, and its value box in the scene."""
    eq = window.quick_bridge.model["eq"]
    proxy = eq["bands"][eq["band"].index][name]
    return proxy, _find(window.centralWidget().rootObject(), "ValueBox", proxy)


def _escape_event() -> QKeyEvent:
    return QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)


def test_escape_puts_the_shown_value_back_in_a_value_box(quick_window, wide_page) -> None:
    page = wide_page
    view = quick_window.centralWidget()
    index = quick_window.quick_bridge.model["eq"]["band"].index
    frequency, box = _band_box(quick_window, "frequency")
    gain, readout = _band_box(quick_window, "gain")
    shown = frequency.text
    stored = quick_window.eq_state.get_band(index).frequency_hz

    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=_reveal(page, box))
    field = view.quickWindow().activeFocusItem()
    QTest.keyClick(view, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(view, "777")
    assert field.property("text") == "777"
    QTest.keyClick(view, Qt.Key.Key_Escape)
    assert field.property("text") == shown

    # Focus leaving the box applies nothing.
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=_reveal(page, readout))
    assert frequency.text == shown
    assert quick_window.eq_state.get_band(index).frequency_hz == stored


def test_escape_is_swallowed_only_when_a_value_box_has_an_edit_to_cancel(
    quick_window, wide_page
) -> None:
    page = wide_page
    view = quick_window.centralWidget()
    _frequency, box = _band_box(quick_window, "frequency")

    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=_reveal(page, box))
    untouched = _escape_event()
    QCoreApplication.sendEvent(view, untouched)
    assert not untouched.isAccepted()

    QTest.keyClicks(view, "7")
    edited = _escape_event()
    QCoreApplication.sendEvent(view, edited)
    assert edited.isAccepted()


def test_a_clicked_readout_lets_the_wheel_scroll_the_page(quick_window, wide_page) -> None:
    page = wide_page
    view = quick_window.centralWidget()
    _gain, readout = _band_box(quick_window, "gain")

    centre = _reveal(page, readout)
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=centre)
    field = view.quickWindow().activeFocusItem()
    assert field.property("readOnly") and field.hasActiveFocus()
    before = page.property("contentY")
    QTest.wheelEvent(view.quickWindow(), QPointF(centre), QPoint(0, -120))

    assert _wait_until(lambda: page.property("contentY") > before)


def test_a_focused_editable_box_still_steps_with_the_wheel(quick_window, wide_page) -> None:
    page = wide_page
    view = quick_window.centralWidget()
    frequency, box = _band_box(quick_window, "frequency")

    centre = _reveal(page, box)
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=centre)
    before, scrolled = frequency.value, page.property("contentY")
    QTest.wheelEvent(view.quickWindow(), QPointF(centre), QPoint(0, 120))

    assert _wait_until(lambda: frequency.value > before)
    assert page.property("contentY") == scrolled


def test_the_export_names_the_quick_view_and_the_graphics_backend_qt_chose(quick_window) -> None:
    snapshot = quick_window._interface_snapshot()
    assert snapshot["view"] == "quick" and snapshot["classic_reason"] == "none"
    interface = quick_window.centralWidget().quickWindow().rendererInterface()
    expected = {
        QSGRendererInterface.GraphicsApi.Software: "software",
        QSGRendererInterface.GraphicsApi.OpenGL: "opengl",
        QSGRendererInterface.GraphicsApi.Direct3D11: "direct3d11",
        QSGRendererInterface.GraphicsApi.Vulkan: "vulkan",
        QSGRendererInterface.GraphicsApi.Null: "null",
    }
    assert snapshot["graphics_api"] == (
        "unknown" if interface is None else expected.get(interface.graphicsApi(), "unknown")
    )


def _load_config_without_devices(monkeypatch) -> None:
    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
    monkeypatch.setattr("mic_eq.ui.main_window.list_presets", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_input_devices", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_output_devices", lambda: [])


def test_level_history_follows_the_meters_and_clears_when_they_stop(quick_window) -> None:
    bridge = quick_window.quick_bridge
    quick_window.input_meter.set_levels(-30.0, -20.0)
    quick_window.output_meter.set_levels(-24.0, -12.0)
    bridge._sample_levels()
    bridge._sample_levels()
    assert bridge.levels == [[-30.0, -24.0], [-30.0, -24.0]]
    quick_window.input_meter.set_unavailable()
    quick_window.output_meter.set_unavailable()
    bridge._sample_levels()
    assert bridge.levels == []


def test_settings_rows_tell_switches_menus_and_actions_apart(quick_window) -> None:
    cards = quick_window.quick_bridge.model["settings"]["cards"]
    kinds = {row["proxy"].text: row["kind"] for card in cards for row in card["rows"]}
    assert kinds["Startup Preset"] == "menu"
    assert kinds["Run Guided Setup"] == "action"
    assert kinds["Keep running in tray when window is closed"] == "toggle"


def test_scene_gets_structured_values_instead_of_parsing_labels(quick_window) -> None:
    model = quick_window.quick_bridge.model
    status = model["presets"]["status"]
    assert status.data == {
        "name": quick_window.current_preset_name,
        "modified": quick_window.preset_modified,
    }
    assert model["eq"]["diagnostics"].data == {"calibrated": False}
    noise, gate = model["stages"][:2]
    assert noise["trace"] and noise["alert"] is not None
    assert not gate["trace"] and gate["alert"] is None


def _settings_row(window, text: str):
    cards = window.quick_bridge.model["settings"]["cards"]
    return next(
        row["proxy"] for card in cards for row in card["rows"] if row["proxy"].text.startswith(text)
    )


def test_settings_switch_follows_an_action_the_window_corrects_silently(
    quick_window, monkeypatch
) -> None:
    """A refused global shortcut unchecks its action with signals blocked."""

    monkeypatch.setattr(
        "mic_eq.ui.main_window.GlobalMuteHotkey.register", lambda _self: (False, "in use")
    )
    action = quick_window._mute_hotkey_action
    row = _settings_row(quick_window, "Enable global mute shortcut")
    with QSignalBlocker(action):
        action.setChecked(True)
    action.changed.emit()
    assert row.checked

    assert not quick_window._register_mute_hotkey("Ctrl+Alt+M")
    assert not action.isChecked()
    assert not row.checked


def test_scene_control_reads_back_a_mode_the_processor_refused(quick_window) -> None:
    mode = quick_window.quick_bridge.model["top"]["mode"]
    choice = quick_window.processing_mode_choice
    native = quick_window.processor

    class RejectBypass:
        def __getattr__(self, name):
            return getattr(native, name)

        def set_bypass(self, enabled):
            if enabled:
                raise RuntimeError("bypass unavailable")
            native.set_bypass(enabled)

    quick_window.processor = RejectBypass()
    notified: list[bool] = []
    mode.changed.connect(lambda: notified.append(True))
    mode.setIndex(1)
    assert choice.currentIndex() == 0 and mode.index == 0
    assert notified


def test_redo_button_follows_the_history(quick_window) -> None:
    redo = quick_window.quick_bridge.model["presets"]["redo"]
    assert not quick_window.redo_button_action.isEnabled()
    quick_window.gate_state.set_value("enabled", not quick_window.gate_state.get_settings()["enabled"])
    quick_window.undo_configuration()
    assert quick_window.redo_button_action.isEnabled() == quick_window._redo_action.isEnabled()
    assert quick_window.redo_button_action.isEnabled()
    assert redo.enabled


def test_keyboard_focus_scrolls_its_control_into_view(quick_window, qapp) -> None:
    quick_window.resize(900, 600)
    quick_window.activateWindow()
    view = quick_window.centralWidget()
    view.setFocus()
    qapp.processEvents()
    limiter = quick_window.quick_bridge.model["stages"][-1]
    toggle = _find(view.rootObject(), "Toggle", limiter["toggle"])
    assert toggle is not None
    page = toggle
    while page is not None and page.property("contentY") is None:
        page = page.parentItem()
    assert page is not None and page.property("contentY") == 0
    toggle.forceActiveFocus()
    qapp.processEvents()
    assert page.property("contentY") > 0


def _queue_limiter_ceiling(window, *, reject: bool = False) -> list[float]:
    state = window.limiter_state
    native = state.processor
    state.set_settings({"ceiling_db": -1.0})
    written: list[float] = []

    class CeilingProbe:
        def __getattr__(self, name):
            return getattr(native, name)

        def set_limiter_ceiling(self, value):
            if reject and value == -3.0:
                raise RuntimeError("queued ceiling rejected")
            native.set_limiter_ceiling(value)
            written.append(value)

    state.processor = CeilingProbe()
    state._rate_limiter.interval_ms = 60000
    state.set_value("ceiling_db", -3.0)
    assert state.get_settings()["ceiling_db"] == -3.0
    assert state._rate_limiter._pending_fn is not None
    assert written == []
    return written


def test_failed_bulk_configuration_restores_applied_not_queued_settings(quick_window) -> None:
    window = quick_window
    candidate = window._get_current_preset()
    candidate.deesser.threshold_db = -22.0
    written = _queue_limiter_ceiling(window)
    native = window.deesser_state.processor

    class RejectCandidate:
        def __getattr__(self, name):
            return getattr(native, name)

        def apply_deesser_settings(self, settings):
            if settings["threshold_db"] == -22.0:
                raise RuntimeError("bulk de-esser candidate rejected")
            native.apply_deesser_settings(settings)

    window.deesser_state.processor = RejectCandidate()
    with pytest.raises(RuntimeError, match="previous sound restored"):
        window.apply_processing_configuration(candidate)
    assert window.limiter_state.get_settings()["ceiling_db"] == -1.0
    assert window.limiter_state._rate_limiter._pending_fn is None
    assert written == [-1.0]


@pytest.mark.parametrize("entrypoint", ["owned", "save_as", "prompt"])
@pytest.mark.parametrize("reject", [True, False], ids=["rejected", "accepted"])
def test_save_entrypoints_capture_only_accepted_pending_settings(
    quick_window, monkeypatch, tmp_path, entrypoint, reject,
) -> None:
    window = quick_window
    written = _queue_limiter_ceiling(window, reject=reject)
    window.current_preset_path = tmp_path / "owned.json"
    monkeypatch.setattr("mic_eq.ui.main_window.get_presets_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "mic_eq.ui.main_window.QInputDialog.getText", lambda *_args, **_kwargs: ("Saved sound", True),
    )
    monkeypatch.setattr(
        "mic_eq.ui.main_window.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    saved: list[float] = []

    def capture_save(preset, *, filepath=None):
        assert window.limiter_state._rate_limiter._pending_fn is None
        assert written == [-3.0]
        saved.append(preset.limiter.ceiling_db)
        return filepath or tmp_path / "saved.json"

    monkeypatch.setattr(window, "_save_preset_file", capture_save)
    if entrypoint == "owned":
        result = window._save_preset()
    elif entrypoint == "save_as":
        result = window._save_preset_as()
    else:
        result = window._prompt_save_current_preset(
            title="Save", question="Save this sound?", preset_name="Saved sound", description="",
        )
    assert result is (None if entrypoint == "prompt" else not reject)
    assert saved == ([] if reject else [-3.0])
    assert written == ([-1.0] if reject else [-3.0])
    assert window.limiter_state.get_settings()["ceiling_db"] == (-1.0 if reject else -3.0)


def _picker_text(picker) -> str:
    return picker.property("contentItem").property("text")


def test_health_rows_show_their_plain_text_without_opening_technical_details(quick_window) -> None:
    rows = quick_window.quick_bridge.model["health"]["stream"]
    assert all("coded" not in row for row in rows)
    qml = (Path(__file__).parents[1] / "mic_eq" / "ui" / "qml" / "Main.qml").read_text(encoding="utf-8")
    assert "coded" not in qml and "technical" not in qml.split("component HealthRow")[1].split("component Page")[0]


def test_choice_state_exposes_its_placeholder_to_the_scene(quick_window) -> None:
    choice = quick_window.output_choice
    assert choice.property("placeholder") == ""
    choice.setPlaceholderText("Select a replacement destination")
    assert choice.property("placeholder") == "Select a replacement destination"


def test_quick_picker_shows_the_placeholder_while_nothing_is_selected(quick_window, qapp) -> None:
    choice = quick_window.output_choice
    choice.clear()
    choice.setPlaceholderText("Select a replacement destination")
    qapp.processEvents()
    picker = _find(quick_window.centralWidget().rootObject(), "Picker", choice)
    assert picker is not None
    assert _picker_text(picker) == "Select a replacement destination"
    choice.addItem("Cable Input", DeviceIdentity(name="Cable Input", direction="output"))
    qapp.processEvents()
    assert _picker_text(picker) == "Cable Input"


def test_a_device_name_cut_short_at_the_minimum_window_size_is_whole_in_the_tooltip(
    quick_window, qapp
) -> None:
    from pathlib import Path

    from PySide6.QtGui import QFontDatabase

    # Qt's offscreen platform has no fonts, which makes every name look short.
    font = Path("C:/Windows/Fonts/segoeui.ttf")
    if not font.is_file():
        pytest.skip("The Windows UI font is not installed")
    families = QFontDatabase.applicationFontFamilies(QFontDatabase.addApplicationFont(str(font)))
    previous = qapp.font()
    qapp.setFont(QFont(families[0], 9))
    try:
        quick_window.resize(900, 640)
        names = {
            quick_window.input_choice: "Studio Microphone (USB)",
            quick_window.output_choice: "CABLE Input (VB-Audio Virtual Cable)",
        }
        for choice, name in names.items():
            choice.addItem(name, DeviceIdentity(name=name, direction="input"))
        for _ in range(3):
            qapp.processEvents()
        root = quick_window.centralWidget().rootObject()
        for choice, name in names.items():
            picker = _find(root, "Picker", choice)
            assert picker is not None
            assert picker.property("contentItem").property("truncated")
            hint = next(c for c in picker.children() if "Hint" in c.metaObject().className())
            assert hint.property("text") == name
    finally:
        qapp.setFont(previous)


def test_quick_meters_describe_a_lit_clip_marker_to_assistive_technology(quick_window, qapp) -> None:
    from PySide6.QtGui import QAccessible

    meter = quick_window.input_meter
    root = quick_window.centralWidget().rootObject()

    def find_meter(item):
        for child in item.childItems():
            if isinstance(child, QuickMeterItem) and child.property("source") is meter:
                return child
            found = find_meter(child)
            if found is not None:
                return found
        return None

    item = find_meter(root)
    assert item is not None
    was_active = QAccessible.isActive()
    QAccessible.setActive(True)

    def description() -> str:
        qapp.processEvents()
        return QAccessible.queryAccessibleInterface(item).text(QAccessible.Text.Description)

    try:
        assert description() == ""
        meter.set_levels(-20.0, -10.0)
        meter.note_peak(None, None, full_scale_count=0)
        meter.note_peak(1.0, 1.0, full_scale_count=CLIP_FLAT_TOP_SAMPLES)
        assert "Clipping" in description()
    finally:
        QAccessible.setActive(was_active)


class _AccessibleScene:
    """Qt's accessibility layer on, for reading the names assistive technology gets."""

    def __enter__(self):
        from PySide6.QtGui import QAccessible

        self._accessible, self._was_active = QAccessible, QAccessible.isActive()
        QAccessible.setActive(True)
        return self

    def __exit__(self, *_exc) -> None:
        self._accessible.setActive(self._was_active)

    def name(self, item) -> str:
        return self._accessible.queryAccessibleInterface(item).text(self._accessible.Text.Name)

    def description(self, item) -> str:
        return self._accessible.queryAccessibleInterface(item).text(self._accessible.Text.Description)


def _tab_stops(window, qapp, presses: int = 120) -> list:
    """The items that real Tab presses focus in turn, wrapping around."""
    window.resize(1400, 900)
    window.activateWindow()
    view = window.centralWidget()
    view.setFocus()
    qapp.processEvents()
    stops = []
    for _ in range(presses):
        QTest.keyClick(view, Qt.Key.Key_Tab)
        stops.append(view.quickWindow().activeFocusItem())
    return stops


def test_the_health_chip_is_a_toggle_the_keyboard_reaches_and_names(quick_window, qapp) -> None:
    from PySide6.QtGui import QAccessible

    window = quick_window
    view = window.centralWidget()
    with _AccessibleScene() as scene:
        stops = _tab_stops(window, qapp)
        named = [item for item in stops if item.objectName() == "healthToggle"]
        assert named, "Tab never reaches the Health chip"
        toggle = named[0]
        # Voice control says what it sees: the visible text names the button.
        chip = window.shell_texts["health_summary_label"]
        assert chip.text and scene.name(toggle) == chip.text
        assert scene.description(toggle) == "Show audio health details"
        window._set_health_chip("health_summary_label", "Health: Check input", "warn")
        qapp.processEvents()
        assert scene.name(toggle) == "Health: Check input"
        interface = QAccessible.queryAccessibleInterface(toggle)
        assert interface.role() == QAccessible.Role.Button
        assert not interface.state().checked

        assert window.shell_page.currentIndex() == 0
        # Focus the chip through Tab alone, then use the keys a button answers to.
        while view.quickWindow().activeFocusItem().objectName() != toggle.objectName():
            QTest.keyClick(view, Qt.Key.Key_Tab)
        QTest.keyClick(view, Qt.Key.Key_Space)
        assert window.shell_page.currentIndex() == window.HEALTH_PAGE_INDEX
        assert QAccessible.queryAccessibleInterface(toggle).state().checked
        QTest.keyClick(view, Qt.Key.Key_Return)
        assert window.shell_page.currentIndex() == 0

        centre = toggle.mapToScene(QPointF(toggle.width() / 2, toggle.height() / 2)).toPoint()
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=centre)
        assert window.shell_page.currentIndex() == window.HEALTH_PAGE_INDEX
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=centre)
        assert window.shell_page.currentIndex() == 0


def test_the_equalizer_help_says_what_delete_does_and_delete_does_it(quick_window, qapp) -> None:
    from mic_eq.config import EQSettings

    window = quick_window
    assert "Delete resets the selected band." in window.quick_bridge.model["eq"]["help"]

    window.resize(1400, 900)
    qapp.processEvents()
    state = window.eq_state
    state.set_band_value(4, "gain_db", 3.0, edited=False)
    state.flush()
    window.eq_presentation.graph_model.select_band(4)
    graph = _graph_item(window.centralWidget().rootObject())
    assert graph is not None
    graph.forceActiveFocus()
    QTest.keyClick(window.centralWidget(), Qt.Key.Key_Delete)
    assert state.get_band(4) == EQSettings().bands[4]


def test_the_preset_menu_button_name_contains_its_visible_label(quick_window, qapp) -> None:
    # Voice control finds a control by what it reads, so the name must contain it.
    menu = quick_window.quick_bridge.model["presets"]["menu"]
    button = _find(quick_window.centralWidget().rootObject(), "Btn", menu)
    assert button is not None
    with _AccessibleScene() as scene:
        qapp.processEvents()
        assert button.property("text") in scene.name(button)


def test_a_read_only_value_box_is_no_tab_stop_copies_and_shows_no_edit_border(
    quick_window, wide_page, qapp
) -> None:
    window, page = quick_window, wide_page
    view = window.centralWidget()
    gain, readout = _band_box(window, "gain")
    assert readout is not None

    stops = _tab_stops(window, qapp)
    assert not [item for item in stops if item.property("readOnly")]
    assert any(item.metaObject().className().startswith("QQuickTextInput") for item in stops)

    field = next(
        child for child in readout.childItems() if child.metaObject().className().startswith("QQuickTextInput")
    )
    assert field.property("readOnly") and not field.property("activeFocusOnTab")
    with _AccessibleScene() as scene:
        assert scene.name(field) == gain.name

    def border_colour(box):
        return QQmlProperty.read(box, "border.color").name()

    accent = window.quick_bridge.model["theme"]["accent"].lower()
    QGuiApplication.clipboard().clear()
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=_reveal(page, readout))
    QTest.keyClick(view, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClick(view, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    assert QGuiApplication.clipboard().text() == field.property("text") != ""
    assert border_colour(readout).lower() != accent

    # An editable box still shows it has focus.
    _frequency, editable = _band_box(window, "frequency")
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=_reveal(page, editable))
    assert border_colour(editable).lower() == accent


def test_the_drops_tooltip_does_not_promise_a_right_click_quick_lacks(quick_window) -> None:
    drops = quick_window.shell_texts["dropped_label"]
    assert "right-click" not in drops.toolTip.lower()
    quick_window._update_diagnostic_labels(
        diagnostics={}, latency_ms=10.0, dsp_time_ms=0.2, input_buf=0, output_buf=0, rnnoise_buf=0,
    )
    assert "Technical counters" in drops.toolTip
    assert "right-click" not in drops.toolTip.lower()


def test_the_health_text_is_exposed_once_as_the_buttons_name(quick_window, qapp) -> None:
    from PySide6.QtGui import QAccessible

    def texts(interface) -> list[str]:
        found = [interface.text(QAccessible.Text.Name), interface.text(QAccessible.Text.Value)]
        for index in range(interface.childCount()):
            child = interface.child(index)
            found += texts(child) if child is not None else []
        return found

    with _AccessibleScene():
        view = quick_window.centralWidget()
        qapp.processEvents()
        health = quick_window.shell_texts["health_summary_label"].text
        tree = texts(QAccessible.queryAccessibleInterface(view))
        assert sum(health in text for text in tree) == 1


def test_a_held_return_toggles_the_health_page_once(quick_window, qapp) -> None:
    window = quick_window
    view = window.centralWidget()
    window.activateWindow()
    view.setFocus()
    toggle = view.rootObject().findChild(QObject, "healthToggle")
    toggle.forceActiveFocus(Qt.FocusReason.TabFocusReason)
    qapp.processEvents()

    def press(repeat: bool) -> None:
        QCoreApplication.sendEvent(
            view,
            QKeyEvent(
                QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier,
                "\r", repeat, 1,
            ),
        )

    press(False)
    assert window.shell_page.currentIndex() == window.HEALTH_PAGE_INDEX
    for _ in range(3):
        press(True)
    assert window.shell_page.currentIndex() == window.HEALTH_PAGE_INDEX


class _Engine:
    """The audio engine, faked: Start and Stop flip a flag and open no device."""

    mute_error = False

    def __init__(self, native) -> None:
        self.native, self.running = native, False

    def __getattr__(self, name):
        return getattr(self.native, name)

    def is_running(self) -> bool:
        return self.running

    def stop(self) -> None:
        self.running = False

    def set_output_mute(self, muted: bool) -> None:
        if self.mute_error:
            raise RuntimeError("unavailable")
        self.native.set_output_mute(muted)


@pytest.fixture
def engine_window(quick_window, monkeypatch):
    """The window with a faked engine and a route, driven by hand instead of by its timers."""
    window = quick_window
    window.meter_timer.stop()
    window.diagnostics_timer.stop()
    engine: Any = _Engine(window.processor)
    window.processor = engine
    window.input_choice.addItem("Microphone", DeviceIdentity(name="Microphone", direction="input"))
    window.output_choice.addItem("Destination", DeviceIdentity(name="Destination", direction="output"))
    monkeypatch.setattr(window, "_apply_input_preferences_for_current_route", lambda: None)

    def start(processor: _Engine, *_devices: object) -> str:
        processor.running = True
        return "started"

    monkeypatch.setattr("mic_eq.ui.main_window.start_processor_for_route", start)
    return window, engine


def _status_dot(window):
    strip = window.centralWidget().rootObject().findChild(QObject, "statusStrip")
    return next(child for child in strip.childItems() if child.width() == 8 and child.height() == 8)


def _diagnostic_labels(window, **changes) -> None:
    window._update_diagnostic_labels(**{
        "diagnostics": {}, "latency_ms": 10.0, "dsp_time_ms": 0.2, "input_buf": 0, "output_buf": 0,
        "rnnoise_buf": 0, "input_rms_db": -30.0, "output_rms_db": -30.0,
        "input_callback_age_ms": 0, "output_callback_age_ms": 0, **changes,
    })


def test_the_status_dot_shows_sound_running_and_heard(engine_window, qapp) -> None:
    window, engine = engine_window
    dot = _status_dot(window)

    def colour() -> str:
        qapp.processEvents()
        return dot.property("color").name()

    def palette(name: str) -> str:
        return QColor(getattr(PALETTE, name)).name()

    assert colour() == palette("border_strong")
    # Stop is offered while a login start waits for its devices, but nothing is running.
    window._login_startup_deadline = time.monotonic() + 60
    window._sync_processing_controls()
    assert window.stop_btn_action.isEnabled()
    assert window.shell_texts["transmission_status_label"].text.startswith("Transmission: Stopped")
    assert colour() == palette("border_strong")
    window._login_startup_deadline = None
    window._sync_processing_controls()

    window.start_btn_action.trigger()
    assert colour() == palette("status_ok_text")
    window.user_mute_checkbox_action.setChecked(True)
    assert colour() == palette("status_warn_text")
    window.user_mute_checkbox_action.setChecked(False)
    assert colour() == palette("status_ok_text")
    engine.mute_error = True
    window.user_mute_checkbox_action.setChecked(True)
    assert window.shell_texts["transmission_status_label"].text.endswith("Mute pending")
    assert colour() == palette("status_warn_text")

    window.stop_btn_action.trigger()
    assert colour() == palette("border_strong")


def test_the_device_grace_starts_over_after_stop_and_start(engine_window, monkeypatch) -> None:
    window, _engine = engine_window
    clock = [1000.0]
    monkeypatch.setattr("mic_eq.ui.main_window.time.monotonic", lambda: clock[0])
    never = 2**64 - 1  # what the engine reports before a device's first callback
    row = window.shell_texts["callback_health_label"]

    def tick() -> None:
        _diagnostic_labels(window, input_callback_age_ms=never, output_callback_age_ms=never)

    window.start_btn_action.trigger()
    tick()
    clock[0] += main_window.DEVICE_STARTUP_GRACE_SECONDS + 1
    tick()
    assert row.state == "warn"

    window.stop_btn_action.trigger()
    clock[0] += 1
    window.start_btn_action.trigger()
    tick()

    assert row.state != "warn"


def _meter_items(item):
    for child in item.childItems():
        if isinstance(child, QuickMeterItem):
            yield child
        yield from _meter_items(child)


def test_a_muted_output_says_its_levels_are_from_before_the_mute(engine_window, qapp) -> None:
    from PySide6.QtGui import QAccessible

    window, engine = engine_window
    window.start_btn_action.trigger()
    row = window.shell_texts["output_health_label"]
    summary = window.shell_texts["health_summary_label"]
    # With no limiter, full scale is a fault, unless nobody hears it.
    loud = {
        "diagnostics": {"output_true_peak_db": -30.4}, "output_rms_db": -36.0,
        "output_at_full_scale": True, "limiter_enabled": False,
    }
    root = window.centralWidget().rootObject()
    item = next(child for child in _meter_items(root) if child.property("source") is window.output_meter)

    def description() -> str:
        qapp.processEvents()
        return QAccessible.queryAccessibleInterface(item).text(QAccessible.Text.Description)

    was_active = QAccessible.isActive()
    QAccessible.setActive(True)
    try:
        _diagnostic_labels(window, **loud)
        assert row.state == "bad" and summary.state == "bad"
        assert description() == ""

        window.user_mute_checkbox_action.setChecked(True)
        _diagnostic_labels(window, **loud)
        assert (row.text, row.state) == (
            "Output: Muted (before mute: -36 dB average, peaks -30.4 dB)", "info",
        )
        assert summary.state == "ok"
        assert "before the mute" in description()

        window.user_mute_checkbox_action.setChecked(False)
        assert description() == ""

        # The row, the meter and the dot follow one state: a mute change the engine
        # refused is pending, not muted, whatever the engine still reports.
        window.user_mute_checkbox_action.setChecked(True)
        engine.mute_error = True
        window.user_mute_checkbox_action.setChecked(False)
        _diagnostic_labels(window, **{**loud, "diagnostics": {**loud["diagnostics"], "output_muted": True}})
        assert window.shell_texts["transmission_status_label"].text.endswith("Mute pending")
        assert row.state == "bad"
        assert description() == ""
    finally:
        QAccessible.setActive(was_active)
