"""The Qt Quick view must drive the same widgets the widget view does."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt, qInstallMessageHandler
from PySide6.QtGui import QImage, QMouseEvent, QPainter
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QLabel

from mic_eq.config import AppConfig
from mic_eq.ui.main_window import MainWindow
from mic_eq.ui.quick_shell import (
    QUICK_VIEW_UNAVAILABLE,
    WidgetItem,
    restore_widget_shell,
)


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


def test_scene_loads_and_replaces_the_widget_shell(quick_window) -> None:
    view = quick_window.centralWidget()
    assert isinstance(view, QQuickWidget)
    assert view.status() == QQuickWidget.Status.Ready
    assert view.errors() == []
    # Binding errors are only warnings to Qt; here they fail the test.
    assert vars(quick_window)["scene_messages"] == []
    # The widget shell keeps running off screen so its timers and layouts work.
    assert quick_window.widget_shell.isVisible()
    assert quick_window.widget_shell.testAttribute(
        Qt.WidgetAttribute.WA_DontShowOnScreen
    )


def test_field_proxy_writes_through_to_the_panel(quick_window) -> None:
    gate = _rows(quick_window.quick_bridge.model["stages"][1])
    threshold = gate["Threshold"]["proxy"]
    spinbox = quick_window.gate_panel.threshold_spinbox

    threshold.setValue(-33.0)
    assert spinbox.value() == pytest.approx(-33.0)
    assert threshold.value == pytest.approx(-33.0)
    assert threshold.text == spinbox.text()

    threshold.setText("-28 dB")
    assert spinbox.value() == pytest.approx(-28.0)
    threshold.setText("not a number")
    assert spinbox.value() == pytest.approx(-28.0)

    threshold.setValue(1000.0)
    assert spinbox.value() == spinbox.maximum()


def test_toggle_combo_and_page_proxies(quick_window) -> None:
    model = quick_window.quick_bridge.model
    gate_card = model["stages"][1]
    switch = quick_window.gate_panel.enabled_checkbox
    before = switch.isChecked()
    gate_card["toggle"].click()
    assert switch.isChecked() is not before
    assert gate_card["toggle"].checked is switch.isChecked()

    mode = model["top"]["mode"]
    combo = quick_window.processing_mode_combo
    assert mode.items == [combo.itemText(i) for i in range(combo.count())]
    mode.setIndex(1)
    assert combo.currentIndex() == 1

    model["page"].setIndex(MainWindow.HEALTH_PAGE_INDEX)
    assert quick_window.page_stack.currentIndex() == MainWindow.HEALTH_PAGE_INDEX


def test_proxy_follows_widget_changes_made_elsewhere(quick_window) -> None:
    bridge = quick_window.quick_bridge
    status = bridge.model["health"]["signal"][0]["proxy"]
    seen: list[str] = []
    status.changed.connect(lambda: seen.append(status.text))
    quick_window._set_health_chip(quick_window.input_health_label, "Input: hot", "warn")
    bridge.refresh()
    assert seen == ["Input: hot"]
    assert status.state == "warn"


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


def test_widget_item_paints_and_forwards_pointer_input(quick_window) -> None:
    eq = quick_window.quick_bridge.model["eq"]
    curve = quick_window.eq_panel.curve_widget
    item = WidgetItem()
    item.setWidth(900)
    item.setHeight(300)
    item.proxy = eq["curve"]

    image = QImage(900, 300, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    item.paint(painter)
    painter.end()
    assert curve.width() == 900
    assert image.pixelColor(450, 150).alpha() > 0

    x, y = curve.band_handle_position(3)
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(x, y),
        QPointF(x, y),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    item._send_mouse(press, press.type(), press.button(), press.buttons())
    quick_window.quick_bridge.refresh()
    assert eq["band"].index == 3


def _find(item, kind: str, proxy):
    for child in item.childItems():
        if kind in child.metaObject().className() and child.property("proxy") is proxy:
            return child
        found = _find(child, kind, proxy)
        if found is not None:
            return found
    return None


def test_scene_controls_drive_the_widgets(quick_window, qapp) -> None:
    """Click and drag in the scene itself, not on the proxies."""

    from PySide6.QtTest import QTest

    quick_window.resize(1400, 900)
    qapp.processEvents()
    view = quick_window.centralWidget()
    model = quick_window.quick_bridge.model

    mute = _find(view.rootObject(), "Toggle", model["top"]["mute"])
    assert mute is not None
    assert not quick_window.user_mute_checkbox.isChecked()
    QTest.mouseClick(
        view,
        Qt.MouseButton.LeftButton,
        pos=mute.mapToScene(QPointF(12, mute.height() / 2)).toPoint(),
    )
    assert quick_window.user_mute_checkbox.isChecked()

    slider = quick_window.eq_panel.band_sliders[0].slider
    fader = _find(view.rootObject(), "Fader", model["eq"]["bands"][0]["gain"])
    assert fader is not None
    start = fader.mapToScene(QPointF(fader.width() / 2, fader.height() / 2)).toPoint()
    end = fader.mapToScene(QPointF(fader.width() * 0.8, fader.height() / 2)).toPoint()
    assert slider.value() == 0
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view, end)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=end)
    assert slider.value() > 30


def _fallback_notice(window) -> bool:
    return any(
        label.text() == QUICK_VIEW_UNAVAILABLE
        for label in window.status_bar.findChildren(QLabel)
    )


def test_render_failure_falls_back_to_the_widget_view_and_says_so(quick_window) -> None:
    shell = quick_window.widget_shell
    restore_widget_shell(quick_window, "no graphics adapter")
    assert quick_window.centralWidget() is shell
    assert not shell.testAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    assert quick_window.status_bar.isVisible()
    assert _fallback_notice(quick_window)
    # The repaint hook is gone with the scene, so the widgets draw as before.
    assert "update" not in quick_window.eq_panel.curve_widget.__dict__
    quick_window.input_meter.set_levels(-20.0, -10.0)


def test_scene_that_does_not_load_keeps_the_widget_view(qapp, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("mic_eq.ui.quick_shell.QML_DIR", tmp_path)
    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
    monkeypatch.setattr("mic_eq.ui.main_window.list_presets", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_input_devices", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_output_devices", lambda: [])
    window = MainWindow()
    assert not isinstance(window.centralWidget(), QQuickWidget)
    assert _fallback_notice(window)


def test_opt_out_keeps_the_widget_view(qapp, monkeypatch) -> None:
    monkeypatch.setenv("AUDIOFORGE_QML", "0")
    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
    monkeypatch.setattr("mic_eq.ui.main_window.list_presets", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_input_devices", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_output_devices", lambda: [])
    window = MainWindow()
    assert not isinstance(window.centralWidget(), QQuickWidget)
    assert "quick_bridge" not in window.__dict__


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
        "modified": quick_window.current_preset_modified,
    }
    assert model["eq"]["diagnostics"].data == {"calibrated": False}
    noise, gate = model["stages"][:2]
    assert noise["trace"] and noise["alert"] is not None
    assert not gate["trace"] and gate["alert"] is None
    assert [row["title"] for row in model["health"]["stream"] if row["coded"]] == [
        "Backend",
        "Drops",
        "Recovery",
    ]
