"""Both frontends share settings; Quick does not need hidden control panels."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPointF, QSignalBlocker, Qt, qInstallMessageHandler
from PySide6.QtGui import QImage, QMouseEvent, QPainter
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QLabel, QMessageBox

from mic_eq.config import AppConfig
from mic_eq.ui.main_window import MainWindow
from mic_eq.ui.quick_shell import (
    SettingControl,
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


def test_quick_processing_runs_without_constructing_classic_panels(quick_window) -> None:
    window = quick_window
    assert "_gate_panel" not in vars(window)
    assert "_model_combo" not in vars(window)
    before = window._get_current_preset()
    window.apply_processing_configuration(before)
    window._commit_pending_configuration_snapshot()
    window._update_meters()
    window._update_diagnostics()
    window.quick_bridge.stepBand(1)
    assert "_gate_panel" not in vars(window)
    assert "_model_combo" not in vars(window)
    assert window._get_current_preset().eq.to_dict() == before.eq.to_dict()


def test_classic_fallback_only_attaches_views_without_native_writes(quick_window, monkeypatch) -> None:
    window = quick_window
    preset = window._get_current_preset()
    preset.compressor.threshold_db = -23.257
    preset.limiter.ceiling_db = -3.257
    preset.rnnoise.strength = 0.6723
    window.apply_processing_configuration(preset)
    expected = window._get_current_preset()

    def unexpected(*_args, **_kwargs):
        pytest.fail("Attaching the classic view must not write DSP settings")

    for state in (window.gate_state, window.eq_state, window.compressor_state,
                  window.deesser_state, window.limiter_state, window.noise_suppression_state):
        monkeypatch.setattr(state, "set_settings", unexpected)
        if hasattr(state, "set_value"):
            monkeypatch.setattr(state, "set_value", unexpected)
    restore_widget_shell(window, "test fallback")
    assert window.compressor_panel.compressor_state is window.compressor_state
    assert window.eq_panel.curve_widget is window.eq_presentation.curve_widget
    assert window.gate_panel.confidence_meter is window.processing_meters.confidence
    assert window._get_current_preset() == expected


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
    bridge._timer.stop()
    rows = _rows(bridge.model["stages"][2])
    state = quick_window.deesser_state
    state.set_settings({"auto_enabled": False, "attack_ms": 2.125})
    attack = rows["Attack"]["proxy"]
    assert attack.value == quick_window.deesser_panel.attack_spinbox.value()
    rows["Auto"]["proxy"].click()
    rows["Auto"]["proxy"].released()
    assert not rows["Threshold"]["proxy"].enabled
    assert not rows["Ratio"]["proxy"].enabled
    rows["Low cut"]["proxy"].setValue(12000)
    rows["Low cut"]["proxy"].released()
    assert rows["High cut"]["proxy"].value == 12200
    assert state.get_settings()["attack_ms"] == 2.125
    assert quick_window._get_current_preset().deesser.high_cut_hz == 12200


def test_state_controls_preserve_widget_presentation_contract(quick_window) -> None:
    gate = quick_window.gate_panel
    deesser = quick_window.deesser_panel
    limiter = quick_window.compressor_panel
    for stage_index, bindings in (
        (1, {
            "Threshold": gate.threshold_spinbox, "Attack": gate.attack_spinbox,
            "Release": gate.release_spinbox, "VAD threshold": gate.vad_threshold_spinbox,
            "Hold time": gate.vad_hold_spinbox, "VAD pre-gain": gate.vad_pre_gain_spinbox,
            "Margin": gate.margin_spinbox,
        }),
        (2, {
            "Amount": deesser.auto_amount_spinbox,
            "Low cut": deesser.low_cut_spinbox,
            "High cut": deesser.high_cut_spinbox,
            "Threshold": deesser.threshold_spinbox,
            "Ratio": deesser.ratio_spinbox,
            "Attack": deesser.attack_spinbox,
            "Release": deesser.release_spinbox,
            "Max reduction": deesser.max_reduction_spinbox,
        }),
        (3, {
            "Threshold": limiter.threshold_spinbox, "Ratio": limiter.ratio_spinbox,
            "Attack": limiter.attack_spinbox, "Release": limiter.release_spinbox,
            "Makeup gain": limiter.makeup_spinbox, "Base release": limiter.base_release_spinbox,
            "Target LUFS": limiter.target_lufs_spinbox,
        }),
        (4, {"Ceiling": limiter.ceiling_spinbox, "Release": limiter.limiter_release_spinbox}),
    ):
        rows = _rows(quick_window.quick_bridge.model["stages"][stage_index])
        for label, widget in bindings.items():
            control = rows[label]["proxy"]
            assert control.name == widget.accessibleName()
            assert control.minimum == widget.minimum()
            assert control.maximum == widget.maximum()
            assert control.step == widget.singleStep()
            assert control.enabled == widget.isEnabled()
            assert control.text == widget.text()
            assert control.toolTip == widget.toolTip()


def test_limiter_reads_shared_state_without_polling_widgets(quick_window) -> None:
    bridge = quick_window.quick_bridge
    bridge._timer.stop()
    control = _rows(bridge.model["stages"][-1])["Ceiling"]["proxy"]
    assert isinstance(control, SettingControl)
    widget = quick_window.compressor_panel.ceiling_spinbox
    assert all(proxy.target is not widget for proxy in bridge._proxies)
    seen = []
    control.changed.connect(lambda: seen.append(control.value))

    quick_window.limiter_state.set_settings({"ceiling_db": -3.257})
    assert seen[-1] == -3.26
    assert control.text == widget.text()
    assert quick_window._get_current_preset().limiter.ceiling_db == -3.257
    # Formatting or a signal-blocked change in a view cannot become product state.
    with QSignalBlocker(widget):
        widget.setValue(-9.0)
    assert control.value == -3.26
    assert quick_window.compressor_panel.get_limiter_settings()["ceiling_db"] == -3.257
    quick_window.limiter_state.changed.emit()
    assert widget.value() == -3.26


def test_limiter_quick_edits_preserve_exact_preset_undo_and_fallback(quick_window) -> None:
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
    restore_widget_shell(quick_window, "test rendering recovery")
    assert quick_window.compressor_panel.ceiling_spinbox.value() == -2.25
    assert quick_window._get_current_preset().limiter.ceiling_db == -2.25


@pytest.mark.parametrize("value", [-1.125, -2.125, -2.005])
def test_limiter_uses_qt_rounding_in_both_views(quick_window, value) -> None:
    state = quick_window.limiter_state
    control = _rows(quick_window.quick_bridge.model["stages"][-1])["Ceiling"]["proxy"]
    widget = quick_window.compressor_panel.ceiling_spinbox
    state.set_settings({"ceiling_db": value})
    assert control.value == widget.value()
    assert control.text == widget.text()
    assert state.get_settings()["ceiling_db"] == value
    rounded = widget.value()
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
    assert "test limiter write refused" in quick_window.status_bar.currentMessage()
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
    assert "recovery failed" in quick_window.status_bar.currentMessage()
    assert not quick_window._commit_pending_configuration_snapshot()
    state.processor = native
    quick_window.apply_processing_configuration(preset)
    assert "configuration" not in quick_window._temporary_mute_reasons


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
    quick_window.quick_bridge.refresh()
    assert row.checked

    assert not quick_window._register_mute_hotkey("Ctrl+Alt+M")
    quick_window.quick_bridge.refresh()
    assert not action.isChecked()
    assert not row.checked


def test_scene_control_is_told_to_read_back_a_choice_the_widget_refused(quick_window) -> None:
    mode = quick_window.quick_bridge.model["top"]["mode"]
    combo = quick_window.processing_mode_combo
    # The handler puts the old entry back, as a failed backend switch does.
    combo.currentIndexChanged.connect(
        lambda _index: (combo.blockSignals(True), combo.setCurrentIndex(0), combo.blockSignals(False))
    )
    notified: list[bool] = []
    mode.changed.connect(lambda: notified.append(True))
    mode.setIndex(1)
    assert combo.currentIndex() == 0 and mode.index == 0
    assert notified


def test_redo_button_follows_the_history(quick_window) -> None:
    redo = quick_window.quick_bridge.model["presets"]["redo"]
    assert not quick_window.redo_button.isEnabled()
    quick_window.gate_panel.enabled_checkbox.click()
    quick_window.undo_configuration()
    assert quick_window.redo_button.isEnabled() == quick_window._redo_action.isEnabled()
    assert quick_window.redo_button.isEnabled()
    quick_window.quick_bridge.refresh()
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


def test_fallback_gives_painted_widgets_their_size_limits_back(quick_window) -> None:
    curve = quick_window.eq_panel.curve_widget
    item = WidgetItem()
    item.setWidth(640)
    item.setHeight(200)
    item.proxy = quick_window.quick_bridge.model["eq"]["curve"]
    image = QImage(640, 200, QImage.Format.Format_ARGB32_Premultiplied)
    painter = QPainter(image)
    item.paint(painter)
    painter.end()
    assert curve.minimumWidth() == curve.maximumWidth() == 640
    restore_widget_shell(quick_window, "no graphics adapter")
    # Free to follow the window width again, as in the widget view.
    assert curve.minimumWidth() < 640 < curve.maximumWidth()


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

        def set_deesser_threshold_db(self, value):
            if value == -22.0:
                raise RuntimeError("bulk de-esser candidate rejected")
            native.set_deesser_threshold_db(value)

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
