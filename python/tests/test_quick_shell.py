"""Both frontends share settings; Quick builds no widget view."""

from __future__ import annotations

from typing import Any

import pytest
from PySide6.QtCore import (
    QCoreApplication,
    QEvent,
    QPointF,
    QRectF,
    QSignalBlocker,
    Qt,
    QTimer,
    qInstallMessageHandler,
)
from PySide6.QtGui import QImage, QMouseEvent, QPainter
from PySide6.QtTest import QTest
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import (
    QLabel,
    QMessageBox,
    QStatusBar,
)
from shiboken6 import isValid

from mic_eq.config import AppConfig, DeviceIdentity
from mic_eq.ui.main_window import CLASSIC_ATTRIBUTES, MainWindow
from mic_eq.ui.eq_quick_graph import EQGraphItem
from mic_eq.ui.eq_graph import EQGraphGeometry
from mic_eq.ui.quick_meter import QuickMeterItem
from mic_eq.ui.shell_specs import ACTION_SPECS, CHOICE_SPECS
from mic_eq.ui.quick_shell import (
    SettingControl,
    QUICK_VIEW_UNAVAILABLE,
    restore_classic_view,
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


def test_a_quick_session_never_builds_the_widget_view(quick_window, monkeypatch) -> None:
    """Start, run, change pages and close without one classic widget or status bar."""

    window = quick_window
    refreshes: list[int] = []
    monkeypatch.setattr(
        window, "_refresh_login_startup_action", lambda: refreshes.append(window.shell_page.currentIndex())
    )

    def assert_no_widget_view() -> None:
        assert window._classic_root is None
        assert not vars(window).keys() & set(CLASSIC_ATTRIBUTES)
        assert window.findChildren(QStatusBar) == []
        assert isinstance(window.centralWidget(), QQuickWidget)
        assert not window.quick_view_failed

    def text(key: str) -> str:
        return window.shell_texts[key].text

    assert_no_widget_view()
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
    assert_no_widget_view()
    window._quitting = True
    window.close()
    assert_no_widget_view()


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
    restore_classic_view(window, "test fallback")
    assert window.compressor_panel.compressor_state is window.compressor_state
    assert window.eq_panel.curve_widget is window.eq_presentation.curve_widget
    assert window.gate_panel.confidence_meter is window.processing_meters.confidence
    assert window._get_current_preset() == expected
    for name in ("transmission_status_label", "calibration_status_label", "health_summary_label"):
        assert window.shell_texts[name].shown
        assert not getattr(window, name).isHidden()


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
    control = _rows(bridge.model["stages"][-1])["Ceiling"]["proxy"]
    assert isinstance(control, SettingControl)
    widget = quick_window.compressor_panel.ceiling_spinbox
    assert not hasattr(bridge, "_timer")
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
    restore_classic_view(quick_window, "test rendering recovery")
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


def test_scene_loads_without_a_widget_view(quick_window) -> None:
    view = quick_window.centralWidget()
    assert isinstance(view, QQuickWidget)
    assert view.status() == QQuickWidget.Status.Ready
    assert view.errors() == []
    # Binding errors are only warnings to Qt; here they fail the test.
    assert vars(quick_window)["scene_messages"] == []
    assert quick_window._classic_root is None
    assert quick_window.findChildren(QStatusBar) == []
    # Meters age their own peak holds, so nothing needs a visible widget to run.
    meter = quick_window.input_meter
    meter.set_levels(-30.0, -0.2)
    assert meter.clip_flash_counter == 10 and meter.decay_timer.isActive()


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


def _fallback_notice(window) -> bool:
    return any(
        label.text() == QUICK_VIEW_UNAVAILABLE and not label.isHidden()
        for label in window.status_bar.findChildren(QLabel)
    )


def _change_shared_state(window) -> None:
    """Move every kind of shared state away from its startup value."""

    window.status_message.showMessage("Changed while the scene was active")
    window._set_health_chip("input_health_label", "Input: hot", "warn")
    window._set_health_chip("dropped_label", "Drops: 3", "bad")
    window.shell_texts["config_warning_banner"].update(text="Settings were reset", shown=True)
    window.shell_page.setIndex(window.HEALTH_PAGE_INDEX)
    window.processing_mode_choice.setIndex(1)
    window.input_cleanup_mode_choice.setIndex(2)
    window.user_mute_checkbox_action.setChecked(True)
    window.start_btn_action.setEnabled(False)
    window.stop_btn_action.setEnabled(True)
    window.input_meter.set_levels(-20.0, -10.0)


def _assert_widgets_show_the_state(window) -> None:
    """The widget view is the window's content and every widget renders its state."""

    root = window._classic_root
    assert root is not None
    assert window.centralWidget() is root
    assert not root.isHidden() and not window.status_bar.isHidden()
    assert window.status_bar.currentMessage() == window.status_message.text
    for key, state in window.shell_texts.items():
        label = getattr(window, key)
        assert (label.text(), label.toolTip(), label.isHidden()) == (
            state.text, state.toolTip, not state.shown
        ), key
    for key in CHOICE_SPECS:
        combo, choice = getattr(window, f"{key}_combo"), getattr(window, f"{key}_choice")
        assert combo.currentIndex() == choice.currentIndex(), key
        assert [combo.itemText(i) for i in range(combo.count())] == choice.items, key
    for key in ACTION_SPECS:
        button, action = getattr(window, key), getattr(window, f"{key}_action")
        assert (button.isEnabled(), button.isChecked()) == (action.isEnabled(), action.isChecked()), key
        if key != "refresh_btn":  # An icon button keeps its glyph as its text.
            assert button.text() == action.text(), key
    assert window.page_stack.currentIndex() == window.shell_page.currentIndex()
    assert window.nav_group.checkedId() == window.shell_page.currentIndex()
    assert window.health_details_button.isChecked() == (
        window.shell_page.currentIndex() == window.HEALTH_PAGE_INDEX
    )
    running = window.stop_btn_action.isEnabled()
    assert window.stop_btn.isHidden() is not running
    assert window.start_btn.isHidden() is running
    assert window.gate_panel.confidence_meter is window.processing_meters.confidence
    assert window._responsive_layout_compact is (window.width() < window.COMPACT_LAYOUT_BREAKPOINT)
    # Later changes keep rendering into the same widgets.
    window.status_message.showMessage("After the switch")
    window._set_health_chip("input_health_label", "Input: ok", "ok")
    assert window.status_bar.currentMessage() == "After the switch"
    assert window.input_health_label.text() == "Input: ok"
    assert window.input_health_label.property("health_state") == "ok"


def test_render_failure_falls_back_to_the_widget_view_and_says_so(quick_window, qapp) -> None:
    window = quick_window
    scene = window.centralWidget()
    assert isinstance(scene, QQuickWidget)
    _change_shared_state(window)
    assert not window.quick_view_failed

    restore_classic_view(window, "no graphics adapter")

    assert window.quick_view_failed is True
    assert "quick_bridge" not in vars(window)
    assert _fallback_notice(window)
    _assert_widgets_show_the_state(window)
    # The window is on screen, so the new content is too.
    qapp.processEvents()
    assert window.status_bar.isVisible() and window.page_stack.isVisible()
    # The repaint hook is gone with the scene, so the widgets draw as before.
    assert "update" not in window.eq_panel.curve_widget.__dict__
    # The scene is deleted, not left behind a second central widget.
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(scene)
    # A second report from the dead scene changes nothing.
    root = window._classic_root
    restore_classic_view(window, "reported twice")
    assert window._classic_root is root and window.centralWidget() is root


def test_a_widget_view_built_while_the_scene_runs_is_the_one_a_failure_shows(quick_window) -> None:
    window = quick_window
    _change_shared_state(window)
    # Reading a widget builds the view now, behind the scene.
    label = window.dropped_label
    root = window._classic_root
    assert root is not None and root.isHidden() and window.status_bar.isHidden()
    assert isinstance(window.centralWidget(), QQuickWidget)
    assert label.text() == "Drops: 3"
    states = [window.shell_texts[key] for key in ("dropped_label", "input_health_label")]
    receivers = [state.receivers("2changed()") for state in states]
    window._ensure_classic_view()
    assert window._classic_root is root
    assert [state.receivers("2changed()") for state in states] == receivers

    restore_classic_view(window, "no graphics adapter")

    assert window.dropped_label is label and window.centralWidget() is root
    assert [state.receivers("2changed()") for state in states] == receivers
    assert _fallback_notice(window)
    _assert_widgets_show_the_state(window)


def _load_config_without_devices(monkeypatch) -> None:
    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
    monkeypatch.setattr("mic_eq.ui.main_window.list_presets", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_input_devices", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_output_devices", lambda: [])


def test_scene_that_does_not_load_keeps_the_widget_view(qapp, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("mic_eq.ui.quick_shell.QML_DIR", tmp_path)
    _load_config_without_devices(monkeypatch)
    window = MainWindow()
    assert window.quick_view_failed is True
    assert "quick_bridge" not in vars(window)
    assert not isinstance(window.centralWidget(), QQuickWidget)
    assert _fallback_notice(window)
    _assert_widgets_show_the_state(window)
    window.show()
    qapp.processEvents()
    assert window.status_bar.isVisible() and window.page_stack.isVisible()


def test_scene_that_cannot_be_prepared_keeps_the_widget_view(qapp, monkeypatch) -> None:
    def refuse(_window):
        raise RuntimeError("model could not be built")

    monkeypatch.setattr("mic_eq.ui.quick_shell.QuickBridge._build_model", refuse)
    _load_config_without_devices(monkeypatch)
    window = MainWindow()
    assert window.quick_view_failed is True
    assert _fallback_notice(window)
    _assert_widgets_show_the_state(window)


def test_opt_out_keeps_the_widget_view(qapp, monkeypatch) -> None:
    monkeypatch.setenv("AUDIOFORGE_QML", "0")
    _load_config_without_devices(monkeypatch)
    window = MainWindow()
    assert not isinstance(window.centralWidget(), QQuickWidget)
    assert "quick_bridge" not in vars(window)
    # Opting out is a choice, not a failure: no notice, and a smoke test passes.
    assert not window.quick_view_failed
    assert not _fallback_notice(window)
    _assert_widgets_show_the_state(window)
    window.show()
    qapp.processEvents()
    assert window.status_bar.isVisible() and window.page_stack.isVisible()


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


def test_quick_graph_does_not_construct_or_resize_classic_graph(quick_window) -> None:
    presentation = quick_window.eq_presentation
    item = EQGraphItem()
    item.setWidth(640)
    item.setHeight(200)
    item.graph = presentation.graph_model
    image = QImage(640, 200, QImage.Format.Format_ARGB32_Premultiplied)
    painter = QPainter(image)
    item.paint(painter)
    painter.end()
    assert presentation._curve_widget is None
    restore_classic_view(quick_window, "no graphics adapter")
    curve = presentation.curve_widget
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
