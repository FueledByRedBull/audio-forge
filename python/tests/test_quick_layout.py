"""The Qt Quick scene keeps long text inside its column at the minimum window size."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from PySide6.QtCore import QObject, QRectF
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtQuick import QQuickItem
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtTest import QTest

from mic_eq.config import AppConfig
from mic_eq.ui.main_window import MINIMUM_WINDOW_HEIGHT, MINIMUM_WINDOW_WIDTH, MainWindow

INPUT = SimpleNamespace(name="CABLE Output (VB-Audio Virtual Cable)", is_default=True)
OUTPUT = SimpleNamespace(name="Realtek Digital Output (Realtek USB Audio)", is_default=True)
STALE_CALIBRATION = "Saved calibration is stale: devices or settings differ; recalibrate"
LONG_PRESET = "Hands-on test: <weird>/name that keeps going well past any sensible width"
SEGOE_UI = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"


def build_minimum_window(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    preset: str,
    status: str,
    *,
    modified: bool = False,
    width: int = MINIMUM_WINDOW_WIDTH,
) -> MainWindow:
    """The real window at its minimum size; only the device lists are faked."""
    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
    monkeypatch.setattr("mic_eq.ui.main_window.list_presets", lambda: [])
    monkeypatch.setattr("mic_eq.ui.main_window.list_input_devices", lambda: [INPUT])
    monkeypatch.setattr("mic_eq.ui.main_window.list_output_devices", lambda: [OUTPUT])
    window = MainWindow()
    window.meter_timer.stop()
    window.diagnostics_timer.stop()
    window.current_preset_name = preset
    window.preset_modified = modified
    window._update_session_summary()
    # The platform minimum is clamped to the offscreen screen; ask for the real one.
    window.setMinimumSize(MINIMUM_WINDOW_WIDTH, MINIMUM_WINDOW_HEIGHT)
    window.resize(width, MINIMUM_WINDOW_HEIGHT)
    window.show()
    window.status_message.showMessage(status)
    window.shell_texts["calibration_status_label"].update(text=STALE_CALIBRATION)
    for _ in range(3):
        qapp.processEvents()
        QTest.qWait(30)
    return window


@pytest.fixture
def segoe_font(qapp):
    """Text widths depend on the font, and Qt's offscreen platform has none.

    The icon font decides whether undo and redo are icons or wider text buttons.
    """
    if not SEGOE_UI.is_file():
        pytest.skip("The Windows UI font is not installed")
    families = QFontDatabase.applicationFontFamilies(QFontDatabase.addApplicationFont(str(SEGOE_UI)))
    for icons in ("SegoeIcons.ttf", "segmdl2.ttf"):
        QFontDatabase.addApplicationFont(str(SEGOE_UI.with_name(icons)))
    previous = qapp.font()
    qapp.setFont(QFont(families[0], 9))
    yield
    qapp.setFont(previous)


@pytest.fixture(params=["short", "short-unsaved", "long"])
def minimum_window(request, qapp, monkeypatch, segoe_font):
    preset = LONG_PRESET if request.param == "long" else "DaBest"
    window = build_minimum_window(
        qapp, monkeypatch, preset, f"Loaded preset: {preset}", modified=request.param == "short-unsaved"
    )
    yield window
    window.close()
    window.deleteLater()
    qapp.processEvents()


def scene_rect(item: QQuickItem) -> QRectF:
    return item.mapRectToScene(QRectF(0, 0, item.width(), item.height()))


def named(window: MainWindow, name: str) -> QQuickItem:
    item = cast(QQuickWidget, window.centralWidget()).rootObject().findChild(QObject, name)
    assert isinstance(item, QQuickItem), name
    return item


def shown_items(item: QQuickItem):
    """Every visible item below ``item`` that takes up room."""
    for child in item.childItems():
        if child.isVisible() and child.width() > 0 and child.height() > 0:
            yield child
            yield from shown_items(child)


def test_the_scene_is_built_at_its_minimum_width(minimum_window) -> None:
    assert named(minimum_window, "meterPanel").width() > 0
    assert minimum_window.centralWidget().width() == MINIMUM_WINDOW_WIDTH


def test_nothing_in_the_content_column_runs_under_the_meters(minimum_window) -> None:
    meters_left = scene_rect(named(minimum_window, "meterPanel")).left()
    column = named(minimum_window, "contentColumn")
    overflowing = [
        f"{item.metaObject().className()} {item.objectName()!r} right={scene_rect(item).right():.1f}"
        for item in shown_items(column)
        if scene_rect(item).right() > meters_left + 0.5
    ]
    assert not overflowing, f"meter panel starts at {meters_left:.1f}: {overflowing}"


@pytest.mark.parametrize("name", ["micPage", "presetBar", "voiceSetupButton", "bandEditorRow", "gainBox"])
def test_the_mic_page_rows_fit_the_page(minimum_window, name) -> None:
    meters_left = scene_rect(named(minimum_window, "meterPanel")).left()
    box = scene_rect(named(minimum_window, name))
    assert box.right() <= meters_left + 0.5, f"{name} ends at {box.right():.1f}, meters at {meters_left:.1f}"


def test_a_long_preset_name_is_elided_and_whole_in_the_tooltip(minimum_window) -> None:
    name = named(minimum_window, "presetName")
    long_name = minimum_window.current_preset_name == LONG_PRESET
    assert name.property("truncated") is long_name
    hint = next(child for child in name.children() if "Hint" in child.metaObject().className())
    assert hint.property("text") == (LONG_PRESET if long_name else "")


def caption_tip(item: QQuickItem) -> str:
    hint = next(child for child in item.children() if "Hint" in child.metaObject().className())
    return hint.property("text")


def status_caption(window: MainWindow, text: str) -> QQuickItem:
    strip = named(window, "statusStrip")
    return next(child for child in strip.childItems() if child.property("text") == text)


def test_an_elided_status_caption_has_its_full_text_as_its_tip(minimum_window) -> None:
    message = named(minimum_window, "statusMessage")
    calibration = status_caption(minimum_window, STALE_CALIBRATION)
    if minimum_window.current_preset_name == LONG_PRESET:
        assert message.property("truncated") is True
    for caption in (message, calibration):
        assert caption_tip(caption) == (caption.property("text") if caption.property("truncated") else "")


def test_the_calibration_text_keeps_its_natural_width_when_there_is_room(
    qapp, monkeypatch, segoe_font
) -> None:
    window = build_minimum_window(qapp, monkeypatch, "DaBest", "Loaded preset: DaBest", width=1600)
    try:
        assert window.centralWidget().width() == 1600
        calibration = status_caption(window, STALE_CALIBRATION)
        health = max(named(window, "statusStrip").childItems(), key=lambda child: scene_rect(child).left())
        assert calibration.property("truncated") is False
        # The message takes the spare width, so the text ends beside the Health chip.
        text_end = scene_rect(calibration).left() + calibration.property("contentWidth")
        assert scene_rect(health).left() - text_end == pytest.approx(14, abs=1)
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_the_status_strip_texts_do_not_overlap(minimum_window) -> None:
    strip = named(minimum_window, "statusStrip")
    boxes = sorted(
        (scene_rect(child) for child in strip.childItems() if child.isVisible()),
        key=QRectF.left,
    )
    assert len(boxes) == 5
    assert min(box.width() for box in boxes) > 0, "a text was squeezed out"
    for left, right in zip(boxes, boxes[1:]):
        assert left.right() <= right.left(), f"{left} runs into {right}"
    assert boxes[-1].right() <= scene_rect(named(minimum_window, "meterPanel")).left()


LONG_HEALTH_TEXTS = [
    "Drops: 256 samples dropped, 0 dropouts, 1 restart (needs attention: the audio engine reported a fault)",
    "Device activity: input has not started sending sound yet, output has not started playing sound yet",
    "Output: LIMITER WORKING HARD (7.5 dB)",
]


def settle(qapp) -> None:
    for _ in range(3):
        qapp.processEvents()
        QTest.qWait(30)


def health_chip(window: MainWindow) -> QQuickItem:
    strip = named(window, "statusStrip")
    return next(child for child in strip.childItems() if "Chip" in child.metaObject().className())


@pytest.mark.parametrize("text", LONG_HEALTH_TEXTS)
def test_a_long_health_text_is_cut_short_and_whole_in_the_tooltip_and_the_name(
    minimum_window, qapp, text
) -> None:
    from PySide6.QtGui import QAccessible

    window = minimum_window
    window._set_health_chip("health_summary_label", text, "warn")
    window.shell_texts["health_summary_label"].update(tip="What to do about it.")
    settle(qapp)

    # The other texts keep their room and nothing runs past the strip or into the meters.
    test_the_status_strip_texts_do_not_overlap(window)
    chip = health_chip(window)
    shown = next(child for child in chip.childItems() if child.metaObject().className() == "QQuickText")
    assert shown.property("truncated") is True
    assert caption_tip(chip) == f"{text}\n\nWhat to do about it."
    was_active = QAccessible.isActive()
    QAccessible.setActive(True)
    try:
        toggle = named(window, "healthToggle")
        assert QAccessible.queryAccessibleInterface(toggle).text(QAccessible.Text.Name) == text
    finally:
        QAccessible.setActive(was_active)


def test_a_short_health_text_keeps_its_natural_width_and_its_own_tooltip(minimum_window, qapp) -> None:
    window = minimum_window
    window._set_health_chip("health_summary_label", "Health: OK", "ok")
    window.shell_texts["health_summary_label"].update(tip="Everything looks fine.")
    settle(qapp)

    chip = health_chip(window)
    shown = next(child for child in chip.childItems() if child.metaObject().className() == "QQuickText")
    assert shown.property("truncated") is False
    assert chip.width() == pytest.approx(chip.implicitWidth())
    assert caption_tip(chip) == "Everything looks fine."


def test_the_status_message_keeps_a_quarter_of_the_strip_at_the_worst_combination(
    minimum_window, qapp
) -> None:
    window = minimum_window
    transmission = "Transmission: Running / Raw monitor / Mute pending"
    window.shell_texts["transmission_status_label"].update(text=transmission, status="warn")
    window._set_health_chip("health_summary_label", LONG_HEALTH_TEXTS[0], "warn")
    window.status_message.showMessage("Loaded preset: DaBest")
    settle(qapp)

    # The note gives way first, then the transmission text and the chip; the message does not.
    test_the_status_strip_texts_do_not_overlap(window)
    strip = named(window, "statusStrip")
    message = named(window, "statusMessage")
    assert message.width() >= strip.width() / 4 - 1
    for text in (transmission, STALE_CALIBRATION):
        caption = status_caption(window, text)
        assert caption.property("truncated") is True
        assert caption_tip(caption) == text
