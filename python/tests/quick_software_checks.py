"""The Qt Quick scene checks that matter under software rendering.

Not collected by the normal run: ``test_quick_software_rendering.py`` runs this
file in a child pytest with ``QT_QUICK_BACKEND=software``. The shared tests are
repeated here; the checks below are the ones only this backend needs. The offscreen
platform renders in software anyway, so these show the scene works there, not that the
variable chose the backend.
"""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

import os

from PySide6.QtCore import QObject, QPointF, Qt
from PySide6.QtQuick import QQuickItem
from PySide6.QtTest import QTest

from test_quick_shell import (  # noqa: F401  (tests and fixtures repeated under software rendering)
    mic_page,
    quick_window,
    test_a_drag_moves_the_band_by_the_pointers_travel,
    test_live_data_changes_repaint_existing_qml_items,
    test_scene_controls_drive_the_shared_state,
    test_scene_loads_cleanly,
)
from mic_eq.ui.main_window import MainWindow
from mic_eq.ui.quick_shell import graphics_api


def test_the_scene_renders_with_the_software_adapter_and_without_shadows(quick_window) -> None:  # noqa: F811
    # The request reached this process, and the live scene reports software.
    assert os.environ["QT_QUICK_BACKEND"] == "software"
    root = quick_window.centralWidget().rootObject()
    assert graphics_api(quick_window) == "software"
    assert quick_window._interface_snapshot()["graphics_api"] == "software"
    assert root.property("effects") is False

    # No card builds a shadow effect, and the window still paints.
    pending: list[QQuickItem] = [root]
    while pending:
        item = pending.pop()
        assert "MultiEffect" not in item.metaObject().className()
        pending.extend(item.childItems())
    assert not quick_window.centralWidget().grabFramebuffer().isNull()


def test_a_click_on_the_health_chip_switches_the_page_and_back(quick_window, qapp) -> None:  # noqa: F811
    quick_window.resize(1400, 900)
    qapp.processEvents()
    view = quick_window.centralWidget()
    toggle = view.rootObject().findChild(QObject, "healthToggle")
    mic = view.rootObject().findChild(QObject, "micPage")
    assert toggle is not None and mic is not None
    centre = toggle.mapToScene(QPointF(toggle.width() / 2, toggle.height() / 2)).toPoint()

    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=centre)
    assert quick_window.shell_page.currentIndex() == MainWindow.HEALTH_PAGE_INDEX
    assert not mic.isVisible()
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=centre)
    assert quick_window.shell_page.currentIndex() == 0
    assert mic.isVisible()

