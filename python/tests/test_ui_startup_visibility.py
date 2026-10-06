"""Constructing either frontend must not flash unparented desktop controls."""

import pytest
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QWidget

from mic_eq.config import AppConfig
from mic_eq.ui.main_window import MainWindow


@pytest.mark.parametrize("quick", ["0", "1"], ids=["classic", "quick"])
def test_startup_shows_only_the_completed_main_window(qapp, monkeypatch, quick):
    monkeypatch.setenv("AUDIOFORGE_QML", quick)
    monkeypatch.setattr("mic_eq.ui.main_window.load_config", AppConfig)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)
    for name in ("list_presets", "list_input_devices", "list_output_devices"):
        monkeypatch.setattr(f"mic_eq.ui.main_window.{name}", lambda: [])
    monkeypatch.setattr(MainWindow, "_setup_desktop_integration", lambda _self: None)
    monkeypatch.setattr(MainWindow, "_maybe_show_first_run_setup", lambda _self: None)
    shown = []

    class WindowShows(QObject):
        def eventFilter(self, obj, event):
            if (event.type() == QEvent.Type.Show and isinstance(obj, QWidget)
                    and obj.isWindow()
                    and not obj.testAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)):
                shown.append((type(obj).__name__, obj.accessibleName()))
            return False

    observer = WindowShows()
    qapp.installEventFilter(observer)
    try:
        window = MainWindow()
        assert shown == []
        assert not window.isVisible()
        window.show()
        qapp.processEvents()
        assert shown == [("MainWindow", "")]
    finally:
        qapp.removeEventFilter(observer)
