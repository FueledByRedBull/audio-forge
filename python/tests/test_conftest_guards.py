"""The conftest guards must fail a test even when the offender runs in a Qt slot."""

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox


class _Emitter(QObject):
    fired = Signal()


def test_guard_violation_inside_a_slot_is_recorded(qapp, guard_violations):
    emitter = _Emitter()
    emitter.fired.connect(lambda: QMessageBox.warning(None, "Title", "Body"))

    emitter.fired.emit()  # PySide prints and swallows an exception raised in a slot

    assert len(guard_violations) == 1
    guard_violations.clear()  # handled here, so the fixture's teardown assertion stays quiet
