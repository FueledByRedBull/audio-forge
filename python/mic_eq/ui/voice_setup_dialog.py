"""Wizard dialog for Auto Voice Setup."""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QDialog, QMessageBox, QVBoxLayout

from .components import session_is_ending
from .layout_constants import configure_resizable_dialog
from .processing_configuration import ProcessingConfiguration
from .quick_scene import QML_DIR, load_scene, quick_theme
from .quick_shell import register_qml_types
from .voice_setup_flow import VoiceSetupFlow

logger = logging.getLogger(__name__)


class VoiceSetupDialog(QDialog):
    """The window around a ``VoiceSetupFlow``: its scene, its message boxes, its close."""

    # The flow's ``setup_applied``, for the main window.
    setup_applied = Signal(str)

    def __init__(self, parent=None, *, configuration: ProcessingConfiguration | None = None):
        super().__init__(parent)
        self.setWindowTitle("Auto Voice Setup")
        self.setModal(True)
        self.flow = VoiceSetupFlow(
            host=parent, configuration=configuration, prompts=self, parent=self
        )
        self.flow.closed.connect(self._done)
        self.flow.setup_applied.connect(self.setup_applied)
        self._load_error = self._setup_ui()
        if self._load_error is not None:
            # Nothing is recorded; exec() says why and closes.
            logger.error("Voice Setup did not open: %s", self._load_error)
            return
        self.flow.watch_application_quit()
        # The scene's own minimum grows with the text size; the dialog opens at least that wide.
        minimum_width = max(500, int(self.quick_view.rootObject().property("minimumWidth")))
        configure_resizable_dialog(
            self,
            preferred_width=max(760, minimum_width),
            preferred_height=820,
            minimum_width=minimum_width,
            minimum_height=380,
        )

    @property
    def load_error(self) -> str | None:
        """Why the scene did not load, or None; a dialog that did not load records nothing."""
        return self._load_error

    def exec(self) -> int:
        if self._load_error is not None:
            QMessageBox.critical(
                self.parentWidget(),
                self.windowTitle(),
                f"This window could not open.\n\n{self._load_error}",
            )
            return int(QDialog.DialogCode.Rejected)
        return super().exec()

    def _setup_ui(self) -> str | None:
        """Load the scene over the flow's view model; return why it did not load, or None."""
        register_qml_types()
        self.quick_view = QQuickWidget(self)
        reason = load_scene(
            self.quick_view,
            QML_DIR / "VoiceSetup.qml",
            quick_theme(),
            {"vm": self.flow.view_model},
        )
        self.quick_view.setAccessibleName("Auto Voice Setup content")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.quick_view)
        return reason

    def ask(self, title: str, text: str) -> bool:
        """The ``SetupPrompts`` answer the flow passes to ``borrow_audio_for``, with ``fail``."""
        reply = QMessageBox.question(
            self, title, text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        return reply == QMessageBox.StandardButton.Yes

    def fail(self, title: str, text: str) -> None:
        QMessageBox.critical(self, title, text)

    def ask_defaulting_no(self, title: str, text: str) -> bool:
        """``ask`` with No as the default button, so Enter does not say yes."""
        reply = QMessageBox.question(
            self,
            title,
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def inform(self, title: str, text: str) -> None:
        QMessageBox.information(self, title, text)

    def warn(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    def show_comparison(self, **comparison: Any) -> bool | None:
        """Show the listening comparison: whether the proposal was kept, or None when the
        window could not open."""
        from .listening_comparison_dialog import ListeningComparisonDialog

        dialog = ListeningComparisonDialog(**comparison, parent=self)
        try:
            keep = dialog.exec() == int(QDialog.DialogCode.Accepted)
            # A window that could not open says so itself; it is not an answer.
            if not keep and dialog.load_error is not None:
                return None
            return keep
        finally:
            dialog.deleteLater()

    def persist_calibration(
        self, owner: Any, scope: str, curve: str, stages: tuple[str, ...]
    ) -> None:
        from .calibration_history import persist_calibration

        persist_calibration(self, owner, scope, curve, stages)

    def _done(self, accepted: bool) -> None:
        QDialog.done(self, int(QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected))

    def closeEvent(self, event) -> None:
        if session_is_ending():
            # A veto here would stop Windows from signing out: stop the work, join its threads, close.
            self.flow.wait_for_analysis_workers()
            self.flow.finish_close()
            event.accept()
            return
        self.flow.request_close(False)
        event.ignore()

    def accept(self) -> None:
        self.flow.request_close(True)

    def reject(self) -> None:
        """Never leave a temporary candidate active when the dialog closes."""
        self.flow.request_close(False)
