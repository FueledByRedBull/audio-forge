"""Shared EQ graph data and menus, rendered by the Qt Quick view."""

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QMenu, QWidget

from .eq_graph import EQGraphModel
from .eq_state import EQState
from ..config_parts.app_config import INPUT_CLEANUP_MODES


class EQPresentation(QObject):
    """Shared graph data and menus."""

    def __init__(self, state: EQState, parent: QObject | None = None):
        super().__init__(parent)
        self.eq_state = state
        self.graph_model = EQGraphModel(self)
        widget_parent = parent if isinstance(parent, QWidget) else None
        tone_menu = QMenu(widget_parent)
        for name, tooltip, handler in (
            (
                "Voice",
                "Voice clarity tone; preserves the Auto-EQ layer and other processing",
                lambda: state.apply_catalog_preset("voice"),
            ),
            (
                "Bass Cut",
                "EQ-only bass trimming; not a rumble high-pass filter",
                lambda: state.apply_catalog_preset("bass_cut"),
            ),
            (
                "Presence",
                "Presence tone; preserves the Auto-EQ layer and other processing",
                lambda: state.apply_catalog_preset("presence"),
            ),
            (
                "Warm & Clear",
                "Strong EQ-only replacement: trims bass, lifts low mids, and cuts "
                "harsh upper mids; no rumble high-pass filter",
                state.apply_warm_clear_preset,
            ),
            (
                "Flat",
                "Neutral character gains; preserves the Auto-EQ layer and other processing",
                state.reset_tone,
            ),
        ):
            action = tone_menu.addAction(name.replace("&", "&&"))
            action.setToolTip(tooltip)
            action.triggered.connect(handler)
        tone_menu.setToolTipsVisible(True)
        self.tone_menu = tone_menu

        options_menu = QMenu(widget_parent)
        clear_action = options_menu.addAction("Clear Auto-EQ")
        clear_action.setToolTip("Remove the calibrated EQ layer while keeping the tone")
        clear_action.triggered.connect(state.clear_correction)
        reset_action = options_menu.addAction("Reset tone")
        reset_action.setToolTip(
            "Reset character gains to zero while retaining the Auto-EQ layer"
        )
        reset_action.triggered.connect(state.reset_tone)
        options_menu.setToolTipsVisible(True)
        self.options_menu = options_menu
        self.tone_actions = tuple(self.tone_menu.actions())
        self.options_actions = tuple(self.options_menu.actions())

        self.graph_model.bandSelected.connect(state.select_band)
        self.graph_model.bandResetRequested.connect(state.reset_band)
        self.graph_model.bandDragStarted.connect(state.begin_curve_edit)
        self.graph_model.bandDragged.connect(state.edit_curve_band)
        self.graph_model.bandDragFinished.connect(state.finish_curve_edit)
        self.graph_model.bandDragCancelled.connect(
            lambda index, frequency, gain: state.finish_curve_edit(
                index, frequency, gain, cancelled=True
            )
        )
        state.changed.connect(self._render_curve)
        state.presentationChanged.connect(self._render_presentation)
        # The input conditioning in the curve is whatever the processor runs.
        # The window applies the cleanup mode from several places, some with
        # signals blocked, so the processor is asked on a timer, never at
        # construction, instead of listened to.
        self._conditioning_timer = QTimer(self)
        self._conditioning_timer.setInterval(300)
        self._conditioning_timer.timeout.connect(self.sync_input_conditioning)
        self._conditioning_timer.start()
        for owned in (
            self.tone_menu,
            self.options_menu,
        ):
            self.destroyed.connect(owned.deleteLater)
        self._render_curve()

    def _render_curve(self) -> None:
        settings = self.eq_state.get_eq_settings()
        self.graph_model.set_all_params(
            [band.to_native() for band in settings.bands],
            correction=[band.to_native() for band in settings.correction_bands or ()],
            enabled=settings.enabled,
        )
        self._render_presentation()

    def sync_input_conditioning(self) -> None:
        """Draw what the processor runs: its processing mode, input filter cleanup mode and rate."""
        processor = self.eq_state.processor
        # Raw monitor wins over bypass, as the DSP loop selects its path.
        raw = getattr(processor, "is_raw_monitor_enabled", lambda: False)()
        bypass = getattr(processor, "is_bypass", lambda: False)()
        self.graph_model.set_engine_mode(
            "raw" if raw is True else "bypass" if bypass is True else "normal"
        )
        mode = getattr(processor, "get_input_cleanup_mode", lambda: None)()
        rate = getattr(processor, "sample_rate", lambda: None)()
        if mode in INPUT_CLEANUP_MODES and isinstance(rate, int) and rate > 0:
            self.graph_model.set_input_conditioning(mode, float(rate))

    def _render_presentation(self) -> None:
        if self.eq_state.show_markers:
            self.graph_model.set_band_markers(
                self.eq_state.get_eq_settings().band_freqs
            )
        else:
            self.graph_model.clear_band_markers()
        if self.eq_state.selected_band is not None:
            self.graph_model.select_band(self.eq_state.selected_band)
