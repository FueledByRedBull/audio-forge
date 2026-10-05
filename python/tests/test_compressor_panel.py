"""Focused compressor panel update behavior."""

from __future__ import annotations

import pytest

from mic_eq import AudioProcessor
from mic_eq.ui.compressor_panel import CompressorPanel
from mic_eq.ui.compressor_state import CompressorState
from mic_eq.ui.limiter_state import LimiterState


def test_compressor_views_render_shared_exact_values_and_control_availability(qapp):
    native = AudioProcessor()
    state = CompressorState(native)
    panel = CompressorPanel(native, compressor_state=state)
    other = CompressorPanel(native, compressor_state=state)
    try:
        state.set_settings({
            "threshold_db": -20.123456, "ratio": 3.123456,
            "release_ms": 222.123456, "base_release_ms": 60.123456,
        })
        assert panel.threshold_spinbox.value() == -20.12
        assert other.ratio_spinbox.value() == 3.12
        panel.adaptive_release_checkbox.setChecked(True)
        assert other.adaptive_release_checkbox.isChecked()
        assert other.base_release_spinbox.isEnabled()
        assert not other.release_spinbox.isEnabled()
        assert state.get_settings()["threshold_db"] == -20.123456
        assert state.get_settings()["ratio"] == 3.123456
        assert native.get_compressor_release() == 60.123456
        other.auto_makeup_checkbox.setChecked(True)
        assert not panel.makeup_slider.isEnabled()
        assert panel.target_lufs_spinbox.isEnabled()
        assert panel.get_compressor_settings() == other.get_compressor_settings()
    finally:
        native.stop()
        panel.deleteLater()
        other.deleteLater()
        qapp.processEvents()


def test_limiter_views_share_exact_state_without_rounding_unedited_fields(qapp):
    native = AudioProcessor()
    state = LimiterState(native)
    panel = CompressorPanel(native, state)
    other_panel = CompressorPanel(native, state)
    edits = []
    panel.configurationEdited.connect(edits.append)
    try:
        panel.set_limiter_settings({
            "ceiling_db": -1.234567,
            "release_ms": 83.987654,
            "careful_output_enabled": False,
        })
        assert panel.ceiling_spinbox.value() == -1.23
        assert other_panel.ceiling_spinbox.value() == -1.23
        assert panel.get_limiter_settings()["ceiling_db"] == -1.234567
        assert not edits

        other_panel.limiter_enabled_checkbox.setChecked(False)
        state.flush()
        assert not panel.limiter_enabled_checkbox.isChecked()
        assert not native.is_limiter_enabled()
        assert panel.get_limiter_settings() == other_panel.get_limiter_settings()
        assert panel.get_limiter_settings()["ceiling_db"] == -1.234567
        assert panel.get_limiter_settings()["release_ms"] == 83.987654
        assert edits == ["Limiter edit"]

        panel.ceiling_slider.setValue(-25)
        panel.ceiling_slider.sliderReleased.emit()
        assert other_panel.ceiling_spinbox.value() == -2.5
        assert state.get_settings()["ceiling_db"] == -2.5
        assert state.get_settings()["release_ms"] == 83.987654
        assert native.get_limiter_effective_ceiling_db() == -2.5
    finally:
        native.stop()
        panel.deleteLater()
        other_panel.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("adaptive", [False, True])
def test_release_mode_preserves_active_value_across_edits_and_toggle(qapp, adaptive):
    native = AudioProcessor()
    panel = CompressorPanel(native)
    try:
        panel.set_compressor_settings({
            "adaptive_release": adaptive,
            "release_ms": 220.0,
            "base_release_ms": 60.0,
        })
        assert native.get_compressor_release() == (60.0 if adaptive else 220.0)
        panel.threshold_spinbox.setValue(-25.0)
        panel.compressor_state.flush()
        assert native.get_compressor_release() == (60.0 if adaptive else 220.0)
        panel.threshold_spinbox.setValue(-26.0)
        panel.adaptive_release_checkbox.setChecked(not adaptive)
        panel.compressor_state.flush()
        assert native.get_compressor_release() == (220.0 if adaptive else 60.0)
        settings = panel.get_compressor_settings()
        assert settings["release_ms"] == 220.0
        assert settings["base_release_ms"] == 60.0
    finally:
        native.stop()
        panel.deleteLater()
        qapp.processEvents()


def test_bulk_compressor_apply_propagates_native_failure(qapp):
    native = AudioProcessor()

    class FailingProcessor:
        fail = False

        def __getattr__(self, name):
            return getattr(native, name)

        def set_compressor_adaptive_release(self, enabled):
            if self.fail:
                raise RuntimeError("adaptive release write failed")
            native.set_compressor_adaptive_release(enabled)

    processor = FailingProcessor()
    panel = CompressorPanel(processor)
    processor.fail = True

    try:
        with pytest.raises(RuntimeError, match="adaptive release write failed"):
            panel.set_compressor_settings({"adaptive_release": True})
    finally:
        native.stop()
        panel.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize(
    ("running", "compressor_enabled", "bypass", "raw", "expected"),
    [
        (True, True, False, False, "83 ms"),
        (True, False, False, False, "--"),
        (True, True, True, False, "--"),
        (True, True, False, True, "--"),
        (False, True, False, False, "--"),
    ],
)
def test_current_release_is_only_shown_when_compressor_is_processing(
    qapp, running, compressor_enabled, bypass, raw, expected
):
    native = AudioProcessor()

    class ProcessorState:
        def __init__(self):
            self.running = running
            self.compressor_enabled = compressor_enabled
            self.bypass = bypass
            self.raw = raw

        def __getattr__(self, name):
            return getattr(native, name)

        def is_running(self):
            return self.running

        def is_compressor_enabled(self):
            return self.compressor_enabled

        def is_bypass(self):
            return self.bypass

        def is_raw_monitor_enabled(self):
            return self.raw

        def get_compressor_current_release(self):
            return 83.4

    processor = ProcessorState()
    panel = CompressorPanel(processor)

    try:
        panel._update_current_release()
        assert panel.current_release_label.text() == expected
    finally:
        native.stop()
        panel.deleteLater()
        qapp.processEvents()
