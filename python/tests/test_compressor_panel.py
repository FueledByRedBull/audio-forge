"""Focused compressor panel update behavior."""

from __future__ import annotations

import pytest

from mic_eq import AudioProcessor
from mic_eq.ui.compressor_panel import CompressorPanel


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
        panel._comp_rate_limiter.flush()
        assert native.get_compressor_release() == (60.0 if adaptive else 220.0)
        panel.threshold_spinbox.setValue(-26.0)
        panel.adaptive_release_checkbox.setChecked(not adaptive)
        panel._comp_rate_limiter.flush()
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
