"""Compressor and limiter updates through the shared state and Quick controls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mic_eq import AudioProcessor
from mic_eq.ui.compressor_state import CompressorState
from mic_eq.ui.limiter_state import LimiterState
from mic_eq.ui.processing_controls import SettingControl
from mic_eq.ui.processing_meters import ProcessingMeters


@pytest.fixture
def native(qapp):
    processor = AudioProcessor()
    yield processor
    processor.stop()


def _compressor(state: CompressorState, key: str) -> SettingControl:
    return SettingControl(state, "compressor", key, None)


def _limiter(state: LimiterState, key: str) -> SettingControl:
    return SettingControl(state, "limiter", key, None)


def test_compressor_controls_render_shared_exact_values_and_control_availability(native):
    state = CompressorState(native)
    threshold, ratio = _compressor(state, "threshold_db"), _compressor(state, "ratio")
    release, base_release = _compressor(state, "release_ms"), _compressor(state, "base_release_ms")
    adaptive = _compressor(state, "adaptive_release")
    other_adaptive = _compressor(state, "adaptive_release")
    makeup, target = _compressor(state, "makeup_gain_db"), _compressor(state, "target_lufs")
    auto_makeup = _compressor(state, "auto_makeup_enabled")

    state.set_settings({
        "threshold_db": -20.123456, "ratio": 3.123456,
        "release_ms": 222.123456, "base_release_ms": 60.123456,
    })
    assert threshold.value == -20.12
    assert ratio.value == 3.12

    adaptive.click()
    assert other_adaptive.checked
    assert base_release.enabled
    assert not release.enabled
    assert state.get_settings()["threshold_db"] == -20.123456
    assert state.get_settings()["ratio"] == 3.123456
    assert native.get_compressor_release() == 60.123456

    auto_makeup.click()
    assert not makeup.enabled
    assert target.enabled


def test_limiter_controls_share_exact_state_without_rounding_unedited_fields(native):
    state = LimiterState(native)
    ceiling, other_ceiling = _limiter(state, "ceiling_db"), _limiter(state, "ceiling_db")
    enabled, other_enabled = _limiter(state, "enabled"), _limiter(state, "enabled")
    edits = []
    state.configurationEdited.connect(edits.append)

    state.set_settings({
        "ceiling_db": -1.234567,
        "release_ms": 83.987654,
        "careful_output_enabled": False,
    })
    assert ceiling.value == -1.23
    assert other_ceiling.value == -1.23
    assert state.get_settings()["ceiling_db"] == -1.234567
    assert not edits

    other_enabled.click()
    state.flush()
    assert not enabled.checked
    assert not native.is_limiter_enabled()
    assert state.get_settings()["ceiling_db"] == -1.234567
    assert state.get_settings()["release_ms"] == 83.987654
    assert edits == ["Limiter edit"]

    ceiling.setValue(-2.5)
    ceiling.released()
    assert other_ceiling.value == -2.5
    assert state.get_settings()["ceiling_db"] == -2.5
    assert state.get_settings()["release_ms"] == 83.987654
    assert native.get_limiter_effective_ceiling_db() == -2.5


@pytest.mark.parametrize("adaptive", [False, True])
def test_release_mode_preserves_active_value_across_edits_and_toggle(native, adaptive):
    state = CompressorState(native)
    threshold = _compressor(state, "threshold_db")
    adaptive_release = _compressor(state, "adaptive_release")

    state.set_settings({
        "adaptive_release": adaptive,
        "release_ms": 220.0,
        "base_release_ms": 60.0,
    })
    assert native.get_compressor_release() == (60.0 if adaptive else 220.0)
    threshold.setValue(-25.0)
    state.flush()
    assert native.get_compressor_release() == (60.0 if adaptive else 220.0)
    threshold.setValue(-26.0)
    adaptive_release.click()
    state.flush()
    assert native.get_compressor_release() == (220.0 if adaptive else 60.0)
    settings = state.get_settings()
    assert settings["release_ms"] == 220.0
    assert settings["base_release_ms"] == 60.0


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
    processor = SimpleNamespace(
        is_running=lambda: running,
        is_compressor_enabled=lambda: compressor_enabled,
        is_bypass=lambda: bypass,
        is_raw_monitor_enabled=lambda: raw,
        get_compressor_current_release=lambda: 83.4,
    )
    meters = ProcessingMeters(processor, None)

    meters.update_current_release()

    assert meters.text_states["current_release"].text == expected
