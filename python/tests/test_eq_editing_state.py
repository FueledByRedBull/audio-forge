"""EQ editing, Auto-EQ results and typed settings through the shared EQ state.

Everything is driven the way the Quick view drives it: the bridge's EQ controls,
the tone menu actions, and mouse and key events in the real scene.
"""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

from typing import Any

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QSignalSpy, QTest

from mic_eq.config import (
    BUILTIN_PRESETS,
    EQ_SLOPES_DB_PER_OCTAVE,
    EQSettings,
    Preset,
    q_from_bandwidth_octaves,
)
from mic_eq.ui.eq_graph import EQGraphGeometry
from mic_eq.ui.eq_state import EQ_FILTER_OPTIONS
import test_quick_shell as quick_shell

# Fixtures and helpers of the Quick shell tests: a real window with the Qt Quick scene.
quick_window = quick_shell.quick_window
mic_page = quick_shell.mic_page
_graph_item = quick_shell._graph_item

TYPES = [choice for _name, choice in EQ_FILTER_OPTIONS]
SLOPES = sorted(EQ_SLOPES_DB_PER_OCTAVE)
AUTO_EQ_BANDS = [
    (72.0, -2.5, 0.7),
    (145.0, -1.2, 1.0),
    (290.0, 0.0, 1.2),
    (580.0, 1.8, 1.4),
    (1150.0, 3.0, 1.6),
    (2300.0, 4.5, 2.0),
    (4600.0, 2.5, 1.8),
    (7600.0, 1.0, 1.2),
    (11100.0, -0.5, 0.9),
    (15100.0, -1.5, 0.7),
]
FLAT_BANDS = [
    (frequency, 0.0, 1.0)
    for frequency in (80.0, 160.0, 320.0, 640.0, 1280.0, 2500.0, 5000.0, 8000.0, 12000.0, 16000.0)
]


def eq_control(window, key: str, index: int | None = None) -> Any:
    """The control the scene binds: one band's, or the section's when no index is given."""
    model = window.quick_bridge.model["eq"]
    return model[key] if index is None else model["bands"][index][key]


def tone_action(window, name: str):
    """The tone menu entry the scene's tone button opens."""
    return next(a for a in window.eq_presentation.tone_menu.actions() if a.text() == name)


def edit(control, value: float) -> None:
    """Set a numeric control and release it, as a drag or a typed value does."""
    control.setValue(value)
    control.released()


def test_auto_eq_results_fill_the_correction_layer_and_a_manual_edit_moves_one_band(
    quick_window, qapp
):
    window = quick_window
    state, graph, processor = window.eq_state, window.eq_presentation.graph_model, window.processor

    state.apply_auto_eq_results(AUTO_EQ_BANDS)
    qapp.processEvents()

    assert graph.band_markers == [freq for freq, _gain, _q in AUTO_EQ_BANDS]
    # Calibration occupies its own stage; editable tone starts flat.
    correction = state.get_eq_settings().correction_bands
    assert correction is not None
    assert [(band.frequency_hz, band.gain_db, band.q) for band in correction] == AUTO_EQ_BANDS
    for i, (expected_freq, _expected_gain, expected_q) in enumerate(AUTO_EQ_BANDS):
        assert eq_control(window, "gain", i).value == 0.0
        assert abs(eq_control(window, "q", i).value - expected_q) <= 0.1
        assert abs(eq_control(window, "frequency", i).value - expected_freq) <= 0.1
        assert state.get_eq_settings().band_freqs[i] == expected_freq
        assert eq_control(window, "frequencyLabel", i).text

    # Existing getters expose the independently editable tone stage.
    for i in range(10):
        params = processor.get_eq_band_params(i)
        assert params is not None
        freq, gain, q = params
        assert abs(freq - AUTO_EQ_BANDS[i][0]) <= 0.1
        assert gain == 0.0
        assert abs(q - AUTO_EQ_BANDS[i][2]) <= 0.1

    # Manual frequency edits should move the active band and marker too.
    edit(eq_control(window, "frequency", 5), 2310.0)
    qapp.processEvents()

    params = processor.get_eq_band_params(5)
    assert params is not None
    freq, gain, q = params
    assert abs(freq - 2310.0) <= 0.1
    assert gain == 0.0
    assert abs(q - AUTO_EQ_BANDS[5][2]) <= 0.1
    assert abs(state.get_eq_settings().band_freqs[5] - 2310.0) <= 0.1
    assert abs(state.get_settings()["band_freqs"][5] - 2310.0) <= 0.1
    assert abs(graph.band_markers[5] - 2310.0) <= 0.1
    assert state.get_eq_settings().correction_bands == correction


def test_catalog_tone_presets_reuse_values_and_preserve_enabled_state(quick_window, qapp):
    window = quick_window
    toggle = eq_control(window, "toggle")
    toggle.click()

    tone_action(window, "Voice").trigger()
    qapp.processEvents()

    expected = BUILTIN_PRESETS["voice"].eq.bands
    actual = window.eq_state.get_eq_settings().bands
    assert toggle.checked is False
    assert [(band.gain_db, band.q) for band in actual] == [
        (band.gain_db, band.q) for band in expected
    ]


def _drag(window, item, band_index: int, target: QPoint, while_dragging=lambda: None) -> None:
    """Press the band's handle and drag to ``target`` (graph coordinates) in the scene."""
    view = window.centralWidget()
    handle = window.eq_presentation.graph_model.band_handle_position(
        band_index, item.width(), item.height()
    )
    start = item.mapToScene(QPointF(*handle)).toPoint()
    destination = item.mapToScene(QPointF(target)).toPoint()
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view, destination, delay=5)
    while_dragging()
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=destination)


def test_eq_graph_drag_updates_numeric_controls_native_dsp_and_signals(
    quick_window, mic_page, qapp
):
    item, _page = mic_page
    window = quick_window
    state, processor = window.eq_state, window.processor
    graph = window.eq_presentation.graph_model
    started = QSignalSpy(state.configurationEditStarted)
    finished = QSignalSpy(state.configurationEditFinished)
    band_index = 4
    width, height = item.width(), item.height()
    start_x, start_y = graph.band_handle_position(band_index, width, height)
    press = QPoint(int(round(start_x)), int(round(start_y)))
    target = QPoint(
        int(round(EQGraphGeometry.frequency_to_x(2100.0, width, height))),
        int(round(EQGraphGeometry.gain_to_y(3.2, width, height))),
    )
    # The handle moves by the pointer's travel from where it was pressed.
    expected_frequency = EQGraphGeometry.x_to_frequency(start_x + target.x() - press.x(), width, height)
    expected_gain = EQGraphGeometry.y_to_gain(start_y + target.y() - press.y(), width, height)

    _drag(window, item, band_index, target)
    qapp.processEvents()

    assert eq_control(window, "frequency", band_index).value == pytest.approx(expected_frequency, abs=1.0)
    # The gain control counts tenths of a dB.
    assert eq_control(window, "gain", band_index).value / 10.0 == pytest.approx(expected_gain, abs=0.1)
    native = processor.get_eq_band_config(band_index)
    assert native is not None
    assert native[1] == pytest.approx(expected_frequency, abs=1.0)
    assert native[2] == pytest.approx(expected_gain, abs=0.1)
    assert started.count() == 1
    assert finished.count() == 1
    assert finished.at(0)[0] == "EQ graph edit"


def test_eq_graph_pass_filter_is_horizontal_only_and_clamped(quick_window, qapp):
    window = quick_window
    # Wide enough that the pointer can travel past the graph's right edge.
    window.resize(1500, 1000)
    qapp.processEvents()
    item = _graph_item(window.centralWidget().rootObject())
    assert item is not None
    band_index = 3
    window.eq_state.set_band_value(band_index, "gain_db", 5.0)
    eq_control(window, "type", band_index).setIndex(TYPES.index("high_pass"))
    window.eq_state.flush()
    qapp.processEvents()

    graph = window.eq_presentation.graph_model

    def pass_filter_follows_the_pointer_horizontally_only() -> None:
        assert graph.dragging_band_index == band_index
        assert graph.bands[band_index][1] == pytest.approx(20_000.0)
        assert graph.bands[band_index][2] == pytest.approx(5.0)

    # Past the right edge and above the top of the graph.
    _drag(
        window, item, band_index, QPoint(int(item.width()) + 100, -100),
        pass_filter_follows_the_pointer_horizontally_only,
    )
    qapp.processEvents()

    native = window.processor.get_eq_band_config(band_index)
    assert native is not None
    assert native[0] == "high_pass"
    assert native[1] == pytest.approx(20_000.0)
    assert native[2] == pytest.approx(5.0)
    assert eq_control(window, "gain", band_index).value / 10.0 == pytest.approx(5.0)


def test_eq_graph_coordinate_mapping_is_deterministic_at_every_width():
    for width in (320, 800, 1600):
        for frequency in (20.0, 80.0, 1000.0, 20_000.0):
            x = EQGraphGeometry.frequency_to_x(frequency, width, 100)
            assert EQGraphGeometry.x_to_frequency(x, width, 100) == pytest.approx(frequency, abs=1.0)
    assert EQGraphGeometry.x_to_frequency(-1000.0, 1600, 100) == 20.0
    assert EQGraphGeometry.x_to_frequency(100_000.0, 1600, 100) == 20_000.0
    assert EQGraphGeometry.y_to_gain(-1000.0, 1600, 100) == 12.0
    assert EQGraphGeometry.y_to_gain(100_000.0, 1600, 100) == -12.0


def test_eq_graph_keyboard_edit_selects_and_moves_a_band(quick_window, mic_page, qapp):
    item, _page = mic_page
    window = quick_window
    view = window.centralWidget()
    window.activateWindow()
    view.setFocus()
    item.forceActiveFocus()
    qapp.processEvents()
    frequency, gain = eq_control(window, "frequency", 0), eq_control(window, "gain", 0)

    QTest.keyClick(view, Qt.Key.Key_BracketRight)
    before = frequency.value
    QTest.keyClick(view, Qt.Key.Key_Right)
    QTest.keyClick(view, Qt.Key.Key_Up)
    qapp.processEvents()

    assert eq_control(window, "band").index == 0
    assert frequency.value > before
    assert gain.value == 1


def test_eq_curve_interaction_warnings(quick_window, qapp):
    window = quick_window
    graph = window.eq_presentation.graph_model

    window.eq_state.apply_preset([0.0] * 10, [1.41] * 10)
    qapp.processEvents()
    assert graph.interaction_warnings == []

    risky_bands = [
        (80.0, 0.0, 1.0),
        (160.0, 0.0, 1.0),
        (300.0, 6.0, 4.5),
        (340.0, 6.0, 4.5),
        (1280.0, 0.0, 1.0),
        (2500.0, 0.0, 1.0),
        (5000.0, 0.0, 1.0),
        (8000.0, 0.0, 1.0),
        (12000.0, 0.0, 1.0),
        (16000.0, 0.0, 1.0),
    ]
    window.eq_state.apply_auto_eq_results(risky_bands)
    qapp.processEvents()

    assert graph.interaction_warnings


def test_auto_eq_diagnostics_are_shown_in_the_eq_diagnostics_control(quick_window, qapp):
    diagnostics = {
        "analysis_confidence": 0.82,
        "eq_confidence": 0.76,
        "capture_confidence": 0.88,
        "validation_confidence": 0.79,
        "validation_before_error_db": 4.2,
        "validation_after_error_db": 2.1,
        "validation_gain_scale": 0.85,
        "target_profile": "broadcast:adaptive",
        "band_confidences": [0.8] * 10,
        "recommendation_status": "apply",
        "low_confidence_active_bands": 0,
    }

    quick_window.eq_state.apply_auto_eq_results(FLAT_BANDS, diagnostics=diagnostics)
    qapp.processEvents()

    text = eq_control(quick_window, "diagnostics").text
    assert "overall 82%" in text
    assert "Correction ready" in text
    assert "EQ 76%" in text
    assert "4.2 dB -> 2.1 dB" in text


def test_manual_filter_type_slope_and_bypass_sync_to_native_and_curve(quick_window, qapp):
    window = quick_window
    graph, processor = window.eq_presentation.graph_model, window.processor
    control = lambda key: eq_control(window, key, 4)  # noqa: E731

    control("type").setIndex(TYPES.index("notch"))
    edit(control("frequency"), 1000.0)
    edit(control("q"), 8.0)
    qapp.processEvents()

    assert processor.get_eq_band_config(4) == ("notch", 1000.0, 0.0, 8.0, 12, True)
    assert not control("gain").enabled
    assert control("q").enabled
    assert not control("slope").enabled
    assert min(graph.response_db) < -20.0

    control("type").setIndex(TYPES.index("high_pass"))
    control("slope").setIndex(SLOPES.index(48))
    qapp.processEvents()

    assert processor.get_eq_band_config(4) == ("high_pass", 1000.0, 0.0, 8.0, 48, True)
    assert not control("gain").enabled
    assert not control("q").enabled
    assert control("slope").enabled

    control("enabled").click()
    qapp.processEvents()
    config = processor.get_eq_band_config(4)
    assert config is not None
    assert config[5] is False
    assert max(abs(value) for value in graph.response_db) < 1e-9


def test_typed_eq_settings_round_trip_through_state_and_preset(quick_window, qapp):
    window = quick_window
    state, processor = window.eq_state, window.processor
    typed = EQSettings().to_dict()
    typed["bands"][0].update(
        {"filter_type": "high_pass", "frequency_hz": 70.0, "slope_db_per_octave": 36}
    )
    typed["bands"][4].update({"filter_type": "notch", "frequency_hz": 2100.0, "q": 7.5})
    exact_frequency = 2345.678
    exact_gain = 1.234
    bandwidth_octaves = 1.234
    exact_q = q_from_bandwidth_octaves(exact_frequency, bandwidth_octaves)
    typed["bands"][5].update(
        {
            "filter_type": "bell",
            "frequency_hz": exact_frequency,
            "gain_db": exact_gain,
            "q": exact_q,
            "bandwidth_mode": "octaves",
            "bandwidth_octaves": bandwidth_octaves,
        }
    )
    typed["bands"][9]["enabled"] = False

    state.set_settings(typed)
    qapp.processEvents()
    settings = state.get_eq_settings()
    restored = Preset.from_dict(Preset(name="Typed", eq=settings).to_dict())

    assert restored.eq == settings
    assert settings.bands[0].filter_type == "high_pass"
    assert settings.bands[0].slope_db_per_octave == 36
    assert settings.bands[4].filter_type == "notch"
    assert settings.bands[4].q == 7.5
    assert settings.bands[5].frequency_hz == exact_frequency
    assert settings.bands[5].gain_db == exact_gain
    assert settings.bands[5].q == exact_q
    assert settings.bands[5].bandwidth_mode == "octaves"
    assert settings.bands[5].bandwidth_octaves == bandwidth_octaves
    assert not settings.bands[9].enabled
    low = processor.get_eq_band_config(0)
    middle = processor.get_eq_band_config(4)
    high = processor.get_eq_band_config(9)
    assert low is not None
    assert middle is not None
    assert high is not None
    assert low[0] == "high_pass"
    assert middle[0] == "notch"
    precise = processor.get_eq_band_config(5)
    assert precise is not None
    assert precise[1] == pytest.approx(exact_frequency, abs=1e-9)
    assert precise[2] == pytest.approx(exact_gain, abs=1e-9)
    assert precise[3] == pytest.approx(exact_q, abs=1e-9)
    assert high[5] is False


def test_auto_eq_explicitly_restores_historical_filter_layout(quick_window, qapp):
    window = quick_window
    eq_control(window, "type", 4).setIndex(TYPES.index("notch"))
    window.eq_state.flush()

    window.eq_state.apply_auto_eq_results([(frequency, 0.0, 1.41) for frequency, _g, _q in FLAT_BANDS])
    qapp.processEvents()

    low = window.processor.get_eq_band_config(0)
    middle = window.processor.get_eq_band_config(4)
    high = window.processor.get_eq_band_config(9)
    assert low is not None
    assert middle is not None
    assert high is not None
    assert low[0] == "low_shelf"
    assert middle[0] == "bell"
    assert high[0] == "high_shelf"


class CountingProcessor:
    def __init__(self, delegate):
        self.delegate = delegate
        self.calls = {"layers": 0, "gain": 0, "q": 0, "frequency": 0}

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def apply_eq_layers(self, correction, tone):
        self.calls["layers"] += 1
        return self.delegate.apply_eq_layers(correction, tone)

    def set_eq_band_gain(self, *args):
        self.calls["gain"] += 1
        return self.delegate.set_eq_band_gain(*args)

    def set_eq_band_q(self, *args):
        self.calls["q"] += 1
        return self.delegate.set_eq_band_q(*args)

    def set_eq_band_frequency(self, *args):
        self.calls["frequency"] += 1
        return self.delegate.set_eq_band_frequency(*args)


def _count_curve_refreshes(graph, monkeypatch) -> list[int]:
    calls = [0]
    original = graph.set_all_params

    def counted(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(graph, "set_all_params", counted)
    return calls


def test_single_band_edits_make_one_layer_update_and_one_curve_refresh(
    quick_window, monkeypatch
):
    window = quick_window
    counted = CountingProcessor(window.processor)
    window.eq_state.processor = counted
    curve_calls = _count_curve_refreshes(window.eq_presentation.graph_model, monkeypatch)

    def reset_counts():
        counted.calls = {"layers": 0, "gain": 0, "q": 0, "frequency": 0}
        curve_calls[0] = 0

    reset_counts()
    edit(eq_control(window, "gain", 4), 10.0)
    assert counted.calls == {"layers": 1, "gain": 0, "q": 0, "frequency": 0}
    assert curve_calls[0] == 1

    reset_counts()
    edit(eq_control(window, "q", 4), 2.1)
    assert counted.calls == {"layers": 1, "gain": 0, "q": 0, "frequency": 0}
    assert curve_calls[0] == 1

    reset_counts()
    edit(eq_control(window, "frequency", 4), 3000.0)
    assert counted.calls == {"layers": 1, "gain": 0, "q": 0, "frequency": 0}
    assert curve_calls[0] == 1
    band_config = window.processor.get_eq_band_config(4)
    assert band_config is not None
    assert band_config[1] == 3000.0


def test_eq_drag_coalesces_native_updates_and_curve_refreshes(quick_window, monkeypatch):
    window = quick_window
    state = window.eq_state
    counted = CountingProcessor(window.processor)
    state.processor = counted
    curve_calls = _count_curve_refreshes(window.eq_presentation.graph_model, monkeypatch)
    state._curve_rate_limiter._last_call_time = 0.0

    state.edit_curve_band(4, 3200.0, 2.0)
    state.edit_curve_band(4, 3300.0, 2.1)
    state.edit_curve_band(4, 3400.0, 2.2)
    state.finish_curve_edit(4, 3400.0, 2.2)

    assert counted.calls["layers"] == 2
    assert curve_calls[0] == 2
    config = window.processor.get_eq_band_config(4)
    assert config is not None and config[1:3] == (3400.0, 2.2)


def test_eq_activity_summary_matches_effective_filters_and_layer_state(quick_window):
    window = quick_window
    layers = eq_control(window, "layers")
    payload = EQSettings().to_dict()
    correction = [dict(band) for band in payload["bands"]]
    correction[0].update(filter_type="high_pass", gain_db=0.0, enabled=True)
    correction[1].update(filter_type="notch", gain_db=0.0, enabled=True)
    correction[2].update(filter_type="bell", gain_db=3.0, enabled=False)
    correction[3].update(filter_type="low_shelf", gain_db=0.0, enabled=True)
    payload["layers"] = {
        "correction": correction,
        "tone": [dict(band) for band in payload["bands"]],
    }

    window.eq_state.set_settings(payload)
    assert "Auto-EQ: 2 active bands" in layers.text
    assert "Tone: 0 active bands" in layers.text

    payload["enabled"] = False
    window.eq_state.set_settings(payload)
    assert "Auto-EQ: 0 active bands" in layers.text
    assert "Tone: 0 active bands" in layers.text
