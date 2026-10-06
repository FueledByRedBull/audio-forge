"""Exact EQ ownership, queued edits and layer recovery without UI controls."""

from dataclasses import replace

import pytest

from mic_eq.config import EQSettings
from mic_eq.ui.eq_state import EQState


class RecordingProcessor:
    def __init__(self):
        self.enabled = True
        self.correction = [band.to_native() for band in EQSettings().bands]
        self.tone = self.correction.copy()
        self.writes = []
        self.fail_next = False
        self.fail_all = False
        self.stopped = False

    def set_eq_enabled(self, enabled):
        self.enabled = enabled

    def apply_eq_layers(self, correction, tone):
        if self.fail_next or self.fail_all:
            self.fail_next = False
            raise RuntimeError("EQ layer write rejected")
        self.correction = correction
        self.tone = tone
        self.writes.append((self.enabled, correction, tone))

    def stop(self):
        self.stopped = True


def test_eq_coalesces_exact_pending_values_and_bulk_settings_supersede_them(qapp):
    native = RecordingProcessor()
    state = EQState(native)
    original = state.get_eq_settings()
    original.bands = tuple(
        replace(band, frequency_hz=2345.678, q=1.234567) if index == 4 else band
        for index, band in enumerate(original.bands)
    )
    state.set_settings(original.to_dict())
    state._rate_limiter.interval_ms = 5000
    edits = []
    state.configurationEdited.connect(edits.append)
    state.set_band_value(4, "gain_db", 1.234567)
    state.set_band_value(4, "gain_db", 2.345678)
    assert len(native.writes) == 1
    assert state.get_band(4).gain_db == 2.345678
    assert not edits
    state.flush()
    assert native.tone[4][1:4] == (2345.678, 2.345678, 1.234567)
    assert edits == ["EQ band edit"]
    state.set_band_value(4, "gain_db", 5.0)
    state.set_settings(original.to_dict())
    state.flush()
    assert state.get_eq_settings() == original
    assert len(native.writes) == 3


def test_eq_layer_commands_preserve_independent_correction_and_enable_state(qapp):
    native = RecordingProcessor()
    state = EQState(native)
    correction = [
        (band.frequency_hz, 2.0 if index == 4 else 0.0, band.q)
        for index, band in enumerate(EQSettings().bands)
    ]
    state.apply_auto_eq_results(correction, {"analysis_confidence": 0.8})
    stored_correction = state.get_eq_settings().correction_bands
    assert stored_correction is not None
    assert all(band.gain_db == 0 for band in state.get_eq_settings().bands)
    state.set_enabled(False)
    state.apply_catalog_preset("voice")
    assert not state.get_eq_settings().enabled
    assert not native.enabled
    assert state.get_eq_settings().correction_bands == stored_correction
    tone = state.get_eq_settings().bands
    state.clear_correction()
    assert state.get_eq_settings().correction_bands is None
    assert state.get_eq_settings().bands == tone
    assert all(band[2] == 0 for band in native.correction)
    state.reset_tone()
    assert all(band.gain_db == 0 for band in state.get_eq_settings().bands)
    assert not state.show_markers


def test_array_preset_retains_defaults_for_omitted_q_and_frequency_values(qapp):
    state = EQState(RecordingProcessor())
    state.apply_preset([0.0] * 10, [2.5], [90.0])
    assert state.get_band(0).q == 2.5
    assert state.get_band(0).frequency_hz == 90.0
    assert state.get_band(1) == EQSettings().bands[1]


def test_eq_returned_snapshots_and_diagnostics_cannot_mutate_authoritative_state(qapp):
    state = EQState(RecordingProcessor())
    snapshot = state.get_eq_settings()
    snapshot.enabled = False
    snapshot.band_gains = [5.0] * 10
    assert state.get_eq_settings().enabled
    assert state.get_band(0).gain_db == 0
    diagnostics = {"headroom_validation": {"safe": True}}
    state.set_auto_eq_diagnostics(diagnostics)
    diagnostics["headroom_validation"]["safe"] = False
    assert state.auto_eq_diagnostics == {"headroom_validation": {"safe": True}}
    state.set_band_value(0, "gain_db", state.get_band(0).gain_db)
    state.select_band(1)
    assert state.auto_eq_diagnostics == {"headroom_validation": {"safe": True}}


def test_eq_graph_cancel_restores_origin_and_emits_one_complete_transaction(qapp):
    native = RecordingProcessor()
    state = EQState(native)
    origin = state.get_band(4)
    events = []
    state.configurationEditStarted.connect(lambda: events.append("start"))
    state.configurationEditFinished.connect(events.append)
    state.configurationEdited.connect(events.append)
    state.begin_curve_edit(4)
    state.edit_curve_band(4, 3200.0, 2.0)
    state.edit_curve_band(4, 3300.0, 2.5)
    state.finish_curve_edit(4, origin.frequency_hz, origin.gain_db, cancelled=True)
    assert state.get_band(4) == origin
    assert native.tone[4] == origin.to_native()
    assert events == ["start", "Cancelled EQ graph edit"]


def test_eq_graph_pass_filter_ignores_vertical_gain_and_clamps_frequency(qapp):
    native = RecordingProcessor()
    state = EQState(native)
    state.set_band_value(4, "filter_type", "high_pass")
    state.set_band_value(4, "gain_db", 5.0)
    state.flush()
    state.edit_curve_band(4, 25000.0, -12.0)
    state.flush()
    assert state.get_band(4).frequency_hz == 20000.0
    assert state.get_band(4).gain_db == 5.0


@pytest.mark.parametrize("graph_last", [False, True])
def test_interleaved_graph_and_numeric_edits_keep_the_last_value_and_other_band_edits(
    qapp, graph_last
):
    native = RecordingProcessor()
    state = EQState(native)
    state.set_settings(state.get_settings())
    state._rate_limiter.interval_ms = 5000
    state.set_band_value(3, "q", 2.34567)

    def graph():
        state.edit_curve_band(4, 2800.0, 2.0)

    def numeric():
        state.set_band_value(4, "frequency_hz", 3100.0)

    first, last = (numeric, graph) if graph_last else (graph, numeric)
    first()
    last()
    state.flush()
    expected = 2800.0 if graph_last else 3100.0
    assert state.get_band(4).frequency_hz == expected
    assert native.tone[4][1:3] == (expected, 2.0)
    assert native.tone[3][3] == 2.34567
    assert len(native.writes) == 2


def test_eq_graph_rejects_nonfinite_coordinates_before_scheduling(qapp):
    native = RecordingProcessor()
    state = EQState(native)
    previous = state.get_eq_settings()
    with pytest.raises(ValueError, match="finite numbers"):
        state.edit_curve_band(4, float("nan"), 1.0)
    state.flush()
    assert state.get_eq_settings() == previous
    assert not native.writes


@pytest.mark.parametrize("interactive", [False, True])
def test_eq_failed_write_restores_native_and_visible_snapshot_without_history(
    qapp, interactive
):
    native = RecordingProcessor()
    state = EQState(native)
    previous = state.get_eq_settings()
    state.set_settings(previous.to_dict())
    state._rate_limiter.interval_ms = 5000
    edits, errors = [], []
    state.configurationEdited.connect(edits.append)
    state.writeFailed.connect(errors.append)
    native.fail_next = True
    with pytest.raises(RuntimeError, match="EQ layer write rejected"):
        if interactive:
            state.set_band_value(4, "gain_db", 3.0)
            state.flush()
        else:
            candidate = state.get_eq_settings()
            candidate.enabled = False
            candidate.band_gains = [4.0] * 10
            state.set_settings(candidate.to_dict())
    assert state.get_eq_settings() == previous
    assert native.enabled == previous.enabled
    assert native.tone == [band.to_native() for band in previous.bands]
    assert not edits
    assert errors == ["EQ layer write rejected"]


def test_eq_failed_rollback_signals_recovery_before_audio_stop(qapp):
    native = RecordingProcessor()
    state = EQState(native)
    native.fail_all = True
    recovery = []
    state.recoveryFailed.connect(
        lambda message: recovery.append((message, native.stopped))
    )
    with pytest.raises(RuntimeError, match="previous settings also failed"):
        state.set_settings({"enabled": False})
    assert len(recovery) == 1
    assert not recovery[0][1]
    assert native.stopped


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("gain_db", float("nan")),
        ("gain_db", 13.0),
        ("frequency_hz", 0.0),
        ("q", True),
        ("filter_type", "unknown"),
        ("enabled", "false"),
        ("slope_db_per_octave", 13),
        ("missing", 1),
    ],
)
def test_invalid_band_edits_do_not_change_state_or_write_native(qapp, key, value):
    native = RecordingProcessor()
    state = EQState(native)
    previous = state.get_eq_settings()
    with pytest.raises(ValueError):
        state.set_band_value(4, key, value)
    assert state.get_eq_settings() == previous
    assert not native.writes
