"""Compressor state owns native ordering, precise controls and transient metadata."""

import pytest

from mic_eq.config import CompressorSettings
from mic_eq.ui.compressor_state import CompressorState


class RecordingProcessor:
    def __init__(self):
        self.calls = []
        self.values = {}
        self.failures_left = 0
        self.stopped = False

    def __getattr__(self, name):
        if not name.startswith("set_compressor_"):
            raise AttributeError(name)

        def write(value):
            if name == "set_compressor_target_lufs" and self.failures_left:
                self.failures_left -= 1
                raise OSError("compressor write rejected")
            self.values[name] = value
            self.calls.append((name.removeprefix("set_compressor_"), value))

        return write

    def stop(self):
        self.stopped = True


def test_adaptive_mode_flushes_old_release_before_applying_new_mode(qapp):
    processor = RecordingProcessor()
    state = CompressorState(processor)
    state.set_settings({"release_ms": 220.123456, "base_release_ms": 60.987654})
    state._rate_limiter.interval_ms = 5000
    processor.calls.clear()
    state.set_value("release_ms", 400.123456)
    assert not processor.calls
    state.set_value("adaptive_release", True)
    assert processor.calls.index(("release", 400.123456)) < processor.calls.index(
        ("adaptive_release", True)
    ) < processor.calls.index(("base_release", 60.987654))
    assert state.get_settings()["release_ms"] == 400.123456
    assert state.get_settings()["base_release_ms"] == 60.987654
    assert not state.control_enabled("release_ms")
    assert state.control_enabled("base_release_ms")

    processor.calls.clear()
    state.set_value("adaptive_release", False)
    assert processor.calls.index(("adaptive_release", False)) < processor.calls.index(
        ("release", 400.123456)
    )
    assert state.control_enabled("release_ms")
    assert not state.control_enabled("base_release_ms")


def test_coalesced_edits_keep_exact_metadata_and_preset_cancels_them(qapp):
    processor = RecordingProcessor()
    state = CompressorState(processor)
    state.set_settings({
        "ratio": 3.123456, "dynamics_profile": "gentle",
        "noise_reference_reliability": 0.7,
    })
    state._rate_limiter.interval_ms = 5000
    edits = []
    state.configurationEdited.connect(edits.append)
    processor.calls.clear()
    state.set_value("threshold_db", -25.123456)
    state.set_value("threshold_db", -26.123456)
    assert not processor.calls
    assert not edits
    state.flush()
    settings = state.get_settings(include_calibration=True)
    assert settings["ratio"] == 3.123456
    assert settings["threshold_db"] == -26.123456
    assert settings["dynamics_profile"] == "gentle"
    assert settings["dynamics_intensity"] == "customized"
    assert settings["noise_reference_reliability"] == 0.7
    assert edits == ["Compressor edit"]

    state.set_value("threshold_db", -27.0)
    state.set_settings({"threshold_db": -22.123456})
    processor.calls.clear()
    state.flush()
    assert not processor.calls
    settings = state.get_settings(include_calibration=True)
    assert settings["threshold_db"] == -22.123456
    assert settings["dynamics_profile"] == "custom"
    assert not settings["dynamics_customized"]
    assert settings["noise_reference_reliability"] == 0.0
    assert edits == ["Compressor edit"]
    CompressorSettings(**state.get_settings())


def test_target_lufs_is_independent_of_calibrated_dynamics_and_auto_enables_controls(qapp):
    state = CompressorState(RecordingProcessor())
    state.set_settings({"dynamics_profile": "gentle", "target_lufs": -18.123456})
    assert not state.control_enabled("target_lufs")
    assert state.control_enabled("makeup_gain_db")
    state.set_value("target_lufs", -17.123456)
    assert not state.get_settings(include_calibration=True)["dynamics_customized"]
    state.set_value("auto_makeup_enabled", True)
    assert state.control_enabled("target_lufs")
    assert not state.control_enabled("makeup_gain_db")


def test_compressor_recommendation_diagnostics_are_ignored_by_bulk_setter(qapp):
    state = CompressorState(RecordingProcessor())
    state.set_settings({
        "ratio": 3.123456,
        "dynamics_profile": "gentle",
        "noise_reference_reliability": 0.7,
        "measured_short_term_lufs": -21.0,
        "recommendation_diagnostics": {"confidence": 0.9},
    })
    settings = state.get_settings(include_calibration=True)
    assert settings["ratio"] == 3.123456
    assert settings["dynamics_profile"] == "gentle"
    assert settings["noise_reference_reliability"] == 0.7
    assert "measured_short_term_lufs" not in settings
    assert "recommendation_diagnostics" not in settings
    with pytest.raises(ValueError, match="Unknown compressor control"):
        state.set_value("measured_short_term_lufs", -21.0)


@pytest.mark.parametrize("interactive", [False, True])
def test_failed_compressor_write_restores_native_and_calibration_metadata(qapp, interactive):
    processor = RecordingProcessor()
    state = CompressorState(processor)
    state.set_settings({
        "ratio": 3.123456, "dynamics_profile": "gentle",
        "noise_reference_reliability": 0.7,
    })
    state._rate_limiter.interval_ms = 5000
    before, native = state.get_settings(include_calibration=True), processor.values.copy()
    edits, errors = [], []
    state.configurationEdited.connect(edits.append)
    state.writeFailed.connect(errors.append)
    processor.failures_left = 1
    with pytest.raises(RuntimeError, match="compressor write rejected"):
        if interactive:
            state.set_value("ratio", 4.0)
            state.flush()
        else:
            state.set_settings({
                "ratio": 4.0, "dynamics_profile": "dense",
                "noise_reference_reliability": 0.1,
            })
    assert state.get_settings(include_calibration=True) == before
    assert processor.values == native
    assert errors == ["compressor write rejected"]
    assert not edits
    assert not processor.stopped


def test_compressor_rollback_failure_signals_barrier_before_stopping(qapp):
    processor = RecordingProcessor()
    state = CompressorState(processor)
    state.set_settings({})
    observed = []
    state.recoveryFailed.connect(lambda _message: observed.append(processor.stopped))
    processor.failures_left = 2
    with pytest.raises(RuntimeError, match="previous settings also failed"):
        state.set_settings({"adaptive_release": True})
    assert observed == [False]
    assert processor.stopped


@pytest.mark.parametrize(
    ("key", "value"),
    [("ratio", float("nan")), ("base_release_ms", 201.0), ("target_lufs", -11.9),
     ("adaptive_release", "false"), ("threshold_db", True)],
)
def test_invalid_compressor_commands_do_not_change_either_side(qapp, key, value):
    processor = RecordingProcessor()
    state = CompressorState(processor)
    before = state.get_settings(include_calibration=True)
    with pytest.raises(ValueError):
        state.set_value(key, value)
    with pytest.raises(ValueError):
        state.set_settings({key: value})
    assert state.get_settings(include_calibration=True) == before
    assert not processor.calls
