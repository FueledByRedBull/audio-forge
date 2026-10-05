"""View-independent de-esser edits, cutoff normalization and native recovery."""

from dataclasses import asdict

import pytest

from mic_eq.config import DeEsserSettings
from mic_eq.ui.deesser_state import DeEsserState


class RecordingProcessor:
    def __init__(self):
        self.settings = asdict(DeEsserSettings())
        self.calls = []
        self.failures_left = 0
        self.stopped = False

    def __getattr__(self, name):
        if not name.startswith("set_deesser_"):
            raise AttributeError(name)
        key = name.removeprefix("set_deesser_")
        if key not in self.settings:
            raise AttributeError(name)

        def write(value):
            if key == "threshold_db" and self.failures_left:
                self.failures_left -= 1
                raise OSError("de-esser write rejected")
            if key == "low_cut_hz":
                value = min(value, self.settings["high_cut_hz"] - 200.0)
            elif key == "high_cut_hz":
                value = max(value, self.settings["low_cut_hz"] + 200.0)
            self.settings[key] = value
            self.calls.append((key, value))

        return write

    def get_deesser_high_cut_hz(self):
        return self.settings["high_cut_hz"]

    def stop(self):
        self.stopped = True


def test_deesser_coalesces_exact_edits_and_preset_cancels_stale_work(qapp):
    processor = RecordingProcessor()
    state = DeEsserState(processor)
    state.set_settings({})
    state._rate_limiter.interval_ms = 5000
    edits = []
    state.configurationEdited.connect(edits.append)
    initial_calls = len(processor.calls)
    state.set_value("threshold_db", -30.123456)
    state.set_value("threshold_db", -32.123456)
    state.set_value("auto_enabled", False)
    assert state.get_settings()["threshold_db"] == -32.123456
    assert len(processor.calls) == initial_calls
    assert not edits
    state.flush()
    assert processor.settings == state.get_settings()
    assert edits == ["De-esser edit"]

    state.set_value("auto_amount", 0.789123)
    state.set_settings({"auto_amount": 0.234567})
    calls = len(processor.calls)
    state.flush()
    assert len(processor.calls) == calls
    assert processor.settings == asdict(DeEsserSettings()) | {"auto_amount": 0.234567}
    assert state.get_settings() == processor.settings
    assert edits == ["De-esser edit"]


def test_deesser_cutoffs_expand_first_and_normalize_the_other_endpoint(qapp):
    processor = RecordingProcessor()
    state = DeEsserState(processor)
    for low, high in [(4000.0, 4200.0), (8000.123, 8200.123), (3000.0, 3200.0)]:
        state.set_settings({"low_cut_hz": low, "high_cut_hz": high})
        assert processor.settings["low_cut_hz"] == low
        assert processor.settings["high_cut_hz"] == high
        assert state.get_settings() == processor.settings

    state.set_value("low_cut_hz", 12000.0)
    state.flush()
    assert processor.settings["high_cut_hz"] == 12200.0
    state.set_value("high_cut_hz", 2200.0)
    state.flush()
    assert processor.settings["low_cut_hz"] == 2000.0
    state.set_settings({"low_cut_hz": 7000.123, "high_cut_hz": 4000.0})
    assert state.get_settings()["high_cut_hz"] == 7200.123
    assert state.get_settings() == processor.settings


@pytest.mark.parametrize("interactive", [False, True])
def test_deesser_rejection_restores_cutoffs_without_committing_history(qapp, interactive):
    processor = RecordingProcessor()
    state = DeEsserState(processor)
    state.set_settings({"low_cut_hz": 4000.0, "high_cut_hz": 4200.0})
    state._rate_limiter.interval_ms = 5000
    before = state.get_settings()
    edits, errors = [], []
    state.configurationEdited.connect(edits.append)
    state.writeFailed.connect(errors.append)
    processor.failures_left = 1

    with pytest.raises(RuntimeError, match="de-esser write rejected"):
        if interactive:
            state.set_value("low_cut_hz", 8000.0)
            state.flush()
        else:
            state.set_settings({"low_cut_hz": 8000.0, "high_cut_hz": 8200.0})

    assert state.get_settings() == before
    assert processor.settings == before
    assert not edits
    assert errors == ["de-esser write rejected"]
    assert not processor.stopped


def test_failed_deesser_rollback_stops_audio(qapp):
    processor = RecordingProcessor()
    state = DeEsserState(processor)
    recovery_failures = []
    state.recoveryFailed.connect(
        lambda message: recovery_failures.append((message, processor.stopped))
    )
    processor.failures_left = 2
    with pytest.raises(RuntimeError, match="previous settings also failed"):
        state.set_settings({"threshold_db": -30.0})
    assert processor.stopped
    assert len(recovery_failures) == 1
    assert "previous settings also failed" in recovery_failures[0][0]
    assert recovery_failures[0][1] is False


@pytest.mark.parametrize(
    ("key", "value"),
    [("auto_amount", float("nan")), ("low_cut_hz", 1999.0),
     ("high_cut_hz", 16001.0), ("ratio", True), ("auto_enabled", "false")],
)
def test_invalid_deesser_edits_do_not_modify_state_or_native(qapp, key, value):
    processor = RecordingProcessor()
    state = DeEsserState(processor)
    before = state.get_settings()
    with pytest.raises(ValueError):
        state.set_value(key, value)
    with pytest.raises(ValueError):
        state.set_settings({key: value})
    assert state.get_settings() == before
    assert not processor.calls


def test_auto_control_availability_and_cancel_follow_shared_state(qapp):
    state = DeEsserState(RecordingProcessor())
    state.set_settings({})
    state._rate_limiter.interval_ms = 5000
    assert not state.control_enabled("threshold_db")
    assert not state.control_enabled("ratio")
    assert state.control_enabled("auto_amount")
    state.set_value("auto_enabled", False)
    assert state.control_enabled("threshold_db")
    assert state.control_enabled("ratio")
    state.cancel()
    state.flush()
    assert not state.control_enabled("threshold_db")
    assert state.get_settings()["auto_enabled"]
