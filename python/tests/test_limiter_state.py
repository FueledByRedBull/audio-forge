"""Limiter ownership, coalescing, and rejected-write behavior without a view."""

from dataclasses import asdict

import pytest

from mic_eq.config import LimiterSettings
from mic_eq.ui.limiter_state import LimiterState


class RecordingProcessor:
    def __init__(self):
        self.settings = asdict(LimiterSettings())
        self.writes = []
        self.fail_next_release = False

    def set_limiter_enabled(self, value):
        self.settings["enabled"] = value

    def set_limiter_careful_output_enabled(self, value):
        self.settings["careful_output_enabled"] = value

    def set_limiter_ceiling(self, value):
        self.settings["ceiling_db"] = value

    def set_limiter_release(self, value):
        if self.fail_next_release:
            self.fail_next_release = False
            raise RuntimeError("release write rejected")
        self.settings["release_ms"] = value
        self.writes.append(self.settings.copy())


def test_limiter_coalesces_changes_and_preset_supersedes_pending_values(qapp):
    processor = RecordingProcessor()
    state = LimiterState(processor)
    edits = []
    state.configurationEdited.connect(edits.append)
    state.set_settings({})
    state._rate_limiter.interval_ms = 5000

    state.set_value("ceiling_db", -2.0)
    state.set_value("ceiling_db", -3.0)
    state.set_value("release_ms", 62.34567)
    assert state.get_settings()["ceiling_db"] == -3.0
    assert processor.settings["ceiling_db"] == -0.5
    assert len(processor.writes) == 1
    assert not edits

    state.flush()
    assert processor.settings == state.get_settings()
    assert len(processor.writes) == 2
    assert edits == ["Limiter edit"]

    state.set_value("ceiling_db", -4.0)
    state.set_settings({"ceiling_db": -1.234567, "release_ms": 83.987654})
    state.flush()
    assert processor.settings == state.get_settings()
    assert processor.settings["ceiling_db"] == -1.234567
    assert processor.settings["release_ms"] == 83.987654
    assert len(processor.writes) == 3
    assert edits == ["Limiter edit"]


@pytest.mark.parametrize("interactive", [False, True])
def test_rejected_write_restores_native_and_visible_state_without_history(qapp, interactive):
    processor = RecordingProcessor()
    state = LimiterState(processor)
    state.set_settings({"ceiling_db": -1.234567, "release_ms": 83.987654})
    state._rate_limiter.interval_ms = 5000
    previous = state.get_settings()
    edits, errors, observed = [], [], []
    state.configurationEdited.connect(edits.append)
    state.writeFailed.connect(errors.append)
    state.changed.connect(lambda: observed.append(state.get_settings()))
    processor.fail_next_release = True

    with pytest.raises(RuntimeError, match="release write rejected"):
        if interactive:
            state.set_value("ceiling_db", -4.0)
            assert state.get_settings()["ceiling_db"] == -4.0
            state.flush()
        else:
            state.set_settings({"enabled": False, "ceiling_db": -4.0})

    assert state.get_settings() == previous
    assert processor.settings == previous
    assert not edits
    assert errors == ["release write rejected"]
    if interactive:
        assert observed[-1] == previous
    state.flush()
    assert state.get_settings() == previous


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ceiling_db", float("nan")),
        ("ceiling_db", float("inf")),
        ("ceiling_db", -12.1),
        ("ceiling_db", True),
        ("release_ms", 9.9),
        ("release_ms", 500.1),
        ("enabled", "false"),
        ("careful_output_enabled", 1),
        ("missing", 1),
    ],
)
def test_invalid_limiter_edits_do_not_modify_state_or_write_native(qapp, name, value):
    processor = RecordingProcessor()
    state = LimiterState(processor)
    previous = state.get_settings()
    with pytest.raises(ValueError):
        state.set_value(name, value)
    with pytest.raises(ValueError):
        state.set_settings({name: value})
    assert state.get_settings() == previous
    assert not processor.writes


def test_cancel_discards_pending_edit_and_notifies_views(qapp):
    processor = RecordingProcessor()
    state = LimiterState(processor)
    state.set_settings({})
    state._rate_limiter.interval_ms = 5000
    observed = []
    state.changed.connect(lambda: observed.append(state.get_settings()))
    state.set_value("enabled", False)
    state.cancel()
    state.flush()
    assert [settings["enabled"] for settings in observed] == [False, True]
    assert state.get_settings() == processor.settings
    assert len(processor.writes) == 1


def test_failed_native_rollback_stops_audio_and_reports_the_failure(qapp):
    class FailedProcessor(RecordingProcessor):
        stopped = False

        def set_limiter_release(self, _value):
            raise OSError("native unavailable")

        def stop(self):
            self.stopped = True

    processor = FailedProcessor()
    state = LimiterState(processor)
    errors = []
    state.writeFailed.connect(errors.append)
    with pytest.raises(RuntimeError, match="previous settings also failed"):
        state.set_settings({"ceiling_db": -2.0})
    assert processor.stopped
    assert len(errors) == 1
    assert "native unavailable" in errors[0]
