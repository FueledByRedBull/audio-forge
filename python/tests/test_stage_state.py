"""The shared stage write protocol, tested once against a fake processor."""

import pytest

from mic_eq.ui import rate_limiter
from mic_eq.ui.stage_state import RateLimitedStageState, StageState, validated_settings


class FakeProcessor:
    def __init__(self, events):
        self.events = events
        self.fail_writes = 0
        self.fail_stop = False

    def write(self, settings):
        if self.fail_writes:
            self.fail_writes -= 1
            self.events.append(("rejected", settings.copy()))
            raise RuntimeError(f"rejected {settings['value']}")
        self.events.append(("write", settings.copy()))

    def stop(self):
        self.events.append(("stop",))
        if self.fail_stop:
            raise RuntimeError("device busy")


class QueuedStage(RateLimitedStageState[dict]):
    _stage_name = "Fake"

    def __init__(self, processor, *, stop_audio=None):
        super().__init__(processor, {"value": 0.0}, stop_audio=stop_audio)
        self.after_apply = 0

    def _write(self, settings):
        self.processor.write(settings)

    def _after_apply(self):
        self.after_apply += 1
        self.processor.events.append(("after_apply", self._settings.copy()))


class SyncStage(StageState[dict]):
    _stage_name = "Sync"

    def __init__(self, processor):
        super().__init__(processor, {"value": 0.0})

    def _write(self, settings):
        self.processor.write(settings)


def _record(state, events):
    state.changed.connect(lambda: events.append(("changed", state._settings.copy())))
    state.configurationEdited.connect(lambda label: events.append(("edited", label)))
    state.writeFailed.connect(lambda message: events.append(("writeFailed", message)))
    state.recoveryFailed.connect(lambda message: events.append(("recoveryFailed", message)))


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: now[0])
    return now


def test_interactive_edits_show_at_once_and_coalesce_within_the_rate_limit(qapp, clock):
    events = []
    processor = FakeProcessor(events)
    state = QueuedStage(processor)
    _record(state, events)

    state._queue({"value": 1.0}, label="Fake edit")
    assert events == [
        ("changed", {"value": 1.0}),
        ("write", {"value": 1.0}),
        ("after_apply", {"value": 1.0}),
        ("edited", "Fake edit"),
    ]
    events.clear()

    clock[0] += 0.010
    state._queue({"value": 2.0}, label="Fake edit")
    state._queue({"value": 3.0}, label="Fake edit")
    assert events == [("changed", {"value": 2.0}), ("changed", {"value": 3.0})]
    assert state._settings == {"value": 3.0}
    assert state._applied_settings == {"value": 1.0}
    assert state._rate_limiter._timer.isActive()
    events.clear()

    state.flush()
    assert events == [
        ("write", {"value": 3.0}),
        ("after_apply", {"value": 3.0}),
        ("edited", "Fake edit"),
    ]
    assert state._applied_settings == {"value": 3.0}
    assert not state._rate_limiter._timer.isActive()


def test_interval_elapsed_writes_the_next_edit_immediately(qapp, clock):
    events = []
    processor = FakeProcessor(events)
    state = QueuedStage(processor)
    state._queue({"value": 1.0}, label="Fake edit")
    clock[0] += 0.010
    state._queue({"value": 2.0}, label="Fake edit")
    assert state._applied_settings == {"value": 1.0}
    clock[0] += 0.033
    state._queue({"value": 3.0}, label="Fake edit")
    assert state._applied_settings == {"value": 3.0}
    assert [event for event in events if event[0] == "write"] == [
        ("write", {"value": 1.0}),
        ("write", {"value": 3.0}),
    ]
    assert not state._rate_limiter._timer.isActive()


def test_immediate_request_and_bulk_write_supersede_a_queued_edit(qapp, clock):
    events = []
    processor = FakeProcessor(events)
    state = QueuedStage(processor)
    _record(state, events)
    state._apply_inline({"value": 1.0})
    clock[0] += 0.010
    state._queue({"value": 2.0}, label="Fake edit")
    events.clear()

    state._apply_inline({"value": 5.0})
    # A bulk write records no edit, and the view hears about the value it did not request.
    assert events == [
        ("write", {"value": 5.0}),
        ("changed", {"value": 5.0}),
        ("after_apply", {"value": 5.0}),
    ]
    assert state._rate_limiter._pending_fn is None
    events.clear()

    state._queue({"value": 6.0}, label="Fake edit")
    state._queue({"value": 7.0}, label="Mode edit", immediate=True)
    assert events == [
        ("changed", {"value": 6.0}),
        ("changed", {"value": 7.0}),
        ("write", {"value": 7.0}),
        ("after_apply", {"value": 7.0}),
        ("edited", "Mode edit"),
    ]
    assert state._rate_limiter._pending_fn is None


def test_unchanged_accepted_write_records_no_edit(qapp, clock):
    events = []
    processor = FakeProcessor(events)
    state = QueuedStage(processor)
    _record(state, events)
    state._apply_inline({"value": 0.0}, label="Fake edit")
    assert ("edited", "Fake edit") not in events
    state._apply_inline({"value": 1.0}, label="Fake edit")
    assert events[-1] == ("edited", "Fake edit")


def test_unnotified_request_notifies_the_view_when_its_write_is_accepted(qapp, clock):
    events = []
    processor = FakeProcessor(events)
    state = QueuedStage(processor)
    _record(state, events)
    state._apply_inline({"value": 1.0})
    clock[0] += 0.010
    events.clear()

    state._queue({"value": 2.0}, notify=False)
    assert events == []
    assert state._settings == {"value": 2.0}
    state.flush()
    assert events == [
        ("write", {"value": 2.0}),
        ("changed", {"value": 2.0}),
        ("after_apply", {"value": 2.0}),
    ]


def test_accepted_callback_runs_after_the_write_is_recorded_and_before_views_hear(qapp, clock):
    events = []
    processor = FakeProcessor(events)
    state = QueuedStage(processor)
    _record(state, events)

    def accepted():
        events.append(("accepted", state._applied_settings.copy(), state._settings.copy()))

    state._apply_inline({"value": 4.0}, on_accepted=accepted)
    assert events == [
        ("write", {"value": 4.0}),
        ("accepted", {"value": 4.0}, {"value": 0.0}),
        ("changed", {"value": 4.0}),
        ("after_apply", {"value": 4.0}),
    ]
    events.clear()
    processor.fail_writes = 1
    with pytest.raises(RuntimeError):
        state._apply_inline({"value": 5.0}, on_accepted=accepted)
    assert not any(event[0] == "accepted" for event in events)


def test_rejected_write_restores_the_applied_values_and_reports_the_error(qapp, clock):
    events = []
    processor = FakeProcessor(events)
    state = QueuedStage(processor)
    _record(state, events)
    state._apply_inline({"value": 1.0})
    clock[0] += 0.010
    state._queue({"value": 2.0}, label="Fake edit")
    state._queue({"value": 3.0}, label="Fake edit")
    events.clear()
    processor.fail_writes = 1

    with pytest.raises(RuntimeError, match="^rejected 3.0$") as raised:
        state.flush()
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert events == [
        ("rejected", {"value": 3.0}),
        ("changed", {"value": 1.0}),
        ("write", {"value": 1.0}),
        ("writeFailed", "rejected 3.0"),
    ]
    assert state._settings == state._applied_settings == {"value": 1.0}
    assert state._rate_limiter._pending_fn is None
    assert not state._rate_limiter._timer.isActive()


def test_rejected_bulk_write_leaves_unchanged_desired_values_without_notifying(qapp, clock):
    events = []
    processor = FakeProcessor(events)
    state = QueuedStage(processor)
    _record(state, events)
    processor.fail_writes = 1
    with pytest.raises(RuntimeError, match="^rejected 9.0$"):
        state._apply_inline({"value": 9.0}, label="Fake edit")
    assert events == [
        ("rejected", {"value": 9.0}),
        ("write", {"value": 0.0}),
        ("writeFailed", "rejected 9.0"),
    ]
    assert state._settings == state._applied_settings == {"value": 0.0}


@pytest.mark.parametrize("routed", [False, True])
def test_failed_rollback_reports_the_barrier_then_stops_audio(qapp, clock, routed):
    events = []
    processor = FakeProcessor(events)
    stop_audio = (lambda: events.append(("routed stop",))) if routed else None
    state = QueuedStage(processor, stop_audio=stop_audio)
    _record(state, events)
    processor.fail_writes = 2
    message = (
        "Fake update failed: rejected 2.0; "
        "restoring its previous settings also failed: rejected 0.0"
    )

    with pytest.raises(RuntimeError) as raised:
        state._apply_inline({"value": 2.0})
    assert str(raised.value) == message
    assert str(raised.value.__cause__) == "rejected 2.0"
    assert events == [
        ("rejected", {"value": 2.0}),
        ("rejected", {"value": 0.0}),
        ("recoveryFailed", message),
        ("routed stop",) if routed else ("stop",),
        ("writeFailed", message),
    ]
    assert state._applied_settings == {"value": 0.0}


def test_failed_stop_after_failed_rollback_is_part_of_the_reported_error(qapp, clock):
    events = []
    processor = FakeProcessor(events)
    processor.fail_stop = True
    state = QueuedStage(processor)
    _record(state, events)
    processor.fail_writes = 2
    barrier = (
        "Fake update failed: rejected 2.0; "
        "restoring its previous settings also failed: rejected 0.0"
    )
    with pytest.raises(RuntimeError) as raised:
        state._apply_inline({"value": 2.0})
    assert str(raised.value) == f"{barrier}; stopping audio also failed: device busy"
    assert [event for event in events if event[0] in {"recoveryFailed", "writeFailed"}] == [
        ("recoveryFailed", barrier),
        ("writeFailed", f"{barrier}; stopping audio also failed: device busy"),
    ]


def test_synchronous_stage_writes_inline_and_rereads_views_on_any_rejection(qapp):
    events = []
    processor = FakeProcessor(events)
    state = SyncStage(processor)
    _record(state, events)
    partial = []

    state._apply(
        {"value": 1.0}, label="Sync edit",
        write=lambda settings: partial.append(settings["value"]),
    )
    assert partial == [1.0]
    assert events == [("changed", {"value": 1.0}), ("edited", "Sync edit")]
    assert state._settings == state._applied_settings == {"value": 1.0}
    events.clear()

    processor.fail_writes = 1
    with pytest.raises(RuntimeError, match="^rejected 2.0$"):
        state._apply({"value": 2.0}, label="Sync edit")
    # Nothing was stored, yet the view may already show the rejected value.
    assert events == [
        ("rejected", {"value": 2.0}),
        ("changed", {"value": 1.0}),
        ("write", {"value": 1.0}),
        ("writeFailed", "rejected 2.0"),
    ]
    assert state._settings == state._applied_settings == {"value": 1.0}


def test_validated_settings_checks_each_kind_and_keeps_omitted_values():
    current = {"enabled": False, "mode": 0, "level": -10.0}

    def mode(value):
        if value not in (0, 1):
            raise ValueError("Mode must be 0 or 1")
        return value

    kinds = {
        "section": "fake",
        "noun": "fake",
        "booleans": {"enabled"},
        "ranges": {"mode": (0, 5), "level": (-20.0, 0.0)},
        "special": {"mode": mode},
    }
    assert validated_settings(current, {"level": -12.5, "mode": 1}, **kinds) == {
        "enabled": False, "mode": 1, "level": -12.5,
    }
    assert current == {"enabled": False, "mode": 0, "level": -10.0}
    # A special check wins over a range of the same name.
    with pytest.raises(ValueError, match="^Mode must be 0 or 1$"):
        validated_settings(current, {"mode": 3}, **kinds)
    with pytest.raises(ValueError, match="^Unknown fake setting: other$"):
        validated_settings(current, {"other": 1}, **kinds)
    with pytest.raises(ValueError, match="Invalid enabled in fake") as raised:
        validated_settings(current, {"enabled": 1}, **kinds)
    assert type(raised.value) is ValueError
    with pytest.raises(ValueError, match="Invalid level in fake"):
        validated_settings(current, {"level": 1.0}, **kinds)
    with pytest.raises(ValueError, match="Invalid level in fake"):
        validated_settings(current, {"level": "-5"}, **kinds)
