"""Noise suppression ownership and rejected writes without either view."""

from dataclasses import asdict

import pytest

from mic_eq.config import RNNoiseSettings
from mic_eq.ui.noise_suppression_state import NoiseSuppressionState


class RecordingProcessor:
    models: tuple[tuple[str, str], ...] = (
        ("rnnoise", "RNNoise"),
        ("deepfilter-ll", "DeepFilterNet LL"),
        ("deepfilter", "DeepFilterNet"),
    )

    def __init__(self):
        self.settings = asdict(RNNoiseSettings())
        self.writes = []
        self.rejected_models = set()
        self.fail_strength = 0
        self.stopped = False

    def list_noise_models(self):
        return list(self.models)

    def set_noise_model(self, value):
        self.writes.append(("model", value))
        if value in self.rejected_models:
            return False
        self.settings["model"] = value
        return True

    def set_rnnoise_enabled(self, value):
        self.writes.append(("enabled", value))
        self.settings["enabled"] = value

    def set_rnnoise_strength(self, value):
        self.writes.append(("strength", value))
        if self.fail_strength:
            self.fail_strength -= 1
            raise RuntimeError("strength write rejected")
        self.settings["strength"] = value

    def stop(self):
        self.stopped = True


def test_defaults_initialization_and_immediate_exact_interactive_edits(qapp):
    processor = RecordingProcessor()
    state = NoiseSuppressionState(processor)
    assert not processor.writes
    assert state.get_settings() == asdict(RNNoiseSettings())
    assert state.models == processor.models
    assert state.latency_text == "Latency: ~10ms (RNNoise)"
    edits, observed = [], []
    state.configurationEdited.connect(edits.append)
    state.changed.connect(lambda: observed.append(state.get_settings()))

    state.set_settings({"enabled": False, "strength": 0.123456789})
    assert processor.writes == [("model", "rnnoise"), ("enabled", False), ("strength", 0.123456789)]
    assert not edits
    processor.writes.clear()
    state.set_value("strength", 0.987654321)
    assert processor.writes == [("strength", 0.987654321)]
    assert state.get_settings() == processor.settings == {
        "enabled": False, "strength": 0.987654321, "model": "rnnoise",
    }
    assert observed[-1] == state.get_settings()
    assert edits == ["Noise suppression edit"]
    state.set_value("strength", 0.987654321)
    assert len(processor.writes) == len(edits) == 1
    snapshot = state.get_settings()
    snapshot["strength"] = 0.0
    assert state.get_settings()["strength"] == 0.987654321


@pytest.mark.parametrize("interactive", [False, True])
def test_rejected_model_restores_exact_state_and_notifies_views(qapp, interactive):
    processor = RecordingProcessor()
    state = NoiseSuppressionState(processor)
    state.set_settings({"enabled": False, "strength": 0.123456789})
    previous = state.get_settings()
    processor.rejected_models.add("deepfilter")
    edits, errors, observed = [], [], []
    state.configurationEdited.connect(edits.append)
    state.writeFailed.connect(errors.append)
    state.changed.connect(lambda: observed.append(state.get_settings()))

    with pytest.raises(RuntimeError, match="previous model retained"):
        if interactive:
            state.set_value("model", "deepfilter")
        else:
            state.set_settings({"model": "deepfilter", "enabled": True, "strength": 0.9})

    assert state.get_settings() == processor.settings == previous
    assert observed == [previous]
    assert not edits
    assert errors == ["Noise model 'deepfilter' is unavailable; previous model retained"]
    assert not processor.stopped


def test_bulk_partial_failure_restores_all_fields_without_history(qapp):
    processor = RecordingProcessor()
    state = NoiseSuppressionState(processor)
    state.set_settings({"strength": 0.123456789})
    previous = state.get_settings()
    edits = []
    state.configurationEdited.connect(edits.append)
    processor.fail_strength = 1

    with pytest.raises(RuntimeError, match="strength write rejected"):
        state.set_settings({"model": "deepfilter-ll", "enabled": False, "strength": 0.5})

    assert state.get_settings() == processor.settings == previous
    assert not edits
    assert not processor.stopped


def test_rollback_failure_reports_barrier_before_stopping_audio(qapp):
    processor = RecordingProcessor()
    state = NoiseSuppressionState(processor)
    events = []
    state.recoveryFailed.connect(lambda message: events.append(("recovery", processor.stopped, message)))
    state.writeFailed.connect(lambda message: events.append(("write", processor.stopped, message)))
    processor.fail_strength = 2

    with pytest.raises(RuntimeError, match="previous settings also failed"):
        state.set_settings({"enabled": False, "strength": 0.5})

    assert processor.stopped
    assert [(event, stopped) for event, stopped, _ in events] == [("recovery", False), ("write", True)]
    assert all("strength write rejected" in message for _, _, message in events)
    assert state.get_settings() == asdict(RNNoiseSettings())


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("strength", float("nan")), ("strength", float("inf")),
        ("strength", -0.01), ("strength", 1.01), ("strength", True),
        ("strength", "0.5"), ("enabled", 1), ("enabled", "false"),
        ("model", "missing"), ("model", None), ("missing", 1),
    ],
)
def test_invalid_settings_never_write_or_change_state(qapp, name, value):
    processor = RecordingProcessor()
    state = NoiseSuppressionState(processor)
    with pytest.raises(ValueError):
        state.set_value(name, value)
    with pytest.raises(ValueError):
        state.set_settings({name: value})
    assert not processor.writes
    assert state.get_settings() == asdict(RNNoiseSettings())


@pytest.mark.parametrize(
    ("model", "label"),
    [("deepfilter", "Latency: ~30ms (DeepFilterNet)"),
     ("deepfilter-ll", "Latency: ~10ms (DeepFilterNet LL)"),
     ("rnnoise", "Latency: ~10ms (RNNoise)")],
)
def test_model_selection_updates_latency_without_touching_other_controls(qapp, model, label):
    processor = RecordingProcessor()
    state = NoiseSuppressionState(processor)
    state.set_value("model", model)
    assert state.latency_text == label
    assert state.get_settings()["model"] == model
    assert all(name == "model" for name, _ in processor.writes)


def test_valid_but_unlisted_model_is_rejected_before_native_switch(qapp):
    processor = RecordingProcessor()
    processor.models = (("rnnoise", "RNNoise"),)
    state = NoiseSuppressionState(processor)
    with pytest.raises(RuntimeError, match="unavailable"):
        state.set_value("model", "deepfilter")
    assert processor.writes == [("model", "rnnoise")]
    assert state.get_settings() == processor.settings == asdict(RNNoiseSettings())
