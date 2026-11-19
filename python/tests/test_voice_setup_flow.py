"""Contracts for the Voice Setup flow, which runs without a window."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

from copy import deepcopy
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np

from mic_eq.config import Preset
from mic_eq.ui.processing_configuration import Checkpoint
from mic_eq.ui.voice_setup_flow import VoiceSetupFlow, _PreSetup
from test_voice_setup import _full_voice_candidate_metadata


def test_the_flow_module_does_not_import_the_widget_toolkit() -> None:
    script = (
        "import sys\n"
        "import mic_eq.ui.voice_setup_flow\n"
        "print('PySide6.QtWidgets' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


class _Prompts:
    """Records every prompt the flow shows; ``ask_defaulting_no`` answers ``answer``."""

    def __init__(self, answer: bool = False) -> None:
        self.answer = answer
        self.calls: list[tuple[str, str, str]] = []

    def ask(self, title: str, text: str) -> bool:
        self.calls.append(("ask", title, text))
        return True

    def fail(self, title: str, text: str) -> None:
        self.calls.append(("fail", title, text))

    def ask_defaulting_no(self, title: str, text: str) -> bool:
        self.calls.append(("ask_defaulting_no", title, text))
        return self.answer

    def inform(self, title: str, text: str) -> None:
        self.calls.append(("inform", title, text))

    def warn(self, title: str, text: str) -> None:
        self.calls.append(("warn", title, text))

    def compare(self, **_comparison: Any) -> bool | None:
        raise AssertionError("no comparison is expected")

    def persist_calibration(self, *_args: Any) -> None:
        raise AssertionError("nothing is expected to be saved")


class _Configuration:
    """One preset that ``apply`` replaces and ``restore`` puts back; either can be made to fail."""

    calibration_context_key = None
    processing_mode = "normal"

    def __init__(self) -> None:
        self.current = Preset()
        self.applied: list[Preset] = []
        self.fail_apply = False
        self.fail_restore = False

    def checkpoint(self) -> Checkpoint:
        return Checkpoint(Preset.from_dict(self.current.to_dict()), self.processing_mode, None, None)

    def apply(self, preset: Preset, **_kwargs: Any) -> None:
        # A failing apply has already changed part of the chain, as a real one can.
        self.current = preset
        self.applied.append(preset)
        if self.fail_apply:
            raise RuntimeError("sentinel apply failure")

    def restore(self, checkpoint: Checkpoint, **_calibration: Any) -> None:
        if self.fail_restore:
            raise RuntimeError("sentinel restore failure")
        self.current = Preset.from_dict(checkpoint.preset.to_dict())


def _windowless_flow(*, answer: bool = False):
    """A flow over an owner stand-in and a recording prompts fake, with no dialog or scene."""
    configuration = _Configuration()
    records: list[dict[str, Any]] = []
    processor = Mock()
    processor.sample_rate.return_value = 48_000
    processor.get_input_cleanup_mode.return_value = "off"
    owner = SimpleNamespace(
        processor=processor,
        configuration=configuration,
        record_setup_result=records.append,
    )
    prompts = _Prompts(answer)
    flow = VoiceSetupFlow(host=owner, configuration=cast(Any, configuration), prompts=prompts)
    return flow, configuration, prompts, records


def _result(configuration: _Configuration, *, recommended: bool = True) -> dict[str, Any]:
    """A complete candidate over the configuration's own sections, with a louder EQ."""
    current = configuration.current.to_dict()
    return {
        "_candidate": _full_voice_candidate_metadata(),
        "diagnostics": {
            "apply_recommended": recommended,
            "uncertainty_reasons": [] if recommended else ["capture confidence is weak"],
            "setup_confidence": 0.92 if recommended else 0.62,
            "capture_confidence": 0.9 if recommended else 0.7,
            "recommendation_uncertainty": 0.08 if recommended else 0.4,
            "gate_mode_label": "Threshold Gate",
        },
        "gate_settings": deepcopy(current["gate"]),
        "deesser_settings": deepcopy(current["deesser"]),
        "compressor_settings": {**current["compressor"], "threshold_db": -21.0},
        "limiter_settings": {**current["limiter"], "ceiling_db": -2.5},
        "eq_settings": {
            "band_freqs": list(np.geomspace(60.0, 16_000.0, 10)),
            "band_gains": [1.0] * 10,
            "band_qs": [1.41] * 10,
        },
    }


def _analysed(flow: VoiceSetupFlow, result: dict[str, Any]) -> None:
    flow._candidate_metadata = deepcopy(result["_candidate"])
    flow.noise_audio = flow.voice_audio = np.zeros(16, dtype=np.float32)
    flow._workers.generation = 1
    flow.analysis_finished(result, generation=1)


def test_an_apply_failure_puts_the_previous_settings_back(qapp) -> None:
    flow, configuration, prompts, records = _windowless_flow()
    before = configuration.current.to_dict()
    _analysed(flow, _result(configuration))
    assert flow.setup_state == "completed"
    configuration.fail_apply = True

    flow.apply()

    assert len(configuration.applied) == 1, "the candidate was tried"
    assert configuration.current.to_dict() == before
    assert flow._pre_setup_snapshot is None
    assert flow.setup_state == "completed"
    assert prompts.calls == [(
        "fail",
        "Error",
        "The new settings could not be applied, so your previous settings were put back "
        "(sentinel apply failure).",
    )]
    assert (records[-1]["outcome"], records[-1]["reason_codes"]) == ("failed", ["apply_failed"])


def test_an_analysis_from_an_older_take_is_ignored(qapp) -> None:
    flow, configuration, prompts, records = _windowless_flow()
    result = _result(configuration)
    flow._candidate_metadata = deepcopy(result["_candidate"])
    flow.noise_audio = flow.voice_audio = np.zeros(16, dtype=np.float32)
    flow._workers.generation = 2

    flow.analysis_finished(result, generation=1)

    assert flow.setup_result is None
    assert flow.setup_state == "idle"
    assert records == [] and prompts.calls == []

    flow.analysis_finished(result, generation=2)
    assert flow.setup_result is not None and flow.setup_state == "completed"


def test_close_is_refused_while_the_previous_settings_cannot_be_put_back(qapp) -> None:
    flow, configuration, _prompts, _records = _windowless_flow()
    closed: list[bool] = []
    flow.closed.connect(closed.append)
    snapshot = _PreSetup(configuration.checkpoint(), {})
    flow._pre_setup_snapshot = snapshot
    configuration.fail_restore = True

    flow.request_close(False)

    assert closed == []
    assert not flow._close_requested
    assert flow._pre_setup_snapshot is snapshot
    assert flow.view_model.shown["phaseText"] == (
        "Close blocked; previous settings not fully put back"
    )
    assert flow.view_model.warning_text == (
        "Your previous settings could not be fully put back. Press Retake to try again."
    )

    configuration.fail_restore = False
    flow.request_close(False)
    assert closed == [False]


def test_applying_suggestions_asks_with_no_as_the_default(qapp) -> None:
    flow, configuration, prompts, _records = _windowless_flow(answer=False)
    before = configuration.current.to_dict()
    _analysed(flow, _result(configuration, recommended=False))
    assert flow._advisory_only

    flow.apply()

    assert len(prompts.calls) == 1
    kind, title, text = prompts.calls[0]
    assert (kind, title) == ("ask_defaulting_no", "Apply Suggested Settings?")
    assert text.startswith("These settings did not pass every check:\n\n- ")
    assert text.endswith("\n\nApply them anyway?")
    assert configuration.applied == []
    assert configuration.current.to_dict() == before
    assert flow._pre_setup_snapshot is None
    assert flow.setup_state == "completed"
