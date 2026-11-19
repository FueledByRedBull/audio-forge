"""Every dialog scene opens with the shared header: its title, and the cross and Esc key that close it."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PySide6.QtMultimedia import QAudioDevice, QMediaDevices

from capture_helpers import wired_latency_dialog
from mic_eq.ui.calibration_dialog import CalibrationDialog
from mic_eq.ui.listening_comparison_dialog import ListeningComparisonDialog, RawProcessedDialog
from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog
from test_auto_eq_candidate_lifecycle import _Owner as _CalibrationOwner
from test_first_run_setup import opening_setup_dialogs, scene_items
from test_first_run_setup_scene import _owner as _first_run_owner
from test_listening_comparison import _settings

TITLES = {
    "VoiceSetup": "Auto Voice Setup",
    "FirstRunSetup": "AudioForge Setup",
    "AutoEqCalibration": "Auto-EQ Voice Tone",
    "LatencyCalibration": "Latency Calibration",
    "ListeningComparison": "Before/After Listening Comparison",
}


def _headers(dialog: Any) -> list[Any]:
    return [
        item
        for item in scene_items(dialog.quick_view.rootObject())
        if item.metaObject().className().startswith("DialogHeader")
    ]


@pytest.fixture
def dialogs(qapp, monkeypatch):
    """Each dialog scene, opened the way its tests open it."""
    monkeypatch.setattr(ListeningComparisonDialog, "_start_render", lambda self: None)
    monkeypatch.setattr(QMediaDevices, "audioOutputs", lambda: [QAudioDevice()])
    made: list[Any] = []
    owners: list[Any] = []
    first_run = opening_setup_dialogs(qapp)
    open_first_run = next(first_run)

    def shown(dialog: Any) -> Any:
        made.append(dialog)
        dialog.show()
        qapp.processEvents()
        return dialog

    def voice_setup() -> Any:
        owner = _CalibrationOwner()
        owners.append(owner)
        owner.config = SimpleNamespace(  # type: ignore[attr-defined]
            voice_setup_dynamics_intensity="balanced",
            voice_setup_custom_p95_db=3.5,
            voice_setup_custom_peak_cap_db=8.0,
        )
        return shown(VoiceSetupDialog(owner, configuration=owner.configuration))

    def auto_eq() -> Any:
        owner = _CalibrationOwner()
        owners.append(owner)
        return shown(CalibrationDialog(owner, configuration=owner.configuration))

    capture = {"audio_data": np.zeros(480, dtype=np.float32), "sample_rate": 48_000}
    makers: dict[str, Callable[[], Any]] = {
        "VoiceSetup": voice_setup,
        "FirstRunSetup": lambda: open_first_run(_first_run_owner()),
        "AutoEqCalibration": auto_eq,
        "LatencyCalibration": lambda: shown(wired_latency_dialog(None)),
        "ListeningComparison": lambda: shown(
            ListeningComparisonDialog(current_settings=_settings(), proposed_settings=_settings(), **capture)
        ),
    }
    yield makers
    for dialog in made:
        dialog.hide()
        dialog.deleteLater()
    for owner in owners:
        owner.close()
    list(first_run)
    qapp.processEvents()


def test_every_dialog_scene_has_a_dialog_header(dialogs) -> None:
    for scene, title in TITLES.items():
        dialog = dialogs[scene]()
        headers = _headers(dialog)
        assert len(headers) == 1, scene
        assert headers[0].property("title") == title, scene


def test_test_my_sound_keeps_its_own_title_in_the_header(qapp, monkeypatch) -> None:
    monkeypatch.setattr(ListeningComparisonDialog, "_start_render", lambda self: None)
    monkeypatch.setattr(QMediaDevices, "audioOutputs", lambda: [QAudioDevice()])
    dialog = RawProcessedDialog(
        settings=_settings(), audio_data=np.zeros(480, dtype=np.float32), sample_rate=48_000
    )
    try:
        dialog.show()
        qapp.processEvents()
        (header,) = _headers(dialog)
        assert header.property("title") == dialog.windowTitle() == "Test my sound"
    finally:
        dialog.hide()
        dialog.deleteLater()
        qapp.processEvents()
