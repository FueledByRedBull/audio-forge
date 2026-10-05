"""Focused Auto-EQ candidate lifecycle tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from PySide6.QtWidgets import QMessageBox, QWidget

from mic_eq.config import (
    EQSettings,
    Preset,
    build_eq_candidate_settings,
)
from mic_eq.ui.calibration_dialog import CalibrationDialog


class _SignalStub:
    def connect(self, _slot: Any) -> None:
        return None


class _AnalysisWorkerStub:
    def __init__(self, _audio, _sample_rate, target_preset, **kwargs):
        self.target_preset = target_preset
        self.target_mode = kwargs["target_mode"]
        self.smoothing_strength = kwargs["smoothing_strength"]
        self.step_progress = _SignalStub()
        self.result_ready = _SignalStub()
        self.failed = _SignalStub()
        self.finished = _SignalStub()

    def start(self) -> None:
        return None

    def isRunning(self) -> bool:
        return False

    def stop(self) -> None:
        return None

    def deleteLater(self) -> None:
        return None


class _ProcessorStub:
    def sample_rate(self) -> int:
        return 48_000

    def is_running(self) -> bool:
        return False

    def stop_raw_recording(self) -> None:
        return None

    def set_output_mute(self, _muted: bool) -> None:
        return None

    def set_recovery_suppressed(self, _suppressed: bool) -> None:
        return None

    def get_input_cleanup_mode(self) -> str:
        return "gentle"


class _EqStateStub:
    def __init__(self) -> None:
        self.enabled = False
        self.settings = EQSettings(enabled=False)
        self.operations: list[str] = []
        self.applied_bands: list[tuple[float, float, float]] | None = None
        self.diagnostics: dict | None = None
        self.fail_apply = False

    def get_settings(self) -> dict:
        return {"enabled": self.enabled}

    def get_eq_settings(self) -> EQSettings:
        return self.settings

    def apply_auto_eq_results(self, bands, diagnostics=None) -> None:
        self.operations.append("apply")
        if self.fail_apply:
            raise RuntimeError("native apply failed")
        self.applied_bands = list(bands)
        self.diagnostics = diagnostics

    def set_settings(self, settings: dict) -> None:
        self.operations.append("typed")
        if self.fail_apply:
            raise RuntimeError("native typed apply failed")
        self.settings = EQSettings.from_dict(settings)
        self.enabled = self.settings.enabled

    def set_auto_eq_diagnostics(self, diagnostics: dict | None) -> None:
        self.operations.append("diagnostics")
        self.diagnostics = diagnostics


class _Owner(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.processor = _ProcessorStub()
        self.eq_state = _EqStateStub()
        self.applied_preset: Preset | None = None
        self.preset = Preset(eq=EQSettings(enabled=self.eq_state.enabled))

    def _get_current_preset(self) -> Preset:
        return Preset.from_dict(self.preset.to_dict())

    def _processing_mode(self) -> str:
        return "normal"

    def apply_processing_configuration(
        self,
        preset: Preset,
        *,
        noise_reference_reliability: float | None = None,
        processing_mode: str | None = None,
        compressor_metadata: dict[str, object] | None = None,
    ) -> None:
        del noise_reference_reliability, processing_mode, compressor_metadata
        self.applied_preset = Preset.from_dict(preset.to_dict())
        self.preset = self.applied_preset
        self.eq_state.set_settings(preset.eq.to_dict())


def _candidate(gain: float = 1.0, base_eq: EQSettings | None = None) -> dict:
    base_eq = base_eq or EQSettings(enabled=False)
    frequencies = [
        80.0,
        160.0,
        320.0,
        640.0,
        1280.0,
        2500.0,
        5000.0,
        8000.0,
        12000.0,
        16000.0,
    ]
    return {
        "band_freqs": frequencies,
        "band_gains": [gain] * 10,
        "band_qs": [1.41] * 10,
        "apply_recommended": True,
        "validated_candidate_eq": build_eq_candidate_settings(
            base_eq,
            frequencies,
            [gain] * 10,
            [1.41] * 10,
            enabled=True,
        ).to_dict(),
        "headroom_validation": {
            "safe": True,
            "authoritative": True,
            "advisory": False,
            "status": "safe",
        },
    }


def test_correction_candidate_preserves_typed_incumbent_tone() -> None:
    tone = list(EQSettings().bands)
    tone[0] = replace(
        tone[0],
        filter_type="high_pass",
        frequency_hz=125.0,
        slope_db_per_octave=48,
    )
    tone[4] = replace(
        tone[4], filter_type="notch", frequency_hz=2200.0, gain_db=-6.0, q=3.2
    )
    incumbent = EQSettings(bands=tone)

    candidate = build_eq_candidate_settings(
        incumbent,
        [80.0, 160.0, 320.0, 640.0, 1280.0, 2500.0, 5000.0, 8000.0, 12000.0, 16000.0],
        [1.0] * 10,
        [1.41] * 10,
    )

    assert candidate.tone_bands == tuple(tone)
    assert candidate.bands == tuple(tone)
    assert candidate.correction_bands is not None
    assert candidate.correction_bands[0].gain_db == 1.0


def _tone_eq() -> EQSettings:
    tone = list(EQSettings().bands)
    tone[0] = replace(
        tone[0],
        filter_type="high_pass",
        frequency_hz=125.0,
        slope_db_per_octave=48,
    )
    tone[4] = replace(
        tone[4], filter_type="notch", frequency_hz=2200.0, gain_db=-4.0, q=3.2
    )
    return EQSettings(enabled=False, bands=tone)


def _capture(dialog: CalibrationDialog, monkeypatch, curve: str) -> None:
    monkeypatch.setattr(
        "mic_eq.ui.calibration_dialog.AnalysisWorker", _AnalysisWorkerStub
    )
    dialog.curve_combo.setCurrentIndex(dialog.curve_combo.findData(curve))
    dialog.recording_state = "recording"
    dialog._on_recording_complete(np.ones(128, dtype=np.float32))


def test_retake_and_failed_next_analysis_cannot_apply_old_candidate(qapp, monkeypatch):
    owner = _Owner()
    dialog = CalibrationDialog(owner)
    try:
        _capture(dialog, monkeypatch, "broadcast")
        generation_a = dialog._analysis_generation
        assert dialog.recording_state == "analyzing"
        assert dialog.start_button.isEnabled() is False
        assert dialog.curve_combo.isEnabled() is False

        dialog._on_analysis_complete(_candidate(1.0), generation=generation_a)
        assert dialog.recording_state == "ready"
        assert dialog.eq_settings is not None
        assert dialog.curve_combo.isEnabled() is False

        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
        )
        dialog._on_retake_clicked()
        assert dialog.recording_state == "idle"
        assert dialog.eq_settings is None
        assert dialog._candidate_target_metadata is None

        _capture(dialog, monkeypatch, "podcast")
        generation_b = dialog._analysis_generation
        assert dialog.recording_state == "analyzing"
        assert dialog.eq_settings is None
        assert dialog.curve_combo.isEnabled() is False

        dialog._on_analysis_failed("B failed", generation=generation_b)
        assert dialog.recording_state == "ready"
        assert dialog.eq_settings is None
        assert dialog.start_button.text() == "Record Again"
        dialog._apply_eq_settings()
        assert owner.eq_state.applied_bands is None
    finally:
        dialog.reject()
        owner.close()


def test_apply_uses_captured_target_and_enables_eq_after_bands(qapp, monkeypatch):
    owner = _Owner()
    dialog = CalibrationDialog(owner)
    applied_targets: list[str] = []
    dialog.auto_eq_applied.connect(applied_targets.append)
    try:
        _capture(dialog, monkeypatch, "broadcast")
        generation = dialog._analysis_generation
        dialog._on_analysis_complete(
            _candidate(2.0, owner.preset.eq), generation=generation
        )

        dialog.curve_combo.setCurrentIndex(dialog.curve_combo.findData("podcast"))
        assert owner.eq_state.enabled is False
        dialog._on_start_clicked()

        assert owner.eq_state.operations == ["typed", "diagnostics"]
        assert owner.eq_state.enabled is True
        assert owner.applied_preset is not None
        assert owner.eq_state.get_eq_settings().to_dict() == (
            owner.applied_preset.eq.to_dict()
        )
        assert applied_targets == ["broadcast"]
    finally:
        if not dialog._close_requested:
            dialog.reject()
        owner.close()


def test_typed_tone_candidate_matches_audition_and_applied_configuration(
    qapp, monkeypatch
):
    from PySide6.QtWidgets import QDialog

    from mic_eq.analysis import listening_comparison as analysis_comparison
    from mic_eq.ui import listening_comparison_dialog

    owner = _Owner()
    owner.preset = Preset(eq=_tone_eq())
    owner.eq_state.settings = owner.preset.eq
    dialog = CalibrationDialog(owner)
    captured_comparison: dict[str, Any] = {}

    class _ComparisonStub:
        def __init__(self, **kwargs):
            captured_comparison.update(kwargs)

        def exec(self):
            return int(QDialog.DialogCode.Accepted)

        def deleteLater(self):
            return None

    monkeypatch.setattr(
        listening_comparison_dialog,
        "ListeningComparisonDialog",
        _ComparisonStub,
    )
    try:
        _capture(dialog, monkeypatch, "broadcast")
        dialog._on_analysis_complete(
            _candidate(1.5, owner.preset.eq), generation=dialog._analysis_generation
        )

        assert dialog.eq_settings is not None
        expected = EQSettings.from_dict(
            dialog.eq_settings["validated_candidate_eq"]
        ).to_dict()
        assert dialog._candidate_proposal is not None
        proposal = dialog._candidate_proposal
        assert proposal.proposed_preset.eq.to_dict() == expected
        raw_capture = dialog.preview_audio_data
        assert raw_capture is not None

        dialog._compare_recording()

        assert captured_comparison["current_settings"] == (
            proposal.incumbent_preset.eq.to_dict()
        )
        assert captured_comparison["proposed_settings"] == expected
        assert captured_comparison["current_chain_settings"] == (
            captured_comparison["proposed_chain_settings"]
        )

        rendered_eq: list[dict[str, Any]] = []

        def render_candidate(audio, _sample_rate, settings, _chain):
            rendered_eq.append(settings)
            return {
                "output_audio": np.asarray(audio, dtype=np.float32).tolist(),
                "simulation_backend": "rust",
            }

        monkeypatch.setattr(
            analysis_comparison, "simulate_candidate_chain", render_candidate
        )
        render_result = analysis_comparison.render_comparison(
            raw_capture,
            48_000,
            proposal.incumbent_preset.eq.to_dict(),
            proposal.proposed_preset.eq.to_dict(),
            current_chain_settings=proposal.chain,
            proposed_chain_settings=proposal.chain,
        )
        assert render_result.native_authoritative is True
        assert len(rendered_eq) == 2
        proposed_render_eq = EQSettings.from_dict(
            {
                key: rendered_eq[1][key]
                for key in ("schema_version", "enabled", "bands", "layers")
            }
        )
        assert proposed_render_eq.to_dict() == expected

        assert owner.applied_preset is not None
        assert owner.applied_preset.eq.to_dict() == expected
        assert owner.eq_state.get_eq_settings().to_dict() == expected
    finally:
        if not dialog._close_requested:
            dialog.reject()
        owner.close()


def test_candidate_is_stale_after_incumbent_tone_changes(qapp, monkeypatch):
    owner = _Owner()
    owner.preset = Preset(eq=_tone_eq())
    owner.eq_state.settings = owner.preset.eq
    dialog = CalibrationDialog(owner)
    try:
        _capture(dialog, monkeypatch, "broadcast")
        dialog._on_analysis_complete(
            _candidate(1.0, owner.preset.eq), generation=dialog._analysis_generation
        )
        candidate = dialog.eq_settings
        assert candidate is not None

        changed_tone = list(owner.preset.eq.bands)
        changed_tone[1] = replace(changed_tone[1], gain_db=2.0)
        owner.preset = Preset(eq=EQSettings(enabled=False, bands=changed_tone))
        owner.eq_state.settings = owner.preset.eq

        identity_error = dialog._candidate_identity_error(candidate, owner)
        assert identity_error is not None
        assert "changed after analysis" in identity_error
    finally:
        dialog.reject()
        owner.close()


def test_unavailable_headroom_never_enables_apply_or_audition(qapp, monkeypatch):
    owner = _Owner()
    dialog = CalibrationDialog(owner)
    try:
        _capture(dialog, monkeypatch, "broadcast")
        result = _candidate(base_eq=owner.preset.eq)
        result["validated_candidate_eq"] = None
        result["headroom_validation"] = {
            "safe": False,
            "authoritative": False,
            "advisory": False,
            "status": "unavailable",
            "reason": "native simulator unavailable",
        }
        result["apply_recommended"] = False

        dialog._on_analysis_complete(result, generation=dialog._analysis_generation)

        assert dialog._candidate_proposal is None
        assert dialog.eq_settings is None
        assert dialog.compare_button.isEnabled() is False
        assert dialog.start_button.text() == "Record Again"
        assert "could not confirm the correction is safe" in dialog.warning_label.text()
        assert "native simulator unavailable" in dialog.warning_label.toolTip()
    finally:
        dialog.reject()
        owner.close()


def test_failed_typed_apply_does_not_enable_eq(qapp, monkeypatch):
    owner = _Owner()
    owner.eq_state.fail_apply = True
    dialog = CalibrationDialog(owner)
    try:
        _capture(dialog, monkeypatch, "broadcast")
        generation = dialog._analysis_generation
        dialog._on_analysis_complete(_candidate(), generation=generation)
        monkeypatch.setattr(QMessageBox, "critical", lambda *_args, **_kwargs: None)

        dialog._on_start_clicked()

        assert owner.eq_state.enabled is False
        assert owner.eq_state.operations == ["typed", "typed"]
        assert dialog.recording_state == "ready"
    finally:
        if not dialog._close_requested:
            dialog.reject()
        owner.close()
