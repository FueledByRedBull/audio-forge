"""Focused Auto-EQ candidate lifecycle tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox, QWidget

from mic_eq.config import (
    CompressorSettings,
    EQ_FREQUENCIES,
    EQSettings,
    LimiterSettings,
    Preset,
    build_eq_candidate_settings,
)
from mic_eq.ui.calibration_dialog import CalibrationDialog
from mic_eq.ui.calibration_support import AutoEqCandidate, chain_settings


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
    dialog._on_recording_complete(np.full(128, 0.1, dtype=np.float32))


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


def _render(headroom: float) -> dict[str, float]:
    return {
        "pre_limiter_true_peak_headroom_db": headroom,
        "limiter_gain_reduction_db": 0.0,
        "true_peak_limiter_gain_reduction_db": 0.0,
    }


def _owner_with(**stages: Any) -> _Owner:
    """An owner whose current chain has the given stages (the compressor off unless given)."""
    owner = _Owner()
    stages.setdefault("compressor", CompressorSettings(enabled=False))
    owner.preset = Preset(eq=EQSettings(enabled=False), **stages)
    return owner


def _over_the_rule(base_eq: EQSettings, headroom: dict[str, Any] | None = None) -> dict:
    """A proposal kept although the current EQ fails the headroom rule.

    The figures are the owner's take on the usual chain: the full correction misses the
    rule by 1.283 dB, the current EQ by 1.348, and no correction at all by 0.695.
    """
    result = _candidate(1.0, base_eq)
    result["headroom_validation"] = {
        "safe": False,
        "authoritative": True,
        "advisory": False,
        "meets_advisory_thresholds": False,
        "gain_scale": 1.0,
        "status": "risk_not_worsened",
        "before": _render(-0.283),
        "after": _render(-0.283),
        "reference_excess_db": 1.348,
        "flat_excess_db": 0.695,
        "incumbent_excess_db": 1.348,
        "incumbent_output_true_peak_db": -0.5,
        **(headroom or {}),
    }
    return result


def _no_correction(result: dict) -> dict:
    result.update(
        apply_recommended=False,
        recommendation_status="abstain",
        abstention_reasons=["headroom validation removed the usable correction"],
    )
    result["headroom_validation"]["gain_scale"] = 0.0
    return result


def _left_out(base_eq: EQSettings, headroom: dict[str, Any] | None = None) -> dict:
    """The correction was left out because it would leave less room than the current EQ.

    The current EQ meets the rule here (a half decibel inside it); no correction does not.
    """
    result = _over_the_rule(
        base_eq,
        {
            "status": "risk_worse",
            "gain_scale": 0.0,
            "reference_excess_db": -0.5,
            "incumbent_excess_db": -0.5,
            "incumbent_output_true_peak_db": -3.0,
            **(headroom or {}),
        },
    )
    result.update(
        apply_recommended=False,
        recommendation_status="abstain",
        abstention_reasons=[
            "headroom validation removed the usable correction",
            "no safe candidate remained after full-chain headroom validation",
        ],
        validated_candidate_eq=None,
    )
    return result


def _complete(dialog: CalibrationDialog, monkeypatch, result: dict) -> None:
    _capture(dialog, monkeypatch, "broadcast")
    dialog._on_analysis_complete(result, generation=dialog._analysis_generation)


_READY = (
    "Correction ready. Every band that changed is well supported by your recording. "
    "Largest change 1.0 dB. "
)


def test_proposal_kept_over_the_rule_is_offered_with_the_cause_and_the_fix(qapp, monkeypatch):
    owner = _owner_with()
    dialog = CalibrationDialog(owner)
    try:
        _complete(dialog, monkeypatch, _over_the_rule(owner.preset.eq))

        assert dialog._candidate_proposal is not None
        assert dialog.eq_settings is not None
        assert dialog.compare_button.isEnabled() is True
        assert dialog.start_button.text() == "Apply EQ Settings"
        assert dialog.warning_label.text() == (
            _READY + "Your current settings already leave too little room for loud peaks. "
            "This EQ is no worse than your current one. "
            "Lower the microphone gain by about 2 dB (in Windows or on the microphone) "
            "to leave enough room."
        )
        assert dialog.gain_scale_label.text() == "Headroom check: not met even without the EQ."
        assert dialog._candidate_identity_error(dialog.eq_settings, owner) is None

        dialog._on_start_clicked()

        assert owner.eq_state.enabled is True
        assert owner.applied_preset is not None
    finally:
        if not dialog._close_requested:
            dialog.reject()
        owner.close()


def test_output_that_already_clips_with_the_limiter_off_gets_the_plain_warning(qapp, monkeypatch):
    owner = _owner_with(limiter=LimiterSettings(enabled=False))
    dialog = CalibrationDialog(owner)
    try:
        _complete(
            dialog,
            monkeypatch,
            _over_the_rule(owner.preset.eq, {"incumbent_output_true_peak_db": 0.05}),
        )

        assert dialog._candidate_proposal is not None
        assert dialog.warning_label.text().endswith(
            "Your output already clips on loud peaks. "
            "This EQ is no worse than your current one. "
            "Lower the microphone gain by about 2 dB (in Windows or on the microphone) "
            "to leave enough room. "
            "Turning the limiter on would catch the peaks, but it would not make this check pass."
        )
    finally:
        dialog.reject()
        owner.close()


def test_no_correction_over_the_rule_still_names_the_cause_and_the_fix(qapp, monkeypatch):
    owner = _owner_with()
    dialog = CalibrationDialog(owner)
    try:
        _complete(dialog, monkeypatch, _no_correction(_over_the_rule(owner.preset.eq)))

        assert dialog._candidate_proposal is None
        assert dialog.compare_button.isEnabled() is False
        assert dialog.start_button.text() == "Record Again"
        assert dialog.warning_label.text() == (
            "No correction applied. Not enough headroom was left for a useful correction. "
            "Your current settings already leave too little room for loud peaks. "
            "Lower the microphone gain by about 2 dB (in Windows or on the microphone), "
            "then record again."
        )
    finally:
        dialog.reject()
        owner.close()


def test_a_working_compressor_gets_the_remedy_without_a_figure(qapp, monkeypatch):
    owner = _owner_with(compressor=CompressorSettings(enabled=True))
    dialog = CalibrationDialog(owner)
    try:
        _complete(dialog, monkeypatch, _over_the_rule(owner.preset.eq))

        assert dialog.warning_label.text().endswith(
            "Lower the microphone gain (in Windows or on the microphone) to leave enough room."
        )
    finally:
        dialog.reject()
        owner.close()


def test_automatic_makeup_sends_the_user_to_the_target_loudness(qapp, monkeypatch):
    owner = _owner_with(compressor=CompressorSettings(enabled=True, auto_makeup_enabled=True))
    dialog = CalibrationDialog(owner)
    try:
        _complete(dialog, monkeypatch, _no_correction(_over_the_rule(owner.preset.eq)))

        assert dialog.warning_label.text().endswith(
            "Lower the compressor's target loudness, then record again. "
            "Automatic makeup would undo a lower microphone gain."
        )
    finally:
        dialog.reject()
        owner.close()


def test_a_disabled_compressor_with_automatic_makeup_left_on_gets_the_gain_wording(
    qapp, monkeypatch
):
    owner = _owner_with(compressor=CompressorSettings(enabled=False, auto_makeup_enabled=True))
    dialog = CalibrationDialog(owner)
    try:
        _complete(dialog, monkeypatch, _no_correction(_over_the_rule(owner.preset.eq)))

        assert dialog.warning_label.text().endswith(
            "Lower the microphone gain by about 2 dB (in Windows or on the microphone), "
            "then record again."
        )
    finally:
        dialog.reject()
        owner.close()


def test_a_correction_that_headroom_made_gentler_says_what_would_allow_the_full_one(
    qapp, monkeypatch
):
    owner = _owner_with()
    dialog = CalibrationDialog(owner)
    try:
        result = _candidate(1.0, owner.preset.eq)
        result["headroom_validation"] = {
            "safe": True,
            "authoritative": True,
            "advisory": False,
            "gain_scale": 0.55,
            "status": "safe",
            "before": _render(0.4),
            "after": _render(1.1),
        }
        _complete(dialog, monkeypatch, result)

        assert dialog.warning_label.text() == (
            _READY + "Lower the microphone gain by about 1 dB (in Windows or on the microphone) "
            "to allow the full correction."
        )
        assert dialog.gain_scale_label.text() == "Headroom check: passed."
    finally:
        dialog.reject()
        owner.close()


def test_a_correction_over_the_rule_that_was_made_gentler_does_not_say_there_was_room(
    qapp, monkeypatch
):
    owner = _owner_with()
    dialog = CalibrationDialog(owner)
    try:
        result = _over_the_rule(owner.preset.eq, {"gain_scale": 0.55})
        result.update(
            recommendation_status="reduced",
            recommendation_reasons=["headroom validation reduced the fitted correction"],
        )
        _complete(dialog, monkeypatch, result)

        text = dialog.warning_label.text()
        assert "leave enough headroom" not in text
        assert text.startswith(
            "Conservative correction ready. "
            "The correction was made gentler so it leaves no less room than your current EQ. "
        )
        assert text.endswith(
            "Your current settings already leave too little room for loud peaks. "
            "Lower the microphone gain by about 2 dB (in Windows or on the microphone) "
            "to allow the full correction."
        )
    finally:
        dialog.reject()
        owner.close()


def test_a_correction_left_out_for_leaving_less_room_says_so_and_offers_nothing(qapp, monkeypatch):
    owner = _owner_with()
    dialog = CalibrationDialog(owner)
    try:
        _complete(dialog, monkeypatch, _left_out(owner.preset.eq))

        assert dialog._candidate_proposal is None
        assert dialog.eq_settings is None
        assert dialog.compare_button.isEnabled() is False
        assert dialog.start_button.text() == "Record Again"
        assert dialog.warning_label.text() == (
            "No correction applied. "
            "The correction would leave less room than your current EQ does. "
            "Lower the microphone gain by about 2 dB (in Windows or on the microphone), "
            "then record again."
        )
        assert "Full-chain headroom validation failed" not in dialog.warning_label.toolTip()
        assert dialog.gain_scale_label.text() == (
            "Headroom check: the correction would leave less room than your current EQ."
        )
        # The figures shown are the ones that ruled the full correction out.
        assert "-0.3 dB" in dialog.gain_scale_label.toolTip()
    finally:
        dialog.reject()
        owner.close()


def test_a_correction_left_out_is_never_offered_whatever_else_the_result_says(qapp, monkeypatch):
    owner = _owner_with()
    dialog = CalibrationDialog(owner)
    try:
        result = _left_out(owner.preset.eq)
        result.update(
            apply_recommended=True,
            recommendation_status="apply",
            validated_candidate_eq=_candidate(1.0, owner.preset.eq)["validated_candidate_eq"],
        )
        _complete(dialog, monkeypatch, result)

        assert dialog._candidate_proposal is None
        assert dialog.eq_settings is None
        assert dialog.start_button.text() == "Record Again"
    finally:
        dialog.reject()
        owner.close()


def test_only_an_authoritative_not_worsened_result_opens_the_gate(qapp, monkeypatch):
    for headroom in (
        {"authoritative": False, "advisory": True, "status": "advisory"},
        {"authoritative": False, "advisory": True},
        {"status": "risk"},
    ):
        owner = _Owner()
        dialog = CalibrationDialog(owner)
        try:
            _capture(dialog, monkeypatch, "broadcast")
            dialog._on_analysis_complete(
                _over_the_rule(owner.preset.eq, headroom),
                generation=dialog._analysis_generation,
            )

            assert dialog._candidate_proposal is None
            assert dialog.eq_settings is None
            assert dialog.start_button.text() == "Record Again"
            assert "Full-chain headroom validation failed" in dialog.warning_label.toolTip()
        finally:
            dialog.reject()
            owner.close()


def _record(dialog: CalibrationDialog, monkeypatch, take: np.ndarray) -> None:
    monkeypatch.setattr("mic_eq.ui.calibration_dialog.AnalysisWorker", _AnalysisWorkerStub)
    dialog.recording_state = "recording"
    dialog._on_recording_complete(take)


def _clipped() -> np.ndarray:
    take = np.full(480, 0.1, dtype=np.float32)
    take[:8] = 1.0
    return take


@pytest.mark.parametrize(
    ("take", "message"),
    [
        (
            _clipped(),
            "The recording clipped: its peaks were held at full scale. Lower the "
            "microphone level by about 6 dB (in Windows or on the microphone) or move back a little, "
            "then record again.",
        ),
        (
            np.full(480, np.nan, dtype=np.float32),
            "The recording held invalid audio data, which usually means the audio "
            "driver glitched. Restart processing and record again.",
        ),
        (
            np.zeros(480, dtype=np.float32),
            "No sound came from the microphone. Check that it is not muted and "
            "that the right input is selected, then record again.",
        ),
    ],
)
def test_a_clipped_unreadable_or_silent_take_is_refused_before_any_analysis(
    qapp, monkeypatch, take, message
):
    owner = _Owner()
    dialog = CalibrationDialog(owner)
    try:
        _record(dialog, monkeypatch, take)

        assert dialog.warning_label.text() == message
        assert dialog.recording_state == "idle"
        assert dialog.analysis_worker is None
        assert dialog.audio_data is None and dialog.preview_audio_data is None
        assert dialog.start_button.isEnabled() is True
    finally:
        dialog.reject()
        owner.close()


class _ModeOwner(_Owner):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.applied_mode: str | None = None

    def _processing_mode(self) -> str:
        return self.mode

    def apply_processing_configuration(self, preset, *, processing_mode=None, **kwargs) -> None:
        self.applied_mode = processing_mode
        super().apply_processing_configuration(preset, processing_mode=processing_mode, **kwargs)


@pytest.mark.parametrize("mode", ["raw", "bypass"])
def test_headroom_checks_the_normal_path_where_the_eq_will_act(qapp, monkeypatch, mode):
    owner = _ModeOwner(mode)
    dialog = CalibrationDialog(owner)
    try:
        _capture(dialog, monkeypatch, "broadcast")
        assert dialog._analysis_headroom_chain is not None
        assert dialog._analysis_headroom_chain["processing_mode"] == "normal"

        dialog._on_analysis_complete(
            _candidate(1.0, owner.preset.eq), generation=dialog._analysis_generation
        )
        assert dialog.eq_settings is not None
        assert dialog._candidate_identity_error(dialog.eq_settings, owner) is None

        dialog._on_start_clicked()

        # Applying leaves the mode the user is in; the EQ acts once they leave it.
        assert owner.applied_mode == mode
    finally:
        if not dialog._close_requested:
            dialog.reject()
        owner.close()


def test_the_abstention_tooltip_does_not_outlive_its_result(qapp, monkeypatch):
    owner = _Owner()
    dialog = CalibrationDialog(owner)
    try:
        _complete(dialog, monkeypatch, _no_correction(_over_the_rule(owner.preset.eq)))
        assert dialog.warning_label.toolTip() == "headroom validation removed the usable correction"

        _capture(dialog, monkeypatch, "broadcast")
        dialog._on_analysis_failed("B failed", generation=dialog._analysis_generation)
        assert dialog.warning_label.toolTip() == ""

        _complete(dialog, monkeypatch, _no_correction(_over_the_rule(owner.preset.eq)))
        assert dialog.warning_label.toolTip() != ""
        monkeypatch.setattr(
            QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.Yes
        )
        dialog._on_retake_clicked()
        assert dialog.warning_label.toolTip() == ""
    finally:
        dialog.reject()
        owner.close()


def test_progress_is_a_busy_bar_while_the_analysis_runs(qapp, monkeypatch):
    owner = _Owner()
    dialog = CalibrationDialog(owner)
    try:
        _capture(dialog, monkeypatch, "broadcast")
        generation = dialog._analysis_generation
        assert (dialog.progress_bar.minimum(), dialog.progress_bar.maximum()) == (0, 0)

        dialog._on_analysis_step("fitting the correction and checking headroom...", 10, generation)
        assert dialog.progress_bar.maximum() == 0

        dialog._on_analysis_complete(_candidate(1.0, owner.preset.eq), generation=generation)
        assert (dialog.progress_bar.maximum(), dialog.progress_bar.value()) == (100, 100)

        _capture(dialog, monkeypatch, "broadcast")
        assert dialog.progress_bar.maximum() == 0
        dialog._on_analysis_failed("B failed", generation=dialog._analysis_generation)
        assert dialog.progress_bar.maximum() == 100
    finally:
        dialog.reject()
        owner.close()


def test_the_tone_modes_are_named_for_what_they_do_and_keep_their_stored_values(
    qapp, monkeypatch
):
    owner = _Owner()
    dialog = CalibrationDialog(owner)
    combo = dialog.target_mode_combo
    try:
        assert [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())] == [
            ("Subtle", "adaptive"),
            ("Full strength", "static"),
        ]
        assert combo.toolTip() == (
            "Subtle aims for small tone changes, never more than 2 dB from flat. "
            "Full strength aims for the tone preset's whole curve, up to 5 dB. "
            "Your recording decides whether a correction is applied and how far "
            "headroom lets it go."
        )
        assert dialog.get_selected_target_mode() == "adaptive"

        for label, stored in (("Subtle", "adaptive"), ("Full strength", "static")):
            combo.setCurrentIndex(combo.findText(label))
            _capture(dialog, monkeypatch, "broadcast")

            # Exactly what was stored before the names changed.
            assert dialog._candidate_target_metadata == ("broadcast", stored, "conservative")
            assert dialog._candidate_metadata is not None
            assert dialog._candidate_metadata["target"] == {
                "curve": "broadcast", "mode": stored, "smoothing": "conservative",
            }
            assert dialog._candidate_metadata["options"] == {
                "target_mode": stored, "smoothing_strength": "conservative",
            }
            assert dialog.analysis_worker is not None
            assert dialog.analysis_worker.target_mode == stored  # type: ignore[attr-defined]
            dialog._reset_recording_ui()
    finally:
        dialog.reject()
        owner.close()


def test_no_text_in_the_dialog_claims_to_adapt_to_the_voice(qapp):
    from PySide6.QtWidgets import QComboBox, QLabel

    owner = _Owner()
    dialog = CalibrationDialog(owner)
    try:
        texts = []
        for label in dialog.findChildren(QLabel):
            texts += [label.text(), label.toolTip()]
        for combo in dialog.findChildren(QComboBox):
            texts += [combo.toolTip()] + [combo.itemText(i) for i in range(combo.count())]
        shown = " ".join(texts)

        for phrase in ("Adapt to my voice", "Exact tone preset", "suit your voice", "Adapting"):
            assert phrase not in shown
    finally:
        dialog.reject()
        owner.close()


def test_the_quiet_verdict_waits_out_the_pauses_between_words(qapp):
    owner = _Owner()
    dialog = CalibrationDialog(owner)
    try:
        for rms_db in [-25.0] * 3 + [-70.0] * 14:
            dialog._on_level_update(rms_db, -12.0)
        assert dialog.warning_label.text().startswith("The level is good")

        dialog._on_level_update(-70.0, -12.0)  # the last loud block has left the window
        assert dialog.warning_label.text().startswith(
            "Little or no speech was heard: the loudest words were -70 dB"
        )

        dialog._on_level_update(-25.0, -12.0)
        assert dialog.warning_label.text().startswith("The level is good")
    finally:
        dialog.reject()
        owner.close()


def test_the_level_verdict_waits_for_a_full_window_but_loudness_does_not(qapp):
    owner = _Owner()
    dialog = CalibrationDialog(owner)
    try:
        dialog.warning_label.setText("Recording...")
        for _ in range(14):  # a speaker who has not started yet
            dialog._on_level_update(-70.0, -60.0)
        assert dialog.warning_label.text() == "Recording..."

        dialog._on_level_update(-70.0, -60.0)  # the 15th block makes a window
        assert dialog.warning_label.text().startswith("Your voice is quiet: -70 dB")

        dialog._recent_rms_db.clear()
        dialog._on_level_update(-20.0, -1.0)  # too loud is said at once
        assert dialog.warning_label.text().startswith(
            "Your voice peaks at -1.0 dB, close to the limit: lower the microphone level by about 5 dB "
            "(in Windows or on the microphone)"
        )
    finally:
        dialog.reject()
        owner.close()


def test_a_new_take_starts_with_an_empty_level_window(qapp, monkeypatch):
    class _Session:
        def __init__(self, *_args) -> None:
            pass

        def start(self, *_args, **_kwargs) -> None:
            pass

        def cleanup(self) -> None:
            pass

    monkeypatch.setattr("mic_eq.ui.calibration_dialog.CaptureSession", _Session)
    owner = _Owner()
    dialog = CalibrationDialog(owner)
    try:
        dialog._recent_rms_db.extend([-20.0] * 15)  # the last take's levels
        dialog._recent_peak_db.extend([-6.0] * 15)
        dialog.recording_state = "recording"

        dialog._begin_recording_capture()
        dialog.recording_timer.stop()

        assert len(dialog._recent_rms_db) == 0
        assert len(dialog._recent_peak_db) == 0
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


class _EqState:
    def __init__(self) -> None:
        self.restore_error = False
        self.apply_calls = 0
        self.typed_settings = EQSettings(
            enabled=False,
            band_freqs=list(EQ_FREQUENCIES),
            band_gains=[0.0] * 10,
            band_qs=[1.41] * 10,
        )
        self.state = {
            "enabled": False,
            "band_freqs": list(EQ_FREQUENCIES),
            "band_gains": [0.0] * 10,
            "band_qs": [1.41] * 10,
        }

    def get_settings(self) -> dict:
        return deepcopy(self.state)

    def get_eq_settings(self) -> EQSettings:
        return self.typed_settings

    def set_eq_settings(self, settings: EQSettings) -> None:
        self.typed_settings = EQSettings.from_dict(settings.to_dict())

    def apply_auto_eq_results(self, _bands, diagnostics=None) -> None:
        self.apply_calls += 1
        self.state = {
            "enabled": True,
            "band_freqs": [100.0] * 10,
            "band_gains": [6.0] * 10,
            "band_qs": [1.0] * 10,
            "diagnostics": diagnostics,
        }
        raise RuntimeError("native apply failed after mutation")

    def set_settings(self, settings: dict) -> None:
        if self.restore_error:
            raise RuntimeError("restore unavailable")
        self.state.update(deepcopy(settings))
        self.state.pop("diagnostics", None)

    def set_auto_eq_diagnostics(self, _diagnostics) -> None:
        return None


class _CalibrationOwner(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.processor = Mock()
        self.processor.sample_rate.return_value = 48_000
        self.processor.get_input_cleanup_mode.return_value = "off"
        self.eq_state = _EqState()

    def _get_current_preset(self) -> Preset:
        return Preset(eq=self.eq_state.get_eq_settings())

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
        correction = preset.eq.correction_bands
        if correction is not None:
            self.eq_state.apply_auto_eq_results(
                [
                    (band.frequency_hz, band.gain_db, band.q)
                    for band in correction
                ]
            )
            self.eq_state.set_settings({"enabled": preset.eq.enabled})
            self.eq_state.set_eq_settings(preset.eq)
            return
        self.eq_state.set_settings(
            {
                "enabled": preset.eq.enabled,
                "band_freqs": [band.frequency_hz for band in preset.eq.bands],
                "band_gains": [band.gain_db for band in preset.eq.bands],
                "band_qs": [band.q for band in preset.eq.bands],
            }
        )
        self.eq_state.set_eq_settings(preset.eq)


class _ContextCalibrationOwner(_CalibrationOwner):
    def __init__(self) -> None:
        super().__init__()
        self.context_key = "route-a"

    def _calibration_context_key(self) -> str:
        return self.context_key


class _ReadbackEqState(_EqState):
    def apply_auto_eq_results(self, _bands, diagnostics=None) -> None:
        self.apply_calls += 1
        self.state = {
            "enabled": True,
            "band_freqs": [100.0] * 10,
            "band_gains": [6.0] * 10,
            "band_qs": [1.0] * 10,
            "diagnostics": diagnostics,
        }

    def set_settings(self, settings: dict) -> None:
        self.state.update(deepcopy(settings))


def _eq_candidate_metadata(context_key: str | None = None) -> dict:
    return {
        "scope": "eq_only",
        "allowed_scope": ["eq"],
        "target": {
            "curve": "broadcast",
            "mode": "adaptive",
            "smoothing": "conservative",
        },
        "capture_identity": {
            "generation": 1,
            "sample_rate": 48_000,
            "sample_count": 16,
            "context_key": context_key,
        },
        "options": {
            "target_mode": "adaptive",
            "smoothing_strength": "conservative",
        },
        "verified_stages": [],
    }


def _eq_candidate(
    *,
    band_freqs: list[float] | None = None,
    band_gains: list[float] | None = None,
    band_qs: list[float] | None = None,
) -> dict:
    return {
        "band_freqs": (
            [80.0, 160.0, 320.0, 640.0, 1280.0, 2500.0, 5000.0, 8000.0, 12000.0, 16000.0]
            if band_freqs is None
            else band_freqs
        ),
        "band_gains": [1.0] * 10 if band_gains is None else band_gains,
        "band_qs": [1.41] * 10 if band_qs is None else band_qs,
        "apply_recommended": True,
    }


def _install_eq_candidate(
    dialog: CalibrationDialog,
    owner: _CalibrationOwner,
    *,
    band_freqs: list[float] | None = None,
    band_gains: list[float] | None = None,
    band_qs: list[float] | None = None,
) -> None:
    candidate = _eq_candidate(
        band_freqs=band_freqs,
        band_gains=band_gains,
        band_qs=band_qs,
    )
    incumbent = owner._get_current_preset()
    candidate_eq = build_eq_candidate_settings(
        incumbent.eq,
        candidate["band_freqs"],
        candidate["band_gains"],
        candidate["band_qs"],
        layer="correction",
        enabled=True,
    )
    chain = chain_settings(
        owner,
        full_chain=True,
        input_pre_filtered=False,
        preset=incumbent,
    )
    proposal = AutoEqCandidate.create(incumbent, candidate_eq, chain)
    context_getter = getattr(owner, "_calibration_context_key", None)
    context_value = context_getter() if callable(context_getter) else None
    context_key = context_value if isinstance(context_value, str) else None
    metadata = _eq_candidate_metadata(context_key)
    metadata["incumbent_signature"] = proposal.incumbent_signature
    metadata["configuration_signature"] = proposal.proposed_signature
    candidate["validated_candidate_eq"] = candidate_eq.to_dict()
    candidate["headroom_validation"] = {"safe": True, "authoritative": True}
    candidate["_candidate"] = deepcopy(metadata)

    dialog.recording_state = "ready"
    dialog.audio_data = np.zeros(16, dtype=np.float32)
    dialog._analysis_generation = 1
    dialog.eq_settings = candidate
    dialog._candidate_target_metadata = ("broadcast", "adaptive", "conservative")
    dialog._candidate_metadata = deepcopy(metadata)
    dialog._candidate_proposal = proposal


def test_eq_apply_failure_restores_partial_native_mutation(qapp, monkeypatch) -> None:
    owner = _CalibrationOwner()
    dialog = CalibrationDialog(owner)
    before = owner.eq_state.get_settings()
    try:
        _install_eq_candidate(dialog, owner)
        monkeypatch.setattr(QMessageBox, "critical", lambda *_args, **_kwargs: None)

        dialog._apply_eq_settings()

        assert owner.eq_state.get_settings() == before
        assert dialog.recording_state == "ready"
    finally:
        dialog.reject()
        owner.close()


def test_eq_apply_requires_snapshot_and_reports_restore_failure(qapp, monkeypatch) -> None:
    owner = _CalibrationOwner()
    dialog = CalibrationDialog(owner)
    messages: list[str] = []
    try:
        _install_eq_candidate(dialog, owner)
        monkeypatch.setattr(
            QMessageBox,
            "critical",
            lambda _parent, _title, message: messages.append(str(message)),
        )

        get_current_preset = owner._get_current_preset
        snapshot_calls = 0

        def fail_apply_snapshot() -> Preset:
            nonlocal snapshot_calls
            snapshot_calls += 1
            if snapshot_calls == 2:
                raise RuntimeError("snapshot unavailable")
            return get_current_preset()

        monkeypatch.setattr(owner, "_get_current_preset", fail_apply_snapshot)
        dialog._apply_eq_settings()
        assert owner.eq_state.apply_calls == 0
        assert "nothing was changed" in messages[-1]

        monkeypatch.setattr(owner, "_get_current_preset", get_current_preset)
        owner.eq_state.restore_error = True
        dialog._apply_eq_settings()
        assert "could not be fully put back (restore unavailable)" in messages[-1]
    finally:
        dialog.reject()
        owner.close()


def test_eq_apply_reads_back_accepted_values(qapp, monkeypatch) -> None:
    owner = _CalibrationOwner()
    owner.eq_state = _ReadbackEqState()
    dialog = CalibrationDialog(owner)
    monkeypatch.setattr(QMessageBox, "critical", lambda *_args, **_kwargs: None)
    try:
        _install_eq_candidate(
            dialog,
            owner,
            band_freqs=[100.0] * 10,
            band_gains=[6.0] * 10,
            band_qs=[1.0] * 10,
        )
        proposal = dialog._candidate_proposal
        assert proposal is not None
        expected_eq = proposal.proposed_preset.eq.to_dict()
        dialog._apply_eq_settings()
        assert owner.eq_state.get_settings()["band_gains"] == [6.0] * 10
        assert owner.eq_state.get_settings()["enabled"] is True
        assert owner.eq_state.get_eq_settings().to_dict() == expected_eq
    finally:
        dialog.reject()
        owner.close()


def test_eq_candidate_rejects_changed_route_context(qapp, monkeypatch) -> None:
    owner = _ContextCalibrationOwner()
    dialog = CalibrationDialog(owner)
    messages: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, message: messages.append(str(message)),
    )
    try:
        _install_eq_candidate(dialog, owner)
        owner.context_key = "route-b"
        dialog._apply_eq_settings()
        assert owner.eq_state.apply_calls == 0
        assert messages[-1] == (
            "The microphone or its settings changed after you recorded. "
            "Record again so the settings match your microphone. Nothing was changed."
        )
    finally:
        dialog.reject()
        owner.close()
