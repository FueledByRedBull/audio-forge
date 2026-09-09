"""Focused PR63 calibration lifecycle and tone-state checks."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import Mock

import numpy as np
from PyQt6.QtWidgets import QMessageBox, QWidget

from mic_eq.analysis.voice_setup import _recommend_compressor_settings
from mic_eq.ui.calibration_dialog import CalibrationDialog, _set_temporary_mute
from mic_eq.ui.compressor_panel import CompressorPanel
from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog


class _MuteOwner:
    user_muted = True

    def __init__(self) -> None:
        self.calls: list[tuple[bool, str]] = []

    def set_temporary_output_mute(self, muted: bool, reason: str) -> None:
        self.calls.append((muted, reason))


class _ProcessorOnlyOwner:
    user_muted = True

    def __init__(self) -> None:
        self.processor = Mock()


class _EqPanel:
    def __init__(self) -> None:
        self.snapshot_error = False
        self.restore_error = False
        self.apply_calls = 0
        self.state = {
            "enabled": False,
            "band_freqs": [80.0],
            "band_gains": [0.0],
            "band_qs": [1.41],
        }

    def get_settings(self) -> dict:
        if self.snapshot_error:
            raise RuntimeError("snapshot unavailable")
        return deepcopy(self.state)

    def apply_auto_eq_results(self, _bands, diagnostics=None) -> None:
        self.apply_calls += 1
        self.state = {
            "enabled": True,
            "band_freqs": [100.0],
            "band_gains": [6.0],
            "band_qs": [1.0],
            "diagnostics": diagnostics,
        }
        raise RuntimeError("native apply failed after mutation")

    def set_settings(self, settings: dict) -> None:
        if self.restore_error:
            raise RuntimeError("restore unavailable")
        self.state = deepcopy(settings)

    def set_auto_eq_diagnostics(self, _diagnostics) -> None:
        return None


class _CalibrationOwner(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.processor = Mock()
        self.processor.sample_rate.return_value = 48_000
        self.eq_panel = _EqPanel()


class _ContextCalibrationOwner(_CalibrationOwner):
    def __init__(self) -> None:
        super().__init__()
        self.context_key = "route-a"

    def _calibration_context_key(self) -> str:
        return self.context_key


class _ReadbackEqPanel(_EqPanel):
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


class _RestoreStage:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def set_settings(self, _settings) -> None:
        self.calls += 1
        if self.fail:
            raise RuntimeError("stage restore failed")

    def set_compressor_settings(self, _settings) -> None:
        self.set_settings(_settings)

    def set_limiter_settings(self, _settings) -> None:
        self.set_settings(_settings)


class _RestoreOwner:
    def __init__(self) -> None:
        self.gate_panel = _RestoreStage()
        self.deesser_panel = _RestoreStage()
        self.compressor_panel = _RestoreStage()
        self.eq_panel = _RestoreStage()


class _ContextRestoreStage(_RestoreStage):
    def __init__(self) -> None:
        super().__init__()
        self.last_settings: dict | None = None

    def set_compressor_settings(self, settings) -> None:
        self.last_settings = deepcopy(settings)
        super().set_compressor_settings(settings)


class _ContextRestoreOwner(_RestoreOwner):
    def __init__(self) -> None:
        super().__init__()
        self.context_key = "route-a"
        self.compressor_panel = _ContextRestoreStage()

    def _calibration_context_key(self) -> str:
        return self.context_key


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


def _eq_candidate() -> dict:
    return {
        "band_freqs": [80.0, 160.0, 320.0, 640.0, 1280.0, 2500.0, 5000.0, 8000.0, 12000.0, 16000.0],
        "band_gains": [1.0] * 10,
        "band_qs": [1.41] * 10,
        "apply_recommended": True,
        "_candidate": _eq_candidate_metadata(),
    }


def test_temporary_mute_keeps_owner_reason_and_fallback_user_mute() -> None:
    owner = _MuteOwner()
    _set_temporary_mute(owner, "auto_eq_calibration", True)
    _set_temporary_mute(owner, "auto_eq_calibration", False)
    assert owner.calls == [
        (True, "auto_eq_calibration"),
        (False, "auto_eq_calibration"),
    ]

    fallback_owner = _ProcessorOnlyOwner()
    _set_temporary_mute(fallback_owner, "auto_voice_setup", False)
    fallback_owner.processor.set_output_mute.assert_called_once_with(True)


def test_eq_apply_failure_restores_partial_native_mutation(qapp, monkeypatch) -> None:
    owner = _CalibrationOwner()
    dialog = CalibrationDialog(owner)
    before = owner.eq_panel.get_settings()
    try:
        dialog.recording_state = "ready"
        dialog.audio_data = np.zeros(16, dtype=np.float32)
        dialog._analysis_generation = 1
        dialog.eq_settings = _eq_candidate()
        dialog._candidate_target_metadata = ("broadcast", "adaptive", "conservative")
        dialog._candidate_metadata = _eq_candidate_metadata()
        monkeypatch.setattr(QMessageBox, "critical", lambda *_args, **_kwargs: None)

        dialog._apply_eq_settings()

        assert owner.eq_panel.get_settings() == before
        assert dialog.recording_state == "ready"
    finally:
        dialog.reject()
        owner.close()


def test_eq_apply_requires_snapshot_and_reports_restore_failure(qapp, monkeypatch) -> None:
    owner = _CalibrationOwner()
    dialog = CalibrationDialog(owner)
    messages: list[str] = []
    try:
        dialog.recording_state = "ready"
        dialog.audio_data = np.zeros(16, dtype=np.float32)
        dialog._analysis_generation = 1
        dialog.eq_settings = _eq_candidate()
        dialog._candidate_target_metadata = ("broadcast", "adaptive", "conservative")
        dialog._candidate_metadata = _eq_candidate_metadata()
        monkeypatch.setattr(
            QMessageBox,
            "critical",
            lambda _parent, _title, message: messages.append(str(message)),
        )

        owner.eq_panel.snapshot_error = True
        dialog._apply_eq_settings()
        assert owner.eq_panel.apply_calls == 0
        assert "no changes were applied" in messages[-1]

        owner.eq_panel.snapshot_error = False
        owner.eq_panel.restore_error = True
        dialog._apply_eq_settings()
        assert "EQ restoration failed" in messages[-1]
    finally:
        dialog.reject()
        owner.close()


def test_voice_rollback_attempts_all_stages_and_keeps_failed_snapshot(qapp, monkeypatch) -> None:
    owner = _RestoreOwner()
    dialog = VoiceSetupDialog()
    snapshot = {
        "gate": {},
        "deesser": {},
        "compressor": {},
        "limiter": {},
        "eq": {},
    }
    dialog._pre_setup_snapshot = snapshot
    owner.gate_panel.fail = True
    monkeypatch.setattr(
        "mic_eq.ui.voice_setup_dialog._find_eq_panel_owner",
        lambda _widget: owner,
    )
    try:
        assert dialog._restore_pre_setup_snapshot() is False
        assert dialog._pre_setup_snapshot is snapshot
        assert owner.gate_panel.calls == 1
        assert owner.deesser_panel.calls == 1
        assert owner.compressor_panel.calls == 2
        assert owner.eq_panel.calls == 1
        dialog._on_verification_failed("verification failed")
        assert "restore is incomplete" in dialog.phase_label.text()

        owner.gate_panel.fail = False
        assert dialog._restore_pre_setup_snapshot() is True
        assert dialog._pre_setup_snapshot is None
    finally:
        dialog.reject()


def test_explicit_lufs_target_is_independent_of_tone_curve() -> None:
    common = {
        "speech_body_db": -22.0,
        "speech_loudness_lufs": -20.0,
        "loudness_range_db": 5.0,
        "speech_snr_db": 20.0,
        "capture_confidence": 0.8,
        "dynamics_intensity": "balanced",
        "custom_target_p95_db": 3.5,
        "custom_peak_cap_db": 8.0,
        "target_lufs": -19.0,
    }
    broadcast, _ = _recommend_compressor_settings(
        target_preset="broadcast", **common
    )
    podcast, _ = _recommend_compressor_settings(target_preset="podcast", **common)
    assert broadcast["target_lufs"] == podcast["target_lufs"] == -19.0


def test_manual_compressor_change_is_marked_customized_after_calibration(qapp) -> None:
    panel = CompressorPanel(Mock())
    try:
        panel.set_compressor_settings(
            {"ratio": 3.0, "dynamics_profile": "gentle", "dynamics_customized": False}
        )
        panel.ratio_spinbox.setValue(3.5)
        panel._mark_dynamics_customized()

        settings = panel.get_compressor_settings(include_calibration=True)
        assert settings["dynamics_profile"] == "gentle"
        assert settings["dynamics_customized"] is True
        assert settings["dynamics_intensity"] == "customized"
    finally:
        panel.close()
        panel.deleteLater()
        qapp.processEvents()


def test_lufs_target_change_keeps_compression_intensity_marker(qapp) -> None:
    panel = CompressorPanel(Mock())
    try:
        panel.set_compressor_settings(
            {"dynamics_profile": "gentle", "dynamics_customized": False}
        )
        panel.target_lufs_spinbox.setValue(-17.0)
        panel._mark_dynamics_customized()
        settings = panel.get_compressor_settings(include_calibration=True)
        assert settings["target_lufs"] == -17.0
        assert settings["dynamics_profile"] == "gentle"
        assert settings["dynamics_customized"] is False
    finally:
        panel.close()
        panel.deleteLater()
        qapp.processEvents()


def test_eq_apply_reads_back_accepted_values(qapp, monkeypatch) -> None:
    owner = _CalibrationOwner()
    owner.eq_panel = _ReadbackEqPanel()
    dialog = CalibrationDialog(owner)
    monkeypatch.setattr(QMessageBox, "critical", lambda *_args, **_kwargs: None)
    try:
        dialog.recording_state = "ready"
        dialog.audio_data = np.zeros(16, dtype=np.float32)
        dialog._analysis_generation = 1
        dialog.eq_settings = _eq_candidate()
        dialog._candidate_target_metadata = ("broadcast", "adaptive", "conservative")
        dialog._candidate_metadata = _eq_candidate_metadata()
        dialog._apply_eq_settings()
        assert owner.eq_panel.get_settings()["band_gains"] == [6.0] * 10
        assert owner.eq_panel.get_settings()["enabled"] is True
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
        metadata = _eq_candidate_metadata("route-a")
        dialog.recording_state = "ready"
        dialog.audio_data = np.zeros(16, dtype=np.float32)
        dialog._analysis_generation = 1
        dialog.eq_settings = _eq_candidate()
        dialog.eq_settings["_candidate"] = deepcopy(metadata)
        dialog._candidate_target_metadata = ("broadcast", "adaptive", "conservative")
        dialog._candidate_metadata = metadata
        owner.context_key = "route-b"
        dialog._apply_eq_settings()
        assert owner.eq_panel.apply_calls == 0
        assert "context changed" in messages[-1]
    finally:
        dialog.reject()
        owner.close()


def test_rollback_drops_stale_noise_reliability_but_restores_numeric_settings(
    qapp, monkeypatch
) -> None:
    owner = _ContextRestoreOwner()
    dialog = VoiceSetupDialog()
    snapshot = {
        "gate": {},
        "deesser": {},
        "compressor": {"ratio": 2.5, "noise_reference_reliability": 0.8},
        "limiter": {},
        "eq": {},
        "calibration_context_key": "route-a",
    }
    dialog._pre_setup_snapshot = snapshot
    owner.context_key = "route-b"
    monkeypatch.setattr(
        "mic_eq.ui.voice_setup_dialog._find_eq_panel_owner", lambda _widget: owner
    )
    try:
        assert dialog._restore_pre_setup_snapshot() is True
        assert owner.compressor_panel.last_settings == {"ratio": 2.5}
    finally:
        dialog.reject()


def test_voice_close_and_reset_block_when_restore_owner_is_missing(qapp, monkeypatch) -> None:
    dialog = VoiceSetupDialog()
    worker = Mock()
    worker.isRunning.return_value = True
    dialog._analysis_workers.append(worker)
    dialog.setup_state = "verification_analyzing"
    generation = dialog._analysis_generation
    dialog._pre_setup_snapshot = {"gate": {}}
    monkeypatch.setattr(
        "mic_eq.ui.voice_setup_dialog._find_eq_panel_owner", lambda _widget: None
    )
    dialog.reject()
    worker.stop.assert_called_once()
    assert dialog._analysis_generation > generation
    assert not dialog._close_requested
    assert dialog._pre_setup_snapshot is not None
    assert "could not be fully restored" in dialog.warning_label.text()
    dialog._reset_setup_ui()
    assert dialog._pre_setup_snapshot is not None
    dialog._pre_setup_snapshot = None
    worker.isRunning.return_value = False
    dialog.reject()
    dialog.deleteLater()
    qapp.processEvents()


def test_voice_verification_marks_only_accepted_stages(qapp, monkeypatch) -> None:
    owner = _CalibrationOwner()
    dialog = VoiceSetupDialog(owner)
    candidate = {
        "scope": "full_voice_setup",
        "allowed_scope": ["eq", "gate", "deesser", "compressor", "limiter"],
        "target": {
            "curve": "broadcast",
            "target_lufs": -18.0,
            "dynamics_intensity": "balanced",
        },
        "capture_identity": {
            "generation": 1,
            "sample_rate": 48_000,
            "noise_samples": 16,
            "voice_samples": 16,
            "context_key": None,
        },
        "options": {
            "vad_available": False,
            "dynamics_intensity": "balanced",
            "custom_target_p95_db": 3.5,
            "custom_peak_cap_db": 8.0,
        },
        "verified_stages": [],
    }
    dialog.setup_result = {"_candidate": deepcopy(candidate)}
    dialog._candidate_metadata = deepcopy(candidate)
    dialog.noise_audio = dialog.voice_audio = np.zeros(16, dtype=np.float32)
    dialog._analysis_generation = 1
    monkeypatch.setattr(QMessageBox, "information", lambda *_args, **_kwargs: None)
    try:
        dialog._on_verification_complete({"decision": "retry"})
        assert dialog.setup_result["_candidate"]["verified_stages"] == []
        dialog._on_verification_complete({"decision": "accept"})
        assert dialog.setup_result["_candidate"]["verified_stages"] == [
            "eq",
            "deesser",
            "compressor",
            "limiter",
        ]
    finally:
        dialog.reject()
        owner.close()


def test_numeric_compressor_preset_recovers_exact_intensity_without_marker(qapp) -> None:
    common = {
        "speech_body_db": -22.0,
        "speech_loudness_lufs": -20.0,
        "loudness_range_db": 5.0,
        "speech_snr_db": 20.0,
        "capture_confidence": 0.8,
        "custom_target_p95_db": 3.5,
        "custom_peak_cap_db": 8.0,
        "target_lufs": -18.0,
    }
    for intensity in ("gentle", "balanced", "dense"):
        settings, _ = _recommend_compressor_settings(
            target_preset="broadcast", dynamics_intensity=intensity, **common
        )
        source = CompressorPanel(Mock())
        restored = CompressorPanel(Mock())
        try:
            source.set_compressor_settings(settings)
            saved = source.get_compressor_settings()
            restored.set_compressor_settings(saved)
            assert restored.get_compressor_settings(include_calibration=True)[
                "dynamics_profile"
            ] == intensity
        finally:
            source.close()
            restored.close()
            source.deleteLater()
            restored.deleteLater()
    custom = CompressorPanel(Mock())
    try:
        custom.set_compressor_settings(
            {
                "ratio": 4.0,
                "attack_ms": 10.0,
                "release_ms": 200.0,
                "base_release_ms": 50.0,
            }
        )
        assert custom.get_compressor_settings(include_calibration=True)[
            "dynamics_profile"
        ] == "custom"
    finally:
        custom.close()
        custom.deleteLater()
    qapp.processEvents()
