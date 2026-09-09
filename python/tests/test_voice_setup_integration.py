"""Small end-to-end checks for accepted voice setup state."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from PyQt6.QtTest import QSignalSpy

from mic_eq.analysis.voice_setup import analyze_voice_setup
from mic_eq.config import load_preset
from mic_eq.ui.main_window import MainWindow
from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog


@pytest.fixture
def real_main_window(qapp, monkeypatch, tmp_path):
    from mic_eq.config_parts import app_config, shared
    from mic_eq.ui import main_window

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    monkeypatch.setattr(main_window, "list_presets", lambda: [])
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [])
    assert main_window.load_config is app_config.load_config
    assert main_window.save_config is app_config.save_config
    window = MainWindow()
    window.meter_timer.stop()
    window.diagnostics_timer.stop()
    try:
        yield window
    finally:
        try:
            window.processor.stop()
        except Exception:
            pass
        window.close()
        window.deleteLater()
        qapp.processEvents()


def _voice_setup_result(
    window: MainWindow,
    dialog: VoiceSetupDialog,
    *,
    target_curve: str = "broadcast",
    target_lufs: float = -19.0,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dialog.curve_combo.setCurrentIndex(dialog.curve_combo.findData(target_curve))
    dialog.target_lufs_spin.setValue(target_lufs)
    eq_settings = deepcopy(
        (analysis or {}).get("eq_settings") or window.eq_panel.get_settings()
    )
    if analysis is None:
        eq_settings["band_gains"] = [
            3.0,
            2.0,
            1.0,
            0.5,
            0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -2.5,
        ]
    compressor_settings = deepcopy(
        (analysis or {}).get("compressor_settings")
        or window.compressor_panel.get_compressor_settings(include_calibration=True)
    )
    compressor_settings["target_lufs"] = target_lufs
    noise_audio = dialog.noise_audio
    voice_audio = dialog.voice_audio
    assert noise_audio is not None and voice_audio is not None
    assert dialog._analysis_generation > 0
    candidate = {
        "scope": "full_voice_setup",
        "allowed_scope": ["eq", "gate", "deesser", "compressor", "limiter"],
        "target": {
            "curve": target_curve,
            "target_lufs": target_lufs,
            "dynamics_intensity": dialog.dynamics_combo.currentData(),
        },
        "capture_identity": {
            "generation": dialog._analysis_generation,
            "sample_rate": int(window.processor.sample_rate()),
            "noise_samples": int(noise_audio.size),
            "voice_samples": int(voice_audio.size),
            "context_key": window._calibration_context_key(),
        },
        "options": {
            "vad_available": False,
            "dynamics_intensity": dialog.dynamics_combo.currentData(),
            "custom_target_p95_db": dialog.custom_p95_spin.value(),
            "custom_peak_cap_db": dialog.custom_peak_spin.value(),
        },
        "verified_stages": [],
    }
    dialog._candidate_metadata = deepcopy(candidate)
    return {
        "_candidate": candidate,
        "diagnostics": {
            **deepcopy((analysis or {}).get("diagnostics") or {}),
            "apply_recommended": True,
            "setup_confidence": 0.92,
            "capture_confidence": 0.9,
            "recommendation_uncertainty": 0.08,
            "gate_mode_label": "Threshold Gate",
            "compressor_calibration": {},
        },
        "gate_settings": deepcopy(
            (analysis or {}).get("gate_settings") or window.gate_panel.get_settings()
        ),
        "deesser_settings": deepcopy(
            (analysis or {}).get("deesser_settings")
            or window.deesser_panel.get_settings()
        ),
        "compressor_settings": compressor_settings,
        "limiter_settings": deepcopy(
            (analysis or {}).get("limiter_settings")
            or window.compressor_panel.get_limiter_settings()
        ),
        "eq_settings": eq_settings,
    }


def _bind_capture_context(
    dialog: VoiceSetupDialog, noise: np.ndarray, speech: np.ndarray
) -> None:
    """Give the dialog the same capture identity the production flow records."""
    dialog.noise_audio = noise
    dialog.voice_audio = speech
    dialog._analysis_generation = 1


def _short_voice_captures(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    noise_rng = np.random.default_rng(1200)
    noise = (0.0012 * noise_rng.normal(size=int(sample_rate * 1.6))).astype(
        np.float32
    )
    t = np.arange(int(sample_rate * 3.1), dtype=np.float64) / sample_rate
    voice_rng = np.random.default_rng(2401)
    gate = ((np.floor(t * 3.0) % 4.0) != 3.0).astype(np.float64)
    envelope = gate * (0.55 + 0.25 * np.sin(2.0 * np.pi * 2.1 * t) ** 2)
    voiced = (
        0.11 * np.sin(2.0 * np.pi * 140.0 * t)
        + 0.07 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.05 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.004 * np.sin(2.0 * np.pi * 6500.0 * t)
        + 0.30
    )
    speech = (
        envelope * voiced + 0.0025 * voice_rng.normal(size=t.size)
    ).astype(np.float32)
    return noise, speech


def _apply_complete_candidate(
    window: MainWindow,
    dialog: VoiceSetupDialog,
    result: dict[str, Any],
) -> None:
    dialog._on_analysis_complete(result)
    dialog._apply_setup()
    assert dialog.setup_state == "verification_ready"


def test_accepted_full_voice_setup_is_one_undo_transaction(
    real_main_window, monkeypatch
) -> None:
    window = real_main_window
    before = window._get_current_preset().to_dict()
    history_size = window._configuration_history.size
    dialog = VoiceSetupDialog(parent=window)
    noise, speech = _short_voice_captures(int(window.processor.sample_rate()))
    _bind_capture_context(dialog, noise, speech)
    result = _voice_setup_result(window, dialog)
    dialog.setup_applied.connect(window.on_voice_setup_applied)
    monkeypatch.setattr(
        "mic_eq.ui.voice_setup_dialog.QMessageBox.information",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(window, "_prompt_save_current_preset", lambda **_kwargs: None)

    window._history_transaction_depth += 1
    try:
        _apply_complete_candidate(window, dialog, result)
        monkeypatch.setattr(
            'mic_eq.ui.voice_setup_dialog.validate_voice_setup_verification',
            lambda *_args, **_kwargs: {'decision': 'accept'},
        )
        accepted = QSignalSpy(dialog.setup_applied)
        dialog._complete_verification(speech)
        assert accepted.wait(5000), dialog.warning_label.text()
        assert len(accepted) == 1
    finally:
        window._history_transaction_depth -= 1

    try:
        assert window._configuration_history.size == history_size + 1
        assert window._get_current_preset().to_dict() != before

        window.undo_configuration()

        assert window._get_current_preset().to_dict() == before
        assert window._configuration_history.cursor == 0
    finally:
        dialog.deleteLater()


def test_voice_setup_rejects_a_different_route_before_applying(
    real_main_window, monkeypatch
) -> None:
    window = real_main_window
    monkeypatch.setattr(window, "_calibration_context_key", lambda: "route-a")
    dialog = VoiceSetupDialog(parent=window)
    _bind_capture_context(dialog, *_short_voice_captures(int(window.processor.sample_rate())))
    result = _voice_setup_result(window, dialog)
    dialog._on_analysis_complete(result)
    before = window._get_current_preset().to_dict()
    messages = []
    monkeypatch.setattr(
        "mic_eq.ui.voice_setup_dialog.QMessageBox.critical",
        lambda _parent, _title, message: messages.append(message),
    )
    monkeypatch.setattr(window, "_calibration_context_key", lambda: "route-b")
    try:
        dialog._apply_setup()
        assert window._get_current_preset().to_dict() == before
        assert "context changed" in messages[-1]
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_voice_setup_tone_and_lufs_survive_apply_save_and_restart(real_main_window) -> None:
    window = real_main_window
    dialog = VoiceSetupDialog(parent=window)
    noise, speech = _short_voice_captures(int(window.processor.sample_rate()))
    _bind_capture_context(dialog, noise, speech)
    analysis = analyze_voice_setup(
        noise,
        speech,
        int(window.processor.sample_rate()),
        "broadcast",
        vad_available=False,
        target_lufs=-19.0,
        limiter_settings=window.compressor_panel.get_limiter_settings(),
    )
    assert analysis["eq_settings"] is not None
    result = _voice_setup_result(window, dialog, analysis=analysis)
    expected_gains = list(result["eq_settings"]["band_gains"])
    _apply_complete_candidate(window, dialog, result)

    candidate_target = result["_candidate"]["target"]
    assert candidate_target["curve"] == "broadcast"
    assert candidate_target["target_lufs"] == -19.0
    assert window.compressor_panel.get_compressor_settings()["target_lufs"] == -19.0

    preset = window._get_current_preset()
    preset.name = "Broadcast Voice"
    preset.description = "Auto-generated voice chain for the Broadcast tone target"
    filepath = window._save_preset_file(preset)
    assert filepath is not None and filepath.is_file()
    assert window._last_preset_identity_persisted
    loaded = load_preset(filepath)
    assert loaded.description == preset.description
    assert loaded.compressor.target_lufs == pytest.approx(-19.0)

    reloaded = MainWindow()
    reloaded.meter_timer.stop()
    reloaded.diagnostics_timer.stop()
    try:
        restored = reloaded._get_current_preset()
        assert reloaded.current_preset_path == filepath
        assert reloaded.current_preset_name == "Broadcast Voice"
        assert restored.compressor.target_lufs == pytest.approx(-19.0)
        assert restored.eq.band_gains == pytest.approx(expected_gains)
    finally:
        try:
            reloaded.processor.stop()
        except Exception:
            pass
        reloaded.close()
        reloaded.deleteLater()
        dialog.deleteLater()
