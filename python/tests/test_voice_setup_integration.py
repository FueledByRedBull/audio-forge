"""Small end-to-end checks for accepted voice setup state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PySide6.QtTest import QSignalSpy
from PySide6.QtCore import QElapsedTimer, QEventLoop, QThread, QTimer
from PySide6.QtWidgets import QDialog, QMessageBox

from mic_eq.analysis.voice_setup import analyze_voice_setup
from mic_eq.config import EQSettings, build_eq_candidate_settings, load_preset
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
    monkeypatch.setattr(window, "_calibration_context_key", lambda: "test-route/48000/mono")
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
    suppressor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dialog.curve_combo.setCurrentIndex(dialog.curve_combo.findData(target_curve))
    dialog.target_lufs_spin.setValue(target_lufs)
    eq_settings = deepcopy(
        (analysis or {}).get("eq_settings") or window.eq_state.get_settings()
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
        or window.compressor_state.get_settings(include_calibration=True)
    )
    compressor_settings["target_lufs"] = target_lufs
    noise_audio = dialog.noise_audio
    voice_audio = dialog.voice_audio
    assert noise_audio is not None and voice_audio is not None
    assert dialog._analysis_generation > 0
    candidate = {
        "scope": "full_voice_setup",
        "allowed_scope": [
            "eq", "gate", "deesser", "compressor", "limiter",
            *(["suppression"] if suppressor is not None else []),
        ],
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
            (analysis or {}).get("gate_settings") or window.gate_state.get_settings()
        ),
        "deesser_settings": deepcopy(
            (analysis or {}).get("deesser_settings")
            or window.deesser_state.get_settings()
        ),
        "compressor_settings": compressor_settings,
        "limiter_settings": deepcopy(
            (analysis or {}).get("limiter_settings")
            or window.limiter_state.get_settings()
        ),
        "eq_settings": eq_settings,
        **({"suppressor_settings": deepcopy(suppressor)} if suppressor is not None else {}),
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
    real_main_window, monkeypatch, qapp
) -> None:
    window = real_main_window
    before = window._get_current_preset().to_dict()
    history_size = window._configuration_history.size
    dialog = VoiceSetupDialog(parent=window)
    noise, speech = _short_voice_captures(int(window.processor.sample_rate()))
    _bind_capture_context(dialog, noise, speech)
    result = _voice_setup_result(window, dialog)
    dialog.setup_applied.connect(window.on_voice_setup_applied)
    information_messages: list[str] = []
    monkeypatch.setattr(
        "mic_eq.ui.voice_setup_dialog.QMessageBox.information",
        lambda *_args, **_kwargs: information_messages.append(str(_args[-1]))
        if _args
        else None,
    )
    monkeypatch.setattr(
        "mic_eq.ui.voice_setup_dialog.QMessageBox.warning",
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
        result_on_gui_thread = []
        original_complete = dialog._on_verification_complete

        def record_result_thread(*args):
            result_on_gui_thread.append(QThread.currentThread() == qapp.thread())
            original_complete(*args)

        monkeypatch.setattr(dialog, "_on_verification_complete", record_result_thread)
        dialog._complete_verification(speech)
        verification_worker = dialog.analysis_worker
        assert verification_worker is not None
        # Run the same Qt event loop as the application; QSignalSpy.wait can
        # block a Python QThread body in PySide rather than yield to it.
        loop = QEventLoop()
        deadline = QTimer(loop)
        deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit)
        verification_worker.finished.connect(loop.quit)
        if verification_worker.isRunning():
            deadline.start(30_000)
            loop.exec()
        deadline.stop()
        qapp.processEvents()
        assert result_on_gui_thread == [True]
        assert dialog.setup_state == "final_comparison_ready"

        class _AcceptedComparison:
            def __init__(self, **_kwargs):
                pass

            def exec(self):
                return int(QDialog.DialogCode.Accepted)

            def deleteLater(self):
                pass

        monkeypatch.setattr(
            "mic_eq.ui.listening_comparison_dialog.ListeningComparisonDialog",
            _AcceptedComparison,
        )
        dialog._on_start_clicked()
        assert accepted.count() == 1
        assert len(window.config.calibration_results) == 1
        accepted_message = information_messages[-1].casefold()
        assert "passed" in accepted_message
        assert "listening comparison" not in accepted_message
        assert "was heard" not in accepted_message
        assert window.config.calibration_results[0].verified_stages == (
            "input_cleanup",
            "gate",
            "suppression",
            "eq",
            "deesser",
            "compressor",
            "limiter",
        )
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
        assert messages[-1] == (
            "The microphone or its settings changed after you recorded. "
            "Record again so the settings match your microphone. Nothing was changed."
        )
    finally:
        dialog.reject()
        dialog.deleteLater()


@pytest.mark.parametrize("mode", ["raw", "bypass"])
def test_unprocessed_mode_verification_capture_allows_intended_mode_transition(
    real_main_window, monkeypatch, mode
) -> None:
    window = real_main_window
    monkeypatch.setattr(
        window,
        "_calibration_context_key",
        lambda: (
            '["test-route","phase_safe_mono","off",null,"'
            f'{window._processing_mode()}"]'
        ),
    )
    window._set_processing_mode(mode)
    window.compressor_state.set_settings({"noise_reference_reliability": 0.73})
    dialog = VoiceSetupDialog(parent=window)
    noise, speech = _short_voice_captures(int(window.processor.sample_rate()))
    _bind_capture_context(dialog, noise, speech)
    dialog._capture_context_key = window._calibration_context_key()
    _apply_complete_candidate(window, dialog, _voice_setup_result(window, dialog))
    assert window._processing_mode() == "normal"
    assert dialog._candidate_identity_error(window) is None

    monkeypatch.setattr(
        type(window.processor),
        "start_raw_recording",
        lambda *_args, **_kwargs: None,
    )
    dialog.setup_state = "verification_recording"
    try:
        dialog._begin_recording_capture()
        assert dialog.setup_state == "verification_recording"
        assert dialog.recording_timer.isActive()
    finally:
        dialog.recording_timer.stop()
        dialog._cleanup_recording_tap()
        assert dialog._restore_pre_setup_snapshot()
        assert window._processing_mode() == mode
        assert window.compressor_state.get_settings(include_calibration=True)[
            "noise_reference_reliability"
        ] == pytest.approx(0.73)
        dialog.reject()
        dialog.deleteLater()


def test_repeated_bad_verification_takes_stop_and_restore(real_main_window, monkeypatch, qapp):
    window = real_main_window
    before = window._get_current_preset().to_dict()
    dialog = VoiceSetupDialog(parent=window)
    noise, speech = _short_voice_captures(int(window.processor.sample_rate()))
    _bind_capture_context(dialog, noise, speech)
    _apply_complete_candidate(window, dialog, _voice_setup_result(window, dialog))
    monkeypatch.setattr(
        "mic_eq.ui.voice_setup_dialog.validate_voice_setup_verification",
        lambda *_args, **_kwargs: {
            "decision": "retry",
            "reasons": ["verification passage was clipped; lower microphone gain"],
        },
    )
    try:
        for attempt in range(3):
            dialog._complete_verification(speech)
            timer = QElapsedTimer()
            timer.start()
            while dialog.setup_state == "verification_analyzing" and timer.elapsed() < 5000:
                qapp.processEvents()
                time.sleep(0.01)
            assert dialog.setup_state != "verification_analyzing"
            if attempt < 2:
                assert dialog.setup_state == "verification_ready"
        assert dialog.setup_state == "completed"
        assert window._get_current_preset().to_dict() == before
        assert "clipped" in dialog.warning_label.text()
    finally:
        dialog.reject()
        dialog.deleteLater()


@pytest.mark.parametrize("sample", [np.nan, np.inf], ids=["nan", "infinite"])
def test_an_unreadable_verification_take_restores_and_offers_another_try(
    real_main_window, sample
) -> None:
    window = real_main_window
    before = window._get_current_preset().to_dict()
    dialog = VoiceSetupDialog(parent=window)
    noise, speech = _short_voice_captures(int(window.processor.sample_rate()))
    _bind_capture_context(dialog, noise, speech)
    _apply_complete_candidate(window, dialog, _voice_setup_result(window, dialog))
    assert window._get_current_preset().to_dict() != before
    take = speech.copy()
    take[100] = sample
    dialog.setup_state = "verification_recording"
    try:
        dialog._on_recording_complete(take)

        assert dialog.setup_state == "completed"
        assert window._get_current_preset().to_dict() == before
        assert dialog.warning_label.text() == (
            "The recording held invalid audio data, which usually means the audio "
            "driver glitched. Restart processing and record again."
        )
        assert dialog.start_button.text() == "Apply & Verify Again"
        assert dialog.start_button.isEnabled()
        assert dialog.preview_verification_audio is None
        assert dialog.phase_label.text() == "Second recording failed; previous settings put back"

        dialog._on_start_clicked()
        assert dialog.setup_state == "verification_ready"
    finally:
        dialog.reject()
        dialog.deleteLater()


DULLING_SENTENCE = (
    "noise suppression is dulling consonants by about 8 dB; "
    "lower its strength or use DeepFilterNet LL"
)
ADVICE = (
    "Noise suppression is dulling your S and T sounds by about 8 dB. "
    "Lower its strength or try DeepFilterNet LL."
)
METRICS = (
    "Tone: 0.8 dB from the preset before, 0.9 dB after. "
    "The compressor evened out your voice by 6.0 dB. The loudest peak is -120.0 dB. "
    "Your voice against the background noise changed by -5.0 dB."
)
OUT_OF_ADJUSTMENTS = (
    "Setup could not make the settings safe on your recording after three adjustments. "
    "Lower the microphone level and record again."
)
FIRST_REASON = {
    "accept": "repeatability and full-chain DSP constraints passed",
    "rollback": "SNR fell by 5.0 dB",
    "reduce": "compressor p95 reduction exceeded target by 2.5 dB",
    "retry": (
        "verification delivery differs too much from the setup passage "
        "(delivery level delta was 9.0 dB). Repeat the same passage at the same "
        "microphone distance and natural speaking level."
    ),
}


def _verdict_with_dulling_advisory(decision: str, *, advised: bool = True) -> dict[str, Any]:
    """The analysis's verdict; one for a proposal without a suppressor has no advice."""
    return {
        "decision": decision,
        "reason_codes": [f"{decision}_reason", *(["suppressor_dulling"] if advised else [])],
        "reduction_targets": ["compressor"] if decision == "reduce" else [],
        "reasons": [FIRST_REASON[decision]],
        **({"suppressor_dulling_advice": DULLING_SENTENCE} if advised else {}),
        "spectral_target_error_before_db": 0.8,
        "spectral_target_error_after_db": 0.9,
        "compressor_gain_reduction_p95_db": 6.0,
        "snr_change_db": -5.0,
    }


def _verifying_dialog(window: MainWindow) -> tuple[VoiceSetupDialog, np.ndarray]:
    """A dialog whose applied proposal awaits a verdict."""
    dialog = VoiceSetupDialog(parent=window)
    noise, speech = _short_voice_captures(int(window.processor.sample_rate()))
    _bind_capture_context(dialog, noise, speech)
    _apply_complete_candidate(window, dialog, _voice_setup_result(window, dialog))
    return dialog, speech


def _reduce_out_of_adjustments(dialog, speech, _monkeypatch) -> None:
    dialog._verification_audio = speech
    dialog._verification_adjustment_count = 3


def _reduce_without_progress(dialog, speech, _monkeypatch) -> None:
    dialog._verification_audio = speech
    dialog._verification_last_reduction_metrics = {"compressor": 6.0}


def _reduce_that_changes_nothing(dialog, speech, monkeypatch) -> None:
    dialog._verification_audio = speech
    monkeypatch.setattr(dialog, "_apply_verification_reduction", lambda *_args: (False, None))


def _last_take(dialog, _speech, _monkeypatch) -> None:
    dialog._verification_bad_capture_count = 2


# decision, setup of the dialog's state, the text just before any advice, whether it is advised
DULLING_MESSAGES = [
    pytest.param(
        "accept", None, "closing the comparison puts your old settings back.", True,
        id="accept",
    ),
    pytest.param("rollback", None, METRICS, True, id="rollback"),
    pytest.param(
        "reduce", _reduce_out_of_adjustments,
        OUT_OF_ADJUSTMENTS, True,
        id="reduce-out-of-adjustments",
    ),
    pytest.param(
        "reduce", _reduce_without_progress,
        "The last adjustment did not help. Lower the microphone level and record again.", True,
        id="reduce-without-progress",
    ),
    pytest.param(
        "reduce", _reduce_that_changes_nothing, "Setup could not adjust the settings any further. Lower the microphone level and record again.", True,
        id="reduce-that-changes-nothing",
    ),
    pytest.param("retry", None, METRICS, False, id="retry"),
    pytest.param(
        "retry", _last_take, "Setup stopped after three tries.", False,
        id="last-retry"
    ),
]


@pytest.mark.parametrize("has_advice", [True, False], ids=["advice-given", "no-advice"])
@pytest.mark.parametrize(("decision", "setup", "tail", "advised"), DULLING_MESSAGES)
def test_the_dulling_advice_ends_the_message_as_its_own_sentence_or_is_left_out(
    real_main_window, monkeypatch, decision, setup, tail, advised, has_advice
) -> None:
    dialog, speech = _verifying_dialog(real_main_window)
    try:
        if setup is not None:
            setup(dialog, speech, monkeypatch)
        dialog._on_verification_complete(_verdict_with_dulling_advisory(decision, advised=has_advice))
        text = dialog.warning_label.text()
    finally:
        dialog.reject()
        dialog.deleteLater()

    assert ".;" not in text
    if advised and has_advice:
        assert text.endswith(f"{tail} {ADVICE}")
        assert text.count(ADVICE) == 1
    else:
        assert text.endswith(tail)
        assert DULLING_SENTENCE not in text
        assert "dulling" not in text


@pytest.mark.parametrize("has_advice", [True, False], ids=["advice-given", "no-advice"])
def test_the_verified_box_carries_the_dulling_advice_past_the_comparison(
    real_main_window, monkeypatch, has_advice
) -> None:
    dialog, _speech = _verifying_dialog(real_main_window)
    boxes: list[str] = []
    monkeypatch.setattr(
        "mic_eq.ui.voice_setup_dialog.QMessageBox.information",
        lambda *args, **_kwargs: boxes.append(str(args[-1])),
    )
    monkeypatch.setattr(real_main_window, "_prompt_save_current_preset", lambda **_kwargs: None)

    class _AcceptedComparison:
        def __init__(self, **_kwargs):
            pass

        def exec(self):
            return int(QDialog.DialogCode.Accepted)

        def deleteLater(self):
            pass

    monkeypatch.setattr(
        "mic_eq.ui.listening_comparison_dialog.ListeningComparisonDialog", _AcceptedComparison
    )
    try:
        dialog._on_verification_complete(_verdict_with_dulling_advisory("accept", advised=has_advice))
        assert dialog.setup_state == "final_comparison_ready"
        dialog._on_start_clicked()
    finally:
        dialog.reject()
        dialog.deleteLater()

    assert len(boxes) == 1
    if has_advice:
        assert f"\n\n{ADVICE}" in boxes[0]
    else:
        assert "dulling" not in boxes[0]
        assert DULLING_SENTENCE not in boxes[0]


@pytest.mark.parametrize("target,auto_makeup", [("limiter", True), ("limiter", False), ("deesser", False)])
def test_stage_reductions_change_only_relevant_live_controls(real_main_window, target, auto_makeup):
    window = real_main_window
    dialog = VoiceSetupDialog(parent=window)
    _bind_capture_context(dialog, *_short_voice_captures(int(window.processor.sample_rate())))
    result = _voice_setup_result(window, dialog)
    result["compressor_settings"].update(auto_makeup_enabled=auto_makeup, makeup_gain_db=4.0)
    _apply_complete_candidate(window, dialog, result)
    before = window._get_current_preset().to_dict()
    try:
        changed, _ = dialog._apply_verification_reduction(window, [target])
        assert changed
        after = window._get_current_preset().to_dict()
        expected = deepcopy(before)
        if target == "deesser":
            assert after["deesser"]["auto_amount"] < before["deesser"]["auto_amount"]
            expected["deesser"]["auto_amount"] = after["deesser"]["auto_amount"]
        else:
            field = "target_lufs" if auto_makeup else "makeup_gain_db"
            assert after["compressor"][field] == pytest.approx(before["compressor"][field] - 1.0)
            expected["compressor"][field] = after["compressor"][field]
            if auto_makeup:
                assert dialog.setup_result is not None
                assert dialog.setup_result["verification_adjusted_target_lufs"] == after["compressor"][field]
        assert after == expected
    finally:
        dialog.reject()
        dialog.deleteLater()


@pytest.mark.parametrize("converges", [True, False])
def test_verification_adjustments_reuse_take_and_stop(real_main_window, monkeypatch, converges, qapp):
    window = real_main_window
    before = window._get_current_preset().to_dict()
    dialog = VoiceSetupDialog(parent=window)
    noise, speech = _short_voice_captures(int(window.processor.sample_rate()))
    _bind_capture_context(dialog, noise, speech)
    _apply_complete_candidate(window, dialog, _voice_setup_result(window, dialog))
    calls = []

    def verify(_noise, _original, take, _rate, candidate, *_args, **_kwargs):
        calls.append(deepcopy(candidate))
        np.testing.assert_array_equal(take, speech)
        accepted = converges and len(calls) > 1
        return {
            "decision": "accept" if accepted else "reduce",
            "reason_codes": [] if accepted else ["compressor_excess"],
            "reduction_targets": [] if accepted else ["compressor"],
            "reasons": ["passed" if accepted else "compressor p95 exceeds target"],
            "compressor_gain_reduction_p95_db": 1.0 if accepted else 6.0,
        }

    monkeypatch.setattr("mic_eq.ui.voice_setup_dialog.validate_voice_setup_verification", verify)
    monkeypatch.setattr(
        dialog, "_start_recording_phase",
        lambda *_args: pytest.fail("Valid verification audio must be reused"),
    )
    try:
        dialog._complete_verification(speech)
        timer = QElapsedTimer()
        timer.start()
        while dialog.setup_state == "verification_analyzing" and timer.elapsed() < 5000:
            qapp.processEvents()
            time.sleep(0.01)
        assert 2 <= len(calls) <= 4
        assert calls[1]["compressor_settings"]["threshold_db"] > calls[0]["compressor_settings"]["threshold_db"]
        assert calls[1]["deesser_settings"] == calls[0]["deesser_settings"]
        assert calls[1]["limiter_settings"] == calls[0]["limiter_settings"]
        if converges:
            assert dialog.setup_state == "final_comparison_ready"
            assert dialog._pre_setup_snapshot is not None
            dialog.reject()
        else:
            assert len(calls) == 2
            assert dialog.setup_state == "completed"
        assert window._get_current_preset().to_dict() == before
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
        limiter_settings=window.limiter_state.get_settings(),
    )
    assert analysis["eq_settings"] is not None
    result = _voice_setup_result(window, dialog, analysis=analysis)
    expected_gains = list(result["eq_settings"]["band_gains"])
    _apply_complete_candidate(window, dialog, result)

    candidate_target = result["_candidate"]["target"]
    assert candidate_target["curve"] == "broadcast"
    assert candidate_target["target_lufs"] == -19.0
    assert window.compressor_state.get_settings()["target_lufs"] == -19.0

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
        assert restored.eq.correction_bands is not None
        assert [band.gain_db for band in restored.eq.correction_bands] == pytest.approx(expected_gains)
        assert restored.eq.band_gains == [0.0] * 10
    finally:
        try:
            reloaded.processor.stop()
        except Exception:
            pass
        reloaded.close()
        reloaded.deleteLater()
        dialog.deleteLater()


def test_the_analysis_builds_the_typed_candidate_eq_from_what_the_dialog_sends(
    isolated_main_window, monkeypatch, qapp
) -> None:
    from mic_eq.ui import voice_setup_dialog

    window = isolated_main_window
    dialog = VoiceSetupDialog(parent=window)
    noise, speech = _short_voice_captures(int(window.processor.sample_rate()))
    _bind_capture_context(dialog, noise, speech)
    dialog.preview_noise_audio, dialog.preview_voice_audio = noise, speech
    sent: list[dict[str, Any]] = []
    analyze = voice_setup_dialog.analyze_voice_setup

    def recording_analysis(*args, **kwargs):
        sent.append(deepcopy(kwargs["incumbent_settings"]))
        return analyze(*args, **kwargs)

    monkeypatch.setattr(voice_setup_dialog, "analyze_voice_setup", recording_analysis)
    try:
        dialog._start_analysis()
        worker = dialog.analysis_worker
        assert worker is not None
        loop = QEventLoop()
        deadline = QTimer(loop)
        deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit)
        worker.finished.connect(loop.quit)
        if worker.isRunning():
            deadline.start(120_000)
            loop.exec()
        deadline.stop()
        qapp.processEvents()

        # The payload is the real EQState's: typed fields and the legacy arrays side by side.
        assert sent[0]["eq"] == window.eq_state.get_settings()
        assert {"band_freqs", "band_gains", "band_qs", "bands"} <= set(sent[0]["eq"])
        result = dialog.setup_result
        assert result is not None
        diagnostics = result["diagnostics"]
        assert diagnostics["compressor_calibration"]["backend"] != "unavailable"
        assert diagnostics["offline_validation"]["simulation_backend"] == "rust"
        assert diagnostics["offline_validation_passed"] is True
        reasons = diagnostics["uncertainty_reasons"]
        assert not [reason for reason in reasons if "unavailable" in reason or "validation" in reason]
    finally:
        dialog.reject()
        dialog.deleteLater()


@pytest.mark.parametrize(
    ("apply_recommended", "text", "style"),
    [
        (True, "Good capture. All checks passed.", "ok"),
        (False, "Good capture. These settings did not pass every check, so they are suggestions only.", "warn"),
    ],
    ids=["validated", "advisory"],
)
def test_the_summary_row_does_not_call_advisory_settings_highly_confident(
    real_main_window, apply_recommended, text, style
) -> None:
    from mic_eq.ui.theme import status_chip_style

    window = real_main_window
    dialog = VoiceSetupDialog(parent=window)
    _bind_capture_context(dialog, *_short_voice_captures(int(window.processor.sample_rate())))
    result = _voice_setup_result(window, dialog)
    result["diagnostics"]["apply_recommended"] = apply_recommended
    try:
        dialog._on_analysis_complete(result)

        assert dialog.overall_label.text() == text
        assert dialog.overall_label.styleSheet() == status_chip_style(style)
        assert dialog.warning_label.text().startswith(
            "All checks passed." if apply_recommended else "These settings did not pass every check"
        )
    finally:
        dialog.reject()
        dialog.deleteLater()


@pytest.mark.parametrize("outcome", ["accept", "rollback", "retry", "recording-failed"])
def test_the_review_page_keeps_the_proposal_after_the_verification_take(
    real_main_window, monkeypatch, outcome
) -> None:
    dialog, _speech = _verifying_dialog(real_main_window)
    monkeypatch.setattr(dialog, "_ensure_processor_ready", lambda: True)
    try:
        assert dialog.summary_group.isVisibleTo(dialog)
        dialog._on_start_clicked()
        dialog._capture_start_timer.stop()
        assert dialog.setup_state == "verification_recording"
        if outcome == "recording-failed":
            dialog._on_recording_failed("The microphone stopped.")
        else:
            dialog._on_verification_complete(_verdict_with_dulling_advisory(outcome, advised=False))

        assert dialog.setup_state in {"final_comparison_ready", "completed", "verification_ready"}
        assert dialog.review_page.isVisibleTo(dialog)
        assert dialog.summary_group.isVisibleTo(dialog)
    finally:
        dialog.reject()
        dialog.deleteLater()


def _start_noise_take(dialog: VoiceSetupDialog) -> None:
    """Pass the level check with a comfortable voice, then start the room-noise take."""
    t = np.arange(3 * 48_000) / 48_000
    dialog._on_start_clicked()
    dialog._capture_start_timer.stop()
    dialog._on_recording_complete((0.3 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32))
    assert dialog.setup_state == "level_result"
    dialog._on_start_clicked()
    dialog._capture_start_timer.stop()


@pytest.mark.parametrize("capture", ["noise", "voice"])
def test_a_rejected_capture_gives_the_target_controls_back(real_main_window, monkeypatch, capture) -> None:
    window = real_main_window
    dialog = VoiceSetupDialog(parent=window)
    monkeypatch.setattr(dialog, "_ensure_processor_ready", lambda: True)
    try:
        if capture == "voice":
            dialog.noise_audio = np.zeros(4800, dtype=np.float32)
            dialog.setup_state = "noise_ready"
            dialog._on_start_clicked()
            dialog._capture_start_timer.stop()
        else:
            _start_noise_take(dialog)
        assert dialog.setup_state == f"{capture}_recording"
        assert not dialog.dynamics_group.isEnabled()
        assert not dialog.target_lufs_spin.isEnabled()

        dialog._on_recording_failed("The microphone stopped.")

        assert dialog.setup_state == ("idle" if capture == "noise" else "noise_ready")
        assert dialog.curve_combo.isEnabled()
        assert dialog.dynamics_group.isEnabled()
        assert dialog.target_lufs_spin.isEnabled()
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_a_clipped_room_noise_take_leaves_the_target_page_usable(real_main_window, monkeypatch) -> None:
    dialog = VoiceSetupDialog(parent=real_main_window)
    monkeypatch.setattr(dialog, "_ensure_processor_ready", lambda: True)
    try:
        _start_noise_take(dialog)
        dialog._on_recording_complete(np.ones(48_000, dtype=np.float32))

        assert dialog.setup_state == "idle"
        assert dialog.phase_label.text() == "Room-noise capture rejected"
        assert dialog.dynamics_group.isEnabled()
        assert dialog.target_lufs_spin.isEnabled()
    finally:
        dialog.reject()
        dialog.deleteLater()


@pytest.mark.parametrize(
    ("before_enabled", "expected"),
    [
        (
            False,
            "Compressor: on, 4.0:1 above -20.0 dB, aiming for -18 LUFS\n"
            "Before: off, 4.0:1 above -20.0 dB, aiming for -18 LUFS",
        ),
        (True, "Compressor: on, 4.0:1 above -20.0 dB, aiming for -18 LUFS (unchanged)"),
    ],
    ids=["switched-on", "already-on"],
)
def test_the_review_says_whether_the_compressor_is_on(real_main_window, before_enabled, expected) -> None:
    window = real_main_window
    dialog = VoiceSetupDialog(parent=window)
    _bind_capture_context(dialog, *_short_voice_captures(int(window.processor.sample_rate())))
    window.compressor_state.set_settings({"enabled": before_enabled, "auto_makeup_enabled": True})
    result =_voice_setup_result(window, dialog, target_lufs=-18.0)
    result["compressor_settings"].update(enabled=True, auto_makeup_enabled=True)
    try:
        dialog._show_summary(result)

        assert dialog.compressor_label.text() == expected
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_a_context_change_during_setup_names_everything_the_context_holds(
    real_main_window, monkeypatch
) -> None:
    window = real_main_window
    dialog = VoiceSetupDialog(parent=window)
    dialog.noise_audio = np.zeros(4800, dtype=np.float32)
    dialog.setup_state = "voice_recording"
    dialog._capture_context_key = "route-a"
    monkeypatch.setattr(window, "_calibration_context_key", lambda: "route-b")
    try:
        dialog._begin_recording_capture()

        assert dialog.warning_label.text() == (
            "The audio route, input settings or processing mode changed during setup. "
            "Start the setup again without changing them."
        )
        assert dialog.setup_state == "noise_ready"
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_declining_the_device_switch_keeps_the_state_the_dialog_had(real_main_window, monkeypatch) -> None:
    from mic_eq.ui import voice_setup_dialog

    window = real_main_window
    dialog, _speech = _verifying_dialog(window)
    monkeypatch.setattr(window, "processor", SimpleNamespace(is_running=lambda: True))
    monkeypatch.setattr(voice_setup_dialog, "_active_device_identities", lambda _processor: (None, None))
    monkeypatch.setattr(voice_setup_dialog, "_route_identities_match", lambda *_args: False)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.StandardButton.No
    )
    try:
        assert dialog.setup_state == "verification_ready"
        text = dialog.start_button.text()
        curve_enabled = dialog.curve_combo.isEnabled()

        dialog._on_start_clicked()

        # The proposal is still applied, so the next step is still the verification take.
        assert dialog.setup_state == "verification_ready"
        assert dialog.start_button.text() == text == "Record Verification Passage"
        assert dialog.start_button.isEnabled()
        assert dialog.curve_combo.isEnabled() == curve_enabled
        assert dialog.warning_label.text() == (
            "Setup canceled. The microphone and speakers that were already running were kept."
        )
    finally:
        dialog.reject()
        dialog.deleteLater()


def _legacy_eq(window: MainWindow, gains: list[float]) -> dict[str, Any]:
    """The correction layer as the analysis hands it over: arrays, no typed bands."""
    current = window.eq_state.get_settings()
    return {
        "enabled": True,
        "band_freqs": current["band_freqs"],
        "band_gains": gains,
        "band_qs": current["band_qs"],
    }


def _set_live_eq(
    window: MainWindow,
    *,
    tone: list[float] | None = None,
    correction: list[float] | None = None,
    off_bands: tuple[int, ...] = (),
    enabled: bool = True,
) -> None:
    """Put the live EQ into a state with the given tone and correction layers."""
    current = window.eq_state.get_eq_settings()
    freqs, qs = current.band_freqs, current.band_qs
    window.eq_state.set_settings({"band_gains": tone or [0.0] * 10, "enabled": enabled})
    if correction is None:
        return
    eq = build_eq_candidate_settings(
        window.eq_state.get_eq_settings(), freqs, correction, qs, enabled=enabled
    )
    assert eq.correction_bands is not None
    window.eq_state.set_settings(
        EQSettings(
            enabled=enabled,
            bands=eq.tone_bands,
            correction_bands=tuple(
                replace(band, enabled=False) if index in off_bands else band
                for index, band in enumerate(eq.correction_bands)
            ),
            tone_bands=eq.tone_bands,
        ).to_dict()
    )


def _review_dialog(window: MainWindow) -> tuple[VoiceSetupDialog, np.ndarray]:
    dialog = VoiceSetupDialog(parent=window)
    noise, speech = _short_voice_captures(int(window.processor.sample_rate()))
    _bind_capture_context(dialog, noise, speech)
    return dialog, speech


def _comparison_dialog(monkeypatch, shown: dict[str, Any], *, keep: bool = False) -> None:
    """Stand in for the listening comparison: keep what it was given, then answer."""

    class _Comparison:
        def __init__(self, **kwargs):
            shown.update(kwargs)

        def exec(self):
            return int(QDialog.DialogCode.Accepted if keep else QDialog.DialogCode.Rejected)

        def deleteLater(self):
            pass

    monkeypatch.setattr(
        "mic_eq.ui.listening_comparison_dialog.ListeningComparisonDialog", _Comparison
    )


@pytest.mark.parametrize("mode", ["normal", "raw", "bypass"])
def test_the_analysis_assumes_the_processing_the_candidate_will_run(
    real_main_window, monkeypatch, mode
) -> None:
    from mic_eq.ui import voice_setup_dialog

    window = real_main_window
    window._set_processing_mode(mode)
    sent: list[dict[str, Any]] = []

    class _Signal:
        def connect(self, _slot):
            pass

    class _Worker:
        step_progress = result_ready = failed = finished = _Signal()

        def __init__(self, *_args, **kwargs):
            sent.append(kwargs)

        def start(self):
            pass

        def isRunning(self):
            return False

        def stop(self):
            pass

        def deleteLater(self):
            pass

    monkeypatch.setattr(voice_setup_dialog, "VoiceSetupWorker", _Worker)
    dialog, _speech = _review_dialog(window)
    try:
        dialog._start_analysis()

        assert sent[0]["processing_mode"] == "normal"
    finally:
        dialog.reject()
        dialog.deleteLater()


@pytest.mark.parametrize("mode", ["normal", "bypass"])
def test_the_pre_apply_comparison_plays_the_eq_that_apply_writes(
    real_main_window, monkeypatch, mode
) -> None:
    window = real_main_window
    window._set_processing_mode(mode)
    _set_live_eq(window, tone=[3.0] + [0.0] * 9)
    dialog, _speech = _review_dialog(window)
    result = _voice_setup_result(window, dialog)
    result["eq_settings"] = _legacy_eq(window, [1.5] * 10)
    shown: dict[str, Any] = {}
    _comparison_dialog(monkeypatch, shown)
    try:
        dialog._on_analysis_complete(result)
        dialog._compare_recording()

        proposed = EQSettings.from_dict(shown["proposed_settings"])
        assert [band.gain_db for band in proposed.correction_bands or ()] == [1.5] * 10
        assert [band.gain_db for band in proposed.tone_bands or ()] == [3.0] + [0.0] * 9
        assert shown["current_chain_settings"]["processing_mode"] == mode
        assert shown["proposed_chain_settings"]["processing_mode"] == "normal"

        dialog._apply_setup()

        assert window.eq_state.get_eq_settings().to_dict() == shown["proposed_settings"]
    finally:
        dialog.reject()
        dialog.deleteLater()


@pytest.mark.parametrize(
    ("live", "before"),
    [
        ({"correction": [4.0] * 10}, "4.0"),
        ({"tone": [12.0] + [0.0] * 9}, "0.0"),
        ({"correction": [2.0, 0.0, 0.0, 6.0] + [0.0] * 6, "off_bands": (3,)}, "2.0"),
    ],
    ids=["correction-only", "tone-only", "disabled-band"],
)
def test_the_tone_line_compares_correction_layers_with_enabled_bands(
    real_main_window, live, before
) -> None:
    window = real_main_window
    _set_live_eq(window, **live)
    dialog, _speech = _review_dialog(window)
    result = _voice_setup_result(window, dialog)
    result["eq_settings"] = _legacy_eq(window, [1.0] * 10)
    try:
        dialog._show_summary(result)

        assert dialog.eq_label.text() == (
            f"Tone: max correction 1.0 dB\nBefore: max correction {before} dB"
        )
    finally:
        dialog.reject()
        dialog.deleteLater()


@pytest.mark.parametrize(
    ("tone", "enabled", "note"),
    [
        (None, False, "\nThe EQ is off now; Apply will switch it on."),
        (
            [3.0] + [0.0] * 9,
            False,
            "\nThe EQ is off now; Apply will switch it on, "
            "and your current tone settings will take effect too.",
        ),
        ([3.0] + [0.0] * 9, True, ""),
    ],
    ids=["off-without-tone", "off-with-tone", "on"],
)
def test_the_tone_line_says_when_apply_switches_the_eq_on(
    real_main_window, tone, enabled, note
) -> None:
    window = real_main_window
    _set_live_eq(window, tone=tone, enabled=enabled)
    dialog, _speech = _review_dialog(window)
    result = _voice_setup_result(window, dialog)
    result["eq_settings"] = _legacy_eq(window, [1.0] * 10)
    try:
        dialog._show_summary(result)

        assert dialog.eq_label.text() == (
            "Tone: max correction 1.0 dB\nBefore: max correction 0.0 dB" + note
        )
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_a_verification_reduction_refreshes_the_compressor_line_and_keeps_its_before(
    real_main_window, monkeypatch
) -> None:
    window = real_main_window
    dialog, speech = _review_dialog(window)
    result = _voice_setup_result(window, dialog)
    result["compressor_settings"]["ratio"] = 3.0
    _apply_complete_candidate(window, dialog, result)
    before_line = dialog.compressor_label.text().split("\n")[1]
    assert before_line == "Before: on, 4.0:1 above -20.0 dB, 0.0 dB makeup"
    dialog._verification_audio = speech
    monkeypatch.setattr(dialog, "_start_verification_worker", lambda *_args: None)
    try:
        dialog._on_verification_complete(
            {
                "decision": "reduce",
                "reduction_targets": ["compressor"],
                "reasons": ["compressor p95 reduction exceeded target"],
            }
        )

        applied = window.compressor_state.get_settings()
        assert applied["ratio"] == pytest.approx(2.7)
        assert dialog.compressor_label.text() == (
            f"Compressor: on, {applied['ratio']:.1f}:1 above "
            f"{applied['threshold_db']:.1f} dB, 0.0 dB makeup\n{before_line}"
        )
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_the_confidence_row_follows_the_verification_and_the_restore(
    real_main_window, monkeypatch
) -> None:
    window = real_main_window
    dialog, _speech = _review_dialog(window)
    result = _voice_setup_result(window, dialog)
    result["diagnostics"]["apply_recommended"] = False
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes
    )
    _apply_complete_candidate(window, dialog, result)
    advisory = "Good capture. These settings did not pass every check, so they are suggestions only."
    _comparison_dialog(monkeypatch, {})
    try:
        assert dialog.overall_label.text() == advisory

        dialog._on_verification_complete(_verdict_with_dulling_advisory("accept", advised=False))

        assert dialog.setup_state == "final_comparison_ready"
        assert dialog.overall_label.text() == (
            "These settings passed the check on your second recording."
        )

        dialog._compare_recording()

        assert dialog.setup_state == "completed"
        assert dialog.overall_label.text() == advisory
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_the_final_comparison_opens_from_the_enabled_primary_button(
    real_main_window, monkeypatch
) -> None:
    window = real_main_window
    dialog, _speech = _review_dialog(window)
    result = _voice_setup_result(window, dialog)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes
    )
    _apply_complete_candidate(window, dialog, result)
    shown: dict[str, Any] = {}
    _comparison_dialog(monkeypatch, shown)
    try:
        dialog._on_verification_complete(_verdict_with_dulling_advisory("accept", advised=False))

        assert dialog.setup_state == "final_comparison_ready"
        assert dialog.start_button.text() == "Compare Final Candidate"
        assert dialog.start_button.isEnabled()

        dialog.start_button.click()

        assert shown
        assert dialog.setup_state == "completed"
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_the_manual_makeup_gain_is_named_instead_of_a_loudness_target(
    real_main_window
) -> None:
    window = real_main_window
    dialog, _speech = _review_dialog(window)
    window.compressor_state.set_settings({"auto_makeup_enabled": False, "makeup_gain_db": 3.0})
    result = _voice_setup_result(window, dialog)
    result["compressor_settings"].update(auto_makeup_enabled=False, makeup_gain_db=5.0)
    try:
        dialog._show_summary(result)

        assert dialog.compressor_label.text() == (
            "Compressor: on, 4.0:1 above -20.0 dB, 5.0 dB makeup\n"
            "Before: on, 4.0:1 above -20.0 dB, 3.0 dB makeup"
        )
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_a_failed_verification_take_clears_the_recording_status(
    real_main_window, monkeypatch
) -> None:
    dialog, _speech = _verifying_dialog(real_main_window)
    monkeypatch.setattr(dialog, "_ensure_processor_ready", lambda: True)
    try:
        dialog._on_start_clicked()
        dialog._capture_start_timer.stop()
        assert dialog.phase_label.text() == "Capturing a second passage"
        assert not dialog.retake_btn.isVisibleTo(dialog)

        dialog._on_recording_failed("The microphone stopped.")

        assert dialog.setup_state == "completed"
        assert dialog.phase_label.text() == "Second recording failed; previous settings put back"
        assert dialog.retake_btn.isVisibleTo(dialog)
        assert dialog.compare_button.isEnabled()
    finally:
        dialog.reject()
        dialog.deleteLater()


def _roll_back(dialog, _speech, _monkeypatch) -> None:
    dialog._on_verification_complete(_verdict_with_dulling_advisory("rollback", advised=False))


def _fail_verification(dialog, _speech, _monkeypatch) -> None:
    dialog._on_verification_failed("The check could not finish.")


def _cancel_verification(dialog, _speech, _monkeypatch) -> None:
    dialog.setup_state = "verification_analyzing"
    dialog._cancel_analysis_workers()


def _accept_a_stale_candidate(dialog, _speech, monkeypatch) -> None:
    monkeypatch.setattr(dialog, "_candidate_identity_error", lambda _owner: "The audio route changed")
    dialog._on_verification_complete(_verdict_with_dulling_advisory("accept", advised=False))


@pytest.mark.parametrize(
    "leave",
    [_roll_back, _fail_verification, _cancel_verification, _accept_a_stale_candidate],
    ids=["rollback", "failed", "canceled", "stale"],
)
def test_the_comparison_comes_back_wherever_the_dialog_returns_to_completed(
    real_main_window, monkeypatch, leave
) -> None:
    dialog, speech = _verifying_dialog(real_main_window)
    try:
        assert not dialog.compare_button.isEnabled()

        leave(dialog, speech, monkeypatch)

        assert dialog.setup_state == "completed"
        assert dialog.compare_button.isEnabled()
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_the_tone_line_and_confidence_row_follow_apply_and_restore(
    real_main_window, monkeypatch
) -> None:
    window = real_main_window
    _set_live_eq(window, enabled=False)
    dialog, _speech = _review_dialog(window)
    result = _voice_setup_result(window, dialog)
    result["eq_settings"] = _legacy_eq(window, [1.0] * 10)
    result["diagnostics"]["apply_recommended"] = False
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes
    )
    advisory = "Good capture. These settings did not pass every check, so they are suggestions only."
    tone = "Tone: max correction 1.0 dB\nBefore: max correction 0.0 dB"
    note = "\nThe EQ is off now; Apply will switch it on."
    try:
        dialog._on_analysis_complete(result)
        assert dialog.eq_label.text() == tone + note

        dialog._apply_setup()

        assert window.eq_state.get_settings()["enabled"]
        assert dialog.eq_label.text() == tone
        assert dialog.overall_label.text() == advisory

        assert dialog._restore_pre_setup_snapshot()

        assert not window.eq_state.get_settings()["enabled"]
        assert dialog.eq_label.text() == tone + note
        assert dialog.overall_label.text() == advisory
    finally:
        dialog.reject()
        dialog.deleteLater()


@pytest.mark.parametrize(
    ("mode", "sentence"),
    [
        ("bypass", "Bypass was switched to Normal so the settings take effect."),
        ("raw", "Raw monitor was switched to Normal so the settings take effect."),
        ("normal", ""),
    ],
)
def test_leaving_an_unprocessed_mode_is_said_on_apply_and_in_the_verified_box(
    real_main_window, monkeypatch, mode, sentence
) -> None:
    window = real_main_window
    window._set_processing_mode(mode)
    boxes: list[str] = []
    monkeypatch.setattr(
        "mic_eq.ui.voice_setup_dialog.QMessageBox.information",
        lambda *args, **_kwargs: boxes.append(str(args[-1])),
    )
    monkeypatch.setattr(window, "_prompt_save_current_preset", lambda **_kwargs: None)
    _comparison_dialog(monkeypatch, {}, keep=True)
    dialog, _speech = _verifying_dialog(window)
    try:
        applied = dialog.warning_label.text()
        dialog._on_verification_complete(_verdict_with_dulling_advisory("accept", advised=False))
        dialog._compare_recording()
    finally:
        dialog.reject()
        dialog.deleteLater()

    assert len(boxes) == 1
    for text in (applied, boxes[0]):
        if sentence:
            assert sentence in text
        else:
            assert "switched to Normal" not in text


def test_the_tone_line_is_unchanged_when_the_correction_layer_is_the_same(
    real_main_window
) -> None:
    window = real_main_window
    _set_live_eq(window, correction=[1.0] * 10)
    dialog, _speech = _review_dialog(window)
    result = _voice_setup_result(window, dialog)
    result["eq_settings"] = _legacy_eq(window, [1.0] * 10)
    try:
        dialog._show_summary(result)

        assert dialog.eq_label.text() == "Tone: max correction 1.0 dB (unchanged)"
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_a_zero_gain_filter_still_counts_as_a_tone_setting(real_main_window) -> None:
    window = real_main_window
    _set_live_eq(window, enabled=False)
    window.eq_state.set_band_value(0, "filter_type", "high_pass")
    dialog, _speech = _review_dialog(window)
    result = _voice_setup_result(window, dialog)
    result["eq_settings"] = _legacy_eq(window, [1.0] * 10)
    try:
        dialog._show_summary(result)

        assert dialog.eq_label.text().endswith(
            "Apply will switch it on, and your current tone settings will take effect too."
        )
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_a_stale_final_candidate_restores_the_review_and_ends_the_sentence(
    real_main_window, monkeypatch
) -> None:
    window = real_main_window
    dialog, _speech = _review_dialog(window)
    result = _voice_setup_result(window, dialog)
    result["diagnostics"]["apply_recommended"] = False
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes
    )
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: None)
    _apply_complete_candidate(window, dialog, result)
    try:
        dialog._on_verification_complete(_verdict_with_dulling_advisory("accept", advised=False))
        assert "passed the check" in dialog.overall_label.text()
        monkeypatch.setattr(window, "_calibration_context_key", lambda: "route-b")

        dialog._finalize_verified_setup()

        assert dialog.setup_state == "completed"
        assert dialog.overall_label.text() == (
            "Good capture. These settings did not pass every check, so they are suggestions only."
        )
        assert dialog.warning_label.text() == (
            "The microphone or its settings changed after you recorded. "
            "Record again so the settings match your microphone. "
            "Your previous settings were put back."
        )
    finally:
        dialog.reject()
        dialog.deleteLater()


@pytest.mark.parametrize("restored", [True, False], ids=["restored", "restore-failed"])
@pytest.mark.parametrize("last_take", [False, True], ids=["rollback", "third-bad-take"])
def test_every_rollback_text_follows_the_actual_restore_result(
    real_main_window, monkeypatch, restored, last_take
) -> None:
    dialog, _speech = _verifying_dialog(real_main_window)
    monkeypatch.setattr(dialog, "_restore_pre_setup_snapshot", lambda: restored)
    try:
        if last_take:
            dialog._verification_bad_capture_count = 2
        dialog._on_verification_complete(
            _verdict_with_dulling_advisory("retry" if last_take else "rollback", advised=False)
        )
        phase, warning = dialog.phase_label.text(), dialog.warning_label.text()
    finally:
        dialog.reject()
        dialog.deleteLater()

    if restored:
        assert phase.endswith("previous settings put back")
        assert "could not" not in warning
    else:
        assert phase.endswith("previous settings not fully put back")
        assert "Your previous settings could not be fully put back; press Retake to try again." in warning
        assert "were put back" not in warning and "put your previous settings back" not in warning
    if last_take:
        assert "Setup stopped after three tries." in warning
    else:
        assert phase.startswith("The new settings did not pass the check;")
        assert ("Your previous settings were put back." in warning) == restored
