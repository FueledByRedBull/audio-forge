"""Small end-to-end checks for accepted voice setup state."""

from __future__ import annotations

from copy import deepcopy
import time
from typing import Any

import numpy as np
import pytest
from PySide6.QtTest import QSignalSpy
from PySide6.QtCore import QElapsedTimer, QEventLoop, QThread, QTimer
from PySide6.QtWidgets import QDialog

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
        assert "accepted" in accepted_message
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
        assert "context changed" in messages[-1]
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_raw_monitor_verification_capture_allows_intended_mode_transition(
    real_main_window, monkeypatch
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
    window._set_processing_mode("raw")
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
        assert window._processing_mode() == "raw"
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
            "The recording could not be read. Record it again."
        )
        assert dialog.start_button.text() == "Apply & Verify Again"
        assert dialog.start_button.isEnabled()
        assert dialog.preview_verification_audio is None

        dialog._on_start_clicked()
        assert dialog.setup_state == "verification_ready"
    finally:
        dialog.reject()
        dialog.deleteLater()


DULLING_SENTENCE = (
    "noise suppression is dulling consonants by about 8 dB; "
    "lower its strength or use DeepFilterNet LL"
)
ADVICE = f"Advice: {DULLING_SENTENCE}."
METRICS = (
    "Target error 0.8→0.9 dB; compressor p95 6.0 dB; "
    "true peak -120.0 dBTP; SNR change -5.0 dB."
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


def _verdict_with_dulling_advisory(decision: str) -> dict[str, Any]:
    return {
        "decision": decision,
        "reason_codes": [f"{decision}_reason", "suppressor_dulling"],
        "reduction_targets": ["compressor"] if decision == "reduce" else [],
        "reasons": [FIRST_REASON[decision], DULLING_SENTENCE],
        "spectral_target_error_before_db": 0.8,
        "spectral_target_error_after_db": 0.9,
        "compressor_gain_reduction_p95_db": 6.0,
        "snr_change_db": -5.0,
    }


def _verifying_dialog(
    window: MainWindow, *, suppressor_enabled: bool
) -> tuple[VoiceSetupDialog, np.ndarray]:
    """A dialog whose applied proposal, with the suppressor on or off, awaits a verdict."""
    dialog = VoiceSetupDialog(parent=window)
    noise, speech = _short_voice_captures(int(window.processor.sample_rate()))
    _bind_capture_context(dialog, noise, speech)
    suppressor = {**window.noise_suppression_state.get_settings(), "enabled": suppressor_enabled}
    _apply_complete_candidate(
        window, dialog, _voice_setup_result(window, dialog, suppressor=suppressor)
    )
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
        "accept", None, "closing the comparison restores the original settings.", True,
        id="accept",
    ),
    pytest.param("rollback", None, METRICS, True, id="rollback"),
    pytest.param(
        "reduce", _reduce_out_of_adjustments,
        "Verification made no safe progress after three automatic adjustments.", True,
        id="reduce-out-of-adjustments",
    ),
    pytest.param(
        "reduce", _reduce_without_progress,
        "Verification made no measurable progress after the previous adjustment.", True,
        id="reduce-without-progress",
    ),
    pytest.param(
        "reduce", _reduce_that_changes_nothing, "Verification stopped: no progress.", True,
        id="reduce-that-changes-nothing",
    ),
    pytest.param("retry", None, METRICS, False, id="retry"),
    pytest.param(
        "retry", _last_take, "Maximum verification attempts reached.", False, id="last-retry"
    ),
]


@pytest.mark.parametrize("suppressor_enabled", [True, False], ids=["suppressor-on", "suppressor-off"])
@pytest.mark.parametrize(("decision", "setup", "tail", "advised"), DULLING_MESSAGES)
def test_the_dulling_advice_ends_the_message_as_its_own_sentence_or_is_left_out(
    real_main_window, monkeypatch, decision, setup, tail, advised, suppressor_enabled
) -> None:
    dialog, speech = _verifying_dialog(real_main_window, suppressor_enabled=suppressor_enabled)
    try:
        if setup is not None:
            setup(dialog, speech, monkeypatch)
        dialog._on_verification_complete(_verdict_with_dulling_advisory(decision))
        text = dialog.warning_label.text()
    finally:
        dialog.reject()
        dialog.deleteLater()

    assert ".;" not in text
    if advised and suppressor_enabled:
        assert text.endswith(f"{tail} {ADVICE}")
        assert text.count(DULLING_SENTENCE) == 1
    else:
        assert text.endswith(tail)
        assert DULLING_SENTENCE not in text
        assert "Advice" not in text


@pytest.mark.parametrize("suppressor_enabled", [True, False], ids=["suppressor-on", "suppressor-off"])
def test_the_verified_box_carries_the_dulling_advice_past_the_comparison(
    real_main_window, monkeypatch, suppressor_enabled
) -> None:
    dialog, _speech = _verifying_dialog(real_main_window, suppressor_enabled=suppressor_enabled)
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
        dialog._on_verification_complete(_verdict_with_dulling_advisory("accept"))
        assert dialog.setup_state == "final_comparison_ready"
        dialog._on_start_clicked()
    finally:
        dialog.reject()
        dialog.deleteLater()

    assert len(boxes) == 1
    if suppressor_enabled:
        assert f"\n\n{ADVICE}\n\n" in boxes[0]
    else:
        assert "Advice" not in boxes[0]
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
