# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
"""Tests for UI sample-rate propagation and diagnostics formatting."""

from __future__ import annotations

import re
import time
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, call

import numpy as np
import pytest
from PySide6.QtCore import QEventLoop, QThread, QTimer, Signal
from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import QMessageBox

from mic_eq.analysis.cancellation import AnalysisCancelled
from mic_eq.ui.calibration_dialog import CalibrationDialog
from mic_eq.ui.analysis_worker import AnalysisWorker
from mic_eq.ui.latency_calibration_dialog import (
    LatencyCalibrationDialog,
    _capture_sample_rate,
    _output_sample_rate,
    _resample_probe,
)
from mic_eq.ui.capture_session import device_name
from mic_eq.ui.device_selection import (
    default_device_index,
    find_identity_index,
    identity_is_persistable,
    preferred_output_index,
    start_processor_for_route,
)
from mic_eq.ui import main_window, rate_limiter
from mic_eq.ui.main_window import (
    NOISE_SELECTION_DROPPED_WARNING,
    MainWindow,
    _normalize_startup_preset_id,
    _startup_builtin_id,
)
from mic_eq.ui.auto_eq_explanation import explain_auto_eq
from mic_eq.ui.health import RecentStreamHealth
from mic_eq.ui.noise_suppression_state import NoiseSuppressionState
from mic_eq.ui.startup_presets import startup_custom_id as _startup_custom_id
from mic_eq.ui.theme import status_chip_style
from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog
from mic_eq.config import (
    AppConfig,
    CompressorSettings,
    DeviceIdentity,
    DevicePresetBinding,
    InputDevicePreference,
    LimiterSettings,
    Preset,
    build_device_route_key,
    build_input_device_preference_key,
    legacy_latency_profile_key,
)


def _freeze_throttle_clock(monkeypatch) -> None:
    """Make "applied at once, then throttled" independent of pauses such as garbage collection."""
    frozen = time.monotonic() + 1.0
    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: frozen)


def _use_processor(window, processor) -> None:
    """Swap the audio engine under a real window and the states that write to it."""
    window.processor = processor
    for state in (
        window.gate_state, window.eq_state, window.deesser_state,
        window.compressor_state, window.limiter_state, window.noise_suppression_state,
    ):
        state.processor = processor


def _status_log(window) -> list[str]:
    """Collect every status message the window shows from now on."""
    shown: list[str] = []
    window.status_message.changed.connect(
        lambda: shown.append(window.status_message.text)
    )
    return shown


@pytest.fixture
def window(qapp, monkeypatch):
    """A real MainWindow; tests stand in for the audio processor, never for Qt."""
    monkeypatch.setattr(main_window, "load_config", AppConfig)
    monkeypatch.setattr(main_window, "save_config", lambda _config: True)
    monkeypatch.setattr(main_window, "list_presets", lambda: [])
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [])
    window = MainWindow()
    native = window.processor
    # Tests drive updates directly instead of through the event loop.
    window.meter_timer.stop()
    window.diagnostics_timer.stop()
    yield window
    _use_processor(window, native)
    window.meter_timer.stop()
    window.diagnostics_timer.stop()
    # Closing saves settings; a failing save would open a modal warning.
    monkeypatch.setattr(main_window, "save_config", lambda _config: True)
    window.close()
    window.deleteLater()
    qapp.processEvents()


def test_setup_level_warnings_use_actual_peak_not_rms_estimate(window):
    _use_processor(window, _FakeProcessor())
    calibration = CalibrationDialog(parent=window)
    calibration._on_level_update(-20.0, -1.0)
    assert (calibration.level_meter.rms_db, calibration.level_meter.peak_db) == (-20.0, -1.0)
    assert "peaks at -1.0 dB" in calibration.warning_label.text()

    voice_setup = VoiceSetupDialog(parent=window)
    voice_setup.setup_state = "voice_recording"
    voice_setup._update_level(-20.0, -1.0)
    assert (voice_setup.level_meter.rms_db, voice_setup.level_meter.peak_db) == (-20.0, -1.0)
    assert "peaks at -1.0 dB" in voice_setup.warning_label.text()


class _SignalStub:
    def connect(self, _slot):
        return None


class _CaptureWorkerStub:
    last_init: dict | None = None

    def __init__(
        self,
        audio_data,
        sample_rate,
        target_preset,
        target_mode="adaptive",
        smoothing_strength="conservative",
        chain_settings=None,
        headroom_candidate_base_eq_settings=None,
        headroom_audio_data=None,
        headroom_chain_settings=None,
    ):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.target_preset = target_preset
        self.target_mode = target_mode
        self.smoothing_strength = smoothing_strength
        self.chain_settings = chain_settings
        _CaptureWorkerStub.last_init = {
            "audio_len": len(audio_data),
            "sample_rate": sample_rate,
            "target_preset": target_preset,
            "target_mode": target_mode,
            "smoothing_strength": smoothing_strength,
            "chain_settings": chain_settings,
            "headroom_candidate_base_eq_settings": headroom_candidate_base_eq_settings,
            "headroom_audio_data": headroom_audio_data,
            "headroom_chain_settings": headroom_chain_settings,
        }
        self.step_progress = _SignalStub()
        self.result_ready = _SignalStub()
        self.finished = _SignalStub()
        self.failed = _SignalStub()

    def start(self):
        return None

    def isRunning(self):
        return False

    def stop(self):
        return None

    def wait(self, _timeout=None):
        return True

    def deleteLater(self):
        return None


class _SlowAnalysisWorker(QThread):
    """Non-cooperative worker used to verify asynchronous dialog teardown."""

    step_progress = Signal(str, int)
    result_ready = Signal(dict)
    failed = Signal(str)
    instances: list["_SlowAnalysisWorker"] = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.stop_requested = False
        self.instances.append(self)

    def stop(self):
        self.stop_requested = True

    def run(self):
        time.sleep(1.6)
        self.result_ready.emit({})


class _FakeProcessor:
    def __init__(self, sample_rate: int = 44_100, output_sample_rate: int = 48_000):
        self._sample_rate = sample_rate
        self._output_sample_rate = output_sample_rate

    def sample_rate(self) -> int:
        return self._sample_rate

    def output_sample_rate(self) -> int:
        return self._output_sample_rate

    def get_input_cleanup_mode(self) -> str:
        return "gentle"


@pytest.mark.parametrize(
    ("save_results", "overwrite_reply"),
    [
        ([Path("saved.json")], None),
        ([FileExistsError("exists")], QMessageBox.StandardButton.No),
        (
            [FileExistsError("exists"), Path("saved.json")],
            QMessageBox.StandardButton.Yes,
        ),
        (
            [FileExistsError("exists"), OSError("cannot write")],
            QMessageBox.StandardButton.Yes,
        ),
    ],
    ids=("success", "collision-declined", "collision-accepted", "overwrite-failure"),
)
def test_save_preset_file_handles_collision_and_errors(
    window, monkeypatch, save_results, overwrite_reply
):
    save_mock = Mock(side_effect=save_results)
    question = Mock(return_value=overwrite_reply)
    critical = Mock()
    monkeypatch.setattr(main_window, "save_preset", save_mock)
    monkeypatch.setattr(main_window.QMessageBox, "question", question)
    monkeypatch.setattr(main_window.QMessageBox, "critical", critical)
    window.preset_modified = True
    preset = Preset(name="My Preset")
    result = window._save_preset_file(preset)

    expected_path = save_results[-1] if isinstance(save_results[-1], Path) else None
    overwrite_flags = [False] + (
        [True] if overwrite_reply == QMessageBox.StandardButton.Yes else []
    )
    error = isinstance(save_results[-1], OSError) and not isinstance(
        save_results[-1], FileExistsError
    )
    assert result == expected_path
    if expected_path is not None:
        assert window.preset_modified is False
    assert [kwargs["overwrite"] for _args, kwargs in save_mock.call_args_list] == [
        *overwrite_flags
    ]
    assert question.call_count == int(isinstance(save_results[0], FileExistsError))
    assert critical.call_count == int(error)
    if error:
        assert "cannot write" in critical.call_args.args[2]


class _MeterProcessor:
    def __init__(self):
        self.diagnostics_calls = 0
        self._diagnostics = {
            "input_dropped_samples": 12,
            "lock_contention_count": 1,
            "suppressor_non_finite_count": 0,
            "stream_restart_count": 2,
            "output_underrun_total": 3,
            "output_underrun_streak": 0,
            "output_retime_adjustment_count": 606,
            "output_recovery_event_count": 4,
            "output_recovery_count": 4,
            "output_short_write_dropped_samples": 0,
            "rt_buffer_overflow_count": 8,
            "input_callback_error_count": 9,
            "output_callback_error_count": 10,
            "rt_error_name": "fixed real-time buffer overflow",
            "input_backlog_recovery_count": 5,
            "input_backlog_dropped_samples": 6,
            "clip_event_count": 7,
            "clip_peak_db": -1.25,
            "output_clip_event_count": 2,
            "output_clip_peak_db": -0.75,
            "output_true_peak_event_count": 1,
            "output_true_peak_db": 0.4,
            "limiter_careful_output_enabled": True,
            "limiter_effective_ceiling_db": -1.5,
            "gate_chatter_event_count": 3,
            "input_channel_mode": "average",
            "input_stereo_correlation": 1.0,
            "input_phase_warning_count": 0,
            "input_resampler_active": True,
            "output_resampler_active": False,
            "noise_model": "rnnoise",
            "noise_backend_available": True,
            "noise_backend_failed": False,
            "noise_backend_error": None,
            "recovery_suppressed": False,
            "last_restart_reason": None,
        }

    def is_running(self) -> bool:
        return True

    def get_runtime_diagnostics(self):
        self.diagnostics_calls += 1
        return self._diagnostics

    def get_input_rms_db(self) -> float:
        return -20.0

    def get_input_peak_db(self) -> float:
        return -10.0

    def get_output_rms_db(self) -> float:
        return -18.0

    def get_output_peak_db(self) -> float:
        return -8.0

    def get_compressor_gain_reduction_db(self) -> float:
        return 1.5

    def get_deesser_gain_reduction_db(self) -> float:
        return 0.25

    def get_latency_ms(self) -> float:
        return 24.0

    def get_dsp_time_smoothed_ms(self) -> float:
        return 3.2

    def get_input_buffer_smoothed_samples(self) -> int:
        return 512

    def get_output_buffer_samples(self) -> int:
        return 1024

    def get_buffer_smoothed_samples(self) -> int:
        return 256

    def get_compressor_auto_makeup_enabled(self) -> bool:
        return False

    def get_vad_probability(self) -> float:
        return 0.0

    def get_input_callback_age_ms(self) -> int:
        return 5

    def get_output_callback_age_ms(self) -> int:
        return 10

    def take_input_peak_since_read(self) -> float:
        return 0.0

    take_output_peak_since_read = take_output_pre_limiter_peak_since_read = take_input_peak_since_read

    def get_input_full_scale_count(self) -> int:
        return 0

    get_output_full_scale_count = get_input_full_scale_count

    def service_recovery(self):
        return None

    def get_last_restart_reason(self):
        return None

    def get_last_stream_error(self):
        return None


class _StoppedRecoveryProcessor(_MeterProcessor):
    def __init__(self):
        super().__init__()
        self.running = False
        self.recovery_calls = 0

    def is_running(self) -> bool:
        return self.running

    def service_recovery(self):
        self.recovery_calls += 1
        if self.recovery_calls == 2:
            self.running = True
            return True
        return False


def _recovery_setup(window, processor=None):
    """Run recovery and diagnostics against a processor that reports healthy telemetry."""
    _use_processor(window, processor or _MeterProcessor())
    window._processing = True


def _healthy_runtime_diagnostics(**overrides):
    diagnostics = {
        "input_dropped_samples": 0,
        "lock_contention_count": 0,
        "suppressor_non_finite_count": 0,
        "stream_restart_count": 0,
        "output_underrun_total": 0,
        "output_underrun_streak": 0,
        "output_retime_adjustment_count": 0,
        "output_recovery_event_count": 0,
        "output_recovery_count": 0,
        "output_short_write_dropped_samples": 0,
        "rt_buffer_overflow_count": 0,
        "input_callback_error_count": 0,
        "output_callback_error_count": 0,
        "rt_error_name": "none",
        "clip_event_count": 0,
        "clip_peak_db": -120.0,
        "output_clip_event_count": 0,
        "output_clip_peak_db": -120.0,
        "output_true_peak_event_count": 0,
        "output_true_peak_db": -120.0,
        "gate_chatter_event_count": 0,
        "deesser_detector_confidence": 0.0,
        "noise_model": "rnnoise",
        "noise_backend_available": True,
        "noise_backend_failed": False,
        "noise_backend_error": None,
        "recovery_suppressed": False,
        "last_restart_reason": None,
    }
    diagnostics.update(overrides)
    return diagnostics


def test_calibration_analysis_uses_processor_sample_rate(window, monkeypatch):
    _use_processor(window, _FakeProcessor(sample_rate=44_100))
    dialog = CalibrationDialog(parent=window)
    dialog.audio_data = np.ones(256, dtype=np.float32)
    dialog.preview_audio_data = dialog.audio_data.copy()
    monkeypatch.setattr(
        "mic_eq.ui.calibration_dialog.AnalysisWorker", _CaptureWorkerStub
    )

    dialog._start_analysis()

    assert _CaptureWorkerStub.last_init is not None
    assert _CaptureWorkerStub.last_init["sample_rate"] == 44_100
    assert _CaptureWorkerStub.last_init["target_mode"] == "adaptive"
    assert _CaptureWorkerStub.last_init["smoothing_strength"] == "conservative"
    assert _CaptureWorkerStub.last_init["chain_settings"]["full_chain"] is True
    assert _CaptureWorkerStub.last_init["headroom_chain_settings"]["full_chain"] is True
    assert _CaptureWorkerStub.last_init["headroom_candidate_base_eq_settings"] == (
        window._get_current_preset().eq.to_dict()
    )
    np.testing.assert_array_equal(
        _CaptureWorkerStub.last_init["headroom_audio_data"], dialog.audio_data
    )
    assert dialog.analysis_worker is not None
    assert dialog.analysis_worker.sample_rate == 44_100

    dialog.close()


def _process_events_for(qapp, duration_ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(duration_ms, loop.quit)
    loop.exec()


def test_calibration_cancel_invalidates_delayed_capture_start(qapp, window):
    processor = Mock()
    processor.sample_rate.return_value = 48_000
    _use_processor(window, processor)
    dialog = CalibrationDialog(parent=window)
    dialog.recording_state = "recording"
    dialog._capture_start_timer.start(1)

    dialog.reject()
    _process_events_for(qapp, 25)

    processor.start_raw_recording.assert_not_called()
    assert not dialog._capture_start_timer.isActive()


def test_voice_setup_cancel_invalidates_delayed_capture_start(qapp, window):
    processor = Mock()
    processor.sample_rate.return_value = 48_000
    _use_processor(window, processor)
    dialog = VoiceSetupDialog(parent=window)
    dialog.setup_state = "voice_recording"
    dialog._capture_start_timer.start(1)

    dialog.reject()
    _process_events_for(qapp, 25)

    processor.start_raw_recording.assert_not_called()
    assert not dialog._capture_start_timer.isActive()


def test_calibration_ignores_stale_analysis_generation(window):
    _use_processor(window, _FakeProcessor())
    dialog = CalibrationDialog(parent=window)
    dialog._analysis_generation = 2
    dialog.recording_state = "analyzing"
    previous_step = dialog.warning_label.text()

    dialog._on_analysis_step("stale step", 55, generation=1)

    assert dialog.warning_label.text() == previous_step

    dialog._on_analysis_complete(
        {"band_gains": [0.0] * 10, "apply_recommended": True},
        generation=1,
    )

    assert dialog.eq_settings is None
    assert dialog.recording_state == "analyzing"
    dialog.close()


def test_analysis_close_waits_for_a_slow_worker_and_ignores_its_result(
    qapp, window, monkeypatch
):
    _SlowAnalysisWorker.instances.clear()
    monkeypatch.setattr(
        "mic_eq.ui.calibration_dialog.AnalysisWorker", _SlowAnalysisWorker
    )
    _use_processor(window, _FakeProcessor())
    dialog = CalibrationDialog(parent=window)
    dialog.audio_data = np.ones(256, dtype=np.float32)
    dialog.preview_audio_data = dialog.audio_data.copy()
    dialog._start_analysis()
    worker = dialog.analysis_worker
    assert worker is not None and worker.isRunning()

    dialog.reject()
    assert dialog._close_requested is True
    assert worker.stop_requested is True
    _process_events_for(qapp, 2_100)

    assert dialog._analysis_workers == []
    assert dialog._close_requested is False
    assert dialog.eq_settings is None


def test_analysis_worker_passes_cooperative_cancellation_into_pipeline(
    qapp, monkeypatch
):
    entered = threading.Event()
    callbacks: list[object] = []

    def blocking_analysis(*_args, cancel_check=None, **_kwargs):
        callbacks.append(cancel_check)
        entered.set()
        if not callable(cancel_check):
            raise AssertionError("worker did not provide a cancellation callback")
        while not cancel_check():
            time.sleep(0.005)
        raise AnalysisCancelled("sentinel")

    monkeypatch.setattr(
        "mic_eq.ui.analysis_worker.analyze_auto_eq", blocking_analysis
    )
    worker = AnalysisWorker(np.zeros(256, dtype=np.float32), 48_000)
    worker.start()
    assert entered.wait(1.0)

    worker.stop()
    assert worker.wait(2_000)
    assert len(callbacks) == 1
    assert callable(callbacks[0])


def test_analysis_worker_passes_exact_candidate_headroom_inputs(monkeypatch):
    received: dict[str, object] = {}
    analysis_audio = np.arange(256, dtype=np.float32)
    raw_audio = np.arange(256, dtype=np.float32) / 2.0
    incumbent_eq = Preset().eq.to_dict()
    analysis_chain = {"full_chain": True, "input_pre_filtered": True}
    headroom_chain = {"full_chain": True, "input_pre_filtered": False}

    def analyze(*_args, **kwargs):
        received.update(kwargs)
        return {}, SimpleNamespace(passed=True, reason="")

    monkeypatch.setattr("mic_eq.ui.analysis_worker.analyze_auto_eq", analyze)
    worker = AnalysisWorker(
        analysis_audio,
        48_000,
        chain_settings=analysis_chain,
        headroom_candidate_base_eq_settings=incumbent_eq,
        headroom_audio_data=raw_audio,
        headroom_chain_settings=headroom_chain,
    )

    worker.run()

    assert received["chain_settings"] == analysis_chain
    assert received["headroom_candidate_base_eq_settings"] == incumbent_eq
    np.testing.assert_array_equal(received["headroom_audio_data"], raw_audio)
    assert received["headroom_chain_settings"] == headroom_chain


def test_calibration_dialog_shows_auto_eq_diagnostics(window):
    _use_processor(window, _FakeProcessor(sample_rate=44_100))
    dialog = CalibrationDialog(parent=window)
    eq_settings = {
        "band_freqs": [
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
        ],
        "band_gains": [0.0] * 10,
        "band_qs": [1.41] * 10,
        "analysis_confidence": 0.76,
        "eq_confidence": 0.74,
        "capture_confidence": 0.81,
        "validation_confidence": 0.78,
        "validation_before_error_db": 5.0,
        "validation_after_error_db": 2.5,
        "validation_gain_scale": 0.9,
        "target_profile": "broadcast:adaptive",
        "smoothing_strength": "conservative",
        "residual_regularization": {
            "max_requested_correction_db": 6.0,
            "max_regularized_correction_db": 4.0,
            "max_narrow_residual_db": 2.0,
        },
    }

    dialog.recording_state = "analyzing"
    dialog._candidate_target_metadata = ("broadcast", "adaptive", "conservative")
    dialog._on_analysis_complete(eq_settings)

    assert not dialog.diagnostics_group.isHidden()
    assert dialog.confidence_label.text().startswith("Good capture: ")
    assert "clear and steady enough to base a correction on" in dialog.confidence_label.text()
    assert dialog.confidence_label.styleSheet() == status_chip_style("ok")
    assert (
        "Capture quality score 81%." in dialog.confidence_label.toolTip()
    )
    assert "Overall score: 76%." in dialog.confidence_label.toolTip()
    assert (
        dialog.error_label.text()
        == "Distance from the tone preset: 5.0 dB; no correction was applied."
    )
    assert (
        dialog.error_label.toolTip()
        == "The correction is used at 90% of its fitted strength. "
        "Smoothing took the largest change from 6.0 dB to 4.0 dB."
    )
    assert dialog.gain_scale_label.text() == "Headroom check: not available."
    assert (
        dialog.target_profile_label.text()
        == f"Tone preset: {dialog.curve_combo.currentText()}"
    )
    assert (
        dialog.target_profile_label.toolTip() == "Tone preset and matching: broadcast, Subtle."
    )

    dialog.close()


def test_calibration_dialog_does_not_offer_abstained_eq(window):
    _use_processor(window, _FakeProcessor(sample_rate=48_000))
    dialog = CalibrationDialog(parent=window)
    eq_settings = {
        "band_freqs": [
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
        ],
        "band_gains": [0.0] * 10,
        "band_qs": [1.41] * 10,
        "analysis_confidence": 0.20,
        "apply_recommended": False,
        "recommendation_status": "abstain",
        "abstention_reasons": ["noise-referenced SNR is too low"],
    }

    dialog.recording_state = "analyzing"
    dialog._candidate_target_metadata = ("broadcast", "adaptive", "conservative")
    dialog._on_analysis_complete(eq_settings)

    assert dialog.eq_settings is None
    assert dialog.start_button.text() == "Record Again"
    # Plain language up front; the raw reason stays available on hover.
    text = dialog.warning_label.text()
    explanation = explain_auto_eq(eq_settings)
    assert explanation.outcome_code == "unusable_capture" and explanation.state == "bad"
    assert text.startswith("No correction applied. ")
    assert "Your voice was too close to the background noise in the recording." in text
    assert "noise-referenced SNR is too low" not in text
    assert "noise-referenced SNR is too low" in dialog.warning_label.toolTip()
    # Review step: the setup and recording widgets are out of the way.
    assert dialog.curve_group.isHidden()
    assert dialog.instructions_group.isHidden()
    assert dialog.level_meter_widget.isHidden()

    dialog.close()


def test_latency_calibration_keeps_capture_and_output_rates_distinct(window):
    _use_processor(window, _FakeProcessor(sample_rate=48_000, output_sample_rate=44_100))
    assert _capture_sample_rate(window) == 48_000
    assert _output_sample_rate(window) == 44_100


def test_latency_calibration_resamples_selected_route_probe():
    probe = np.linspace(-0.5, 0.5, 480, dtype=np.float32)
    resampled = _resample_probe(probe, 48_000, 44_100)

    assert resampled.dtype == np.float32
    assert resampled.flags.c_contiguous
    assert abs(len(resampled) - 441) <= 1
    assert np.all(np.isfinite(resampled))


def test_main_window_diagnostics_include_new_metrics():
    diagnostics = {
        "input_dropped_samples": 12,
        "lock_contention_count": 1,
        "suppressor_non_finite_count": 0,
        "stream_restart_count": 2,
        "output_underrun_total": 3,
        "output_underrun_streak": 2,
        "output_retime_adjustment_count": 606,
        "output_recovery_event_count": 4,
        "output_recovery_count": 4,
        "output_short_write_dropped_samples": 11,
        "rt_buffer_overflow_count": 8,
        "input_callback_error_count": 9,
        "output_callback_error_count": 10,
        "rt_error_name": "fixed real-time buffer overflow",
        "input_backlog_recovery_count": 5,
        "input_backlog_dropped_samples": 6,
        "clip_event_count": 7,
        "clip_peak_db": -1.25,
        "output_clip_event_count": 2,
        "output_clip_peak_db": -0.75,
        "output_true_peak_event_count": 1,
        "output_true_peak_db": 0.4,
        "limiter_effective_ceiling_db": -1.5,
        "gate_chatter_event_count": 3,
        "deesser_detector_confidence": 0.64,
        "input_channel_mode": "average",
        "input_stereo_correlation": -0.9,
        "input_phase_warning_count": 3,
        "input_resampler_active": True,
        "output_resampler_active": False,
    }

    dropped_bits: list[str] = []
    MainWindow._extend_diag_tokens(
        dropped_bits,
        diagnostics,
        [
            ("input_backlog_recovery_count", "IBR"),
            ("input_backlog_dropped_samples", "IBD"),
            ("output_short_write_dropped_samples", "OSW"),
            ("rt_buffer_overflow_count", "RTO"),
            ("input_callback_error_count", "ICE"),
            ("output_callback_error_count", "OCE"),
            ("clip_event_count", "CL"),
            ("clip_peak_db", "PK"),
            ("output_clip_event_count", "OCL"),
            ("output_clip_peak_db", "OPK"),
            ("output_true_peak_event_count", "OTP"),
            ("output_true_peak_db", "TPK"),
            ("limiter_effective_ceiling_db", "LIM"),
            ("gate_chatter_event_count", "GCH"),
            ("deesser_detector_confidence", "DSC"),
        ],
    )
    backend_bits = ["rnnoise"]
    MainWindow._extend_diag_tokens(
        backend_bits,
        diagnostics,
        [
            ("input_resampler_active", "IR"),
            ("output_resampler_active", "OR"),
        ],
    )

    assert "IBR:5" in dropped_bits
    assert "IBD:6" in dropped_bits
    assert "OSW:11" in dropped_bits
    assert "RTO:8" in dropped_bits
    assert "ICE:9" in dropped_bits
    assert "OCE:10" in dropped_bits
    assert "CL:7" in dropped_bits
    assert "PK:-1.2" in dropped_bits
    assert "OCL:2" in dropped_bits
    assert "OPK:-0.8" in dropped_bits
    assert "OTP:1" in dropped_bits
    assert "TPK:0.4" in dropped_bits
    assert "LIM:-1.5" in dropped_bits
    assert "GCH:3" in dropped_bits
    assert "DSC:0.6" in dropped_bits
    assert "IR:Y" in backend_bits
    assert "OR:N" in backend_bits


def test_health_panel_reports_input_output_and_callback_states(window):
    _recovery_setup(window)

    window._update_diagnostic_labels(
        diagnostics=_healthy_runtime_diagnostics(),
        latency_ms=24.0,
        dsp_time_ms=0.5,
        input_buf=128,
        output_buf=256,
        rnnoise_buf=0,
        input_rms_db=-72.0,
        output_rms_db=-18.0,
        input_callback_age_ms=12,
        output_callback_age_ms=14,
    )

    assert "Input: LOW" in window.shell_texts["input_health_label"].text
    assert window.shell_texts["input_health_label"].state == "warn"
    assert "Output: OK" in window.shell_texts["output_health_label"].text
    assert window.shell_texts["output_health_label"].state == "ok"
    activity = window.shell_texts["callback_health_label"].text
    assert activity.startswith("Device activity:")
    assert re.search(r"input\s+12 ms\b.*output\s+14 ms\b", activity)
    assert window.shell_texts["callback_health_label"].state == "ok"

    window._update_diagnostic_labels(
        diagnostics=_healthy_runtime_diagnostics(
            clip_event_count=1,
            output_clip_event_count=1,
        ),
        latency_ms=24.0,
        dsp_time_ms=0.5,
        input_buf=128,
        output_buf=256,
        rnnoise_buf=0,
        input_rms_db=-12.0,
        output_rms_db=-0.5,
        input_callback_age_ms=300,
        output_callback_age_ms=20,
        input_peak_db=0.0,
        input_full_scale_samples=9,
        output_peak_db=1.5,
        output_at_full_scale=True,
        limiter_enabled=False,
    )

    clipped = window.shell_texts["input_health_label"].text
    assert clipped.startswith("Input: CLIPPED")
    assert re.search(r"(?<![\w.-])0\.0 dB\b", clipped) and re.search(r"(?<![\w.-])9(?![\w.])", clipped)
    assert window.shell_texts["input_health_label"].state == "bad"
    full_scale = window.shell_texts["output_health_label"].text
    assert full_scale.startswith("Output: AT FULL SCALE")
    assert re.search(r"(?<![\w.-])1\.5 dB\b", full_scale)
    assert window.shell_texts["output_health_label"].state == "bad"
    assert window.shell_texts["callback_health_label"].state == "warn"


def _nothing_dropped(text: str) -> bool:
    return text.startswith("Drops:") and "nothing dropped" in text and not re.search(r"\b[1-9]\d* samples?\b", text)


def _labels(window, **changes):
    args = dict(
        diagnostics=_healthy_runtime_diagnostics(),
        latency_ms=24.0, dsp_time_ms=0.5, input_buf=128, output_buf=256, rnnoise_buf=0,
        input_rms_db=-18.0, output_rms_db=-18.0,
        input_callback_age_ms=10, output_callback_age_ms=10,
    )
    args.update(changes)
    window._update_diagnostic_labels(**args)


NEVER = 2**64 - 1


def test_a_device_that_has_not_called_back_yet_is_unknown_not_a_huge_number(window):
    from mic_eq.ui.health import advice_for

    _recovery_setup(window)
    window._processing = True  # still starting up
    never = 2**64 - 1  # what the engine reports before a device's first callback
    chip = window.shell_texts["callback_health_label"]

    _labels(window, input_callback_age_ms=never, output_callback_age_ms=never)
    assert chip.text.startswith("Device activity:")
    assert re.search(r"input[^,]*not started", chip.text) and re.search(r"output[^,]*not started", chip.text)
    assert not any(ch.isdigit() for ch in chip.text)  # no age, least of all the sentinel
    assert chip.state == "idle"

    _labels(window, input_callback_age_ms=never, output_callback_age_ms=14)
    assert re.search(r"input[^,]*not started", chip.text) and re.search(r"output[^,]*\b14 ms\b", chip.text)
    assert "1844" not in chip.text
    assert chip.state == "ok"

    _labels(window, input_callback_age_ms=never, output_callback_age_ms=1400)
    assert chip.state == "bad"
    advice = advice_for(chip.text)
    assert "for 1400 ms" in advice and "1844" not in advice and "1844" not in chip.text


@pytest.mark.parametrize(
    ("input_age", "output_age", "input_says", "output_says"),
    [
        (NEVER, 14, "not started", r"\b14 ms\b"),
        (12, NEVER, r"\b12 ms\b", "not started"),
        (NEVER, NEVER, "not started", "not started"),
    ],
)
def test_a_device_that_never_starts_is_a_warning_once_the_startup_grace_is_over(
    window, monkeypatch, input_age, output_age, input_says, output_says
):
    clock = [1000.0]
    monkeypatch.setattr("mic_eq.ui.main_window.time.monotonic", lambda: clock[0])
    _recovery_setup(window)
    chip = window.shell_texts["callback_health_label"]

    _labels(window, input_callback_age_ms=input_age, output_callback_age_ms=output_age)
    clock[0] += main_window.DEVICE_STARTUP_GRACE_SECONDS - 2
    _labels(window, input_callback_age_ms=input_age, output_callback_age_ms=output_age)
    text = chip.text
    assert chip.state != "warn"
    assert text.startswith("Device activity:")
    assert re.search(rf"input[^,]*{input_says}", text) and re.search(rf"output[^,]*{output_says}", text)

    clock[0] += 3
    _labels(window, input_callback_age_ms=input_age, output_callback_age_ms=output_age)
    assert (chip.text, chip.state) == (text, "warn")
    summary = window.shell_texts["health_summary_label"]
    assert (summary.text, summary.state) == (text, "warn")
    assert "has not started" in window.shell_texts["health_advice_label"].text
    assert " ms" not in window.shell_texts["health_advice_label"].text

    window._processing = False  # not processing: nothing to warn about
    _labels(window, input_callback_age_ms=input_age, output_callback_age_ms=output_age)
    assert chip.state != "warn"


def test_a_restart_long_after_processing_began_gets_the_startup_grace_again(window, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("mic_eq.ui.main_window.time.monotonic", lambda: clock[0])
    _recovery_setup(window)
    window._processing = True  # the engine has run for a while
    chip = window.shell_texts["callback_health_label"]

    _labels(window)
    assert chip.state == "ok"

    # An engine restart that bypasses _start_processing: both ages are back to the sentinel.
    _labels(window, input_callback_age_ms=NEVER, output_callback_age_ms=NEVER)
    assert chip.state != "warn"
    clock[0] += main_window.DEVICE_STARTUP_GRACE_SECONDS - 1
    _labels(window, input_callback_age_ms=NEVER, output_callback_age_ms=NEVER)
    assert chip.state != "warn"

    clock[0] += 2
    _labels(window, input_callback_age_ms=NEVER, output_callback_age_ms=NEVER)
    assert chip.state == "warn"

    # The device recovers, then a second restart waits out a fresh grace.
    _labels(window)
    _labels(window, input_callback_age_ms=NEVER, output_callback_age_ms=NEVER)
    assert chip.state != "warn"


def test_gate_chatter_alone_is_the_gate_not_an_engine_fault(window, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("mic_eq.ui.health.time.monotonic", lambda: clock[0])
    _recovery_setup(window)
    chip = window.shell_texts["dropped_label"]
    summary = window.shell_texts["health_summary_label"]
    _labels(window, diagnostics=_healthy_runtime_diagnostics())
    assert chip.state == "ok"

    # Everything is zero except a gate-chatter counter that keeps rising.
    for events in (1, 2, 3):
        clock[0] += 1
        _labels(window, diagnostics=_healthy_runtime_diagnostics(gate_chatter_event_count=events))
        assert chip.state == "warn" and _nothing_dropped(chip.text) and "chatter" in chip.text
        assert "fault" not in chip.text
    assert window.shell_texts["gate_health_label"].text.startswith("Gate: CHATTERING")
    assert summary.state == "warn"


def test_a_real_error_counter_still_reads_as_an_engine_fault(window, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("mic_eq.ui.health.time.monotonic", lambda: clock[0])
    _recovery_setup(window)
    chip = window.shell_texts["dropped_label"]
    _labels(window, diagnostics=_healthy_runtime_diagnostics())

    clock[0] += 1
    _labels(window, diagnostics=_healthy_runtime_diagnostics(
        gate_chatter_event_count=1, input_callback_error_count=1,
    ))

    assert chip.state == "warn" and _nothing_dropped(chip.text) and "fault" in chip.text


def test_the_drops_row_never_claims_attention_for_zero_drops(window, monkeypatch):
    from mic_eq.ui.health import advice_for

    clock = [10.0]
    monkeypatch.setattr("mic_eq.ui.health.time.monotonic", lambda: clock[0])
    _recovery_setup(window)
    chip = window.shell_texts["dropped_label"]
    _labels(window, diagnostics=_healthy_runtime_diagnostics(rt_error_name="input_queue_full"))
    assert chip.state == "warn"
    assert _nothing_dropped(chip.text) and "error" in chip.text and "fault" not in chip.text
    assert advice_for(chip.text) == ""
    assert "0 samples" not in chip.text

    # With real numbers the row keeps them and names the first cause.
    _labels(window, diagnostics=_healthy_runtime_diagnostics(
        input_dropped_samples=12, stream_restart_count=1, rt_error_name="input_queue_full",
    ))
    assert chip.text.startswith("Drops:")
    assert all(re.search(rf"\b{part}\b", chip.text) for part in ("12 samples", "0 dropouts", "1 restart"))
    assert "fault" in chip.text and "attention" in chip.text
    assert "12 samples so far" in advice_for(chip.text)

    clock[0] += 5.1  # the drops are no longer recent
    _labels(window, diagnostics=_healthy_runtime_diagnostics(input_dropped_samples=12, stream_restart_count=1))
    assert chip.state == "ok" and "attention" not in chip.text
    assert all(re.search(rf"\b{part}\b", chip.text) for part in ("12 samples", "0 dropouts", "1 restart"))


def test_health_panel_reports_gate_chatter_and_underruns(window):
    _recovery_setup(window)

    window._update_diagnostic_labels(
        diagnostics=_healthy_runtime_diagnostics(
            gate_chatter_event_count=1,
            output_underrun_total=2,
        ),
        latency_ms=24.0,
        dsp_time_ms=0.5,
        input_buf=128,
        output_buf=256,
        rnnoise_buf=0,
        input_rms_db=-18.0,
        output_rms_db=-18.0,
        input_callback_age_ms=10,
        output_callback_age_ms=10,
    )

    assert "Gate: CHATTER" in window.shell_texts["gate_health_label"].text
    assert window.shell_texts["gate_health_label"].state == "warn"
    assert window.shell_texts["underrun_health_label"].text == "Dropouts: 2"
    assert window.shell_texts["underrun_health_label"].state == "warn"


def test_input_channel_mode_change_persists_and_applies(window, monkeypatch):
    class _Processor:
        def __init__(self):
            self.modes: list[str] = []

        def set_input_channel_mode(self, mode: str):
            self.modes.append(mode)

    saved = []
    window.processor = _Processor()
    window.config.input_channel_mode = "average"
    window.compressor_state.set_settings({"noise_reference_reliability": 0.8})
    window.input_channel_mode_choice.setCurrentIndex(
        window.input_channel_mode_choice.findData("phase_safe_mono")
    )
    monkeypatch.setattr(
        main_window, "save_config", lambda cfg: saved.append(cfg) or True
    )

    window._on_input_channel_mode_changed()

    assert window.config.input_channel_mode == "phase_safe_mono"
    assert window.processor.modes == ["phase_safe_mono"]
    reliability = window.compressor_state.get_settings(include_calibration=True)
    assert reliability["noise_reference_reliability"] == 0.0
    assert saved == [window.config]


@pytest.mark.parametrize("save_failure", [False, OSError("config locked")])
def test_route_input_preference_reports_unsaved_failure(window, monkeypatch, save_failure):
    window.processor = Mock()
    window.input_choice.addItem(
        "Mic", DeviceIdentity(name="Mic", endpoint_id="input", direction="input")
    )
    window.output_choice.addItem(
        "Cable", DeviceIdentity(name="Cable", endpoint_id="output", direction="output")
    )
    window.input_channel_mode_choice.setCurrentIndex(
        window.input_channel_mode_choice.findData("left")
    )
    window.input_cleanup_mode_choice.setCurrentIndex(
        window.input_cleanup_mode_choice.findData("gentle")
    )
    route_key = window._current_device_route_key()
    assert route_key is not None
    previous = InputDevicePreference(channel_mode="average", cleanup_mode="off")
    window.config.route_input_preferences[route_key] = previous
    if isinstance(save_failure, Exception):
        monkeypatch.setattr(
            main_window,
            "save_config",
            Mock(side_effect=save_failure),
        )
    else:
        monkeypatch.setattr(
            main_window,
            "save_config",
            Mock(return_value=save_failure),
        )

    assert not window._save_current_route_input_preference()

    assert window.config.route_input_preferences[route_key] == previous
    assert "could not be saved" in window.status_message.text.lower()

    window._on_input_channel_mode_changed()
    window.processor.set_input_channel_mode.assert_called_with("left")
    assert "applied for this session" in window.status_message.text
    window._on_input_cleanup_mode_changed()
    window.processor.set_input_cleanup_mode.assert_called_with("gentle")
    assert "could not be saved" in window.status_message.text


def test_refresh_devices_defers_while_processing(window, monkeypatch):
    window.processor = Mock()
    window.processor.is_running.return_value = True
    window.input_choice.addItem(
        "Mic", DeviceIdentity(name="Mic", endpoint_id="input", direction="input")
    )
    window.output_choice.addItem(
        "Cable", DeviceIdentity(name="Cable", endpoint_id="output", direction="output")
    )
    monkeypatch.setattr(
        main_window,
        "list_input_devices",
        Mock(side_effect=AssertionError("refresh enumerated while running")),
    )
    monkeypatch.setattr(
        main_window,
        "list_output_devices",
        Mock(side_effect=AssertionError("refresh enumerated while running")),
    )

    window._refresh_devices()

    assert not window.refresh_btn_action.isEnabled()
    assert "stop processing" in window.status_message.text.lower()


def test_input_phase_warning_marks_diagnostics_warn(window):
    _recovery_setup(window)
    diagnostics = {
        "input_dropped_samples": 0,
        "lock_contention_count": 0,
        "suppressor_non_finite_count": 0,
        "stream_restart_count": 0,
        "output_underrun_total": 0,
        "output_underrun_streak": 0,
        "output_recovery_event_count": 0,
        "output_recovery_count": 0,
        "output_short_write_dropped_samples": 0,
        "rt_buffer_overflow_count": 0,
        "input_callback_error_count": 0,
        "output_callback_error_count": 0,
        "rt_error_name": "none",
        "input_phase_warning_count": 1,
        "input_stereo_correlation": -0.91,
        "noise_model": "rnnoise",
        "noise_backend_available": True,
        "noise_backend_failed": False,
        "noise_backend_error": None,
        "recovery_suppressed": False,
        "last_restart_reason": None,
    }

    window._update_diagnostic_labels(
        diagnostics=diagnostics,
        latency_ms=24.0,
        dsp_time_ms=0.5,
        input_buf=128,
        output_buf=512,
        rnnoise_buf=0,
    )

    # Nothing was dropped, so the row says what raised it instead of printing zeros.
    assert window.shell_texts["dropped_label"].text == (
        "Drops: nothing dropped, but the microphone channels cancel each other"
    )
    assert "PH:1" in window.shell_texts["dropped_label"].toolTip
    assert "COR:-0.91" in window.shell_texts["dropped_label"].toolTip
    assert window.shell_texts["dropped_label"].state == "warn"


def test_device_selection_policy_matches_exact_then_unique_name():
    identities = [
        DeviceIdentity(name="Mic A", is_default=False),
        DeviceIdentity(name="Mic B", is_default=True),
    ]

    assert (
        find_identity_index(identities, DeviceIdentity(name="Mic B", is_default=True))
        == 1
    )
    assert (
        find_identity_index(identities, DeviceIdentity(name="Mic B", is_default=False))
        == 1
    )
    assert (
        find_identity_index(
            identities, DeviceIdentity(name="Missing", is_default=False)
        )
        == -1
    )


def test_device_selection_policy_uses_endpoint_id_across_rename_and_rejects_ambiguity():
    identities = [
        DeviceIdentity(
            name="USB Audio",
            endpoint_id="endpoint-a",
            direction="input",
            name_ordinal=0,
        ),
        DeviceIdentity(
            name="USB Audio",
            endpoint_id="endpoint-b",
            direction="input",
            name_ordinal=1,
        ),
    ]

    assert (
        find_identity_index(
            identities,
            DeviceIdentity(
                name="Old Microphone Name",
                endpoint_id="endpoint-b",
                direction="input",
                name_ordinal=0,
            ),
        )
        == 1
    )
    assert find_identity_index(identities, DeviceIdentity(name="USB Audio")) == -1
    assert (
        find_identity_index(
            identities,
            DeviceIdentity(name="USB Audio", direction="input", name_ordinal=1),
        )
        == -1
    )


def test_endpoint_identity_does_not_fall_back_to_replacement_with_same_name():
    identities = [
        DeviceIdentity(
            name="USB Audio",
            endpoint_id="replacement-endpoint",
            direction="input",
        )
    ]

    assert (
        find_identity_index(
            identities,
            DeviceIdentity(
                name="USB Audio",
                endpoint_id="disconnected-endpoint",
                direction="input",
            ),
        )
        == -1
    )


def test_duplicate_name_without_endpoint_id_is_not_persistable():
    identities = [
        DeviceIdentity(name="USB Audio", direction="input", name_ordinal=0),
        DeviceIdentity(name="USB Audio", direction="input", name_ordinal=1),
    ]

    assert identity_is_persistable(identities, identities[0]) is False
    assert identity_is_persistable([identities[0]], identities[0]) is True


def test_start_processor_for_route_forwards_duplicate_name_ordinals():
    class Processor:
        def __init__(self):
            self.args = None
            self.kwargs = None

        def start(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            return "started"

    processor = Processor()
    result = start_processor_for_route(
        processor,
        DeviceIdentity(name="USB Audio", name_ordinal=1),
        DeviceIdentity(name="USB Audio", name_ordinal=2),
    )

    assert result == "started"
    assert processor.args == ("USB Audio", "USB Audio", 1, 2)
    assert processor.kwargs == {
        "input_device_endpoint_id": None,
        "output_device_endpoint_id": None,
    }


def test_start_processor_for_route_forwards_endpoint_ids():
    class Processor:
        def __init__(self):
            self.kwargs = None

        def start(self, *_args, **kwargs):
            self.kwargs = kwargs
            return "started"

    processor = Processor()
    result = start_processor_for_route(
        processor,
        DeviceIdentity(name="USB Audio", endpoint_id="input-endpoint"),
        DeviceIdentity(name="USB Audio", endpoint_id="output-endpoint"),
    )

    assert result == "started"
    assert processor.kwargs == {
        "input_device_endpoint_id": "input-endpoint",
        "output_device_endpoint_id": "output-endpoint",
    }


def test_start_processor_for_route_rejects_missing_output_device():
    class Processor:
        def start(self, *_args, **_kwargs):
            raise AssertionError("processor must not start without a destination")

    with pytest.raises(ValueError, match="destination"):
        start_processor_for_route(
            Processor(),
            DeviceIdentity(name="USB Microphone", direction="input"),
            None,
        )


def test_route_preset_binding_applies_only_to_exact_stable_route(window):
    input_identity = DeviceIdentity(
        name="Mic",
        endpoint_id="input-id",
        host_api="WASAPI",
        direction="input",
        name_ordinal=0,
    )
    output_identity = DeviceIdentity(
        name="Cable",
        endpoint_id="output-id",
        host_api="WASAPI",
        direction="output",
        name_ordinal=0,
    )
    window.input_choice.addItem("Mic", input_identity)
    window.output_choice.addItem("Cable", output_identity)
    route_key = build_device_route_key(input_identity, output_identity)
    window.config = AppConfig(
        device_preset_bindings={route_key: DevicePresetBinding("builtin:broadcast")}
    )
    loaded: list[str] = []
    window._load_device_preset_id = lambda preset_id: loaded.append(preset_id) or True

    assert window._apply_bound_preset_for_current_route() is True
    assert loaded == ["builtin:broadcast"]

    window.output_choice.clear()
    window.output_choice.addItem(
        "Other",
        DeviceIdentity(
            name="Other",
            endpoint_id="other-output",
            host_api="WASAPI",
            direction="output",
            name_ordinal=0,
        ),
    )
    assert window._apply_bound_preset_for_current_route() is False
    assert loaded == ["builtin:broadcast"]


def test_device_selection_policy_prefers_default_and_virtual_output():
    identities = [
        DeviceIdentity(name="Speakers", is_default=True),
        DeviceIdentity(name="VB-Audio Cable", is_default=False),
    ]

    assert default_device_index(identities) == 0
    assert preferred_output_index(identities) == 1


def test_refresh_devices_preserves_existing_selection(window, monkeypatch):
    window.input_choice.addItem("Mic A", DeviceIdentity(name="Mic A", is_default=False))
    window.input_choice.addItem("Mic B", DeviceIdentity(name="Mic B", is_default=True))
    window.output_choice.addItem("Out A", DeviceIdentity(name="Out A", is_default=False))
    window.output_choice.addItem("Out B", DeviceIdentity(name="Out B", is_default=True))
    window.config = AppConfig(
        last_input_device="Mic A",
        last_output_device="Out A",
        last_input_device_identity=DeviceIdentity(name="Mic A", is_default=False),
        last_output_device_identity=DeviceIdentity(name="Out A", is_default=False),
    )
    shown = _status_log(window)

    monkeypatch.setattr(
        main_window,
        "list_input_devices",
        lambda: [
            type("Dev", (), {"name": "Mic A", "is_default": False})(),
            type("Dev", (), {"name": "Mic B", "is_default": True})(),
        ],
    )
    monkeypatch.setattr(
        main_window,
        "list_output_devices",
        lambda: [
            type("Dev", (), {"name": "Out A", "is_default": False})(),
            type("Dev", (), {"name": "Out B", "is_default": True})(),
        ],
    )

    window._refresh_devices()

    assert window.output_choice.currentData() == DeviceIdentity(
        name="Out A", is_default=False, direction="output"
    )
    assert window.input_choice.currentData() == DeviceIdentity(
        name="Mic A", is_default=False, direction="input"
    )
    assert shown == []
    window.compressor_state.set_settings({"noise_reference_reliability": 0.8})
    window._refresh_devices()
    reliability = window.compressor_state.get_settings(include_calibration=True)
    assert reliability["noise_reference_reliability"] == 0.8


def test_refresh_devices_restores_all_control_signal_states(window, monkeypatch):
    window.input_choice.addItem("Mic A", DeviceIdentity(name="Mic A", is_default=True))
    window.output_choice.addItem("Out A", DeviceIdentity(name="Out A", is_default=True))
    window.input_choice.blockSignals(True)
    window.input_channel_mode_choice.blockSignals(True)
    window.config = AppConfig(
        last_input_device_identity=DeviceIdentity(name="Mic A"),
        last_output_device_identity=DeviceIdentity(name="Out A"),
        last_input_device="Mic A",
        last_output_device="Out A",
    )
    monkeypatch.setattr(
        main_window,
        "list_input_devices",
        lambda: [type("Dev", (), {"name": "Mic A", "is_default": True})()],
    )
    monkeypatch.setattr(
        main_window,
        "list_output_devices",
        lambda: [type("Dev", (), {"name": "Out A", "is_default": True})()],
    )

    window._refresh_devices()

    assert window.input_choice.signalsBlocked()
    assert not window.output_choice.signalsBlocked()
    assert window.input_channel_mode_choice.signalsBlocked()
    assert not window.input_cleanup_mode_choice.signalsBlocked()


def test_refresh_devices_preserves_missing_output_for_reconnect(window, monkeypatch):
    window.compressor_state.set_settings({"noise_reference_reliability": 0.8})
    window.input_choice.addItem("Mic A", DeviceIdentity(name="Mic A", is_default=True))
    window.output_choice.addItem(
        "Out Old", DeviceIdentity(name="Out Old", is_default=False)
    )
    window.config = AppConfig(
        last_input_device="Mic A",
        last_output_device="Out Old",
        last_input_device_identity=DeviceIdentity(name="Mic A", is_default=True),
        last_output_device_identity=DeviceIdentity(name="Out Old", is_default=False),
    )
    shown = _status_log(window)

    monkeypatch.setattr(
        main_window,
        "list_input_devices",
        lambda: [type("Dev", (), {"name": "Mic A", "is_default": True})()],
    )
    monkeypatch.setattr(
        main_window,
        "list_output_devices",
        lambda: [type("Dev", (), {"name": "Out New", "is_default": True})()],
    )

    window._refresh_devices()

    reliability = window.compressor_state.get_settings(include_calibration=True)
    assert reliability["noise_reference_reliability"] == 0.0
    assert window.config.last_output_device == "Out Old"
    assert window.config.last_output_device_identity == DeviceIdentity(
        name="Out Old", is_default=False
    )
    assert window.output_choice.currentData() is None
    assert any("Destination 'Out Old' is disconnected" in message for message in shown)


@pytest.mark.parametrize("write_failure", [False, OSError("config locked")])
def test_startup_preset_write_failure_keeps_previous_selection(window, monkeypatch, write_failure):
    previous = _startup_builtin_id("flat")
    window.config.startup_preset = previous
    window._update_startup_preset_menu()
    save = Mock(
        side_effect=write_failure if isinstance(write_failure, Exception) else None,
        return_value=write_failure,
    )
    monkeypatch.setattr(main_window, "save_config", save)
    actions = {
        action.data(): action
        for action in window._startup_preset_menu.actions()
        if action.isCheckable()
    }

    actions[_startup_builtin_id("voice")].trigger()

    assert window.config.startup_preset == previous
    assert "previous selection kept" in window.status_message.text
    assert [data for data, action in actions.items() if action.isChecked()] == [previous]


@pytest.mark.parametrize("write_failure", [False, OSError("config locked")])
def test_refresh_device_write_failure_keeps_live_route_and_reports_session_state(
    window, monkeypatch, write_failure
):
    window.config = AppConfig(
        last_input_device_identity=DeviceIdentity(name="Mic", direction="input"),
        last_output_device_identity=DeviceIdentity(name="Cable", direction="output"),
    )
    monkeypatch.setattr(
        main_window,
        "list_input_devices",
        lambda: [
            SimpleNamespace(
                name="Mic",
                is_default=True,
                endpoint_id="input-id",
                host_api="WASAPI",
                sample_rate=48_000,
                channels=1,
                name_ordinal=0,
            )
        ],
    )
    monkeypatch.setattr(
        main_window,
        "list_output_devices",
        lambda: [
            SimpleNamespace(
                name="Cable",
                is_default=True,
                endpoint_id="output-id",
                host_api="WASAPI",
                sample_rate=48_000,
                channels=2,
                name_ordinal=0,
            )
        ],
    )
    save = Mock(
        side_effect=write_failure if isinstance(write_failure, Exception) else None,
        return_value=write_failure,
    )
    monkeypatch.setattr(main_window, "save_config", save)

    window._refresh_devices()

    assert window.input_choice.currentData().endpoint_id == "input-id"
    assert window.output_choice.currentData().endpoint_id == "output-id"
    assert "applied for this session" in window.status_message.text
    assert "could not be saved" in window.status_message.text


@pytest.mark.parametrize("write_failure", [False, OSError("config locked")])
def test_restore_write_failure_keeps_canonicalized_route_in_session(
    window, monkeypatch, write_failure
):
    input_identity = DeviceIdentity(
        name="Mic", endpoint_id="input-id", direction="input"
    )
    output_identity = DeviceIdentity(
        name="Cable", endpoint_id="output-id", direction="output"
    )
    window.input_choice.addItem("Mic", input_identity)
    window.output_choice.addItem("Cable", output_identity)
    window.config = AppConfig(
        last_input_device_identity=DeviceIdentity(name="Mic", direction="input"),
        last_output_device_identity=DeviceIdentity(name="Cable", direction="output"),
    )
    save = Mock(
        side_effect=write_failure if isinstance(write_failure, Exception) else None,
        return_value=write_failure,
    )
    monkeypatch.setattr(main_window, "save_config", save)

    window._restore_from_config()

    assert window.config.last_input_device_identity == input_identity
    assert window.config.last_output_device_identity == output_identity
    assert "applied for this session" in window.status_message.text
    assert "could not be saved" in window.status_message.text


@pytest.mark.parametrize("write_failure", [False, OSError("config locked")])
def test_device_change_write_failure_keeps_active_route(window, monkeypatch, write_failure):
    input_identity = DeviceIdentity(
        name="Mic", endpoint_id="input-id", direction="input"
    )
    output_identity = DeviceIdentity(
        name="Cable", endpoint_id="output-id", direction="output"
    )
    window.input_choice.addItem("Mic", input_identity)
    window.output_choice.addItem("Cable", output_identity)
    save = Mock(
        side_effect=write_failure if isinstance(write_failure, Exception) else None,
        return_value=write_failure,
    )
    monkeypatch.setattr(main_window, "save_config", save)

    window._on_device_changed()

    assert window.config.last_input_device_identity == input_identity
    assert window.config.last_output_device_identity == output_identity
    assert "applied for this session" in window.status_message.text
    assert "could not be saved" in window.status_message.text


def test_latency_profile_key_uses_structured_device_identity(window):
    window.input_choice.addItem("Mic A", DeviceIdentity(name="Mic A", is_default=True))
    window.output_choice.addItem("Out B", DeviceIdentity(name="Out B", is_default=False))
    key = window._latency_profile_key()

    assert key == build_device_route_key(
        DeviceIdentity(name="Mic A", is_default=True),
        DeviceIdentity(name="Out B", is_default=False),
    )
    assert key != legacy_latency_profile_key("Mic A", "Out B")


def test_device_name_coerces_identity_to_name():
    assert device_name(DeviceIdentity(name="Mic A", is_default=False)) == "Mic A"
    assert device_name("Out B") == "Out B"
    assert device_name(None) is None


def test_update_diagnostics_surfaces_output_recovery_and_reuses_diagnostics(window):
    _recovery_setup(window)
    shown = _status_log(window)

    window._update_diagnostics()

    assert window.processor.diagnostics_calls == 1
    assert "4 output recoveries" in window.shell_texts["recovery_diag_label"].text
    assert "2 restarts" in window.shell_texts["recovery_diag_label"].text
    assert "4 restarts" not in window.shell_texts["dropped_label"].text
    assert "RTO:8" in window.shell_texts["dropped_label"].toolTip
    assert "ICE:9" in window.shell_texts["dropped_label"].toolTip
    assert "OCE:10" in window.shell_texts["dropped_label"].toolTip
    assert "RT:fixed real-time buffer overflow" in window.shell_texts["dropped_label"].toolTip
    assert not shown


def test_stopped_diagnostics_continue_native_recovery_and_reconcile_controls(window):
    processor = _StoppedRecoveryProcessor()
    _recovery_setup(window, processor)

    window._update_diagnostics()

    assert processor.diagnostics_calls == 0
    assert processor.recovery_calls == 1
    assert window.start_btn_action.isEnabled()
    assert not window.stop_btn_action.isEnabled()

    window._update_diagnostics()

    assert processor.recovery_calls == 2
    assert processor.is_running() is True
    assert not window.start_btn_action.isEnabled()
    assert window.stop_btn_action.isEnabled()


def _poll_dropped_selections(window, processor, count) -> str:
    """One diagnostics tick with the engine reporting ``count`` dropped selections."""
    processor._diagnostics["noise_selection_dropped_count"] = count
    window._update_diagnostics()
    return window.status_message.text


def test_a_dropped_noise_selection_warns_once_and_says_what_to_do(window):
    processor = _MeterProcessor()
    _recovery_setup(window, processor)
    window.status_message.clearMessage()

    assert _poll_dropped_selections(window, processor, 0) == ""
    assert _poll_dropped_selections(window, processor, 1) == NOISE_SELECTION_DROPPED_WARNING
    for expected in ("could not be applied", "previous one is still running",
                     "Choose it again", "Help > Export Diagnostics"):
        assert expected in NOISE_SELECTION_DROPPED_WARNING

    # The same count is not a new drop: nothing comes back once the message has gone.
    window.status_message.clearMessage()
    assert _poll_dropped_selections(window, processor, 1) == ""

    # The next drop warns again.
    assert _poll_dropped_selections(window, processor, 2) == NOISE_SELECTION_DROPPED_WARNING


def test_a_count_that_is_already_nonzero_does_not_warn_at_the_first_poll(window):
    processor = _MeterProcessor()
    processor._diagnostics["noise_selection_dropped_count"] = 3
    _recovery_setup(window, processor)
    window.status_message.clearMessage()

    assert window.status_message.text == ""
    assert _poll_dropped_selections(window, processor, 3) == ""
    assert _poll_dropped_selections(window, processor, 3) == ""
    assert _poll_dropped_selections(window, processor, 4) == NOISE_SELECTION_DROPPED_WARNING


def test_a_counter_that_starts_over_does_not_warn_until_it_grows_again(window):
    processor = _MeterProcessor()
    _recovery_setup(window, processor)
    _poll_dropped_selections(window, processor, 5)
    window.status_message.clearMessage()

    assert _poll_dropped_selections(window, processor, 0) == ""
    assert _poll_dropped_selections(window, processor, 1) == NOISE_SELECTION_DROPPED_WARNING


def test_the_window_watches_the_count_the_engine_exports(window):
    # A fresh engine counts from zero, and the window starts from that.
    assert window.processor.get_runtime_diagnostics()["noise_selection_dropped_count"] == 0
    assert window._noise_drops_seen == (window.processor, 0)


def test_the_dropped_selection_warning_is_announced_to_assistive_technology(window, monkeypatch):
    announced: list[str] = []
    monkeypatch.setattr(QAccessible, "isActive", staticmethod(lambda: True))
    monkeypatch.setattr(
        QAccessible, "updateAccessibility",
        staticmethod(lambda event: announced.append(event.message())),
    )
    processor = _MeterProcessor()
    _recovery_setup(window, processor)
    window.status_message.clearMessage()

    _poll_dropped_selections(window, processor, 0)
    assert NOISE_SELECTION_DROPPED_WARNING not in announced
    _poll_dropped_selections(window, processor, 1)
    assert announced.count(NOISE_SELECTION_DROPPED_WARNING) == 1


def test_stale_output_underrun_and_recovery_totals_do_not_warn(window, monkeypatch):
    monkeypatch.setattr("mic_eq.ui.health.time.monotonic", lambda: 10.0)
    _recovery_setup(window)
    diagnostics = {
        "input_dropped_samples": 0,
        "lock_contention_count": 0,
        "suppressor_non_finite_count": 0,
        "stream_restart_count": 0,
        "output_underrun_total": 4,
        "output_underrun_streak": 0,
        "output_retime_adjustment_count": 606,
        "output_recovery_event_count": 0,
        "output_recovery_count": 0,
        "output_short_write_dropped_samples": 0,
        "rt_buffer_overflow_count": 0,
        "input_callback_error_count": 0,
        "output_callback_error_count": 0,
        "rt_error_name": "none",
        "input_backlog_recovery_count": 0,
        "input_backlog_dropped_samples": 0,
        "clip_event_count": 0,
        "clip_peak_db": -120.0,
        "input_resampler_active": False,
        "output_resampler_active": False,
        "noise_model": "deepfilter",
        "noise_backend_available": True,
        "noise_backend_failed": False,
        "noise_backend_error": None,
        "recovery_suppressed": False,
        "last_restart_reason": None,
    }
    window._stream_health = RecentStreamHealth()
    window._stream_health.observe(diagnostics, now=0.0)
    window._last_output_underrun_total = 4

    window._update_diagnostic_labels(
        diagnostics=diagnostics,
        latency_ms=64.0,
        dsp_time_ms=0.6,
        input_buf=512,
        output_buf=512,
        rnnoise_buf=0,
    )

    assert "4 dropouts" in window.shell_texts["dropped_label"].text
    assert "606" not in window.shell_texts["dropped_label"].text
    assert window.shell_texts["dropped_label"].state == "ok"
    assert window.shell_texts["recovery_diag_label"].state == "ok"


def test_new_output_underrun_total_warns_for_stability_window(window, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("mic_eq.ui.health.time.monotonic", lambda: clock[0])
    _recovery_setup(window)
    diagnostics = {
        "input_dropped_samples": 0,
        "lock_contention_count": 0,
        "suppressor_non_finite_count": 0,
        "stream_restart_count": 0,
        "output_underrun_total": 5,
        "output_underrun_streak": 0,
        "output_retime_adjustment_count": 0,
        "output_recovery_event_count": 0,
        "output_recovery_count": 0,
        "output_short_write_dropped_samples": 0,
        "rt_buffer_overflow_count": 0,
        "input_callback_error_count": 0,
        "output_callback_error_count": 0,
        "rt_error_name": "none",
        "noise_model": "rnnoise",
        "noise_backend_available": True,
        "noise_backend_failed": False,
        "noise_backend_error": None,
        "recovery_suppressed": False,
        "last_restart_reason": None,
    }
    window._last_output_underrun_total = 4

    window._update_diagnostic_labels(
        diagnostics=diagnostics,
        latency_ms=64.0,
        dsp_time_ms=0.6,
        input_buf=512,
        output_buf=512,
        rnnoise_buf=0,
    )

    assert window.shell_texts["dropped_label"].state == "warn"
    clock[0] += 5.1

    window._update_diagnostic_labels(
        diagnostics=diagnostics,
        latency_ms=64.0,
        dsp_time_ms=0.6,
        input_buf=512,
        output_buf=512,
        rnnoise_buf=0,
    )

    assert window.shell_texts["dropped_label"].state == "ok"


def test_the_output_clip_counter_is_a_support_counter_and_does_not_warn(window):
    _recovery_setup(window)
    diagnostics = {
        "input_dropped_samples": 0,
        "lock_contention_count": 0,
        "suppressor_non_finite_count": 0,
        "stream_restart_count": 0,
        "output_underrun_total": 0,
        "output_underrun_streak": 0,
        "output_retime_adjustment_count": 0,
        "output_recovery_event_count": 0,
        "output_recovery_count": 0,
        "output_short_write_dropped_samples": 0,
        "rt_buffer_overflow_count": 0,
        "input_callback_error_count": 0,
        "output_callback_error_count": 0,
        "rt_error_name": "none",
        "clip_event_count": 0,
        "clip_peak_db": -120.0,
        "output_clip_event_count": 1,
        "output_clip_peak_db": -0.25,
        "limiter_effective_ceiling_db": -1.5,
        "noise_model": "rnnoise",
        "noise_backend_available": True,
        "noise_backend_failed": False,
        "noise_backend_error": None,
        "recovery_suppressed": False,
        "last_restart_reason": None,
    }

    window._update_diagnostic_labels(
        diagnostics=diagnostics,
        latency_ms=32.0,
        dsp_time_ms=0.4,
        input_buf=256,
        output_buf=256,
        rnnoise_buf=0,
    )

    assert "OCL:1" in window.shell_texts["dropped_label"].toolTip
    assert "OPK:-0.2" in window.shell_texts["dropped_label"].toolTip
    assert "LIM:-1.5" in window.shell_texts["dropped_label"].toolTip
    assert window.shell_texts["dropped_label"].state == "ok"


def test_new_output_true_peak_event_warns_once(window):
    _recovery_setup(window)
    diagnostics = {
        "input_dropped_samples": 0,
        "lock_contention_count": 0,
        "suppressor_non_finite_count": 0,
        "stream_restart_count": 0,
        "output_underrun_total": 0,
        "output_underrun_streak": 0,
        "output_retime_adjustment_count": 0,
        "output_recovery_event_count": 0,
        "output_recovery_count": 0,
        "output_short_write_dropped_samples": 0,
        "rt_buffer_overflow_count": 0,
        "input_callback_error_count": 0,
        "output_callback_error_count": 0,
        "rt_error_name": "none",
        "clip_event_count": 0,
        "clip_peak_db": -120.0,
        "output_clip_event_count": 0,
        "output_clip_peak_db": -120.0,
        "output_true_peak_event_count": 1,
        "output_true_peak_db": 0.4,
        "limiter_effective_ceiling_db": -1.5,
        "noise_model": "rnnoise",
        "noise_backend_available": True,
        "noise_backend_failed": False,
        "noise_backend_error": None,
        "recovery_suppressed": False,
        "last_restart_reason": None,
    }

    window._update_diagnostic_labels(
        diagnostics=diagnostics,
        latency_ms=32.0,
        dsp_time_ms=0.4,
        input_buf=256,
        output_buf=256,
        rnnoise_buf=0,
    )

    assert "Output: PEAK ABOVE CEILING" in window.shell_texts["output_health_label"].text
    assert window.shell_texts["output_health_label"].state == "warn"
    assert "OTP:1" in window.shell_texts["dropped_label"].toolTip
    assert "TPK:0.4" in window.shell_texts["dropped_label"].toolTip
    assert window.shell_texts["dropped_label"].state == "warn"

    window._update_diagnostic_labels(
        diagnostics=diagnostics,
        latency_ms=32.0,
        dsp_time_ms=0.4,
        input_buf=256,
        output_buf=256,
        rnnoise_buf=0,
        output_rms_db=-18.0,
    )

    assert "Output: OK" in window.shell_texts["output_health_label"].text
    assert window.shell_texts["dropped_label"].state == "ok"


def test_new_gate_chatter_event_warns_for_stability_window(window, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("mic_eq.ui.health.time.monotonic", lambda: clock[0])
    _recovery_setup(window)
    diagnostics = {
        "input_dropped_samples": 0,
        "lock_contention_count": 0,
        "suppressor_non_finite_count": 0,
        "stream_restart_count": 0,
        "output_underrun_total": 0,
        "output_underrun_streak": 0,
        "output_retime_adjustment_count": 0,
        "output_recovery_event_count": 0,
        "output_recovery_count": 0,
        "output_short_write_dropped_samples": 0,
        "rt_buffer_overflow_count": 0,
        "input_callback_error_count": 0,
        "output_callback_error_count": 0,
        "rt_error_name": "none",
        "clip_event_count": 0,
        "clip_peak_db": -120.0,
        "output_clip_event_count": 0,
        "output_clip_peak_db": -120.0,
        "gate_chatter_event_count": 1,
        "noise_model": "rnnoise",
        "noise_backend_available": True,
        "noise_backend_failed": False,
        "noise_backend_error": None,
        "recovery_suppressed": False,
        "last_restart_reason": None,
    }

    window._update_diagnostic_labels(
        diagnostics=diagnostics,
        latency_ms=32.0,
        dsp_time_ms=0.4,
        input_buf=256,
        output_buf=256,
        rnnoise_buf=0,
    )

    assert "GCH:1" in window.shell_texts["dropped_label"].toolTip
    assert window.shell_texts["dropped_label"].state == "warn"
    clock[0] += 5.1

    window._update_diagnostic_labels(
        diagnostics=diagnostics,
        latency_ms=32.0,
        dsp_time_ms=0.4,
        input_buf=256,
        output_buf=256,
        rnnoise_buf=0,
    )

    assert window.shell_texts["dropped_label"].state == "ok"


def test_startup_preset_ids_normalize_builtin_and_custom_legacy_names():
    assert _normalize_startup_preset_id("Voice Clarity") == _startup_builtin_id("voice")
    assert _normalize_startup_preset_id("voice") == _startup_builtin_id("voice")
    assert _normalize_startup_preset_id(
        "Custom Voice", ("Custom Voice",)
    ) == _startup_custom_id("Custom Voice")
    assert _normalize_startup_preset_id(
        _startup_builtin_id("flat")
    ) == _startup_builtin_id("flat")


def test_apply_preset_passes_advanced_compressor_fields(window):
    preset = Preset(
        name="Compressor Advanced",
        compressor=CompressorSettings(
            enabled=True,
            threshold_db=-18.0,
            ratio=3.0,
            attack_ms=8.0,
            release_ms=150.0,
            makeup_gain_db=2.0,
            adaptive_release=True,
            base_release_ms=75.0,
            auto_makeup_enabled=True,
            target_lufs=-16.0,
            sidechain_highpass_enabled=False,
        ),
        limiter=LimiterSettings(careful_output_enabled=False),
    )

    window._write_processing_configuration(preset)

    assert window.gate_state.get_settings() == asdict(preset.gate)
    assert window.eq_state.get_eq_settings().to_dict() == preset.eq.to_dict()
    assert window.deesser_state.get_settings() == asdict(preset.deesser)
    assert window.compressor_state.get_settings() == asdict(preset.compressor)
    assert window.compressor_state.get_settings()["adaptive_release"] is True
    assert window.compressor_state.get_settings()["base_release_ms"] == 75.0
    assert window.compressor_state.get_settings()["auto_makeup_enabled"] is True
    assert window.compressor_state.get_settings()["target_lufs"] == -16.0
    assert (
        window.compressor_state.get_settings()["sidechain_highpass_enabled"]
        is False
    )
    assert window.limiter_state.get_settings()["careful_output_enabled"] is False


def test_configuration_writer_rejects_model_load_failure_before_other_state_edits(window):
    def set_noise_model(model):
        if model == "deepfilter":
            raise RuntimeError("test model load failure")
        return True

    processor = Mock()
    processor.list_noise_models.return_value = [
        ("rnnoise", "RNNoise"),
        ("deepfilter", "DeepFilter"),
    ]
    processor.set_noise_model.side_effect = set_noise_model
    window.noise_suppression_state = NoiseSuppressionState(processor, window)
    _use_processor(window, processor)

    preset = Preset()
    preset.rnnoise.model = "deepfilter"

    with pytest.raises(RuntimeError):
        window._write_processing_configuration(preset)
    assert window.noise_suppression_state.get_settings()["model"] == "rnnoise"
    # Only the noise model's own writes (and its rollback) reached the processor.
    assert {name for name, _args, _kwargs in processor.mock_calls} <= {
        "list_noise_models", "set_noise_model", "set_rnnoise_enabled", "set_rnnoise_strength",
    }


def _configuration_processor():
    """A processor that accepts every native write; tests read its call order."""
    processor = Mock()
    processor.is_running.return_value = False
    processor.get_deesser_high_cut_hz.return_value = 0.0
    return processor


_WRITES = ("set_", "apply_")


def _writes_between_mutes(calls):
    """The native writes made after the configuration mute and before its release."""
    mute, unmute = call.set_output_mute(True), call.set_output_mute(False)
    assert mute in calls and unmute in calls
    start = calls.index(mute)
    end = len(calls) - 1 - calls[::-1].index(unmute)
    writes = [c for c in calls[start + 1:end] if c[0].startswith(_WRITES)]
    assert not [c for c in calls[end + 1:] if c[0].startswith(_WRITES)]
    return writes


def test_configuration_finishes_rate_limited_native_writes_before_unmute(window, monkeypatch):
    _freeze_throttle_clock(monkeypatch)
    processor = _configuration_processor()
    _use_processor(window, processor)
    defaults = Preset()
    states = (
        (window.gate_state, "threshold_db", processor.apply_gate_settings,
         defaults.gate.threshold_db),
        (window.deesser_state, "threshold_db", processor.apply_deesser_settings,
         defaults.deesser.threshold_db),
        (window.compressor_state, "threshold_db", processor.apply_compressor_settings,
         defaults.compressor.threshold_db),
        (window.limiter_state, "ceiling_db", processor.set_limiter_ceiling,
         defaults.limiter.ceiling_db),
    )
    for state, name, _write, _default in states:
        state.set_value(name, -11.0)  # applied at once
        state.set_value(name, -12.0)  # waits behind the throttle
        assert state._rate_limiter._timer.isActive()
    queued = len(processor.mock_calls)

    window.apply_processing_configuration(defaults)

    writes = _writes_between_mutes(processor.mock_calls[queued:])
    assert {name for name, _args, _kwargs in writes} >= {
        "apply_gate_settings", "apply_deesser_settings",
        "apply_compressor_settings", "set_limiter_ceiling",
    }
    for state, name, write, default in states:
        # The last write carries the configured value, not the queued -12.0.
        last = write.call_args.args[0]
        assert (last[name] if isinstance(last, dict) else last) == default
        assert not state._rate_limiter._timer.isActive()


def test_setter_failure_rolls_back_before_unmute(window):
    processor = _configuration_processor()
    failed = []

    def apply_settings(settings):
        if settings["threshold_db"] == -11.0 and not failed:
            failed.append(settings["threshold_db"])
            raise RuntimeError("de-esser setter failed")

    processor.apply_deesser_settings.side_effect = apply_settings
    _use_processor(window, processor)
    candidate = Preset()
    candidate.deesser.threshold_db = -11.0

    with pytest.raises(RuntimeError, match="de-esser setter failed"):
        window.apply_processing_configuration(candidate)

    writes = _writes_between_mutes(processor.mock_calls)
    thresholds = [
        c.args[0]["threshold_db"] for c in writes if c[0] == "apply_deesser_settings"
    ]
    # The rejected write comes first, and the last write restores the old value.
    assert thresholds[0] == -11.0
    assert thresholds[-1] == -28.0


def test_configuration_cancels_superseded_eq_edits_before_snapshot_and_unmute(window, monkeypatch):
    _freeze_throttle_clock(monkeypatch)
    processor = _configuration_processor()
    _use_processor(window, processor)
    window.eq_state.set_band_value(0, "gain_db", 3.0)  # applied at once
    window.eq_state.set_band_value(0, "gain_db", 4.0)  # waits behind the throttle
    assert window.eq_state._rate_limiter._timer.isActive()
    queued = len(processor.mock_calls)

    window.apply_processing_configuration(Preset())

    writes = _writes_between_mutes(processor.mock_calls[queued:])
    assert [name for name, _args, _kwargs in writes].count("apply_eq_layers") == 1
    assert window.eq_state.get_eq_settings().bands[0].gain_db == 0.0
    assert not window.eq_state._rate_limiter._timer.isActive()


class _LatencyProcessor:
    def __init__(self, running: bool, fail_recording: bool = True):
        self.running = running
        self.fail_recording = fail_recording
        self.started = 0
        self.stopped = 0
        self.recovery_suppressed: list[bool] = []

    def is_running(self):
        return self.running

    def start(
        self,
        _input_device=None,
        _output_device=None,
        _input_device_name_ordinal=0,
        _output_device_name_ordinal=0,
        input_device_endpoint_id=None,
        output_device_endpoint_id=None,
    ):
        self.running = True
        self.started += 1

    def stop(self):
        self.running = False
        self.stopped += 1

    def output_sample_rate(self):
        return 48_000

    def set_recovery_suppressed(self, value):
        self.recovery_suppressed.append(bool(value))

    def start_raw_recording(self, _duration, *, before_cleanup=False):
        if self.fail_recording:
            raise RuntimeError("recording setup failed")

    def stop_raw_recording(self):
        return None

    def set_output_mute(self, _muted):
        return None


def _latency_window(window, processor):
    _use_processor(window, processor)
    window.input_choice.addItem("Mic", DeviceIdentity(name="Mic", is_default=False))
    window.output_choice.addItem("Out", DeviceIdentity(name="Out", is_default=False))
    return window


def test_latency_calibration_failure_stops_owned_processor(window):
    processor = _LatencyProcessor(running=False)
    owner = _latency_window(window, processor)
    dialog = LatencyCalibrationDialog(owner)

    dialog._on_run_clicked()

    assert processor.started == 1
    assert processor.stopped == 1
    assert not processor.running


def test_latency_calibration_respects_user_mute(window, monkeypatch):
    processor = _LatencyProcessor(running=False)
    processor.set_output_mute = Mock()
    owner = _latency_window(window, processor)
    owner.user_muted = True
    notice = Mock()
    monkeypatch.setattr(QMessageBox, "information", notice)
    dialog = LatencyCalibrationDialog(owner)

    dialog._on_run_clicked()
    assert processor.started == 0
    notice.assert_called_once()
    dialog.reject()
    processor.set_output_mute.assert_not_called()


def test_latency_calibration_failure_does_not_stop_preexisting_processor(window):
    processor = _LatencyProcessor(running=True)
    owner = _latency_window(window, processor)
    dialog = LatencyCalibrationDialog(owner)

    dialog._on_run_clicked()

    assert processor.started == 0
    assert processor.stopped == 0
    assert processor.running


def test_voice_setup_dynamics_intensity_controls_custom_targets(qapp):
    dialog = VoiceSetupDialog()

    assert dialog.dynamics_combo.currentData() == "balanced"
    assert dialog.custom_p95_spin.isEnabled() is False
    assert dialog.custom_peak_spin.isEnabled() is False

    dialog.dynamics_combo.setCurrentIndex(dialog.dynamics_combo.findData("custom"))
    qapp.processEvents()

    assert dialog.custom_p95_spin.isEnabled() is True
    assert dialog.custom_peak_spin.isEnabled() is True
    dialog.close()


def test_start_applies_persisted_mute_before_and_after_native_start(
    qapp, monkeypatch, tmp_path
) -> None:
    from mic_eq.config_parts import shared
    from mic_eq.ui import main_window

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [])

    window = MainWindow()
    events: list[tuple[str, bool | None]] = []

    class Processor:
        running = False

        def is_running(self) -> bool:
            return self.running

        def set_output_mute(self, muted: bool) -> None:
            events.append(("mute", muted))

        def stop(self) -> None:
            self.running = False

    processor = Processor()
    cast(Any, window).processor = processor
    window.output_choice.addItem("Destination", DeviceIdentity(name="Destination", direction="output"))
    window.user_muted = True
    window._temporary_mute_reasons = set()
    window._apply_input_preferences_for_current_route = lambda: None

    def start(_processor: Processor, *_devices: object) -> str:
        events.append(("start", None))
        processor.running = True
        return "started"

    monkeypatch.setattr(main_window, "start_processor_for_route", start)
    try:
        window._start_processing()
        assert events == [("mute", True), ("start", None), ("mute", True)]
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_start_refuses_retained_configuration_mute() -> None:
    owner = cast(Any, MainWindow.__new__(MainWindow))
    owner.processor = SimpleNamespace(is_running=lambda: False, start=Mock())
    owner._temporary_mute_reasons = {"configuration"}
    owner.status_message = Mock()
    owner._sync_processing_controls = Mock()

    MainWindow._start_processing(owner)

    owner.processor.start.assert_not_called()
    owner._sync_processing_controls.assert_called_once_with()
    assert "reload the preset" in owner.status_message.showMessage.call_args.args[0]


def test_user_mute_survives_temporary_owner_release() -> None:
    calls: list[bool] = []
    owner = cast(Any, MainWindow.__new__(MainWindow))
    owner.processor = SimpleNamespace(set_output_mute=calls.append)
    owner.user_muted = True
    owner._temporary_mute_reasons = set()
    owner._update_session_summary = lambda: None

    MainWindow.set_temporary_output_mute(owner, True, "calibration")
    MainWindow.set_temporary_output_mute(owner, False, "calibration")

    assert calls == [True, True]


@pytest.mark.parametrize("checked,temporary", [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize("write_error", [False, True])
def test_user_mute_applies_before_failed_persistence(
    monkeypatch, checked: bool, temporary: bool, write_error: bool
) -> None:
    calls: list[bool] = []
    owner = cast(Any, MainWindow.__new__(MainWindow))
    owner.processor = SimpleNamespace(set_output_mute=calls.append)
    owner.config = SimpleNamespace(user_muted=not checked)
    owner._temporary_mute_reasons = {"calibration"} if temporary else set()
    owner._update_session_summary = lambda: None
    owner.status_message = Mock()

    def save(_config) -> bool:
        assert calls == [checked or temporary]
        if write_error:
            raise OSError("config is not writable")
        return False

    monkeypatch.setattr("mic_eq.ui.main_window.save_config", save)
    MainWindow._on_user_mute_toggled(owner, checked)

    assert owner.user_muted is checked
    assert owner.config.user_muted is checked
    assert calls == [checked or temporary]
    message = owner.status_message.showMessage.call_args.args[0]
    assert "preference could not be saved" in message
    assert ("Output muted" if checked or temporary else "Output unmuted") in message


def test_user_mute_does_not_claim_native_failure_is_muted(monkeypatch) -> None:
    owner = cast(Any, MainWindow.__new__(MainWindow))
    owner.processor = SimpleNamespace(set_output_mute=Mock(side_effect=RuntimeError("unavailable")))
    owner.config = SimpleNamespace(user_muted=False)
    owner._temporary_mute_reasons = set()
    owner._update_session_summary = lambda: None
    owner.status_message = Mock()
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)

    MainWindow._on_user_mute_toggled(owner, True)

    message = owner.status_message.showMessage.call_args.args[0]
    assert "could not be applied" in message
    assert "preference saved" in message


def test_failed_mute_application_stays_pending() -> None:
    owner = cast(Any, MainWindow.__new__(MainWindow))

    def fail(_muted: bool) -> None:
        raise RuntimeError("processor unavailable")

    owner.processor = SimpleNamespace(set_output_mute=fail)
    owner.user_muted = True
    owner._temporary_mute_reasons = set()
    owner._update_session_summary = lambda: None

    MainWindow._apply_output_mute(owner)

    assert owner._output_mute_error == "unavailable"


def test_legacy_global_modes_without_stable_identity_use_safe_defaults() -> None:
    config = AppConfig.from_dict(
        {
            "last_input_device": "Old mic",
            "input_channel_mode": "left",
            "input_cleanup_mode": "gentle",
        }
    )

    assert config.input_device_preferences == {}
    owner = cast(Any, MainWindow.__new__(MainWindow))
    owner.config = config
    owner._current_device_route_key = lambda: None
    owner._current_input_preference_key = lambda: None
    assert MainWindow._input_preference_for_current_route(owner) == InputDevicePreference()


def test_missing_route_override_does_not_fall_back_to_global_modes() -> None:
    owner = cast(Any, MainWindow.__new__(MainWindow))
    owner.config = AppConfig(
        input_channel_mode="left",
        input_cleanup_mode="gentle",
        route_input_preferences={"another-route": InputDevicePreference("right", "strong")},
    )
    owner._current_device_route_key = lambda: "missing-route"
    owner._current_input_preference_key = lambda: None

    assert MainWindow._input_preference_for_current_route(owner) == InputDevicePreference()


def test_new_input_endpoint_does_not_inherit_another_device_preference() -> None:
    old_identity = DeviceIdentity(
        name="Mic",
        endpoint_id="old-endpoint",
        direction="input",
    )
    new_identity = DeviceIdentity(
        name="Mic",
        endpoint_id="new-endpoint",
        direction="input",
    )
    owner = cast(Any, MainWindow.__new__(MainWindow))
    owner.config = AppConfig(
        input_device_preferences={
            build_input_device_preference_key(old_identity): InputDevicePreference(
                "left", "gentle"
            )
        }
    )
    owner._current_device_route_key = lambda: None
    owner._current_input_preference_key = lambda: build_input_device_preference_key(
        new_identity
    )

    assert MainWindow._input_preference_for_current_route(owner) == InputDevicePreference()


def test_route_input_preference_action_saves_visible_controls_for_current_route(
    qapp, monkeypatch, tmp_path
) -> None:
    from mic_eq.config_parts import shared
    from mic_eq.ui import main_window

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [
        SimpleNamespace(
            name="Mic",
            is_default=True,
            endpoint_id="input-route",
            host_api="WASAPI",
            sample_rate=48_000,
            channels=1,
            name_ordinal=0,
        )
    ])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [
        SimpleNamespace(
            name="Cable",
            is_default=True,
            endpoint_id="output-route",
            host_api="WASAPI",
            sample_rate=48_000,
            channels=2,
            name_ordinal=0,
        )
    ])

    window = MainWindow()
    try:
        window.input_channel_mode_choice.setCurrentIndex(
            window.input_channel_mode_choice.findData("left")
        )
        window.input_cleanup_mode_choice.setCurrentIndex(
            window.input_cleanup_mode_choice.findData("gentle")
        )
        window.save_route_input_preference_action.trigger()

        route_key = window._current_device_route_key()
        assert route_key is not None
        assert window.config.route_input_preferences[route_key] == InputDevicePreference(
            "left", "gentle", "explicit_user"
        )
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def _fake_audio_device(name: str, endpoint_id: str, direction: str, **formats: int):
    return SimpleNamespace(
        name=name,
        is_default=True,
        endpoint_id=endpoint_id,
        host_api="WASAPI",
        sample_rate=formats.get("sample_rate", 48_000),
        channels=formats.get("channels", 1 if direction == "input" else 2),
        name_ordinal=0,
    )


def test_refresh_reapplies_latency_and_route_preset_after_reconnect(
    qapp, monkeypatch, tmp_path
) -> None:
    from mic_eq.config_parts import shared
    from mic_eq.ui import main_window

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    devices = {
        "input": [_fake_audio_device("Mic A", "input-a", "input")],
        "output": [_fake_audio_device("Cable A", "output-a", "output")],
    }
    monkeypatch.setattr(main_window, "list_input_devices", lambda: devices["input"])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: devices["output"])

    window = MainWindow()
    try:
        window._on_device_changed()
        window._apply_latency_compensation_for_current_devices = Mock()
        window._apply_bound_preset_for_current_route = Mock(return_value=False)
        devices["input"] = [_fake_audio_device("Mic B", "input-b", "input")]
        window._refresh_devices()

        window._apply_latency_compensation_for_current_devices.assert_called_once()
        window._apply_bound_preset_for_current_route.assert_called_once()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_refresh_reapplies_latency_when_endpoint_format_changes(
    qapp, monkeypatch, tmp_path
) -> None:
    from mic_eq.config_parts import shared
    from mic_eq.ui import main_window

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    devices = {
        "input": [_fake_audio_device("Mic", "input-route", "input")],
        "output": [_fake_audio_device("Cable", "output-route", "output")],
    }
    monkeypatch.setattr(main_window, "list_input_devices", lambda: devices["input"])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: devices["output"])

    window = MainWindow()
    try:
        window._on_device_changed()
        window._apply_latency_compensation_for_current_devices = Mock()
        window._apply_bound_preset_for_current_route = Mock(return_value=False)
        devices["input"] = [
            _fake_audio_device(
                "Mic", "input-route", "input", sample_rate=44_100, channels=2
            )
        ]
        window._refresh_devices()

        window._apply_latency_compensation_for_current_devices.assert_called_once()
        window._apply_bound_preset_for_current_route.assert_not_called()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
