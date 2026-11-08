# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
"""Tests for UI sample-rate propagation and diagnostics formatting."""

from __future__ import annotations

import re
import time
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

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
from mic_eq.ui import main_window
from mic_eq.ui.main_window import MainWindow
from mic_eq.ui.processing_configuration import LoadOutcome, LoadStatus
from mic_eq.ui.startup_presets import (
    normalize_startup_preset_id as _normalize_startup_preset_id,
    startup_builtin_id as _startup_builtin_id,
)
from mic_eq.ui.auto_eq_explanation import explain_auto_eq
from mic_eq.runtime_snapshot import RuntimeSnapshot
from mic_eq.ui.health import (
    DEVICE_STARTUP_GRACE_SECONDS,
    NOISE_SELECTION_DROPPED_WARNING,
    RecentStreamHealth,
    diagnostic_tokens,
)
from mic_eq.ui.startup_presets import startup_custom_id as _startup_custom_id
from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog
from mic_eq.ui.voice_setup_view import DYNAMICS_KEYS
from mic_eq.config import (
    AppConfig,
    DeviceIdentity,
    DevicePresetBinding,
    InputDevicePreference,
    Preset,
    build_device_route_key,
    build_input_device_preference_key,
    legacy_latency_profile_key,
)
from test_first_run_setup import scene_item
from test_quick_shell import _load_config_without_devices


def _use_processor(window, processor) -> None:
    """Swap the audio engine under a real window, its engine session and the states that write to it."""
    window.processor = window.engine_session.processor = processor
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
    _load_config_without_devices(monkeypatch)
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
    calibration = CalibrationDialog(parent=window, configuration=window.configuration)
    calibration._on_level_update(-20.0, -1.0)
    assert (calibration.level_meter.rms_db, calibration.level_meter.peak_db) == (-20.0, -1.0)
    assert "peaks at -1.0 dB" in calibration.view_model.warning_text

    voice_setup = VoiceSetupDialog(parent=window, configuration=window.configuration)
    voice_setup.setup_state = "voice_recording"
    voice_setup._update_level(-20.0, -1.0)
    assert (voice_setup.level_meter.rms_db, voice_setup.level_meter.peak_db) == (-20.0, -1.0)
    assert "peaks at -1.0 dB" in voice_setup.view_model.warning_text


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
    window.configuration.note_edit()
    assert window.configuration.identity.modified
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
        assert window.configuration.identity.modified is False
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


def _diag_counters(**overrides):
    """The runtime counters every diagnostics test starts from, all quiet; each test adds its own."""
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
        "noise_model": "rnnoise",
        "noise_backend_available": True,
        "noise_backend_failed": False,
        "noise_backend_error": None,
        "recovery_suppressed": False,
        "last_restart_reason": None,
    }
    diagnostics.update(overrides)
    return diagnostics


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
    dialog = CalibrationDialog(parent=window, configuration=window.configuration)
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
        window.configuration.current.eq.to_dict()
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


@pytest.mark.parametrize(
    ("dialog_class", "state_attribute", "recording_state"),
    [
        (CalibrationDialog, "recording_state", "recording"),
        (VoiceSetupDialog, "setup_state", "voice_recording"),
    ],
    ids=["calibration", "voice-setup"],
)
def test_calibration_and_voice_setup_cancel_invalidate_delayed_capture_start(
    qapp, window, dialog_class, state_attribute, recording_state
):
    processor = Mock()
    processor.sample_rate.return_value = 48_000
    _use_processor(window, processor)
    dialog = dialog_class(parent=window)
    setattr(dialog, state_attribute, recording_state)
    dialog._capture_start_timer.start(1)

    dialog.reject()
    _process_events_for(qapp, 25)

    processor.start_raw_recording.assert_not_called()
    assert not dialog._capture_start_timer.isActive()


def test_calibration_ignores_stale_analysis_generation(window):
    _use_processor(window, _FakeProcessor())
    dialog = CalibrationDialog(parent=window, configuration=window.configuration)
    dialog._analysis_generation = 2
    dialog.recording_state = "analyzing"
    previous_step = dialog.view_model.warning_text

    dialog._on_analysis_step("stale step", 55, generation=1)

    assert dialog.view_model.warning_text == previous_step

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
    dialog = CalibrationDialog(parent=window, configuration=window.configuration)
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
    dialog = CalibrationDialog(parent=window, configuration=window.configuration)
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

    assert dialog.view_model.shown["diagnosticsShown"]
    assert dialog.view_model.chip("confidence")["text"].startswith("Good capture: ")
    assert "clear and steady enough to base a correction on" in dialog.view_model.chip("confidence")["text"]
    assert dialog.view_model.chip("confidence")["state"] == "ok"
    assert (
        "Capture quality score 81%." in dialog.view_model.chip("confidence")["tip"]
    )
    assert "Overall score: 76%." in dialog.view_model.chip("confidence")["tip"]
    assert (
        dialog.view_model.chip("error")["text"]
        == "Distance from the tone preset: 5.0 dB; no correction was applied."
    )
    assert (
        dialog.view_model.chip("error")["tip"]
        == "The correction is used at 90% of its fitted strength. "
        "Smoothing took the largest change from 6.0 dB to 4.0 dB."
    )
    assert dialog.view_model.chip("gain")["text"] == "Headroom check: not available."
    assert (
        dialog.view_model.chip("target")["text"]
        == f"Tone preset: {dialog.view_model.curve_name()}"
    )
    assert (
        dialog.view_model.chip("target")["tip"] == "Tone preset and matching: broadcast, Subtle."
    )

    dialog.close()


def test_calibration_dialog_does_not_offer_abstained_eq(window):
    _use_processor(window, _FakeProcessor(sample_rate=48_000))
    dialog = CalibrationDialog(parent=window, configuration=window.configuration)
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
    assert dialog.view_model.shown["startText"] == "Record Again"
    # Plain language up front; the raw reason stays available on hover.
    text = dialog.view_model.warning_text
    explanation = explain_auto_eq(eq_settings)
    assert explanation.outcome_code == "unusable_capture" and explanation.state == "bad"
    assert text.startswith("No correction applied. ")
    assert "Your voice was too close to the background noise in the recording." in text
    assert "noise-referenced SNR is too low" not in text
    assert "noise-referenced SNR is too low" in dialog.view_model.warning_tip
    # Review step: the setup and recording widgets are out of the way.
    assert not dialog.view_model.shown["curveShown"]
    assert not dialog.view_model.shown["instructionsShown"]
    assert not dialog.view_model.shown["meterShown"]

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

    snapshot = RuntimeSnapshot.parse(diagnostics)
    dropped_bits = diagnostic_tokens(
        snapshot,
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
    backend_bits = diagnostic_tokens(
        snapshot,
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
        latency_ms=24.0, dsp_time_ms=0.5, input_buf=128, rnnoise_buf=0,
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
    clock[0] += DEVICE_STARTUP_GRACE_SECONDS - 2
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

    # An engine restart that bypasses the window's Start: both ages are back to the sentinel.
    _labels(window, input_callback_age_ms=NEVER, output_callback_age_ms=NEVER)
    assert chip.state != "warn"
    clock[0] += DEVICE_STARTUP_GRACE_SECONDS - 1
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
    window.processor = window.engine_session.processor = _Processor()
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
    window.processor = window.engine_session.processor = Mock()
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
    window.processor = window.engine_session.processor = Mock()
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
    diagnostics = _diag_counters(input_phase_warning_count=1, input_stereo_correlation=-0.91)

    window._update_diagnostic_labels(
        diagnostics=diagnostics,
        latency_ms=24.0,
        dsp_time_ms=0.5,
        input_buf=128,
        rnnoise_buf=0,
    )

    # Nothing was dropped, so the row says what raised it instead of printing zeros.
    assert window.shell_texts["dropped_label"].text == (
        "Drops: nothing dropped, but the microphone channels cancel each other"
    )
    assert "PH:1" in window.shell_texts["dropped_label"].toolTip
    assert "COR:-0.91" in window.shell_texts["dropped_label"].toolTip
    assert window.shell_texts["dropped_label"].state == "warn"


_TWO_MICS = [
    DeviceIdentity(name="Mic A", is_default=False),
    DeviceIdentity(name="Mic B", is_default=True),
]
_USB_PAIR = [
    DeviceIdentity(name="USB Audio", endpoint_id="endpoint-a", direction="input", name_ordinal=0),
    DeviceIdentity(name="USB Audio", endpoint_id="endpoint-b", direction="input", name_ordinal=1),
]
_REPLACEMENT_ENDPOINT = [
    DeviceIdentity(name="USB Audio", endpoint_id="replacement-endpoint", direction="input")
]


@pytest.mark.parametrize(
    ("identities", "wanted", "index"),
    [
        (_TWO_MICS, DeviceIdentity(name="Mic B", is_default=True), 1),
        (_TWO_MICS, DeviceIdentity(name="Mic B", is_default=False), 1),
        (_TWO_MICS, DeviceIdentity(name="Missing", is_default=False), -1),
        (
            _USB_PAIR,
            DeviceIdentity(
                name="Old Microphone Name", endpoint_id="endpoint-b", direction="input", name_ordinal=0
            ),
            1,
        ),
        (_USB_PAIR, DeviceIdentity(name="USB Audio"), -1),
        (_USB_PAIR, DeviceIdentity(name="USB Audio", direction="input", name_ordinal=1), -1),
        (
            _REPLACEMENT_ENDPOINT,
            DeviceIdentity(name="USB Audio", endpoint_id="disconnected-endpoint", direction="input"),
            -1,
        ),
    ],
    ids=[
        "exact-default",
        "name-only-unique",
        "missing-name",
        "endpoint-id-across-rename",
        "ambiguous-name",
        "ambiguous-ordinal",
        "no-fallback-to-replacement-with-same-name",
    ],
)
def test_device_selection_policy_matches_endpoint_then_exact_then_unique_name(identities, wanted, index):
    assert find_identity_index(identities, wanted) == index


def test_duplicate_name_without_endpoint_id_is_not_persistable():
    identities = [
        DeviceIdentity(name="USB Audio", direction="input", name_ordinal=0),
        DeviceIdentity(name="USB Audio", direction="input", name_ordinal=1),
    ]

    assert identity_is_persistable(identities, identities[0]) is False
    assert identity_is_persistable([identities[0]], identities[0]) is True


@pytest.mark.parametrize(
    ("input_identity", "output_identity", "expected_args", "expected_kwargs"),
    [
        (
            DeviceIdentity(name="USB Audio", name_ordinal=1),
            DeviceIdentity(name="USB Audio", name_ordinal=2),
            ("USB Audio", "USB Audio", 1, 2),
            {"input_device_endpoint_id": None, "output_device_endpoint_id": None},
        ),
        (
            DeviceIdentity(name="USB Audio", endpoint_id="input-endpoint"),
            DeviceIdentity(name="USB Audio", endpoint_id="output-endpoint"),
            ("USB Audio", "USB Audio", 0, 0),
            {
                "input_device_endpoint_id": "input-endpoint",
                "output_device_endpoint_id": "output-endpoint",
            },
        ),
    ],
    ids=["duplicate-name-ordinals", "endpoint-ids"],
)
def test_start_processor_for_route_forwards_name_ordinals_and_endpoint_ids(
    input_identity, output_identity, expected_args, expected_kwargs
):
    class Processor:
        def __init__(self):
            self.args = None
            self.kwargs = None

        def start(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            return "started"

    processor = Processor()
    result = start_processor_for_route(processor, input_identity, output_identity)

    assert result == "started"
    assert processor.args == expected_args
    assert processor.kwargs == expected_kwargs


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
    window.configuration.load_id = lambda preset_id, *, remember: (
        loaded.append(preset_id) or LoadOutcome(LoadStatus.LOADED, preset=Preset())
    )

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


def test_device_selection_prefers_the_audioforge_mic_over_any_cable():
    identities = [
        DeviceIdentity(name="VB-Audio Cable"),
        DeviceIdentity(name="Speakers", is_default=True),
        DeviceIdentity(
            name="AudioForge Mic Input", endpoint_id="pipewire:audioforge_mic_input"
        ),
    ]

    assert preferred_output_index(identities) == 2


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
    monkeypatch.setattr(main_window, "save_config", _failing_save(write_failure))
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
        lambda: [_fake_audio_device("Mic", "input-id", "input")],
    )
    monkeypatch.setattr(
        main_window,
        "list_output_devices",
        lambda: [_fake_audio_device("Cable", "output-id", "output")],
    )
    monkeypatch.setattr(main_window, "save_config", _failing_save(write_failure))

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
    monkeypatch.setattr(main_window, "save_config", _failing_save(write_failure))

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
    monkeypatch.setattr(main_window, "save_config", _failing_save(write_failure))

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


def test_a_counter_that_starts_over_does_not_warn_until_it_grows_again(window):
    processor = _MeterProcessor()
    _recovery_setup(window, processor)
    assert _poll_dropped_selections(window, processor, 5) == NOISE_SELECTION_DROPPED_WARNING
    window.status_message.clearMessage()

    assert _poll_dropped_selections(window, processor, 0) == ""
    assert _poll_dropped_selections(window, processor, 1) == NOISE_SELECTION_DROPPED_WARNING


def test_the_window_starts_from_the_zero_the_engine_exports(window):
    # A fresh engine counts from zero, and the window has seen nothing yet.
    assert window.processor.get_runtime_diagnostics()["noise_selection_dropped_count"] == 0
    assert window._health_memory.noise_drops_seen == 0


class _RefusingProcessor(_MeterProcessor):
    """An engine that cannot load DeepFilterNet and says so to the caller."""

    def list_noise_models(self):
        return [("rnnoise", "RNNoise"), ("deepfilter", "DeepFilterNet")]

    def set_noise_model(self, model: str) -> bool:
        return model == "rnnoise"


def test_a_selection_refused_to_its_caller_keeps_its_own_message(window, monkeypatch):
    # The caller is told why, and the engine does not count what it already
    # reported, so the next poll leaves that message on the line.
    monkeypatch.setattr(QMessageBox, "warning", Mock())
    processor = _RefusingProcessor()
    _recovery_setup(window, processor)
    window.noise_suppression_state.models = tuple(processor.list_noise_models())

    window._on_model_changed(1)

    specific = window.status_message.text
    assert "unavailable; previous model retained" in specific
    assert _poll_dropped_selections(window, processor, 0) == specific
    assert specific != NOISE_SELECTION_DROPPED_WARNING


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
    diagnostics = _diag_counters(
        output_underrun_total=4, output_retime_adjustment_count=606, input_backlog_recovery_count=0,
        input_backlog_dropped_samples=0, clip_event_count=0, clip_peak_db=-120.0,
        input_resampler_active=False, output_resampler_active=False, noise_model="deepfilter",
    )
    window._stream_health = RecentStreamHealth()
    window._stream_health.observe(RuntimeSnapshot.parse(diagnostics), now=0.0)
    window._health_memory = replace(window._health_memory, output_underrun_total=4)

    window._update_diagnostic_labels(
        diagnostics=diagnostics,
        latency_ms=64.0,
        dsp_time_ms=0.6,
        input_buf=512,
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
    diagnostics = _diag_counters(output_underrun_total=5, output_retime_adjustment_count=0)
    window._health_memory = replace(window._health_memory, output_underrun_total=4)

    window._update_diagnostic_labels(
        diagnostics=diagnostics,
        latency_ms=64.0,
        dsp_time_ms=0.6,
        input_buf=512,
        rnnoise_buf=0,
    )

    assert window.shell_texts["dropped_label"].state == "warn"
    clock[0] += 5.1

    window._update_diagnostic_labels(
        diagnostics=diagnostics,
        latency_ms=64.0,
        dsp_time_ms=0.6,
        input_buf=512,
        rnnoise_buf=0,
    )

    assert window.shell_texts["dropped_label"].state == "ok"


def test_the_output_clip_counter_is_a_support_counter_and_does_not_warn(window):
    _recovery_setup(window)
    diagnostics = _diag_counters(
        output_retime_adjustment_count=0, clip_event_count=0, clip_peak_db=-120.0,
        output_clip_event_count=1, output_clip_peak_db=-0.25, limiter_effective_ceiling_db=-1.5,
    )

    window._update_diagnostic_labels(
        diagnostics=diagnostics,
        latency_ms=32.0,
        dsp_time_ms=0.4,
        input_buf=256,
        rnnoise_buf=0,
    )

    assert "OCL:1" in window.shell_texts["dropped_label"].toolTip
    assert "OPK:-0.2" in window.shell_texts["dropped_label"].toolTip
    assert "LIM:-1.5" in window.shell_texts["dropped_label"].toolTip
    assert window.shell_texts["dropped_label"].state == "ok"


def test_new_output_true_peak_event_warns_once(window):
    _recovery_setup(window)
    diagnostics = _diag_counters(
        output_retime_adjustment_count=0, clip_event_count=0, clip_peak_db=-120.0,
        output_clip_event_count=0, output_clip_peak_db=-120.0, output_true_peak_event_count=1,
        output_true_peak_db=0.4, limiter_effective_ceiling_db=-1.5,
    )

    window._update_diagnostic_labels(
        diagnostics=diagnostics,
        latency_ms=32.0,
        dsp_time_ms=0.4,
        input_buf=256,
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
        rnnoise_buf=0,
        output_rms_db=-18.0,
    )

    assert "Output: OK" in window.shell_texts["output_health_label"].text
    assert window.shell_texts["dropped_label"].state == "ok"


def test_new_gate_chatter_event_warns_for_stability_window(window, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("mic_eq.ui.health.time.monotonic", lambda: clock[0])
    _recovery_setup(window)
    diagnostics = _diag_counters(
        output_retime_adjustment_count=0, clip_event_count=0, clip_peak_db=-120.0,
        output_clip_event_count=0, output_clip_peak_db=-120.0, gate_chatter_event_count=1,
    )

    window._update_diagnostic_labels(
        diagnostics=diagnostics,
        latency_ms=32.0,
        dsp_time_ms=0.4,
        input_buf=256,
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


@pytest.mark.parametrize(
    ("running", "started", "stopped"), [(False, 1, 1), (True, 0, 0)], ids=["owned", "preexisting"]
)
def test_latency_calibration_failure_stops_only_a_processor_it_started(window, running, started, stopped):
    processor = _LatencyProcessor(running=running)
    owner = _latency_window(window, processor)
    dialog = LatencyCalibrationDialog(owner)

    dialog._on_run_clicked()

    assert processor.started == started
    assert processor.stopped == stopped
    assert processor.running is running


class _StopRefusedOnce(_LatencyProcessor):
    def stop(self):
        if self.stopped == 0:
            self.stopped = -1
            raise RuntimeError("stop refused")
        super().stop()


@pytest.mark.parametrize("dialog_type", [LatencyCalibrationDialog, CalibrationDialog])
def test_a_dialog_whose_stop_fails_says_so_and_still_owns_the_audio(window, caplog, dialog_type):
    processor = _StopRefusedOnce(running=False)
    owner = _latency_window(window, processor)
    shown = _status_log(window)
    dialog = dialog_type(owner)
    with caplog.at_level("WARNING"):
        if dialog_type is LatencyCalibrationDialog:
            dialog._on_run_clicked()  # its capture fails, so it stops what it started
        else:
            dialog._start_recording("level_check")
            dialog._capture_start_timer.stop()
            dialog._reset_recording_ui()
    assert processor.started == 1 and processor.running
    assert any("stop refused" in record.getMessage() for record in caplog.records)
    assert any("Stopping audio failed: stop refused" in text for text in shown)

    dialog.reject()  # the close stops it, as the dialog still owns it

    assert not processor.running and processor.stopped == 0


def test_latency_calibration_respects_user_mute(window, monkeypatch):
    processor = _LatencyProcessor(running=False)
    processor.set_output_mute = Mock()
    owner = _latency_window(window, processor)
    owner.engine_session.set_user_muted(True)
    processor.set_output_mute.reset_mock()
    notice = Mock()
    monkeypatch.setattr(QMessageBox, "information", notice)
    dialog = LatencyCalibrationDialog(owner)

    dialog._on_run_clicked()
    assert processor.started == 0
    notice.assert_called_once()
    dialog.reject()
    processor.set_output_mute.assert_not_called()


def test_voice_setup_dynamics_intensity_controls_custom_targets(qapp):
    dialog = VoiceSetupDialog()

    assert dialog.view_model.selected("dynamics") == "balanced"
    assert scene_item(dialog, "p95Number").isEnabled() is False
    assert scene_item(dialog, "peakNumber").isEnabled() is False

    dialog.view_model.pickDynamics(DYNAMICS_KEYS.index("custom"))
    qapp.processEvents()

    assert scene_item(dialog, "p95Number").isEnabled() is True
    assert scene_item(dialog, "peakNumber").isEnabled() is True
    dialog.close()


def test_start_refuses_retained_configuration_mute(window, monkeypatch) -> None:
    processor = _LatencyProcessor(running=False)
    _latency_window(window, processor)
    window.engine_session.hold_mute("configuration", True)
    shown = _status_log(window)
    synced: list[bool] = []
    monkeypatch.setattr(window, "_sync_processing_controls", lambda: synced.append(True))

    assert window.engine_session.start_now() is False

    assert processor.started == 0 and synced == [True]
    assert "reload the preset" in shown[-1]


@pytest.mark.parametrize("checked,temporary", [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize("write_error", [False, True])
def test_user_mute_applies_before_failed_persistence(
    window, monkeypatch, checked: bool, temporary: bool, write_error: bool
) -> None:
    calls: list[bool] = []
    processor = SimpleNamespace(set_output_mute=calls.append, is_running=lambda: False)
    if temporary:
        window.engine_session.hold_mute("calibration", True)
    window.engine_session.processor = processor
    window.config.user_muted = not checked
    shown = _status_log(window)

    def save(_config) -> bool:
        assert calls == [checked or temporary]
        if write_error:
            raise OSError("config is not writable")
        return False

    monkeypatch.setattr("mic_eq.ui.main_window.save_config", save)
    window._on_user_mute_toggled(checked)

    assert window.engine_session.view.user_muted is checked
    assert window.config.user_muted is checked
    assert calls == [checked or temporary]
    assert "preference could not be saved" in shown[-1]
    assert ("Output muted" if checked or temporary else "Output unmuted") in shown[-1]
    window.engine_session.processor = window.processor
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)


def test_user_mute_does_not_claim_native_failure_is_muted(window, monkeypatch) -> None:
    window.engine_session.processor = SimpleNamespace(
        set_output_mute=Mock(side_effect=RuntimeError("unavailable")), is_running=lambda: False
    )
    shown = _status_log(window)
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _config: True)

    window._on_user_mute_toggled(True)

    window.engine_session.processor = window.processor
    assert "could not be applied" in shown[-1]
    assert "preference saved" in shown[-1]
    assert window.shell_texts["transmission_status_label"].text.endswith("/ Mute pending")


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


@pytest.mark.parametrize(
    ("config", "route_key", "input_key"),
    [
        (
            AppConfig(
                input_channel_mode="left",
                input_cleanup_mode="gentle",
                route_input_preferences={"another-route": InputDevicePreference("right", "strong")},
            ),
            "missing-route",
            None,
        ),
        (
            AppConfig(
                input_device_preferences={
                    build_input_device_preference_key(
                        DeviceIdentity(name="Mic", endpoint_id="old-endpoint", direction="input")
                    ): InputDevicePreference("left", "gentle")
                }
            ),
            None,
            build_input_device_preference_key(
                DeviceIdentity(name="Mic", endpoint_id="new-endpoint", direction="input")
            ),
        ),
    ],
    ids=["missing-route-override", "new-input-endpoint"],
)
def test_missing_route_or_input_override_does_not_fall_back_to_global_modes(config, route_key, input_key) -> None:
    owner = cast(Any, MainWindow.__new__(MainWindow))
    owner.config = config
    owner._current_device_route_key = lambda: route_key
    owner._current_input_preference_key = lambda: input_key

    assert MainWindow._input_preference_for_current_route(owner) == InputDevicePreference()


def test_route_input_preference_action_saves_visible_controls_for_current_route(
    qapp, monkeypatch, tmp_path
) -> None:
    from mic_eq.config_parts import shared
    from mic_eq.ui import main_window

    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    monkeypatch.setattr(main_window, "list_input_devices", lambda: [_fake_audio_device("Mic", "input-route", "input")])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: [_fake_audio_device("Cable", "output-route", "output")])

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


def _failing_save(write_failure) -> Mock:
    """A save_config that raises ``write_failure`` if it is an exception, else returns it."""
    return Mock(
        side_effect=write_failure if isinstance(write_failure, Exception) else None,
        return_value=write_failure,
    )


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


@pytest.mark.parametrize(
    ("initial", "changed", "preset_calls"),
    [
        (
            ("Mic A", "input-a", "Cable A", "output-a"),
            _fake_audio_device("Mic B", "input-b", "input"),
            1,
        ),
        (
            ("Mic", "input-route", "Cable", "output-route"),
            _fake_audio_device("Mic", "input-route", "input", sample_rate=44_100, channels=2),
            0,
        ),
    ],
    ids=["reconnected-as-another-device", "endpoint-format-changed"],
)
def test_refresh_reapplies_latency_but_the_route_preset_only_for_another_device(
    qapp, monkeypatch, tmp_path, initial, changed, preset_calls
) -> None:
    from mic_eq.config_parts import shared
    from mic_eq.ui import main_window

    input_name, input_id, output_name, output_id = initial
    monkeypatch.setattr(shared, "_config_base_dir", lambda: tmp_path)
    devices = {
        "input": [_fake_audio_device(input_name, input_id, "input")],
        "output": [_fake_audio_device(output_name, output_id, "output")],
    }
    monkeypatch.setattr(main_window, "list_input_devices", lambda: devices["input"])
    monkeypatch.setattr(main_window, "list_output_devices", lambda: devices["output"])

    window = MainWindow()
    try:
        window._on_device_changed()
        window._apply_latency_compensation_for_current_devices = Mock()
        window._apply_bound_preset_for_current_route = Mock(return_value=False)
        devices["input"] = [changed]
        window._refresh_devices()

        window._apply_latency_compensation_for_current_devices.assert_called_once()
        assert window._apply_bound_preset_for_current_route.call_count == preset_calls
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
