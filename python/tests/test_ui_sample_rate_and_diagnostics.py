# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
"""Tests for UI sample-rate propagation and diagnostics formatting."""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QWidget

from mic_eq.ui.calibration_dialog import CalibrationDialog, _selected_device_pair
from mic_eq.ui.latency_calibration_dialog import (
    LatencyCalibrationDialog,
    _capture_sample_rate,
    _device_name as latency_device_name,
)
from mic_eq.ui.main_window import (
    MainWindow,
    _normalize_startup_preset_id,
    _startup_builtin_id,
    _startup_custom_id,
)
from mic_eq.config import (
    CompressorSettings,
    DeviceIdentity,
    Preset,
    build_latency_profile_key,
    legacy_latency_profile_key,
)


class _SignalStub:
    def connect(self, _slot):
        return None


class _CaptureWorkerStub:
    last_init: dict | None = None

    def __init__(self, audio_data, sample_rate, target_preset):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.target_preset = target_preset
        _CaptureWorkerStub.last_init = {
            "audio_len": len(audio_data),
            "sample_rate": sample_rate,
            "target_preset": target_preset,
        }
        self.step_progress = _SignalStub()
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


class _FakeProcessor:
    def __init__(self, sample_rate: int = 44_100, output_sample_rate: int = 48_000):
        self._sample_rate = sample_rate
        self._output_sample_rate = output_sample_rate

    def sample_rate(self) -> int:
        return self._sample_rate

    def output_sample_rate(self) -> int:
        return self._output_sample_rate


class _FakeOwner(QWidget):
    def __init__(self, processor: _FakeProcessor):
        super().__init__()
        self.processor = processor


class _FakeCombo:
    def __init__(self, items: list[tuple[str, object]] | None = None):
        self._items = list(items or [])
        self._index = 0 if self._items else -1
        self._signals_blocked = False

    def blockSignals(self, blocked: bool):
        self._signals_blocked = blocked

    def clear(self):
        self._items.clear()
        self._index = -1

    def addItem(self, label, data=None):
        self._items.append((label, data))

    def setCurrentIndex(self, index: int):
        self._index = index

    def currentData(self):
        if 0 <= self._index < len(self._items):
            return self._items[self._index][1]
        return None

    def currentText(self):
        if 0 <= self._index < len(self._items):
            return self._items[self._index][0]
        return ""

    def count(self):
        return len(self._items)

    def itemData(self, index: int):
        return self._items[index][1]


class _FakeControl:
    def __init__(self, value=None):
        self.value = value

    def setChecked(self, value):
        self.value = bool(value)

    def isChecked(self):
        return bool(self.value)

    def setValue(self, value):
        self.value = value


class _FakeLabel:
    def __init__(self):
        self.text = ""
        self.stylesheet = ""
        self.visible = True

    def setText(self, text: str):
        self.text = text

    def setStyleSheet(self, stylesheet: str):
        self.stylesheet = stylesheet

    def setVisible(self, visible: bool):
        self.visible = visible


class _FakeStatusBar:
    def __init__(self):
        self.messages: list[tuple[str, int | None]] = []

    def showMessage(self, message: str, timeout: int | None = None):
        self.messages.append((message, timeout))


class _PresetProcessor:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def set_rnnoise_enabled(self, value):
        self.calls.append(("rnnoise_enabled", value))

    def set_rnnoise_strength(self, value):
        self.calls.append(("rnnoise_strength", value))

    def set_noise_model(self, value):
        self.calls.append(("noise_model", value))
        return True

    def set_bypass(self, value):
        self.calls.append(("bypass", value))


class _PresetPanel:
    def __init__(self):
        self.settings = None
        self.compressor_settings = None
        self.limiter_settings = None

    def set_settings(self, settings):
        self.settings = settings

    def set_compressor_settings(self, settings):
        self.compressor_settings = settings

    def set_limiter_settings(self, settings):
        self.limiter_settings = settings


class _FakeMeter:
    def __init__(self):
        self.levels: tuple[float, float] | None = None

    def set_levels(self, rms: float, peak: float):
        self.levels = (rms, peak)


class _FakePanel:
    def __init__(self):
        self.gain_reduction: float | None = None
        self.current_release_updates = 0
        self.auto_makeup_updates: list[tuple[float, float]] = []
        self.vad_updates: list[float] = []

    def update_gain_reduction(self, value: float):
        self.gain_reduction = value

    def _update_current_release(self):
        self.current_release_updates += 1

    def update_auto_makeup_meters(self, current_lufs: float, makeup_gain: float):
        self.auto_makeup_updates.append((current_lufs, makeup_gain))

    def update_vad_confidence(self, value: float):
        self.vad_updates.append(value)


class _MeterProcessor:
    def __init__(self):
        self.diagnostics_calls = 0
        self._diagnostics = {
            "input_dropped_samples": 12,
            "lock_contention_count": 1,
            "suppressor_non_finite_count": 0,
            "stream_restart_count": 2,
            "output_underrun_total": 3,
            "output_recovery_count": 4,
            "input_backlog_recovery_count": 5,
            "input_backlog_dropped_samples": 6,
            "clip_event_count": 7,
            "clip_peak_db": -1.25,
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

    def service_recovery(self):
        return None

    def get_last_restart_reason(self):
        return None

    def get_last_stream_error(self):
        return None


class _RecoveryWindow:
    def __init__(self):
        self.processor = _MeterProcessor()
        self.input_meter = _FakeMeter()
        self.output_meter = _FakeMeter()
        self.compressor_panel = _FakePanel()
        self.deesser_panel = _FakePanel()
        self.gate_panel = _FakePanel()
        self.latency_label = _FakeLabel()
        self.buffer_label = _FakeLabel()
        self.dropped_label = _FakeLabel()
        self.backend_diag_label = _FakeLabel()
        self.recovery_diag_label = _FakeLabel()
        self.status_bar = _FakeStatusBar()
        self._last_diag_poll = -1.0
        self._last_backend_warning = None
        self._calibration_dialog_open = False
        self._output_stall_started_at = None
        self._output_callback_stall_started_at = None
        self._processing_started_at = 0.0
        self._last_output_recovery_at = 0.0

    def _maybe_recover_output_stall(self, *args, **kwargs):
        return None

    def _maybe_recover_callback_stall(self, *args, **kwargs):
        return None

    def _reset_health_labels(self):
        return None

    @staticmethod
    def _diag_token(label: str, value) -> str | None:
        return MainWindow._diag_token(label, value)

    @classmethod
    def _extend_diag_tokens(cls, tokens: list[str], diagnostics: dict, keys: list[tuple[str, str]]) -> None:
        MainWindow._extend_diag_tokens(tokens, diagnostics, keys)

    def _set_health_chip(self, label: _FakeLabel, text: str, state: str) -> None:
        label.setText(text)


def test_calibration_analysis_uses_processor_sample_rate(qapp, monkeypatch):
    owner = _FakeOwner(_FakeProcessor(sample_rate=44_100))
    dialog = CalibrationDialog(parent=owner)
    dialog.audio_data = np.ones(256, dtype=np.float32)
    monkeypatch.setattr("mic_eq.ui.calibration_dialog.AnalysisWorker", _CaptureWorkerStub)

    dialog._start_analysis()

    assert _CaptureWorkerStub.last_init is not None
    assert _CaptureWorkerStub.last_init["sample_rate"] == 44_100
    assert dialog.analysis_worker is not None
    assert dialog.analysis_worker.sample_rate == 44_100

    dialog.close()
    owner.close()


def test_calibration_recorded_audio_reports_processor_rate(qapp):
    owner = _FakeOwner(_FakeProcessor(sample_rate=44_100))
    dialog = CalibrationDialog(parent=owner)
    dialog.audio_data = np.ones(32, dtype=np.float32)

    audio, sample_rate = dialog.get_recorded_audio()

    assert audio is not None
    assert sample_rate == 44_100

    dialog.close()
    owner.close()


def test_latency_calibration_uses_processor_output_sample_rate(qapp):
    owner = _FakeOwner(_FakeProcessor(output_sample_rate=44_100))
    assert _capture_sample_rate(owner) == 44_100
    owner.close()


def test_main_window_diagnostics_include_new_metrics():
    diagnostics = {
        "input_dropped_samples": 12,
        "lock_contention_count": 1,
        "suppressor_non_finite_count": 0,
        "stream_restart_count": 2,
        "output_underrun_total": 3,
        "output_recovery_count": 4,
        "input_backlog_recovery_count": 5,
        "input_backlog_dropped_samples": 6,
        "clip_event_count": 7,
        "clip_peak_db": -1.25,
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
            ("clip_event_count", "CL"),
            ("clip_peak_db", "PK"),
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
    assert "CL:7" in dropped_bits
    assert "PK:-1.2" in dropped_bits
    assert "IR:Y" in backend_bits
    assert "OR:N" in backend_bits


def test_refresh_devices_preserves_existing_selection(qapp, monkeypatch):
    window = MainWindow.__new__(MainWindow)
    window.input_combo = _FakeCombo(
        [
            ("Mic A", DeviceIdentity(name="Mic A", is_default=False)),
            ("Mic B", DeviceIdentity(name="Mic B", is_default=True)),
        ]
    )
    window.output_combo = _FakeCombo(
        [
            ("Out A", DeviceIdentity(name="Out A", is_default=False)),
            ("Out B", DeviceIdentity(name="Out B", is_default=True)),
        ]
    )
    window.input_combo.setCurrentIndex(0)
    window.output_combo.setCurrentIndex(0)
    window.device_warning_banner = _FakeLabel()
    window.status_bar = _FakeStatusBar()
    window.config = type(
        "Cfg",
        (),
        {
            "last_input_device": "Mic A",
            "last_output_device": "Out A",
            "last_input_device_identity": DeviceIdentity(name="Mic A", is_default=False),
            "last_output_device_identity": DeviceIdentity(name="Out A", is_default=False),
        },
    )()

    monkeypatch.setattr(
        "mic_eq.ui.main_window.list_input_devices",
        lambda: [
            type("Dev", (), {"name": "Mic A", "is_default": False})(),
            type("Dev", (), {"name": "Mic B", "is_default": True})(),
        ],
    )
    monkeypatch.setattr(
        "mic_eq.ui.main_window.list_output_devices",
        lambda: [
            type("Dev", (), {"name": "Out A", "is_default": False})(),
            type("Dev", (), {"name": "Out B", "is_default": True})(),
        ],
    )
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _cfg: None)

    window._refresh_devices()

    assert window.output_combo.currentData() == DeviceIdentity(name="Out A", is_default=False)
    assert window.input_combo.currentData() == DeviceIdentity(name="Mic A", is_default=False)
    assert window.status_bar.messages == []


def test_refresh_devices_clears_missing_output_selection(qapp, monkeypatch):
    window = MainWindow.__new__(MainWindow)
    window.input_combo = _FakeCombo(
        [("Mic A", DeviceIdentity(name="Mic A", is_default=True))]
    )
    window.output_combo = _FakeCombo(
        [("Out Old", DeviceIdentity(name="Out Old", is_default=False))]
    )
    window.input_combo.setCurrentIndex(0)
    window.output_combo.setCurrentIndex(0)
    window.device_warning_banner = _FakeLabel()
    window.status_bar = _FakeStatusBar()
    window.config = type(
        "Cfg",
        (),
        {
            "last_input_device": "Mic A",
            "last_output_device": "Out Old",
            "last_input_device_identity": DeviceIdentity(name="Mic A", is_default=True),
            "last_output_device_identity": DeviceIdentity(name="Out Old", is_default=False),
        },
    )()

    monkeypatch.setattr(
        "mic_eq.ui.main_window.list_input_devices",
        lambda: [type("Dev", (), {"name": "Mic A", "is_default": True})()],
    )
    monkeypatch.setattr(
        "mic_eq.ui.main_window.list_output_devices",
        lambda: [type("Dev", (), {"name": "Out New", "is_default": True})()],
    )
    monkeypatch.setattr("mic_eq.ui.main_window.save_config", lambda _cfg: None)

    window._refresh_devices()

    assert window.config.last_output_device == ""
    assert window.config.last_output_device_identity is None
    assert window.output_combo.currentData() == DeviceIdentity(name="Out New", is_default=True)
    assert any("Previous output device 'Out Old' not found" in message for message, _ in window.status_bar.messages)


def test_latency_profile_key_uses_structured_device_identity():
    window = MainWindow.__new__(MainWindow)
    window.input_combo = _FakeCombo(
        [("Mic A", DeviceIdentity(name="Mic A", is_default=True))]
    )
    window.output_combo = _FakeCombo(
        [("Out B", DeviceIdentity(name="Out B", is_default=False))]
    )
    key = MainWindow._latency_profile_key(window)

    assert key == build_latency_profile_key(
        DeviceIdentity(name="Mic A", is_default=True),
        DeviceIdentity(name="Out B", is_default=False),
    )
    assert key != legacy_latency_profile_key("Mic A", "Out B")


def test_calibration_dialog_selected_device_pair_returns_names():
    owner = type("Owner", (), {})()
    owner.input_combo = _FakeCombo([("Mic A", DeviceIdentity(name="Mic A", is_default=False))])
    owner.output_combo = _FakeCombo([("Out B", DeviceIdentity(name="Out B", is_default=True))])

    assert _selected_device_pair(owner) == ("Mic A", "Out B")


def test_latency_dialog_device_name_coerces_identity_to_name():
    assert latency_device_name(DeviceIdentity(name="Mic A", is_default=False)) == "Mic A"
    assert latency_device_name("Out B") == "Out B"
    assert latency_device_name(None) is None


def test_update_meters_surfaces_output_recovery_and_reuses_diagnostics(qapp):
    window = _RecoveryWindow()

    MainWindow._update_meters(window)

    assert window.processor.diagnostics_calls == 1
    assert "ORC:4" in window.recovery_diag_label.text
    assert "R:2" in window.recovery_diag_label.text
    assert not window.status_bar.messages


def test_startup_preset_ids_normalize_builtin_and_custom_legacy_names():
    assert _normalize_startup_preset_id("Voice Clarity") == _startup_builtin_id("voice")
    assert _normalize_startup_preset_id("voice") == _startup_builtin_id("voice")
    assert _normalize_startup_preset_id("Custom Voice", ("Custom Voice",)) == _startup_custom_id("Custom Voice")
    assert _normalize_startup_preset_id(_startup_builtin_id("flat")) == _startup_builtin_id("flat")


def test_apply_preset_passes_advanced_compressor_fields(qapp):
    window = MainWindow.__new__(MainWindow)
    window.gate_panel = _PresetPanel()
    window.eq_panel = _PresetPanel()
    window.deesser_panel = _PresetPanel()
    window.compressor_panel = _PresetPanel()
    window.rnnoise_checkbox = _FakeControl()
    window.strength_slider = _FakeControl()
    window.model_combo = _FakeCombo([])
    window.bypass_checkbox = _FakeControl()
    window.processor = _PresetProcessor()
    window.status_bar = _FakeStatusBar()

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
        ),
    )

    MainWindow._apply_preset(window, preset)

    assert window.compressor_panel.compressor_settings["adaptive_release"] is True
    assert window.compressor_panel.compressor_settings["base_release_ms"] == 75.0
    assert window.compressor_panel.compressor_settings["auto_makeup_enabled"] is True
    assert window.compressor_panel.compressor_settings["target_lufs"] == -16.0


class _LatencyProcessor:
    def __init__(self, running: bool, fail_recording: bool = True):
        self.running = running
        self.fail_recording = fail_recording
        self.started = 0
        self.stopped = 0
        self.recovery_suppressed: list[bool] = []

    def is_running(self):
        return self.running

    def start(self, _input_device=None, _output_device=None):
        self.running = True
        self.started += 1

    def stop(self):
        self.running = False
        self.stopped += 1

    def output_sample_rate(self):
        return 48_000

    def set_recovery_suppressed(self, value):
        self.recovery_suppressed.append(bool(value))

    def start_raw_recording(self, _duration):
        if self.fail_recording:
            raise RuntimeError("recording setup failed")

    def stop_raw_recording(self):
        return None

    def set_output_mute(self, _muted):
        return None


class _LatencyOwner(QWidget):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.input_combo = _FakeCombo([("Mic", DeviceIdentity(name="Mic", is_default=False))])
        self.output_combo = _FakeCombo([("Out", DeviceIdentity(name="Out", is_default=False))])


def test_latency_calibration_failure_stops_owned_processor(qapp):
    processor = _LatencyProcessor(running=False)
    owner = _LatencyOwner(processor)
    dialog = LatencyCalibrationDialog(owner)

    dialog._on_run_clicked()

    assert processor.started == 1
    assert processor.stopped == 1
    assert not processor.running


def test_latency_calibration_failure_does_not_stop_preexisting_processor(qapp):
    processor = _LatencyProcessor(running=True)
    owner = _LatencyOwner(processor)
    dialog = LatencyCalibrationDialog(owner)

    dialog._on_run_clicked()

    assert processor.started == 0
    assert processor.stopped == 0
    assert processor.running
