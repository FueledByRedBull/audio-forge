"""Before/after listening comparison for one captured voice passage."""

from __future__ import annotations

from collections.abc import Mapping
import math
import threading
from typing import Any

import numpy as np
from PyQt6.QtCore import QByteArray, QBuffer, QIODevice, QThread, QTimer, pyqtSignal
from PyQt6.QtMultimedia import (
    QAudio,
    QAudioDevice,
    QAudioFormat,
    QAudioSink,
    QMediaDevices,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
)

from ..analysis.cancellation import AnalysisCancelled
from ..analysis.listening_comparison import (
    MAX_LEVEL_MATCH_DB,
    ComparisonRenderResult,
    RenderedComparisonClip,
    _playback_safe,
    render_comparison,
)
from ..config import save_config
from .accessibility import bind_label, set_accessible_group
from .layout_constants import (
    SUBDUED_TEXT_STYLE,
    configure_resizable_dialog,
    configure_responsive_combo,
    create_scrollable_dialog_body,
    status_chip_style,
)
from .theme import (
    DESCRIPTION_LABEL_STYLE,
    PRIMARY_ACTION_BUTTON_STYLE,
    SECONDARY_ACTION_BUTTON_STYLE,
    PROGRESS_BAR_STYLE,
    message_text_style,
)


def _device_key(device: QAudioDevice | None) -> bytes | None:
    if device is None:
        return None
    try:
        return device.id().data()
    except (AttributeError, TypeError):
        return None


def _parent_config(widget: Any) -> Any:
    """Find an owning config without coupling the dialog to MainWindow."""
    current = widget
    while current is not None:
        config = getattr(current, "config", None)
        if config is not None:
            return config
        parent = getattr(current, "parent", None)
        current = parent() if callable(parent) else None
    return None


def _stored_preview_device_key(widget: Any) -> str | None:
    config = _parent_config(widget)
    value = getattr(config, "preview_playback_device_id", None)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _persist_preview_device(widget: Any, device: QAudioDevice | None) -> bool | None:
    """Persist the preview route without changing the live output route."""
    config = _parent_config(widget)
    if config is None or not hasattr(config, "preview_playback_device_id"):
        return None
    key = _device_key(device)
    setattr(config, "preview_playback_device_id", key.hex() if key else "")
    try:
        return bool(save_config(config))
    except (OSError, TypeError, ValueError):
        return False


def _preview_samples(
    clip: RenderedComparisonClip,
    *,
    level_match: bool,
) -> np.ndarray:
    """Apply the stored level match at playback time, without rerendering."""
    if not level_match or clip.label == "original":
        return np.ascontiguousarray(clip.samples, dtype=np.float32)
    gain = clip.level_match_gain_db
    if not gain and np.isfinite(clip.actual_level_delta_db):
        gain = -float(clip.actual_level_delta_db)
    gain = float(np.clip(gain, -MAX_LEVEL_MATCH_DB, MAX_LEVEL_MATCH_DB))
    matched, _ = _playback_safe(clip.samples, level_match_gain_db=gain)
    return matched


def _format_for_device(
    device: QAudioDevice,
    sample_rate: int,
) -> tuple[QAudioFormat, int, int]:
    """Prefer a mono 16-bit format, then use the device's native format."""
    requested = QAudioFormat()
    requested.setSampleRate(int(sample_rate))
    requested.setChannelCount(1)
    requested.setSampleFormat(QAudioFormat.SampleFormat.Int16)
    try:
        if device.isFormatSupported(requested):
            return requested, int(sample_rate), 1
    except RuntimeError:
        pass

    preferred = device.preferredFormat()
    preferred_rate = int(preferred.sampleRate()) or int(sample_rate)
    preferred_channels = max(1, int(preferred.channelCount()))
    if preferred.sampleFormat() == QAudioFormat.SampleFormat.Unknown:
        preferred.setSampleFormat(QAudioFormat.SampleFormat.Int16)
    return preferred, preferred_rate, preferred_channels


def _pcm_bytes(samples: np.ndarray, audio_format: QAudioFormat, source_rate: int) -> bytes:
    """Convert bounded float samples to the selected device's PCM format."""
    audio = np.asarray(samples, dtype=np.float32)
    audio = audio.copy()
    ramp = min(audio.size // 2, max(1, int(source_rate * 0.005)))
    if ramp:
        audio[:ramp] *= np.linspace(0.0, 1.0, ramp)
        audio[-ramp:] *= np.linspace(1.0, 0.0, ramp)
    target_rate = int(audio_format.sampleRate()) or int(source_rate)
    if target_rate != int(source_rate):
        from scipy.signal import resample_poly

        divisor = math.gcd(target_rate, int(source_rate))
        audio = np.asarray(
            resample_poly(audio, target_rate // divisor, int(source_rate) // divisor),
            dtype=np.float32,
        )

    channels = max(1, int(audio_format.channelCount()))
    if channels > 1:
        audio = np.repeat(audio[:, None], channels, axis=1).reshape(-1)
    audio = np.clip(audio, -1.0, 1.0)
    sample_format = audio_format.sampleFormat()
    if sample_format == QAudioFormat.SampleFormat.Int16:
        return np.asarray(np.round(audio * 32767.0), dtype="<i2").tobytes()
    if sample_format == QAudioFormat.SampleFormat.Int32:
        return np.asarray(np.round(audio.astype(np.float64) * 2147483647.0), dtype="<i4").tobytes()
    if sample_format == QAudioFormat.SampleFormat.Float:
        return np.asarray(audio, dtype="<f4").tobytes()
    if sample_format == QAudioFormat.SampleFormat.UInt8:
        return np.asarray(np.round((audio + 1.0) * 127.5), dtype="u1").tobytes()
    raise RuntimeError("selected playback device has no supported PCM format")


class ListeningComparisonWorker(QThread):
    """Render the three clips away from the Qt event loop."""

    progress = pyqtSignal(str, int)
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    canceled = pyqtSignal()

    def __init__(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        current_settings: Mapping[str, Any],
        proposed_settings: Mapping[str, Any],
        *,
        current_chain_settings: Mapping[str, Any] | None = None,
        proposed_chain_settings: Mapping[str, Any] | None = None,
        level_match: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.audio_data = np.ascontiguousarray(audio_data, dtype=np.float32).copy()
        self.sample_rate = int(sample_rate)
        self.current_settings = dict(current_settings)
        self.proposed_settings = dict(proposed_settings)
        self.current_chain_settings = (
            dict(current_chain_settings) if current_chain_settings is not None else None
        )
        self.proposed_chain_settings = (
            dict(proposed_chain_settings)
            if proposed_chain_settings is not None
            else None
        )
        self.level_match = bool(level_match)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def _should_stop(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        try:
            self.progress.emit("Rendering current and proposed settings...", 15)
            result = render_comparison(
                self.audio_data,
                self.sample_rate,
                self.current_settings,
                self.proposed_settings,
                current_chain_settings=self.current_chain_settings,
                proposed_chain_settings=self.proposed_chain_settings,
                level_match=self.level_match,
                cancel_check=self._should_stop,
            )
            if self._should_stop():
                self.canceled.emit()
                return
            self.progress.emit("Comparison ready", 100)
            self.result_ready.emit(result)
        except AnalysisCancelled:
            self.canceled.emit()
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")


class ListeningComparisonDialog(QDialog):
    """Play one capture as original, current, or proposed processing."""

    comparison_decided = pyqtSignal(bool)

    def __init__(
        self,
        parent=None,
        *,
        audio_data: np.ndarray,
        sample_rate: int,
        current_settings: Mapping[str, Any],
        proposed_settings: Mapping[str, Any],
        current_chain_settings: Mapping[str, Any] | None = None,
        proposed_chain_settings: Mapping[str, Any] | None = None,
        playback_device: QAudioDevice | None = None,
        playback_device_key: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Before/After Listening Comparison")
        self.setModal(True)
        self._audio_data = np.ascontiguousarray(audio_data, dtype=np.float32).copy()
        self._sample_rate = int(sample_rate)
        self._current_settings = dict(current_settings)
        self._proposed_settings = dict(proposed_settings)
        self._current_chain_settings = (
            dict(current_chain_settings) if current_chain_settings is not None else None
        )
        self._proposed_chain_settings = (
            dict(proposed_chain_settings)
            if proposed_chain_settings is not None
            else None
        )
        self._requested_playback_device = playback_device
        self._requested_playback_device_key = (
            playback_device_key or _stored_preview_device_key(self)
        )
        self._outputs: list[QAudioDevice] = []
        self._result: ComparisonRenderResult | None = None
        self._worker: ListeningComparisonWorker | None = None
        self._restart_after_cancel = False
        self._close_requested = False
        self._close_result = False
        self._decision: bool | None = None
        self._audio_sink: QAudioSink | None = None
        self._audio_buffer = None
        self._fade_timer = QTimer(self)
        self._fade_timer.setInterval(4)
        self._fade_timer.timeout.connect(self._fade_playback)
        self._pending_clip: str | None = None
        self._playing_clip_key: str | None = None
        self._playback_start_sample = 0
        self._playback_sample_rate = self._sample_rate
        self._media_devices = QMediaDevices(self)
        self._media_devices.audioOutputsChanged.connect(self._on_audio_outputs_changed)

        self._setup_ui()
        configure_resizable_dialog(
            self,
            preferred_width=680,
            preferred_height=560,
            minimum_width=480,
            minimum_height=360,
        )
        self._refresh_playback_devices()
        self._start_render()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._wait_for_worker)

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        self.content_scroll_area, layout = create_scrollable_dialog_body(self)
        outer.addWidget(self.content_scroll_area)
        layout.setSpacing(10)

        self.scope_label = QLabel(
            "Same captured passage; listen to the original, current processing, "
            "and proposed processing before deciding."
        )
        self.scope_label.setWordWrap(True)
        self.scope_label.setStyleSheet(DESCRIPTION_LABEL_STYLE)
        self.scope_label.setAccessibleName("Listening comparison scope")
        layout.addWidget(self.scope_label)

        self.scope_warning = QLabel(
            "Preview includes the captured input stage, cleanup, gate/VAD, "
            "noise suppression, correction and tone EQ, de-esser, compressor, "
            "and limiter."
        )
        self.scope_warning.setWordWrap(True)
        self.scope_warning.setStyleSheet(status_chip_style("info"))
        self.scope_warning.setAccessibleName("Listening comparison limitations")
        layout.addWidget(self.scope_warning)

        playback_group = QGroupBox("Playback")
        playback_layout = QFormLayout(playback_group)
        output_label = QLabel("Output device:")
        self.output_combo = QComboBox()
        self.output_combo.setAccessibleName("Listening comparison output device")
        self.output_combo.currentIndexChanged.connect(self._on_output_changed)
        self.output_combo.setToolTip(
            "Choose the speakers or headphones that should receive the preview."
        )
        configure_responsive_combo(self.output_combo)
        output_controls = QHBoxLayout()
        output_controls.addWidget(self.output_combo, 1)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setAccessibleName("Refresh listening comparison output devices")
        self.refresh_button.setToolTip("Refresh available playback devices.")
        self.refresh_button.clicked.connect(self._refresh_playback_devices)
        output_controls.addWidget(self.refresh_button)
        playback_layout.addRow(output_label, output_controls)
        bind_label(output_label, self.output_combo)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setAccessibleName("Stop listening comparison playback")
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip("Stop the current preview.")
        self.stop_button.clicked.connect(self._stop_playback)
        playback_layout.addRow("Playback:", self.stop_button)

        self.level_match_checkbox = QCheckBox(
            "Match speech level (actual level difference remains visible)"
        )
        self.level_match_checkbox.setAccessibleName("Match speech level")
        self.level_match_checkbox.setToolTip(
            "Apply a bounded preview-only gain so level does not dominate the comparison."
        )
        self.level_match_checkbox.toggled.connect(self._on_level_match_changed)
        playback_layout.addRow("", self.level_match_checkbox)
        layout.addWidget(playback_group)

        self.progress_label = QLabel("Preparing comparison...")
        self.progress_label.setStyleSheet(SUBDUED_TEXT_STYLE)
        self.progress_label.setAccessibleName("Listening comparison progress")
        layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(PROGRESS_BAR_STYLE)
        self.progress_bar.setAccessibleName("Listening comparison progress bar")
        layout.addWidget(self.progress_bar)

        clips_group = QGroupBox("Listen")
        clips_layout = QFormLayout(clips_group)
        self._play_buttons: dict[str, QPushButton] = {}
        self._clip_labels: dict[str, QLabel] = {}
        for key, label in (
            ("original", "Original recording"),
            ("current", "Current settings"),
            ("proposed", "Proposed settings"),
        ):
            button = QPushButton(f"Play {label}")
            button.setEnabled(False)
            button.clicked.connect(lambda _checked=False, clip=key: self._play_clip(clip))
            button.setAccessibleName(f"Play {label}")
            detail = QLabel("Waiting for render...")
            detail.setWordWrap(True)
            detail.setStyleSheet(SUBDUED_TEXT_STYLE)
            detail.setAccessibleName(f"{label} listening comparison details")
            self._play_buttons[key] = button
            self._clip_labels[key] = detail
            row = QHBoxLayout()
            row.addWidget(button)
            row.addWidget(detail, 1)
            clips_layout.addRow(label + ":", row)
        layout.addWidget(clips_group)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setAccessibleName("Listening comparison status")
        layout.addWidget(self.status_label)

        actions = QHBoxLayout()
        self.keep_button = QPushButton("Keep Proposed")
        self.keep_button.setStyleSheet(PRIMARY_ACTION_BUTTON_STYLE)
        self.keep_button.setEnabled(False)
        self.keep_button.clicked.connect(lambda: self._request_close(True))
        self.reject_button = QPushButton("Reject Proposed")
        self.reject_button.setStyleSheet(SECONDARY_ACTION_BUTTON_STYLE)
        self.reject_button.clicked.connect(lambda: self._request_close(False))
        actions.addWidget(self.keep_button)
        actions.addWidget(self.reject_button)
        layout.addLayout(actions)
        set_accessible_group(
            (
                (self.refresh_button, "Refresh playback devices", None),
                (self.stop_button, "Stop playback", None),
                (self.keep_button, "Keep proposed settings", None),
                (self.reject_button, "Reject proposed settings", None),
            )
        )

    def _refresh_playback_devices(self) -> None:
        self._stop_playback()
        requested_key = _device_key(self._requested_playback_device)
        requested_key_text = (
            requested_key.hex().lower() if requested_key is not None else None
        )
        if requested_key_text is None and self._requested_playback_device_key:
            requested_key_text = self._requested_playback_device_key.strip().lower()
        self.output_combo.blockSignals(True)
        self.output_combo.clear()
        self.output_combo.addItem("Choose headphones or speakers…", None)
        self._outputs = list(QMediaDevices.audioOutputs())
        selected_index = -1
        for index, device in enumerate(self._outputs):
            self.output_combo.addItem(device.description(), device)
            device_key = _device_key(device)
            if requested_key_text is not None and device_key is not None and (
                device_key.hex().lower() == requested_key_text
            ):
                selected_index = index + 1
        if not self._outputs:
            self.output_combo.addItem("No playback devices detected", None)
            self.output_combo.setEnabled(False)
            self.status_label.setText(
                "No playback device is available. Connect speakers or headphones, "
                "then reopen this comparison."
            )
            self.status_label.setStyleSheet(message_text_style("warn"))
        else:
            self.output_combo.setEnabled(True)
            if selected_index >= 0:
                self.output_combo.setCurrentIndex(selected_index)
            elif requested_key_text is not None:
                self.status_label.setText(
                    "The saved preview device is unavailable. Choose another output."
                )
                self.status_label.setStyleSheet(message_text_style("warn"))
        self.output_combo.blockSignals(False)
        self._update_playback_buttons()

    def _current_playback_device(self) -> QAudioDevice | None:
        value = self.output_combo.currentData()
        return value if isinstance(value, QAudioDevice) else None

    def _on_output_changed(self, _index: int) -> None:
        self._stop_playback()
        device = self._current_playback_device()
        if device is not None:
            self._requested_playback_device = device
            device_key = _device_key(device)
            self._requested_playback_device_key = (
                device_key.hex() if device_key is not None else None
            )
            saved = _persist_preview_device(self, device)
            if saved is False:
                self.status_label.setText(
                    "Preview device selected for this session, but saving it failed."
                )
                self.status_label.setStyleSheet(message_text_style("warn"))
        self._update_playback_buttons()

    def _on_audio_outputs_changed(self) -> None:
        self._refresh_playback_devices()

    def _update_playback_buttons(self) -> None:
        enabled = self._result is not None and self._current_playback_device() is not None
        for button in self._play_buttons.values():
            button.setEnabled(enabled)
        self.stop_button.setEnabled(self._audio_sink is not None)

    def _start_render(self) -> None:
        if self._close_requested:
            return
        if self._worker is not None and self._worker.isRunning():
            self._restart_after_cancel = True
            self._worker.stop()
            return
        self._result = None
        self.keep_button.setEnabled(False)
        for button in self._play_buttons.values():
            button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Rendering comparison...")
        self._worker = ListeningComparisonWorker(
            self._audio_data,
            self._sample_rate,
            self._current_settings,
            self._proposed_settings,
            current_chain_settings=self._current_chain_settings,
            proposed_chain_settings=self._proposed_chain_settings,
            # Level matching is a playback preference.  Keeping rendering
            # independent of it makes toggling instant and preserves the same
            # DSP result for every A/B pass.
            level_match=False,
            parent=self,
        )
        worker = self._worker
        worker.progress.connect(self._on_render_progress)
        worker.result_ready.connect(self._on_render_ready)
        worker.failed.connect(self._on_render_failed)
        worker.canceled.connect(self._on_render_canceled)
        worker.finished.connect(lambda finished_worker=worker: self._on_worker_finished(finished_worker))
        worker.start()

    def _on_level_match_changed(self, _enabled: bool) -> None:
        self._stop_playback()
        if self._result is not None:
            for key, detail in self._clip_labels.items():
                detail.setText(
                    self._clip_details(
                        getattr(self._result, key),
                        level_match=self.level_match_checkbox.isChecked(),
                    )
                )
            self.status_label.setText(
                "Speech level matching applied to the next preview playback."
            )
            self.status_label.setStyleSheet(message_text_style("info"))

    def _on_render_progress(self, message: str, percentage: int) -> None:
        self.progress_label.setText(message)
        self.progress_bar.setValue(int(percentage))

    def _on_render_ready(self, result: ComparisonRenderResult) -> None:
        if self._close_requested or self._restart_after_cancel:
            return
        self._result = result
        self.progress_bar.setValue(100)
        self.progress_label.setText("Comparison ready")
        for key, button in self._play_buttons.items():
            button.setEnabled(self._current_playback_device() is not None)
            self._clip_labels[key].setText(
                self._clip_details(
                    getattr(result, key),
                    level_match=self.level_match_checkbox.isChecked(),
                )
            )
        self.keep_button.setEnabled(True)
        self._update_playback_buttons()
        self.status_label.setText(
            f"{result.scope_label}. Native delay compensation: "
            f"{result.alignment_ms:.1f} ms. "
            f"Speech level source: {result.speech_detection}. "
            "The original clip is the unchanged capture; processed clips use "
            "the same captured microphone signal before the input pre-filter."
        )
        self.status_label.setStyleSheet(
            message_text_style("ok" if result.native_authoritative else "warn")
        )

    def _on_render_failed(self, error: str) -> None:
        if self._close_requested:
            return
        self._result = None
        self.progress_label.setText("Comparison failed")
        self.status_label.setText(f"Could not render listening comparison: {error}")
        self.status_label.setStyleSheet(message_text_style("bad"))

    def _on_render_canceled(self) -> None:
        self.progress_label.setText("Comparison canceled")

    def _on_worker_finished(self, worker: ListeningComparisonWorker) -> None:
        if self._worker is worker:
            self._worker = None
        worker.deleteLater()
        if self._close_requested:
            self._finish_close()
        elif self._restart_after_cancel:
            self._restart_after_cancel = False
            self._start_render()

    @staticmethod
    def _clip_details(
        clip: RenderedComparisonClip, *, level_match: bool = False
    ) -> str:
        match_gain = clip.level_match_gain_db
        if level_match and clip.label != "original" and not match_gain:
            match_gain = float(
                np.clip(
                    -clip.actual_level_delta_db,
                    -MAX_LEVEL_MATCH_DB,
                    MAX_LEVEL_MATCH_DB,
                )
            )
        match = (
            f"preview gain {match_gain:+.1f} dB; "
            if level_match and match_gain
            else ""
        )
        if level_match:
            return (
                f"Actual level vs original: {clip.actual_level_delta_db:+.1f} dB; "
                f"{match}bounded preview gain and peak-safe playback"
            )
        return (
            f"Actual level vs original: {clip.actual_level_delta_db:+.1f} dB; "
            f"{match}safety gain {clip.safety_gain_db:+.1f} dB; "
            f"playback peak {clip.peak_db:.1f} dBFS"
        )

    def _current_playback_sample(self) -> int:
        if self._audio_sink is None or self._playing_clip_key is None:
            return int(self._playback_start_sample)
        try:
            elapsed_us = max(0, int(self._audio_sink.processedUSecs()))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return int(self._playback_start_sample)
        offset = int(round(elapsed_us * self._playback_sample_rate / 1_000_000.0))
        if self._result is not None:
            try:
                clip = self._result.clip(self._playing_clip_key)
                offset = min(offset + self._playback_start_sample, int(clip.samples.size))
            except (KeyError, TypeError, ValueError):
                pass
        return max(0, offset)

    def _play_clip(self, key: str, *, start_sample: int = 0) -> None:
        if self._audio_sink is not None:
            self._pending_clip = key
            self._fade_timer.start()
            return
        result = self._result
        device = self._current_playback_device()
        if result is None or device is None:
            return
        clip = result.clip(key)
        self._stop_playback()
        try:
            audio_format, _, _ = _format_for_device(device, result.sample_rate)
            samples = _preview_samples(
                clip,
                level_match=self.level_match_checkbox.isChecked(),
            )
            start = min(max(0, int(start_sample)), int(samples.size))
            payload = _pcm_bytes(samples[start:], audio_format, result.sample_rate)
            audio_buffer = QBuffer(self)
            audio_buffer.setData(QByteArray(payload))
            if not audio_buffer.open(QIODevice.OpenModeFlag.ReadOnly):
                raise RuntimeError("could not open preview buffer")
            sink = QAudioSink(device, audio_format, self)
            sink.setVolume(1.0)
            sink.stateChanged.connect(self._on_audio_state_changed)
            self._audio_buffer = audio_buffer
            self._audio_sink = sink
            self._playing_clip_key = key
            self._playback_start_sample = start
            self._playback_sample_rate = int(result.sample_rate)
            sink.start(audio_buffer)
            if sink.error() != QAudio.Error.NoError:
                raise RuntimeError(f"audio output failed ({sink.error().name})")
            self.status_label.setText(f"Playing {clip.label} on {device.description()}.")
            self.status_label.setStyleSheet(message_text_style("info"))
            self._update_playback_buttons()
        except Exception as error:
            self._stop_playback()
            self.status_label.setText(f"Playback failed: {error}")
            self.status_label.setStyleSheet(message_text_style("bad"))

    def _on_audio_state_changed(self, state: Any) -> None:
        if self._audio_sink is None:
            return
        if state in (
            QAudio.State.IdleState,
            QAudio.State.StoppedState,
        ):
            self._stop_playback()

    def _stop_playback(self) -> None:
        self._fade_timer.stop()
        self._pending_clip = None
        sink, buffer = self._audio_sink, self._audio_buffer
        self._audio_sink = None
        self._audio_buffer = None
        self._playing_clip_key = None
        self._playback_start_sample = 0
        self._playback_sample_rate = self._sample_rate
        if sink is not None:
            sink.stop()
            sink.deleteLater()
        if buffer is not None:
            buffer.close()
            buffer.deleteLater()
        if hasattr(self, "stop_button"):
            self.stop_button.setEnabled(False)

    def _fade_playback(self) -> None:
        sink = self._audio_sink
        if sink is not None and sink.volume() > 0.21:
            sink.setVolume(sink.volume() - 0.2)
            return
        key = self._pending_clip
        start_sample = self._current_playback_sample()
        self._stop_playback()
        if key is not None:
            self._play_clip(key, start_sample=start_sample)

    def _wait_for_worker(self) -> None:
        self._request_close(False)
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()
        self._finish_close()

    def _request_close(self, accepted: bool) -> None:
        if self._close_requested:
            return
        self._close_requested = True
        self._close_result = bool(accepted)
        self._restart_after_cancel = False
        self._stop_playback()
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self.keep_button.setEnabled(False)
            self.reject_button.setEnabled(False)
            self.progress_label.setText("Canceling comparison...")
            return
        self._finish_close()

    def _finish_close(self) -> None:
        if not self._close_requested:
            return
        if self._worker is not None and self._worker.isRunning():
            return
        self._audio_data = np.empty(0, dtype=np.float32)
        self._result = None
        self._decision = self._close_result
        self.comparison_decided.emit(self._close_result)
        self._close_requested = False
        QDialog.done(self, int(QDialog.DialogCode.Accepted if self._close_result else QDialog.DialogCode.Rejected))

    def closeEvent(self, event) -> None:
        self._request_close(False)
        event.ignore()

    def accept(self) -> None:
        self._request_close(True)

    def reject(self) -> None:
        self._request_close(False)

    @property
    def decision(self) -> bool | None:
        return self._decision


__all__ = ["ListeningComparisonDialog", "ListeningComparisonWorker"]
