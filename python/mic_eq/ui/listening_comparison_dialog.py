"""Before/after listening comparison for one captured voice passage."""

from __future__ import annotations

from collections.abc import Mapping
import logging
import math
import threading
from typing import Any

import numpy as np
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QThread, QTimer, Signal, Slot
from PySide6.QtMultimedia import (
    QtAudio,
    QAudioDevice,
    QAudioFormat,
    QAudioSink,
    QMediaDevices,
)
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QVBoxLayout

from ..analysis.cancellation import AnalysisCancelled
from ..analysis.listening_comparison import (
    MAX_LEVEL_MATCH_DB,
    ComparisonRenderResult,
    RenderedComparisonClip,
    _common_safety_gain_db,
    _playback_safe,
    render_comparison,
)
from ..config import save_config
from .calibration_support import chain_settings
from .capture_session import (
    CaptureSession,
    active_device_identities,
    processor_sample_rate,
    route_identities_match,
    selected_device_identities,
    set_temporary_mute,
    start_selected_route,
    sync_owner_processing_controls,
)
from .components import session_is_ending
from .layout_constants import configure_resizable_dialog
from .listening_comparison_view import ListeningComparisonView, RecordingProgressView
from .quick_scene import QML_DIR, load_scene, quick_theme

logger = logging.getLogger(__name__)

_NO_PLAYBACK_DEVICE_TEXT = (
    "No playback device is available. Connect speakers or headphones, "
    "then reopen this comparison."
)


def _device_key(device: QAudioDevice | None) -> bytes | None:
    if device is None:
        return None
    try:
        return bytes(device.id().data())
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


def _preview_level_match_gain(
    clip: RenderedComparisonClip,
    *,
    level_match: bool,
    rendered_level_matched: bool,
) -> float:
    if not level_match or rendered_level_matched or clip.label == "original":
        return 0.0
    gain = clip.level_match_gain_db
    if not gain and np.isfinite(clip.actual_level_delta_db):
        gain = -float(clip.actual_level_delta_db)
    return float(np.clip(gain, -MAX_LEVEL_MATCH_DB, MAX_LEVEL_MATCH_DB))


def _preview_samples(
    clip: RenderedComparisonClip,
    *,
    level_match: bool,
    common_safety_gain_db: float | None = None,
    rendered_level_matched: bool = False,
) -> np.ndarray:
    """Prepare one already-rendered clip for playback without rerendering."""
    if not level_match and common_safety_gain_db is None:
        return np.ascontiguousarray(clip.samples, dtype=np.float32)
    match_gain = _preview_level_match_gain(
        clip,
        level_match=level_match,
        rendered_level_matched=rendered_level_matched,
    )
    matched, _ = _playback_safe(
        clip.samples,
        level_match_gain_db=match_gain,
        safety_gain_db=common_safety_gain_db,
    )
    return matched


def _preview_safety_gain(
    result: ComparisonRenderResult,
    *,
    level_match: bool,
) -> float:
    """Find one peak-safe gain for all clips in the current playback mode."""
    rendered_level_matched = bool(result.level_match)
    prepared = (
        _preview_samples(
            clip,
            level_match=level_match,
            common_safety_gain_db=0.0,
            rendered_level_matched=rendered_level_matched,
        )
        for clip in result.clips
    )
    return _common_safety_gain_db(prepared)


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
        from ..analysis.signal_processing import resample_poly

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

    progress = Signal(str, int)
    result_ready = Signal(object)
    failed = Signal(str)
    canceled = Signal()

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
            logger.exception("Listening comparison failed")
            self.failed.emit(f"{type(error).__name__}: {error}")


class ListeningComparisonDialog(QDialog):
    """Play one capture as original, current, or proposed processing."""

    _CLIPS: tuple[tuple[str, str], ...] = (
        ("original", "Original recording"),
        ("current", "Current settings"),
        ("proposed", "Proposed settings"),
    )

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
        # The device behind each choice of the output list; the prompt and the notice are None.
        self._choice_devices: list[QAudioDevice | None] = []

        self._load_error = self._setup_ui()
        if self._load_error is not None:
            # Nothing renders or plays; exec() says why and closes.
            logger.error("Listening comparison did not open: %s", self._load_error)
            return
        self._media_devices.audioOutputsChanged.connect(self._on_audio_outputs_changed)
        # The scene's own minimum grows with the text size; the dialog opens at least that wide.
        minimum_width = max(480, int(self.quick_view.rootObject().property("minimumWidth")))
        configure_resizable_dialog(
            self,
            preferred_width=max(680, minimum_width),
            preferred_height=560,
            minimum_width=minimum_width,
            minimum_height=360,
        )
        self._refresh_playback_devices()
        self._start_render()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._wait_for_worker)

    def _setup_ui(self) -> str | None:
        """Build the view model and load the scene; return why it did not load, or None."""
        self.view_model = ListeningComparisonView(
            clips=self._CLIPS,
            pick_output=self._on_output_changed,
            refresh=self._refresh_playback_devices,
            stop=self._stop_playback,
            level_match=self._on_level_match_changed,
            play=self._play_clip,
            keep=lambda: self._request_close(True),
            reject=lambda: self._request_close(False),
            parent=self,
        )
        self.view_model.set_page(
            scope="Same captured passage; playback is optional. Keep the rendered "
            "proposal or close to discard it.",
            scopeWarning="Each processed clip follows its configured mode and enabled stages.",
        )
        self.quick_view = QQuickWidget(self)
        reason = load_scene(
            self.quick_view,
            QML_DIR / "ListeningComparison.qml",
            quick_theme(),
            {"vm": self.view_model},
        )
        self.quick_view.setAccessibleName("Listening comparison content")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.quick_view)
        return reason

    def exec(self) -> int:
        if self._load_error is not None:
            QMessageBox.critical(
                self.parentWidget(),
                self.windowTitle(),
                f"This window could not open.\n\n{self._load_error}",
            )
            self._decision = False
            return int(QDialog.DialogCode.Rejected)
        return super().exec()

    def _refresh_playback_devices(self) -> None:
        self._stop_playback()
        requested_key = _device_key(self._requested_playback_device)
        requested_key_text = (
            requested_key.hex().lower() if requested_key is not None else None
        )
        if requested_key_text is None and self._requested_playback_device_key:
            requested_key_text = self._requested_playback_device_key.strip().lower()
        choices = ["Choose headphones or speakers…"]
        devices: list[QAudioDevice | None] = [None]
        self._outputs = list(QMediaDevices.audioOutputs())
        selected_index = 0
        for index, device in enumerate(self._outputs):
            choices.append(device.description())
            devices.append(device)
            device_key = _device_key(device)
            if requested_key_text is not None and device_key is not None and (
                device_key.hex().lower() == requested_key_text
            ):
                selected_index = index + 1
        if not self._outputs:
            choices.append("No playback devices detected")
            devices.append(None)
            self.view_model.set_page(outputEnabled=False)
            self.view_model.set_status(_NO_PLAYBACK_DEVICE_TEXT, "warn")
        else:
            self.view_model.set_page(outputEnabled=True)
            if self.view_model.status_text == _NO_PLAYBACK_DEVICE_TEXT:
                self.view_model.set_status("")
            if not selected_index and requested_key_text is not None:
                self.view_model.set_status(
                    "The saved preview device is unavailable. Choose another output.", "warn"
                )
        self._choice_devices = devices
        self.view_model.set_outputs(choices, selected_index)
        self._update_playback_buttons()

    def _current_playback_device(self) -> QAudioDevice | None:
        index = self.view_model.output_index
        return self._choice_devices[index] if 0 <= index < len(self._choice_devices) else None

    def _on_output_changed(self) -> None:
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
                self.view_model.set_status(
                    "Preview device selected for this session, but saving it failed.", "warn"
                )
        self._update_playback_buttons()

    def _on_audio_outputs_changed(self) -> None:
        self._refresh_playback_devices()

    def _update_playback_buttons(self) -> None:
        enabled = self._result is not None and self._current_playback_device() is not None
        self.view_model.set_page(playEnabled=enabled, stopEnabled=self._audio_sink is not None)

    def _start_render(self) -> None:
        if self._close_requested:
            return
        if self._worker is not None and self._worker.isRunning():
            self._restart_after_cancel = True
            self._worker.stop()
            return
        self._result = None
        self.view_model.set_page(
            keepEnabled=False, playEnabled=False, progress=0, progressText="Rendering comparison..."
        )
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
            self._show_clip_details(self._result)
            self.view_model.set_status(
                "Speech level matching applied to the next preview playback.", "info"
            )

    def _show_clip_details(self, result: ComparisonRenderResult) -> None:
        level_match = self.view_model.level_match
        safety_gain = _preview_safety_gain(result, level_match=level_match)
        for key, _label in self._CLIPS:
            self.view_model.set_detail(
                key,
                self._clip_details(
                    getattr(result, key),
                    level_match=level_match,
                    common_safety_gain_db=safety_gain,
                    rendered_level_matched=result.level_match,
                ),
            )

    def _on_render_progress(self, message: str, percentage: int) -> None:
        self.view_model.set_page(progressText=message, progress=int(percentage))

    def _on_render_ready(self, result: ComparisonRenderResult) -> None:
        if self._close_requested or self._restart_after_cancel:
            return
        self._result = result
        self.view_model.set_page(progress=100, progressText="Comparison ready", keepEnabled=True)
        self._show_clip_details(result)
        self._update_playback_buttons()
        self.view_model.set_page(scopeWarning="\n".join(
            f"{clip.label.capitalize()}: {clip.render_scope}; "
            + ", ".join(clip.rendered_stages)
            for clip in (result.current, result.proposed)
        ))
        self.view_model.set_status(
            f"{result.scope_label}. Native delay compensation: "
            f"{result.alignment_ms:.1f} ms. "
            f"Speech level source: {result.speech_detection}. "
            "The original clip is the unchanged capture; processed clips use "
            "the same captured microphone signal.",
            "ok" if result.native_authoritative else "warn",
        )

    def _on_render_failed(self, error: str) -> None:
        if self._close_requested:
            return
        self._result = None
        self.view_model.set_page(progressText="Comparison failed")
        self.view_model.set_status(f"Could not render listening comparison: {error}", "bad")

    def _on_render_canceled(self) -> None:
        self.view_model.set_page(progressText="Comparison canceled")

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
        clip: RenderedComparisonClip,
        *,
        level_match: bool = False,
        common_safety_gain_db: float | None = None,
        rendered_level_matched: bool = False,
    ) -> str:
        match_gain = (
            float(clip.level_match_gain_db)
            if level_match
            and rendered_level_matched
            and clip.label != "original"
            else _preview_level_match_gain(
                clip,
                level_match=level_match,
                rendered_level_matched=rendered_level_matched,
            )
        )
        match = (
            f"preview gain {match_gain:+.1f} dB; "
            if level_match and match_gain
            else ""
        )
        if level_match:
            safety_gain = clip.safety_gain_db + (common_safety_gain_db or 0.0)
            return (
                f"Actual level vs original: {clip.actual_level_delta_db:+.1f} dB; "
                f"{match}remaining level difference "
                f"{clip.actual_level_delta_db + match_gain:+.1f} dB; "
                f"shared peak safety gain {safety_gain:+.1f} dB"
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
            level_match = self.view_model.level_match
            safety_gain = _preview_safety_gain(result, level_match=level_match)
            samples = _preview_samples(
                clip,
                level_match=level_match,
                common_safety_gain_db=safety_gain,
                rendered_level_matched=result.level_match,
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
            if sink.error() != QtAudio.Error.NoError:
                raise RuntimeError(f"audio output failed ({sink.error().name})")
            self.view_model.set_status(f"Playing {clip.label} on {device.description()}.", "info")
            self._update_playback_buttons()
        except Exception as error:
            self._stop_playback()
            self.view_model.set_status(f"Playback failed: {error}", "bad")

    @Slot()
    def _on_audio_state_changed(self) -> None:
        if self._audio_sink is None:
            return
        if self._audio_sink.state() in (
            QtAudio.State.IdleState,
            QtAudio.State.StoppedState,
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
        self.view_model.set_page(stopEnabled=False)

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
            self.view_model.set_page(
                keepEnabled=False, rejectEnabled=False, progressText="Canceling comparison..."
            )
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
        self._close_requested = False
        QDialog.done(self, int(QDialog.DialogCode.Accepted if self._close_result else QDialog.DialogCode.Rejected))

    def closeEvent(self, event) -> None:
        if session_is_ending():
            # A veto here would stop Windows from signing out: stop playback and the render, close.
            self._wait_for_worker()
            event.accept()
            return
        self._request_close(False)
        event.ignore()

    def accept(self) -> None:
        self._request_close(True)

    def reject(self) -> None:
        self._request_close(False)

    @property
    def decision(self) -> bool | None:
        return self._decision


class RawProcessedDialog(ListeningComparisonDialog):
    """Play one capture raw or through the current settings; nothing to decide."""

    _CLIPS = (("original", "Raw"), ("current", "Processed"))

    def __init__(
        self,
        parent=None,
        *,
        audio_data: np.ndarray,
        sample_rate: int,
        settings: Mapping[str, Any],
        chain_settings: Mapping[str, Any] | None = None,
        playback_device: QAudioDevice | None = None,
        playback_device_key: str | None = None,
    ) -> None:
        # shortcut: the shared worker renders the same settings twice (current
        # and proposed), add a single-render path to render_comparison if a
        # five-second clip ever takes noticeably long.
        super().__init__(
            parent,
            audio_data=audio_data,
            sample_rate=sample_rate,
            current_settings=settings,
            proposed_settings=settings,
            current_chain_settings=chain_settings,
            proposed_chain_settings=chain_settings,
            playback_device=playback_device,
            playback_device_key=playback_device_key,
        )
        self.setWindowTitle("Test my sound")

    def _setup_ui(self) -> str | None:
        reason = super()._setup_ui()
        self.view_model.set_page(
            pageName="Test my sound page",
            scope="This is one recording of your microphone. Raw is what the "
            "microphone picked up. Processed is the same recording with your "
            "current settings.",
            scopeWarning="Preparing the processed clip.",
            keepShown=False,
            rejectText="Close",
            rejectName="Close test my sound",
        )
        self.quick_view.setAccessibleName("Test my sound content")
        return reason

    def _play_clip(self, key: str, *, start_sample: int = 0) -> None:
        # The playing clip's button reads Stop, so pressing it again stops.
        if key == self._playing_clip_key:
            self._stop_playback()
            return
        super()._play_clip(key, start_sample=start_sample)

    def _update_playback_buttons(self) -> None:
        super()._update_playback_buttons()
        self._show_playing_clip()

    def _stop_playback(self) -> None:
        super()._stop_playback()
        self._show_playing_clip()

    def _show_playing_clip(self) -> None:
        playing = self._playing_clip_key
        for key, label in self._CLIPS:
            self.view_model.set_play_label(key, f"Stop {label}" if key == playing else f"Play {label}")
        names = dict(self._CLIPS)
        if playing in names:
            self.view_model.set_status(f"Playing: {names[playing]}", "info")
        elif self.view_model.status_text.startswith("Playing: "):
            self.view_model.set_status("Playback stopped.")

    def _on_render_ready(self, result: ComparisonRenderResult) -> None:
        super()._on_render_ready(result)
        if self._result is not result:
            return
        stages = ", ".join(result.current.rendered_stages) or "none"
        self.view_model.set_page(
            progressText="Ready to play",
            scopeWarning=f"Stages applied to the processed clip: {stages}.",
        )
        self.view_model.set_status(
            "Play each clip to hear the difference. You can switch while one is playing.", "ok"
        )

    def _on_render_failed(self, error: str) -> None:
        super()._on_render_failed(error)
        if self._close_requested:
            return
        self.view_model.set_page(progressText="Processing failed")
        self.view_model.set_status(f"Could not process the recording: {error}")


TEST_CAPTURE_SECONDS = 5.0
_TEST_MY_SOUND_REASON = "test_my_sound"


def _show_recording_progress(dialog: QDialog) -> RecordingProgressView:
    """Give the recording window its scene; raise when the scene does not load."""
    dialog.setWindowTitle("Test my sound")
    dialog.setModal(True)
    view = RecordingProgressView(
        prompt=f"Recording. Speak normally for {TEST_CAPTURE_SECONDS:.0f} seconds.",
        cancel=dialog.reject,
        parent=dialog,
    )
    quick_view = QQuickWidget(dialog)
    reason = load_scene(quick_view, QML_DIR / "RecordingProgress.qml", quick_theme(), {"vm": view})
    if reason is not None:
        raise RuntimeError(f"The recording window could not open: {reason}")
    quick_view.setAccessibleName("Test my sound recording")
    quick_view.setMinimumSize(quick_view.sizeHint())
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(quick_view)
    return view


def _record_test_clip(window: Any) -> np.ndarray | None:
    """Record one raw clip behind a modal progress dialog; None when canceled."""
    session = CaptureSession(window, _TEST_MY_SOUND_REASON)
    session.start(TEST_CAPTURE_SECONDS, before_cleanup=True)

    dialog = QDialog(window)
    failures: list[str] = []
    timer = QTimer(dialog)
    timer.setInterval(50)
    try:
        view = _show_recording_progress(dialog)

        def poll() -> None:
            try:
                progress = float(window.processor.recording_progress())
                failure = session.failure_reason(progress)
                if failure is not None:
                    raise RuntimeError(failure)
                view.set_progress(int(progress * 100))
                if progress >= 1.0 or window.processor.is_recording_complete():
                    timer.stop()
                    dialog.accept()
            except Exception as exc:
                timer.stop()
                failures.append(str(exc))
                dialog.reject()

        timer.timeout.connect(poll)
        timer.start()
        accepted = dialog.exec() == int(QDialog.DialogCode.Accepted)
        if failures:
            raise RuntimeError(failures[0])
        if not accepted:
            return None
        audio = session.stop_recording()
        return np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
    finally:
        timer.stop()
        session.cleanup()
        dialog.deleteLater()


def open_test_my_sound(window: Any) -> None:
    """Record a short clip, then let the user play it raw and processed."""

    def fail(message: str) -> None:
        QMessageBox.warning(window, "Test my sound", message)

    processor = getattr(window, "processor", None)
    if processor is None:
        fail("Audio processing is not available.")
        return
    selected = selected_device_identities(window)
    if selected[0] is None:
        fail("No microphone is selected. Choose an input device, then try again.")
        return

    started_here = False
    try:
        if not processor.is_running():
            start_selected_route(window)
            started_here = True
            sync_owner_processing_controls(window)
        elif not route_identities_match(selected, active_device_identities(processor)):
            fail(
                "Processing is running on different devices than the ones "
                "selected. Stop and start processing, then try again."
            )
            return
    except Exception as exc:
        fail(
            f"Could not start the microphone.\n\n{exc}\n\n"
            "Check that your devices are connected and not in use by another app."
        )
        return

    try:
        try:
            audio = _record_test_clip(window)
            sample_rate = processor_sample_rate(window)
        finally:
            if started_here:
                processor.stop()
                sync_owner_processing_controls(window)
    except Exception as exc:
        fail(f"Recording did not work.\n\n{exc}")
        return
    if audio is None:
        return
    if audio.size == 0:
        fail("The recording was empty. Check the microphone and try again.")
        return

    try:
        dialog = RawProcessedDialog(
            window,
            audio_data=audio,
            sample_rate=sample_rate,
            settings=window.eq_state.get_settings(),
            chain_settings=chain_settings(
                window, full_chain=True, input_pre_filtered=False
            ),
        )
    except Exception as exc:
        fail(f"Could not prepare the processed clip.\n\n{exc}")
        return
    # Keep the preview out of the live output while it plays near the microphone.
    set_temporary_mute(window, _TEST_MY_SOUND_REASON, True)
    try:
        dialog.exec()
    finally:
        set_temporary_mute(window, _TEST_MY_SOUND_REASON, False)
        dialog.deleteLater()


__all__ = [
    "ListeningComparisonDialog",
    "ListeningComparisonWorker",
    "RawProcessedDialog",
    "open_test_my_sound",
]
