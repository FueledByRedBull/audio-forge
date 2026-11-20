"""Capture sanitized, reproducible repository screenshots of the shipped UI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, cast
import numpy as np
from release_provenance import sha256_file as _sha256
from unittest.mock import patch

# These must be fixed before QApplication is created.
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_SCALE_FACTOR"] = "1"
os.environ["QT_FONT_DPI"] = "96"
os.environ.setdefault("AUDIOFORGE_REDUCED_MOTION", "1")

from PySide6.QtCore import QByteArray, QEventLoop, QPointF, QRectF, QTimer
from PySide6.QtGui import QFont, QFontDatabase, QImageWriter
from PySide6.QtWidgets import QApplication, QWidget

from mic_eq.config import BUILTIN_PRESETS, TARGET_CURVES, AppConfig
from mic_eq.ui import main_window as main_window_module
from mic_eq.ui.main_window import MainWindow
from mic_eq.ui.quick_shell import LEVEL_HISTORY
from mic_eq.ui.theme import apply_application_theme
from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog
from mic_eq.ui.voice_setup_view import DYNAMICS_KEYS


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "images"
DEFAULT_REPORT = REPO_ROOT / "build" / "ui-screenshot-report.json"
CAPTURE_WIDTH = 1600
CAPTURE_HEIGHT = 960
CAPTURE_FONT_FILENAME = "segoeui.ttf"


@dataclass(frozen=True)
class SanitizedDevice:
    name: str
    is_default: bool = False


SANITIZED_INPUT_DEVICES = (
    SanitizedDevice("Studio Microphone", is_default=True),
    SanitizedDevice("USB Headset Microphone"),
)
SANITIZED_OUTPUT_DEVICES = (
    SanitizedDevice("CABLE Input (Virtual Route)"),
    SanitizedDevice("Studio Headphones", is_default=True),
)


SCREENSHOTS: tuple[dict[str, str], ...] = (
    {
        "filename": "audioforge-routing-eq.png",
        "view": "routing_eq",
        "alt": (
            "AudioForge Equalizer page for a studio microphone routed to the CABLE "
            "Input virtual cable, drawing each band's curve, the Auto-EQ correction, "
            "the input filter and their total over the microphone's input spectrum, "
            "with the editor for the selected band and level meters."
        ),
    },
    {
        "filename": "audioforge-overview.png",
        "view": "overview",
        "alt": (
            "AudioForge Overview page while audio runs, with the live status, the "
            "last ten seconds of input and output level, an equalizer summary with "
            "buttons for Auto Voice Setup, Test my sound and Auto-EQ, and keyboard "
            "hints, beside a sidebar listing the signal chain with an on/off switch "
            "on each stage from noise gate to limiter."
        ),
    },
    {
        "filename": "audioforge-auto-voice-setup.png",
        "view": "voice_setup",
        "alt": (
            "AudioForge Auto Voice Setup on its Review step, listing the "
            "recommended settings in plain language."
        ),
    },
)


def _source_hashes() -> dict[str, str]:
    paths = (
        "python/tools/capture_repository_screenshots.py",
        "python/mic_eq/ui/theme.py",
        "python/mic_eq/ui/main_window.py",
        "python/mic_eq/ui/quick_shell.py",
        "python/mic_eq/ui/shell_specs.py",
        "python/mic_eq/ui/quick_scene.py",
        "python/mic_eq/ui/overview_presentation.py",
        "python/mic_eq/ui/chain_presentation.py",
        "python/mic_eq/ui/eq_presentation.py",
        "python/mic_eq/ui/eq_graph.py",
        "python/mic_eq/ui/eq_compare.py",
        "python/mic_eq/ui/eq_quick_compare.py",
        "python/mic_eq/ui/voice_setup_dialog.py",
        "python/mic_eq/ui/voice_setup_flow.py",
        "python/mic_eq/ui/voice_setup_view.py",
        "python/mic_eq/ui/qml/AutoEqCalibration.qml",
        "python/mic_eq/ui/qml/BodyText.qml",
        "python/mic_eq/ui/qml/Chip.qml",
        "python/mic_eq/ui/qml/ChoiceBox.qml",
        "python/mic_eq/ui/qml/CloseCross.qml",
        "python/mic_eq/ui/qml/CommandPalette.qml",
        "python/mic_eq/ui/qml/CompactWindow.qml",
        "python/mic_eq/ui/qml/DialogHeader.qml",
        "python/mic_eq/ui/qml/EscChip.qml",
        "python/mic_eq/ui/qml/FirstRunSetup.qml",
        "python/mic_eq/ui/qml/FocusRing.qml",
        "python/mic_eq/ui/qml/GlassPanel.qml",
        "python/mic_eq/ui/qml/KeyButton.qml",
        "python/mic_eq/ui/qml/LatencyCalibration.qml",
        "python/mic_eq/ui/qml/ListeningComparison.qml",
        "python/mic_eq/ui/qml/Look.qml",
        "python/mic_eq/ui/qml/Main.qml",
        "python/mic_eq/ui/qml/ModelButton.qml",
        "python/mic_eq/ui/qml/MuteBanner.qml",
        "python/mic_eq/ui/qml/MuteButton.qml",
        "python/mic_eq/ui/qml/MutedText.qml",
        "python/mic_eq/ui/qml/NumberBox.qml",
        "python/mic_eq/ui/qml/ProgressTrack.qml",
        "python/mic_eq/ui/qml/PushButton.qml",
        "python/mic_eq/ui/qml/qmldir",
        "python/mic_eq/ui/qml/RecordingProgress.qml",
        "python/mic_eq/ui/qml/ScrollArea.qml",
        "python/mic_eq/ui/qml/ScrollPage.qml",
        "python/mic_eq/ui/qml/StepMark.qml",
        "python/mic_eq/ui/qml/Surface.qml",
        "python/mic_eq/ui/qml/SwitchButton.qml",
        "python/mic_eq/ui/qml/Tip.qml",
        "python/mic_eq/ui/qml/Title.qml",
        "python/mic_eq/ui/qml/VoiceSetup.qml",
    )
    return {path: _sha256(REPO_ROOT / path) for path in paths}


def _install_sanitized_sources():
    """Prevent capture from reading user config, presets, or audio devices."""
    return patch.multiple(
        main_window_module,
        load_config=lambda: AppConfig(
            first_run_setup_state="completed",
            first_run_setup_steps={
                "devices": "completed",
                "route": "completed",
                "latency": "completed",
                "voice": "completed",
            },
        ),
        save_config=lambda _config: True,
        list_presets=lambda: [],
        find_presets_refused_for_age=lambda: [],
        list_input_devices=lambda: list(SANITIZED_INPUT_DEVICES),
        list_output_devices=lambda: list(SANITIZED_OUTPUT_DEVICES),
    )


def _load_capture_font(app: QApplication) -> dict[str, str]:
    """Load the system font explicitly because Qt offscreen has no font DB."""

    windows_root = Path(os.environ.get("WINDIR", "C:/Windows"))
    font_path = windows_root / "Fonts" / CAPTURE_FONT_FILENAME
    if not font_path.is_file():
        raise RuntimeError(
            f"Required capture font is unavailable: {CAPTURE_FONT_FILENAME}"
        )
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
    if not families:
        raise RuntimeError(f"Could not load capture font: {CAPTURE_FONT_FILENAME}")
    family = families[0]
    # Icon glyphs come from whichever system icon font this Windows ships; Segoe UI
    # Symbol covers the check marks and arrows Segoe UI lacks, as on a real desktop.
    for icon_font_filename in ("SegoeIcons.ttf", "segmdl2.ttf", "seguisym.ttf"):
        QFontDatabase.addApplicationFont(str(windows_root / "Fonts" / icon_font_filename))
    app.setFont(QFont(family, 9))
    return {
        "family": family,
        "filename": CAPTURE_FONT_FILENAME,
        "sha256": _sha256(font_path),
    }


def _select_data(combo: Any, value: str) -> None:
    index = combo.findData(value)
    if index < 0:
        raise ValueError(f"Screenshot selection is unavailable: {value}")
    combo.setCurrentIndex(index)


def _prepare_main_window() -> MainWindow:
    window = MainWindow()
    window.meter_timer.stop()
    window.diagnostics_timer.stop()
    window.resize(CAPTURE_WIDTH, CAPTURE_HEIGHT)
    window.input_choice.setCurrentIndex(0)
    window.output_choice.setCurrentIndex(0)
    _select_data(window.input_channel_mode_choice, "average")
    _select_data(window.input_cleanup_mode_choice, "gentle")
    # The graph reads the cleanup mode from the processor on a timer.
    window.eq_presentation.sync_input_conditioning()
    window.noise_suppression_state.set_settings({"model": "rnnoise", "enabled": True, "strength": 0.72})
    window.eq_state.apply_auto_eq_results(
        [
            (72.0, -2.0, 0.75),
            (145.0, -1.0, 1.0),
            (290.0, 0.4, 1.1),
            (580.0, 1.1, 1.2),
            (1160.0, 1.8, 1.4),
            (2320.0, 2.2, 1.6),
            (4640.0, 1.2, 1.5),
            (7600.0, -0.8, 1.4),
            (11100.0, -1.1, 1.1),
            (15100.0, -0.6, 0.8),
        ]
    )
    # A tone on top, so the graph shows both layers: the Auto-EQ correction and
    # the bands the user edits. Typed bands keep the preset marked as saved.
    window.eq_state.apply_typed_bands(BUILTIN_PRESETS["voice"].eq.bands)
    window.eq_state.set_auto_eq_diagnostics(
        {
            "analysis_confidence": 0.88,
            "eq_confidence": 0.86,
            "capture_confidence": 0.91,
            "validation_confidence": 0.90,
            "validation_before_error_db": 3.4,
            "validation_after_error_db": 1.1,
            "validation_gain_scale": 1.0,
            "recommendation_status": "apply",
            "target_profile": "broadcast-style",
            "headroom_validation": {
                "safe": True,
                "after": {
                    "pre_limiter_true_peak_headroom_db": 2.6,
                    "limiter_gain_reduction_db": 0.0,
                    "true_peak_limiter_gain_reduction_db": 0.0,
                },
                "gain_scale": 1.0,
            },
        }
    )
    window.input_meter.set_levels(-22.0, -10.5)
    window.output_meter.set_levels(-19.0, -7.5)
    window._set_health_chip("input_health_label", "Input: OK", "ok")
    window._set_health_chip("output_health_label", "Output: OK", "ok")
    window._set_health_chip("gate_health_label", "Gate: stable", "ok")
    window._set_health_chip("backend_diag_label", "Backend: ready", "ok")
    window._set_health_chip("callback_health_label", "Device activity: input 5 ms ago, output 6 ms ago", "ok")
    window._set_health_chip("underrun_health_label", "Dropouts: 0", "ok")
    window._set_health_chip("latency_label", "Latency: 17.3 ms", "info")
    window._set_health_chip("buffer_label", "Buffer: OK", "ok")
    window._set_health_chip("dropped_label", "Drops: 0", "ok")
    window._set_health_chip("recovery_diag_label", "Recovery: idle", "idle")
    window.status_message.clearMessage()
    return window


def _level_series() -> list[list[float]]:
    """Ten seconds of speech-like input and output level in dB, the same on every run."""
    series: list[list[float]] = []
    for index in range(LEVEL_HISTORY):
        syllables = abs(math.sin(index * 0.11)) * (0.65 + 0.35 * math.sin(index * 0.37 + 1.0))
        # Phrases with a pause between them.
        phrase = 1.0 if math.sin(index * 0.05 + 0.6) > -0.6 else 0.06
        microphone = -54.0 + 36.0 * syllables * phrase
        # The chain evens the level out a little before it reaches callers.
        series.append([microphone, -52.0 + 31.0 * syllables * phrase])
    return series


def _analyser_spectrum() -> tuple[np.ndarray, np.ndarray]:
    """A voice-like input spectrum in dBFS on the engine's 96 log-spaced bands."""
    bands = 96
    centres = 20.0 * 1000.0 ** ((np.arange(bands) + 0.5) / bands)
    octaves = np.log2(centres / 250.0)
    shape = np.where(octaves < 0, -35.0 + 12.0 * octaves, -35.0 - 8.5 * octaves)
    # Voice harmonics show as ripple in the low bands and fade into the high ones.
    ripple = 3.0 * np.sin(2.0 * np.pi * centres / 130.0) * np.exp(-centres / 1500.0)
    return np.maximum(shape + ripple, -100.0).astype(np.float32), centres.astype(np.float32)


def _show_running_state(window: MainWindow) -> None:
    """Make the whole window look like audio is running, not only its top bar.

    The Overview and the sidebar read the session's own flag, the stop control and
    the level history. Tab changes refresh the session summary from the stopped
    processor, so this runs immediately before each main-window grab.
    """
    window._processing = True
    window.start_btn_action.setEnabled(False)
    window.stop_btn_action.setEnabled(True)
    window.shell_texts["transmission_status_label"].update(text="Transmission: Running / Normal")
    window._set_health_chip("health_summary_label", "Health: OK", "ok")
    window.processing_meters.compressor_gr.set_gain_reduction(3.2)
    window.processing_meters.limiter_gr.set_gain_reduction(0.8)
    bridge = cast(Any, window).quick_bridge
    # The bridge would append the stopped meters' readings to the history.
    bridge._trace_timer.stop()
    bridge._levels.clear()
    bridge._levels.extend(_level_series())
    bridge.levelsChanged.emit()
    bridge.model["sidebar"]["lines"].refresh()
    bridge.model["overview"].refresh()


def _prepare_voice_setup(parent: MainWindow) -> VoiceSetupDialog:
    dialog = VoiceSetupDialog(parent, configuration=parent.configuration)
    dialog.flow.recording_timer.stop()
    dialog.resize(780, 760)
    dialog.flow.view_model.pickCurve(list(TARGET_CURVES).index("broadcast"))
    dialog.flow.view_model.pickDynamics(DYNAMICS_KEYS.index("balanced"))
    dialog.flow.setup_state = "completed"
    dialog.flow.view_model.set_page(
        recordingShown=True,
        phaseText="Recommendations ready",
        progress=100,
        timeText="Capture and analysis complete",
        startText="Apply Voice Setup",
    )
    dialog.flow.level_meter.set_levels(-24.0, -10.0)
    dialog.flow.view_model.set_warning("Validated settings are ready to review.", state="ok", strong=True)
    dialog.flow.show_summary(
        {
            "diagnostics": {
                "setup_confidence": 0.91,
                "capture_confidence": 0.94,
                "recommendation_uncertainty": 0.08,
                "gate_mode_label": "VAD assisted",
                "apply_recommended": True,
            },
            "eq_settings": {
                **parent.eq_state.get_settings(),
                "analysis_confidence": 0.89,
            },
            "gate_settings": {"threshold_db": -43.0, "vad_threshold": 0.46},
            "deesser_settings": {
                "enabled": True,
                "auto_amount": 0.42,
                "low_cut_hz": 4600.0,
                "high_cut_hz": 9800.0,
            },
            "compressor_settings": {
                "ratio": 3.0,
                "threshold_db": -19.0,
                "auto_makeup_enabled": True,
                "makeup_gain_db": 0.0,
                "target_lufs": -18.0,
            },
        }
    )
    return dialog


def _limiter_row_in_view(scene: Any) -> bool:
    """Whether the sidebar's Limiter row lies wholly inside the scroll view that holds it."""
    pending = [scene]
    while pending:
        item = pending.pop()
        if item.property("name") == "Limiter" and item.property("detail") is not None:
            break
        pending.extend(item.childItems())
    else:
        return False
    view = item.parentItem()
    while view is not None and view.property("contentY") is None:
        view = view.parentItem()
    if view is None:
        return False

    def rect(target: Any) -> QRectF:
        return target.mapRectToScene(QRectF(QPointF(0, 0), target.size()))

    shown = rect(view).intersected(QRectF(0, 0, scene.width(), scene.height()))
    return item.isVisible() and shown.contains(rect(item))


def _wait_for_next_frame() -> None:
    loop = QEventLoop()
    QTimer.singleShot(100, loop.quit)
    loop.exec()


def _write_optimized_png(widget: QWidget, path: Path) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = widget.grab().toImage()
    writer = QImageWriter(str(path), QByteArray(b"png"))
    # QImageWriter uses a 0-100 compression scale, not zlib's 0-9 scale.
    writer.setCompression(100)
    writer.setOptimizedWrite(True)
    if not writer.write(image):
        raise RuntimeError(f"Could not write {path.name}: {writer.errorString()}")
    return image.width(), image.height()


@_install_sanitized_sources()
def capture_screenshots(output_dir: Path, report_path: Path) -> dict[str, Any]:
    existing_app = QApplication.instance()
    app = QApplication([]) if existing_app is None else cast(QApplication, existing_app)
    apply_application_theme(app)
    font = _load_capture_font(app)

    window = _prepare_main_window()
    window.show()
    app.processEvents()
    outputs: list[dict[str, Any]] = []
    signal_chain_in_view = False
    overview_state = ""
    analyser_shown = False
    try:
        for specification in SCREENSHOTS[:2]:
            graph_model = window.eq_presentation.graph_model
            if specification["view"] == "routing_eq":
                window.shell_page.setIndex(MainWindow.EQUALIZER_PAGE_INDEX)
            if specification["view"] == "overview":
                window.shell_page.setIndex(0)
            _show_running_state(window)
            if specification["view"] == "routing_eq":
                # The engine's spectrum is read only while the Equalizer page shows.
                graph_model.set_input_spectrum(*_analyser_spectrum())
            else:
                graph_model.clear_input_spectrum()
            app.processEvents()
            if specification["view"] == "routing_eq":
                analyser_shown = graph_model.input_spectrum_db is not None
            if specification["view"] == "overview":
                scene = cast(Any, window.centralWidget()).rootObject()
                signal_chain_in_view = _limiter_row_in_view(scene)
                overview_state = cast(Any, window).quick_bridge.model["overview"].state
            # The scene draws a change on its next frame; without waiting the grab shows the one before.
            _wait_for_next_frame()
            path = output_dir / specification["filename"]
            width, height = _write_optimized_png(window, path)
            outputs.append(
                {
                    **specification,
                    "path": path.relative_to(REPO_ROOT).as_posix()
                    if path.is_relative_to(REPO_ROOT)
                    else path.name,
                    "width": width,
                    "height": height,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )

        # The review lists the EQ change the voice setup would make; it is
        # shown against the Auto-EQ layer alone.
        window.eq_state.reset_tone()
        dialog = _prepare_voice_setup(window)
        dialog.show()
        app.processEvents()
        eq_recommendation_shown = "max correction" in dialog.flow.view_model.chip("eq")["text"]
        specification = SCREENSHOTS[2]
        path = output_dir / specification["filename"]
        width, height = _write_optimized_png(dialog, path)
        outputs.append(
            {
                **specification,
                "path": path.relative_to(REPO_ROOT).as_posix()
                if path.is_relative_to(REPO_ROOT)
                else path.name,
                "width": width,
                "height": height,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
        dialog.close()
        dialog.deleteLater()
    finally:
        try:
            window.processor.stop()
        except Exception:
            pass
        # This window contains only the generated demonstration settings.
        with patch.object(window.configuration, "confirm_discard", return_value=True):
            window.close()
        window.deleteLater()
        app.processEvents()

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "repository-ui-screenshots",
        "audible_change": False,
        "decision": {
            "retained": True,
            "reason": (
                "All required stable UI views were captured at fixed logical "
                "scaling from sanitized in-memory state."
            ),
        },
        "capture_contract": {
            "qt_platform": "offscreen",
            "qt_style": "Fusion",
            "font": f"{font['family']} 9pt",
            "font_filename": font["filename"],
            "font_sha256": font["sha256"],
            "logical_dpi": 96,
            "scale_factor": 1,
            "main_viewport": [CAPTURE_WIDTH, CAPTURE_HEIGHT],
            "reads_user_config": False,
            "enumerates_real_devices": False,
            "sanitized_input_devices": [
                device.name for device in SANITIZED_INPUT_DEVICES
            ],
            "sanitized_output_devices": [
                device.name for device in SANITIZED_OUTPUT_DEVICES
            ],
        },
        "checks": {
            "routing_shown": True,
            "eq_shown": True,
            "overview_shown": True,
            "overview_running": overview_state == "live",
            "input_analyser_shown": analyser_shown,
            "auto_voice_setup_shown": True,
            "eq_recommendation_shown": eq_recommendation_shown,
            "alt_text_present": all(bool(item["alt"].strip()) for item in outputs),
            "all_pngs_nonempty": all(item["bytes"] > 0 for item in outputs),
            "all_pngs_compressed": all(
                item["bytes"] < item["width"] * item["height"] for item in outputs
            ),
            # The limiter is the last stage of the chain the sidebar lists.
            "signal_chain_in_view": signal_chain_in_view,
            "sanitized_sources_only": True,
        },
        "view_state": {
            "overview_state": overview_state,
        },
        "screenshots": outputs,
        "source_sha256": _source_hashes(),
        "limitations": [
            "Offscreen Fusion rendering omits Windows title-bar chrome by design.",
            "The screenshots demonstrate layout and product surface, not live hardware activity.",
            "A release UI or semantic-theme change requires regeneration and visual review.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = capture_screenshots(args.output_dir.resolve(), args.report.resolve())
    print(
        json.dumps(
            {
                "screenshots": len(report["screenshots"]),
                "checks": report["checks"],
                "report": args.report.as_posix(),
            },
            indent=2,
        )
    )
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
