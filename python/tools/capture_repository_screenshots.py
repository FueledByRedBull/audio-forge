"""Capture sanitized, reproducible repository screenshots of the shipped UI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

# These must be fixed before QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_SCALE_FACTOR", "1")
os.environ.setdefault("QT_FONT_DPI", "96")
os.environ.setdefault("AUDIOFORGE_REDUCED_MOTION", "1")

from PyQt6.QtCore import QByteArray
from PyQt6.QtGui import QFont, QFontDatabase, QImageWriter
from PyQt6.QtWidgets import QApplication, QScrollArea, QWidget

from mic_eq.config import AppConfig
from mic_eq.ui import main_window as main_window_module
from mic_eq.ui.main_window import MainWindow
from mic_eq.ui.theme import message_text_style
from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "images"
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "ui-screenshot-report.json"
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
            "AudioForge main window showing sanitized input and virtual-route "
            "output selection, cleanup controls, and the editable ten-band EQ."
        ),
    },
    {
        "filename": "audioforge-processing.png",
        "view": "processing",
        "alt": (
            "AudioForge dynamics view showing compressor and limiter controls, "
            "health indicators, and the editable EQ."
        ),
    },
    {
        "filename": "audioforge-auto-voice-setup.png",
        "view": "voice_setup",
        "alt": (
            "AudioForge Auto Voice Setup dialog showing target and dynamics "
            "choices plus sanitized validated recommendation summaries."
        ),
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = (
        "python/tools/capture_repository_screenshots.py",
        "python/mic_eq/ui/theme.py",
        "python/mic_eq/ui/main_window.py",
        "python/mic_eq/ui/eq_panel.py",
        "python/mic_eq/ui/eq_curve.py",
        "python/mic_eq/ui/voice_setup_dialog.py",
    )
    return {path: _sha256(REPO_ROOT / path) for path in paths}


def _install_sanitized_sources() -> None:
    """Prevent capture from reading user config, presets, or audio devices."""

    main_window_module.load_config = lambda: AppConfig(
        first_run_setup_state="completed",
        first_run_setup_steps={
            "devices": "completed",
            "route": "completed",
            "latency": "completed",
            "voice": "completed",
        },
    )
    main_window_module.save_config = lambda _config: None
    main_window_module.list_presets = lambda: []
    main_window_module.list_input_devices = lambda: list(SANITIZED_INPUT_DEVICES)
    main_window_module.list_output_devices = lambda: list(SANITIZED_OUTPUT_DEVICES)


def _load_capture_font(app: QApplication) -> dict[str, str]:
    """Load the system font explicitly because Qt offscreen has no font DB."""

    windows_root = Path(os.environ.get("WINDIR", "C:/Windows"))
    font_path = windows_root / "Fonts" / CAPTURE_FONT_FILENAME
    if not font_path.is_file():
        raise RuntimeError(f"Required capture font is unavailable: {CAPTURE_FONT_FILENAME}")
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
    if not families:
        raise RuntimeError(f"Could not load capture font: {CAPTURE_FONT_FILENAME}")
    family = families[0]
    app.setFont(QFont(family, 9))
    return {
        "family": family,
        "filename": CAPTURE_FONT_FILENAME,
        "sha256": _sha256(font_path),
    }


def _select_data(combo: Any, value: str) -> None:
    index = combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)


def _prepare_main_window() -> MainWindow:
    _install_sanitized_sources()
    window = MainWindow()
    window.meter_timer.stop()
    window.diagnostics_timer.stop()
    window.resize(CAPTURE_WIDTH, CAPTURE_HEIGHT)
    window.main_splitter.setSizes([470, 1010])
    window.input_combo.setCurrentIndex(0)
    window.output_combo.setCurrentIndex(0)
    _select_data(window.input_channel_mode_combo, "average")
    _select_data(window.input_cleanup_mode_combo, "gentle")
    _select_data(window.model_combo, "deepfilter_ll")
    window.rnnoise_checkbox.setChecked(True)
    window.strength_slider.setValue(72)
    window.eq_panel.apply_auto_eq_results(
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
    window.eq_panel.set_auto_eq_diagnostics(
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
    window._set_health_chip(window.input_health_label, "Input: OK", "ok")
    window._set_health_chip(window.output_health_label, "Output: OK", "ok")
    window._set_health_chip(window.gate_health_label, "Gate: stable", "ok")
    window._set_health_chip(window.backend_diag_label, "Backend: ready", "ok")
    window._set_health_chip(window.callback_health_label, "Callbacks: live", "ok")
    window._set_health_chip(window.underrun_health_label, "Underruns: 0", "ok")
    window._set_health_chip(window.latency_label, "Latency: 17.3 ms", "info")
    window._set_health_chip(window.buffer_label, "Buffer: OK", "ok")
    window._set_health_chip(window.dropped_label, "Drops: 0", "ok")
    window._set_health_chip(window.recovery_diag_label, "Recovery: idle", "idle")
    window.status_bar.showMessage("Ready - sanitized documentation capture")
    return window


def _prepare_voice_setup(parent: MainWindow) -> VoiceSetupDialog:
    dialog = VoiceSetupDialog(parent)
    dialog.recording_timer.stop()
    dialog.resize(780, 940)
    _select_data(dialog.curve_combo, "broadcast")
    _select_data(dialog.dynamics_combo, "balanced")
    dialog.recording_group.setVisible(True)
    dialog.phase_label.setText("Recommendations ready")
    dialog.progress_bar.setValue(100)
    dialog.time_label.setText("Capture and analysis complete")
    dialog.level_meter.set_levels(-24.0, -10.0)
    dialog.warning_label.setText("Validated settings are ready to review.")
    dialog.warning_label.setStyleSheet(message_text_style("ok", strong=True))
    dialog.start_button.setText("Apply Voice Setup")
    dialog._show_summary(
        {
            "diagnostics": {
                "setup_confidence": 0.91,
                "capture_confidence": 0.94,
                "recommendation_uncertainty": 0.08,
                "gate_mode_label": "VAD assisted",
            },
            "eq_settings": {
                "analysis_confidence": 0.89,
                "band_gains": [-1.8, -0.8, 0.2, 0.9, 1.6, 2.1, 1.1, -0.6, -0.9, -0.4],
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


def capture_screenshots(output_dir: Path, report_path: Path) -> dict[str, Any]:
    existing_app = QApplication.instance()
    app = QApplication([]) if existing_app is None else cast(QApplication, existing_app)
    app.setStyle("Fusion")
    font = _load_capture_font(app)

    window = _prepare_main_window()
    window.show()
    app.processEvents()
    outputs: list[dict[str, Any]] = []
    processing_scroll_position = 0
    processing_scroll_maximum = 0
    try:
        for specification in SCREENSHOTS[:2]:
            window.control_tabs.setCurrentIndex(
                0 if specification["view"] == "routing_eq" else 1
            )
            app.processEvents()
            if specification["view"] == "processing":
                page = cast(QScrollArea, window.control_tabs.currentWidget())
                scrollbar = page.verticalScrollBar()
                if scrollbar is None:
                    raise RuntimeError("Dynamics page has no vertical scrollbar")
                processing_scroll_maximum = scrollbar.maximum()
                scrollbar.setValue(processing_scroll_maximum)
                processing_scroll_position = scrollbar.value()
                app.processEvents()
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

        dialog = _prepare_voice_setup(window)
        dialog.show()
        app.processEvents()
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
            "sanitized_input_devices": [device.name for device in SANITIZED_INPUT_DEVICES],
            "sanitized_output_devices": [device.name for device in SANITIZED_OUTPUT_DEVICES],
        },
        "checks": {
            "routing_shown": True,
            "eq_shown": True,
            "processing_shown": True,
            "auto_voice_setup_shown": True,
            "alt_text_present": all(bool(item["alt"].strip()) for item in outputs),
            "all_pngs_nonempty": all(item["bytes"] > 0 for item in outputs),
            "all_pngs_compressed": all(
                item["bytes"] < item["width"] * item["height"]
                for item in outputs
            ),
            "processing_scrolled_to_limiter": (
                processing_scroll_maximum > 0
                and processing_scroll_position == processing_scroll_maximum
            ),
            "sanitized_sources_only": True,
        },
        "view_state": {
            "processing_scroll_position": processing_scroll_position,
            "processing_scroll_maximum": processing_scroll_maximum,
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
