"""Contracts for sanitized repository screenshot generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QWidget

from tools.capture_repository_screenshots import (
    SANITIZED_INPUT_DEVICES,
    SANITIZED_OUTPUT_DEVICES,
    SCREENSHOTS,
    _write_optimized_png,
    capture_screenshots,
)


def test_screenshot_manifest_covers_required_views_and_alt_text() -> None:
    views = {item["view"] for item in SCREENSHOTS}
    assert views == {"routing_eq", "processing", "voice_setup"}
    assert len({item["filename"] for item in SCREENSHOTS}) == len(SCREENSHOTS)
    assert all(item["filename"].endswith(".png") for item in SCREENSHOTS)
    assert all(len(item["alt"].split()) >= 8 for item in SCREENSHOTS)
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(
        encoding="utf-8"
    )
    for item in SCREENSHOTS:
        expected = f"![{item['alt']}](docs/images/{item['filename']})"
        assert expected in readme


def test_sanitized_capture_devices_are_explicit_and_non_local() -> None:
    names = [
        device.name
        for device in (*SANITIZED_INPUT_DEVICES, *SANITIZED_OUTPUT_DEVICES)
    ]
    assert names
    assert len(names) == len(set(names))
    assert all("ancha" not in name.lower() for name in names)
    assert any("Virtual Route" in name for name in names)


def test_capture_generates_nonempty_portable_pngs_and_report(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "images"
    report_path = tmp_path / "report.json"
    report = capture_screenshots(output_dir, report_path)

    assert all(report["checks"].values())
    assert report["capture_contract"]["reads_user_config"] is False
    assert report["capture_contract"]["enumerates_real_devices"] is False
    assert len(report["screenshots"]) == 3
    for item in report["screenshots"]:
        path = output_dir / item["filename"]
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert item["bytes"] == path.stat().st_size
        assert item["width"] > 600
        assert item["height"] > 500

    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    serialized = json.dumps(loaded)
    assert str(tmp_path) not in serialized


def test_png_writer_failure_is_not_silent(monkeypatch, qapp, tmp_path: Path) -> None:
    class FailingWriter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def setCompression(self, _value: int) -> None:
            pass

        def setOptimizedWrite(self, _value: bool) -> None:
            pass

        def write(self, _image) -> bool:
            return False

        def errorString(self) -> str:
            return "synthetic failure"

    monkeypatch.setattr(
        "tools.capture_repository_screenshots.QImageWriter",
        FailingWriter,
    )
    widget = QWidget()
    widget.resize(100, 100)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        _write_optimized_png(widget, tmp_path / "bad.png")
    widget.deleteLater()
    qapp.processEvents()
