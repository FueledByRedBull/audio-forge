"""License access resolves the packaged/source notices without launching a viewer in tests."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from mic_eq.ui import main_window


@pytest.mark.parametrize("frozen", [False, True])
def test_license_action_opens_the_trusted_notice_directory(monkeypatch, tmp_path, frozen):
    root = tmp_path / "path with spaces"
    notices = root / "licenses"
    notices.mkdir(parents=True)
    (notices / "THIRD_PARTY_NOTICES.md").write_text("notices", encoding="utf-8")
    monkeypatch.setattr(main_window, "__file__", str(root / "python/mic_eq/ui/main_window.py"))
    if frozen:
        monkeypatch.setattr(main_window.sys, "_MEIPASS", str(root), raising=False)
    else:
        monkeypatch.delattr(main_window.sys, "_MEIPASS", raising=False)
    opened = []
    monkeypatch.setattr(main_window.QDesktopServices, "openUrl", lambda url: opened.append(Path(url.toLocalFile())) or True)
    main_window.MainWindow._show_licenses(cast(Any, SimpleNamespace()))
    assert opened == [notices]


@pytest.mark.parametrize("present", [False, True])
def test_license_action_reports_missing_notices_or_viewer_failure(monkeypatch, tmp_path, present):
    monkeypatch.setattr(main_window.sys, "_MEIPASS", str(tmp_path), raising=False)
    if present:
        (tmp_path / "licenses").mkdir()
        (tmp_path / "licenses/THIRD_PARTY_NOTICES.md").write_text("notices", encoding="utf-8")
    opened, warnings = [], []
    monkeypatch.setattr(main_window.QDesktopServices, "openUrl", lambda url: opened.append(url) or False)
    monkeypatch.setattr(main_window.QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    main_window.MainWindow._show_licenses(cast(Any, SimpleNamespace()))
    assert bool(opened) is present
    assert len(warnings) == 1
    assert str(tmp_path / "licenses") in warnings[0]
