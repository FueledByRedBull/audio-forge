from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock
from typing import cast
from uuid import uuid4

from PyQt6.QtWidgets import QMainWindow

from mic_eq.ui import app_bootstrap, desktop_integration


def _server_name() -> str:
    return f"AudioForge.test.{uuid4().hex}"


def test_tray_tooltip_reflects_processing_mute_health_and_route():
    tooltip = desktop_integration.build_tray_tooltip(
        processing=True,
        muted=True,
        health="Healthy",
        route="Mic -> Cable",
    )

    assert tooltip == "AudioForge: Running / Muted | Health: Healthy | Mic -> Cable"


def test_activate_window_restores_and_focuses_in_order():
    calls: list[str] = []
    window = SimpleNamespace(
        showNormal=lambda: calls.append("showNormal"),
        show=lambda: calls.append("show"),
        raise_=lambda: calls.append("raise"),
        activateWindow=lambda: calls.append("activate"),
    )

    desktop_integration.activate_window(cast(QMainWindow, window))

    assert calls == ["showNormal", "show", "raise", "activate"]


def test_second_instance_forwards_activation_without_removing_primary_server(
    qapp, tmp_path
):
    server_name = _server_name()
    lock_path = tmp_path / "audioforge.lock"
    primary = desktop_integration.SingleInstanceCoordinator(
        server_name=server_name,
        lock_path=lock_path,
    )
    secondary = desktop_integration.SingleInstanceCoordinator(
        server_name=server_name,
        lock_path=lock_path,
    )
    seen: list[bool] = []
    primary.set_activation_callback(lambda: seen.append(True))

    try:
        assert primary.acquire() == primary.ACQUIRED
        assert secondary.acquire() == secondary.FORWARDED
        for _ in range(10):
            qapp.processEvents()

        assert seen == [True]
        assert secondary.acquire() == secondary.FORWARDED
    finally:
        secondary.close()
        primary.close()


def test_second_instance_fails_closed_when_owner_has_no_activation_endpoint(
    tmp_path,
):
    from PyQt6.QtCore import QLockFile

    lock_path = tmp_path / "audioforge.lock"
    owner_lock = QLockFile(str(lock_path))
    owner_lock.setStaleLockTime(0)
    assert owner_lock.tryLock(0)
    secondary = desktop_integration.SingleInstanceCoordinator(
        server_name=_server_name(),
        lock_path=lock_path,
    )

    try:
        assert secondary.acquire() == secondary.FAILED
        assert secondary.last_error
    finally:
        secondary.close()
        owner_lock.unlock()


def test_smoke_test_skips_single_instance_coordinator(monkeypatch):
    class UnexpectedCoordinator:
        def __init__(self):
            raise AssertionError("smoke tests must not acquire the real instance lock")

    monkeypatch.setattr(app_bootstrap, "SingleInstanceCoordinator", UnexpectedCoordinator)
    monkeypatch.setattr(
        app_bootstrap,
        "_run_qt_app",
        lambda _window_cls, *, smoke_test: 0 if smoke_test else 1,
    )

    assert app_bootstrap.run_qt_app(cast(type[QMainWindow], object), smoke_test=True) == 0


def test_failed_instance_lock_shows_error_before_audio_setup(qapp, monkeypatch):
    instance = SimpleNamespace(
        acquire=lambda: "failed", last_error="owner unavailable",
    )
    factory = Mock(return_value=instance, ACQUIRED="acquired", FORWARDED="forwarded")
    monkeypatch.setattr(app_bootstrap, "SingleInstanceCoordinator", factory)
    monkeypatch.setattr(app_bootstrap, "QApplication", lambda _args: qapp)
    monkeypatch.setattr(app_bootstrap, "configure_app_logging", lambda: None)
    monkeypatch.setattr(app_bootstrap, "configure_windows_app_id", lambda: None)
    audio_setup = Mock()
    monkeypatch.setattr(app_bootstrap, "configure_deepfilter_env", audio_setup)
    message = Mock()
    monkeypatch.setattr(app_bootstrap.QMessageBox, "critical", message)

    assert app_bootstrap._run_qt_app(QMainWindow, smoke_test=False) == 1
    audio_setup.assert_not_called()
    assert "owner unavailable" in message.call_args.args[2]
