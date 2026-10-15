"""Picking a noise suppression backend reports its outcome in the scene."""

from __future__ import annotations

import re

import pytest
from PySide6.QtQml import QQmlExpression
from PySide6.QtQuickWidgets import QQuickWidget

from mic_eq.ui import main_window as main_window_module

BACKENDS = [("rnnoise", "RNNoise"), ("deepfilter-ll", "DeepFilterNet LL"), ("deepfilter", "DeepFilterNet")]


def _names_the_refused_backend(text: str) -> bool:
    """The model, why it was refused, and that the previous one is kept."""
    return bool(
        "deepfilter-ll" in text
        and "unavailable" in text
        and re.search(r"previous.*(retain|kept|keep)", text)
    )


@pytest.fixture
def engine(monkeypatch):
    """The native engine with three listed backends; a switch to a refused one fails."""

    native = main_window_module.AudioProcessor

    class Engine:
        def __init__(self) -> None:
            self._native = native()
            self.refused: set[str] = set()
            self.switches: list[str] = []

        def __getattr__(self, name):
            return getattr(self._native, name)

        def list_noise_models(self):
            return BACKENDS

        def set_noise_model(self, model: str) -> bool:
            self.switches.append(model)
            return model not in self.refused

    monkeypatch.setattr(main_window_module, "AudioProcessor", Engine)
    return Engine


@pytest.fixture
def window(engine, isolated_main_window):
    isolated_main_window.processor.switches.clear()
    return isolated_main_window


@pytest.fixture
def warnings(monkeypatch) -> list[tuple[str, str]]:
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "mic_eq.ui.main_window.QMessageBox.warning",
        lambda _parent, title, text, *_args: shown.append((title, text)),
    )
    return shown


def _pick(window, index: int) -> None:
    # The call the Backend picker makes from the scene.
    scene = window.centralWidget()
    assert isinstance(scene, QQuickWidget)
    call = QQmlExpression(scene.rootContext(), scene, f"ui.model.stages[0].rows[0].proxy.setIndex({index})")
    call.evaluate()
    assert not call.hasError()


def _shown(window) -> tuple[int, str]:
    proxy = window.quick_bridge.model["stages"][0]["rows"][0]["proxy"]
    return proxy.index, proxy.text


def test_refused_backend_warns_and_the_picker_keeps_the_previous_one(window, warnings) -> None:
    window.processor.refused.add("deepfilter-ll")
    edits: list[str] = []
    window.noise_suppression_state.configurationEdited.connect(edits.append)
    history = (window._configuration_history.cursor, window._configuration_history.size)
    _pick(window, 1)
    assert len(warnings) == 1
    title, body = warnings[0]
    assert "switch" in title.lower() and "fail" in title.lower() and _names_the_refused_backend(body)
    assert _shown(window) == (0, "RNNoise")
    assert window.noise_suppression_state.get_settings()["model"] == "rnnoise"
    assert _names_the_refused_backend(window.status_message.text)
    assert window.processor.switches == ["deepfilter-ll", "rnnoise"]
    assert edits == []
    assert not window._history_timer.isActive()
    assert (window._configuration_history.cursor, window._configuration_history.size) == history


def test_switched_backend_is_confirmed_in_the_status_line(window, warnings) -> None:
    edits: list[str] = []
    window.noise_suppression_state.configurationEdited.connect(edits.append)
    _pick(window, 1)
    assert "DeepFilterNet LL" in window.status_message.text
    assert _shown(window) == (1, "DeepFilterNet LL")
    assert window.noise_suppression_state.get_settings()["model"] == "deepfilter-ll"
    assert window.processor.switches == ["deepfilter-ll"]
    assert edits == ["Noise suppression edit"]
    assert not warnings


def test_picking_the_active_backend_again_changes_and_reports_nothing(window, warnings) -> None:
    _pick(window, 1)
    window.status_message.showMessage("Probe")
    window.processor.switches.clear()
    # The Quick picker calls the handler even for the backend it already shows.
    _pick(window, 1)
    assert window.status_message.text == "Probe"
    assert _shown(window) == (1, "DeepFilterNet LL")
    assert window.processor.switches == []
    assert not warnings


@pytest.mark.parametrize("index", [-1, len(BACKENDS)])
def test_an_index_outside_the_backend_list_changes_and_reports_nothing(window, warnings, index) -> None:
    window._on_model_changed(index)
    assert _shown(window) == (0, "RNNoise")
    assert window.processor.switches == []
    assert not warnings
