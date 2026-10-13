"""Picking a noise suppression backend reports the same outcome in both views."""

from __future__ import annotations

import pytest
from PySide6.QtQml import QQmlExpression
from PySide6.QtQuickWidgets import QQuickWidget

from mic_eq.ui import main_window as main_window_module

BACKENDS = [("rnnoise", "RNNoise"), ("deepfilter-ll", "DeepFilterNet LL"), ("deepfilter", "DeepFilterNet")]
REFUSED = "Noise model 'deepfilter-ll' is unavailable; previous model retained"


@pytest.fixture(params=["quick", "classic"])
def view(request, monkeypatch) -> str:
    monkeypatch.setenv("AUDIOFORGE_QML", "1" if request.param == "quick" else "0")
    return request.param


@pytest.fixture
def engine(view, monkeypatch):
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
def window(view, engine, isolated_main_window):
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


def _pick(window, view: str, index: int) -> None:
    if view == "classic":
        window.model_combo.setCurrentIndex(index)
        return
    # The call the Backend picker makes from the scene.
    scene = window.centralWidget()
    assert isinstance(scene, QQuickWidget)
    call = QQmlExpression(scene.rootContext(), scene, f"ui.model.stages[0].rows[0].proxy.setIndex({index})")
    call.evaluate()
    assert not call.hasError()


def _shown(window, view: str) -> tuple[int, str]:
    if view == "classic":
        return window.model_combo.currentIndex(), window.model_combo.currentText()
    assert window._classic_root is None
    proxy = window.quick_bridge.model["stages"][0]["rows"][0]["proxy"]
    return proxy.index, proxy.text


def test_refused_backend_warns_and_the_picker_keeps_the_previous_one(window, view, warnings) -> None:
    window.processor.refused.add("deepfilter-ll")
    edits: list[str] = []
    window.noise_suppression_state.configurationEdited.connect(edits.append)
    history = (window._configuration_history.cursor, window._configuration_history.size)
    _pick(window, view, 1)
    assert warnings == [("Model Switch Failed", REFUSED)]
    assert _shown(window, view) == (0, "RNNoise")
    assert window.noise_suppression_state.get_settings()["model"] == "rnnoise"
    assert window.status_message.text == REFUSED
    assert window.processor.switches == ["deepfilter-ll", "rnnoise"]
    assert edits == []
    assert not window._history_timer.isActive()
    assert (window._configuration_history.cursor, window._configuration_history.size) == history


def test_switched_backend_is_confirmed_in_the_status_line(window, view, warnings) -> None:
    edits: list[str] = []
    window.noise_suppression_state.configurationEdited.connect(edits.append)
    _pick(window, view, 1)
    assert window.status_message.text == "Switched to DeepFilterNet LL"
    assert _shown(window, view) == (1, "DeepFilterNet LL")
    assert window.noise_suppression_state.get_settings()["model"] == "deepfilter-ll"
    assert window.processor.switches == ["deepfilter-ll"]
    assert edits == ["Noise suppression edit"]
    assert not warnings


def test_picking_the_active_backend_again_changes_and_reports_nothing(window, view, warnings) -> None:
    _pick(window, view, 1)
    window.status_message.showMessage("Probe")
    window.processor.switches.clear()
    if view == "classic":
        # A combo box emits nothing for the index it already shows; the Quick picker calls the handler.
        window._on_model_changed(1)
    else:
        _pick(window, view, 1)
    assert window.status_message.text == "Probe"
    assert _shown(window, view) == (1, "DeepFilterNet LL")
    assert window.processor.switches == []
    assert not warnings


@pytest.mark.parametrize("index", [-1, len(BACKENDS)])
def test_an_index_outside_the_backend_list_changes_and_reports_nothing(window, view, warnings, index) -> None:
    window._on_model_changed(index)
    assert _shown(window, view) == (0, "RNNoise")
    assert window.processor.switches == []
    assert not warnings
