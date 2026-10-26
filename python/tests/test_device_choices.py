"""The device pickers: what they say, and what a change in one of them saves."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from mic_eq.config import DeviceIdentity
from mic_eq.ui import main_window


def _device(name: str, endpoint_id: str, *, default: bool = False, ordinal: int = 0):
    return SimpleNamespace(
        name=name, is_default=default, endpoint_id=endpoint_id, host_api="WASAPI",
        sample_rate=48_000, channels=2, name_ordinal=ordinal,
    )


def _plug(monkeypatch, inputs, outputs) -> None:
    monkeypatch.setattr(main_window, "list_input_devices", lambda: inputs)
    monkeypatch.setattr(main_window, "list_output_devices", lambda: outputs)


def _labels(combo) -> list[str]:
    return [combo.itemText(i) for i in range(combo.count())]


def test_the_markers_that_tell_devices_apart_come_before_the_name(isolated_main_window, monkeypatch):
    window: Any = isolated_main_window
    _plug(
        monkeypatch,
        [
            _device("Microphone (USB Audio Device)", "usb-1", ordinal=0),
            _device("Microphone (USB Audio Device)", "usb-2", default=True, ordinal=1),
            _device("Line In", "line"),
        ],
        [_device("Speakers", "spk", default=True)],
    )

    window._refresh_devices()

    # The pickers cut a long label off at its end, so nothing that matters may sit there.
    assert _labels(window.input_choice) == [
        "[#1] Microphone (USB Audio Device)",
        "[#2] (Default) Microphone (USB Audio Device)",
        "Line In",
    ]
    assert _labels(window.output_choice) == ["(Default) Speakers"]
    # Only the label changed; the saved identity of each device did not.
    second = window.input_choice.itemData(1)
    assert (second.name, second.name_ordinal, second.endpoint_id) == (
        "Microphone (USB Audio Device)", 1, "usb-2"
    )


def test_choosing_an_output_keeps_the_saved_microphone_that_is_unplugged(
    isolated_main_window, monkeypatch
):
    window: Any = isolated_main_window
    saved = DeviceIdentity(name="USB Mic", endpoint_id="usb-mic", direction="input", host_api="WASAPI")
    window.config.last_input_device_identity = saved
    window.config.last_input_device = "USB Mic"
    _plug(
        monkeypatch,
        [_device("Built-in Mic", "builtin", default=True)],
        [_device("Cable", "cable", default=True), _device("Speakers", "spk")],
    )
    window._refresh_devices()
    assert window.input_choice.currentData().endpoint_id == "builtin"  # only until the USB mic returns

    window.output_choice.setCurrentIndex(1)

    assert window.output_choice.currentData().endpoint_id == "spk"
    assert window.config.last_output_device_identity.endpoint_id == "spk"
    assert window.config.last_input_device_identity == saved
    assert window.config.last_input_device == "USB Mic"


def test_choosing_a_microphone_keeps_the_saved_destination_that_is_unplugged(
    isolated_main_window, monkeypatch
):
    window: Any = isolated_main_window
    saved = DeviceIdentity(name="Virtual Cable", endpoint_id="cable", direction="output", host_api="WASAPI")
    window.config.last_output_device_identity = saved
    window.config.last_output_device = "Virtual Cable"
    _plug(
        monkeypatch,
        [_device("Mic A", "a", default=True), _device("Mic B", "b")],
        [_device("Speakers", "spk", default=True)],
    )
    window._refresh_devices()
    assert window.output_choice.currentIndex() == -1  # nothing chosen until it returns

    window.input_choice.setCurrentIndex(1)

    assert window.config.last_input_device_identity.endpoint_id == "b"
    assert window.config.last_output_device_identity == saved
    assert window.config.last_output_device == "Virtual Cable"
