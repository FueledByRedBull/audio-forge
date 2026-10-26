"""The input-format probe reports what the native enumeration returns."""

from __future__ import annotations

import json
from types import SimpleNamespace

import probe_input_formats as probe_tool


def _device(name, *, default=False, default_config=None, supported=(), error=None):
    return SimpleNamespace(
        name=name,
        endpoint_id=f"{{{name}}}",
        is_default=default,
        default_config=default_config,
        supported=list(supported),
        error=error,
    )


FAKE_DEVICES = [
    _device(
        "Microphone (USB)",
        default=True,
        default_config=("f32", 2, 48000),
        supported=[("f32", 2, 48000, 48000), ("i16", 1, 44100, 96000)],
    ),
    _device(
        "Line In (Interface)",
        default_config=("i24", 2, 48000),
        supported=[("i24", 2, 48000, 48000)],
    ),
    _device("Virtual Cable", error="device is busy"),
]


def _fake_list():
    return FAKE_DEVICES


def test_probe_flattens_native_tuples_into_records():
    first, second, third = probe_tool.probe(_fake_list)
    assert first["is_default"] is True
    assert first["default_config"] == {"sample_format": "f32", "channels": 2, "sample_rate": 48000}
    assert first["supported"][1] == {
        "sample_format": "i16",
        "channels": 1,
        "min_sample_rate": 44100,
        "max_sample_rate": 96000,
    }
    assert second["is_default"] is False
    assert third["default_config"] is None and third["error"] == "device is busy"


def test_twenty_four_bit_devices_come_from_default_or_supported_configs():
    records = probe_tool.probe(_fake_list)
    assert probe_tool.twenty_four_bit_devices(records) == ["Line In (Interface)"]
    only_supported = probe_tool.probe(
        lambda: [_device("A", default_config=("f32", 2, 48000), supported=[("u24", 2, 48000, 48000)])]
    )
    assert probe_tool.twenty_four_bit_devices(only_supported) == ["A"]
    assert probe_tool.twenty_four_bit_devices(probe_tool.probe(lambda: [])) == []


def test_text_report_lists_every_device_and_the_24_bit_summary(capsys):
    assert probe_tool.main([], list_formats=_fake_list) == 0
    out = capsys.readouterr().out
    assert "3 input device(s)" in out
    assert "* Microphone (USB)" in out and "  Line In (Interface)" in out
    assert "default: f32, 2 ch, 48000 Hz" in out
    assert "supported: i16, 1 ch, 44100-96000 Hz" in out
    assert "supported: f32, 2 ch, 48000 Hz" in out
    assert "default: not reported" in out and "could not be read (device is busy)" in out
    assert out.rstrip().endswith("24-bit formats (i24/u24): Line In (Interface)")


def test_report_says_when_no_device_reports_24_bit(capsys):
    quiet = [_device("Mic", default_config=("f32", 1, 48000), supported=[("f32", 1, 48000, 48000)])]
    assert probe_tool.main([], list_formats=lambda: quiet) == 0
    assert capsys.readouterr().out.rstrip().endswith("24-bit formats (i24/u24): none reported")


def test_json_report_carries_records_and_the_24_bit_names(capsys):
    assert probe_tool.main(["--json"], list_formats=_fake_list) == 0
    report = json.loads(capsys.readouterr().out)
    assert [device["name"] for device in report["devices"]] == [
        "Microphone (USB)",
        "Line In (Interface)",
        "Virtual Cable",
    ]
    assert report["twenty_four_bit"] == ["Line In (Interface)"]


def test_an_enumeration_failure_is_reported_not_raised(capsys):
    def failing():
        raise RuntimeError("no audio host")

    assert probe_tool.main([], list_formats=failing) == 2
    assert "no audio host" in capsys.readouterr().err
