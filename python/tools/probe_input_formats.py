"""
List the sample formats and channel counts every input device offers.

Read-only: the audio host is asked what each input device supports and no
stream is opened. Close AudioForge first so that nothing else holds an
endpoint.

    python python/tools/probe_input_formats.py [--json] [--bundle-root DIR]

The last line says which devices default to or offer a 24-bit format
(i24/u24), the formats stream setup has to convert.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

TWENTY_FOUR_BIT_FORMATS = frozenset({"i24", "u24"})
_DEFAULT_FIELDS = ("sample_format", "channels", "sample_rate")
_RANGE_FIELDS = ("sample_format", "channels", "min_sample_rate", "max_sample_rate")


def _record(device: Any) -> dict[str, Any]:
    default = device.default_config
    return {
        "name": device.name,
        "endpoint_id": device.endpoint_id,
        "is_default": device.is_default,
        "default_config": None if default is None else dict(zip(_DEFAULT_FIELDS, default)),
        "supported": [dict(zip(_RANGE_FIELDS, entry)) for entry in device.supported],
        "error": device.error,
    }


def probe(list_formats: Callable[[], Sequence[Any]]) -> list[dict[str, Any]]:
    """One record per input device, from the native ``list_input_device_formats``."""
    return [_record(device) for device in list_formats()]


def twenty_four_bit_devices(devices: Sequence[dict[str, Any]]) -> list[str]:
    """Names of the devices whose default or supported configs include i24/u24."""
    names = []
    for device in devices:
        formats = {entry["sample_format"] for entry in device["supported"]}
        if device["default_config"] is not None:
            formats.add(device["default_config"]["sample_format"])
        if formats & TWENTY_FOUR_BIT_FORMATS:
            names.append(device["name"])
    return names


def _rate_text(entry: dict[str, Any]) -> str:
    low, high = entry["min_sample_rate"], entry["max_sample_rate"]
    return f"{low} Hz" if low == high else f"{low}-{high} Hz"


def render(devices: Sequence[dict[str, Any]]) -> str:
    lines = [f"{len(devices)} input device(s); the default is marked with *"]
    for device in devices:
        lines.append("")
        lines.append(f"{'*' if device['is_default'] else ' '} {device['name']}")
        lines.append(f"    endpoint: {device['endpoint_id']}")
        default = device["default_config"]
        if default is None:
            lines.append("    default: not reported")
        else:
            lines.append(
                f"    default: {default['sample_format']}, "
                f"{default['channels']} ch, {default['sample_rate']} Hz"
            )
        if device["error"]:
            lines.append(f"    supported: could not be read ({device['error']})")
        for entry in device["supported"]:
            lines.append(
                f"    supported: {entry['sample_format']}, "
                f"{entry['channels']} ch, {_rate_text(entry)}"
            )
    lines.append("")
    offering = twenty_four_bit_devices(devices)
    lines.append(
        "24-bit formats (i24/u24): " + (", ".join(offering) if offering else "none reported")
    )
    return "\n".join(lines)


def _native_lister(bundle_root: Path | None) -> Callable[[], Sequence[Any]]:
    if bundle_root is None:
        from mic_eq import mic_eq_core as core
    else:
        from bundle_runtime import load_bundled_core

        core = load_bundled_core(bundle_root)
    return core.list_input_device_formats


def main(
    argv: Sequence[str] | None = None,
    *,
    list_formats: Callable[[], Sequence[Any]] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="List the formats every input device offers.")
    parser.add_argument("--json", action="store_true", help="print the records as JSON")
    parser.add_argument(
        "--bundle-root",
        type=Path,
        help="probe with the native module of an extracted release bundle",
    )
    args = parser.parse_args(argv)

    try:
        lister = list_formats or _native_lister(args.bundle_root)
    except (AttributeError, ImportError, OSError, RuntimeError) as error:
        print(f"The native core cannot list input formats: {error}", file=sys.stderr)
        return 2
    try:
        devices = probe(lister)
    except RuntimeError as error:
        print(f"The audio host could not list the input devices: {error}", file=sys.stderr)
        return 2
    if args.json:
        report = {"devices": devices, "twenty_four_bit": twenty_four_bit_devices(devices)}
        print(json.dumps(report, indent=2))
    else:
        print(render(devices))
    return 0


if __name__ == "__main__":
    sys.exit(main())
