"""Tests for package/native-module identity compatibility."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


def test_package_init_keeps_top_level_native_extension_fallback(monkeypatch):
    package_dir = Path(__file__).resolve().parents[1] / "mic_eq"
    init_path = package_dir / "__init__.py"
    package_name = "mic_eq_fallback_test"

    fake_core: Any = types.ModuleType("mic_eq_core")

    class FakeAudioProcessor:
        pass

    class FakeDeviceInfo:
        pass

    fake_core.AudioProcessor = FakeAudioProcessor
    fake_core.DeviceInfo = FakeDeviceInfo
    fake_core.list_input_devices = lambda: ["input"]
    fake_core.list_output_devices = lambda: ["output"]

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "mic_eq_core", fake_core)
    monkeypatch.setitem(sys.modules, f"{package_name}.mic_eq_core", None)
    monkeypatch.setitem(sys.modules, package_name, module)

    spec.loader.exec_module(module)

    assert module.CORE_AVAILABLE is True
    assert module.AudioProcessor is FakeAudioProcessor
    assert module.DeviceInfo is FakeDeviceInfo
    assert module.list_input_devices() == ["input"]
    assert module.list_output_devices() == ["output"]
