"""AudioForge low-latency microphone processing."""

from __future__ import annotations

import os
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ORT_DLL_DIRECTORY_HANDLES: list[object] = []


def _ort_dll_root() -> Path | None:
    """Return the one trusted directory that may contain the CPU ORT DLLs."""
    package_root = Path(__file__).resolve().parent
    if getattr(sys, "frozen", False):
        # PyInstaller places this package and the spec's ``.`` binaries under
        # the same content directory.  Deriving it from __file__ avoids CWD,
        # PATH, and executable-adjacent search fallbacks.
        root = package_root.parent
    else:
        # Editable/source builds hydrate this exact project-owned directory.
        root = _PROJECT_ROOT / "target" / "onnxruntime-cpu" / "lib"
    resolved = root.resolve()
    if not resolved.is_dir():
        return None
    if not (resolved / "onnxruntime.dll").is_file() or not (
        resolved / "onnxruntime_providers_shared.dll"
    ).is_file():
        return None
    return resolved


def _configure_ort_dll_directory() -> None:
    """Register the packaged CPU ORT directory before loading the Rust extension."""
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return
    root = _ort_dll_root()
    if root is None:
        raise ImportError(
            "CPU ONNX Runtime DLLs are missing from the trusted project/bundle directory"
        )
    try:
        handle = add_dll_directory(str(root))
    except OSError as error:
        raise ImportError(f"cannot register CPU ORT DLL directory: {root}") from error
    _ORT_DLL_DIRECTORY_HANDLES.append(handle)


__version__ = "1.12.2"

try:
    _configure_ort_dll_directory()
    from .mic_eq_core import (
        AudioProcessor,
        DeviceInfo,
        analyze_vad_probabilities,
        configure_deepfilter_runtime_paths,
        eq_magnitude_response,
        eq_magnitude_response_v2,
        list_input_devices,
        list_output_devices,
    )

    CORE_AVAILABLE = True
except ImportError as error:
    _CORE_IMPORT_ERROR = error
    CORE_AVAILABLE = False

    def _missing_core(*args, **kwargs):
        raise ImportError(
            "mic_eq_core is unavailable. Build it with: maturin develop --release. "
            f"Native import error: {_CORE_IMPORT_ERROR}"
        ) from _CORE_IMPORT_ERROR

    class _MissingCoreType:
        def __init__(self, *args, **kwargs):
            _missing_core()

    AudioProcessor = type("AudioProcessor", (_MissingCoreType,), {"__module__": __name__})
    DeviceInfo = type("DeviceInfo", (_MissingCoreType,), {"__module__": __name__})
    list_input_devices = _missing_core
    list_output_devices = _missing_core
    configure_deepfilter_runtime_paths = _missing_core
    analyze_vad_probabilities = _missing_core
    eq_magnitude_response = _missing_core
    eq_magnitude_response_v2 = _missing_core

from .config import (  # noqa: E402
    BUILTIN_PRESETS,
    Preset,
    list_presets,
    load_preset,
    save_preset,
)

__all__ = [
    "AudioProcessor",
    "DeviceInfo",
    "list_input_devices",
    "list_output_devices",
    "eq_magnitude_response",
    "eq_magnitude_response_v2",
    "analyze_vad_probabilities",
    "configure_deepfilter_runtime_paths",
    "CORE_AVAILABLE",
    "Preset",
    "save_preset",
    "load_preset",
    "list_presets",
    "BUILTIN_PRESETS",
]
