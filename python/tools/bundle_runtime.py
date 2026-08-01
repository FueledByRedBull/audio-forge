"""Load AudioForge's native runtime directly from an extracted release bundle."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType


_DLL_DIRECTORY_HANDLES: list[object] = []


def resolve_bundle_layout(bundle_root: Path) -> dict[str, Path]:
    root = bundle_root.resolve(strict=True)
    internal = root / "_internal"
    executable = root / "AudioForge.exe"
    library = internal / "df.dll"
    model_root = internal / "models"
    native_candidates = sorted(
        (internal / "mic_eq").glob("mic_eq_core*.pyd"),
        key=lambda path: path.name.casefold(),
    )
    if not executable.is_file():
        raise FileNotFoundError(f"bundle executable is missing: {executable}")
    if not internal.is_dir():
        raise FileNotFoundError(f"bundle internal directory is missing: {internal}")
    if not library.is_file():
        raise FileNotFoundError(f"bundle DeepFilter library is missing: {library}")
    if not model_root.is_dir():
        raise FileNotFoundError(f"bundle model directory is missing: {model_root}")
    if len(native_candidates) != 1:
        raise FileNotFoundError(
            "bundle must contain exactly one _internal/mic_eq/mic_eq_core*.pyd"
        )
    return {
        "root": root,
        "internal": internal,
        "executable": executable,
        "library": library,
        "model_root": model_root,
        "native_extension": native_candidates[0],
    }


def load_bundled_core(bundle_root: Path) -> ModuleType:
    layout = resolve_bundle_layout(bundle_root)
    if not hasattr(os, "add_dll_directory"):
        raise RuntimeError("loading a bundled runtime is supported only on Windows")
    _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(layout["internal"])))
    spec = importlib.util.spec_from_file_location(
        "mic_eq_core",
        layout["native_extension"],
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"cannot create import spec for {layout['native_extension']}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    configure = getattr(module, "configure_deepfilter_runtime_paths", None)
    if not callable(configure):
        raise ImportError("bundled native runtime lacks DeepFilter path configuration")
    configure(str(layout["library"]), str(layout["model_root"]))
    return module
