"""Load AudioForge's native runtime directly from an extracted release bundle."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType


_DLL_DIRECTORY_HANDLES: list[object] = []
# The ORT build release-assets.json pins for Linux, loaded by absolute path.
LINUX_ORT_LIBRARY = "libonnxruntime.so.1.23.2"


def is_linux_bundle(root: Path) -> bool:
    return not (root / "AudioForge.exe").is_file() and (root / "AudioForge").is_file()


def resolve_bundle_layout(bundle_root: Path) -> dict[str, Path]:
    root = bundle_root.resolve(strict=True)
    internal = root / "_internal"
    linux = is_linux_bundle(root)
    executable = root / ("AudioForge" if linux else "AudioForge.exe")
    library = internal / ("libdf.so" if linux else "df.dll")
    model_root = internal / "models"
    vad_model = model_root / "silero_vad.onnx"
    extension = "*.so" if linux else "*.pyd"
    native_candidates = sorted(
        (internal / "mic_eq").glob(f"mic_eq_core{extension}"),
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
    if not vad_model.is_file():
        raise FileNotFoundError(f"bundle VAD model is missing: {vad_model}")
    if len(native_candidates) != 1:
        raise FileNotFoundError(
            f"bundle must contain exactly one _internal/mic_eq/mic_eq_core{extension}"
        )
    layout = {}
    if linux:
        layout["ort_library"] = internal / LINUX_ORT_LIBRARY
        if not layout["ort_library"].is_file():
            raise FileNotFoundError(f"bundle ONNX Runtime is missing: {layout['ort_library']}")
    return {
        **layout,
        "root": root,
        "internal": internal,
        "executable": executable,
        "library": library,
        "model_root": model_root,
        "vad_model": vad_model,
        "native_extension": native_candidates[0],
    }


def load_bundled_core(bundle_root: Path) -> ModuleType:
    layout = resolve_bundle_layout(bundle_root)
    os.environ["VAD_MODEL_PATH"] = str(layout["vad_model"])
    if "ort_library" in layout:
        # The core's ort loads this library on first use, so set it before import.
        os.environ["ORT_DYLIB_PATH"] = str(layout["ort_library"])
    elif hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(layout["internal"])))
    else:
        raise RuntimeError("a Windows bundle loads only on Windows")
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
