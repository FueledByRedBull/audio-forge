"""Checks for trusted native runtime DLL registration."""

from __future__ import annotations

import pytest

import mic_eq


def test_source_ort_directory_is_registered_before_native_load(tmp_path, monkeypatch):
    ort_root = tmp_path / "target" / "onnxruntime-cpu" / "lib"
    ort_root.mkdir(parents=True)
    (ort_root / "onnxruntime.dll").write_bytes(b"ort")
    (ort_root / "onnxruntime_providers_shared.dll").write_bytes(b"providers")
    registered: list[str] = []

    class Handle:
        pass

    monkeypatch.setattr(mic_eq, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        mic_eq.os,
        "add_dll_directory",
        lambda path: registered.append(path) or Handle(),
        raising=False,
    )
    monkeypatch.setattr(mic_eq, "_ORT_DLL_DIRECTORY_HANDLES", [])

    mic_eq._configure_ort_dll_directory()

    assert registered == [str(ort_root.resolve())]
    assert len(mic_eq._ORT_DLL_DIRECTORY_HANDLES) == 1


def test_frozen_runtime_ignores_source_tree_decoy(tmp_path, monkeypatch):
    bundle_package = tmp_path / "bundle" / "_internal" / "mic_eq"
    bundle_package.mkdir(parents=True)
    bundle_root = bundle_package.parent
    (bundle_root / "onnxruntime.dll").write_bytes(b"bundle")
    (bundle_root / "onnxruntime_providers_shared.dll").write_bytes(b"bundle")
    source_root = tmp_path / "source" / "target" / "onnxruntime-cpu" / "lib"
    source_root.mkdir(parents=True)
    (source_root / "onnxruntime.dll").write_bytes(b"source")
    (source_root / "onnxruntime_providers_shared.dll").write_bytes(b"source")
    registered: list[str] = []

    monkeypatch.setattr(mic_eq, "__file__", str(bundle_package / "__init__.py"))
    monkeypatch.setattr(mic_eq, "_PROJECT_ROOT", tmp_path / "source")
    monkeypatch.setattr(mic_eq.sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        mic_eq.os,
        "add_dll_directory",
        lambda path: registered.append(path) or object(),
        raising=False,
    )
    monkeypatch.setattr(mic_eq, "_ORT_DLL_DIRECTORY_HANDLES", [])

    mic_eq._configure_ort_dll_directory()

    assert registered == [str(bundle_root.resolve())]


def test_linux_exports_only_the_trusted_ort_library_path(tmp_path, monkeypatch):
    ort_root = tmp_path / "target" / "onnxruntime-cpu" / "lib"
    ort_root.mkdir(parents=True)
    (ort_root / mic_eq._LINUX_ORT_LIBRARY).write_bytes(b"ort")
    monkeypatch.setattr(mic_eq, "_PROJECT_ROOT", tmp_path)
    monkeypatch.delattr(mic_eq.os, "add_dll_directory", raising=False)
    monkeypatch.setattr(mic_eq.sys, "platform", "linux")
    monkeypatch.setenv("ORT_DYLIB_PATH", "/tmp/untrusted/libonnxruntime.so")

    mic_eq._configure_ort_dll_directory()

    assert mic_eq.os.environ["ORT_DYLIB_PATH"] == str(
        ort_root.resolve() / mic_eq._LINUX_ORT_LIBRARY
    )


def test_linux_missing_ort_library_fails_before_native_import(tmp_path, monkeypatch):
    monkeypatch.setattr(mic_eq, "_PROJECT_ROOT", tmp_path)
    monkeypatch.delattr(mic_eq.os, "add_dll_directory", raising=False)
    monkeypatch.setattr(mic_eq.sys, "platform", "linux")

    with pytest.raises(ImportError, match="CPU ONNX Runtime library is missing"):
        mic_eq._configure_ort_dll_directory()


def test_missing_ort_directory_fails_before_native_import(tmp_path, monkeypatch):
    registered: list[str] = []
    monkeypatch.setattr(mic_eq, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        mic_eq.os,
        "add_dll_directory",
        lambda path: registered.append(path) or object(),
        raising=False,
    )

    with pytest.raises(ImportError, match="CPU ONNX Runtime DLLs are missing"):
        mic_eq._configure_ort_dll_directory()
    assert registered == []
