"""The mic_eq_core.pyi stub must list what the native module exports."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import mic_eq

pytestmark = pytest.mark.skipif(not mic_eq.CORE_AVAILABLE, reason="native core not built")


def _stub_names(class_name: str | None) -> set[str]:
    tree = ast.parse((Path(mic_eq.__file__).parent / "mic_eq_core.pyi").read_text("utf-8"))
    body = tree.body
    if class_name is not None:
        body = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
        ).body
    return {node.name for node in body if isinstance(node, ast.FunctionDef)}


def test_audio_processor_stub_lists_every_native_method():
    from mic_eq import mic_eq_core

    native = {
        name
        for name in dir(mic_eq_core.AudioProcessor)
        if not name.startswith("_") and callable(getattr(mic_eq_core.AudioProcessor, name))
    }

    assert native - _stub_names("AudioProcessor") == set()
