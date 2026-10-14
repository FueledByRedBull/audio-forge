"""Compares the previous spectrum implementation with the current one."""

import importlib.util
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

import test_spectrum_identity as identity
from mic_eq.analysis import spectrum as after

_PATH = Path(__file__).with_name("_spectrum_before.py")
_spec = importlib.util.spec_from_file_location("_spectrum_before", _PATH)
before = importlib.util.module_from_spec(_spec)
sys.modules["_spectrum_before"] = before
_spec.loader.exec_module(before)


def _outcome(module, monkeypatch, run):
    monkeypatch.setattr(identity, "spectrum", module)
    try:
        return "ok", run()
    except Exception as exc:  # compared by type and message below
        return "raised", exc


def _field_diff(old, new):
    """Return a description of the difference, or None when exactly equal."""
    if isinstance(old, np.ndarray) or isinstance(new, np.ndarray):
        if not (isinstance(old, np.ndarray) and isinstance(new, np.ndarray)):
            return f"type {type(old).__name__} vs {type(new).__name__}"
        if old.shape != new.shape or old.dtype != new.dtype:
            return f"shape/dtype {old.shape}/{old.dtype} vs {new.shape}/{new.dtype}"
        if old.dtype.kind in "fc":
            same = (old == new) | (np.isnan(old) & np.isnan(new))
        else:
            same = old == new
        if np.array_equal(old, new, equal_nan=old.dtype.kind in "fc"):
            return None
        with np.errstate(all="ignore"):
            delta = np.abs(old[~same].astype(complex) - new[~same].astype(complex))
        return f"max abs diff {np.nanmax(delta):.17g}, {int((~same).sum())} of {old.size} elements differ"
    if type(old) is not type(new):
        return f"type {type(old).__name__} vs {type(new).__name__}"
    if isinstance(old, float) and old != old and new != new:
        return None
    if old == new:
        return None
    delta = abs(old - new) if isinstance(old, (int, float)) else "n/a"
    return f"{old!r} != {new!r}, abs diff {delta}"


def _assert_same(name, old, new):
    assert old[0] == new[0], f"{name}: old {old[0]}, new {new[0]} ({old[1]!r} / {new[1]!r})"
    if old[0] == "raised":
        assert (type(old[1]), str(old[1])) == (type(new[1]), str(new[1])), (
            f"{name}: old raised {old[1]!r}, new raised {new[1]!r}"
        )
        return
    assert old[1].keys() == new[1].keys(), f"{name}: fields {sorted(old[1])} vs {sorted(new[1])}"
    problems = []
    for field in sorted(old[1]):
        diff = _field_diff(old[1][field], new[1][field])
        if diff:
            problems.append(f"{name}: field {field}: {diff}")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("name", sorted(identity.RUNNERS))
def test_previous_and_current_agree(monkeypatch, name):
    run = identity.RUNNERS[name]
    old = _outcome(before, monkeypatch, run)
    new = _outcome(after, monkeypatch, run)
    _assert_same(name, old, new)


@pytest.mark.parametrize("call", ["analyze", "compute"])
def test_previous_and_current_reject_short_audio_alike(call):
    def run(module):
        func = module.analyze_voice_spectrum if call == "analyze" else module.compute_voice_spectrum
        with pytest.raises(ValueError) as caught:
            func(np.zeros(4095), identity.FS)
        return str(caught.value)

    assert run(before) == run(after)


def test_print_numpy_runtime():
    print("platform:", platform.platform(), platform.machine(), platform.processor())
    print("python:", sys.version)
    print("numpy:", np.__version__)
    np.show_runtime()
    try:
        from numpy._core._multiarray_umath import __cpu_baseline__, __cpu_dispatch__, __cpu_features__
    except ImportError as exc:
        print("cpu feature tables unavailable:", exc)
        return
    print("cpu baseline:", list(__cpu_baseline__))
    print("cpu dispatch:", list(__cpu_dispatch__))
    print("cpu features enabled:", sorted(k for k, v in __cpu_features__.items() if v))
    print("cpu features absent:", sorted(k for k, v in __cpu_features__.items() if not v))
