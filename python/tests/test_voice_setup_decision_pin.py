"""Voice Setup's analysis returns the same settings, decision and report on fixed bench takes.

Each case runs ``analyze_voice_setup`` on the first 6 s of a bench mix, as the app's offline flow does, and
compares what comes back (the chosen settings, the apply recommendation with its reasons, and the whole report)
with ``fixtures/golden/voice_setup_analysis.json``. A change that only makes the analysis faster (parallel
candidates, shared evidence) has to leave all of it alone. Timings (the ``runtime`` fields) are left out. Numbers are
stored to 6 significant digits and match within 1e-4 of their size, so another CPU's last-bit differences pass;
booleans, reasons and counts match exactly.

The Silero cases need the model (``VAD_MODEL_PATH`` or ``models/silero_vad.onnx``) and skip without it or with
another model; the energy case runs wherever the native extension does.

Regenerate only with ``AUDIOFORGE_UPDATE_GOLDEN=1 python -m pytest python/tests/test_voice_setup_decision_pin.py``,
in a pull request meant to change what Voice Setup decides, and say why.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeGuard

import numpy as np
import pytest

import bench_metrics as bm
from mic_eq import CORE_AVAILABLE
from mic_eq.analysis.voice_setup import analyze_voice_setup
from mic_eq.config import EQSettings, Preset
from test_bench_metrics import built_mix

pytestmark = pytest.mark.skipif(not CORE_AVAILABLE, reason="native extension is not built")

SR = bm.SAMPLE_RATE
CROP_SECONDS = 6.0
NOISE_SECONDS = 1.6
PIN_PATH = Path(__file__).resolve().parent / "fixtures" / "golden" / "voice_setup_analysis.json"
UPDATE = os.environ.get("AUDIOFORGE_UPDATE_GOLDEN") == "1"
RELATIVE_TOLERANCE = 1e-4
ABSOLUTE_TOLERANCE = 1e-6
# (case name, bench mix, Silero VAD). The energy case runs without a model.
CASES = (
    ("fan_t01_silero", "fan-t01", True),
    ("cafe_t02_silero", "cafe-t02", True),
    ("fan_t01_energy", "fan-t01", False),
)


def _normalized(value: Any) -> Any:
    """The report as plain JSON values, without timings and with floats at 6 significant digits."""
    if isinstance(value, dict):
        return {str(key): _normalized(child) for key, child in value.items() if "runtime" not in str(key)}
    if isinstance(value, (list, tuple)):
        return [_normalized(child) for child in value]
    if isinstance(value, np.ndarray):
        return _normalized(value.tolist())
    if isinstance(value, np.generic):
        return _normalized(value.item())
    if isinstance(value, float) and math.isfinite(value):
        return float(f"{value:.6g}")
    return value


def _analysis(mix_id: str, vad: bool) -> dict[str, Any]:
    mix = built_mix(mix_id)
    size = int(CROP_SECONDS * SR)
    mix = {**mix, "raw": mix["raw"][:size], "clean": mix["clean"][:size], "labels": mix["labels"][:size],
           "noise_capture": mix["noise_capture"][: int(NOISE_SECONDS * SR)]}
    ctx = bm.make_context(mix)
    preset = Preset()
    result = analyze_voice_setup(
        ctx["noise"].astype(np.float32), ctx["noisy"].astype(np.float32), SR, "broadcast", vad_available=vad,
        incumbent_settings={"gate": asdict(preset.gate), "rnnoise": asdict(preset.rnnoise),
                            "eq": EQSettings().to_dict()},
        raw_noise_audio=ctx["raw_noise"], raw_speech_audio=ctx["raw"])
    return _normalized(result)


def _is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _differences(stored: Any, actual: Any, path: str = "$") -> list[str]:
    if isinstance(stored, dict) and isinstance(actual, dict):
        found = [f"{path}.{key}: only in one of them" for key in stored.keys() ^ actual.keys()]
        for key in stored.keys() & actual.keys():
            found += _differences(stored[key], actual[key], f"{path}.{key}")
        return found
    if isinstance(stored, list) and isinstance(actual, list):
        if len(stored) != len(actual):
            return [f"{path}: {len(stored)} items stored, {len(actual)} now"]
        return [d for i, (a, b) in enumerate(zip(stored, actual)) for d in _differences(a, b, f"{path}[{i}]")]
    if _is_number(stored) and _is_number(actual):
        close = stored == actual or math.isclose(stored, actual, rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE)
        return [] if close else [f"{path}: stored {stored!r}, now {actual!r}"]
    return [] if type(stored) is type(actual) and stored == actual else [f"{path}: stored {stored!r}, now {actual!r}"]


def _read() -> dict[str, Any]:
    return json.loads(PIN_PATH.read_text(encoding="utf-8"))


def _write(data: dict[str, Any]) -> None:
    cases = ",\n".join(f" {json.dumps(name)}: {json.dumps(case, sort_keys=True, separators=(',', ':'))}"
                       for name, case in data["cases"].items())
    PIN_PATH.write_text(
        f'{{\n"model_sha256": {json.dumps(data["model_sha256"])},\n"cases": {{\n{cases}\n}}}}\n',
        encoding="utf-8", newline="\n")


def _silero_model() -> Path | None:
    model = Path(os.environ.get("VAD_MODEL_PATH") or bm.REPO_ROOT / "models" / "silero_vad.onnx")
    return model if model.is_file() else None


@pytest.mark.parametrize(("name", "mix_id", "vad"), CASES, ids=[case[0] for case in CASES])
def test_the_analysis_decides_what_it_decided_before(monkeypatch, name, mix_id, vad):
    model = _silero_model()
    if vad:
        if model is None:
            pytest.skip("Silero model not provisioned (set VAD_MODEL_PATH or place models/silero_vad.onnx)")
        monkeypatch.setenv("VAD_MODEL_PATH", str(model))
        assert bm.silero_available()
    actual = _analysis(mix_id, vad)
    if UPDATE:
        stored = _read() if PIN_PATH.is_file() else {"model_sha256": None, "cases": {}}
        if vad and model is not None:
            stored["model_sha256"] = bm.file_sha256(model)
        stored["cases"][name] = actual
        stored["cases"] = dict(sorted(stored["cases"].items()))
        _write(stored)
        pytest.skip(f"regenerated {name} in {PIN_PATH.name}")
    stored = _read()
    if vad and model is not None and stored["model_sha256"] != bm.file_sha256(model):
        pytest.skip("the provisioned Silero model is not the one the stored values were made with")
    differences = _differences(stored["cases"][name], actual)
    assert not differences, "\n".join(differences[:12])

