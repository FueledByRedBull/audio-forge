"""Voice Setup's analysis returns the same settings, decision and report on fixed bench takes.

Each case runs ``analyze_voice_setup`` on the first 6 s of a bench mix, as the app's offline flow does, and
compares what comes back (the chosen settings, the apply recommendation with its reasons, and the whole report)
with ``fixtures/golden/voice_setup_analysis.json``. A change that only makes the analysis faster (parallel
candidates, shared evidence) has to leave all of it alone. Timings (the ``runtime`` fields) are left out. Numbers are
stored to 6 significant digits and match within 1e-4 of their size, so another CPU's last-bit differences pass;
booleans, reasons and counts match exactly.

The Silero cases need the model (``VAD_MODEL_PATH`` or ``models/silero_vad.onnx``) and skip without it or with
another model. The energy case runs wherever the native extension does, with nothing a Silero lookup can find:
Voice Setup told that VAD is unavailable still has Auto-EQ run the model whenever it is there.

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
from mic_eq.analysis import compressor_calibration, joint_tuning
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
# (case name, bench mix, Silero VAD, strength of the user's noise suppression). The energy cases run without a
# model. A nearly-off suppressor is the incumbent a candidate beats, so that case reaches the held-out check
# and applies; the others keep the factory settings, which the search leaves alone.
CASES = (
    ("fan_t01_silero", "fan-t01", True, None),
    ("cafe_t02_silero", "cafe-t02", True, None),
    ("fan_t01_energy", "fan-t01", False, None),
    ("fan_t01_energy_weak_incumbent", "fan-t01", False, 0.05),
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


def _analysis(mix_id: str, vad: bool, suppressor_strength: float | None = None) -> dict[str, Any]:
    mix = built_mix(mix_id)
    size = int(CROP_SECONDS * SR)
    mix = {**mix, "raw": mix["raw"][:size], "clean": mix["clean"][:size], "labels": mix["labels"][:size],
           "noise_capture": mix["noise_capture"][: int(NOISE_SECONDS * SR)]}
    ctx = bm.make_context(mix)
    preset = Preset()
    suppressor = asdict(preset.rnnoise)
    if suppressor_strength is not None:
        suppressor["strength"] = suppressor_strength
    result = analyze_voice_setup(
        ctx["noise"].astype(np.float32), ctx["noisy"].astype(np.float32), SR, "broadcast", vad_available=vad,
        incumbent_settings={"gate": asdict(preset.gate), "rnnoise": suppressor, "eq": EQSettings().to_dict()},
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


def _without_silero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Leave no model for the native lookup: not by VAD_MODEL_PATH, ./models, ../models or LOCALAPPDATA."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.chdir(run_dir)
    monkeypatch.delenv("VAD_MODEL_PATH", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))


@pytest.mark.parametrize(("name", "mix_id", "vad", "strength"), CASES, ids=[case[0] for case in CASES])
def test_the_analysis_decides_what_it_decided_before(monkeypatch, tmp_path, name, mix_id, vad, strength):
    model = _silero_model()
    if vad:
        if model is None:
            pytest.skip("Silero model not provisioned (set VAD_MODEL_PATH or place models/silero_vad.onnx)")
        if not UPDATE and _read()["model_sha256"] != bm.file_sha256(model):
            pytest.skip("the provisioned Silero model is not the one the stored values were made with")
        monkeypatch.setenv("VAD_MODEL_PATH", str(model))
        assert bm.silero_available()
    else:
        _without_silero(monkeypatch, tmp_path)
    actual = _analysis(mix_id, vad, strength)
    if UPDATE:
        stored = _read() if PIN_PATH.is_file() else {"model_sha256": None, "cases": {}}
        if vad and model is not None:
            stored["model_sha256"] = bm.file_sha256(model)
        stored["cases"][name] = actual
        stored["cases"] = dict(sorted(stored["cases"].items()))
        _write(stored)
        pytest.skip(f"regenerated {name} in {PIN_PATH.name}")
    differences = _differences(_read()["cases"][name], actual)
    assert not differences, "\n".join(differences[:12])


@pytest.mark.parametrize(("name", "mix_id", "vad", "strength"), [case for case in CASES if not case[2]],
                         ids=[case[0] for case in CASES if not case[2]])
def test_one_worker_decides_what_the_default_workers_decide(monkeypatch, tmp_path, name, mix_id, vad, strength):
    """The stored energy cases were made by the serial analysis; the worker count may not change them."""
    monkeypatch.setattr(joint_tuning, "ANALYSIS_WORKERS", 1)
    monkeypatch.setattr(compressor_calibration, "ANALYSIS_WORKERS", 1)
    _without_silero(monkeypatch, tmp_path)

    differences = _differences(_read()["cases"][name], _analysis(mix_id, vad, strength))
    assert not differences, "\n".join(differences[:12])


def test_the_pinned_weak_incumbent_reaches_the_apply_and_held_out_branch():
    """Without this the pin could go on passing with the whole candidate search retaining the incumbent."""
    tuning = _read()["cases"]["fan_t01_energy_weak_incumbent"]["diagnostics"]["joint_tuning"]

    assert tuning["decision"] == "apply_candidate" and tuning["apply_recommended"]
    assert tuning["holdout_evaluated"] and tuning["holdout_passed"]
    assert tuning["suppressor_settings"]["strength"] != 0.05
