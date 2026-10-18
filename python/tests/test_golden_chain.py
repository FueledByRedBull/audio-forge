"""Golden expected outputs of the offline full chain at factory settings, for native rebuilds.

Each case renders a fixed input through ``simulate_auto_eq_chain`` (RNNoise, factory ``Preset()``, the VAD
posterior supplied from the input's known speech frames, so no model is needed) and compares compact
expected values stored in ``fixtures/golden/``: the RMS and peak of every 10 ms block of the aligned output
in dB, and for the cases with a change in the signal (a gate opening, hum starting or moving, a level step,
a sibilant burst, the limiter catching a burst) the largest sample step and the energy and largest step in a
window around each change.

Why these suffice: the block curves pin the steady-state level of every stage within the stated tolerance,
so a fix may not move sound outside its window. The largest step is the size of the worst click, and the
window energy is how much the change disturbs the sound around it, which are the two things a transition fix
has to make smaller. Those artifact numbers fail only when they grow, and the blocks inside a change's window
(20 ms before to 40 ms after) are not compared, so a transition fix that shrinks the artifact and leaves the
steady-state blocks alone passes; it regenerates the transition file in its own pull request. Nothing here stores
audio.

The offline entry points render one fixed configuration per call, so a stage toggled, an EQ retuned or a
parameter changed in the middle of a render cannot be reached from Python; the changes below are the ones the
signal itself makes the native chain track (its gate, hum and rumble cleanup, compressor and limiter). Those
mid-render changes live in the native block-processor tests.

Regenerate only with ``AUDIOFORGE_UPDATE_GOLDEN=1 python -m pytest python/tests/test_golden_chain.py``, in a
pull request meant to change the sound, and say why.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bench_metrics as bm
from mic_eq import CORE_AVAILABLE

pytestmark = pytest.mark.skipif(not CORE_AVAILABLE, reason="native extension is not built")

SR = bm.SAMPLE_RATE
BLOCK = 480
GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures" / "golden"
UPDATE = os.environ.get("AUDIOFORGE_UPDATE_GOLDEN") == "1"
TOLERANCE = {
    # Every 10 ms block's RMS and peak, in dB, after flooring both at floor_db (digital silence reads as -200).
    "block_db": 0.1,
    "floor_db": -90.0,
    # The largest sample step, relative (and an absolute allowance in full-scale units), and a window's energy in dB.
    "step_relative": 0.05,
    "step_absolute": 2e-5,
    "window_energy_db": 0.2,
}
BEFORE, AFTER = int(0.020 * SR), int(0.040 * SR)  # the window around a change: 20 ms before, 40 ms after


def _rng(name: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(name.encode()))


def _db_floor(value: np.ndarray) -> np.ndarray:
    return np.maximum(value, TOLERANCE["floor_db"])


def _active(audio: np.ndarray, level_dbfs: float) -> np.ndarray:
    return audio * 10.0 ** ((level_dbfs - bm.active_level_db(audio)) / 20.0)


def _energy_labels(audio: np.ndarray) -> np.ndarray:
    frames = bm.frame_rms_db(audio)
    return np.repeat(frames > frames.max() - 35.0, BLOCK)[: audio.size]


def _speech_like(name: str, seconds: float, level_dbfs: float = -26.0) -> np.ndarray:
    return _active(bm.speech_like(int(seconds * SR), _rng(name)), level_dbfs)


def _fade(audio: np.ndarray, start: int, stop: int, ms: float = 10.0) -> np.ndarray:
    envelope = np.zeros(audio.size)
    ramp = int(ms * SR / 1000)
    envelope[start:stop] = 1.0
    kernel = np.ones(ramp) / ramp
    return audio * np.convolve(envelope, kernel, "same")


def _case(audio: np.ndarray, labels: np.ndarray, *, changes: tuple[int, ...] = (), chain: dict[str, Any] | None = None,
          edit: Any = None, note: str = "") -> dict[str, Any]:
    assert audio.size % BLOCK == 0 and labels.size == audio.size
    return {"audio": np.asarray(audio, np.float32), "labels": labels, "changes": changes, "chain": chain or {},
            "edit": edit, "note": note}


def _sweep() -> dict[str, Any]:
    t = np.arange(3 * SR) / SR
    phase = 2 * np.pi * 100.0 * 3.0 / np.log(80.0) * (80.0 ** (t / 3.0) - 1.0)  # 100 Hz to 8 kHz, logarithmic
    audio = _fade(0.1 * np.sin(phase), SR // 4, 3 * SR - SR // 4)
    return _case(audio, np.abs(audio) > 1e-4, note="log tone sweep 100 Hz-8 kHz at -23 dBFS peak, 250 ms of silence at each end")


def _bursts() -> dict[str, Any]:
    rng = _rng("bursts")
    audio = 0.0005 * rng.standard_normal(4 * SR)
    labels = np.zeros(audio.size, bool)
    changes = []
    for start in (0.5, 1.4, 2.3, 3.2):
        a, b = int(start * SR), int((start + 0.4) * SR)
        audio[a:b] += 0.03 * rng.standard_normal(b - a)
        labels[a:b] = True
        changes += [a, b]
    return _case(audio, labels, changes=tuple(changes), note="four 0.4 s noise bursts (-30 dBFS) in a -66 dBFS floor; the gate opens and closes at each edge")


def _speech() -> dict[str, Any]:
    audio = _speech_like("speech", 3.0)
    return _case(audio, _energy_labels(audio), note="synthetic voice, syllable-gated, -26 dBFS active")


def _hot() -> dict[str, Any]:
    audio = _speech_like("hot", 3.0)
    audio *= 0.99 / np.max(np.abs(audio))
    return _case(audio, _energy_labels(audio), note="the synthetic voice peaking at 0.99 full scale")


def _bench_crop(mix_id: str, seconds: float = 4.0, chain: dict[str, Any] | None = None) -> dict[str, Any]:
    mix = bm.build_mix(bm.load_bench(), mix_id)
    size = int(seconds * SR)
    return _case(mix["raw"][:size], mix["labels"][:size], chain=chain, note=f"the first {seconds:g} s of bench mix {mix_id}")


def _next_onset(labels: np.ndarray, after_s: float, offset_s: float = 0.0) -> int:
    """The first speech onset at or after ``after_s``, so a change lands where the voice is sounding."""
    onsets = np.flatnonzero(labels[1:] & ~labels[:-1]) + 1
    return int(onsets[onsets >= int(after_s * SR)][0]) + int(offset_s * SR)


def _hum_retune() -> dict[str, Any]:
    speech = _speech_like("hum", 4.0)
    labels = _energy_labels(speech)
    start, move = _next_onset(labels, 1.0), _next_onset(labels, 2.5)
    rng = _rng("hum-tone")
    hum = np.zeros(speech.size)
    for a, b, hz in ((start, move, 50.0), (move, speech.size, 60.0)):
        hum += _fade(bm.hum(speech.size, rng, hz), a, b, 20.0)
    audio = speech + _active(hum, -34.0)
    return _case(audio, labels, changes=(start, move), chain={"input_cleanup_mode": "strong"},
                 note="voice with mains hum that starts at 50 Hz with a phrase after 1.0 s and moves to 60 Hz with the "
                      "phrase after 2.5 s; strong input cleanup retunes its notches")


def _rumble() -> dict[str, Any]:
    speech = _speech_like("rumble", 4.0)
    t = np.arange(speech.size) / SR
    rumble = _fade(np.sin(2 * np.pi * 28.0 * t) + 0.5 * np.sin(2 * np.pi * 41.0 * t), int(1.5 * SR), speech.size, 20.0)
    return _case(speech + _active(rumble, -26.0), _energy_labels(speech), changes=(int(1.5 * SR),),
                 chain={"input_cleanup_mode": "gentle"}, note="voice with low-frequency rumble from 1.5 s; gentle cleanup raises its high-pass")


def _level_step() -> dict[str, Any]:
    speech = _speech_like("step", 4.0, -30.0)
    labels = _energy_labels(speech)
    step = _next_onset(labels, 2.0, 0.1)  # 100 ms into the phrase after 2.0 s
    audio = speech.copy()
    audio[step:] *= 10.0 ** (24.0 / 20.0)
    return _case(audio, labels, changes=(step,),
                 note="the synthetic voice 24 dB louder from 100 ms into the phrase after 2.0 s; the compressor and "
                      "automatic makeup follow")


def _sibilant_burst() -> dict[str, Any]:
    speech = _speech_like("sibilant", 4.0)
    rng = _rng("sibilant-noise")
    spectrum = np.fft.rfft(rng.standard_normal(speech.size))
    freqs = np.fft.rfftfreq(speech.size, 1.0 / SR)
    spectrum[(freqs < 4_000.0) | (freqs > 11_000.0)] = 0.0
    hiss = np.fft.irfft(spectrum, speech.size)
    changes = (int(1.5 * SR), int(1.8 * SR), 3 * SR, int(3.3 * SR))
    gated = _fade(hiss, changes[0], changes[1], 5.0) + _fade(hiss, changes[2], changes[3], 5.0)
    audio = speech + _active(gated, -22.0)
    labels = _energy_labels(speech) | (np.abs(gated) > 0)
    return _case(audio, labels, changes=changes, edit=lambda s: s["deesser_settings"].update(enabled=True),
                 note="voice with two 0.3 s bursts of 4-11 kHz hiss (-22 dBFS) and the de-esser switched on")


def _limiter_burst() -> dict[str, Any]:
    speech = _speech_like("limiter", 3.0, -30.0)
    t = np.arange(speech.size) / SR
    tone = _fade(0.5 * np.sin(2 * np.pi * 1_000.0 * t), SR, 2 * SR, 5.0)
    labels = _energy_labels(speech) | (np.abs(tone) > 0)
    return _case(speech + tone, labels, changes=(SR, 2 * SR),
                 edit=lambda s: s["compressor_settings"].update(auto_makeup_enabled=False, makeup_gain_db=12.0),
                 note="a -6 dBFS tone burst from 1.0 to 2.0 s with 12 dB of manual makeup, so the limiter catches it")


CASES = {
    "sweep": _sweep, "noise_bursts": _bursts, "speech_like": _speech, "hot_near_clipping": _hot,
    "bench_room_t02": lambda: _bench_crop("room-t02"), "bench_cafe_t02": lambda: _bench_crop("cafe-t02"),
    "transition_hum_retune": _hum_retune, "transition_rumble_onset": _rumble,
    "transition_level_step": _level_step, "transition_sibilant_burst": _sibilant_burst,
    "transition_limiter_burst": _limiter_burst,
}


def _probabilities(labels: np.ndarray) -> list[float]:
    """The VAD posterior at the 10 ms cadence: 0.95 in frames with speech, 0.05 elsewhere."""
    frames = labels[: labels.size // BLOCK * BLOCK].reshape(-1, BLOCK).any(axis=1)
    return np.where(frames, 0.95, 0.05).round(2).tolist()


def _render(case: dict[str, Any], *, vad: bool | list[float] = True, edit: Any = None) -> np.ndarray:
    settings = bm.current_settings(True)
    for change in (case["edit"], edit):
        if change is not None:
            change(settings)
    chain = dict(case["chain"])
    if vad is True:
        chain["vad_probabilities"] = _probabilities(case["labels"])
    result = bm.render(settings, case["audio"], chain)
    output = np.asarray(result["output_audio"], np.float32)
    assert output.shape == case["audio"].shape
    return output


def _summary(case: dict[str, Any], output: np.ndarray) -> dict[str, Any]:
    blocks = output.astype(np.float64).reshape(-1, BLOCK)
    rms = _db_floor(10.0 * np.log10(np.mean(blocks**2, axis=1) + 1e-20))
    peak = _db_floor(20.0 * np.log10(np.max(np.abs(blocks), axis=1) + 1e-10))
    summary: dict[str, Any] = {"note": case["note"], "samples": int(output.size), "rms_db": rms.round(2).tolist(),
                               "peak_db": peak.round(2).tolist()}
    if case["changes"]:
        out = output.astype(np.float64)
        windows = [out[max(0, c - BEFORE) : c + AFTER] for c in case["changes"]]
        summary |= {
            "changes": [int(c) for c in case["changes"]],
            "max_step": round(float(np.max(np.abs(np.diff(out)))), 5),
            "window_energy_db": [round(float(_db_floor(10.0 * np.log10(np.sum(w**2) + 1e-20))), 2) for w in windows],
            "window_max_step": [round(float(np.max(np.abs(np.diff(w)))), 5) for w in windows],
        }
    return summary


def _change_blocks(summary: dict[str, Any]) -> np.ndarray:
    """The 10 ms blocks that touch a change's window (20 ms before to 40 ms after); empty without changes."""
    mask = np.zeros(len(summary["rms_db"]), bool)
    for change in summary.get("changes", ()):
        mask[max(0, change - BEFORE) // BLOCK : (change + AFTER - 1) // BLOCK + 1] = True
    return mask


def _differences(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """What differs from the stored values. Steady state is compared both ways, outside the change windows.

    The artifact numbers (the largest step, each window's largest step and energy) fail only when larger than
    stored, and the blocks inside a change window are not compared, so a transition fix that shrinks the artifact
    and leaves the steady state alone passes. It regenerates the transition file in its own pull request.
    """
    found: list[str] = []
    if expected["samples"] != actual["samples"] or expected.get("changes") != actual.get("changes"):
        return ["the input shape differs from the stored one"]
    steady = ~_change_blocks(expected)
    for key in ("rms_db", "peak_db"):
        delta = np.abs(np.array(expected[key]) - np.array(actual[key]))[steady]
        bad = np.flatnonzero(delta > TOLERANCE["block_db"])
        if bad.size:
            found.append(f"{key}: {bad.size} of {delta.size} steady blocks differ by more than {TOLERANCE['block_db']} dB "
                         f"(worst {delta.max():.2f} dB)")
    if "max_step" in expected:
        for key in ("max_step", "window_max_step"):
            want, got = np.atleast_1d(expected[key]), np.atleast_1d(actual[key])
            if np.any(got > want * (1.0 + TOLERANCE["step_relative"]) + TOLERANCE["step_absolute"]):
                found.append(f"{key}: larger than stored {want.tolist()}, got {got.tolist()}")
        want, got = np.array(expected["window_energy_db"]), np.array(actual["window_energy_db"])
        if np.any(got > want + TOLERANCE["window_energy_db"]):
            found.append(f"window_energy_db: more than stored {want.tolist()}, got {got.tolist()}")
    return found


def _write(path: Path, header: dict[str, Any], cases: dict[str, dict[str, Any]]) -> None:
    """One line per case, so a regenerated file diffs by case."""
    lines = [f'  "{key}": {json.dumps(value, separators=(",", ":"))}' for key, value in header.items()]
    body = ",\n".join(f'    "{name}": {json.dumps(case, separators=(",", ":"))}' for name, case in cases.items())
    path.write_text("{\n" + ",\n".join(lines) + ',\n  "cases": {\n' + body + "\n  }\n}\n", encoding="utf-8", newline="\n")


def _header(**extra: Any) -> dict[str, Any]:
    return {"purpose": "expected values of the offline full chain at factory settings with RNNoise; see test_golden_chain.py",
            "sample_rate": SR, "block_samples": BLOCK, "tolerance": TOLERANCE, **extra}


def _load(path: Path) -> dict[str, Any]:
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["tolerance"] == TOLERANCE, "the tolerance in the file and in the test must be the same"
    assert stored["sample_rate"] == SR and stored["block_samples"] == BLOCK
    return stored


def test_cases_match_the_stored_expected_values():
    path = GOLDEN_DIR / "factory_chain.json"
    actual = {name: _summary(case, _render(case)) for name, build in CASES.items() for case in [build()]}
    if UPDATE:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        _write(path, _header(), actual)
        pytest.skip(f"regenerated {path.name}")
    stored = _load(path)
    assert set(stored["cases"]) == set(actual)
    failures = {name: found for name in actual if (found := _differences(stored["cases"][name], actual[name]))}
    assert not failures, failures


def test_the_comparison_fails_when_one_setting_moves():
    """A +0.5 dB boost of the EQ band nearest 1 kHz must show up, on a bench crop and on the synthetic voice."""
    def boost(settings: dict[str, Any]) -> None:
        bands = settings["eq_settings"]["bands"]
        nearest = min(bands, key=lambda band: abs(band["frequency_hz"] - 1_000.0))
        nearest["gain_db"] = nearest["gain_db"] + 0.5

    stored = _load(GOLDEN_DIR / "factory_chain.json") if not UPDATE else None
    if stored is None:
        pytest.skip("regeneration run")
    for name in ("bench_room_t02", "speech_like"):
        case = CASES[name]()
        assert _differences(stored["cases"][name], _summary(case, _render(case, edit=boost))), name


def test_a_transition_fix_passes_and_a_worse_artifact_fails():
    stored = _load(GOLDEN_DIR / "factory_chain.json")["cases"]["transition_limiter_burst"]

    def variant(step_scale: float, energy_db: float, window_block_db: float = 0.0, steady_block_db: float = 0.0) -> dict:
        case = copy.deepcopy(stored)
        case["max_step"] *= step_scale
        case["window_max_step"] = [value * step_scale for value in case["window_max_step"]]
        case["window_energy_db"] = [value + energy_db for value in case["window_energy_db"]]
        inside = np.flatnonzero(_change_blocks(stored))
        outside = np.flatnonzero(~_change_blocks(stored))
        case["rms_db"][inside[0]] += window_block_db
        case["rms_db"][outside[0]] += steady_block_db
        return case

    assert not _differences(stored, variant(1.0, 0.0))
    # Half the step and 1 dB less energy around each change, with the blocks inside the windows moved, passes.
    assert not _differences(stored, variant(0.5, -1.0, window_block_db=-3.0))
    # A larger step, more energy, or a moved steady-state block fails.
    assert _differences(stored, variant(1.5, 0.0))
    assert _differences(stored, variant(1.0, 1.0))
    assert _differences(stored, variant(0.5, -1.0, steady_block_db=0.5))


def test_the_silero_gate_matches_its_stored_values_where_the_model_is_provisioned(monkeypatch):
    path = GOLDEN_DIR / "gate_silero.json"
    model = Path(os.environ.get("VAD_MODEL_PATH") or bm.REPO_ROOT / "models" / "silero_vad.onnx")
    if not model.is_file():
        pytest.skip("Silero model not provisioned (set VAD_MODEL_PATH or place models/silero_vad.onnx)")
    monkeypatch.setenv("VAD_MODEL_PATH", str(model))
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    case = _bench_crop("room-t02")
    actual = {"bench_room_t02_silero_gate": _summary(case, _render(case, vad=False))}
    if UPDATE:
        _write(path, _header(model_sha256=digest), actual)
        pytest.skip(f"regenerated {path.name}")
    stored = _load(path)
    if stored["model_sha256"] != digest:
        pytest.skip("the provisioned Silero model is not the one the stored values were made with")
    assert bm.silero_available()
    assert not _differences(stored["cases"]["bench_room_t02_silero_gate"], actual["bench_room_t02_silero_gate"])


def test_every_case_renders_the_same_twice_and_is_finite():
    for name in ("noise_bursts", "transition_limiter_burst"):
        case = CASES[name]()
        first, second = _render(case), _render(copy.deepcopy(case))
        assert np.isfinite(first).all() and first.tobytes() == second.tobytes(), name
