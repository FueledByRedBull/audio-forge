"""Tests for the Voice Setup refusal study: its judgement, taps, counting, report and guards.

The hand-made world is ``fabricated_rows()``: the 54 study users in 18 speakers, covering every bucket and
verdict. Its first-veto buckets are B1 3, B2 5, B3 2, B4 2, B5 3, C1 2, C2 2, none 34 (rows 18-21 and the
filler) and one analysis failure.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import check_evaluation_hygiene
from tools import evaluate_voice_setup_refusals as ev

voice_setup_eval, loudness, gate_corpus = ev.voice_setup_eval, ev.loudness, ev.gate_corpus
ANALYSIS = Path(__file__).resolve().parents[1] / "mic_eq" / "analysis"
OK = {"estoi": 0.5, "tone_error_db": 2.0, "sibilant_loss_db": 1.0, "deep_pause_db": -10.0, "tail_db": -8.0,
      "onset_db": -1.0, "pause_db": -9.0, "clipped": False, "limiter_gain_reduction_db": 0.5, "true_peak_limiter_gain_reduction_db": 0.1}
IDENTITY = {name: "0" * 64 for name in ("extension_sha256", "manifest_sha256", "vad_model_sha256", "df_dll_sha256")}


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    monkeypatch.setattr(gate_corpus, "BOOTSTRAP", 200)


@pytest.fixture
def no_vad_model(monkeypatch, tmp_path):
    """Hide the Silero model as CI's Python jobs lack it: the native lookup reads VAD_MODEL_PATH and ./models."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.chdir(run_dir)
    monkeypatch.setenv("VAD_MODEL_PATH", str(tmp_path / "missing" / "silero_vad.onnx"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))


@pytest.fixture
def frozen_tree(monkeypatch, tmp_path):
    """A freeze of the real code with stand-in assets (CI has none), written under a scratch directory."""
    assets = {"extension": "1" * 64, "df.dll": "2" * 64, "onnxruntime.dll": "6" * 64,
              "onnxruntime_providers_shared.dll": "7" * 64, "vad_model": "3" * 64, "corpus_manifest": "4" * 64,
              "models/DeepFilterNet3_ll_onnx.tar.gz": "8" * 64, "models/DeepFilterNet3_onnx.tar.gz": "5" * 64}
    monkeypatch.setattr(ev, "asset_hashes", lambda root=None: dict(assets))
    monkeypatch.setattr(ev, "check_assets", lambda: None)  # CI has none of the assets; the real check is tested apart
    for name, value in (("FROZEN", "frozen-files.json"), ("RULES_FINAL", "rules-final.md"), ("DEFAULT_ROWS", "rows.json"),
                        ("DEFAULT_REPORT", "report.json")):
        monkeypatch.setattr(ev, name, tmp_path / "details" / value)
    ev.build_id.cache_clear()
    ev.freeze(ev.FROZEN, ev.RULES_FINAL, ev.DEFAULT_ROWS, tmp_path)
    yield json.loads(ev.FROZEN.read_text(encoding="utf-8"))
    ev.build_id.cache_clear()


def _with(**changes: Any) -> dict[str, Any]:
    return {**OK, **changes}


# ---------------------------------------------------------------------------------------------------------------
# 1. The judgement on hand-made cases


@pytest.mark.parametrize("metric, value, expected", [
    ("estoi", 0.5 - 0.0099, "pass"), ("estoi", 0.5 - 0.0100, "pass"), ("estoi", 0.5 - 0.0101, "fail"), ("estoi", 0.9, "pass"),
    ("tone_error_db", 2.99, "pass"), ("tone_error_db", 3.00, "pass"), ("tone_error_db", 3.01, "fail"),
    ("tone_error_db", 0.0, "pass"),
    ("sibilant_loss_db", 3.99, "pass"), ("sibilant_loss_db", 4.00, "pass"), ("sibilant_loss_db", 4.01, "fail"),
    ("deep_pause_db", -9.01, "pass"), ("deep_pause_db", -9.00, "pass"), ("deep_pause_db", -8.99, "fail"),
    ("deep_pause_db", -20.0, "pass"),
    ("tail_db", -7.01, "pass"), ("tail_db", -7.00, "pass"), ("tail_db", -6.99, "fail"),
    ("onset_db", -1.99, "pass"), ("onset_db", -2.00, "pass"), ("onset_db", -2.01, "fail"), ("onset_db", 3.0, "pass"),
    ("estoi", float("nan"), "unmeasurable"), ("tone_error_db", None, "unmeasurable"),
    ("sibilant_loss_db", None, "unmeasurable"), ("deep_pause_db", float("nan"), "unmeasurable"),
])
def test_boundary_values_pass_and_directions_are_right(metric, value, expected):
    assert ev.judge(_with(**{metric: value}), OK)["clauses"][metric] == expected


@pytest.mark.parametrize("tolerance, value, expected", [
    (1.5, 2.5, "pass"), (1.5, 2.51, "fail"), (3.0, 4.0, "pass"), (6.0, 7.0, "pass"), (6.0, 7.01, "fail"),
])
def test_the_sibilant_sensitivity_readings(tolerance, value, expected):
    judged = ev.judge(_with(sibilant_loss_db=value), OK, sibilant_tolerance=tolerance)
    assert judged["clauses"]["sibilant_loss_db"] == expected


@pytest.mark.parametrize("withheld, current, strict, expected", [
    ({"clipped": True}, {}, False, "fail"),                                   # only the withheld setup clips
    ({"clipped": True}, {"clipped": True}, False, "pass"),                    # both clip: attributed reading
    ({"clipped": True}, {"clipped": True}, True, "fail"),                     # the strict reading beside it
    ({}, {"clipped": True}, False, "pass"),
    ({"limiter_gain_reduction_db": 6.0}, {}, False, "fail"),
    ({"limiter_gain_reduction_db": 5.99}, {}, False, "pass"),
    ({"true_peak_limiter_gain_reduction_db": 3.0}, {}, False, "fail"),
    ({"true_peak_limiter_gain_reduction_db": 2.99}, {}, False, "pass"),
    ({"limiter_gain_reduction_db": 7.0}, {"limiter_gain_reduction_db": 6.5}, False, "pass"),
    ({"limiter_gain_reduction_db": 7.0}, {"limiter_gain_reduction_db": 6.5}, True, "fail"),
    ({"limiter_gain_reduction_db": float("nan")}, {}, False, "unmeasurable"),
])
def test_clipping_and_limiting_count_against_a_setup_the_current_one_does_not_share(withheld, current, strict, expected):
    assert ev.judge(_with(**withheld), _with(**current), strict=strict)["clauses"]["overload"] == expected


def test_verdicts_and_the_two_bounds_of_an_unmeasurable_clause():
    assert ev.judge(OK, OK)["verdict"] == "false"
    two = ev.judge(_with(estoi=0.4, tone_error_db=5.0), OK)
    assert two["verdict"] == "correct" and two["failed"] == ["estoi", "tone_error_db"]
    mixed = ev.judge(_with(tone_error_db=5.0, deep_pause_db=None), OK)
    assert mixed["verdict"] == "correct" and mixed["failed"] == ["tone_error_db"] and mixed["unmeasurable"] == ["deep_pause_db"]
    partial = ev.judge(_with(deep_pause_db=float("nan")), OK)
    assert partial["verdict"] == "partial" and partial["failed"] == []
    assert [ev.false_refusal("partial", bound) for bound in ("lower", "upper")] == [False, True]
    assert [ev.false_refusal("correct", bound) for bound in ("lower", "upper")] == [False, False]
    assert [ev.false_refusal("false", bound) for bound in ("lower", "upper")] == [True, True]


def test_whenever_the_gate_study_says_dominates_no_worse_holds():
    rng = np.random.default_rng(3)
    keys = list(voice_setup_eval.TOLERANCES)
    dominating = 0
    for _ in range(400):
        a, b = ({key: float(rng.choice([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]) * 0.01 ** (key == "estoi"))
                 for key in keys} for _ in range(2))
        if voice_setup_eval.dominates(a, b):
            dominating += 1
            assert ev.no_worse({**OK, **a}, {**OK, **b})
    assert dominating > 20


# ---------------------------------------------------------------------------------------------------------------
# 2. Per-step settings


def _proposal() -> dict[str, Any]:
    current = ev.current_settings(True)
    proposal = deepcopy(current)
    for name in ev.SECTIONS:
        proposal[name] = {**proposal[name], "marker": name}
    return proposal


STEP_SECTIONS = {"tone": ("eq_settings",), "gate_suppression": ("gate_settings", "suppressor_settings"),
                 "deesser": ("deesser_settings",), "compressor": ("compressor_settings",)}


def test_the_steps_are_exactly_these_sections():
    assert ev.STEPS == STEP_SECTIONS


@pytest.mark.parametrize("step, sections", list(STEP_SECTIONS.items()))
def test_a_step_changes_only_its_own_sections(step, sections):
    current, proposal = ev.current_settings(True), _proposal()
    mixed = ev.hybrid(current, proposal, step)
    for name in ev.SECTIONS:
        source = proposal if name in sections else current
        assert loudness._digest(mixed[name]) == loudness._digest(source[name])
    assert mixed["diagnostics"] == current["diagnostics"]


def test_the_analysis_limiter_is_the_factory_limiter_so_there_is_no_limiter_step():
    from mic_eq.analysis.compressor_calibration import _normalise_limiter_settings

    factory = ev.current_settings(True)["limiter_settings"]
    assert loudness._digest(_normalise_limiter_settings(None)) == loudness._digest(factory)


def test_vacuous_means_the_withheld_sections_change_nothing():
    current = ev.current_settings(True)
    flat = {**current["eq_settings"], "band_gains": [0.0, 0.1, 0.2]}
    assert ev.is_vacuous({"eq_settings": flat}, current, ("eq_settings",))
    assert not ev.is_vacuous({"eq_settings": {**flat, "band_gains": [0.0, 0.25]}}, current, ("eq_settings",))
    assert ev.is_vacuous(deepcopy(current), current, ("gate_settings", "compressor_settings"))
    assert not ev.is_vacuous(_proposal(), current, ("gate_settings",))


# ---------------------------------------------------------------------------------------------------------------
# 3. Level matching and measurement


def _case(cycles: int = 4, modulated: bool = False) -> dict[str, Any]:
    """Pauses of 1.5 s and speech-like noise of 1.5 s, ``cycles`` times; ``modulated`` gives the noise a syllable-like
    envelope that a compressor can flatten."""
    rng = np.random.default_rng(1)
    rate = voice_setup_eval.SAMPLE_RATE

    def speech() -> np.ndarray:
        noise = 0.1 * rng.standard_normal(int(1.5 * rate))
        return noise * np.repeat(rng.uniform(0.3, 1.5, 30), int(0.05 * rate)) if modulated else noise

    clean = np.concatenate([np.concatenate([np.zeros(int(1.5 * rate)), speech()]) for _ in range(cycles)])
    labels = np.concatenate([np.concatenate([np.zeros(int(1.5 * rate), bool), np.ones(int(1.5 * rate), bool)])
                             for _ in range(cycles)])
    return {"clean": clean, "noisy": clean + 0.01 * rng.standard_normal(clean.size), "labels": labels}


def test_level_matching_removes_makeup_from_the_pause_and_tail_readings_only():
    case = _case()
    quiet, loud = case["noisy"], case["noisy"] * 10.0 ** (10.0 / 20.0)
    matched_quiet, matched_loud = ev.scored(case, quiet), ev.scored(case, loud)
    for metric in matched_quiet:
        assert matched_loud[metric] == pytest.approx(matched_quiet[metric], abs=1e-6)
    raw_quiet, raw_loud = gate_corpus.score(case, quiet), gate_corpus.score(case, loud)
    for metric in ("deep_pause_db", "tail_db", "pause_db"):
        assert raw_loud[metric] - raw_quiet[metric] == pytest.approx(10.0, abs=1e-6)
    for metric in ("estoi", "onset_db"):
        assert raw_loud[metric] == pytest.approx(raw_quiet[metric], abs=1e-9)
    assert math.isfinite(matched_quiet["deep_pause_db"]) and math.isfinite(matched_quiet["onset_db"])


def test_measure_reads_every_clause_and_the_last_window():
    case = _case()
    ctx = {**case, "noise": case["noisy"][: 2 * voice_setup_eval.SAMPLE_RATE], "raw": case["noisy"]}
    output = case["noisy"].copy()
    rendered = {"output_audio": output, "limiter_gain_reduction_db": 1.5, "true_peak_limiter_gain_reduction_db": 0.25}
    whole = ev.measure(rendered, ctx)
    assert set(whole) >= set(ev.RULES) | {"pause_db", "clipped"}
    assert whole["clipped"] is False and whole["limiter_gain_reduction_db"] == 1.5
    assert whole["tone_error_db"] < 6.0
    output[100] = 1.0
    assert ev.measure(rendered, ctx)["clipped"] is True
    assert ev.measure(rendered, ctx, start=output.size // 2)["clipped"] is False


def _loud_in(signal: np.ndarray, frames: range, seed: int) -> np.ndarray:
    """``signal`` with noise 14 dB above the pause floor added over the given 10 ms frames."""
    louder = signal.copy()
    start, stop = frames.start * gate_corpus.FRAME, frames.stop * gate_corpus.FRAME
    louder[start:stop] += 0.05 * np.random.default_rng(seed).standard_normal(stop - start)
    return louder


def test_deep_pause_and_tail_are_read_from_their_own_frames():
    case = _case()  # pauses are 150 frames; the second one starts at frame 300, after speech that ends at frame 299
    base = ev.scored(case, case["noisy"])
    deep = ev.scored(case, _loud_in(case["noisy"], range(365, 385), 1))  # 60 frames or more from speech on both sides
    tail = ev.scored(case, _loud_in(case["noisy"], range(305, 325), 2))  # within 30 frames after speech
    assert deep["deep_pause_db"] - base["deep_pause_db"] > 3.0 and abs(deep["tail_db"] - base["tail_db"]) < 0.5
    assert tail["tail_db"] - base["tail_db"] > 3.0 and abs(tail["deep_pause_db"] - base["deep_pause_db"]) < 0.5


def test_sibilant_loss_is_positive_when_the_render_takes_the_sibilants_off():
    rate = voice_setup_eval.SAMPLE_RATE
    case = _case(2)
    noisy = case["noisy"]
    ctx = {**case, "noise": noisy[: 2 * rate], "raw": noisy}
    bins = np.fft.rfftfreq(noisy.size, 1.0 / rate)
    dull = np.fft.irfft(np.fft.rfft(noisy) * (bins < 3_000.0), noisy.size)  # nothing above 3 kHz: the 5-9 kHz band is gone
    rendered = {"output_audio": dull, "limiter_gain_reduction_db": 0.0, "true_peak_limiter_gain_reduction_db": 0.0}
    assert ev.measure(rendered, ctx)["sibilant_loss_db"] > 10.0
    untouched = {**rendered, "output_audio": noisy}
    assert abs(ev.measure(untouched, ctx)["sibilant_loss_db"]) < 0.5


def test_a_render_within_float_noise_of_the_current_one_is_marked_identical(monkeypatch):
    case = _case(1)
    rate = voice_setup_eval.SAMPLE_RATE
    ctx = {**case, "noise": case["noisy"][: 2 * rate], "raw": case["noisy"].astype(np.float32)}
    monkeypatch.setattr(loudness, "_render", lambda settings, capture: {
        "output_audio": np.asarray(capture, np.float32) + np.float32(settings["offset"]),
        "limiter_gain_reduction_db": 0.0, "true_peak_limiter_gain_reduction_db": 0.0})
    assert "identical" not in ev.measure_settings({"offset": 0.0}, ctx, as_reference=True)
    assert ev.measure_settings({"offset": 1e-8}, ctx)["identical"] is True
    assert ev.measure_settings({"offset": 1e-3}, ctx)["identical"] is False


def test_a_render_that_fails_is_recorded_not_hidden(monkeypatch):
    def fail(*_args):
        raise RuntimeError("native simulator unavailable")

    monkeypatch.setattr(loudness, "_render", fail)
    result = ev.measure_settings({}, {"raw": np.zeros(10)})
    assert result == {"error": "RuntimeError: native simulator unavailable"}
    assert ev.outcome({"objects": {"whole": {"vacuous": False}}, "metrics": {"current": {"whole": OK}, "whole": result}},
                      "whole")["status"] == "unjudgeable"


def _signals(seconds: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    clean = 0.1 * rng.standard_normal(int(seconds * voice_setup_eval.SAMPLE_RATE))
    return clean, np.ones(clean.size, bool)


def test_tone_error_is_zero_for_identity_and_a_flat_gain_and_reads_a_spectral_step():
    clean, labels = _signals()
    assert ev.tone_error_db(clean, clean, labels) == pytest.approx(0.0, abs=1e-6)
    assert ev.tone_error_db(clean, clean * 2.0, labels) == pytest.approx(0.0, abs=1e-6)
    rate = voice_setup_eval.SAMPLE_RATE
    gain = np.where(np.fft.rfftfreq(clean.size, 1.0 / rate) >= 3_000.0, 10.0 ** (6.0 / 20.0), 1.0)
    stepped = np.fft.irfft(np.fft.rfft(clean) * gain, clean.size)
    bins = np.fft.rfftfreq(ev.TONE_FRAME, 1.0 / rate)
    delta = np.where(bins >= 3_000.0, 6.0, 0.0)[(bins >= 80.0) & (bins <= 12_000.0)]
    expected = float(np.sqrt(np.mean(np.square(delta - np.median(delta)))))
    assert expected == pytest.approx(6.0 * math.sqrt((3_000.0 - 80.0) / (12_000.0 - 80.0)), abs=0.03)
    assert ev.tone_error_db(clean, stepped, labels) == pytest.approx(expected, abs=0.15)


def test_tone_error_uses_only_frames_that_are_speech_in_both_signals_and_needs_enough_of_them():
    clean, labels = _signals()
    half = clean.size // 2
    other = clean.copy()
    other[:half] = np.random.default_rng(9).standard_normal(half)  # different audio where nothing is labelled speech
    labels[:half] = False
    assert ev.tone_error_db(clean, other, labels) == pytest.approx(0.0, abs=1e-6)
    assert math.isnan(ev.tone_error_db(clean[:15_000], clean[:15_000], np.ones(15_000, bool)))
    assert math.isnan(ev.tone_error_db(clean, clean, np.zeros(clean.size, bool)))


# ---------------------------------------------------------------------------------------------------------------
# 4. Taps


def test_the_validation_tap_records_a_failure_returns_a_pass_and_restores_the_original():
    from mic_eq.analysis import failure_detection

    original = failure_detection.validate_analysis
    freqs = np.linspace(0.0, 24_000.0, 2049)
    flat = np.zeros(freqs.size)  # no formant structure: too few peaks, noise-like
    assert not original({"band_gains": [0.0] * 10}, flat, freqs).passed
    with ev._taps() as log:
        result = failure_detection.validate_analysis({"band_gains": [0.0] * 10}, flat, freqs)
        assert result.passed and result.reason is None
    assert [entry["passed"] for entry in log["validation"]] == [False]
    assert any("peak_count" in failure for failure in log["validation"][0]["failures"])
    assert log["validation"][0]["reason"] == "Recording too unclear. Please try again."
    assert failure_detection.validate_analysis is original
    with pytest.raises(RuntimeError), ev._taps():
        raise RuntimeError("boom")
    assert failure_detection.validate_analysis is original


def test_the_optimizer_tap_records_the_validated_gains_and_restores_the_original(monkeypatch):
    from mic_eq.analysis.auto_eq_parts import optimizer

    solution = (np.array([1.0, -2.0]), 0.5, 0.25, 1.0, {})
    monkeypatch.setattr(optimizer, "_validate_and_attenuate_solution", lambda *args, **kwargs: solution)
    fake = optimizer._validate_and_attenuate_solution
    with ev._taps() as log:
        assert cast(Any, optimizer)._validate_and_attenuate_solution(1, 2, cancel_check=None) is solution
    assert log["tone"] == [[1.0, -2.0]]
    assert optimizer._validate_and_attenuate_solution is fake
    with pytest.raises(RuntimeError), ev._taps():
        raise RuntimeError("boom")
    assert optimizer._validate_and_attenuate_solution is fake


# ---------------------------------------------------------------------------------------------------------------
# 5. Veto mapping and drift


def _view(**changes: Any) -> dict[str, Any]:
    return {"reasons": [], "weak_capture": False, "b1": False, "tuner_decision": "apply_candidate",
            "verification": "accept", **changes}


@pytest.mark.parametrize("reason, veto", [
    ("compressor p95 gain reduction target missed: measured 1.20 dB, target 3.50 dB (±0.40 dB)", "B3"),
    ("compressor p95 gain reduction target is unavailable", "B3"),
    ("the microphone level leaves too little headroom; lower the input gain by about 3 dB", "B4"),
    ("final full-chain headroom validation did not pass", "B4"),
    ("full-chain gate/suppressor silenced active speech", "B4"),
    ("full-chain validation failed: boom", "B4"),
    ("offline DSP validation did not pass", "B4"),
    ("raw full-chain capture is unavailable for input validation", "B4"),
    ("Auto-EQ abstained from this capture", "B5"),
    ("too little VAD-active speech", "B2"), ("speech-to-noise ratio is weak", "B2"),
    ("too few voiced frames for a stable spectrum", "B2"), ("overall capture quality is low", "B2"),
    ("room-noise capture is not stationary", "B2"),
    ("offline DSP validation could not run, so these settings are advisory", "X"),
])
def test_every_reason_maps_to_its_veto(reason, veto):
    assert ev.reason_veto(reason) == veto
    expected = [] if veto in ("X", "B2") else [veto]  # B2 is the analysis' weak_capture switch, not its wording
    assert ev.fired_vetoes(_view(reasons=[reason])) == expected
    if veto == "B2":
        assert ev.fired_vetoes(_view(reasons=[reason], weak_capture=True)) == ["B2"]


def test_the_switches_fire_their_vetoes_in_order():
    fired = ev.fired_vetoes(_view(reasons=["Auto-EQ abstained from this capture", "compressor p95 gain reduction target is unavailable"],
                                  weak_capture=True, b1=True, tuner_decision="retain_incumbent", verification="rollback"))
    assert fired == ["B1", "B2", "B3", "B5", "B6", "C1"]
    assert ev.fired_vetoes(_view(verification="gave_up")) == ["C2"]
    assert ev.fired_vetoes(_view(verification="retry")) == []


def _literals(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return ["".join(str(part.value) if isinstance(part, ast.Constant) else "1.5" for part in node.values)]
    if isinstance(node, ast.BinOp):
        return ["".join(_literals(node.left) + _literals(node.right))]
    if isinstance(node, ast.IfExp):
        return _literals(node.body) + _literals(node.orelse)
    if isinstance(node, (ast.Tuple, ast.List)):
        return [text for element in node.elts for text in _literals(element)]
    return []


def _appended(module: str, receiver: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse((ANALYSIS / module).read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("append", "extend")
                and isinstance(node.func.value, ast.Name) and node.func.value.id == receiver):
            found += [text for arg in node.args for text in _literals(arg)]
    return found


def test_every_reason_the_analysis_can_give_has_a_veto():
    from mic_eq.analysis.noise_reference import SILENT_REFERENCE_REASON

    reasons = _appended("voice_setup.py", "uncertainty_reasons") + _appended("noise_reference.py", "reasons")
    reasons.append(SILENT_REFERENCE_REASON)
    for node in ast.walk(ast.parse((ANALYSIS / "compressor_calibration.py").read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef) and node.name == "_compressor_p95_target_status":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Return) and isinstance(inner.value, ast.Tuple):
                    reasons += _literals(inner.value.elts[1])
    assert len(reasons) > 25
    assert "Auto-EQ abstained from this capture" in reasons and "too little VAD-active speech" in reasons
    unmapped = sorted(reason for reason in reasons if ev.reason_veto(reason) is None)
    assert unmapped == [], f"these reasons have no veto in REASON_VETOES: {unmapped}"


# ---------------------------------------------------------------------------------------------------------------
# 6. Counting


def _users(rows=None) -> list[dict[str, Any]]:
    return ev.assess(rows or ev.fabricated_rows())


def test_first_follows_the_precedence_and_c_only_counts_for_a_recommended_setup():
    assert ev.first_whole({"B4", "B2", "C1"}, False) == "B2"
    assert ev.first_whole({"B5", "B3"}, False) == "B3"
    assert ev.first_whole({"C1", "C2"}, True) == "C1"
    assert ev.first_whole({"C1"}, False) is None
    assert ev.first_whole({"B6"}, True) is None  # B6 is the step track's


def test_the_buckets_add_up_and_each_user_has_exactly_one():
    users = _users()
    assert ev.population(users) == {"B1": 3, "B2": 5, "B3": 2, "B4": 2, "B5": 3, "C1": 2, "C2": 2, "none": 34,
                                    "analysis_failure": 1}
    assert len(users) == 54 and len({u["row"]["index"] for u in users}) == 54


def test_first_and_any_count_differently_and_the_tracks_are_independent():
    users = _users()
    first, anywhere = (ev._table_row(users, "B4", "whole", counting) for counting in ("first", "any"))
    assert (first["fired"], anywhere["fired"]) == (2, 3)  # row 2 is B1 first, B4 also
    assert (first["false"], anywhere["false"]) == (1, 2)
    c1 = ev._table_row(users, "C1", "whole", "any")
    assert (c1["fired"], c1["false"], c1["correct"]) == (3, 1, 2)  # row 17 is B2 first, C1 also fired
    assert ev._table_row(users, "C1", "whole", "first")["fired"] == 2
    tone = ev._table_row(users, "B5", "step", "any")
    assert (tone["fired"], tone["false"], tone["correct"], tone["vacuous"]) == (3, 1, 1, 1)
    tuner = ev._table_row(users, "B6", "step", "any")
    assert (tuner["fired"], tuner["false"], tuner["correct"], tuner["unjudgeable"]) == (3, 1, 1, 1)


def test_every_status_is_reachable_and_attributed():
    users = _users()
    b2 = ev._table_row(users, "B2", "whole", "first")
    assert {s: b2[s] for s in ev.STATUSES} == {"false": 2, "correct": 1, "partial": 1, "definition": 1, "vacuous": 0,
                                                "unjudgeable": 0}
    assert b2["correct_by_clause"] == {"tone_error_db": 1}
    assert ev._table_row(users, "B4", "whole", "first")["correct_by_clause"] == {"overload": 1}
    assert ev._table_row(users, "B3", "whole", "first")["unjudgeable"] == 1
    assert ev._table_row(users, "B5", "whole", "first")["vacuous"] == 1
    lower, upper = (b2[f"false_share_{bound}_bound"][0] for bound in ("lower", "upper"))
    assert (lower, upper) == (pytest.approx(2 / 54), pytest.approx(3 / 54))


def test_a_retry_decides_nothing_and_c_needs_a_decision():
    rows = ev.fabricated_rows()
    retry = rows[21]
    assert retry["verification"]["decision"] == "retry" and retry["fired"] == []
    ev.validate_rows(rows)
    broken = deepcopy(rows)
    broken[21]["fired"] = ["C1"]
    with pytest.raises(AssertionError, match="never a retry"):
        ev.validate_rows(broken)
    assert ev.refusal_report(rows)["retake_costs"] == {
        "users_with_a_retake": 2, "retakes": {"D1": 4}, "ended_without_a_verdict": 1}


def test_the_arithmetic_checks_raise_on_a_corrupted_row():
    rows = ev.fabricated_rows()
    for corrupt, message in (
        (lambda r: r.append(deepcopy(r[0])), "twice"),
        (lambda r: r[0].update(fired=["B9"]), "unknown veto"),
        (lambda r: r[53].update(fired=["B3"]), "analysis failure"),
        (lambda r: r[16].update(verification={"decision": "accept"}), "never a retry"),
    ):
        broken = deepcopy(rows)
        corrupt(broken)
        with pytest.raises(AssertionError, match=message):
            ev.assess(broken)


def test_the_counting_arithmetic_raises_when_a_count_does_not_add_up(monkeypatch):
    users = _users()
    monkeypatch.setattr(ev, "outcome", lambda *args, **kwargs: {"status": "weird"})
    with pytest.raises(AssertionError, match="status is unknown"):
        ev.population(users)
    monkeypatch.undo()
    monkeypatch.setattr(ev, "_outcomes", lambda users, veto, track, counting, **kw: [(0, {"status": "false"})] * (2 if counting == "first" else 1))
    with pytest.raises(AssertionError, match="more first than any"):
        ev.population(users)


# ---------------------------------------------------------------------------------------------------------------
# 7. The tuner's withheld step and the verification flow


def _candidate(strength, gate, score, passes, **gates):
    return {"noise_model": "rnnoise", "suppressor_strength": strength, "gate_settings": gate, "score": score,
            "passes": passes, "gates": gates or {"speech": passes}}


def test_the_tuner_step_is_its_own_best_non_incumbent_candidate():
    from mic_eq.config import Preset

    gate = asdict(Preset().gate)
    key = voice_setup_eval._arm_key("rnnoise", 1.0, gate)
    incumbent = _candidate(1.0, gate, 0.1, True)
    slower = _candidate(0.8, {**gate, "release_ms": 300.0}, 0.3, True)
    better = _candidate(0.7, {**gate, "release_ms": 50.0}, 0.2, True)
    failing = _candidate(0.5, {**gate, "release_ms": 20.0}, 0.05, False, speech=False, noise=True)
    chosen = ev.tuner_withheld([incumbent, slower, better, failing], key, {"holdout_evaluated": False})
    assert chosen and chosen["score"] == 0.2 and chosen["cause"] == "margin" and chosen["tested"] == 3
    held = ev.tuner_withheld([incumbent, better], key, {"holdout_evaluated": True})
    assert held and held["cause"] == "holdout"
    none_pass = ev.tuner_withheld([incumbent, failing], key, {})
    assert none_pass and none_pass["score"] == 0.05 and none_pass["failed_gates"] == ["speech"]
    assert none_pass["cause"] == "no_candidate_passed"
    assert ev.tuner_withheld([incumbent], key, {}) is None


def _flow(monkeypatch, outcomes):
    """Run ``verify_flow`` on scripted outcomes; return the flow and what the validator was given each time."""
    from mic_eq.config import EQ_FREQUENCIES

    steps, heard = iter(outcomes), []

    def validate(noise, original, verification, rate, state, preset, **kwargs):
        heard.append((original, verification, kwargs["raw_verification_speech_audio"]))
        return next(steps)

    monkeypatch.setattr("mic_eq.analysis.voice_setup.validate_voice_setup_verification", validate)
    settings = ev.current_settings(True)
    result = {**{name: settings[name] for name in ev.SECTIONS}, "diagnostics": settings["diagnostics"],
              "eq_settings": {"band_freqs": list(EQ_FREQUENCIES), "band_gains": [0.0] * 10, "band_qs": [1.41] * 10}}
    zero = np.zeros(10, np.float32)
    takes = [(np.full(10, 1.0 + i, np.float32), np.full(10, 10.0 * (1 + i), np.float32)) for i in range(3)]
    flow = ev.verify_flow(result, {"noise": zero, "setup": zero, "raw_noise": zero, "takes": takes})
    return flow, heard, takes


def _reduce(p95, targets=("compressor",)):
    return {"decision": "reduce", "reduction_targets": list(targets), "reason_codes": ["compressor_excess"],
            "compressor_gain_reduction_p95_db": p95}


ACCEPT = {"decision": "accept", "reason_codes": ["accepted"]}
RETRY = {"decision": "retry", "reason_codes": ["verification_delivery_mismatch"]}


@pytest.mark.parametrize("outcomes, decision, adjustments, ended, takes_heard, retakes", [
    ([ACCEPT], "accept", 0, None, [0], []),
    ([_reduce(5.0), ACCEPT], "accept", 1, None, [0, 0], []),
    ([_reduce(5.0), _reduce(4.0), _reduce(3.0), _reduce(2.0)], "gave_up", 3, "three_adjustments", [0] * 4, []),
    ([_reduce(5.0), _reduce(5.0)], "gave_up", 1, "no_help", [0, 0], []),
    ([_reduce(5.0, ("limiter",))], "gave_up", 0, "no_change", [0], []),  # no automatic makeup to lower
    ([_reduce(5.0, ())], "gave_up", 0, "no_target", [0], []),
    ([{"decision": "rollback", "reason_codes": ["target_error"]}], "rollback", 0, None, [0], []),
    ([RETRY, ACCEPT], "accept", 0, None, [0, 1], ["D1"]),
    ([{"decision": "retry", "reason_codes": ["verification_capture_invalid"]}, ACCEPT], "accept", 0, None, [0, 1], ["D2"]),
    ([{"decision": "retry", "reason_codes": ["verification_capture_too_short"]}, ACCEPT], "accept", 0, None, [0, 1], ["D2"]),
    ([{"decision": "retry", "reason_codes": ["no_such_code"]}, ACCEPT], "accept", 0, None, [0, 1], ["unknown"]),
    ([RETRY, RETRY, {"decision": "rollback", "reason_codes": ["target_error"]}], "rollback", 0, None, [0, 1, 2], ["D1", "D1"]),
    ([RETRY, _reduce(5.0), ACCEPT], "accept", 1, None, [0, 1, 1], ["D1"]),  # a reduction re-verifies the take it settled on
    ([RETRY, RETRY, RETRY], "retry", 0, "bad_takes", [0, 1, 2], ["D1", "D1", "D1"]),
])
def test_the_verification_flow_mirrors_the_dialog(monkeypatch, outcomes, decision, adjustments, ended, takes_heard, retakes):
    flow, heard, takes = _flow(monkeypatch, outcomes)
    assert (flow["decision"], flow["adjustments"], flow["ended"]) == (decision, adjustments, ended)
    assert flow["first_decision"] == outcomes[0]["decision"] and flow["retakes"] == retakes
    assert flow["takes"] == takes_heard[-1] + 1
    # Every call hears a recorded take (never the setup passage, which is all zeros here), the next one after a retake.
    for call, index in zip(heard, takes_heard, strict=True):
        assert np.array_equal(call[1], takes[index][0]) and np.array_equal(call[2], takes[index][1])
        assert not np.array_equal(call[1], call[0])
    if adjustments:
        assert flow["state"]["compressor_settings"]["threshold_db"] == pytest.approx(-20.0 + 2.0 * adjustments)
    assert ev._verification_row(flow)["retakes"] == retakes


# ---------------------------------------------------------------------------------------------------------------
# 8. The baseline world to the report


def _walk_floats(value: Any, path: str = "$") -> list[tuple[str, float]]:
    if isinstance(value, float):
        return [(path, value)]
    if isinstance(value, dict):
        return [hit for key, item in value.items() for hit in _walk_floats(item, f"{path}.{key}")]
    if isinstance(value, (list, tuple)):
        return [hit for i, item in enumerate(value) for hit in _walk_floats(item, f"{path}[{i}]")]
    return []


def test_the_baseline_world_gives_exact_counts_and_finite_intervals():
    report = ev.refusal_report(ev.fabricated_rows())
    assert (report["users"], report["speakers"]) == (54, 18)
    by = {(r["veto"], r["track"], r["counting"]): r for r in report["vetoes"]}
    assert by[("B1", "whole", "first")]["kind"] == "hard" and by[("B3", "whole", "first")]["kind"] == "soft"
    assert by[("B6", "step", "any")]["kind"] == "step"
    assert report["per_step_breakdown"]["refused_users"] == 7
    assert report["per_step_breakdown"]["no_single_step_worse"] == 0
    assert report["per_step_breakdown"]["steps"]["tone"]["worse_by_clause"] == {"tone_error_db": 7}
    assert report["tuner_step"] == {"fired": 3, "no_candidate": 1, "cause": {"margin": 1, "holdout": 1},
                                    "any_candidate_no_worse_upper_bound": 3}
    assert report["verification_decisions"] == {"accept": 44, "rollback": 3, "gave_up": 2, "retry": 1}
    assert report["sensitivity"]["last_10_s_for_c1_c2"] == {"C1": [1, 1], "C2": [1, 1]}
    assert report["candidates_for_loosening"] == []
    assert report["retake_costs"] == {"users_with_a_retake": 2, "retakes": {"D1": 4}, "ended_without_a_verdict": 1}
    # Nothing fires 8 users here, so every row is underpowered, and the report says so beside its trigger.
    assert len(report["underpowered"]) == len(report["vetoes"]) == 16 and all(r["underpowered"] for r in report["vetoes"])
    assert "underpowered" in report["trigger"] and "counterfactual" in report["b1"]
    for path, value in _walk_floats(report):
        assert math.isfinite(value), path
    for row in report["vetoes"]:
        mean, low, high = row["false_share_lower_bound"]
        assert 0.0 <= low <= mean <= high <= 1.0
        assert row["false_share_upper_bound"][0] >= mean
    assert {name: sum(group["users"] for group in groups.values()) for name, groups in report["strata"].items()} == {
        "level_check": 54, "rt60": 54, "snr": 54, "gender": 54}


def test_the_trigger_is_a_lower_bound_above_five_percent():
    rows = ev.fabricated_rows()
    for row in rows[23:41]:  # 18 users in 6 speakers whose weak-capture refusal withheld a setup that was no worse
        row.update(fired=["B3"], recommended=False, objects={"whole": {"vacuous": False}},
                   metrics={"current": {"whole": OK}, "whole": {"whole": OK}})
    report = ev.refusal_report(rows)
    assert report["candidates_for_loosening"] == ["B3:whole"]
    b3 = next(r for r in report["vetoes"] if (r["veto"], r["counting"]) == ("B3", "first"))
    assert b3["false_share_lower_bound"][1] > 0.05 and b3["candidate_for_loosening"]
    assert b3["fired"] >= ev.MIN_FIRED and not b3["underpowered"]
    assert "B3:whole:first" not in report["underpowered"] and "B1:whole:first" in report["underpowered"]


def test_the_report_is_clean_small_and_states_its_rule(frozen_tree, tmp_path):
    source = {"rows": ev.fabricated_rows(), "identity": {"build": ev.build_of(frozen_tree)}}
    path = tmp_path / "report.json"
    ev._write(path, ev.build_report(source, frozen_tree))
    assert check_evaluation_hygiene.validate_report(path) == []
    assert path.stat().st_size < 100_000
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["audible_change"] is False and report["schema_version"] == 2
    assert report["user_indices"] == {"fit": sorted(ev.STUDY_INDICES)}
    assert report["source_sha256"] == {name: frozen_tree[name] for name in ev.SOURCES}
    assert report["frozen_sha256"] == frozen_tree and report["rule_sha256"] == frozen_tree["rule"]
    assert "tuner" in " ".join(report["flattering"]) and len(report["limitations"]) >= 5
    assert any("counterfactual" in line for line in report["limitations"]) and any("margin" in line for line in report["limitations"])
    assert report["predefined_rule"].endswith("evaluate_voice_setup_refusals.py.")
    assert "Rule (fixed before any study row existed)" in (ev.__doc__ or "")
    assert "counterfactual" in (ev.__doc__ or "") and "underpowered" in (ev.__doc__ or "")


def test_the_report_takes_its_hashes_from_the_freeze_and_hashes_nothing_itself(frozen_tree, monkeypatch):
    def boom(*_args):
        raise AssertionError("hashed at report time")

    monkeypatch.setattr(loudness, "_sha256", boom)
    monkeypatch.setattr(gate_corpus, "_sha256", boom)
    report = ev.build_report({"rows": ev.fabricated_rows(), "identity": {"build": "b"}}, frozen_tree)
    assert report["implementation_sha256"] == {loudness.EXTENSION: frozen_tree["extension"]}
    assert report["asset_hashes"]["models/DeepFilterNet3_onnx.tar.gz"] == frozen_tree["models/DeepFilterNet3_onnx.tar.gz"]


def test_the_unmeasurable_clauses_are_flagged_over_fifteen_percent():
    users = _users()
    coverage = ev.clause_coverage(users)
    assert coverage["unmeasurable"]["deep_pause_db"] == 1 and coverage["over_15_percent"] == []
    rows = ev.fabricated_rows()
    for row in rows:
        for entry in row["metrics"].values():
            for window in entry.values() if "error" not in entry else ():
                if isinstance(window, dict) and "estoi" in window:
                    window["estoi"] = None
    assert "estoi" in ev.clause_coverage(_users(rows))["over_15_percent"]


# ---------------------------------------------------------------------------------------------------------------
# 9. Guards, the freeze, the dry run and the commands


def test_the_study_indices_are_unspent_and_map_to_the_intended_18_speakers():
    assert len(ev.STUDY_INDICES) == 54 and min(ev.STUDY_INDICES) >= ev.SPENT_BELOW
    assert {i % 30 for i in ev.STUDY_INDICES} == set(range(12, 30))
    manifest = {"speakers": {f"p{n:03d}": {"gender": "female" if n % 3 else "male"} for n in range(88)},
                "noise": {f"E{n:02d}": {} for n in range(18)}}
    fit = voice_setup_eval.splits(manifest)["fit"]["speakers"]
    chosen = [fit[i % len(fit)] for i in ev.STUDY_INDICES]
    assert len(fit) == 30 and set(chosen) == set(fit[12:]) and all(chosen.count(s) == 3 for s in set(chosen))
    assert ev.parse_indices(ev.DEFAULT_INDICES) == ("fit", list(ev.STUDY_INDICES))


def test_run_reaches_only_the_study_users_and_probe_only_the_development_ones():
    ev.check_indices("fit", [72, 89, 102, 119, 132, 149], development=False)
    for split, indices in (("fit", [59]), ("test", [0]), ("fit", [0])):
        with pytest.raises(SystemExit, match="spent"):
            ev.check_indices(split, indices, development=False)
    for split, indices in (("fit", [60]), ("fit", [90]), ("fit", [101]), ("fit", [150]), ("test", [100]), ("calibrate", [72])):
        with pytest.raises(SystemExit, match="only the study users"):
            ev.check_indices(split, indices, development=False)
    ev.check_indices("calibrate", [0, 11], development=True)
    for split, indices in (("fit", [12]), ("fit", [72]), ("test", [-1])):
        with pytest.raises(SystemExit, match="development users"):
            ev.check_indices(split, indices, development=True)
    with pytest.raises(SystemExit, match="unknown split"):
        ev.check_indices("train", [0], development=True)
    with pytest.raises(SystemExit, match="spent"):
        ev.main(["run", "--indices", "fit:0-3"])
    with pytest.raises(SystemExit, match="--development"):
        ev.main(["probe"])


def _rows_file(tmp_path, frozen, *, development=False, build=None):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"development": development, "identity": {"build": build or ev.build_of(frozen)},
                                "rows": ev.fabricated_rows()}), encoding="utf-8")
    return path


def test_report_writes_the_report_from_the_freeze(frozen_tree, tmp_path):
    out = tmp_path / "out.json"
    rows = _rows_file(tmp_path, frozen_tree)
    assert ev.main(["report", "--rows", str(rows), "--output", str(out), "--root", str(tmp_path)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["metrics"]["users"] == 54 and report["frozen_sha256"] == frozen_tree
    assert report["build"] == ev.build_of(frozen_tree)


def test_report_refuses_development_rows_rows_of_another_build_and_a_changed_tree(frozen_tree, tmp_path):
    rows = ev.fabricated_rows()
    rows[0]["development"] = True
    with pytest.raises(AssertionError, match="spent users"):
        ev.refusal_report(rows)
    with pytest.raises(AssertionError, match="54 study users"):
        ev.refusal_report(ev.fabricated_rows()[:53])
    out = tmp_path / "out.json"

    def report(path):
        return ev.main(["report", "--rows", str(path), "--output", str(out), "--root", str(tmp_path)])

    with pytest.raises(SystemExit, match="development"):
        report(_rows_file(tmp_path, frozen_tree, development=True))
    with pytest.raises(SystemExit, match="another build"):
        report(_rows_file(tmp_path, frozen_tree, build="other"))
    ev.FROZEN.write_text(json.dumps({**frozen_tree, "python/mic_eq/config_parts/presets.py": "0"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="presets.py"):
        report(_rows_file(tmp_path, frozen_tree))
    assert not out.exists()


def test_defaults_are_the_new_files_never_a_historical_report():
    assert ev.DEFAULT_REPORT == ev.REPO_ROOT / "evaluation" / "voice-setup-refusals-2026-10-report.json"
    details = ev.REPO_ROOT / "models" / "evaluation-details" / "voice-setup-refusals-2026-10"
    assert details in ev.DEFAULT_ROWS.parents and details in ev.DEFAULT_BOOKKEEPING.parents
    assert details in ev.FROZEN.parents and details in ev.DEVELOPMENT_ROWS.parents


JOBS = [("root", "fit", 72, "rnnoise")]


def test_workers_are_at_most_four_and_a_partial_file_resumes_only_its_own_build_and_users(tmp_path):
    assert [ev.worker_count(n) for n in (9, 4, 2, 0, -1)] == [4, 4, 2, 1, 1]
    partial = tmp_path / "rows.run.partial.jsonl"
    ev.check_resume(partial, "build-a", JOBS)  # nothing to resume
    partial.write_text(json.dumps({"split": "fit", "index": 72, "model": "rnnoise", "build": "build-a"}) + "\n", encoding="utf-8")
    ev.check_resume(partial, "build-a", JOBS)
    with pytest.raises(SystemExit, match="another build"):
        ev.check_resume(partial, "build-b", JOBS)
    # Finishing a run unlinks the partial file, so it must not hold users that run does not return.
    with pytest.raises(SystemExit, match="outside this run"):
        ev.check_resume(partial, "build-a", [("root", "fit", 73, "rnnoise")])
    assert partial.exists()


def test_a_truncated_last_line_of_the_partial_file_is_dropped_with_a_message_not_a_traceback(tmp_path, capsys):
    good = json.dumps({"split": "fit", "index": 72, "model": "rnnoise", "build": "b"})
    partial = tmp_path / "rows.run.partial.jsonl"
    partial.write_text(good + "\n" + good + "\n" + good[:25], encoding="utf-8")
    ev.check_resume(partial, "b", JOBS)
    assert "truncated last line (3)" in capsys.readouterr().out
    assert partial.read_text(encoding="utf-8") == good + "\n" + good + "\n"
    partial.write_text(good, encoding="utf-8")  # complete, but the next row would be appended to it
    ev.check_resume(partial, "b", JOBS)
    assert partial.read_text(encoding="utf-8") == good + "\n"
    partial.write_text("{damaged\n" + good + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="line 1 is not JSON"):
        ev.check_resume(partial, "b", JOBS)


def test_the_freeze_covers_every_module_the_study_loads_and_notices_a_change(frozen_tree, tmp_path):
    product = {path.relative_to(ev.REPO_ROOT).as_posix() for path in (ev.REPO_ROOT / "python" / "mic_eq").rglob("*.py")}
    assert len(product) > 50 and product <= set(frozen_tree) and set(ev.SOURCES) <= set(frozen_tree)
    for name in ("python/mic_eq/config_parts/presets.py", "python/mic_eq/analysis/auto_eq_parts/eq_fit.py",
                 "python/mic_eq/analysis/auto_eq_parts/target.py", "python/mic_eq/analysis/vad.py",
                 "python/mic_eq/analysis/deesser_fusion.py", "python/mic_eq/analysis/eq_quality.py",
                 "python/tools/evaluate_voice_setup_refusals.py"):
        assert name in frozen_tree, name
    assert {"rule", "extension", "df.dll", "onnxruntime.dll", "onnxruntime_providers_shared.dll", "vad_model",
            "corpus_manifest", "models/DeepFilterNet3_onnx.tar.gz", "models/DeepFilterNet3_ll_onnx.tar.gz",
            "release_assets:df.dll", "release_assets:onnxruntime.dll", "release_assets:vad_model",
            "runtime:python", "runtime:numpy", "runtime:scipy"} <= set(frozen_tree)
    assert ev.check_frozen(ev.FROZEN, tmp_path) == frozen_tree
    for name in ("python/mic_eq/config_parts/presets.py", "df.dll", "models/DeepFilterNet3_onnx.tar.gz", "rule", "runtime:numpy"):
        ev.FROZEN.write_text(json.dumps({**frozen_tree, name: "changed"}), encoding="utf-8")
        with pytest.raises(SystemExit, match=name.replace(".", r"\.")):
            ev.check_frozen(ev.FROZEN, tmp_path)
    ev.FROZEN.write_text(json.dumps({**frozen_tree, "gone.py": "x"}), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"gone\.py"):
        ev.check_frozen(ev.FROZEN, tmp_path)
    ev.FROZEN.unlink()
    with pytest.raises(SystemExit, match="freeze"):
        ev.check_frozen(ev.FROZEN, tmp_path)


def test_a_module_added_after_the_freeze_is_a_change(frozen_tree, monkeypatch, tmp_path):
    real = ev.source_hashes
    monkeypatch.setattr(ev, "source_hashes", lambda: {**real(), "python/mic_eq/added_later.py": "x"})
    with pytest.raises(SystemExit, match=r"added_later\.py"):
        ev.check_frozen(ev.FROZEN, tmp_path)


def _tree(tmp_path):
    """A stand-in repository holding every file the study loads, and the digest of each."""
    repo = tmp_path / "repo"
    contents = {"df.dll": b"df", "models/DeepFilterNet3_ll_onnx.tar.gz": b"ll", "models/DeepFilterNet3_onnx.tar.gz": b"std",
                "models/silero_vad.onnx": b"vad", loudness.EXTENSION: b"pyd", "ort/onnxruntime.dll": b"ort",
                "ort/onnxruntime_providers_shared.dll": b"providers", "corpus/manifest.json": b"{}"}
    for name, data in contents.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_bytes(data)
    digest = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    return repo, digest


def test_the_assets_are_exactly_the_files_the_study_loads(monkeypatch, tmp_path):
    repo, digest = _tree(tmp_path)
    monkeypatch.setattr(ev, "REPO_ROOT", repo)
    monkeypatch.setattr(ev, "ort_dll_root", lambda: repo / "ort")
    monkeypatch.setenv("VAD_MODEL_PATH", str(repo / "models" / "silero_vad.onnx"))
    assert ev.asset_hashes(repo / "corpus") == {
        "extension": digest[loudness.EXTENSION], "df.dll": digest["df.dll"], "onnxruntime.dll": digest["ort/onnxruntime.dll"],
        "onnxruntime_providers_shared.dll": digest["ort/onnxruntime_providers_shared.dll"],
        "vad_model": digest["models/silero_vad.onnx"], "corpus_manifest": digest["corpus/manifest.json"],
        "models/DeepFilterNet3_ll_onnx.tar.gz": digest["models/DeepFilterNet3_ll_onnx.tar.gz"],
        "models/DeepFilterNet3_onnx.tar.gz": digest["models/DeepFilterNet3_onnx.tar.gz"]}
    ev.check_assets()
    # An absent DeepFilter archive stops the check, the hashing and the freeze: none leaves it out.
    (repo / "models" / "DeepFilterNet3_ll_onnx.tar.gz").unlink()
    with pytest.raises(FileNotFoundError, match="DeepFilterNet3_ll_onnx"):
        ev.check_assets()
    with pytest.raises(FileNotFoundError):
        ev.asset_hashes(repo / "corpus")
    with pytest.raises(FileNotFoundError, match="DeepFilterNet3_ll_onnx"):
        ev.freeze(tmp_path / "frozen.json", tmp_path / "rules.md", tmp_path / "rows.json", repo / "corpus")
    assert not (tmp_path / "frozen.json").exists()
    (repo / "ort" / "onnxruntime.dll").unlink()
    with pytest.raises(FileNotFoundError, match="onnxruntime.dll"):
        ev.check_assets()


def test_release_assets_declares_what_the_study_loads_and_the_report_records_both_hashes(frozen_tree):
    listed = {a["path"]: a["sha256"] for a in json.loads((ev.REPO_ROOT / "release-assets.json").read_text(encoding="utf-8"))["assets"]}
    declared = ev.declared_hashes()
    assert set(declared) == {f"release_assets:{name}" for name in (
        "df.dll", "onnxruntime.dll", "onnxruntime_providers_shared.dll", "vad_model",
        "models/DeepFilterNet3_ll_onnx.tar.gz", "models/DeepFilterNet3_onnx.tar.gz")}
    assert declared["release_assets:df.dll"] == listed["df.dll"] and declared["release_assets:vad_model"] == listed["models/silero_vad.onnx"]
    assert all(frozen_tree[key] == value for key, value in declared.items())
    source = {"rows": ev.fabricated_rows(), "identity": {"build": "b"}}
    report = ev.build_report(source, frozen_tree)
    assert report["release_assets"]["df.dll"] == {"local": "2" * 64, "declared": declared["release_assets:df.dll"], "same": False}
    assert report["asset_hashes"]["onnxruntime_dll"] == "6" * 64 and report["asset_hashes"]["df_dll"] == "2" * 64
    assert any(line.startswith("df.dll is a local build") for line in report["limitations"])
    matching = {**frozen_tree, "df.dll": declared["release_assets:df.dll"]}
    assert ev.build_report(source, matching)["release_assets"]["df.dll"]["same"] is True
    assert not any(line.startswith("df.dll is a local build") for line in ev.build_report(source, matching)["limitations"])


def test_a_freeze_stays_once_rows_exist(frozen_tree, tmp_path):
    ev.freeze(ev.FROZEN, ev.RULES_FINAL, ev.DEFAULT_ROWS, tmp_path)  # nothing was made under it yet
    assert ev.RULES_FINAL.read_text(encoding="utf-8") == ev.__doc__
    for existing in (ev.DEFAULT_ROWS, ev.DEFAULT_ROWS.with_suffix(".run.partial.jsonl")):
        existing.write_text("{}", encoding="utf-8")
        with pytest.raises(SystemExit, match="stays"):
            ev.freeze(ev.FROZEN, ev.RULES_FINAL, ev.DEFAULT_ROWS, tmp_path)
        with pytest.raises(SystemExit, match="stays"):
            ev.main(["freeze", "--root", str(tmp_path)])
        existing.unlink()


def test_run_refuses_to_overwrite_a_finished_rows_file_wherever_output_points(frozen_tree, tmp_path):
    other = tmp_path / "elsewhere.json"
    with pytest.raises(SystemExit, match="always"):
        ev.main(["run", "--root", str(tmp_path), "--output", str(other)])
    ev.DEFAULT_ROWS.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="finished run"):
        ev.main(["run", "--root", str(tmp_path)])
    with pytest.raises(SystemExit, match="always"):
        ev.main(["run", "--root", str(tmp_path), "--output", str(other)])
    with pytest.raises(SystemExit, match="stays"):  # the freeze lock follows the study's rows, not --rows
        ev.main(["freeze", "--root", str(tmp_path), "--rows", str(other)])
    ev.check_not_finished(tmp_path / "none.json")


def test_the_build_comes_from_every_frozen_hash_and_a_change_during_a_run_is_caught(monkeypatch, tmp_path):
    frozen = {"rule": "1", "python/mic_eq/x.py": "2"}
    monkeypatch.setattr(ev, "frozen_hashes", lambda root=None: dict(frozen))
    ev.build_id.cache_clear()
    started = ev.build_id("x")
    assert started == ev.build_of(frozen)
    ev.check_unchanged(tmp_path, started, [{"build": started}])
    with pytest.raises(SystemExit, match="another build|not one build"):
        ev.check_unchanged(tmp_path, started, [{"build": "other"}])
    frozen["python/mic_eq/x.py"] = "3"
    with pytest.raises(SystemExit, match="changed during the run"):
        ev.check_unchanged(tmp_path, started, [{"build": started}])
    ev.build_id.cache_clear()
    assert ev.build_id("x") != started
    ev.build_id.cache_clear()


def test_the_capture_vetoes_a1_to_a4_fire_on_hand_made_arrays():
    assert ev.capture_vetoes_probe() == {"A1": True, "A2": True, "A3": True, "A4": True}


def _development_rows() -> list[dict[str, Any]]:
    """The hand-made world as if it were development rows, each with the probes a real run would carry."""
    bad = {"tone_plus_12_db_3k": {"tone_error_db": 9.0}, "sibilance_minus_12_db_6k5": {"sibilant_loss_db": 12.0},
           "compressor_squash": {"estoi": 0.3}, "suppressor_off": {"deep_pause_db": 0.0}, "gate_threshold_minus_15": {"onset_db": -9.0},
           "gate_release_1000": {"tail_db": 0.0}, "makeup_plus_24": {"clipped": True}}
    rows = ev.fabricated_rows()
    for row in rows:
        row["development"] = True
        row["probes"] = {name: {"whole": _with(**changes)} for name, changes in bad.items()} | {
            "current_again": {"whole": dict(OK)}, "limiter_off": {"whole": dict(OK)}}
        row["tone_cross_check_gap_db"] = 0.2
    return rows


def test_the_dry_run_output_has_counts_and_flags_but_no_rates():
    rows = _development_rows()
    result = ev.bookkeeping(rows)
    assert _walk_floats(result) == []
    text = json.dumps(result).lower()
    for word in ("rate", "share", "interval", "bootstrap", "ci\"", "bucket"):
        assert word not in text
    assert result["development_users"] == 54 and result["arithmetic"] == "checked"
    assert result["constructed_capture_probes"] == {"A1": True, "A2": True, "A3": True, "A4": True}
    # Reachability says whether a veto fired and what became of it, never how many users it fired for.
    assert result["reachability"]["B1"] == {"fired": True, "judged": True, "vacuous": False, "unjudgeable": False}
    assert result["reachability"]["B5_step"]["vacuous"] is True and result["reachability"]["B6_step"]["unjudgeable"] is True
    assert all(isinstance(flag, bool) for entry in result["reachability"].values() for flag in entry.values())
    assert result["never_fired"] == []
    assert result["tone_cross_check"] == {"users": 54, "within_half_db": 54}
    assert set(result["probes"]) == set(ev.probe_settings(ev.current_settings(True)))
    for name, probe in result["probes"].items():
        if "clause" in probe:
            assert probe["tripped"] == probe["measurable"] == 53 and probe["trips_on_most"] and not probe["vacuous"], name
    assert {probe["clause"] for probe in result["probes"].values() if "clause" in probe} >= {"estoi", "overload", "onset_db"}
    assert result["probes"]["current_again"] == {"users": 53, "identical_to_current": 53, "clause_failures": {}}
    with pytest.raises(AssertionError, match="spent users"):
        ev.refusal_report(rows)


def test_a_probe_that_never_trips_or_trips_for_the_same_settings_again_is_vacuous_and_limiter_off_is_not_the_judge():
    rows = _development_rows()
    for row in rows:
        row["probes"]["suppressor_off"] = {"whole": dict(OK)}  # never trips
        row["probes"]["current_again"] = {"whole": _with(tail_db=0.0)}  # the same settings differ: tail cannot be trusted
        row["probes"]["limiter_off"] = {"whole": _with(clipped=True)}  # without the limiter it may legitimately clip
    probes = ev.bookkeeping(rows)["probes"]
    assert probes["suppressor_off"]["tripped"] == 0 and probes["suppressor_off"]["vacuous"]
    assert probes["gate_release_1000"]["vacuous"] is True and probes["tone_plus_12_db_3k"]["vacuous"] is False
    assert probes["makeup_plus_24"]["vacuous"] is False  # limiter_off clipping must not make overload vacuous
    assert probes["limiter_off"]["clause_failures"] == {"overload": 53}


def test_the_fabricated_world_is_the_study_users_and_covers_every_veto():
    rows = ev.fabricated_rows()
    assert [row["index"] for row in rows] == list(ev.STUDY_INDICES)
    assert {veto for row in rows for veto in row["fired"]} == set(ev.VETOES)
    assert not any(row["development"] for row in rows)


# ---------------------------------------------------------------------------------------------------------------
# 10. One user through the worker, with the analysis and the renders stubbed


def _stub_worker(monkeypatch, tmp_path, *, b1, heard=None, verdicts=None, vad="silero", deepfilter="available"):
    """Stub the analysis, the native renders and the corpus; return the job for ``evaluate_user``.

    ``heard`` collects the simulated user and what the verification validator was given. ``verdicts`` are the
    validator's decisions in turn (the last one repeats), a rollback by default.
    """
    from mic_eq.analysis import failure_detection, voice_setup
    from mic_eq.analysis.auto_eq_parts import optimizer
    from mic_eq.config import EQ_FREQUENCIES, Preset

    rate = voice_setup_eval.SAMPLE_RATE
    case = _case(2)  # a 6 s evaluation capture; the setup passage is 3 s of other audio, so takes start at 0, 1.5, 3 s
    rng = np.random.default_rng(5)
    (tmp_path / "speech").mkdir(exist_ok=True)
    (tmp_path / "speech" / "a.wav").write_bytes(b"speech")
    (tmp_path / "noise.wav").write_bytes(b"noise")
    digest = lambda data: hashlib.sha256(data).hexdigest()  # noqa: E731
    (tmp_path / "manifest.json").write_text(json.dumps({
        "speakers": {"p001": {"gender": "male", "files": [{"path": "speech/a.wav", "sha256": digest(b"speech")}]}},
        "noise": {"DKITCHEN": {"path": "noise.wav", "sha256": digest(b"noise")}}}), encoding="utf-8")
    simulated = {
        "split": "fit", "index": 72, "speaker": "p001", "environment": "DKITCHEN", "snr_db": 12.0, "level_dbfs": -28.0,
        "noise_capture": (0.01 * rng.standard_normal(2 * rate)).astype(np.float32),
        "setup_capture": (0.05 * rng.standard_normal(3 * rate)).astype(np.float32),
        "evaluation_capture": case["noisy"].astype(np.float32), "evaluation_clean": case["clean"].astype(np.float32),
        "labels": case["labels"],
    }
    gate = asdict(Preset().gate)
    candidates = [
        {"noise_model": "rnnoise", "suppressor_strength": 1.0, "gate_settings": gate, "score": 0.1, "passes": True,
         "gates": {"speech": True, "runtime": True}, "metrics": {"runtime_factor": 0.4}},
        {"noise_model": "deepfilter-ll", "suppressor_strength": 0.8, "gate_settings": {**gate, "release_ms": 60.0},
         "score": 0.05, "passes": False, "gates": {"speech": False, "runtime": False}, "metrics": {"runtime_factor": 1.7}},
    ]
    current = ev.current_settings(True)
    decisions = iter(verdicts or [])
    last = {"decision": "rollback", "reason_codes": ["target_error"], "reasons": [], "reduction_targets": []}

    def analyze(*_args, **_kwargs):
        if b1:
            failure_detection.validate_analysis({"band_gains": [0.0] * 10}, np.zeros(2049), np.linspace(0.0, 24_000.0, 2049))
        cast(Any, optimizer)._validate_and_attenuate_solution()
        return {
            **{name: deepcopy(current[name]) for name in ev.SECTIONS},
            "compressor_settings": {**current["compressor_settings"], "threshold_db": -25.0},
            "deesser_settings": {**current["deesser_settings"], "auto_amount": 0.9},  # differs, but is off in both
            "eq_settings": {"band_freqs": list(EQ_FREQUENCIES), "band_gains": [0.0] * 10, "band_qs": [1.41] * 10,
                            "enabled": True}, "eq_error": None,
            "diagnostics": {
                "vad_available": True, "weak_capture": False, "apply_recommended": False, "vad_active_duration_s": 5.0,
                "uncertainty_reasons": ["Auto-EQ abstained from this capture"],
                "vad_analysis_backend": "silero", "noise_vad_analysis_backend": "silero",
                "joint_vad_analysis_backend": "silero", "joint_noise_vad_analysis_backend": vad,
                "compressor_auto_makeup_enabled": False, "deesser_enabled": False, "deesser_detection_probability": 0.0,
                "noise_reference_quality": {"status": "usable"},
                "joint_tuning": {
                    "decision": "retain_incumbent", "holdout_evaluated": False, "noise_floor_db": -60.0,
                    "candidates": candidates,
                    "model_evaluations": [{"model": model, "status": deepfilter if model == "deepfilter" else "available"}
                                          for model in ev.TUNER_MODELS]}},
        }

    def render(settings, capture):
        # The output depends on the settings the way a chain's would: the EQ gains and the suppressor model move the
        # level, the proposed compressor clips. The de-esser changes nothing.
        output = np.asarray(capture, np.float64)
        output = output * (1.0 + 0.01 * sum(settings["eq_settings"].get("band_gains", [])))
        output = output * (0.99 if settings["suppressor_settings"]["model"] != "rnnoise" else 1.0)
        if settings["compressor_settings"]["threshold_db"] != -20.0:
            output = np.clip(30.0 * output, -2.0, 2.0)
        return {"output_audio": output, "limiter_gain_reduction_db": 0.0, "true_peak_limiter_gain_reduction_db": 0.0}

    def verify(noise, original, verification, sample_rate, state, preset, **kwargs):
        if heard is not None:
            heard["calls"].append((original, verification, kwargs["raw_verification_speech_audio"]))
        return next(decisions, None) or last

    if heard is not None:
        heard.update(simulated=simulated, calls=[])
    monkeypatch.setattr(ev, "configure_environment", lambda: None)
    monkeypatch.setattr(ev, "build_id", lambda root: "test")
    monkeypatch.setattr(voice_setup_eval, "user", lambda *args: simulated)
    monkeypatch.setattr(voice_setup_eval, "render_arm", lambda capture, arm, floor: np.asarray(capture, np.float64))
    monkeypatch.setattr(voice_setup, "analyze_voice_setup", analyze)
    monkeypatch.setattr(optimizer, "_validate_and_attenuate_solution",
                        lambda *a, **k: (np.array([3.0, -2.0] + [0.0] * 8), 0.0, 0.0, 1.0, {}))
    monkeypatch.setattr(voice_setup, "validate_voice_setup_verification", verify)
    monkeypatch.setattr(loudness, "_render", render)
    return str(tmp_path), "fit", 72, "rnnoise"


def test_a_user_goes_through_the_worker_into_a_row_the_report_can_judge(monkeypatch, tmp_path):
    heard: dict[str, Any] = {}
    row = ev.evaluate_user(_stub_worker(monkeypatch, tmp_path, b1=False, heard=heard), development=False)
    json.dumps(row)  # plain JSON types only
    assert row["fired"] == ["B5", "B6", "C1"] and row["b1"] is None and row["recommended"] is False
    assert row["verification"]["decision"] == "rollback" and row["verification"]["takes"] == 1
    assert set(row["objects"]) >= {"whole", "final", "tone", "tuner"} | {f"step:{s}" for s in ev.STEPS}
    assert set(row["metrics"]["current"]) == {"whole", "last10"} and set(row["metrics"]["final"]) == {"whole", "last10"}
    assert row["tuner"]["withheld"]["noise_model"] == "deepfilter-ll" and row["tuner"]["withheld"]["cause"] == "no_candidate_passed"
    assert row["tuner"]["any_candidate_no_worse"] == {"tested": 1, "no_worse": 1}
    assert 0.2 <= row["rt60_s"] <= 0.7 and row["gender"] == "male" and row["level_check"] in ("good", "too_hot", "too_quiet")
    assert row["taps"] == {"validation": 0, "tone": 1} and row["objects"]["tone"]["digests"].keys() == {"eq_settings"}
    assert row["objects"]["tuner"]["digests"].keys() == {"gate_settings", "suppressor_settings"}
    assert set(row["vad_backends"].values()) == {"silero"} and row["tuner_models"] == dict.fromkeys(ev.TUNER_MODELS, "available")
    assert [c["failed_gates"] for c in row["tuner"]["candidates"]] == [[], ["runtime", "speech"]]
    assert [c["runtime_factor"] for c in row["tuner"]["candidates"]] == [0.4, 1.7]
    status = {name: ev.outcome(row, name)["status"] for name in ("whole", "final", "tone", "tuner")}
    assert status == {"whole": "correct", "final": "correct", "tone": "false", "tuner": "false"}
    assert "overload" in ev.outcome(row, "whole")["failed"]
    # The stub changed the compressor, and the de-esser's settings without changing a sample. The tone and the
    # gate/suppression steps have the current sections: a step that changes nothing is vacuous, by digest or by effect.
    assert [ev.outcome(row, f"step:{s}")["status"] for s in ev.STEPS] == ["vacuous", "vacuous", "vacuous", "correct"]
    assert row["objects"]["step:deesser"]["identical_render"] is True and "step:deesser" not in row["metrics"]
    assert "identical_render" not in row["objects"]["step:gate_suppression"]
    assert ev.assess([row])[0]["first"] == "B5"
    # Verification heard a fresh recorded take, not the setup passage.
    size = heard["simulated"]["setup_capture"].size
    original, verification, raw = heard["calls"][0]
    from mic_eq.ui.calibration_support import filtered_capture_for_analysis

    rate = voice_setup_eval.SAMPLE_RATE
    setup = filtered_capture_for_analysis(heard["simulated"]["setup_capture"], rate)
    take = heard["simulated"]["evaluation_capture"][:size]
    assert np.array_equal(original, setup) and not np.array_equal(verification, setup)
    assert np.array_equal(verification, filtered_capture_for_analysis(take, rate)) and np.array_equal(raw, take)


def test_a_delivery_mismatch_asks_for_the_next_window_and_the_user_is_still_judged(monkeypatch, tmp_path):
    from mic_eq.ui.calibration_support import filtered_capture_for_analysis

    heard: dict[str, Any] = {}
    retry = {"decision": "retry", "reason_codes": ["verification_delivery_mismatch"], "reasons": [], "reduction_targets": []}
    job = _stub_worker(monkeypatch, tmp_path, b1=False, heard=heard, verdicts=[retry])
    row = ev.evaluate_user(job, development=False)
    evaluation, size = heard["simulated"]["evaluation_capture"], heard["simulated"]["setup_capture"].size
    middle = (evaluation.size - size) // 2
    assert [np.array_equal(call[2], evaluation[start : start + size]) for call, start in zip(heard["calls"], (0, middle), strict=True)] == [True, True]
    assert np.array_equal(heard["calls"][1][1], filtered_capture_for_analysis(evaluation[middle : middle + size], voice_setup_eval.SAMPLE_RATE))
    assert row["verification"]["retakes"] == ["D1"] and row["verification"]["takes"] == 2
    assert row["verification"]["decision"] == "rollback" and "C1" in row["fired"]  # not censored
    heard = {}
    row = ev.evaluate_user(_stub_worker(monkeypatch, tmp_path, b1=False, heard=heard, verdicts=[retry] * 3), development=False)
    assert row["verification"]["decision"] == "retry" and row["verification"]["retakes"] == ["D1"] * 3
    # Three takes, the last at the end of the capture, none of them empty or short.
    starts = (0, middle, evaluation.size - size)
    assert len(heard["calls"]) == 3
    assert [np.array_equal(call[2], evaluation[start : start + size]) and call[2].size == size
            for call, start in zip(heard["calls"], starts, strict=True)] == [True, True, True]
    assert "C1" not in row["fired"] and "C2" not in row["fired"] and "final" not in row["objects"]


def test_an_unclear_recording_refusal_keeps_its_proposal_and_never_reaches_verification(monkeypatch, tmp_path):
    row = ev.evaluate_user(_stub_worker(monkeypatch, tmp_path, b1=True), development=False)
    assert row["fired"] == ["B1", "B5", "B6"] and row["verification"] is None and "final" not in row["objects"]
    assert row["b1"]["reason"] == "Recording too unclear. Please try again." and ev.assess([row])[0]["first"] == "B1"
    assert ev.outcome(row, "whole", veto="B1")["status"] == "correct"


def test_a_user_whose_auto_eq_failed_before_validation_is_an_analysis_failure(monkeypatch, tmp_path):
    from mic_eq.analysis import voice_setup

    job = _stub_worker(monkeypatch, tmp_path, b1=False)
    failing = {**cast(Any, voice_setup).analyze_voice_setup(), "eq_settings": None, "eq_error": "boom"}
    monkeypatch.setattr(voice_setup, "analyze_voice_setup", lambda *a, **k: failing)
    row = ev.evaluate_user(job, development=False)
    assert row["analysis_failure"] == "boom" and row["fired"] == [] and ev.assess([row])[0]["bucket"] == "analysis_failure"
    assert set(row["vad_backends"].values()) == {"silero"} and row["tuner_models"]["deepfilter"] == "available"
    refusal = ev.CAPTURE_REFUSALS[1]
    monkeypatch.setattr(voice_setup, "analyze_voice_setup", lambda *a, **k: (_ for _ in ()).throw(ValueError(refusal)))
    row = ev.evaluate_user(job, development=False)
    assert row["analysis_failure"] == refusal and row["vad_backends"] is None and row["tuner_models"] is None
    monkeypatch.setattr(voice_setup, "analyze_voice_setup", lambda *a, **k: (_ for _ in ()).throw(ValueError("a bug")))
    with pytest.raises(ValueError, match="a bug"):
        ev.evaluate_user(job, development=False)


def test_a_row_made_without_silero_or_without_a_tuner_model_stops_the_run(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match=r"VAD not Silero in \['joint_noise_vad_analysis_backend'\]"):
        ev.evaluate_user(_stub_worker(monkeypatch, tmp_path, b1=False, vad="energy_fallback"), development=False)
    with pytest.raises(RuntimeError, match=r"tuner models not available: \['deepfilter'\]"):
        ev.evaluate_user(_stub_worker(monkeypatch, tmp_path, b1=False, deepfilter="unavailable"), development=False)
    diagnostics = {key: "silero" for key in ev.VAD_BACKENDS}
    models = [{"model": model, "status": "available"} for model in ev.TUNER_MODELS]
    assert ev.check_loaded(diagnostics, {"model_evaluations": models})["tuner_models"] == dict.fromkeys(ev.TUNER_MODELS, "available")
    with pytest.raises(RuntimeError):
        ev.check_loaded(diagnostics, {})  # a tuner that reports no models was unavailable


# ---------------------------------------------------------------------------------------------------------------
# 11. The native renders of the current settings and the probes, on a short synthetic signal


def test_the_current_settings_and_every_probe_render_natively_and_the_known_good_ones_change_nothing(no_vad_model):
    from mic_eq.ui.calibration_support import filtered_capture_for_analysis

    rate = voice_setup_eval.SAMPLE_RATE
    case = _case(2, modulated=True)
    rng = np.random.default_rng(4)
    raw = case["noisy"].astype(np.float32)
    ctx = {"raw": raw, "labels": case["labels"], "clean": filtered_capture_for_analysis(case["clean"], rate).astype(np.float64),
           "noisy": filtered_capture_for_analysis(raw, rate).astype(np.float64),
           "noise": filtered_capture_for_analysis((0.01 * rng.standard_normal(2 * rate)).astype(np.float32), rate).astype(np.float64)}
    current = ev.current_settings(False)  # VAD explicitly unavailable: no model is needed
    base = ev.measure_settings(current, ctx, ("whole", "last10"), as_reference=True)
    assert set(base) == {"whole", "last10"} and base["whole"]["clipped"] is False
    judged = {}
    for name, (target, settings) in ev.probe_settings(current).items():
        reading = ev.measure_settings(settings, ctx)
        assert "error" not in reading, (name, reading)
        judged[name] = ev.judge(reading["whole"], base["whole"])
        if not target:
            assert judged[name]["failed"] == [] and ev._same_reading(reading["whole"]["estoi"], base["whole"]["estoi"]), name
        if name == "current_again":
            assert reading["identical"] is True  # the same settings again render the same audio
    assert "tone_error_db" in judged["tone_plus_12_db_3k"]["failed"]
    assert "estoi" in judged["compressor_squash"]["failed"]
    assert "overload" in judged["makeup_plus_24"]["failed"]


def test_the_taps_keep_a_refused_proposal_on_the_real_auto_eq(no_vad_model):
    from mic_eq.analysis.auto_eq import analyze_auto_eq

    rate = voice_setup_eval.SAMPLE_RATE
    flat = (0.05 * np.random.default_rng(1).standard_normal(6 * rate)).astype(np.float32)  # no voice structure
    with pytest.raises(ValueError, match="too unclear"):
        analyze_auto_eq(flat, rate, "broadcast")
    with ev._taps() as log:
        eq, _ = analyze_auto_eq(flat, rate, "broadcast")
    assert log["validation"][0]["passed"] is False and "flatness" in " ".join(log["validation"][0]["failures"])
    assert len(log["tone"]) == 1 and len(log["tone"][0]) == len(eq["band_gains"]) == 10


def test_the_verification_flow_runs_against_the_real_validator_and_retakes_a_mismatched_take(no_vad_model):
    from mic_eq.config import EQ_FREQUENCIES
    from mic_eq.ui.calibration_support import filtered_capture_for_analysis

    rate = voice_setup_eval.SAMPLE_RATE
    rng = np.random.default_rng(1)
    speech = np.concatenate([np.concatenate([np.zeros(rate // 2), 0.1 * rng.standard_normal(rate)]) for _ in range(3)])
    noise = (0.01 * rng.standard_normal(2 * rate)).astype(np.float32)
    take = (speech + 0.01 * rng.standard_normal(speech.size)).astype(np.float32)
    quiet = (0.05 * take).astype(np.float32)  # 26 dB under the setup passage: a delivery mismatch
    settings = ev.current_settings(False)
    result = {**{name: settings[name] for name in ev.SECTIONS}, "diagnostics": settings["diagnostics"],
              "eq_settings": {"band_freqs": list(EQ_FREQUENCIES), "band_gains": [0.0] * 10, "band_qs": [1.41] * 10}}
    f = lambda audio: filtered_capture_for_analysis(audio, rate)  # noqa: E731
    flow = ev.verify_flow(result, {"noise": f(noise), "setup": f(take), "raw_noise": noise,
                                   "takes": [(f(quiet), quiet), (f(take), take), (f(take), take)]})
    assert flow["retakes"] == ["D1"] and flow["takes"] == 2 and flow["first_decision"] == "retry"
    assert flow["decision"] in ("accept", "rollback", "gave_up") and flow["reason_codes"]
    assert "bands" in flow["state"]["eq_settings"]


def test_a_development_user_carries_the_probes_and_the_cross_check_and_gives_counts_only(monkeypatch, tmp_path):
    row = ev.evaluate_user(_stub_worker(monkeypatch, tmp_path, b1=False), development=True)
    assert row["development"] is True
    assert set(row["probes"]) == set(ev.probe_settings(ev.current_settings(True)))
    assert row["probes"]["current_again"]["identical"] is True and row["probes"]["limiter_off"]["identical"] is True
    assert math.isfinite(row["tone_cross_check_gap_db"]) and row["tone_cross_check_gap_db"] < 0.5
    result = ev.bookkeeping([row])
    assert _walk_floats(result) == [] and result["development_users"] == 1
    assert result["reachability"]["B5"]["fired"] is True and result["tone_cross_check"] == {"users": 1, "within_half_db": 1}
    assert result["probes"]["current_again"]["identical_to_current"] == 1


def test_a_missing_model_or_a_current_render_that_fails_stops_the_run_instead_of_filling_rows_with_errors(
        monkeypatch, tmp_path, no_vad_model):
    with pytest.raises(FileNotFoundError, match="silero_vad.onnx"):
        ev.check_assets()
    job = _stub_worker(monkeypatch, tmp_path, b1=False)

    def fail(*_args):
        raise RuntimeError("native simulator failed: full_chain VAD initialization failed: VAD model not found")

    monkeypatch.setattr(loudness, "_render", fail)
    with pytest.raises(RuntimeError, match="current settings do not render.*VAD model not found"):
        ev.evaluate_user(job, development=False)


def test_a_corpus_file_that_is_not_the_manifests_stops_the_run(monkeypatch, tmp_path):
    job = _stub_worker(monkeypatch, tmp_path, b1=False)
    ev.check_corpus(tmp_path, json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")), "p001", "DKITCHEN")
    (tmp_path / "noise.wav").write_bytes(b"replaced")
    with pytest.raises(RuntimeError, match=r"corpus files differ from the manifest: noise\.wav"):
        ev.evaluate_user(job, development=False)


def test_the_take_windows_lie_inside_the_capture():
    capture = np.arange(100.0)
    windows = ev.take_windows(capture, 30)
    assert [window[0] for window in windows] == [0.0, 35.0, 70.0] and all(window.size == 30 for window in windows)
    assert windows[2][-1] == 99.0
    with pytest.raises(ValueError, match="shorter than the setup passage"):
        ev.take_windows(capture, 101)


def test_the_last_window_starts_at_the_middle_of_the_capture(monkeypatch):
    case = _case(1)
    ctx = {**case, "noise": case["noisy"][: 2 * voice_setup_eval.SAMPLE_RATE], "raw": case["noisy"].astype(np.float32)}
    spike = {"at": 100}

    def render(settings, capture):
        output = np.asarray(capture, np.float32).copy()
        output[spike["at"]] = 1.0
        return {"output_audio": output, "limiter_gain_reduction_db": 0.0, "true_peak_limiter_gain_reduction_db": 0.0}

    monkeypatch.setattr(loudness, "_render", render)
    first_half = ev.measure_settings({}, ctx, ("whole", "last10"))
    assert first_half["whole"]["clipped"] is True and first_half["last10"]["clipped"] is False
    spike["at"] = ctx["raw"].size - 100
    second_half = ev.measure_settings({}, ctx, ("whole", "last10"))
    assert second_half["whole"]["clipped"] is True and second_half["last10"]["clipped"] is True


def test_the_trigger_can_fire_from_the_first_count_that_is_not_underpowered(monkeypatch):
    monkeypatch.setattr(gate_corpus, "BOOTSTRAP", 4000)  # the study's own bootstrap, not this file's quick one
    speakers = np.array([f"p{i // 3}" for i in range(54)])

    def lower_bound(firings):  # every firing false, each from another speaker: the most the clusters can give
        flags = np.zeros(54)
        flags[[3 * i for i in range(firings)]] = 1.0
        return gate_corpus.cluster_ci(flags, speakers)[1]

    assert ev.MIN_FIRED == 7
    assert lower_bound(ev.MIN_FIRED - 1) <= ev.TRIGGER_SHARE < lower_bound(ev.MIN_FIRED)
    def b3_row(rows):
        report = ev.refusal_report(rows)
        return next(r for r in report["vetoes"] if (r["veto"], r["track"], r["counting"]) == ("B3", "whole", "first")), report

    def make_b3(row):
        row.update(fired=["B3"], recommended=False, objects={"whole": {"vacuous": False}},
                   metrics={"current": {"whole": OK}, "whole": {"whole": OK}})

    rows = ev.fabricated_rows()  # B3 is first for two users already
    for row in rows[23 : 23 + ev.MIN_FIRED - 3]:
        make_b3(row)
    row3, report = b3_row(rows)
    assert row3["fired"] == ev.MIN_FIRED - 1 and row3["underpowered"] and "B3:whole:first" in report["underpowered"]
    make_b3(rows[23 + ev.MIN_FIRED - 3])
    row3, report = b3_row(rows)
    assert row3["fired"] == ev.MIN_FIRED and not row3["underpowered"] and "B3:whole:first" not in report["underpowered"]
    assert "fewer than 7" in report["trigger"]


# ---------------------------------------------------------------------------------------------------------------
# 12. Running the jobs


def _job(index):
    return ("root", "fit", index, "rnnoise")


def _row(index):
    return {"split": "fit", "index": index, "model": "rnnoise", "build": "b"}


def test_the_first_user_runs_here_and_a_failing_one_stops_the_run_before_the_pool_starts(monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "ProcessPoolExecutor", ThreadPoolExecutor)
    calls = []

    def work(job):
        calls.append(job[2])
        raise OSError("a model is missing")

    partial = tmp_path / "rows.run.partial.jsonl"
    with pytest.raises(RuntimeError, match=r"fit:72 failed: OSError: a model is missing"):
        ev.run_jobs(work, [_job(72 + i) for i in range(8)], 4, partial)
    assert calls == [72] and partial.read_text(encoding="utf-8") == ""


def test_a_worker_that_fails_cancels_the_pending_users_keeps_the_finished_rows_and_names_the_user(monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "ProcessPoolExecutor", ThreadPoolExecutor)
    here, calls = threading.get_ident(), []

    def work(job):
        calls.append((job[2], threading.get_ident()))
        if job[2] == 74:
            time.sleep(0.02)
            raise ValueError("boom")
        time.sleep(0.3)
        return _row(job[2])

    partial = tmp_path / "rows.run.partial.jsonl"
    with pytest.raises(RuntimeError, match=r"fit:74 failed: ValueError: boom"):
        ev.run_jobs(work, [_job(72 + i) for i in range(12)], 2, partial)
    assert calls[0] == (72, here) and len(calls) < 12  # the pending users never started
    kept = [json.loads(line)["index"] for line in partial.read_text(encoding="utf-8").splitlines()]
    assert 72 in kept and 74 not in kept and set(kept) == {index for index, _ in calls} - {74}  # running ones kept theirs


def test_a_run_resumes_from_its_partial_file_and_returns_the_rows_in_job_order(monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "ProcessPoolExecutor", ThreadPoolExecutor)
    partial = tmp_path / "rows.run.partial.jsonl"
    partial.write_text("".join(json.dumps(_row(index)) + "\n" for index in (74, 72)), encoding="utf-8")
    calls = []

    def work(job):
        calls.append(job[2])
        return _row(job[2])

    rows = ev.run_jobs(work, [_job(72 + i) for i in range(5)], 2, partial)
    assert [row["index"] for row in rows] == [72, 73, 74, 75, 76] and sorted(calls) == [73, 75, 76]
    assert len(partial.read_text(encoding="utf-8").splitlines()) == 5


def test_a_first_run_creates_the_folder_of_its_partial_file(monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "ProcessPoolExecutor", ThreadPoolExecutor)
    partial = tmp_path / "study" / "development" / "rows.run.partial.jsonl"
    rows = ev.run_jobs(lambda job: _row(job[2]), [_job(72)], 1, partial)
    assert [row["index"] for row in rows] == [72] and len(partial.read_text(encoding="utf-8").splitlines()) == 1


def test_every_user_of_a_finished_run_is_in_its_rows_so_deleting_the_partial_file_loses_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "ProcessPoolExecutor", ThreadPoolExecutor)
    jobs = [_job(72), _job(73)]
    partial = tmp_path / "rows.run.partial.jsonl"
    partial.write_text(json.dumps(_row(72)) + "\n", encoding="utf-8")
    ev.check_resume(partial, "b", jobs)
    rows = ev.run_jobs(lambda job: _row(job[2]), jobs, 1, partial)
    kept = {json.loads(line)["index"] for line in partial.read_text(encoding="utf-8").splitlines()}
    assert kept == {row["index"] for row in rows}
