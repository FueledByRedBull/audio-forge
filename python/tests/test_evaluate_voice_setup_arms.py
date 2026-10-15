"""Tests for the held-out arm comparison on DAPS: arms, switches, construction, alignment, gates, decision and report.

Everything runs on synthetic arrays and hand-made rows. The last test builds one development user from the real DAPS
files and skips without ``models/daps_script1`` (CI has none); ``DAPS_SCRIPT1_ROOT`` points it elsewhere.
"""

from __future__ import annotations

import functools
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import check_evaluation_hygiene
from tools import evaluate_voice_setup_arms as ev

refusals, loudness, gate_corpus = ev.refusals, ev.loudness, ev.gate_corpus
RATE = ev.SAMPLE_RATE
SIBILANT_FLAG = "current: a sibilant loss above 3 dB for a factory user"
OK = {"estoi": 0.5, "tone_error_db": 2.0, "sibilant_loss_db": 1.0, "deep_pause_db": -10.0, "tail_db": -8.0,
      "onset_db": -1.0, "pause_db": -9.0, "clipped": False, "limiter_gain_reduction_db": 0.5,
      "true_peak_limiter_gain_reduction_db": 0.1}


@pytest.fixture(autouse=True)
def fast_bootstrap(monkeypatch):
    monkeypatch.setattr(gate_corpus, "BOOTSTRAP", 200)


@pytest.fixture
def frozen_tree(monkeypatch, tmp_path):
    """A freeze of the real code with stand-in assets (CI has none), written under a scratch directory."""
    assets = {"extension": "1" * 64, "df.dll": "2" * 64, "onnxruntime.dll": "6" * 64,
              "onnxruntime_providers_shared.dll": "7" * 64, "vad_model": "3" * 64, "daps_manifest": "4" * 64,
              "models/DeepFilterNet3_ll_onnx.tar.gz": "8" * 64, "models/DeepFilterNet3_onnx.tar.gz": "5" * 64}
    monkeypatch.setattr(ev, "asset_hashes", lambda root=None: dict(assets))
    monkeypatch.setattr(refusals, "check_assets", lambda: None)
    for name, value in (("FROZEN", "frozen-files.json"), ("RULES_FINAL", "rules-final.md"), ("DEFAULT_ROWS", "rows.json"),
                        ("DEFAULT_REPORT", "report.json")):
        monkeypatch.setattr(ev, name, tmp_path / "details" / value)
    ev.build_id.cache_clear()
    ev.freeze(ev.FROZEN, ev.RULES_FINAL, ev.DEFAULT_ROWS, tmp_path)
    yield json.loads(ev.FROZEN.read_text(encoding="utf-8"))
    ev.build_id.cache_clear()


# ---------------------------------------------------------------------------------------------------------------
# 1. Arms, switches and the users


def test_the_arm_maps():
    assert ev.ARMS == ("current", "P", "T", "R", "P+T", "P+R", "T+R", "P+T+R")
    assert [ev.analysis_key(arm) for arm in ev.ARMS] == ["current", "P", "current", "R", "P", "P+R", "R", "P+R"]
    assert [ev.existing_twin(arm) for arm in ev.ARMS] == ["current", "P", "T", "current", "P+T", "P", "T", "P+T"]
    # In the existing condition the analyses are those of the non-R twins: two, not four.
    assert {ev.analysis_key(ev.existing_twin(arm)) for arm in ev.ARMS} == {"current", "P"}
    assert {ev.analysis_key(arm) for arm in ev.ARMS} == {"current", "P", "R", "P+R"}
    assert all(ev.arm_of(ev.changes_of(arm)) == arm for arm in ev.ARMS)


def test_the_users_are_the_specified_cohort():
    keys = ev.user_keys(False)
    main = [key for key in keys if key[0] != "balcony"]
    assert len(main) == 180 and len(keys) == 198 and len(set(keys)) == 198
    assert [key[0] for key in keys[:3]] == ["factory"] * 3 and keys[-1][0] == "balcony"
    factory = [key for key in main if key[0] == "factory"]
    assert len(factory) == len(main) - len(factory) == 90
    assert {key[1] for key in main} == set(range(18)) and ev.SPEAKERS[:18] == ev.HELD_OUT
    assert ev.HELD_OUT[:9] == tuple(f"f{i}" for i in range(1, 10)) and "m2" not in ev.HELD_OUT and "f10" not in ev.HELD_OUT
    per_version = {version: len({k for _, k, v in factory if v == version}) for version in ev.VERSIONS}
    assert sum(per_version.values()) == 90 and set(per_version.values()) <= {8, 9, 10}  # (k + j) mod 10 over 18 speakers
    assert all(len({v for _, kk, v in factory if kk == k}) == 5 for k in range(18))
    assert [key[2] for key in keys if key[0] == "balcony"][:2] == ["ipad_balcony1", "iphone_balcony1"]
    development = ev.user_keys(True)
    assert len(development) == 48 and {key[1] for key in development} == {18, 19}
    assert {ev.SPEAKERS[18], ev.SPEAKERS[19]} == {"m2", "f10"}


def test_the_existing_users_tone_eq_follows_the_library_entry():
    for k, version in ((0, "ipad_office1"), (1, "ipad_office1"), (2, "ipad_office1"), (5, "iphone_bedroom1")):
        entry = (k + ev.ALL_VERSIONS.index(version)) % 3
        eq = ev.tone_eq(k, version)
        bands = {round(b["frequency_hz"]): (b["gain_db"], b["filter_type"]) for b in eq["bands"]}
        for frequency, gain, kind in ev.TONE_LIBRARY[entry]:
            assert bands[round(frequency)] == (gain, kind or "bell")
        assert sum(abs(b["gain_db"]) > 0 for b in eq["bands"]) == 2


def test_arm_switches_set_the_three_globals_and_restore_them_even_after_an_exception():
    from mic_eq.analysis import voice_setup
    from mic_eq.config import Preset
    from mic_eq.config_parts import settings

    def state():
        return (voice_setup.PARTIAL_RESULTS, voice_setup.TONAL_CHECK_COUNTS_EVERY_BAND, settings.DEFAULT_SUPPRESSOR_STRENGTH)

    before = state()
    for arm in ev.ARMS:
        changes = ev.changes_of(arm)
        with ev.arm_switches(changes):
            assert state() == ("P" in changes, "T" in changes, 0.7 if "R" in changes else 1.0)
            assert Preset().rnnoise.strength == (0.7 if "R" in changes else 1.0)
        assert state() == before
    with pytest.raises(RuntimeError, match="boom"), ev.arm_switches({"P", "T", "R"}):
        raise RuntimeError("boom")
    assert state() == before and Preset().rnnoise.strength == 1.0
    with ev.arm_switches({"R"}), ev.arm_switches(set()):  # nesting restores to the outer arm
        assert settings.DEFAULT_SUPPRESSOR_STRENGTH == 1.0
    assert state() == before


def test_a_failed_restoration_is_noticed(monkeypatch):
    real, calls = ev._switch_values, []

    def flaky():  # the second reading, after the restoration, differs
        calls.append(1)
        return real() if len(calls) == 1 else {**real(), "R": 0.0}

    monkeypatch.setattr(ev, "_switch_values", flaky)
    with pytest.raises(AssertionError, match="not restored"), ev.arm_switches({"P"}):
        pass
    monkeypatch.undo()
    assert real() == {"P": False, "T": False, "R": 1.0}


def _stub_env(**changes: Any) -> dict[str, Any]:
    audio = np.zeros(RATE, np.float32)
    return {"existing": False, "eq": None, "A": False, "noise": audio, "setup": audio, "raw_noise": audio,
            "raw_setup": audio, **changes}


CENTERS = [float(f) for f in range(100, 1100, 100)]


def _optimize() -> None:
    """The optimizer's validation call, as ``analyze_auto_eq`` makes it: the taps see its arguments."""
    from mic_eq.analysis.auto_eq_parts import optimizer

    cast(Any, optimizer)._validate_and_attenuate_solution(
        None, None, None, np.zeros(10), np.full(10, 1.4), np.array(CENTERS), None)


def _stub_analysis(calls: list[dict[str, Any]], *, withheld: Any = None, fail: bool = False):
    """A stand-in for ``analyze_voice_setup``: with ``fail`` it logs a failed validation and a tone fit through the taps."""
    from mic_eq.analysis import failure_detection

    def analyze(noise, setup, rate, preset, **kwargs):
        calls.append(kwargs)
        if fail:
            failure_detection.validate_analysis({"band_gains": [0.0] * 10}, np.zeros(2049), np.linspace(0.0, 24_000.0, 2049))
            _optimize()
        diagnostics = {"joint_tuning": {}, "tone_withheld": withheld if len(calls) > 1 else None}
        return {"eq_settings": {"band_freqs": [100.0] * 10, "band_gains": [1.0] * 10, "band_qs": [1.4] * 10, "enabled": True},
                "eq_error": None, "diagnostics": diagnostics}

    return analyze


def test_r_builds_preset_at_seven_tenths_and_the_strength_is_passed_explicitly(monkeypatch):
    from mic_eq.analysis import voice_setup

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(voice_setup, "analyze_voice_setup", lambda *a, **k: calls.append(k) or {
        "eq_settings": {"band_freqs": [], "band_gains": [], "band_qs": []}, "diagnostics": {"joint_tuning": {}}})
    monkeypatch.setattr(refusals, "check_loaded", lambda diagnostics, tuning: {})
    for key, existing, strength in (("current", False, 1.0), ("R", False, 0.7), ("P+R", False, 0.7), ("R", True, 1.0)):
        calls.clear()
        ev.analyse(key, _stub_env(existing=existing, eq={"x": 1} if existing else None))
        assert calls[-1]["suppressor_strength"] == strength
        assert calls[-1]["incumbent_settings"]["rnnoise"]["strength"] == strength
        assert calls[-1]["incumbent_settings"]["eq"] == ({"x": 1} if existing else refusals.current_settings(True)["eq_settings"])
    with ev.arm_switches({"R"}):
        assert ev.current_for({"R"}, _stub_env())["suppressor_settings"]["strength"] == 0.7
        with pytest.raises(AssertionError):  # a current built without R's switch does not match R's arm
            ev.current_for(set(), _stub_env())
    assert ev.current_for(set(), _stub_env())["suppressor_settings"]["strength"] == 1.0


def test_p_runs_the_analysis_again_untapped_when_the_tap_hides_a_tone_only_failure(monkeypatch):
    from mic_eq.analysis import voice_setup
    from mic_eq.analysis.auto_eq_parts import optimizer

    monkeypatch.setattr(optimizer, "_validate_and_attenuate_solution", lambda *a, **k: (np.array([3.0, -2.0] + [0.0] * 8), 0.0))
    monkeypatch.setattr(refusals, "check_loaded", lambda diagnostics, tuning: {})
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(voice_setup, "analyze_voice_setup", _stub_analysis(calls, withheld={"cause": ["gain_rms"]}, fail=True))
    product = ev.analyse("P", _stub_env())
    assert len(calls) == 2 and product["b1"] is None and product["result"]["diagnostics"]["tone_withheld"]
    assert product["fit"]["band_gains"][:2] == [3.0, -2.0]  # the withheld tone, as the step object
    assert product["fit"]["band_freqs"] == CENTERS and product["fit"]["band_qs"] == [1.4] * 10  # the optimizer's bands
    calls.clear()
    other = ev.analyse("current", _stub_env())
    assert len(calls) == 1 and other["b1"]["passed"] is False and other["fit"] is not None  # B1, as in the refusal study
    calls.clear()
    monkeypatch.setattr(voice_setup, "analyze_voice_setup", _stub_analysis(calls, withheld=None, fail=True))
    assert ev.analyse("P", _stub_env())["b1"] is not None and len(calls) == 2  # run again; the product refuses too


def test_a_withheld_tone_whose_result_is_the_typed_eq_still_yields_its_fit(monkeypatch):
    """Auto-EQ abstains for a tone-only reason: the product returns the user's typed EQ, which has no band_freqs."""
    from mic_eq.analysis import voice_setup
    from mic_eq.analysis.auto_eq_parts import optimizer
    from mic_eq.config import EQSettings

    monkeypatch.setattr(optimizer, "_validate_and_attenuate_solution", lambda *a, **k: (np.array([3.0, -2.0] + [0.0] * 8), 0.0))
    monkeypatch.setattr(refusals, "check_loaded", lambda diagnostics, tuning: {})

    def analyze(noise, setup, rate, preset, **kwargs):
        _optimize()
        return {"eq_settings": {**EQSettings().to_dict(), "apply_recommended": False}, "eq_error": None,
                "diagnostics": {"joint_tuning": {}, "tone_withheld": {"cause": ["headroom validation removed the usable correction"]}}}

    monkeypatch.setattr(voice_setup, "analyze_voice_setup", analyze)
    product = ev.analyse("P", _stub_env())
    assert "band_freqs" not in product["result"]["eq_settings"] and product["b1"] is None
    assert product["fit"] == {"band_freqs": CENTERS, "band_gains": [3.0, -2.0] + [0.0] * 8, "band_qs": [1.4] * 10, "enabled": True}
    assert ev.stacked(product["fit"], ev.tone_eq(0, "ipad_office1"))["layers"]["correction"]  # the step object can be built


def test_a_capture_refusal_runs_the_analysis_with_the_clip_check_off(monkeypatch):
    from mic_eq.analysis import voice_setup

    seen = []
    monkeypatch.setattr(voice_setup, "analyze_voice_setup", lambda *a, **k: seen.append(voice_setup.raw_capture_is_clipped(
        np.ones(100, np.float32))) or {"eq_settings": {"band_freqs": [], "band_gains": [], "band_qs": []},
                                      "diagnostics": {"joint_tuning": {}}})
    monkeypatch.setattr(refusals, "check_loaded", lambda diagnostics, tuning: {})
    ev.analyse("current", _stub_env(A=True))
    ev.analyse("current", _stub_env(A=False))
    assert seen == [False, True]
    assert voice_setup.raw_capture_is_clipped(np.ones(100, np.float32))


def test_a_proposal_is_stacked_on_the_users_eq_and_matches_the_flat_stacking_for_a_factory_user():
    from mic_eq.config import EQ_FREQUENCIES

    proposal = {"band_freqs": list(EQ_FREQUENCIES), "band_gains": [1.0] * 10, "band_qs": [1.41] * 10, "enabled": True}
    flat = ev.stacked(proposal, None)
    assert flat == loudness._applied_eq(proposal)
    eq = ev.tone_eq(0, "ipad_office1")
    over = ev.stacked(proposal, eq)
    assert over["layers"]["tone"] == eq["bands"] and over["bands"] == eq["bands"]
    assert over["layers"]["correction"] == flat["layers"]["correction"] and over["layers"]["tone"] != flat["layers"]["tone"]
    assert ev.stacked(over, eq) == over  # typed already: unchanged


def test_an_injection_moves_the_nearest_tone_band_and_keeps_the_layers_consistent():
    from mic_eq.config import EQSettings

    eq = ev.stacked({"band_freqs": [100.0] * 10, "band_gains": [0.0] * 10, "band_qs": [1.41] * 10}, ev.tone_eq(1, "ipad_office1"))
    for cell, (frequency, gain) in ev.INJECTIONS.items():
        wrong = ev.inject(eq, frequency, gain)
        assert wrong["layers"]["tone"] == wrong["bands"]
        band = min(wrong["bands"], key=lambda b: abs(np.log(b["frequency_hz"] / frequency)))
        assert (band["frequency_hz"], band["gain_db"]) == (frequency, gain)
        EQSettings.from_dict({k: wrong[k] for k in ("schema_version", "enabled", "bands", "layers")})  # still a valid EQ
    assert {abs(gain) for _, gain in ev.INJECTIONS.values()} == {12.0, 9.0} and len(ev.INJECTIONS) == 8


# ---------------------------------------------------------------------------------------------------------------
# 2. Construction on synthetic labels and audio


def _labels(spec: list[tuple[int, bool]]) -> np.ndarray:
    return np.concatenate([np.full(n, value, bool) for n, value in spec])


def test_speech_labels_follow_the_clean_energy_close_gaps_and_drop_islands(monkeypatch):
    rng = np.random.default_rng(0)

    def burst(frames: int, level: float) -> np.ndarray:
        return level * rng.standard_normal(frames * ev.FRAME)

    quiet = 1e-4
    clean = np.concatenate([burst(100, quiet), burst(200, 0.1), burst(10, quiet), burst(200, 0.1), burst(100, quiet),
                            burst(3, 0.1), burst(100, quiet), burst(100, 0.1), burst(100, quiet)])
    labels = ev.speech_labels(clean)
    assert labels.size == 913
    assert not labels[:100].any() and labels[100:510].all()  # the 100 ms gap between phrases is closed
    assert not labels[510:713].any()  # the 30 ms island is dropped, and a 1 s pause stays a pause
    assert labels[713:813].all() and not labels[813:].any()
    monkeypatch.setattr(ev, "SPEECH_WITHIN_DB", 80.0)  # the bound is the study's constant
    assert ev.speech_labels(clean).all()


def test_noise_prefers_the_leading_silence_and_takes_its_central_two_seconds():
    speech = _labels([(500, False), (400, True), (300, False), (5000, True), (600, False), (4000, True)])
    found = ev.noise_candidates(speech)
    assert [c["kind"] for c in found] == ["leading", "longest", "longest"]
    # trimmed runs: (20, 490) after the first 200 ms and 100 ms before the onset; (6250, 6790); (950, 1190)
    assert [c["pieces"] for c in found] == [[(155, 355)], [(6420, 6620)], [(970, 1170)]]
    assert [c["end"] for c in found] == [355, 6620, 1170]


def test_a_leading_silence_under_one_and_a_half_seconds_gives_way_to_the_longest_run():
    speech = _labels([(150, False), (400, True), (300, False), (400, True), (800, False), (400, True), (300, False), (400, True)])
    found = ev.noise_candidates(speech)
    assert [c["kind"] for c in found] == ["longest"] * 3
    # runs of 740, 240 and 240 frames after the trims; ties go to the earliest
    assert [c["pieces"] for c in found] == [[(1570, 1770)], [(620, 820)], [(2520, 2720)]]


def test_a_run_between_one_and_a_half_and_two_seconds_is_taken_whole():
    speech = _labels([(100, True), (50 + 170 + 10, False), (100, True)])
    (candidate,) = ev.noise_candidates(speech)
    assert candidate["pieces"] == [(150, 320)] and candidate["end"] == 320


def test_with_no_run_of_one_and_a_half_seconds_the_pauses_are_spliced_with_crossfades():
    parts: list[tuple[int, bool]] = [(100, True)]
    for _ in range(6):
        parts += [(50 + 70 + 10, False), (100, True)]  # runs of 0.7 s after the trims
    speech = _labels(parts)
    (candidate,) = ev.noise_candidates(speech)
    assert candidate["kind"] == "spliced" and len(candidate["pieces"]) == 3
    assert all(stop - start >= ev.SPLICE_MIN_RUN for start, stop in candidate["pieces"])
    device = np.arange(speech.size * ev.FRAME, dtype=np.float64)  # a ramp shows where each sample came from
    audio = ev.noise_audio(device, candidate)
    assert audio.size == int(ev.NOISE_S * RATE) and audio.dtype == np.float32
    (first_start, first_stop), (second_start, _) = candidate["pieces"][:2]
    assert audio[0] == np.float32(first_start * ev.FRAME)
    junction = (first_stop - first_start) * ev.FRAME - ev.FADE
    middle = ev.FADE // 2
    weight = 0.5 - 0.5 * np.cos(np.pi * (middle + 0.5) / ev.FADE)
    expected = (1 - weight) * (first_stop * ev.FRAME - ev.FADE + middle) + weight * (second_start * ev.FRAME + middle)
    assert audio[junction + middle] == pytest.approx(expected, abs=1.0)
    assert audio[junction + ev.FADE] == pytest.approx(second_start * ev.FRAME + ev.FADE, abs=1.0)  # the second piece follows


def test_under_two_seconds_of_pause_material_excludes_the_user():
    speech = _labels([(100, True), (50 + 60 + 10, False), (100, True), (50 + 60 + 10, False), (100, True)])
    assert ev.noise_candidates(speech) == []
    clean = np.random.default_rng(1).standard_normal(speech.size * ev.FRAME) * np.repeat(speech, ev.FRAME)
    assert ev.construct(clean, clean, speech, lambda n, v: (None, None)) == {"excluded": "noise_material"}


def test_the_setup_window_starts_half_a_second_before_the_next_onset_and_the_evaluation_window_in_a_deep_pause():
    speech = _labels([(500, False), (400, True), (300, False), (400, True), (5000, False), (300, True), (3000, False),
                      (300, True), (2500, False), (300, True), (400, False), (3000, True)])
    assert ev.setup_starts(speech, 120)[0] == 500 - 50
    assert all(start + 1000 <= speech.size for start in ev.setup_starts(speech, 120))
    starts = ev.evaluation_starts(speech, 1450)
    assert starts and all(m >= 1450 + 2000 and m + 2000 <= speech.size for m in starts)
    counts = [ev.deep_pause_frames(speech[m : m + 2000]) for m in starts]
    assert counts == sorted(counts, reverse=True) and counts[0] == 2000 > counts[-1]
    tied = _labels([(100, True), (3000, False), (100, True), (3000, False), (100, True), (3000, False), (100, True), (3000, False)])
    first = ev.evaluation_starts(tied, 0)
    best = ev.deep_pause_frames(tied[first[0] : first[0] + 2000])
    equal = [m for m in first if ev.deep_pause_frames(tied[m : m + 2000]) == best]
    assert equal == sorted(equal)  # ties go to the earliest
    window = _labels([(100, True), (300, False), (100, True)])
    assert ev.deep_pause_frames(window) == 182  # pause frames 159..340: 600 ms from speech on both sides


def _signals(seconds: float = 100.0, delay: int = 336, seed: int = 0):
    """A clean take of noise bursts (2 s of speech, 2.5 s of pause, after 3 s of silence) and a delayed device take."""
    rng = np.random.default_rng(seed)
    pieces = [np.zeros(3 * RATE)]
    while sum(p.size for p in pieces) < seconds * RATE:
        pieces += [0.1 * rng.standard_normal(2 * RATE), np.zeros(int(2.5 * RATE))]
    clean = np.concatenate(pieces)[: int(seconds * RATE)]
    device = np.concatenate([np.zeros(delay), clean])[: clean.size] * 0.7 + 1e-4 * rng.standard_normal(clean.size)
    return clean, device


def test_alignment_finds_a_synthetic_delay_and_rejects_a_mismatched_pair():
    clean, device = _signals()
    start, size = 40 * RATE, 20 * RATE
    found = ev.align(clean, device, start, size)
    assert found["ok"] and found["lag"] == 336 and found["lags"] == [336, 336]
    window = device[start + found["lag"] : start + found["lag"] + size]
    assert np.corrcoef(clean[start : start + size], window)[0, 1] > 0.99
    speech = ev.speech_labels(clean)[start // ev.FRAME : start // ev.FRAME + 2000]
    assert ev.envelope_correlation(clean[start : start + size], window, speech) > 0.9
    assert ev.gain_db(clean[start : start + size], window, speech) == pytest.approx(20 * np.log10(0.7), abs=0.2)
    assert not ev.align(clean, _signals(seed=5)[1], start, size)["ok"]  # noise bursts that do not match: no agreeing lags


def test_alignment_refuses_lags_over_twenty_milliseconds_and_lags_that_drift_by_more_than_one_millisecond():
    clean, device = _signals()
    start, size = 40 * RATE, 20 * RATE
    late = np.concatenate([np.zeros(1100), device])[: device.size]
    assert not ev.align(clean, late, start, size)["ok"]  # 1100 samples late is over 20 ms
    ratio = 1.0 + 80 / (15 * RATE)  # 80 samples of drift between the first and last 5 s
    drifting = np.interp(np.arange(device.size) * ratio, np.arange(device.size), device)
    found = ev.align(clean, drifting, start, size)
    assert abs(found["lags"][1] - found["lags"][0]) >= 49 and not found["ok"]
    assert not ev.align(clean, device, 10, size)["ok"]  # no room for the +-50 ms margin before the window
    gentle = np.interp(np.arange(device.size) * (1.0 + 30 / (15 * RATE)), np.arange(device.size), device)
    assert ev.align(clean, gentle, start, size)["ok"]  # 30 samples of drift (the takes drift 1-2 a second) passes


def test_a_recording_is_constructed_from_the_clean_labels_and_the_device_take_is_shifted_by_the_lag():
    clean, device = _signals(seconds=120.0)
    speech = ev.speech_labels(clean)
    seen = []
    built = ev.construct(clean, device, speech, lambda noise, voice: seen.append((noise.size, voice.size)) or (None, None))
    assert built["excluded"] is None and not built["A"] and seen == [(2 * RATE, 10 * RATE)]
    readings = built["readings"]
    assert readings["noise_kind"] == "leading" and readings["lag"] == 336 and readings["takes"][0]["vetoes"] == []
    assert built["evaluation"].size == built["clean"].size == 20 * RATE == built["labels"].size
    assert np.corrcoef(built["clean"], built["evaluation"])[0, 1] > 0.99  # shifted by the lag: the pair is aligned
    assert readings["evaluation_start_s"] >= readings["setup_start_s"] + 10 + 20
    assert not readings["drift_flag"] and readings["envelope_correlation"] > 0.9 and readings["deep_pause_frames"] > 0


def test_a_veto_asks_for_the_next_segment_and_three_failures_are_a_capture_refusal_judged_on_the_first_take():
    clean, device = _signals(seconds=120.0)
    speech = ev.speech_labels(clean)
    asked = []

    def noise_once(noise, voice):
        asked.append(noise.copy())
        return ("noise_recapture", None) if len(asked) == 1 else (None, None)

    built = ev.construct(clean, device, speech, noise_once)
    takes = built["readings"]["takes"]
    assert [t["vetoes"] for t in takes] == [["noise_recapture"], []] and takes[1]["noise"] > takes[0]["noise"]
    assert not built["A"] and np.array_equal(built["noise"], asked[1]) and not np.array_equal(asked[0], asked[1])

    def voice_once(noise, voice):
        asked.append(voice)
        return (None, "voice_clipped") if len(asked) == 1 else (None, None)

    asked.clear()
    takes = ev.construct(clean, device, speech, voice_once)["readings"]["takes"]
    assert takes[1]["noise"] == takes[0]["noise"] and takes[1]["setup"] > takes[0]["setup"]
    built = ev.construct(clean, device, speech, lambda n, v: (None, "voice_silent"))
    takes = built["readings"]["takes"]
    assert built["A"] and len(takes) == 3 and built["readings"]["A"]
    assert built["readings"]["setup_start_s"] == takes[0]["setup"] / 100.0  # judged on the first take
    first = takes[0]["setup"]
    assert np.array_equal(built["setup"], device[first * ev.FRAME : (first + 1000) * ev.FRAME].astype(np.float32))


def _too_hot(seconds: float = 120.0):
    """The synthetic pair with the device take normalised near full scale, as the DAPS device files are."""
    clean, device = _signals(seconds=seconds)
    return clean, device * (0.95 / np.max(np.abs(device)))


def _first_take(device, readings):
    start = readings["takes"][0]["setup"] * ev.FRAME
    return device[start : start + 10 * RATE].astype(np.float32)


def test_a_too_hot_recording_is_scaled_so_the_level_checks_peak_reading_is_minus_six_db():
    from mic_eq.ui.level_check import TARGET_PEAK_DB, verdict_from_audio

    clean, device = _too_hot()
    built = ev.construct(clean, device, ev.speech_labels(clean), lambda noise, voice: (None, None))
    level = built["readings"]["level"]
    raw = verdict_from_audio(_first_take(device, built["readings"]), RATE)
    assert raw.code == "too_hot" and level["before"] == "too_hot" and raw.peak_db is not None
    assert level["gain_db"] < 0 and level["after"] == verdict_from_audio(built["setup"], RATE).code != "too_hot"
    assert verdict_from_audio(built["setup"], RATE).peak_db == pytest.approx(TARGET_PEAK_DB, abs=1e-3)
    assert raw.peak_db + level["gain_db"] == pytest.approx(TARGET_PEAK_DB, abs=1e-3)


def test_the_rows_level_check_is_the_verdict_of_the_recording_before_scaling(monkeypatch, tmp_path):
    (tmp_path / "MANIFEST.json").write_text("{}", encoding="utf-8")
    clean, device = _too_hot()
    monkeypatch.setattr(ev, "check_corpus", lambda *args: None)
    monkeypatch.setattr(ev, "read_take", lambda root, speaker, version: clean if version == "clean" else device)
    monkeypatch.setattr(ev, "build_id", lambda root: "test")
    monkeypatch.setattr("mic_eq.ui.calibration_support.filtered_capture_for_analysis", lambda audio, sample_rate: audio)
    row, env = ev.prepare((str(tmp_path), "main", 0, "ipadflat_confroom1"), development=True)
    level = row["construction"]["level"]
    assert env is not None and level["before"] == "too_hot" and level["after"] != "too_hot"
    assert row["level_check"] == "too_hot"


def test_a_recording_that_is_not_too_hot_is_used_as_recorded():
    clean, device = _signals(seconds=120.0)
    built = ev.construct(clean, device, ev.speech_labels(clean), lambda noise, voice: (None, None))
    level = built["readings"]["level"]
    assert level["before"] == level["after"] != "too_hot" and level["gain_db"] == 0
    assert np.array_equal(built["setup"], _first_take(device, built["readings"]))


def test_one_gain_scales_every_capture_of_the_user():
    clean, device = _too_hot()
    speech = ev.speech_labels(clean)
    built = ev.construct(clean, device, speech, lambda noise, voice: (None, None))
    readings = built["readings"]
    gain = 10 ** (readings["level"]["gain_db"] / 20)
    start = round(readings["evaluation_start_s"] * 100) * ev.FRAME
    evaluation = device[start + readings["lag"] : start + readings["lag"] + built["evaluation"].size]
    noise = ev.noise_audio(device, ev.noise_candidates(speech)[readings["takes"][0]["noise"]])
    assert gain < 1
    assert np.allclose(built["noise"], noise * gain, atol=1e-6)
    assert np.allclose(built["setup"], _first_take(device, readings) * gain, atol=1e-6)
    assert np.allclose(built["evaluation"], evaluation * gain, atol=1e-6)
    assert np.array_equal(built["clean"], clean[start : start + built["clean"].size])  # the clean take is not scaled


def test_a_too_hot_take_that_trips_only_the_clip_run_rule_is_not_refused_at_capture_after_scaling():
    from mic_eq.analysis.voice_setup import raw_capture_is_clipped

    clean, device = _too_hot()
    speech = ev.speech_labels(clean)
    first = ev.setup_starts(speech, ev.noise_candidates(speech)[0]["end"])[0] * ev.FRAME
    device[first + 5 * RATE : first + 5 * RATE + 8] = 1.0  # a flat top of 8 samples at full scale
    clipped = lambda noise, voice: (None, "voice_clipped" if raw_capture_is_clipped(voice, RATE) else None)  # noqa: E731
    assert clipped(None, device[first : first + 10 * RATE].astype(np.float32))[1] == "voice_clipped"
    built = ev.construct(clean, device, speech, clipped)
    assert not built["A"] and [t["vetoes"] for t in built["readings"]["takes"]] == [[]]
    assert built["readings"]["level"]["before"] == "too_hot" and built["readings"]["level"]["gain_db"] < 0


def test_the_tuner_runtime_gate_is_lifted_in_every_arm():
    assert ev.UNBOUNDED_TUNER_RUNTIME is True


def test_a_recording_without_an_aligned_evaluation_window_is_excluded():
    clean, device = _signals(seconds=120.0)
    other = _signals(seconds=120.0, seed=9)[1]
    assert ev.construct(clean, other, ev.speech_labels(clean), lambda n, v: (None, None)) == {
        "excluded": "no_aligned_evaluation_window"}
    short, _ = _signals(seconds=30.0)
    assert ev.construct(short, short, ev.speech_labels(short), lambda n, v: (None, None))["excluded"] in (
        "no_setup_window", "no_evaluation_window")


def test_the_capture_vetoes_fire_on_hand_made_takes():
    rng = np.random.default_rng(0)
    quiet = (0.01 * rng.standard_normal(2 * RATE)).astype(np.float32)
    voice = (0.05 * rng.standard_normal(10 * RATE)).astype(np.float32)
    assert ev.capture_vetoes(quiet, voice) == (None, None)
    flat = voice.copy()
    flat[1000:1010] = 1.0
    assert ev.capture_vetoes(quiet, flat) == (None, "voice_clipped")
    assert ev.capture_vetoes(np.concatenate([quiet[:-20], np.ones(20, np.float32)]), voice)[0] == "noise_clipped"
    assert ev.capture_vetoes(quiet, np.zeros_like(voice)) == (None, "voice_silent")
    assert ev.capture_vetoes(np.full(2 * RATE, np.nan, np.float32), voice)[0] == "noise_clipped"


def test_the_corpus_files_are_checked_against_the_manifest(tmp_path):
    for version in ("clean", "ipad_office1"):
        (tmp_path / version).mkdir()
        (tmp_path / version / f"f1_script1_{version}.wav").write_bytes(version.encode())
    entries = [{"path": f"{v}/f1_script1_{v}.wav", "sha256": gate_corpus._sha256(tmp_path / v / f"f1_script1_{v}.wav")}
               for v in ("clean", "ipad_office1")]
    manifest = {"files": entries}
    ev.check_corpus(tmp_path, manifest, "f1", "ipad_office1")
    (tmp_path / "ipad_office1" / "f1_script1_ipad_office1.wav").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="ipad_office1/f1_script1_ipad_office1.wav"):
        ev.check_corpus(tmp_path, manifest, "f1", "ipad_office1")
    with pytest.raises(RuntimeError, match="differ from the manifest"):
        ev.check_corpus(tmp_path, {"files": entries[:1]}, "f1", "ipad_office1")  # a file the manifest does not name


# ---------------------------------------------------------------------------------------------------------------
# 3. Counting, gates and the decision


def _view(*, fired=(), recommended=True, e=OK, whole=OK, accepted=True, tone_withheld=False, a=False, objects=None, **metrics):
    return {"A": a, "fired": list(fired), "recommended": recommended, "short_speech": False, "tone_withheld": tone_withheld,
            "verification": {"decision": "accept"}, "end_state": "candidate" if accepted else "current",
            "metrics": {"current": {"whole": OK, "last10": OK}, "whole": {"whole": whole}, "E": {"whole": e}, **metrics},
            "objects": {"whole": {"vacuous": False}, "E": {"vacuous": not accepted}, **(objects or {})}}


def test_the_flags_of_a_user_condition():
    refused: dict[str, Any] = dict(fired=("B4",), recommended=False, accepted=False)
    assert not any(ev.arm_flags(_view()).values())
    assert ev.arm_flags(_view(**refused))["first:B4"] == 1
    flags = ev.arm_flags(_view(**refused, whole={**OK, "deep_pause_db": -5.0}))
    assert flags["false_first"] == 0 and flags["first:B4"] == 0  # a correct refusal
    flags = ev.arm_flags(_view(fired=("B3",), recommended=False, accepted=False, whole={**OK, "deep_pause_db": None}))
    assert flags["false_first"] == 0 and flags["false_first_upper"] == 1  # the two readings of an unmeasurable clause
    flags = ev.arm_flags(_view(a=True, accepted=False, recommended=False))
    assert flags["first:A"] == flags["false_first"] == 1
    assert ev.arm_flags(_view(fired=("B2",), a=True, accepted=False, recommended=False))["first:A"] == 1  # A comes first
    assert ev.arm_flags(_view(e={**OK, "tone_error_db": 3.5}))["fa"] == 1
    flags = ev.arm_flags(_view(e={**OK, "deep_pause_db": None}))
    assert flags["fa"] == 0 and flags["fa_upper"] == 1
    assert ev.arm_flags(_view(e={**OK, "tone_error_db": 3.0}))["fa"] == 0  # exactly at the tolerance passes
    assert ev.arm_flags(_view(accepted=False))["fa"] == 0  # the current settings are never a false accept
    step = _view(fired=("B5",), recommended=False, accepted=False, objects={"tone": {"vacuous": False}})
    assert ev.arm_flags(step)["step:B5"] == 0  # a step without a render is unjudgeable, not a false refusal
    step["metrics"]["tone"] = {"whole": OK}
    assert ev.arm_flags(step)["step:B5"] == 1
    withheld = _view(tone_withheld=True, objects={"p_tone": {"vacuous": False}}, p_tone={"whole": OK})
    assert ev.arm_flags(withheld)["step:P"] == 1
    withheld["metrics"]["p_tone"] = {"whole": {**OK, "tone_error_db": 4.0}}
    assert ev.arm_flags(withheld)["step:P"] == 0
    assert not ev.arm_flags({**_view(), "analysis_failure": "x", "fired": ["B4"]})["false_first"]


def _gate(inputs, change, name, contrast=0):
    return next(g for g in ev.change_gates(inputs, change) if g["gate"] == name and g["contrast"] == f"K{contrast + 1}")


def test_a_count_gate_passes_at_equality_fails_by_one_and_labels_the_margin():
    speakers = np.array([f"s{u % 18}" for u in range(180)])
    count = lambda n: np.arange(180) < n  # noqa: E731
    assert ev.count_gate("g", count(9), count(0), speakers, budget=9)["pass"]
    assert not ev.count_gate("g", count(10), count(0), speakers, budget=9)["pass"]
    assert ev.count_gate("g", count(14), count(14), speakers, budget=9)["pass"]  # max(budget, off)
    assert not ev.count_gate("g", count(15), count(14), speakers, budget=9)["pass"]
    assert ev.count_gate("g", count(3), count(3), speakers)["pass"] and not ev.count_gate("g", count(4), count(3), speakers)["pass"]
    assert ev.count_gate("g", count(2), count(3), speakers, strict=True)["pass"]
    assert not ev.count_gate("g", count(3), count(3), speakers, strict=True)["pass"]
    near = ev.count_gate("g", count(9), count(0), speakers, budget=9)
    assert near["label"] == ev.NOT_CONFIRMED and near["interval"][2] > 9  # at the budget: the interval reaches past it
    clear = ev.count_gate("g", count(0), count(0), speakers, budget=9)
    assert clear["label"] == "confirmed at 95 %" and len(clear["interval"]) == 3
    assert ev.count_gate("g", count(10), count(0), speakers, budget=9)["label"] == "failed"
    assert near["on"] == 9 and near["off"] == 0 and near["limit"] == 9


def test_the_gate_self_test_finds_no_problem():
    assert ev.gate_self_test() == []


@pytest.mark.parametrize("change", ev.CHANGES)
@pytest.mark.parametrize("contrast", [0, 1])
def test_every_budget_gate_passes_at_its_budget_and_fails_one_over(change, contrast):
    for flag, gate, budget in [*((f"first:{v}", v, ev.WHOLE_BUDGET) for v in ev.WHOLE_VETOES),
                               *((f"step:{v}", f"step:{v}", ev.STEP_BUDGET) for v in ev.STEP_OBJECTS),
                               ("false_first", "combined", ev.COMBINED_BUDGET)]:
        for count, expected in ((budget, True), (budget + 1, False)):
            inputs = ev.fabricated_inputs()
            inputs["flags"][ev.CONTRASTS[change][contrast][1]][flag][:count] = 1
            assert _gate(inputs, change, gate, contrast)["pass"] is expected, (change, contrast, gate, count)


def test_false_accepts_count_against_the_arm_without_the_change_and_t_only_asserts():
    for change in "PR":
        for contrast in (0, 1):
            on, off = ev.CONTRASTS[change][contrast][1:]
            inputs = ev.fabricated_inputs()
            inputs["flags"][on]["fa"][:2], inputs["flags"][off]["fa"][:2] = 1, 1
            assert _gate(inputs, change, "FA", contrast)["pass"]
            inputs["flags"][on]["fa"][:3] = 1
            assert not _gate(inputs, change, "FA", contrast)["pass"]
    inputs = ev.fabricated_inputs()
    assert not any(g["gate"] == "FA" for g in ev.change_gates(inputs, "T"))
    inputs["flags"]["T"]["fa"][:1] = 1
    with pytest.raises(AssertionError, match="only adds rollbacks"):
        ev.change_gates(inputs, "T")


def test_p_needs_fewer_whole_setup_false_refusals_than_without_it():
    for contrast in (0, 1):
        on, off = ev.CONTRASTS["P"][contrast][1:]
        inputs = ev.fabricated_inputs()
        inputs["flags"][off]["false_first"][:5], inputs["flags"][on]["false_first"][:4] = 1, 1
        assert _gate(inputs, "P", "BEN-P", contrast)["pass"]
        inputs["flags"][on]["false_first"][:5] = 1
        assert not _gate(inputs, "P", "BEN-P", contrast)["pass"]


def test_t_needs_thirty_pairs_and_eighty_percent_in_each_twelve_db_cell():
    inputs = ev.fabricated_inputs()
    gates = [g["gate"] for g in ev.change_gates(inputs, "T") if g["gate"].startswith("BEN-T")]
    assert sorted(set(gates)) == sorted(f"BEN-T:{cell}" for cell, (_, gain) in ev.INJECTIONS.items() if abs(gain) == 12)
    for contrast in (0, 1):
        key = ev.analysis_key(ev.CONTRASTS["T"][contrast][1])
        for cell in [c for c, (_, g) in ev.INJECTIONS.items() if abs(g) == 12]:
            for ran, hit, expected in ((30, 24, True), (30, 23, False), (29, 29, False), (50, 40, True), (50, 39, False)):
                inputs = ev.fabricated_inputs()
                inputs["injections"][key][cell]["on"] = [ran, hit]
                assert _gate(inputs, "T", f"BEN-T:{cell}", contrast)["pass"] is expected
        inputs = ev.fabricated_inputs()
        inputs["injections"][key]["100Hz+9dB"]["on"] = [0, 0]  # the 9 dB cells are reported, never gated
        assert all(g["pass"] for g in ev.change_gates(inputs, "T"))


def test_r_needs_the_deep_pause_share_thirty_users_and_fewer_sibilant_losses():
    for contrast in (0, 1):
        on, off = ev.CONTRASTS["R"][contrast][1:]
        for users, lost, expected in ((90, 9, True), (90, 10, False), (30, 3, True), (30, 4, False), (29, 0, False)):
            inputs = ev.fabricated_inputs()
            inputs["deep"][on][users:] = np.nan
            inputs["deep"][on][:lost] = -3.9 + 1.0  # above -4.0: the deep pause is lost
            assert _gate(inputs, "R", "BEN-R1", contrast)["pass"] is expected, (users, lost)
        inputs = ev.fabricated_inputs()
        inputs["deep"][off][:20] = -2.0  # the arm without R keeps 70 of 90: the bar is its own share, under 0.90
        inputs["deep"][on][:20] = -2.0
        assert _gate(inputs, "R", "BEN-R1", contrast)["pass"]
        inputs["deep"][on][:21] = -2.0
        assert not _gate(inputs, "R", "BEN-R1", contrast)["pass"]
        inputs = ev.fabricated_inputs()
        inputs["deep"][on][~inputs["factory"]] = -2.0
        assert _gate(inputs, "R", "BEN-R1", contrast)["pass"]  # only factory users count
        inputs = ev.fabricated_inputs()
        inputs["sibilant"][on][:] = 0.0  # fewer: the arm without R loses one user
        assert _gate(inputs, "R", "BEN-R2", contrast)["pass"]
        inputs["sibilant"][on][0] = ev.SIBILANT_LOSS_DB + 0.01
        assert not _gate(inputs, "R", "BEN-R2", contrast)["pass"]  # equal is not fewer
        inputs["sibilant"][on][0] = ev.SIBILANT_LOSS_DB  # exactly 3.0 dB is not more than 3.0 dB
        assert _gate(inputs, "R", "BEN-R2", contrast)["pass"]
        inputs["sibilant"][on][0] = 4.0  # 4 dB is over the 3.0 dB tolerance (it was under the old 6.0)
        assert not _gate(inputs, "R", "BEN-R2", contrast)["pass"]


def test_the_decision_table_for_zero_one_and_two_failed_changes():
    assert [ev.candidate_of(failed) for failed in ([], ["P"], ["T"], ["R"], ["P", "T"], ["T", "R"], ["P", "T", "R"])] == [
        "P+T+R", "T+R", "P+R", "P+T", None, None, None]
    result = ev.decide(ev.fabricated_inputs())
    assert result["failed_changes"] == [] and result["candidate"] == "P+T+R" and result["ships"]
    broken = ev.fabricated_inputs()
    broken["flags"]["R"]["first:B4"][: ev.WHOLE_BUDGET + 1] = 1
    result = ev.decide(broken)
    assert result["failed_changes"] == ["R"] and result["candidate"] == "P+T" and result["ships"]
    one = ev.fabricated_inputs()
    one["flags"]["T"]["step:B6"][:19] = 1
    result = ev.decide(one)
    assert result["failed_changes"] == ["T"] and result["candidate"] == "P+R" and result["ships"]
    assert [g["gate"] for g in result["shipping_gates"]][:2] == ["S1", "S2:A"]
    two = ev.fabricated_inputs()
    two["flags"]["T"]["step:B6"][:19] = 1
    two["flags"]["P"]["step:B6"][:19] = 1
    result = ev.decide(two)
    assert result["failed_changes"] == ["P", "T"] and result["candidate"] is None and not result["ships"]
    assert result["shipping_gates"] == []


def test_a_candidate_ships_only_if_the_shipping_gates_pass():
    for flag, gate, budget in (("fa", "S1", 0), ("first:B4", "S2:B4", 9), ("step:B6", "S2:step:B6", 18), ("false_first", "S3", 18)):
        for count, expected in ((budget, True), (budget + 1, False)):
            inputs = ev.fabricated_inputs()
            inputs["flags"]["P+T+R"][flag][:count] = 1
            assert {g["gate"]: g["pass"] for g in ev.shipping_gates(inputs, "P+T+R")}[gate] is expected
    # the current build's own count raises the bar of S2, and S1 compares with it too
    inputs = ev.fabricated_inputs()
    inputs["flags"]["current"]["first:B4"][:12] = 1
    inputs["flags"]["P+R"]["first:B4"][:12] = 1
    assert {g["gate"]: g["pass"] for g in ev.shipping_gates(inputs, "P+R")}["S2:B4"]
    inputs["flags"]["P+R"]["first:B4"][:13] = 1
    assert not {g["gate"]: g["pass"] for g in ev.shipping_gates(inputs, "P+R")}["S2:B4"]
    inputs["flags"]["current"]["fa"][:1] = 1
    assert {g["gate"]: g["pass"] for g in ev.shipping_gates(inputs, "P+R")}["S1"]


def test_the_fabricated_world_gives_exact_counts_and_a_decision():
    rows = ev.fabricated_rows()
    assert len(rows) == 198 and len(ev.main_rows(rows)) == 180
    inputs = ev.gate_inputs(rows)
    assert inputs["factory"].sum() == 90 and len(set(inputs["speakers"])) == 18
    flags = inputs["flags"]
    assert flags["current"]["false_first"].sum() == 10 and flags["P"]["false_first"].sum() == 5
    assert flags["current"]["first:B5"].sum() == 5 and flags["P"]["step:P"].sum() == 5 and flags["current"]["step:B5"].sum() == 5
    assert flags["R"]["fa"].sum() == 5 and flags["current"]["fa"].sum() == 0 and flags["P+T+R"]["fa"].sum() == 5
    assert inputs["injections"]["current"]["100Hz+12dB"]["on"] == [90, 90] and inputs["injections"]["P+R"]["100Hz-9dB"]["off"] == [90, 0]
    decision = ev.decide(inputs)
    assert decision["failed_changes"] == ["R"] and decision["candidate"] == "P+T" and decision["ships"]
    assert any(g["gate"] == "FA" and not g["pass"] and g["on_arm"] == "R" for g in decision["change_gates"]["R"])


def test_arm_users_make_a_capture_refusal_the_first_veto():
    rows = ev.fabricated_rows()[:180]
    rows[0] = deepcopy(rows[0])
    rows[0]["arms"]["current"] = {**rows[0]["arms"]["current"], "A": True, "fired": ["B2"]}
    users = ev.arm_users(rows, "current")
    assert users[0]["first"] == users[0]["bucket"] == "A" and users[1]["bucket"] == "none"
    summary = ev.arm_summary(rows, "current")
    assert summary["buckets"]["A"] == 1 and sum(summary["buckets"].values()) == 180


# ---------------------------------------------------------------------------------------------------------------
# 4. Report, freeze and run mechanics


def test_the_report_is_clean_small_and_states_its_rule(frozen_tree, tmp_path):
    source = {"rows": ev.fabricated_rows(), "identity": {"build": ev.refusals.build_of(frozen_tree)}}
    path = tmp_path / "report.json"
    ev.refusals._write(path, ev.build_report(source, frozen_tree))
    assert check_evaluation_hygiene.validate_report(path) == []
    assert path.stat().st_size < 100_000
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["audible_change"] is False and report["schema_version"] == 2
    assert report["source_sha256"] == {name: frozen_tree[name] for name in ev.SOURCES}
    assert report["frozen_sha256"] == frozen_tree and report["rule_sha256"] == frozen_tree["rule"]
    metrics = report["metrics"]
    assert set(metrics["arms"]) == set(ev.ARMS) and metrics["decision"]["candidate"] == "P+T"
    assert metrics["users"]["analysed"] == 180 and metrics["balcony"]["users"] == 18
    text = " ".join(report["limitations"])
    for phrase in ("Loudspeaker playback", "ipadflat", "spliced", "not confirmed at 95 %", "factory users only", "100 and 200 Hz",
                   "counterfactual", "df.dll", "rejected before", "compressor step"):
        assert phrase in text, phrase
    assert report["predefined_rule"].endswith("evaluate_voice_setup_arms.py.")
    for phrase in ("Rule (fixed before any study row existed)", "analysis_key", "equality passes", "within budget on this cohort",
                   "S1-S3", "SPEECH_WITHIN_DB"):
        assert phrase in (ev.__doc__ or ""), phrase
    assert "40 dB" not in (ev.speech_labels.__doc__ or "")
    for phrase in ("first whole-setup veto", "fixed whatever the number of\nexclusions", "ran on the injected candidate for at least half"):
        assert phrase in (ev.__doc__ or ""), phrase
    assert "180 analysed user-conditions in 18 speaker clusters" in text


def test_the_report_counts_its_clusters_and_user_conditions_from_the_included_users(frozen_tree):
    rows = [{**r, "excluded": "noise_material"} if r["index"] == 0 else r for r in ev.fabricated_rows()]
    report = ev.build_report({"rows": rows, "identity": {"build": "b"}}, frozen_tree)
    text = " ".join(report["limitations"])
    assert "170 analysed user-conditions in 17 speaker clusters" in text and "18 speaker clusters" not in text
    assert report["metrics"]["users"]["speakers"] == 17 and len(report["metrics"]["users"]["excluded"]) == 11


def test_the_report_refuses_development_rows_and_missing_main_jobs():
    rows = ev.fabricated_rows()
    ev.check_rows(rows)
    with pytest.raises(AssertionError, match="development"):
        ev.check_rows([{**rows[0], "development": True}, *rows[1:]])
    with pytest.raises(AssertionError, match="missing"):
        ev.check_rows(rows[1:])
    ev.check_rows([r for r in rows if r["condition"] != "balcony"])  # balcony rows are optional


def test_the_report_takes_its_hashes_from_the_freeze_and_hashes_nothing_itself(frozen_tree, monkeypatch):
    def boom(*_args):
        raise AssertionError("hashed at report time")

    monkeypatch.setattr(loudness, "_sha256", boom)
    monkeypatch.setattr(gate_corpus, "_sha256", boom)
    report = ev.build_report({"rows": ev.fabricated_rows(), "identity": {"build": "b"}}, frozen_tree)
    assert report["asset_hashes"]["daps_manifest"] == frozen_tree["daps_manifest"]
    assert report["release_assets"]["df.dll"]["same"] is False


def test_the_freeze_covers_the_rule_the_code_and_the_tools_it_imports_and_notices_a_change(frozen_tree):
    assert frozen_tree["rule"] and frozen_tree["python/tools/evaluate_voice_setup_arms.py"]
    assert frozen_tree["python/tools/evaluate_voice_setup_refusals.py"] and frozen_tree["python/mic_eq/analysis/voice_setup.py"]
    assert "python/mic_eq/config_parts/settings.py" in frozen_tree and "daps_manifest" in frozen_tree
    assert ev.RULES_FINAL.read_text(encoding="utf-8") == ev.__doc__
    assert ev.check_frozen(ev.FROZEN, Path(".")) == frozen_tree
    tampered = dict(frozen_tree, **{"python/mic_eq/analysis/voice_setup.py": "0" * 64})
    ev.FROZEN.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(SystemExit, match="voice_setup.py"):
        ev.check_frozen(ev.FROZEN, Path("."))
    with pytest.raises(SystemExit, match="missing"):
        ev.check_frozen(ev.FROZEN.with_name("absent.json"), Path("."))


def test_a_freeze_stays_once_rows_exist_and_a_change_during_a_run_is_caught(frozen_tree, monkeypatch, tmp_path):
    ev.DEFAULT_ROWS.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="the freeze it was made under stays"):
        ev.freeze(ev.FROZEN, ev.RULES_FINAL, ev.DEFAULT_ROWS, tmp_path)
    build = ev.refusals.build_of(frozen_tree)
    ev.check_unchanged(tmp_path, build, [{"build": build}])
    with pytest.raises(SystemExit, match="changed during the run"):
        ev.check_unchanged(tmp_path, build, [{"build": "other"}])
    monkeypatch.setattr(ev, "frozen_hashes", lambda root=None: {**frozen_tree, "df.dll": "9" * 64})
    with pytest.raises(SystemExit, match="changed during the run"):
        ev.check_unchanged(tmp_path, build, [{"build": build}])


def test_report_refuses_a_changed_tree_and_development_rows_and_run_refuses_a_finished_file(frozen_tree, tmp_path):
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps({"development": True, "identity": {"build": "x"}, "rows": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="development rows"):
        ev.main(["report", "--rows", str(rows), "--root", str(tmp_path)])
    rows.write_text(json.dumps({"development": False, "identity": {"build": "x"}, "rows": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="another build"):
        ev.main(["report", "--rows", str(rows), "--root", str(tmp_path)])
    with pytest.raises(SystemExit, match="finished run"):
        ev.DEFAULT_ROWS.write_text("{}", encoding="utf-8")
        ev.main(["run", "--root", str(tmp_path)])
    with pytest.raises(SystemExit, match="pass --development"):
        ev.main(["probe", "--root", str(tmp_path)])


def test_a_partial_file_resumes_only_its_own_build_and_jobs(tmp_path):
    jobs = ev.jobs_of(tmp_path, True)
    assert len(jobs) == 48 and jobs[0][1:] == ("factory", 18, "ipad_office1")
    partial = tmp_path / "rows.run.partial.jsonl"
    row = {"split": "factory", "index": 18, "model": "ipad_office1", "build": "b"}
    partial.write_text(json.dumps(row) + "\n", encoding="utf-8")
    refusals.check_resume(partial, "b", jobs)
    with pytest.raises(SystemExit, match="another build"):
        refusals.check_resume(partial, "other", jobs)
    foreign = {**row, "index": 3}  # a held-out speaker's row in a development run
    partial.write_text(json.dumps(foreign) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="outside this run"):
        refusals.check_resume(partial, "b", jobs)
    other_condition = {**row, "split": "balcony"}
    partial.write_text(json.dumps(other_condition) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="outside this run"):
        refusals.check_resume(partial, "b", jobs)


def test_the_jobs_are_longest_first_with_balcony_last_and_keys_are_unique(tmp_path):
    jobs = ev.jobs_of(tmp_path, False)
    assert [j[1] for j in jobs] == ["factory"] * 90 + ["existing"] * 90 + ["balcony"] * 18
    assert len({(j[1], j[2], j[3]) for j in jobs}) == 198


def test_defaults_are_the_new_files_never_a_historical_report():
    assert ev.DEFAULT_REPORT.name == "voice-setup-arms-2026-10-report.json" and ev.DEFAULT_REPORT.parent.name == "evaluation"
    assert "voice-setup-arms-2026-10" in ev.DETAILS.as_posix() and ev.DAPS.name == "daps_script1"
    assert ev.MAX_WORKERS == 4 and ev.UNBOUNDED_TUNER_RUNTIME in (True, False)


def test_the_tuner_runtime_gate_is_lifted_only_when_the_constant_says_so(monkeypatch):
    from mic_eq.analysis import joint_tuning

    monkeypatch.setattr(ev, "UNBOUNDED_TUNER_RUNTIME", False)
    with ev.tuner_runtime():
        assert joint_tuning._MAX_RUNTIME_FACTOR == 1.0
    monkeypatch.setattr(ev, "UNBOUNDED_TUNER_RUNTIME", True)
    with ev.tuner_runtime():
        assert joint_tuning._MAX_RUNTIME_FACTOR == float("inf")
    assert joint_tuning._MAX_RUNTIME_FACTOR == 1.0


def test_the_fabricated_world_reachability_and_probe_arithmetic():
    rows = ev.fabricated_rows()
    flags = ev.reachability(rows)
    assert all(value for key, value in flags.items() if key != SIBILANT_FLAG)
    assert not flags[SIBILANT_FLAG]
    assert ev.tuner_runtime_factors(rows) == []


def test_reachability_flags_guard_the_gates_they_stand_for():
    def world():
        rows = [r for r in ev.fabricated_rows() if r["condition"] != "balcony"]
        return rows, [r for r in rows if r["condition"] == "factory"]

    tonal = lambda flags: [v for k, v in flags.items() if k.startswith("T cell")]  # noqa: E731
    deep = "R1: a measurable deep pause for half of the factory users"
    for stripped, expected in ((45, True), (46, False)):  # users refused before verification stay in the denominator
        rows, factory = world()
        for row in factory[:stripped]:
            row["injections"] = {}
        assert len(tonal(ev.reachability(rows))) == 16 and all(v is expected for v in tonal(ev.reachability(rows)))
    for lost, expected in ((45, True), (46, False)):
        rows, factory = world()
        for row in factory[:lost]:
            row["arms"]["current"]["metrics"]["E"] = {"whole": {**OK, "deep_pause_db": None}}
        assert ev.reachability(rows)[deep] is expected
    rows, factory = world()
    sibilant = SIBILANT_FLAG
    existing = next(r for r in rows if r["condition"] == "existing")
    existing["arms"]["current"]["metrics"]["E"] = {"whole": {**OK, "sibilant_loss_db": 9.0}}
    assert not ev.reachability(rows)[sibilant]  # R2 counts factory users only
    factory[0]["arms"]["current"]["metrics"]["E"] = {"whole": {**OK, "sibilant_loss_db": 9.0}}
    assert ev.reachability(rows)[sibilant]


def test_a_failed_render_of_the_end_state_is_unjudgeable_and_counted_not_a_crash():
    view = _view()
    view["metrics"]["E"] = {"error": "RuntimeError: render failed"}
    flags = ev.arm_flags(view)
    assert flags["fa"] == flags["fa_upper"] == 0 and refusals.outcome(view, "E")["status"] == "unjudgeable"
    assert np.isnan(ev._reading(view, "deep_pause_db"))
    rows = ev.fabricated_rows()
    rows[0]["arms"]["current"]["metrics"]["E"] = {"error": "RuntimeError: render failed"}
    rows[0]["arms"]["current"]["objects"]["E"] = {"vacuous": False}
    summary = ev.arm_summary(ev.main_rows(rows), "current")
    assert summary["false_accepts"] == 0 and summary["buckets"]["none"] >= 1
    ev.gate_inputs(rows)
    ev.reachability(rows)


def test_the_consistency_checks_skip_a_user_whose_current_analysis_failed(tmp_path):
    row = ev.fabricated_rows()[0]
    failed = {"A": False, "fired": [], "recommended": False, "short_speech": False, "tone_withheld": False, "verification": None,
              "end_state": "current", "objects": {"E": {"vacuous": True}}, "analysis_failure": "Voice capture was too short for setup.",
              "metrics": {"current": {"whole": OK}, "E": {"whole": OK}}}  # no "b1": the analysis never got that far
    row = {**row, "arms": {arm: failed for arm in ev.ARMS}, "construction": {**row["construction"], "A": False}}
    report = ev.consistency(tmp_path, [row])
    assert report["parity"] == [] and report["p_differs_only_with_tone_withheld"] and report["switches_restored"]


# ---------------------------------------------------------------------------------------------------------------
# 5. One development user from the real DAPS files


DAPS_ROOT = Path(os.environ.get("DAPS_SCRIPT1_ROOT") or ev.DAPS)


@pytest.mark.skipif(not (DAPS_ROOT / "MANIFEST.json").exists(), reason="models/daps_script1 is not present")
def test_one_development_user_runs_end_to_end_on_the_real_files(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.chdir(run_dir)
    monkeypatch.setenv("VAD_MODEL_PATH", str(tmp_path / "missing" / "silero_vad.onnx"))  # the VAD is unavailable
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    from mic_eq.analysis import voice_setup

    monkeypatch.setattr(voice_setup, "analyze_voice_setup", functools.partial(voice_setup.analyze_voice_setup, vad_available=False))
    monkeypatch.setattr(refusals, "configure_environment", lambda: None)
    monkeypatch.setattr(refusals, "check_loaded", lambda diagnostics, tuning: {"vad_backends": {}, "tuner_models": {}})
    monkeypatch.setattr(ev, "build_id", lambda root: "test")
    job = (str(DAPS_ROOT), "existing", 18, "ipadflat_confroom1")  # m2, an existing user with a tone EQ
    row = ev.evaluate_user(job, development=True)
    json.dumps(row, default=float)
    assert row["excluded"] is None and row["speaker"] == "m2" and row["tone_entry"] == (18 + 7) % 3
    assert set(row["arms"]) == set(ev.ARMS) and row["injections"] == {} and row["construction"]["A"] is False
    views = row["arms"]
    for arm in ev.ARMS:
        assert views[arm] is views[ev.existing_twin(arm)]  # R is a no-op for an existing user
        assert set(views[arm]["metrics"]) >= {"current", "E"} and views[arm]["metrics"]["current"]["whole"]["clipped"] is False
    assert set(row["runtime"]["analysis"]) == {"current", "P"} and set(row["probes"]) == set(refusals.probe_settings(
        refusals.current_settings(False)))
    assert ev.arm_flags(ev.arm_user(row, "P+T+R")).keys() == set(ev.FLAGS)
    assert ev._switch_values() == {"P": False, "T": False, "R": 1.0}
