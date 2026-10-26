"""The bench set's files, its mix construction and the measures that read a mix."""

from __future__ import annotations

import functools
import hashlib
from copy import deepcopy

import numpy as np
import pytest
from scipy.signal import fftconvolve

import bench_metrics as bm
from mic_eq import CORE_AVAILABLE
from mic_eq.ui.calibration_support import filtered_capture_for_analysis

SR = bm.SAMPLE_RATE


@pytest.fixture(scope="module")
def bench() -> bm.Bench:
    return bm.load_bench()


@functools.cache
def built_mix(mix_id: str) -> dict[str, np.ndarray]:
    """A bench mix built once for the whole run; its arrays are read-only. Tests that need their own build call
    ``bm.build_mix``."""
    mix = bm.build_mix(bm.load_bench(), mix_id)
    for audio in mix.values():
        audio.setflags(write=False)
    return mix


def test_manifest_hashes_match_the_committed_files():
    manifest = bm.read_manifest()
    files = {**manifest["takes"], **manifest["noises"]}
    for entry in files.values():
        path = bm.BENCH_DIR / entry["file"]
        assert path.stat().st_size == entry["bytes"] < 1_000_000, entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry["file"]
    committed = {p.name for p in bm.BENCH_DIR.iterdir()}
    assert committed == {"manifest.json", *(e["file"] for e in files.values())}
    assert sum(p.stat().st_size for p in bm.BENCH_DIR.iterdir()) <= 12_000_000
    for entry in manifest["takes"].values():
        for row in entry["utterances"]:
            assert len(row["member_sha256"]) == len(row["transcript_sha256"]) == 64


def test_takes_are_long_enough_to_read_pauses_and_cover_both_speakers(bench):
    manifest = bench.manifest
    assert len(manifest["takes"]) == 10
    assert {t["speaker"] for t in manifest["takes"].values()} == {"p232", "p257"}
    assert {t["gender"] for t in manifest["takes"].values()} == {"male", "female"}
    assert sum(t["content"].startswith("sibilant") for t in manifest["takes"].values()) >= 2
    for name, entry in manifest["takes"].items():
        audio = bench.takes[name]
        assert 8.0 <= audio.size / SR <= 12.0 and audio.size / SR == pytest.approx(entry["duration_s"], abs=1e-3)
        speech = entry["speech_samples"]
        gaps = [(b[0] - a[1]) / SR for a, b in zip(speech, speech[1:])]
        assert sum(gap >= 0.7 for gap in gaps) >= 2, name
        # Pause-noise measures read only frames at least 600 ms from speech.
        assert bm.deep_pause_seconds(bm.speech_labels(entry, audio.size)) >= 0.5, name
        assert np.max(np.abs(audio[~bm.speech_labels(entry, audio.size)])) == 0.0, name  # real gaps are digital silence


def test_every_condition_of_the_set_is_present(bench):
    kinds = {r["noise"]["kind"] for r in bench.manifest["mixes"].values()}
    conditions = {r["condition"] for r in bench.manifest["mixes"].values()}
    assert kinds >= {"fan", "keyboard", "real", "hum", "tv", "music", "babble", "pink"}
    assert conditions >= {"fan", "keyboard", "cafe", "traffic", "hum50", "hum60", "reverb", "farmic", "hot", "quiet",
                          "clipped", "tv", "music", "babble"}
    assert {r["noise"]["hz"] for r in bench.manifest["mixes"].values() if r["noise"]["kind"] == "hum"} == {50.0, 60.0}
    assert {r["noise"]["source"] for r in bench.manifest["mixes"].values() if r["noise"]["kind"] == "real"} <= set(
        bench.manifest["noises"])
    for recipe in bench.manifest["mixes"].values():
        assert recipe["take"] in bench.manifest["takes"]
        assert recipe["take"] not in recipe["noise"].get("takes", ())


def test_mix_construction_is_deterministic(bench):
    for mix_id in bench.manifest["mixes"]:
        first, second = bm.build_mix(bench, mix_id), bm.build_mix(bench, mix_id)
        for key in ("raw", "clean", "labels", "noise_capture"):
            assert first[key].tobytes() == second[key].tobytes(), (mix_id, key)


def test_mixes_have_the_levels_and_lengths_their_recipes_state(bench):
    for mix_id, recipe in bench.manifest["mixes"].items():
        mix = built_mix(mix_id)
        assert mix["raw"].shape == mix["clean"].shape == mix["labels"].shape == bench.takes[recipe["take"]].shape
        assert mix["noise_capture"].size == int(bm.NOISE_S * SR)
        assert np.isfinite(mix["raw"]).all()
        post = recipe.get("post") or {}
        peak = float(np.max(np.abs(mix["raw"])))
        if post.get("clip"):
            assert peak == 1.0 and np.count_nonzero(np.abs(mix["raw"]) >= 1.0) > 100
        else:
            assert peak <= 0.98, mix_id
        if "peak" in post and not post.get("clip"):
            assert peak == pytest.approx(post["peak"], abs=1e-3)
    levels = {m: bm.active_level_db(built_mix(m)["raw"].astype(float)) for m in ("hot-t05", "quiet-t10")}
    assert levels["hot-t05"] > levels["quiet-t10"] + 20.0


def test_requested_snr_is_met_in_the_noise_only_stretch(bench):
    recipe = bench.manifest["mixes"]["fan-t01"]
    mix = built_mix("fan-t01")
    noise = mix["raw"].astype(float) - mix["clean"]
    snr = bm.active_level_db(mix["clean"]) - 10 * np.log10(np.mean(noise**2))
    assert snr == pytest.approx(recipe["noise"]["snr_db"], abs=0.1)


def test_room_impulse_has_the_stated_decay_and_direct_ratio():
    rir = bm.room_impulse(0.6, 0.0, 1)
    assert rir.size == int(0.6 * SR) and rir[0] == 1.0
    assert np.sum(rir[1:] ** 2) == pytest.approx(1.0, rel=1e-6)
    assert np.sum(bm.room_impulse(0.35, -4.0, 1)[1:] ** 2) == pytest.approx(10 ** 0.4, rel=1e-6)
    energy = np.cumsum(rir[::-1] ** 2)[::-1]  # Schroeder decay
    decay = 10 * np.log10(energy / energy[0])
    assert -35.0 > decay[int(0.45 * SR)] > -55.0  # the tail has fallen by about 45 dB at three quarters of RT60


def test_estoi_and_si_sdr_reward_a_clean_estimate():
    rng = np.random.default_rng(0)
    voice = np.convolve(rng.standard_normal(3 * SR), np.ones(8) / 8, "same") * np.abs(
        np.sin(np.linspace(0, 40, 3 * SR)))
    noisy = voice + 0.5 * rng.standard_normal(voice.size) * voice.std()
    assert bm.estoi(voice, voice, SR) == pytest.approx(1.0, abs=1e-6)
    assert bm.estoi(voice, noisy, SR) < bm.estoi(voice, voice + 0.05 * (noisy - voice), SR) < 1.0
    assert bm.si_sdr(voice, 3.0 * voice) > 100.0  # scale invariant
    assert bm.si_sdr(voice, noisy) < 10.0


def test_pause_tail_and_onset_readings_see_what_the_output_does():
    rng = np.random.default_rng(1)
    labels = np.zeros(6 * SR, bool)
    labels[SR : 2 * SR] = labels[4 * SR : 5 * SR] = True
    voice = np.where(labels, rng.standard_normal(labels.size), 0.0)
    noisy = voice + 0.01 * rng.standard_normal(labels.size)
    near = fftconvolve(labels, np.ones(int(0.65 * SR)), "same") > 0.5  # within 650 ms of speech
    case = {"clean": voice, "noisy": noisy, "labels": labels}
    quieter_far = noisy * np.where(near, 1.0, 10 ** (-12 / 20))  # only the deep pauses get quieter
    result = bm.score(case, quieter_far)
    assert result["deep_pause_db"] < -3.0 and result["tail_db"] > -1.0
    assert result["estoi"] > 0.9
    assert bm.score(case, noisy * np.where(labels, 1.0, 10 ** (-12 / 20)))["tail_db"] == pytest.approx(-12.0, abs=0.1)
    # level matching removes a pure gain from the pause readings
    assert bm.scored(case, noisy * 0.25)["deep_pause_db"] == pytest.approx(0.0, abs=0.01)


def test_deep_pause_seconds_counts_only_frames_600_ms_from_speech():
    labels = np.zeros(10 * SR, bool)
    labels[SR : 2 * SR] = labels[5 * SR : 6 * SR] = True
    # lead 0-0.41 s, gap 2.59-4.41 s, trail 6.59-10 s
    assert bm.deep_pause_seconds(labels) == pytest.approx(0.41 + 1.82 + 3.41, abs=0.02)
    assert bm.deep_pause_seconds(np.zeros(SR, bool)) == pytest.approx(1.0)


def test_tone_error_is_zero_for_the_same_audio_and_grows_with_a_tilt():
    rng = np.random.default_rng(2)
    clean = rng.standard_normal(4 * SR) * 0.1
    labels = np.ones(clean.size, bool)
    tilted = np.fft.irfft(np.fft.rfft(clean) * np.linspace(1.0, 0.1, clean.size // 2 + 1), clean.size)
    assert bm.tone_error_db(clean, clean, labels) == pytest.approx(0.0, abs=1e-6)
    assert bm.tone_error_db(clean, tilted, labels) > 1.0
    assert np.isnan(bm.tone_error_db(clean[: SR // 4], clean[: SR // 4], labels[: SR // 4]))


def test_veto_bookkeeping_matches_the_retired_rule():
    assert bm.reason_veto("compressor p95 gain reduction target missed") == "B3"
    assert bm.reason_veto("too little VAD-active speech") == "B2"
    assert bm.reason_veto("unrelated") is None
    fired = bm.fired_vetoes(["offline DSP validation did not pass", "Auto-EQ abstained from this capture"], False, False,
                            "retain_incumbent", "rollback")
    assert fired == ["B4", "B5", "B6", "C1"]
    assert bm.first_whole(fired, True) == "B4"
    assert bm.first_whole(["C1", "B6"], False) is None  # verification of an unrecommended setup is not a refusal
    assert bm.first_whole(["C1"], True) == "C1"
    assert bm.fired_vetoes([], True, True, None, None) == ["B1", "B2"]


def test_a_failed_render_is_recorded_and_a_failed_measure_is_not_hidden_as_one(monkeypatch):
    ctx = {"raw": np.zeros(8, np.float32)}

    def failing_render(*_args, **_kwargs):
        raise RuntimeError("no native chain")

    monkeypatch.setattr(bm, "render", failing_render)
    assert bm.measure_settings({}, ctx) == {"error": "RuntimeError: no native chain"}

    def broken_measure(*_args, **_kwargs):
        raise ValueError("metric bug")

    monkeypatch.setattr(bm, "render", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(bm, "measure", broken_measure)
    with pytest.raises(ValueError, match="metric bug"):
        bm.measure_settings({}, ctx)


def _scripted_verification(monkeypatch, outcomes):
    """Make the bench's verification return ``outcomes`` in turn; returns the settings each verdict was read on."""
    from mic_eq.analysis import voice_setup

    seen, script = [], iter(outcomes)

    def verdict(*args, **kwargs):
        seen.append(deepcopy(args[4]["compressor_settings"]))
        return next(script)

    monkeypatch.setattr(voice_setup, "validate_voice_setup_verification", verdict)
    return seen


def _proposal(**compressor):
    settings = bm.current_settings(False)
    settings["compressor_settings"].update(compressor)
    return settings


def _captures():
    silence = np.zeros(SR, np.float32)
    return {"noise": silence, "setup": silence, "raw_noise": silence, "takes": [(silence, silence)]}


def _reduce(pressure):
    return {"decision": "reduce", "reduction_targets": ["limiter"], "limiter_pressure_db": pressure}


def test_the_bench_reduces_a_target_the_way_the_dialog_does(monkeypatch):
    # An auto target above the control's range comes back to its top, not to one step below where it was.
    seen = _scripted_verification(monkeypatch, [_reduce(5.0), {"decision": "accept"}])
    flow = bm.verify_flow(_proposal(auto_makeup_enabled=True, target_lufs=-10.0), _captures())
    assert (flow["decision"], flow["adjustments"], flow["ended"]) == ("accept", 1, None)
    assert [c["target_lufs"] for c in seen] == [-10.0, -12.0]


def test_the_bench_stops_where_the_dialog_stops(monkeypatch):
    _scripted_verification(monkeypatch, [_reduce(5.0), _reduce(4.0), _reduce(3.0), _reduce(2.0)])
    flow = bm.verify_flow(_proposal(auto_makeup_enabled=True, target_lufs=-18.0), _captures())
    assert (flow["decision"], flow["adjustments"], flow["ended"]) == ("gave_up", 3, "three_adjustments")

    _scripted_verification(monkeypatch, [_reduce(5.0), _reduce(5.0)])
    flow = bm.verify_flow(_proposal(auto_makeup_enabled=True, target_lufs=-18.0), _captures())
    assert (flow["decision"], flow["adjustments"], flow["ended"]) == ("gave_up", 1, "no_help")

    _scripted_verification(monkeypatch, [{"decision": "reduce", "reduction_targets": []}])
    assert bm.verify_flow(_proposal(), _captures())["ended"] == "no_target"


def _reading(**changes):
    return {"estoi": 0.95, "tone_error_db": 1.0, "sibilant_loss_db": 2.0, "deep_pause_db": -20.0, "tail_db": -15.0,
            "onset_db": -5.0, "clipped": False, "limiter_overload": False, **changes}


@pytest.mark.parametrize(
    ("changes", "clause", "result"),
    [
        ({"estoi": 0.9399}, "estoi", "fail"), ({"estoi": 0.9401}, "estoi", "pass"),
        ({"tone_error_db": 2.01}, "tone_error_db", "fail"), ({"tone_error_db": 1.99}, "tone_error_db", "pass"),
        ({"sibilant_loss_db": 5.01}, "sibilant_loss_db", "fail"), ({"sibilant_loss_db": 4.99}, "sibilant_loss_db", "pass"),
        ({"deep_pause_db": -18.99}, "deep_pause_db", "fail"), ({"deep_pause_db": -19.01}, "deep_pause_db", "pass"),
        ({"tail_db": -13.99}, "tail_db", "fail"), ({"tail_db": -14.01}, "tail_db", "pass"),
        # Onset is the one measure where larger is better.
        ({"onset_db": -6.01}, "onset_db", "fail"), ({"onset_db": -5.99}, "onset_db", "pass"), ({"onset_db": 9.0}, "onset_db", "pass"),
        ({"onset_db": float("nan")}, "onset_db", "unmeasurable"), ({"tail_db": None}, "tail_db", "unmeasurable"),
        ({"clipped": True}, "overload", "fail"), ({"limiter_overload": True}, "overload", "fail"),
    ],
)
def test_a_clause_fails_beyond_its_tolerance_and_in_its_own_direction(changes, clause, result):
    assert bm.clause_results(_reading(**changes), _reading())[clause] == result


def test_an_overload_the_current_settings_already_have_does_not_count_against_a_setup():
    both = _reading(clipped=True, limiter_overload=True)
    assert bm.clause_results(both, both)["overload"] == "pass"
    assert bm.failed_clauses(both, _reading()) == ["overload"]
    assert bm.failed_clauses(None, _reading()) == []  # a setup that was not rendered fails nothing
    assert set(bm.clause_results({"error": "RuntimeError: x"}, _reading()).values()) == {"unmeasurable"}


# A compressor a little deeper than the factory one (-20 dB, 4:1): ordinary compression of a voice.
GOOD_COMPRESSOR = {"enabled": True, "threshold_db": -23.0, "ratio": 4.0, "attack_ms": 8.0, "release_ms": 200.0,
                   "makeup_gain_db": 0.0, "adaptive_release": False, "auto_makeup_enabled": False,
                   "sidechain_highpass_enabled": True}


@pytest.mark.skipif(not CORE_AVAILABLE, reason="native extension is not built")
def test_the_compression_aware_reading_judges_compression_for_what_it_does_beyond_the_asked_for_gain_reduction():
    ctx = bm.make_context(built_mix("room-t02"))
    current = bm.current_settings(False)
    arms = bm.compressor_arms(current, GOOD_COMPRESSOR, ctx, False)
    plain, aware = bm.measure_settings(current, ctx), bm.fair_settings_readings(current, ctx, False)

    # Read plain, a good compressor lifts the pauses and the tail, and taking the compressor away fails the onset
    # clause: the clauses track how much the compressor compresses, not whether it harms.
    assert {"deep_pause_db", "tail_db"} <= set(bm.failed_clauses(arms["step"]["plain"], plain))
    assert "onset_db" in bm.failed_clauses(arms["off"]["plain"], plain)
    # Read against each setup's own compression, both pass...
    assert bm.failed_clauses(arms["step"]["fair"], aware) == []
    assert bm.failed_clauses(arms["off"]["fair"], aware) == []
    # ...and a compressor that squashes the voice still fails.
    assert "estoi" in bm.failed_clauses(arms["squash"]["fair"], aware)
    # With no compressor there is nothing to take out of the reading.
    assert arms["off"]["fair"] == arms["off"]["plain"]


def test_a_retake_is_two_seconds_of_the_noise_that_played_during_the_take():
    mix = built_mix("tv-t06")
    ctx = bm.make_context(mix)
    noise = mix["raw"].astype(np.float64) - mix["clean"]

    assert len(ctx["retakes"]) == len(bm.RETAKE_STARTS_S) == 3
    assert all(take.shape == (int(bm.NOISE_S * SR),) and take.dtype == np.float32 for take in ctx["retakes"])
    # The voice is taken out of the take exactly, so a retake holds the distractor alone, filtered as the captures are.
    first = noise[int(3.0 * SR) : int(5.0 * SR)]
    assert np.allclose(ctx["retakes"][0], filtered_capture_for_analysis(first.astype(np.float32), SR), atol=1e-5)
    assert np.std(ctx["retakes"][0]) > 0.0
    short = bm.make_context({**mix, "raw": mix["raw"][: 6 * SR], "clean": mix["clean"][: 6 * SR],
                             "labels": mix["labels"][: 6 * SR]})
    assert len(short["retakes"]) == 1  # only the stretch that fits
