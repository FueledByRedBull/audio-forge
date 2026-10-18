"""The bench set's files, its mix construction and the measures that read a mix."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from scipy.signal import fftconvolve

import bench_metrics as bm

SR = bm.SAMPLE_RATE


@pytest.fixture(scope="module")
def bench() -> bm.Bench:
    return bm.load_bench()


def test_manifest_hashes_match_the_committed_files():
    manifest = bm.load_bench().manifest
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
        mix = bm.build_mix(bench, mix_id)
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
    levels = {m: bm.active_level_db(bm.build_mix(bench, m)["raw"].astype(float)) for m in ("hot-t05", "quiet-t10")}
    assert levels["hot-t05"] > levels["quiet-t10"] + 20.0


def test_requested_snr_is_met_in_the_noise_only_stretch(bench):
    recipe = bench.manifest["mixes"]["fan-t01"]
    mix = bm.build_mix(bench, "fan-t01")
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
