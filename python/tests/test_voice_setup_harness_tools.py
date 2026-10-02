"""Contracts for the simulated-user Voice Setup harness."""

from __future__ import annotations

import numpy as np

import evaluate_voice_setup as harness

BASE = {"estoi": 0.80, "onset_db": -1.0, "pause_db": -10.0, "deep_pause_db": -20.0, "tail_db": -6.0}


def test_dominance_needs_a_material_gain_and_no_material_loss():
    assert harness.dominates(BASE | {"pause_db": -12.0}, BASE)
    assert not harness.dominates(BASE | {"pause_db": -10.5}, BASE)  # within tolerance
    # Quieter pauses do not buy clipped onsets.
    assert not harness.dominates(BASE | {"pause_db": -14.0, "onset_db": -3.0}, BASE)
    assert not harness.dominates(BASE | {"estoi": float("nan"), "pause_db": -14.0}, BASE)


def test_dominance_counts_the_noise_floor_in_long_pauses():
    # Gate off: better onsets, but the room noise between phrases is 10 dB louder.
    assert not harness.dominates(BASE | {"onset_db": 4.0, "deep_pause_db": -10.0}, BASE)
    # No pause long enough in either capture: the metric is skipped, not failed.
    no_deep = {"deep_pause_db": float("nan")}
    assert harness.dominates(BASE | no_deep | {"onset_db": 4.0}, BASE | no_deep)


def test_splits_are_disjoint_in_speakers_and_noise():
    manifest = {
        "speakers": {f"p{i:03d}": {"gender": "female" if i % 2 else "male"} for i in range(12)},
        "noise": {f"ENV{i}": {} for i in range(6)},
    }
    parts = harness.splits(manifest)
    for field in ("speakers", "environments"):
        seen = [item for part in parts.values() for item in part[field]]
        assert len(seen) == len(set(seen))
    assert {manifest["speakers"][s]["gender"] for s in parts["test"]["speakers"]} == {"female", "male"}


def test_room_response_starts_with_direct_sound_and_decays():
    response = harness.room_response(np.random.default_rng(3))
    assert response[0] == 1.0
    energy = np.square(response[int(0.03 * harness.SAMPLE_RATE):])
    half = energy.size // 2
    assert energy[:half].sum() > 10 * energy[half:].sum()


def test_onset_proxy_reads_gate_clipping_but_not_quieter_pauses():
    rng = np.random.default_rng(0)
    frame = harness.gate_corpus.FRAME
    ungated = 0.001 * rng.standard_normal(300 * frame)  # suppressed noise floor
    ungated[100 * frame : 250 * frame] += 0.1 * rng.standard_normal(150 * frame)
    quieter_pauses = ungated.copy()
    quieter_pauses[: 100 * frame] *= 0.1
    quieter_pauses[250 * frame :] *= 0.1
    clipped = quieter_pauses.copy()
    clipped[100 * frame : 110 * frame] *= 10 ** (-6 / 20)

    assert abs(harness.onset_proxy_db(quieter_pauses, ungated)) < 0.1
    assert harness.onset_proxy_db(clipped, ungated) < -3.0


def _row(job):
    return {"split": job[1], "index": job[2], "model": job[3], "fresh": True}


def test_run_resumes_from_saved_rows(tmp_path):
    partial = tmp_path / "rows.partial.jsonl"
    partial.write_text('{"split": "fit", "index": 0, "model": "rnnoise", "fresh": false}\n', encoding="utf-8")
    jobs = [("root", "fit", index, "rnnoise") for index in range(2)]

    rows = harness.run(_row, jobs, 1, partial)

    assert [(row["index"], row["fresh"]) for row in rows] == [(0, False), (1, True)]
    assert len(partial.read_text(encoding="utf-8").splitlines()) == 2


def test_report_counts_harm_and_missed_improvement():
    better = BASE | {"pause_db": -13.0}
    rows = [{
        "model": "rnnoise", "speaker": f"p{i}", "current": "a", "chosen": "b",
        "candidates": ["a", "b", "c"],
        "arms": {"a": {"metrics": better}, "b": {"metrics": BASE}, "c": {"metrics": better}},
    } for i in range(4)]
    summary = harness.report(rows)["rnnoise"]
    assert summary["harmed"][0] == 1.0
    assert summary["missed"][0] == 1.0
    assert summary["improved"][0] == 0.0
