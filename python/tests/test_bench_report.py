"""The bench report tool, end to end on one short mix and on its comparison output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

import bench_metrics as bm
import bench_report
from mic_eq import CORE_AVAILABLE

pytestmark = pytest.mark.skipif(not CORE_AVAILABLE, reason="native extension is not built")

SHORT_S = 4.6  # a phrase and a half of take t01, above the three-second minimum of Voice Setup's speech recording


@pytest.fixture
def short_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A bench folder of one 5 s take and one fan mix, so the whole tool runs in seconds."""
    monkeypatch.setenv("VAD_MODEL_PATH", str(tmp_path / "no-silero.onnx"))  # Silero is not needed or assumed
    full = bm.load_bench().manifest
    size = int(SHORT_S * bm.SAMPLE_RATE)
    take = full["takes"]["t01"]
    pcm = np.round(bm.read_wav(bm.BENCH_DIR / take["file"])[:size] * 32768).astype(np.int16)
    folder = tmp_path / "bench"
    folder.mkdir()
    wavfile.write(folder / "take-short.wav", bm.SAMPLE_RATE, pcm)
    data = (folder / "take-short.wav").read_bytes()
    manifest = {
        "takes": {"short": {**take, "file": "take-short.wav", "sha256": hashlib.sha256(data).hexdigest(),
                            "bytes": len(data), "duration_s": SHORT_S,
                            "speech_samples": [[a, min(b, size)] for a, b in take["speech_samples"] if a < size]}},
        "noises": {},
        "mixes": {"fan-short": {**full["mixes"]["fan-t01"], "take": "short"},
                  "fan-short-again": {**full["mixes"]["fan-t01"], "take": "short"}},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def test_report_runs_end_to_end_and_compares_with_an_earlier_run(short_bench: Path, tmp_path: Path, capsys):
    report, numbers = tmp_path / "bench.md", tmp_path / "bench.json"
    assert bench_report.main(["--bench-dir", str(short_bench), "--mixes", "fan-short", "--output", str(report),
                              "--json-output", str(numbers)]) == 0
    assert not numbers.with_name("bench.json.partial.jsonl").exists()  # a finished run leaves no partial file
    result = json.loads(numbers.read_text(encoding="utf-8"))
    entry = result["mixes"]["fan-short"]
    assert result["build"]["vad"] == "unavailable"
    assert len(result["build"]["native_module_sha256"]) == 64
    assert "git_revision" in result["build"] and "manifest_sha256" in result["build"]
    factory = entry["factory"]
    assert factory["finite"] and not factory["clipped"] and 0.0 < factory["estoi"] <= 1.0 and factory["cpu_s"] > 0.0
    voice = entry["voice_setup"]
    assert voice["outcome"] in {"applied", "refused", "unverified", "failed"}
    assert voice["proposal"]["finite"] and voice["cpu_s"] > 0.0
    assert all(v in bm.VETOES for v in voice["vetoes"])
    text = report.read_text(encoding="utf-8")
    assert "## Factory chain" in text and "## Voice Setup with Auto-EQ" in text and "fan-short" in text
    assert "VAD: unavailable" in text
    # The suppressor each path used, and whether DeepFilter was enabled, are in the report and the numbers.
    assert factory["suppressor"]["model"] == "rnnoise" and voice["suppressor"]["model"] in {"rnnoise", "deepfilter-ll", "deepfilter"}
    assert "deepfilter_enabled" in result["build"]
    assert "Suppressor: factory chain rnnoise x1; Voice Setup chose" in text and "AUDIOFORGE_ENABLE_DEEPFILTER:" in text

    # The same numbers compared with themselves: every delta is zero, and the outcome is unchanged.
    after = bench_report.compare_markdown(result, result)
    assert "Outcomes and steps are unchanged." in after and "| fan-short | +0.000 |" in after

    # A later factory-only run compares against the stored report through the command line.
    assert bench_report.main(["--bench-dir", str(short_bench), "--mixes", "fan-short", "--skip-voice-setup",
                              "--compare", str(numbers)]) == 0
    printed = capsys.readouterr().out
    assert "# Bench comparison (after minus before)" in printed
    assert "| fan-short | +0.000 |" in printed
    assert "Voice Setup" not in printed.split("# Bench comparison")[1]


def test_report_refuses_files_that_differ_from_the_manifest(short_bench: Path):
    wav = short_bench / "take-short.wav"
    wav.write_bytes(wav.read_bytes() + b"\0\0")
    with pytest.raises(SystemExit, match="differ from the manifest"):
        bench_report.main(["--bench-dir", str(short_bench), "--skip-voice-setup"])


def test_unknown_mix_ids_are_rejected(short_bench: Path):
    with pytest.raises(SystemExit):
        bench_report.main(["--bench-dir", str(short_bench), "--mixes", "nope"])


def test_a_bad_compare_file_stops_the_run_before_any_mix_is_rendered(short_bench: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(bench_report, "run", lambda *a, **k: pytest.fail("the mixes should not have run"))
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    for compare in (tmp_path / "missing.json", wrong):
        with pytest.raises(SystemExit):
            bench_report.main(["--bench-dir", str(short_bench), "--skip-voice-setup", "--compare", str(compare)])


def test_a_crash_keeps_the_mixes_that_finished(short_bench: Path, tmp_path: Path, monkeypatch):
    measure, calls = bm.measure_settings, []

    def crash_on_the_second_mix(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated crash")
        return measure(*args, **kwargs)

    monkeypatch.setattr(bm, "measure_settings", crash_on_the_second_mix)
    numbers = tmp_path / "bench.json"
    with pytest.raises(RuntimeError, match="simulated crash"):
        bench_report.main(["--bench-dir", str(short_bench), "--skip-voice-setup", "--json-output", str(numbers)])
    lines = (tmp_path / "bench.json.partial.jsonl").read_text(encoding="utf-8").splitlines()
    assert [list(json.loads(line)) for line in lines] == [["fan-short"]] and not numbers.exists()
