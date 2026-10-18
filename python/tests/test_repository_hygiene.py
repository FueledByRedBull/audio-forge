"""Prevent generated payloads and credential files reentering Git history."""

from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
# Public speech and noise inputs for the bench, not generated payloads; the size cap still applies.
BENCH_WAV_DIR = ("python", "tests", "fixtures", "bench")


def check_tracked(names, root=ROOT):
    forbidden_suffixes = {
        ".7z", ".zip", ".gz", ".tar", ".msi", ".exe", ".dll", ".pyd",
        ".pdb", ".onnx", ".wav", ".mp3", ".pem", ".key", ".pfx", ".p12",
    }
    for name in filter(None, names):
        path = Path(name)
        bench_wav = path.suffix.lower() == ".wav" and path.parts[:-1] == BENCH_WAV_DIR
        assert bench_wav or path.suffix.lower() not in forbidden_suffixes, name
        assert not any(part in {".venv", "target", "dist", "__pycache__"}
                       for part in path.parts), name
        assert path.name not in {"AGENTS.md", "AGENTS.override.md", "GPLAN.md"}, name
        assert not path.name.lower().startswith((".env", "credentials.", "secrets.")), name
        assert (root / path).stat().st_size <= 1_000_000, name


def test_tracked_tree_contains_source_and_compact_evidence_only():
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    ).stdout.decode("utf-8").split("\0")
    check_tracked(tracked)


def test_wav_is_allowed_only_in_the_bench_folder_under_the_size_cap(tmp_path):
    for parts in (BENCH_WAV_DIR, ("python", "tests", "fixtures", "other"), ("docs",)):
        folder = tmp_path.joinpath(*parts)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "a.wav").write_bytes(b"\0" * 10)
    big = tmp_path.joinpath(*BENCH_WAV_DIR) / "big.wav"
    big.write_bytes(b"\0" * 1_000_001)
    bench = "/".join((*BENCH_WAV_DIR, "a.wav"))
    check_tracked([bench], tmp_path)
    for rejected in ("/".join((*BENCH_WAV_DIR, "big.wav")), "python/tests/fixtures/other/a.wav",
                     "docs/a.wav", "python/tests/fixtures/bench/sub/a.wav"):
        with pytest.raises(AssertionError):
            check_tracked([rejected], tmp_path)


def test_shared_ignore_rules_allow_tests_and_project_docs(tmp_path):
    root = Path(__file__).resolve().parents[2]
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_bytes((root / ".gitignore").read_bytes())
    result = subprocess.run(
        ["git", "-c", "core.excludesFile=", "check-ignore", "--no-index",
         "python/tests/test_sample.py", "tools/test_sample.py",
         "rust-integration/test_sample.py", "ROADMAP.md", "future-dsp.md"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 1, result.stderr
    assert result.stdout == ""
