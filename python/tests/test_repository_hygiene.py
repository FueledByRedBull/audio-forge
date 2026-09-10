"""Prevent generated payloads and credential files reentering Git history."""

from pathlib import Path
import subprocess


def test_tracked_tree_contains_source_and_compact_evidence_only():
    root = Path(__file__).resolve().parents[2]
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    ).stdout.decode("utf-8").split("\0")
    forbidden_suffixes = {
        ".7z", ".zip", ".gz", ".tar", ".msi", ".exe", ".dll", ".pyd",
        ".pdb", ".onnx", ".wav", ".mp3", ".pem", ".key", ".pfx", ".p12",
    }
    for name in filter(None, tracked):
        path = Path(name)
        assert path.suffix.lower() not in forbidden_suffixes, name
        assert not any(part in {".venv", "target", "dist", "__pycache__"}
                       for part in path.parts), name
        assert path.name not in {"AGENTS.md", "AGENTS.override.md", "GPLAN.md"}, name
        assert not path.name.lower().startswith((".env", "credentials.", "secrets.")), name
        assert (root / path).stat().st_size <= 1_000_000, name


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
