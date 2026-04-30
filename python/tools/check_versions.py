"""Verify release version strings stay in sync."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _single_match(path: str, pattern: str, label: str) -> str:
    match = re.search(pattern, _read(path), re.MULTILINE)
    if not match:
        raise ValueError(f"{label}: version string not found in {path}")
    return match.group(1)


def main() -> int:
    expected = _single_match("pyproject.toml", r'^version\s*=\s*"([^"]+)"', "pyproject")
    checks = {
        "rust-core/Cargo.toml": _single_match(
            "rust-core/Cargo.toml",
            r'^version\s*=\s*"([^"]+)"',
            "rust core",
        ),
        "python/mic_eq/__init__.py": _single_match(
            "python/mic_eq/__init__.py",
            r'^__version__\s*=\s*"([^"]+)"',
            "python package",
        ),
        "README.md": _single_match(
            "README.md",
            r"Current version:\s*`v([^`]+)`",
            "readme",
        ),
        "python/mic_eq/config.py Preset.version": _single_match(
            "python/mic_eq/config.py",
            r'version:\s*str\s*=\s*"([^"]+)"',
            "preset default",
        ),
        "python/mic_eq/config.py built-ins": _single_match(
            "python/mic_eq/config.py",
            r'version="([^"]+)"',
            "built-in preset default",
        ),
        "python/mic_eq/ui/main_window.py auto-eq preset": _single_match(
            "python/mic_eq/ui/main_window.py",
            r'version="([^"]+)"',
            "auto-eq preset default",
        ),
    }

    release_versions = set(re.findall(r"AudioForge-v([0-9]+\.[0-9]+\.[0-9]+)", _read("RELEASING.md")))
    if not release_versions:
        raise ValueError("RELEASING.md: release archive version string not found")
    if release_versions != {expected}:
        checks["RELEASING.md"] = ", ".join(sorted(release_versions))

    mismatches = {
        path: version
        for path, version in checks.items()
        if version != expected
    }
    if mismatches:
        print(f"Version mismatch: pyproject.toml is {expected}")
        for path, version in mismatches.items():
            print(f"  {path}: {version}")
        return 1

    print(f"Version strings are in sync: {expected}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Version check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
