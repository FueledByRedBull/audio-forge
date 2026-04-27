"""Lightweight packaging sanity checks for the portable Windows bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _contains(path: str, needle: str) -> bool:
    return needle in (REPO_ROOT / path).read_text(encoding="utf-8")


def check_source_packaging() -> list[str]:
    errors: list[str] = []

    spec_expectations = [
        ("AudioForge.spec", "DirectML.dll"),
        ("AudioForge.spec", "df.dll"),
        ("AudioForge.spec", "models"),
        ("AudioForge.spec", "mic_eq.ico"),
    ]
    script_expectations = [
        ("build_exe.ps1", "$PSScriptRoot"),
        ("build_exe.ps1", "EXT_SUFFIX"),
        ("build_exe.ps1", "DirectML.dll"),
        ("build_exe.ps1", "DeepFilterNet3_ll_onnx.tar.gz"),
        ("build_exe.ps1", "DeepFilterNet3_onnx.tar.gz"),
        ("build_exe.ps1", "silero_vad.onnx"),
        ("build_exe.ps1", "prune_bundle.py"),
    ]

    for path, needle in [*spec_expectations, *script_expectations]:
        if not _contains(path, needle):
            errors.append(f"{path}: missing expected packaging reference {needle!r}")

    return errors


def check_dist_bundle() -> list[str]:
    dist = REPO_ROOT / "dist" / "AudioForge"
    errors: list[str] = []
    if not (dist / "AudioForge.exe").is_file():
        errors.append("dist/AudioForge/AudioForge.exe is missing")
        return errors

    for filename in ["DirectML.dll", "df.dll"]:
        if not any(path.name == filename for path in dist.rglob(filename)):
            errors.append(f"dist/AudioForge does not contain {filename}")

    models_dir = next((path for path in dist.rglob("models") if path.is_dir()), None)
    if models_dir is None:
        errors.append("dist/AudioForge does not contain a models directory")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Check packaging source files without requiring a built dist/AudioForge bundle.",
    )
    args = parser.parse_args()

    errors = check_source_packaging()
    if not args.source_only:
        errors.extend(check_dist_bundle())

    if errors:
        print("Package smoke check failed:")
        for error in errors:
            print(f"  {error}")
        return 1

    print("Package smoke check passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Package smoke check crashed: {exc}", file=sys.stderr)
        raise SystemExit(1)
