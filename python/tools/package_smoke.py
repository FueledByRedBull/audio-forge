"""Lightweight packaging sanity checks for the portable Windows bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_BUNDLE_FILES = (
    "AudioForge.exe",
    "df.dll",
    "DirectML.dll",
    "models/DeepFilterNet3_ll_onnx.tar.gz",
    "models/DeepFilterNet3_onnx.tar.gz",
    "models/silero_vad.onnx",
)


def _contains(path: str, needle: str) -> bool:
    return needle in (REPO_ROOT / path).read_text(encoding="utf-8")


def _load_asset_manifest() -> tuple[list[dict[str, object]], list[str]]:
    manifest_path = REPO_ROOT / "release-assets.json"
    errors: list[str] = []
    if not manifest_path.is_file():
        return [], ["release-assets.json is missing"]

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [], [f"release-assets.json is invalid JSON: {exc}"]

    assets = raw.get("assets")
    if not isinstance(assets, list) or not assets:
        return [], ["release-assets.json must contain a non-empty assets list"]

    seen_paths: set[str] = set()
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict):
            errors.append(f"release-assets.json assets[{index}] must be an object")
            continue
        path = asset.get("path")
        sha256 = asset.get("sha256")
        source = asset.get("source")
        license_note = asset.get("license")
        if not isinstance(path, str) or not path:
            errors.append(f"release-assets.json assets[{index}].path is required")
            continue
        seen_paths.add(path.replace("\\", "/"))
        bundle_path = asset.get("bundle_path")
        if isinstance(bundle_path, str) and bundle_path:
            seen_paths.add(bundle_path.replace("\\", "/"))
        if not isinstance(sha256, str) or len(sha256) != 64:
            errors.append(f"release-assets.json asset {path} must have a 64-character sha256")
        if not isinstance(source, str) or not source:
            errors.append(f"release-assets.json asset {path} must document source")
        if not isinstance(license_note, str) or not license_note:
            errors.append(f"release-assets.json asset {path} must document license")

    for required in REQUIRED_BUNDLE_FILES[1:]:
        if required not in seen_paths:
            errors.append(f"release-assets.json is missing required asset {required}")

    return assets, errors


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
        ("build_exe.ps1", "verify_release_assets.py"),
        ("build_exe.ps1", "prune_bundle.py"),
    ]

    for path, needle in [*spec_expectations, *script_expectations]:
        if not _contains(path, needle):
            errors.append(f"{path}: missing expected packaging reference {needle!r}")

    if "dist-info" in (REPO_ROOT / "python/tools/prune_bundle.py").read_text(encoding="utf-8"):
        errors.append("python/tools/prune_bundle.py must not prune dependency dist-info metadata")

    _assets, manifest_errors = _load_asset_manifest()
    errors.extend(manifest_errors)

    return errors


def _has_bundle_file(dist: Path, relative_path: str) -> bool:
    normalized = Path(relative_path)
    if (dist / normalized).is_file():
        return True
    return any(
        candidate.is_file() and candidate.as_posix().endswith(relative_path)
        for candidate in dist.rglob(normalized.name)
    )


def _has_dependency_license_metadata(dist: Path) -> bool:
    internal = dist / "_internal"
    if any(path.is_dir() for path in internal.glob("*.dist-info")):
        return True
    license_root = internal / "licenses"
    if license_root.is_dir():
        license_names = {"LICENSE", "LICENSE.txt", "NOTICE", "NOTICE.txt", "COPYING"}
        return any(path.is_file() and path.name in license_names for path in license_root.rglob("*"))
    return False


def check_dist_bundle(dist: Path | None = None) -> list[str]:
    if dist is None:
        dist = REPO_ROOT / "dist" / "AudioForge"
    errors: list[str] = []
    if not dist.is_dir():
        errors.append(f"{dist} is missing")
        return errors

    for relative_path in REQUIRED_BUNDLE_FILES:
        if not _has_bundle_file(dist, relative_path):
            errors.append(f"{dist} does not contain {relative_path}")

    if not any(path.name.startswith("mic_eq_core") and path.suffix == ".pyd" for path in dist.rglob("*.pyd")):
        errors.append(f"{dist} does not contain mic_eq_core*.pyd")

    if not _has_dependency_license_metadata(dist):
        errors.append(f"{dist} does not contain dependency dist-info or collected licenses")

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
