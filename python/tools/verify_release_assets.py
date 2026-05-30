"""Verify local binary/model assets against release-assets.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "release-assets.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    assets = raw.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("manifest must contain a non-empty assets list")
    return assets


def _validate_manifest_path(raw_path: str, field_name: str) -> str | None:
    local_path = Path(raw_path)
    posix_path = PurePosixPath(raw_path.replace("\\", "/"))
    windows_path = PureWindowsPath(raw_path)
    if (
        local_path.is_absolute()
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
    ):
        return f"{field_name} must be repository-relative, got absolute path {raw_path}"
    if ".." in posix_path.parts or ".." in windows_path.parts:
        return f"{field_name} must not contain '..' traversal, got {raw_path}"
    return None


def verify_assets(manifest_path: Path = MANIFEST_PATH) -> list[str]:
    errors: list[str] = []
    for asset in _load_manifest(manifest_path):
        if not isinstance(asset, dict):
            errors.append("asset entry must be an object")
            continue
        raw_path = asset.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            errors.append("asset entry missing path")
            continue
        if path_error := _validate_manifest_path(raw_path, "asset path"):
            errors.append(path_error)
            continue

        raw_bundle_path = asset.get("bundle_path")
        if isinstance(raw_bundle_path, str) and raw_bundle_path:
            if path_error := _validate_manifest_path(raw_bundle_path, "asset bundle_path"):
                errors.append(path_error)

        path = REPO_ROOT / raw_path.replace("\\", "/")
        if not path.is_file():
            errors.append(f"{raw_path}: missing")
            continue

        expected_size = asset.get("size")
        if isinstance(expected_size, int) and path.stat().st_size != expected_size:
            errors.append(
                f"{raw_path}: size mismatch, expected {expected_size}, got {path.stat().st_size}"
            )

        expected_sha = asset.get("sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            errors.append(f"{raw_path}: manifest sha256 must be a 64-character hex string")
            continue

        actual_sha = _sha256(path)
        if actual_sha.lower() != expected_sha.lower():
            errors.append(f"{raw_path}: sha256 mismatch, expected {expected_sha}, got {actual_sha}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Path to release asset manifest.",
    )
    args = parser.parse_args()

    errors = verify_assets(args.manifest.resolve())
    if errors:
        print("Release asset verification failed:")
        for error in errors:
            print(f"  {error}")
        return 1

    print("Release assets verified")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Release asset verification crashed: {exc}", file=sys.stderr)
        raise SystemExit(1)
