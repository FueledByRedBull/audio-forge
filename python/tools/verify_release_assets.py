"""Verify local binary/model assets against release-assets.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "release-assets.json"
SOURCE_BUILD_STATUS = "verified-source-build"
HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
PINNED_ARCHIVE_STATUS = "verified-upstream-archive"
PINNED_ARCHIVE_HOSTS = frozenset(
    {
        "github.com",
        "github-releases.githubusercontent.com",
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
    }
)
KNOWN_ORIGIN_STATUSES = frozenset(
    {
        SOURCE_BUILD_STATUS,
        PINNED_ARCHIVE_STATUS,
        "verified-upstream-git-blob",
        "pinned-upstream-model",
    }
)
BLOCKED_ORIGIN_STATUSES = frozenset(
    {
        "inherited-binary-build-identity-unresolved",
        "license-restricted",
        "proprietary",
        "distribution-blocked",
    }
)
KNOWN_ORIGIN_STATUSES |= BLOCKED_ORIGIN_STATUSES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AssetManifest:
    """Validated release manifest entries shared by live asset consumers."""

    fallback_release_tag: str | None
    assets: tuple[dict[str, Any], ...]
    entries: dict[str, dict[str, Any]]


def load_asset_manifest(
    path: Path = MANIFEST_PATH,
    *,
    require_assets: bool = True,
    require_metadata: bool = False,
) -> AssetManifest:
    """Load and validate the live release asset manifest once."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"manifest could not be read: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("manifest root must be an object")

    errors: list[str] = []
    fallback_tag = raw.get("fallback_release_tag")
    if fallback_tag is not None and (
        not isinstance(fallback_tag, str) or not fallback_tag.startswith("v") or not fallback_tag[1:]
    ):
        errors.append("fallback_release_tag must be a non-empty v-prefixed string")

    raw_assets = raw.get("assets")
    if not isinstance(raw_assets, list):
        errors.append("manifest must contain an assets list")
        raw_assets = []
    elif require_assets and not raw_assets:
        errors.append("manifest must contain a non-empty assets list")

    entries: dict[str, dict[str, Any]] = {}
    bundle_paths: dict[str, str] = {}
    validated_assets: list[dict[str, Any]] = []
    for index, asset in enumerate(raw_assets):
        if not isinstance(asset, dict):
            errors.append(f"assets[{index}] must be an object")
            continue
        raw_path = asset.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            errors.append(f"assets[{index}].path is required")
            continue
        normalized_path = raw_path.replace("\\", "/")
        if path_error := _validate_manifest_path(raw_path, "asset path"):
            errors.append(path_error)
        if normalized_path in entries:
            errors.append(f"manifest repeats asset path {raw_path}")

        raw_bundle_path = asset.get("bundle_path")
        normalized_bundle_path: str | None = None
        if raw_bundle_path is not None:
            if not isinstance(raw_bundle_path, str) or not raw_bundle_path:
                errors.append(f"{raw_path}: bundle_path must be a non-empty string")
            else:
                normalized_bundle_path = raw_bundle_path.replace("\\", "/")
                if path_error := _validate_manifest_path(raw_bundle_path, "asset bundle_path"):
                    errors.append(path_error)
                previous = bundle_paths.get(normalized_bundle_path)
                if previous is not None:
                    errors.append(
                        f"manifest repeats bundle_path {raw_bundle_path} for {previous} and {raw_path}"
                    )

        expected_sha = asset.get("sha256")
        if not isinstance(expected_sha, str) or not HEX_SHA256.fullmatch(expected_sha):
            errors.append(f"{raw_path}: manifest sha256 must be a 64-character hex string")

        expected_size = asset.get("size")
        if expected_size is not None and (
            type(expected_size) is not int or expected_size <= 0
        ):
            errors.append(f"{raw_path}: manifest size must be a positive integer")

        for field in ("source", "license"):
            value = asset.get(field)
            if value is not None and (not isinstance(value, str) or not value):
                errors.append(f"{raw_path}: manifest {field} must be a non-empty string")
            elif require_metadata and not isinstance(value, str):
                errors.append(f"{raw_path}: manifest {field} is required")

        origin = asset.get("origin")
        if origin is not None:
            if not isinstance(origin, dict):
                errors.append(f"{raw_path}: origin must be an object")
            else:
                status = origin.get("status")
                if not isinstance(status, str) or not status:
                    errors.append(f"{raw_path}: origin.status must be a non-empty string")
                elif status not in KNOWN_ORIGIN_STATUSES:
                    errors.append(f"{raw_path}: unsupported origin.status {status!r}")
                if status == PINNED_ARCHIVE_STATUS:
                    errors.extend(_verify_pinned_archive_metadata(asset, raw_path))
                elif status == SOURCE_BUILD_STATUS:
                    attestation_path = origin.get("attestation_path")
                    if not isinstance(attestation_path, str) or not attestation_path:
                        errors.append(
                            f"{raw_path}: source-built asset must declare origin.attestation_path"
                        )
                    elif path_error := _validate_manifest_path(
                        attestation_path, "asset attestation_path"
                    ):
                        errors.append(path_error)
                    provenance_path = origin.get("provenance")
                    if provenance_path is not None:
                        if not isinstance(provenance_path, str) or not provenance_path:
                            errors.append(f"{raw_path}: asset provenance path is invalid")
                        elif path_error := _validate_manifest_path(
                            provenance_path, "asset provenance"
                        ):
                            errors.append(path_error)

        if normalized_path not in entries:
            entries[normalized_path] = asset
        if normalized_bundle_path is not None and normalized_bundle_path not in bundle_paths:
            bundle_paths[normalized_bundle_path] = raw_path
        validated_assets.append(asset)

    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors))
    return AssetManifest(
        fallback_release_tag=fallback_tag if isinstance(fallback_tag, str) else None,
        assets=tuple(validated_assets),
        entries=entries,
    )


def _load_manifest(path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    return list(load_asset_manifest(path).assets)


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


def _read_json(path: Path, label: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"{label}: could not read JSON: {exc}"
    if not isinstance(raw, dict):
        return None, f"{label}: JSON root must be an object"
    return raw, None


def _verify_pinned_archive_metadata(asset: dict[str, Any], raw_path: str) -> list[str]:
    origin = asset.get("origin")
    if not isinstance(origin, dict):
        return [f"{raw_path}: pinned archive asset must contain an origin object"]
    errors: list[str] = []
    source = asset.get("source")
    if not isinstance(source, str) or not source:
        errors.append(f"{raw_path}: pinned archive asset must declare source URL")
    else:
        parsed = urllib.parse.urlsplit(source)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in PINNED_ARCHIVE_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in {None, 443}
            or parsed.fragment
        ):
            errors.append(f"{raw_path}: pinned archive source URL is not trusted GitHub HTTPS")
    archive_sha = origin.get("archive_sha256")
    if not isinstance(archive_sha, str) or not HEX_SHA256.fullmatch(archive_sha):
        errors.append(f"{raw_path}: pinned archive SHA-256 is invalid")
    archive_size = origin.get("archive_size")
    if type(archive_size) is not int or archive_size <= 0:
        errors.append(f"{raw_path}: pinned archive byte count is invalid")
    archive_member = origin.get("archive_member")
    if not isinstance(archive_member, str) or not archive_member:
        errors.append(f"{raw_path}: pinned archive member is missing")
    elif path_error := _validate_manifest_path(archive_member, "pinned archive member"):
        errors.append(path_error)
    runtime = origin.get("runtime")
    if not isinstance(runtime, str) or "cpu-only" not in runtime.casefold():
        errors.append(f"{raw_path}: pinned archive must explicitly declare CPU-only runtime")
    return errors


def _verify_source_build_attestation(
    asset: dict[str, Any], path: Path, raw_path: str
) -> list[str]:
    from release_provenance import (
        DEEPFILTER_RECIPE_FILES,
        _deepfilter_attestation_contract_errors,
        _deepfilter_recipe_sha256,
    )

    errors: list[str] = []
    origin = asset.get("origin")
    if not isinstance(origin, dict):
        return [f"{raw_path}: source-built asset must contain an origin object"]

    raw_attestation = origin.get("attestation_path")
    if not isinstance(raw_attestation, str) or not raw_attestation:
        return [f"{raw_path}: source-built asset must declare origin.attestation_path"]
    if path_error := _validate_manifest_path(raw_attestation, "asset attestation_path"):
        return [path_error]
    attestation_path = REPO_ROOT / raw_attestation.replace("\\", "/")
    if not attestation_path.is_file():
        return [f"{raw_path}: attestation missing: {raw_attestation}"]
    attestation, read_error = _read_json(attestation_path, f"{raw_path} attestation")
    if read_error:
        return [read_error]
    assert attestation is not None

    output = attestation.get("output")
    if not isinstance(output, dict):
        errors.append(f"{raw_path}: attestation output must be an object")
    else:
        output_sha = output.get("sha256")
        if not isinstance(output_sha, str) or not HEX_SHA256.fullmatch(output_sha):
            errors.append(f"{raw_path}: attestation output.sha256 must be a 64-character hex string")
        else:
            actual_sha = _sha256(path)
            if actual_sha.lower() != output_sha.lower():
                errors.append(
                    f"{raw_path}: attestation output sha256 mismatch, expected {output_sha}, got {actual_sha}"
                )
        output_bytes = output.get("bytes")
        if type(output_bytes) is not int or output_bytes <= 0:
            errors.append(f"{raw_path}: attestation output.bytes must be a positive integer")
        elif path.stat().st_size != output_bytes:
            errors.append(
                f"{raw_path}: attestation output byte count mismatch, expected {output_bytes}, got {path.stat().st_size}"
            )

    provenance_relative = origin.get(
        "provenance", "build-support/deepfilter/provenance.json"
    )
    if not isinstance(provenance_relative, str) or not provenance_relative:
        errors.append(f"{raw_path}: source-built asset provenance path is invalid")
        return errors
    if path_error := _validate_manifest_path(provenance_relative, "asset provenance"):
        errors.append(path_error)
        return errors
    provenance_path = REPO_ROOT / provenance_relative.replace("\\", "/")
    if not provenance_path.is_file():
        errors.append(f"{raw_path}: source build provenance is missing: {provenance_relative}")
        return errors
    provenance, read_error = _read_json(provenance_path, f"{raw_path} source provenance")
    if read_error:
        errors.append(read_error)
        return errors
    assert provenance is not None
    errors.extend(
        _deepfilter_attestation_contract_errors(
            attestation,
            provenance,
            label=f"{raw_path}: attestation",
        )
    )

    recipe = attestation.get("recipe")
    recipe_files = recipe.get("files") if isinstance(recipe, dict) else None
    if not isinstance(recipe_files, dict):
        errors.append(f"{raw_path}: attestation recipe.files must be an object")
    else:
        missing = [name for name in DEEPFILTER_RECIPE_FILES if name not in recipe_files]
        if missing:
            errors.append(f"{raw_path}: attestation is missing recipe hashes: {', '.join(missing)}")
        for name, expected_hash in recipe_files.items():
            if not isinstance(name, str):
                errors.append(f"{raw_path}: attestation recipe path must be a string")
                continue
            path_error = _validate_manifest_path(name, "attestation recipe path")
            if path_error:
                errors.append(path_error)
                continue
            if not isinstance(expected_hash, str) or not HEX_SHA256.fullmatch(expected_hash):
                errors.append(f"{raw_path}: recipe hash for {name} is invalid")
                continue
            recipe_path = REPO_ROOT / name.replace("\\", "/")
            if not recipe_path.is_file():
                errors.append(f"{raw_path}: attested recipe file is missing: {name}")
                continue
            actual_hash = _deepfilter_recipe_sha256(recipe_path)
            if actual_hash.lower() != expected_hash.lower():
                errors.append(
                    f"{raw_path}: recipe hash mismatch for {name}, expected {expected_hash}, got {actual_hash}"
                )

    return errors


def verify_assets(
    manifest_path: Path = MANIFEST_PATH,
    selected_paths: set[str] | None = None,
) -> list[str]:
    try:
        manifest = load_asset_manifest(manifest_path)
    except ValueError as exc:
        return [str(exc)]

    errors: list[str] = []
    normalized_selected = (
        {path.replace("\\", "/") for path in selected_paths}
        if selected_paths is not None
        else None
    )
    seen_selected: set[str] = set()
    for asset in manifest.assets:
        raw_path = asset.get("path")
        assert isinstance(raw_path, str) and raw_path
        normalized_path = raw_path.replace("\\", "/")
        if normalized_selected is not None and normalized_path not in normalized_selected:
            continue
        if normalized_selected is not None:
            seen_selected.add(normalized_path)

        path = REPO_ROOT / raw_path.replace("\\", "/")
        if not path.is_file():
            errors.append(f"{raw_path}: missing")
            continue

        origin = asset.get("origin")
        source_built = isinstance(origin, dict) and origin.get("status") == SOURCE_BUILD_STATUS
        if isinstance(origin, dict) and origin.get("status") in BLOCKED_ORIGIN_STATUSES:
            errors.append(f"{raw_path}: origin status is not releasable")
            continue
        expected_size = asset.get("size")
        if (
            not source_built
            and type(expected_size) is int
            and path.stat().st_size != expected_size
        ):
            errors.append(
                f"{raw_path}: size mismatch, expected {expected_size}, got {path.stat().st_size}"
            )
        expected_sha = asset.get("sha256")
        assert isinstance(expected_sha, str) and HEX_SHA256.fullmatch(expected_sha)

        if source_built:
            errors.extend(_verify_source_build_attestation(asset, path, raw_path))
        else:
            actual_sha = _sha256(path)
            if actual_sha.lower() != expected_sha.lower():
                errors.append(f"{raw_path}: sha256 mismatch, expected {expected_sha}, got {actual_sha}")

    if normalized_selected is not None:
        for missing_path in sorted(normalized_selected - seen_selected):
            errors.append(f"{missing_path}: manifest entry missing")
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
