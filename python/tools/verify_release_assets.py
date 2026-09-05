"""Verify local binary/model assets against release-assets.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from release_provenance import sha256_file as _sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "release-assets.json"
SOURCE_BUILD_STATUS = "verified-source-build"
SOURCE_RECIPE_FILES = (
    "build_deepfilter.ps1",
    "build-support/deepfilter/Cargo.toml",
    "build-support/deepfilter/Cargo.lock",
    "build-support/deepfilter/provenance.json",
    "models/DeepFilterNet3_onnx.tar.gz",
    "models/DeepFilterNet3_ll_onnx.tar.gz",
)
SOURCE_RECIPE_TEXT_SUFFIXES = frozenset({".json", ".lock", ".ps1", ".toml"})
HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


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


def _read_json(path: Path, label: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"{label}: could not read JSON: {exc}"
    if not isinstance(raw, dict):
        return None, f"{label}: JSON root must be an object"
    return raw, None


def _recipe_sha256(path: Path) -> str:
    """Hash recipe text after EOL normalization so archives and checkouts agree."""
    if path.suffix.casefold() not in SOURCE_RECIPE_TEXT_SUFFIXES:
        return _sha256(path)
    canonical = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _verify_source_build_attestation(
    asset: dict[str, Any], path: Path, raw_path: str
) -> list[str]:
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
    if attestation.get("schema_version") != 1:
        errors.append(f"{raw_path}: attestation schema_version must be 1")
    if attestation.get("kind") != "audioforge.deepfilter.build":
        errors.append(f"{raw_path}: attestation kind is not the DeepFilter build contract")

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

    source = attestation.get("source")
    expected_upstream = provenance.get("upstream")
    if not isinstance(source, dict) or not isinstance(expected_upstream, dict):
        errors.append(f"{raw_path}: attestation source/provenance upstream data is incomplete")
    else:
        if source.get("repository") != expected_upstream.get("repository"):
            errors.append(f"{raw_path}: attestation source repository does not match provenance")
        if source.get("commit") != expected_upstream.get("commit"):
            errors.append(f"{raw_path}: attestation source commit does not match provenance")

    recipe = attestation.get("recipe")
    recipe_files = recipe.get("files") if isinstance(recipe, dict) else None
    if not isinstance(recipe_files, dict):
        errors.append(f"{raw_path}: attestation recipe.files must be an object")
    else:
        missing = [name for name in SOURCE_RECIPE_FILES if name not in recipe_files]
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
            actual_hash = _recipe_sha256(recipe_path)
            if actual_hash.lower() != expected_hash.lower():
                errors.append(
                    f"{raw_path}: recipe hash mismatch for {name}, expected {expected_hash}, got {actual_hash}"
                )

    expected_build = provenance.get("build")
    attested_abi = attestation.get("abi")
    expected_exports = expected_build.get("required_exports") if isinstance(expected_build, dict) else None
    actual_exports = attested_abi.get("required_exports") if isinstance(attested_abi, dict) else None
    if not isinstance(expected_exports, list) or actual_exports != expected_exports:
        errors.append(f"{raw_path}: attested required exports do not match provenance")
    expected_patch = provenance.get("tract_linalg_patch")
    if not isinstance(expected_patch, dict) or not isinstance(recipe, dict):
        errors.append(f"{raw_path}: attested tract-linalg patch binding is incomplete")
    else:
        patch_bindings = {
            "tract_linalg_archive_sha256": "archive_sha256",
            "tract_linalg_patched_manifest_sha256": "patched_cargo_toml_sha256",
            "tract_linalg_build_rs_sha256": "build_rs_sha256",
        }
        for attested_name, provenance_name in patch_bindings.items():
            if recipe.get(attested_name) != expected_patch.get(provenance_name):
                errors.append(
                    f"{raw_path}: attested {attested_name} does not match provenance"
                )
    if isinstance(recipe, dict) and isinstance(expected_build, dict):
        if recipe.get("target") != expected_build.get("target"):
            errors.append(f"{raw_path}: attested target does not match provenance")
        if recipe.get("profile") != expected_build.get("profile"):
            errors.append(f"{raw_path}: attested build profile does not match provenance")
        if recipe.get("features") != expected_build.get("features"):
            errors.append(f"{raw_path}: attested features do not match provenance")
        if recipe.get("default_features") != expected_build.get("default_features"):
            errors.append(f"{raw_path}: attested default_features does not match provenance")
    toolchain = attestation.get("toolchain")
    tested_rust = expected_build.get("tested_rust") if isinstance(expected_build, dict) else None
    if not isinstance(toolchain, dict) or not isinstance(toolchain.get("rustc"), str):
        errors.append(f"{raw_path}: attested Rust toolchain is missing")
    elif isinstance(tested_rust, str):
        tested_release = tested_rust.split(maxsplit=1)[0]
        if not toolchain["rustc"].startswith(f"rustc {tested_release} "):
            errors.append(f"{raw_path}: attested Rust toolchain does not match provenance")
    return errors


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

        origin = asset.get("origin")
        source_built = isinstance(origin, dict) and origin.get("status") == SOURCE_BUILD_STATUS
        expected_size = asset.get("size")
        if expected_size is not None and type(expected_size) is not int:
            errors.append(f"{raw_path}: manifest size must be an integer")
        if (
            not source_built
            and type(expected_size) is int
            and path.stat().st_size != expected_size
        ):
            errors.append(
                f"{raw_path}: size mismatch, expected {expected_size}, got {path.stat().st_size}"
            )
        expected_sha = asset.get("sha256")
        if not isinstance(expected_sha, str) or not HEX_SHA256.fullmatch(expected_sha):
            errors.append(f"{raw_path}: manifest sha256 must be a 64-character hex string")
            continue

        if source_built:
            errors.extend(_verify_source_build_attestation(asset, path, raw_path))
        else:
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
