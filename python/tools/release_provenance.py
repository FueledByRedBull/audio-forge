"""Create and verify exact-artifact release provenance sidecars."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from verify_release_assets import load_asset_manifest

from hardware_qualification import (
    REQUIRED_DEVICE_CLASSES,
    REQUIRED_OS_RELEASES,
    REQUIRED_SAMPLE_RATES,
    REQUIRED_SCENARIOS,
    SUPPORTED_DEVICE_CLASSES,
    SUPPORTED_OS_RELEASES,
    SUPPORTED_SAMPLE_RATES,
    SUPPORTED_SCENARIOS,
    coverage_missing,
    validate_case,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
DEEPFILTER_RECIPE_FILES = (
    "build_deepfilter.ps1",
    "build-support/deepfilter/Cargo.toml",
    "build-support/deepfilter/Cargo.lock",
    "build-support/deepfilter/provenance.json",
    "models/DeepFilterNet3_onnx.tar.gz",
    "models/DeepFilterNet3_ll_onnx.tar.gz",
)
DEEPFILTER_TEXT_RECIPE_SUFFIXES = frozenset({".json", ".lock", ".ps1", ".toml"})
CPU_ORT_BUNDLE_ASSETS = {
    "target/onnxruntime-cpu/lib/onnxruntime.dll": "_internal/onnxruntime.dll",
    "target/onnxruntime-cpu/lib/onnxruntime_providers_shared.dll": (
        "_internal/onnxruntime_providers_shared.dll"
    ),
}
QUALIFICATION_KINDS = frozenset(
    {
        "exact-artifact-package",
        "exact-artifact-hardware",
        "exact-artifact-hardware-matrix",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _relative_bundle_files(bundle: Path) -> list[Path]:
    files = [path for path in bundle.rglob("*") if path.is_file()]
    files.sort(key=lambda path: path.relative_to(bundle).as_posix().casefold())
    seen: set[str] = set()
    for path in files:
        relative = path.relative_to(bundle)
        normalized = relative.as_posix()
        if relative.is_absolute() or ".." in relative.parts or "\\" in normalized:
            raise ValueError(f"unsafe bundle path: {normalized}")
        folded = normalized.casefold()
        if folded in seen:
            raise ValueError(f"case-insensitive duplicate bundle path: {normalized}")
        seen.add(folded)
    return files


def build_bundle_manifest(bundle: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    if not bundle.is_dir():
        raise ValueError(f"bundle directory is missing: {bundle}")
    entries = [
        {
            "path": path.relative_to(bundle).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _relative_bundle_files(bundle)
    ]
    return {
        "schema_version": 1,
        "bundle_root": bundle.name,
        "file_count": len(entries),
        "total_bytes": sum(entry["size"] for entry in entries),
        "files": entries,
    }


def compare_bundle_sizes(current: dict[str, Any], previous: dict[str, Any]) -> str:
    """Summarize uncompressed payload changes using existing release manifests."""
    def sizes(manifest: dict[str, Any]) -> dict[str, int]:
        entries = manifest.get("files")
        if not isinstance(entries, list):
            raise ValueError("manifest files must be a list")
        result: dict[str, int] = {}
        seen: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("manifest file must be an object")
            path, size = entry.get("path"), entry.get("size")
            if not isinstance(path, str) or not path or type(size) is not int or size < 0:
                raise ValueError("manifest file requires a path and nonnegative integer size")
            if path.casefold() in seen:
                raise ValueError(f"duplicate manifest path: {path}")
            seen.add(path.casefold())
            result[path] = size
        return result

    before, after = sizes(previous), sizes(current)
    old_total, new_total = sum(before.values()), sum(after.values())
    changes = sorted(
        ((path, after.get(path, 0) - before.get(path, 0)) for path in before.keys() | after.keys()),
        key=lambda change: (-abs(change[1]), change[0]),
    )
    lines = [
        "## Bundle size comparison",
        "",
        f"Uncompressed payload: {old_total:,} -> {new_total:,} bytes ({new_total - old_total:+,}).",
        "Compressed download sizes are separate from these payload sizes.",
        "",
        "| Largest file changes (up to 20) | Previous bytes | Current bytes | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for path, delta in [change for change in changes if change[1]][:20]:
        label = path.replace("|", "&#124;").replace("`", "'").replace("\n", " ").replace("\r", " ")
        lines.append(f"| `{label}` | {before.get(path, 0):,} | {after.get(path, 0):,} | {delta:+,} |")
    if not any(delta for _, delta in changes):
        lines.append("| No payload size changes | | | |")
    return "\n".join(lines) + "\n"


def build_path_baseline(manifest: dict[str, Any]) -> dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("manifest files must be a list")
    paths = [
        entry["path"]
        for entry in files
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    ]
    if len(paths) != len(files):
        raise ValueError("manifest contains an invalid file path entry")
    return {"schema_version": 1, "paths": paths}


def compare_path_baseline(
    manifest: dict[str, Any], baseline: dict[str, Any]
) -> tuple[list[str], list[str]]:
    expected_raw = baseline.get("paths")
    actual_raw = manifest.get("files")
    if not isinstance(expected_raw, list) or not all(
        isinstance(path, str) for path in expected_raw
    ):
        raise ValueError("baseline paths must be a list of strings")
    if not isinstance(actual_raw, list):
        raise ValueError("manifest files must be a list")
    actual = {
        entry["path"]
        for entry in actual_raw
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    expected = set(expected_raw)
    return sorted(actual - expected), sorted(expected - actual)


def _project_version() -> str:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def _git_head() -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"git rev-parse HEAD failed: {result.stderr.strip()}")
    head = result.stdout.strip().casefold()
    if GIT_COMMIT_PATTERN.fullmatch(head) is None:
        raise RuntimeError("git rev-parse HEAD returned an invalid commit ID")
    return head


def _git_commit() -> str:
    head = _git_head()
    configured = os.environ.get("AUDIOFORGE_SOURCE_REVISION") or os.environ.get("GITHUB_SHA")
    if configured is not None:
        workflow_commit = configured.strip().casefold()
        if GIT_COMMIT_PATTERN.fullmatch(workflow_commit) is None:
            raise RuntimeError("configured workflow source revision is not a complete Git commit ID")
        if workflow_commit != head:
            raise RuntimeError("configured workflow source revision does not match the checked-out source commit")
    return head


def _git_is_dirty() -> bool:
    result = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=normal"),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"git status failed: {result.stderr.strip()}")
    return bool(result.stdout.strip())


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _deepfilter_recipe_sha256(path: Path) -> str:
    """Hash recipe text canonically while preserving binary model hashes."""
    if path.suffix.casefold() not in DEEPFILTER_TEXT_RECIPE_SUFFIXES:
        return sha256_file(path)
    canonical = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _deepfilter_attestation_contract_errors(
    attestation: dict[str, Any],
    provenance: dict[str, Any],
    *,
    label: str = "DeepFilter attestation",
) -> list[str]:
    """Validate the shared DeepFilter attestation schema and build contract."""
    errors: list[str] = []
    if attestation.get("schema_version") != 1:
        errors.append(f"{label} schema_version must be 1")
    if attestation.get("kind") != "audioforge.deepfilter.build":
        errors.append(f"{label} kind is invalid")

    source = attestation.get("source")
    expected_upstream = provenance.get("upstream")
    if not isinstance(source, dict) or not isinstance(expected_upstream, dict):
        errors.append(f"{label} source/provenance upstream data is incomplete")
    else:
        for field in ("repository", "commit"):
            if source.get(field) != expected_upstream.get(field):
                errors.append(f"{label} source {field} does not match provenance")

    recipe = attestation.get("recipe")
    expected_build = provenance.get("build")
    if not isinstance(expected_build, dict) or not isinstance(recipe, dict):
        errors.append(f"{label} build identity is incomplete")
        return errors

    for field in ("target", "profile", "features", "default_features"):
        if recipe.get(field) != expected_build.get(field):
            errors.append(f"{label} {field} does not match provenance")
    abi = attestation.get("abi")
    expected_exports = expected_build.get("required_exports")
    actual_exports = abi.get("required_exports") if isinstance(abi, dict) else None
    if not isinstance(expected_exports, list) or actual_exports != expected_exports:
        errors.append(f"{label} ABI exports do not match provenance")

    expected_patch = provenance.get("tract_linalg_patch")
    if not isinstance(expected_patch, dict):
        errors.append(f"{label} tract-linalg patch identity is missing from provenance")
    else:
        patch_bindings = {
            "tract_linalg_archive_sha256": "archive_sha256",
            "tract_linalg_patched_manifest_sha256": "patched_cargo_toml_sha256",
            "tract_linalg_build_rs_sha256": "build_rs_sha256",
        }
        for attested_name, provenance_name in patch_bindings.items():
            if recipe.get(attested_name) != expected_patch.get(provenance_name):
                errors.append(f"{label} {attested_name} does not match provenance")

    toolchain = attestation.get("toolchain")
    tested_rust = expected_build.get("tested_rust")
    if not isinstance(toolchain, dict) or not isinstance(toolchain.get("rustc"), str):
        errors.append(f"{label} Rust toolchain is missing")
    elif isinstance(tested_rust, str):
        tested_release = tested_rust.split(maxsplit=1)[0]
        if not toolchain["rustc"].startswith(f"rustc {tested_release} "):
            errors.append(f"{label} Rust toolchain does not match provenance")
    return errors


def _cpu_ort_asset_errors(bundle: Path) -> list[str]:
    """Bind every bundled CPU ORT DLL to the checked-in asset manifest."""
    present = [
        bundle / relative
        for relative in CPU_ORT_BUNDLE_ASSETS.values()
        if (bundle / relative).exists()
    ]
    if not present:
        return []

    manifest_path = REPO_ROOT / "release-assets.json"
    try:
        entries = load_asset_manifest(manifest_path).entries
    except (OSError, ValueError) as exc:
        return [f"CPU ORT release asset manifest could not be loaded: {exc}"]
    errors: list[str] = []
    for manifest_path_name, bundle_path_name in CPU_ORT_BUNDLE_ASSETS.items():
        asset_path = bundle / bundle_path_name
        if not asset_path.is_file():
            errors.append(f"candidate bundle is missing CPU ORT asset: {bundle_path_name}")
            continue
        asset = entries.get(manifest_path_name)
        expected = asset.get("sha256") if isinstance(asset, dict) else None
        if not isinstance(expected, str) or SHA256_PATTERN.fullmatch(expected.casefold()) is None:
            errors.append(f"CPU ORT release asset manifest lacks a valid hash: {manifest_path_name}")
            continue
        if sha256_file(asset_path) != expected.casefold():
            errors.append(f"candidate CPU ORT asset does not match release-assets.json: {bundle_path_name}")
    return errors


def _deepfilter_attestation_errors(
    attestation_path: Path,
    bundle: Path,
) -> list[str]:
    """Verify the generated DeepFilter attestation against a candidate bundle."""
    errors: list[str] = []
    if not attestation_path.is_file():
        return [f"DeepFilter attestation is missing: {attestation_path}"]
    try:
        attestation = _load_json(attestation_path)
    except (OSError, ValueError) as exc:
        return [str(exc)]

    output = attestation.get("output")
    output_sha: str | None = None
    output_bytes: int | None = None
    if not isinstance(output, dict):
        errors.append("DeepFilter attestation output must be an object")
    else:
        raw_sha = output.get("sha256")
        if not isinstance(raw_sha, str) or SHA256_PATTERN.fullmatch(raw_sha.casefold()) is None:
            errors.append("DeepFilter attestation output.sha256 is invalid")
        else:
            output_sha = raw_sha.casefold()
        raw_bytes = output.get("bytes")
        if type(raw_bytes) is not int or raw_bytes <= 0:
            errors.append("DeepFilter attestation output.bytes is invalid")
        else:
            output_bytes = raw_bytes
        output_name = output.get("name")
        if (
            not isinstance(output_name, str)
            or Path(output_name).name != output_name
            or Path(output_name).suffix.casefold() != ".dll"
        ):
            errors.append("DeepFilter attestation output.name must be a safe DLL filename")

    bundled_dll = bundle / "_internal" / "df.dll"
    if not bundled_dll.is_file():
        errors.append(f"candidate bundle is missing its DeepFilter DLL: {bundled_dll}")
    elif output_sha is not None:
        actual_sha = sha256_file(bundled_dll)
        if actual_sha != output_sha:
            errors.append(
                "DeepFilter attestation output hash does not match the candidate DLL"
            )
    if bundled_dll.is_file() and output_bytes is not None and bundled_dll.stat().st_size != output_bytes:
        errors.append("DeepFilter attestation output byte count does not match the candidate DLL")

    manifest_path = REPO_ROOT / "release-assets.json"
    provenance_path = REPO_ROOT / "build-support" / "deepfilter" / "provenance.json"
    try:
        entries = load_asset_manifest(manifest_path).entries
        provenance = _load_json(provenance_path)
    except (OSError, ValueError) as exc:
        errors.append(f"DeepFilter source identity could not be loaded: {exc}")
        return errors
    errors.extend(_deepfilter_attestation_contract_errors(attestation, provenance))
    manifest_asset = entries.get("df.dll")
    origin = manifest_asset.get("origin") if isinstance(manifest_asset, dict) else None
    expected_upstream = provenance.get("upstream")
    if not isinstance(origin, dict) or not isinstance(expected_upstream, dict):
        errors.append("DeepFilter release asset source identity is incomplete")
    else:
        for field in ("repository", "commit"):
            expected = expected_upstream.get(field)
            if origin.get(field) != expected:
                errors.append(f"DeepFilter attestation source {field} does not match provenance")

    recipe = attestation.get("recipe")
    recipe_files = recipe.get("files") if isinstance(recipe, dict) else None
    if not isinstance(recipe_files, dict):
        errors.append("DeepFilter attestation recipe.files is missing")
    else:
        for required in DEEPFILTER_RECIPE_FILES:
            raw_digest = recipe_files.get(required)
            is_model = required.startswith("models/")
            recipe_file = (
                bundle / "_internal" / required
                if is_model
                else REPO_ROOT / required
            )
            if not isinstance(raw_digest, str) or SHA256_PATTERN.fullmatch(raw_digest.casefold()) is None:
                errors.append(f"DeepFilter attestation recipe hash is invalid: {required}")
            elif not recipe_file.is_file():
                location = "candidate bundle" if is_model else "checkout"
                errors.append(f"DeepFilter attestation recipe file is missing from {location}: {required}")
            elif _deepfilter_recipe_sha256(recipe_file) != raw_digest.casefold():
                errors.append(f"DeepFilter attestation recipe hash does not match: {required}")
            if is_model:
                manifest_model = entries.get(required)
                expected_model_hash = (
                    manifest_model.get("sha256")
                    if isinstance(manifest_model, dict)
                    else None
                )
                if (
                    not isinstance(expected_model_hash, str)
                    or SHA256_PATTERN.fullmatch(expected_model_hash.casefold()) is None
                ):
                    errors.append(f"DeepFilter release asset manifest lacks a valid hash: {required}")
                elif recipe_file.is_file() and sha256_file(recipe_file) != expected_model_hash.casefold():
                    errors.append(f"candidate model does not match release-assets.json: {required}")

    return errors


def _qualification_errors(
    report: dict[str, Any],
    report_path: Path,
    *,
    expected_archive_sha256: str | None = None,
    expected_commit: str | None = None,
) -> list[str]:
    """Require a typed qualification shape before accepting a passing report."""
    errors: list[str] = []
    kind = report.get("qualification_kind")
    if kind not in QUALIFICATION_KINDS:
        errors.append(f"{report_path} has no supported qualification_kind")
        return errors
    expected_schema = 1 if kind != "exact-artifact-hardware" else 3
    if report.get("schema_version") != expected_schema:
        errors.append(
            f"{report_path} qualification schema_version must be {expected_schema}"
        )

    producer = report.get("producer")
    if not isinstance(producer, dict):
        errors.append(f"{report_path} producer identity is missing")
    else:
        for field in (
            "repository",
            "workflow",
            "run_id",
            "run_attempt",
            "event",
            "head_sha",
            "ref",
        ):
            if not isinstance(producer.get(field), str) or not producer[field].strip():
                errors.append(f"{report_path} producer.{field} is missing")
        if isinstance(producer.get("run_id"), str) and not producer["run_id"].isdigit():
            errors.append(f"{report_path} producer.run_id is not numeric")
        elif isinstance(producer.get("run_id"), str) and int(producer["run_id"]) < 1:
            errors.append(f"{report_path} producer.run_id is not positive")
        if isinstance(producer.get("run_attempt"), str) and not producer["run_attempt"].isdigit():
            errors.append(f"{report_path} producer.run_attempt is not numeric")
        elif (
            isinstance(producer.get("run_attempt"), str)
            and int(producer["run_attempt"]) < 1
        ):
            errors.append(f"{report_path} producer.run_attempt is not positive")
        if isinstance(producer.get("head_sha"), str) and GIT_COMMIT_PATTERN.fullmatch(
            producer["head_sha"].casefold()
        ) is None:
            errors.append(f"{report_path} producer.head_sha is not a commit ID")
        if (
            expected_commit is not None
            and isinstance(producer.get("head_sha"), str)
            and producer["head_sha"].casefold() != expected_commit.casefold()
        ):
            errors.append(f"{report_path} producer head SHA does not match the release tag")

    if kind == "exact-artifact-package":
        if report.get("schema_version") != 1:
            errors.append(f"{report_path} package qualification schema_version must be 1")
        checks = report.get("checks")
        required = {
            "provenance",
            "package_smoke",
            "hidden_exe_startup",
            "installer_provenance",
            "installer_smoke",
            "installer_upgrade",
            "source_distribution",
        }
        if not isinstance(checks, dict) or any(
            checks.get(name) != "passed" for name in required
        ):
            errors.append(f"{report_path} package qualification checks are incomplete")
    elif kind == "exact-artifact-hardware":
        errors.extend(
            f"{report_path}: {error}"
            for error in validate_case(
                report,
                expected_archive_sha256=expected_archive_sha256,
                expected_source_revision=expected_commit,
            )
        )
    else:
        cases = report.get("cases")
        coverage = report.get("coverage")
        if not isinstance(cases, list) or not cases:
            errors.append(f"{report_path} hardware matrix cases are missing")
        else:
            case_ids = [
                case.get("id") for case in cases if isinstance(case, dict)
            ]
            string_case_ids = [case_id for case_id in case_ids if isinstance(case_id, str)]
            if len(string_case_ids) != len(set(string_case_ids)):
                errors.append(f"{report_path} hardware matrix case IDs are not unique")
        for case in cases if isinstance(cases, list) else []:
            if not isinstance(case, dict):
                errors.append(f"{report_path} hardware matrix case is not an object")
                continue
            if not isinstance(case.get("id"), str) or not case["id"].strip():
                errors.append(f"{report_path} hardware matrix case ID is missing")
            report_file = case.get("report_file")
            if (
                not isinstance(report_file, str)
                or not report_file.strip()
                or "\\" in report_file
                or Path(report_file).is_absolute()
                or ".." in Path(report_file).parts
            ):
                errors.append(f"{report_path} hardware matrix source report filename is invalid")
            try:
                _require_sha256(case.get("report_sha256"), f"{report_path} hardware case report SHA-256")
            except ValueError as exc:
                errors.append(str(exc))
            if (
                not isinstance(case.get("os_release"), str)
                or case.get("os_release") not in SUPPORTED_OS_RELEASES
            ):
                errors.append(f"{report_path} hardware matrix case Windows release is unsupported")
            if (
                not isinstance(case.get("device_class"), str)
                or case.get("device_class") not in SUPPORTED_DEVICE_CLASSES
            ):
                errors.append(f"{report_path} hardware matrix case device class is unsupported")
            if (
                not isinstance(case.get("nominal_sample_rate_hz"), int)
                or isinstance(case.get("nominal_sample_rate_hz"), bool)
                or case.get("nominal_sample_rate_hz") not in SUPPORTED_SAMPLE_RATES
            ):
                errors.append(f"{report_path} hardware matrix case sample rate is unsupported")
            if (
                not isinstance(case.get("scenario"), str)
                or case.get("scenario") not in SUPPORTED_SCENARIOS
            ):
                errors.append(f"{report_path} hardware matrix case scenario is unsupported")
            if (
                not isinstance(case.get("evidence_kind"), str)
                or case.get("evidence_kind") not in {"automated", "operator_observed"}
            ):
                errors.append(f"{report_path} hardware matrix case evidence kind is unsupported")
        if not isinstance(coverage, dict) or not isinstance(coverage.get("missing"), dict):
            errors.append(f"{report_path} hardware matrix coverage is missing")
        else:
            missing = coverage["missing"]
            expected_missing_keys = {
                "automated_baseline_cases",
                "os_releases",
                "device_classes",
                "nominal_sample_rates_hz",
                "scenarios",
            }
            if set(missing) != expected_missing_keys:
                errors.append(f"{report_path} hardware matrix coverage keys are incomplete")
            if (
                not isinstance(missing.get("automated_baseline_cases"), int)
                or isinstance(missing.get("automated_baseline_cases"), bool)
                or missing.get("automated_baseline_cases") != 0
            ):
                errors.append(f"{report_path} hardware matrix baseline coverage is incomplete")
            for dimension in (
                "os_releases",
                "device_classes",
                "nominal_sample_rates_hz",
                "scenarios",
            ):
                value = missing.get(dimension)
                if not isinstance(value, list) or value:
                    errors.append(f"{report_path} hardware matrix {dimension} coverage is incomplete")
        required = coverage.get("required") if isinstance(coverage, dict) else None
        if not isinstance(required, dict):
            errors.append(f"{report_path} hardware matrix required coverage is missing")
        else:
            expected_required = {
                "required_os_releases": sorted(REQUIRED_OS_RELEASES),
                "required_device_classes": sorted(REQUIRED_DEVICE_CLASSES),
                "required_nominal_sample_rates_hz": sorted(REQUIRED_SAMPLE_RATES),
                "required_scenarios": sorted(REQUIRED_SCENARIOS),
            }
            for key, expected in expected_required.items():
                if required.get(key) != expected:
                    errors.append(f"{report_path} hardware matrix required coverage {key} is invalid")
        if not isinstance(report.get("source_revision"), str) or GIT_COMMIT_PATTERN.fullmatch(
            str(report.get("source_revision", "")).casefold()
        ) is None:
            errors.append(f"{report_path} hardware matrix source revision is malformed")
        elif (
            expected_commit is not None
            and report["source_revision"].casefold() != expected_commit.casefold()
        ):
            errors.append(f"{report_path} hardware matrix source revision differs from the release tag")
        artifact = report.get("artifact")
        artifact_hash = artifact.get("archive_sha256") if isinstance(artifact, dict) else None
        validated_artifact_hash = (
            artifact_hash
            if isinstance(artifact_hash, str)
            and SHA256_PATTERN.fullmatch(artifact_hash) is not None
            else None
        )
        if validated_artifact_hash is None:
            errors.append(f"{report_path} hardware matrix artifact SHA-256 is malformed")
        elif (
            expected_archive_sha256 is not None
            and validated_artifact_hash.casefold() != expected_archive_sha256.casefold()
        ):
            errors.append(f"{report_path} hardware matrix artifact differs from the exact archive")

    return errors


def _matrix_source_report_errors(
    matrix: dict[str, Any],
    matrix_path: Path,
    report_root: Path,
    *,
    expected_archive_sha256: str,
    expected_commit: str | None,
) -> list[str]:
    """Verify that every matrix record still names its downloaded source report."""
    errors: list[str] = []
    root = report_root.resolve()
    if not root.is_dir():
        return [f"{matrix_path} matrix source-report directory is missing: {root}"]
    seen: set[Path] = set()
    source_reports: list[dict[str, Any]] = []
    cases = matrix.get("cases")
    if not isinstance(cases, list):
        return errors
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("report_file"), str):
            continue
        source_path = (root / case["report_file"]).resolve()
        try:
            source_path.relative_to(root)
        except ValueError:
            errors.append(f"{matrix_path} matrix source report escapes its report directory")
            continue
        if source_path in seen:
            errors.append(f"{matrix_path} matrix source report is referenced more than once")
            continue
        seen.add(source_path)
        if not source_path.is_file():
            errors.append(f"{matrix_path} matrix source report is missing: {case['report_file']}")
            continue
        try:
            expected_hash = _require_sha256(
                case.get("report_sha256"),
                f"{matrix_path} matrix source report SHA-256",
            )
            if sha256_file(source_path) != expected_hash:
                errors.append(f"{matrix_path} matrix source report hash does not match")
            source_report = _load_json(source_path)
            source_reports.append(source_report)
            errors.extend(
                _qualification_errors(
                    source_report,
                    source_path,
                    expected_archive_sha256=expected_archive_sha256,
                    expected_commit=expected_commit,
                )
            )
            if source_report.get("status") != "passed" or source_report.get("passed") is not True:
                errors.append(f"{source_path} is not a passing hardware source report")
            source_case = source_report.get("case")
            source_machine = source_report.get("machine")
            if not isinstance(source_case, dict) or source_case.get("id") != case.get("id"):
                errors.append(f"{matrix_path} matrix case does not match its source report")
            else:
                summary = {
                    "os_release": (
                        source_machine.get("release")
                        if isinstance(source_machine, dict)
                        else None
                    ),
                    "device_class": source_case.get("device_class"),
                    "nominal_sample_rate_hz": source_case.get("nominal_sample_rate_hz"),
                    "scenario": source_case.get("scenario"),
                    "evidence_kind": source_case.get("evidence_kind"),
                }
                for field, expected in summary.items():
                    if case.get(field) != expected:
                        errors.append(
                            f"{matrix_path} matrix case {case.get('id')!r} "
                            f"{field} does not match its source report"
                        )
        except (OSError, ValueError, TypeError) as exc:
            errors.append(str(exc))
    coverage = matrix.get("coverage")
    if source_reports and isinstance(coverage, dict):
        expected_missing = coverage_missing(source_reports)
        if coverage.get("missing") != expected_missing:
            errors.append(
                f"{matrix_path} matrix coverage.missing does not match verified source reports"
            )
    return errors


def _distribution_inventory_errors(
    bundle: Path,
    *,
    require_complete: bool,
    expected_revision: object,
) -> list[str]:
    inventory_path = bundle / "_internal" / "licenses" / "dependencies" / "inventory.json"
    if not inventory_path.is_file():
        return [f"distribution license inventory is missing: {inventory_path}"]
    try:
        inventory = _load_json(inventory_path)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    if inventory.get("schema_version") != 1:
        return ["distribution license inventory has an unsupported schema version"]
    source_distribution = inventory.get("source_distribution")
    if not isinstance(source_distribution, dict):
        return ["distribution source status is missing from the license inventory"]
    status = source_distribution.get("status")
    if status not in {"pending", "complete"}:
        return ["distribution source status is invalid"]
    blockers = source_distribution.get("blockers")
    if not isinstance(blockers, list) or any(
        not isinstance(blocker, str) or not blocker.strip() for blocker in blockers
    ):
        return ["distribution source blockers must be a list of strings"]
    if require_complete and status != "complete":
        return [
            "distribution source fulfillment is not complete; "
            "release publication is blocked"
        ]
    if require_complete and blockers:
        return ["distribution source fulfillment still has blockers"]
    if require_complete and (
        not isinstance(expected_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None
        or source_distribution.get("revision") != expected_revision
    ):
        return ["distribution source inventory revision does not match the release commit"]
    return []


def create_sidecars(
    bundle: Path,
    archive: Path,
    output_dir: Path,
    *,
    baseline_path: Path | None = None,
    native_attestation: Path | None = None,
    allow_dirty: bool = False,
) -> tuple[Path, Path, Path]:
    bundle = bundle.resolve()
    archive = archive.resolve()
    output_dir = output_dir.resolve()
    if not archive.is_file():
        raise ValueError(f"archive is missing: {archive}")
    source_dirty = _git_is_dirty()
    if source_dirty and not allow_dirty:
        raise ValueError(
            "release provenance refuses a dirty source tree; commit the exact "
            "candidate source or pass --allow-dirty for a non-promotable local artifact"
        )

    ort_errors = _cpu_ort_asset_errors(bundle)
    if ort_errors:
        raise ValueError(
            "CPU ORT assets are not bound to release-assets.json:\n  "
            + "\n  ".join(ort_errors)
        )

    manifest = build_bundle_manifest(bundle)
    if baseline_path is not None:
        additions, removals = compare_path_baseline(
            manifest, _load_json(baseline_path.resolve())
        )
        if additions or removals:
            raise ValueError(
                "bundle path baseline changed; "
                f"additions={additions!r}, removals={removals!r}"
            )

    native_metadata: dict[str, Any] | None = None
    if native_attestation is not None:
        native_attestation = native_attestation.resolve()
        native_errors = _deepfilter_attestation_errors(native_attestation, bundle)
        if native_errors:
            raise ValueError(
                "DeepFilter attestation is not bound to the candidate bundle:\n  "
                + "\n  ".join(native_errors)
            )
        attestation = _load_json(native_attestation)
        output = attestation.get("output")
        if not isinstance(output, dict):
            raise ValueError("DeepFilter attestation output is missing")
        native_metadata = {
            "name": native_attestation.name,
            "sha256": sha256_file(native_attestation),
            "output_sha256": str(output["sha256"]).casefold(),
            "output_bytes": output["bytes"],
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{archive.name}.manifest.json"
    checksum_path = output_dir / f"{archive.name}.sha256"
    metadata_path = output_dir / f"{archive.name}.metadata.json"
    _write_json(manifest_path, manifest)

    archive_hash = sha256_file(archive)
    checksum_path.write_text(
        f"{archive_hash}  {archive.name}\n", encoding="ascii", newline="\n"
    )
    metadata = {
        "schema_version": 1,
        "version": _project_version(),
        "commit": _git_commit(),
        "source_dirty": source_dirty,
        "archive": {
            "name": archive.name,
            "size": archive.stat().st_size,
            "sha256": archive_hash,
            "checksum": checksum_path.name,
        },
        "bundle": {
            "root": manifest["bundle_root"],
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
            "manifest": manifest_path.name,
            "manifest_sha256": sha256_file(manifest_path),
        },
        "toolchain": {
            "python": platform.python_version(),
            "pyinstaller": _distribution_version("pyinstaller"),
        },
        "workflow": {
            "repository": os.environ.get("GITHUB_REPOSITORY", "local"),
            "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "local"),
            "ref": os.environ.get("GITHUB_REF", "local"),
            "runner_os": os.environ.get("RUNNER_OS", platform.system()),
            "runner_arch": os.environ.get("RUNNER_ARCH", platform.machine()),
            "image_os": os.environ.get("ImageOS", "local"),
            "image_version": os.environ.get("ImageVersion", "local"),
        },
    }
    if native_metadata is not None:
        metadata["native_attestation"] = {"deepfilter": native_metadata}
    _write_json(metadata_path, metadata)
    return checksum_path, manifest_path, metadata_path


def verify_sidecars(
    archive: Path,
    checksum_path: Path,
    manifest_path: Path,
    metadata_path: Path,
    *,
    bundle: Path | None = None,
    baseline_path: Path | None = None,
    expected_archive_sha256: str | None = None,
    expected_commit: str | None = None,
    native_attestation: Path | None = None,
    reports: Sequence[Path] = (),
    matrix_report_root: Path | None = None,
    require_source_distribution: bool = False,
) -> list[str]:
    errors: list[str] = []
    archive = archive.resolve()
    checksum_path = checksum_path.resolve()
    manifest_path = manifest_path.resolve()
    metadata_path = metadata_path.resolve()
    for path in (archive, checksum_path, manifest_path, metadata_path):
        if not path.is_file():
            errors.append(f"required provenance file is missing: {path}")
    if errors:
        return errors

    actual_archive_hash = sha256_file(archive)
    if expected_archive_sha256 is not None:
        try:
            expected_hash = _require_sha256(
                expected_archive_sha256.casefold(), "expected archive SHA-256"
            )
            if actual_archive_hash != expected_hash:
                errors.append("exact archive does not match the promotion SHA-256")
        except ValueError as exc:
            errors.append(str(exc))
    checksum_parts = checksum_path.read_text(encoding="ascii").strip().split()
    if checksum_parts != [actual_archive_hash, archive.name]:
        errors.append("checksum sidecar does not match the exact archive")

    try:
        metadata = _load_json(metadata_path)
        manifest = _load_json(manifest_path)
        recorded_hash = _require_sha256(
            metadata.get("archive", {}).get("sha256")
            if isinstance(metadata.get("archive"), dict)
            else None,
            "metadata archive.sha256",
        )
        if recorded_hash != actual_archive_hash:
            errors.append("metadata archive SHA-256 does not match the exact archive")
        archive_data = metadata.get("archive")
        if not isinstance(archive_data, dict):
            errors.append("metadata archive must be an object")
        else:
            if archive_data.get("name") != archive.name:
                errors.append("metadata archive name does not match")
            if archive_data.get("size") != archive.stat().st_size:
                errors.append("metadata archive size does not match")
            if archive_data.get("checksum") != checksum_path.name:
                errors.append("metadata checksum filename does not match")
        bundle_data = metadata.get("bundle")
        if not isinstance(bundle_data, dict):
            errors.append("metadata bundle must be an object")
        else:
            if bundle_data.get("manifest") != manifest_path.name:
                errors.append("metadata manifest filename does not match")
            if bundle_data.get("manifest_sha256") != sha256_file(manifest_path):
                errors.append("metadata manifest SHA-256 does not match")
            if bundle_data.get("file_count") != manifest.get("file_count"):
                errors.append("metadata and manifest file counts differ")
            if bundle_data.get("total_bytes") != manifest.get("total_bytes"):
                errors.append("metadata and manifest byte counts differ")
        if metadata.get("schema_version") != 1 or manifest.get("schema_version") != 1:
            errors.append("unsupported provenance schema version")
        if metadata.get("version") != _project_version():
            errors.append("metadata version does not match the source tree")
        source_dirty = metadata.get("source_dirty")
        if not isinstance(source_dirty, bool):
            errors.append("metadata source_dirty must be a boolean")
        elif expected_commit is not None and source_dirty:
            errors.append("dirty-source release metadata cannot be promoted")
        if expected_commit is not None and metadata.get("commit") != expected_commit:
            errors.append("metadata commit does not match the release tag commit")

        if baseline_path is not None:
            additions, removals = compare_path_baseline(
                manifest, _load_json(baseline_path.resolve())
            )
            if additions or removals:
                errors.append(
                    "bundle path baseline changed; "
                    f"additions={additions!r}, removals={removals!r}"
                )

        if bundle is not None:
            errors.extend(
                _distribution_inventory_errors(
                    bundle,
                    require_complete=require_source_distribution,
                    expected_revision=metadata.get("commit"),
                )
            )
            errors.extend(_cpu_ort_asset_errors(bundle))
            actual_manifest = build_bundle_manifest(bundle)
            file_contract_fields = (
                "schema_version",
                "file_count",
                "total_bytes",
                "files",
            )
            if any(
                actual_manifest.get(field) != manifest.get(field)
                for field in file_contract_fields
            ):
                errors.append("extracted bundle does not match its per-file manifest")

        native_metadata = metadata.get("native_attestation")
        if native_attestation is None:
            if native_metadata is not None:
                errors.append("metadata contains a native attestation but none was supplied")
        else:
            native_attestation = native_attestation.resolve()
            if bundle is None:
                errors.append("native attestation verification requires an extracted bundle")
            else:
                errors.extend(_deepfilter_attestation_errors(native_attestation, bundle))
            if not isinstance(native_metadata, dict):
                errors.append("metadata native_attestation must be an object")
            else:
                deepfilter = native_metadata.get("deepfilter")
                if not isinstance(deepfilter, dict):
                    errors.append("metadata native_attestation.deepfilter is missing")
                else:
                    if deepfilter.get("name") != native_attestation.name:
                        errors.append("metadata native attestation name does not match")
                    if native_attestation.is_file():
                        if deepfilter.get("sha256") != sha256_file(native_attestation):
                            errors.append("metadata native attestation SHA-256 does not match")
                        attestation = _load_json(native_attestation)
                        output = attestation.get("output")
                        if isinstance(output, dict):
                            if deepfilter.get("output_sha256") != str(output.get("sha256", "")).casefold():
                                errors.append("metadata native attestation output SHA-256 does not match")
                            if deepfilter.get("output_bytes") != output.get("bytes"):
                                errors.append("metadata native attestation output size does not match")

        for report_path in reports:
            report = _load_json(report_path.resolve())
            errors.extend(
                _qualification_errors(
                    report,
                    report_path,
                    expected_archive_sha256=actual_archive_hash,
                    expected_commit=expected_commit,
                )
            )
            if (
                matrix_report_root is not None
                and report.get("qualification_kind") == "exact-artifact-hardware-matrix"
            ):
                errors.extend(
                    _matrix_source_report_errors(
                        report,
                        report_path,
                        matrix_report_root,
                        expected_archive_sha256=actual_archive_hash,
                        expected_commit=expected_commit,
                    )
                )
            artifact = report.get("artifact")
            report_hash = (
                artifact.get("sha256") if isinstance(artifact, dict) else None
            )
            if report_hash is None and isinstance(artifact, dict):
                report_hash = artifact.get("archive_sha256")
            if report_hash is None:
                report_hash = report.get("artifact_sha256")
            try:
                report_hash = _require_sha256(
                    report_hash, f"{report_path} artifact SHA-256"
                )
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if report_hash != actual_archive_hash:
                errors.append(
                    f"{report_path} references a different release artifact"
                )
            status = report.get("status")
            passed = report.get("passed")
            if status != "passed" or passed is not True:
                errors.append(f"{report_path} is not a passing qualification report")
            if expected_commit is not None:
                report_commit = report.get("commit")
                if report_commit is None:
                    report_commit = report.get("source_revision")
                if report_commit != expected_commit:
                    errors.append(
                        f"{report_path} source revision does not match the "
                        "release tag commit"
                    )
    except (OSError, ValueError, TypeError) as exc:
        errors.append(str(exc))
    return errors


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="create release sidecars")
    _add_common_paths(create)
    create.add_argument("--output-dir", type=Path, default=Path.cwd())
    create.add_argument(
        "--allow-dirty",
        action="store_true",
        help="mark a local dirty-tree artifact as non-promotable",
    )
    create.add_argument(
        "--native-attestation",
        type=Path,
        help="DeepFilter build attestation bound to the bundle's _internal/df.dll",
    )

    verify = subparsers.add_parser("verify", help="verify release sidecars")
    _add_common_paths(verify)
    verify.add_argument("--checksum", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--metadata", type=Path, required=True)
    verify.add_argument("--expected-archive-sha256")
    verify.add_argument("--expected-commit")
    verify.add_argument(
        "--native-attestation",
        type=Path,
        help="DeepFilter build attestation bound to the bundle's _internal/df.dll",
    )
    verify.add_argument("--report", type=Path, action="append", default=[])
    verify.add_argument(
        "--require-source-distribution",
        action="store_true",
        help="reject pending corresponding-source fulfillment for publication",
    )
    verify.add_argument(
        "--matrix-report-root",
        type=Path,
        help="directory containing the source reports named by a matrix report",
    )

    baseline = subparsers.add_parser(
        "write-baseline", help="write a reviewed bundle path baseline"
    )
    baseline.add_argument("--bundle", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    sizes = subparsers.add_parser("compare-sizes", help="compare existing bundle manifests")
    sizes.add_argument("--manifest", type=Path, required=True)
    sizes.add_argument("--previous-manifest", type=Path, required=True)
    return parser


def _print_errors(errors: Iterable[str]) -> int:
    errors = list(errors)
    if not errors:
        print("Release provenance verification passed")
        return 0
    print("Release provenance verification failed:", file=sys.stderr)
    for error in errors:
        print(f"  {error}", file=sys.stderr)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "compare-sizes":
        print(compare_bundle_sizes(_load_json(args.manifest), _load_json(args.previous_manifest)), end="")
        return 0
    if args.command == "write-baseline":
        baseline = build_path_baseline(build_bundle_manifest(args.bundle))
        _write_json(args.output.resolve(), baseline)
        print(f"Wrote bundle path baseline: {args.output.resolve()}")
        return 0
    if args.command == "create":
        paths = create_sidecars(
            args.bundle,
            args.archive,
            args.output_dir,
            baseline_path=args.baseline,
            native_attestation=args.native_attestation,
            allow_dirty=args.allow_dirty,
        )
        print("Created release sidecars:")
        for path in paths:
            print(f"  {path}")
        return 0
    return _print_errors(
        verify_sidecars(
            args.archive,
            args.checksum,
            args.manifest,
            args.metadata,
            bundle=args.bundle,
            baseline_path=args.baseline,
            expected_archive_sha256=args.expected_archive_sha256,
            expected_commit=args.expected_commit,
            native_attestation=args.native_attestation,
            reports=args.report,
            matrix_report_root=args.matrix_report_root,
            require_source_distribution=args.require_source_distribution,
        )
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Release provenance failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
