"""Validate portable, reproducible AudioForge evaluation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, NoReturn


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_ROOT = REPO_ROOT / "evaluation"
MAX_TRACKED_REPORT_BYTES = 100_000
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[/\\]")
POSIX_HOME = re.compile(r"^/(?:Users|home)/[^/]+(?:/|$)")
AUDIBLE_CONTRACT_FIELDS = {
    "configuration",
    "asset_hashes",
    "runtime",
    "latency",
    "clean_preservation",
}
DEVICE_PSEUDONYM = re.compile(r"^device-[0-9a-f]{16}$")
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
PORTABLE_TEXT_SUFFIXES = {
    ".bat",
    ".c",
    ".h",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".pyi",
    ".rs",
    ".toml",
    ".yaml",
    ".yml",
}
IMPLEMENTATION_SOURCE_SUFFIXES = {
    ".bat",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hh",
    ".hpp",
    ".json",
    ".ps1",
    ".py",
    ".pyi",
    ".rs",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}


def _portable_source_sha256(path: Path) -> set[str]:
    data = path.read_bytes()
    if path.suffix.casefold() not in PORTABLE_TEXT_SUFFIXES or b"\0" in data:
        return {hashlib.sha256(data).hexdigest()}
    lf = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    return {
        hashlib.sha256(lf).hexdigest(),
        hashlib.sha256(crlf).hexdigest(),
    }


def _is_implementation_source_path(raw_path: str) -> bool:
    """Classify source entries in the mixed implementation hash schema.

    Joint-tuning reports retain native binaries and DLLs alongside source and
    tracked manifests in ``implementation_sha256``. Binaries are retained
    identities, while source and manifest entries require source verification.
    """

    normalized = raw_path.replace("\\", "/")
    if normalized.startswith(("models/", "target/", "dist/")):
        return False
    return Path(normalized).suffix.casefold() in IMPLEMENTATION_SOURCE_SUFFIXES


def _declared_source_hash_records(
    report: dict[str, Any],
) -> list[tuple[str, str, str | None]]:
    records: list[tuple[str, str, str | None]] = []

    containers: list[tuple[str, Any, str | None]] = [
        ("source_sha256", report.get("source_sha256"), "source_revision"),
    ]
    provenance = report.get("provenance")
    if isinstance(provenance, dict):
        containers.append(
            ("provenance.source_hashes", provenance.get("source_hashes"), "source_revision")
        )

    measurement_revision = (
        "measurement_source_revision"
        if report.get("measurement_source_revision") is not None
        else "source_revision"
    )
    for field in ("implementation_sha256", "measurement_implementation_sha256"):
        containers.append((field, report.get(field), measurement_revision))

    seen: set[tuple[str, str, str | None]] = set()
    for _field, container, revision_field in containers:
        if not isinstance(container, dict):
            continue
        for raw_path, expected in container.items():
            if (
                isinstance(raw_path, str)
                and isinstance(expected, str)
                and (
                    _field in {"source_sha256", "provenance.source_hashes"}
                    or _is_implementation_source_path(raw_path)
                )
            ):
                record = (raw_path, expected, revision_field)
                if record not in seen:
                    records.append(record)
                    seen.add(record)
    return records


def _declared_source_hashes(report: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (raw_path, expected)
        for raw_path, expected, _revision_field in _declared_source_hash_records(report)
    ]


def _declared_unverified_implementation_hashes(
    report: dict[str, Any],
) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for field in ("implementation_sha256", "measurement_implementation_sha256"):
        container = report.get(field)
        if not isinstance(container, dict):
            continue
        for raw_path, expected in container.items():
            if (
                isinstance(raw_path, str)
                and isinstance(expected, str)
                and not _is_implementation_source_path(raw_path)
            ):
                record = (raw_path, expected)
                if record not in seen:
                    records.append(record)
                    seen.add(record)
    return records


def _git_text(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_blob(revision: str, relative_path: str) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", "cat-file", "blob", f"{revision}:{relative_path}"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def _resolve_source_revision(
    value: object,
    field_name: str = "source_revision",
) -> tuple[str | None, str | None]:
    if not isinstance(value, str) or GIT_COMMIT.fullmatch(value) is None:
        return None, f"{field_name} must be a full lowercase commit ID"
    resolved = _git_text("rev-parse", "--verify", f"{value}^{{commit}}")
    if resolved is None or resolved.casefold() != value:
        return None, f"{field_name} is unavailable or not a commit: {value}"
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", value, "HEAD"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
        )
    except OSError:
        result = None
    if result is None or result.returncode != 0:
        return None, f"{field_name} is not an ancestor of HEAD: {value}"
    return value, None


def _report_path_at_revision(path: Path, report: dict[str, Any], revision: str) -> list[str]:
    try:
        relative_path = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return [f"{path}: pinned report path is outside the repository"]
    raw_historical = _git_blob(revision, relative_path)
    if raw_historical is None:
        return [f"{path}: report is unavailable at source_revision {revision}"]
    try:
        historical = json.loads(raw_historical.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return [f"{path}: pinned report is invalid JSON: {error}"]
    if not isinstance(historical, dict):
        return [f"{path}: pinned report root must be an object"]
    current = dict(report)
    current.pop("source_revision", None)
    historical.pop("source_revision", None)
    if current != historical:
        return [f"{path}: report contents differ from source_revision {revision}"]
    return []


def _walk_strings(value: Any, location: str = "$") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, str):
        found.append((location, value))
    elif isinstance(value, dict):
        for key, child in value.items():
            found.extend(_walk_strings(child, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_strings(child, f"{location}[{index}]"))
    return found


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value}")


def _walk_nonfinite_numbers(value: Any, location: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{location}: non-finite numeric value")
    elif isinstance(value, dict):
        for key, child in value.items():
            errors.extend(_walk_nonfinite_numbers(child, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_walk_nonfinite_numbers(child, f"{location}[{index}]"))
    return errors


def _validated_schema_version(
    path: Path,
    report: dict[str, Any],
) -> tuple[int | None, list[str]]:
    if "schema_version" not in report:
        return None, []
    value = report["schema_version"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None, [f"{path}: schema_version must be a positive integer"]
    return value, []


def _validate_runtime_value(
    path: Path,
    value: Any,
    location: str,
    allow_none: bool,
    require_number: bool = False,
) -> list[str]:
    errors: list[str] = []
    if isinstance(value, bool):
        errors.append(
            f"{path}:{location}: runtime metric must be a finite nonnegative number; "
            "booleans are not measurements"
        )
    elif isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{path}:{location}: runtime metric must be finite")
    elif isinstance(value, (int, float)):
        if value < 0:
            errors.append(
                f"{path}:{location}: runtime metric must be non-negative"
            )
    elif value is None and not allow_none:
        errors.append(
            f"{path}:{location}: runtime metric must be a finite nonnegative number"
        )
    elif require_number:
        errors.append(
            f"{path}:{location}: runtime metric must be a finite nonnegative number"
        )
    elif isinstance(value, dict):
        for key, child in value.items():
            errors.extend(
                _validate_runtime_value(
                    path,
                    child,
                    f"{location}.{key}",
                    allow_none,
                    _is_runtime_metric_key(key),
                )
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(
                _validate_runtime_value(
                    path,
                    child,
                    f"{location}[{index}]",
                    allow_none,
                    require_number,
                )
            )
    return errors


def _is_runtime_metric_key(key: str) -> bool:
    lowered = key.casefold()
    if lowered.endswith("_reason") or lowered.endswith("_scope") or lowered in {
        "measurement",
        "platform",
        "processor",
        "python",
        "machine",
    }:
        return False
    return bool(
        re.search(
            r"(?:^|_)(?:runtime|latency|seconds|milliseconds|ms|factor|ratio|samples)(?:$|_)",
            lowered,
        )
    )


def _validate_runtime_contract(path: Path, runtime: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    location = "evaluation_contract.runtime"
    explanation = any(
        isinstance(value, str)
        and value.strip()
        and (key == "reason" or key == "scope" or key.endswith("_reason"))
        for key, value in runtime.items()
    )
    max_p99 = runtime.get("max_p99_frame_seconds")
    if max_p99 is None:
        if not explanation:
            errors.append(
                f"{path}:{location}.max_p99_frame_seconds: runtime metric must be "
                "a finite nonnegative number or an explicitly justified null"
            )
    elif isinstance(max_p99, bool) or not isinstance(max_p99, (int, float)):
        errors.append(
            f"{path}:{location}.max_p99_frame_seconds: runtime metric must be "
            "a finite nonnegative number"
        )
    else:
        errors.extend(
            _validate_runtime_value(
                path,
                max_p99,
                f"{location}.max_p99_frame_seconds",
                False,
                True,
            )
        )
    for key, value in runtime.items():
        if key == "max_p99_frame_seconds":
            continue
        errors.extend(
            _validate_runtime_value(
                path,
                value,
                f"{location}.{key}",
                not _is_runtime_metric_key(key),
                _is_runtime_metric_key(key),
            )
        )
    return errors


def validate_report(path: Path, *, unverified: list[str] | None = None) -> list[str]:
    errors: list[str] = []
    if path.stat().st_size > MAX_TRACKED_REPORT_BYTES:
        errors.append(
            f"{path}: tracked report exceeds {MAX_TRACKED_REPORT_BYTES} bytes; "
            "move per-case detail to an ignored --details-output artifact"
        )
    try:
        report = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, ValueError) as error:
        return [f"{path}: invalid JSON: {error}"]

    if not isinstance(report, dict):
        return [f"{path}: report root must be an object"]

    errors.extend(
        f"{path}:{error}" for error in _walk_nonfinite_numbers(report)
    )
    schema_version, schema_errors = _validated_schema_version(path, report)
    errors.extend(schema_errors)

    for location, value in _walk_strings(report):
        if WINDOWS_ABSOLUTE.match(value) or POSIX_HOME.match(value):
            errors.append(f"{path}:{location}: machine-local absolute path: {value!r}")

    resolved_revisions: dict[str, str | None] = {}
    revision_errors: set[str] = set()
    for revision_field in ("source_revision", "measurement_source_revision"):
        if revision_field not in report:
            continue
        resolved, revision_error = _resolve_source_revision(
            report.get(revision_field), revision_field
        )
        resolved_revisions[revision_field] = resolved
        if revision_error is not None:
            errors.append(f"{path}: {revision_error}")
            revision_errors.add(revision_field)

    source_revision = resolved_revisions.get("source_revision")
    if "source_revision" in report:
        if source_revision is not None:
            errors.extend(_report_path_at_revision(path, report, source_revision))

    declared_source_hash_records = _declared_source_hash_records(report)
    for raw_path, expected, revision_field in declared_source_hash_records:
        source_path = Path(raw_path)
        if source_path.is_absolute() or ".." in source_path.parts:
            errors.append(f"{path}: non-portable source hash path: {raw_path!r}")
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            errors.append(f"{path}: invalid source SHA-256 for {raw_path}")
            continue
        revision = resolved_revisions.get(revision_field) if revision_field else None
        if revision_field in revision_errors:
            continue
        if revision is not None:
            blob = _git_blob(revision, source_path.as_posix())
            if blob is None:
                errors.append(
                    f"{path}: declared source file is missing at {revision_field}: {raw_path}"
                )
            elif hashlib.sha256(blob).hexdigest() != expected:
                errors.append(
                    f"{path}: stale source SHA-256 in {revision_field} for {raw_path}"
                )
            continue
        resolved = REPO_ROOT / source_path
        if not resolved.is_file():
            errors.append(f"{path}: declared source file is missing: {raw_path}")
        elif not re.fullmatch(r"[0-9a-f]{64}", expected):
            errors.append(f"{path}: invalid source SHA-256 for {raw_path}")
        elif expected not in _portable_source_sha256(resolved):
            errors.append(f"{path}: stale source SHA-256 for {raw_path}")

    for raw_path, expected in _declared_unverified_implementation_hashes(report):
        artifact_path = Path(raw_path)
        if artifact_path.is_absolute() or ".." in artifact_path.parts:
            errors.append(
                f"{path}: non-portable implementation artifact hash path: {raw_path!r}"
            )
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            errors.append(
                f"{path}: retained implementation hash is unverified for "
                f"non-source object {raw_path}: invalid SHA-256"
            )
            continue
        has_historical_revision = any(
            field in report
            for field in ("source_revision", "measurement_source_revision")
        )
        resolved_artifact = REPO_ROOT / artifact_path
        current_matches = (
            not has_historical_revision
            and resolved_artifact.is_file()
            and hashlib.sha256(resolved_artifact.read_bytes()).hexdigest() == expected
        )
        if not current_matches:
            note = (
                f"{path}: retained implementation hash is unverified for "
                f"non-source object {raw_path}"
            )
            if unverified is not None:
                unverified.append(note)

    if report.get("audible_change") is True:
        if schema_version is None or schema_version < 2:
            errors.append(f"{path}: audible-change reports require schema_version >= 2")
        contract = report.get("evaluation_contract")
        if not isinstance(contract, dict):
            errors.append(f"{path}: audible-change report lacks evaluation_contract")
        else:
            missing = sorted(AUDIBLE_CONTRACT_FIELDS - contract.keys())
            if missing:
                errors.append(
                    f"{path}: evaluation_contract lacks {', '.join(missing)}"
                )
            runtime = contract.get("runtime")
            if not isinstance(runtime, dict):
                errors.append(
                    f"{path}: evaluation_contract.runtime must be an object"
                )
            elif "max_p99_frame_seconds" not in runtime:
                errors.append(
                    f"{path}: evaluation_contract.runtime lacks max_p99_frame_seconds"
                )
            else:
                errors.extend(_validate_runtime_contract(path, runtime))
        if not _declared_source_hashes(report):
            errors.append(
                f"{path}: audible-change report lacks verifiable source SHA-256 hashes"
            )
    if path.name.startswith("hardware-validation"):
        errors.extend(_validate_hardware_report_privacy(path, report, schema_version))
    return errors


def _validate_hardware_report_privacy(
    path: Path | str,
    report: dict[str, Any],
    schema_version: int | None = None,
) -> list[str]:
    routes = report.get("routes")
    if not isinstance(routes, dict):
        return []
    errors: list[str] = []
    for route_name, route in routes.items():
        if not isinstance(route, dict):
            errors.append(f"{path}: routes.{route_name} must be an object")
            continue
        for direction in ("input", "output"):
            value = route.get(direction)
            if not isinstance(value, str) or DEVICE_PSEUDONYM.fullmatch(value) is None:
                errors.append(
                    f"{path}: routes.{route_name}.{direction} must use a "
                    "report-local device pseudonym"
                )
    effective_schema_version = schema_version if schema_version is not None else 0
    if effective_schema_version < 3:
        redaction = report.get("privacy_redaction")
        if not isinstance(redaction, dict) or redaction.get("applied") is not True:
            errors.append(
                f"{path}: historical hardware report lacks privacy-redaction provenance"
            )
    return errors


def validate_evaluation_tree(
    root: Path = DEFAULT_EVALUATION_ROOT, *, unverified: list[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    for path in sorted(root.glob("*.json")):
        errors.extend(validate_report(path, unverified=unverified))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    args = parser.parse_args()
    unverified: list[str] = []
    errors = validate_evaluation_tree(args.evaluation_root, unverified=unverified)
    for note in sorted(set(unverified)):
        print(f"Unverified retained metadata: {note}")
    if errors:
        print("Evaluation hygiene check failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Evaluation hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
