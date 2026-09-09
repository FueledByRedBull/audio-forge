"""Validate portable, reproducible AudioForge evaluation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


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


def _declared_source_hashes(report: dict[str, Any]) -> list[tuple[str, str]]:
    containers: list[Any] = [report.get("source_sha256")]
    provenance = report.get("provenance")
    if isinstance(provenance, dict):
        containers.append(provenance.get("source_hashes"))
    found: list[tuple[str, str]] = []
    for container in containers:
        if not isinstance(container, dict):
            continue
        for raw_path, expected in container.items():
            if isinstance(raw_path, str) and isinstance(expected, str):
                found.append((raw_path, expected))
    return found


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


def _resolve_source_revision(value: object) -> tuple[str | None, str | None]:
    if not isinstance(value, str) or GIT_COMMIT.fullmatch(value) is None:
        return None, "source_revision must be a full lowercase commit ID"
    resolved = _git_text("rev-parse", "--verify", f"{value}^{{commit}}")
    if resolved is None or resolved.casefold() != value:
        return None, f"source_revision is unavailable or not a commit: {value}"
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
        return None, f"source_revision is not an ancestor of HEAD: {value}"
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


def validate_report(path: Path) -> list[str]:
    errors: list[str] = []
    if path.stat().st_size > MAX_TRACKED_REPORT_BYTES:
        errors.append(
            f"{path}: tracked report exceeds {MAX_TRACKED_REPORT_BYTES} bytes; "
            "move per-case detail to an ignored --details-output artifact"
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"{path}: invalid JSON: {error}"]

    if not isinstance(report, dict):
        return [f"{path}: report root must be an object"]

    for location, value in _walk_strings(report):
        if WINDOWS_ABSOLUTE.match(value) or POSIX_HOME.match(value):
            errors.append(f"{path}:{location}: machine-local absolute path: {value!r}")

    source_revision: str | None = None
    if "source_revision" in report:
        source_revision, revision_error = _resolve_source_revision(
            report.get("source_revision")
        )
        if revision_error is not None:
            errors.append(f"{path}: {revision_error}")
        elif source_revision is not None:
            errors.extend(_report_path_at_revision(path, report, source_revision))

    declared_source_hashes = _declared_source_hashes(report)
    for raw_path, expected in declared_source_hashes:
        source_path = Path(raw_path)
        if source_path.is_absolute() or ".." in source_path.parts:
            errors.append(f"{path}: non-portable source hash path: {raw_path!r}")
            continue
        if source_revision is not None:
            blob = _git_blob(source_revision, source_path.as_posix())
            if blob is None:
                errors.append(
                    f"{path}: declared source file is missing at source_revision: {raw_path}"
                )
            elif not re.fullmatch(r"[0-9a-f]{64}", expected):
                errors.append(f"{path}: invalid source SHA-256 for {raw_path}")
            elif hashlib.sha256(blob).hexdigest() != expected:
                errors.append(
                    f"{path}: stale source SHA-256 in source_revision for {raw_path}"
                )
            continue
        resolved = REPO_ROOT / source_path
        if not resolved.is_file():
            errors.append(f"{path}: declared source file is missing: {raw_path}")
        elif not re.fullmatch(r"[0-9a-f]{64}", expected):
            errors.append(f"{path}: invalid source SHA-256 for {raw_path}")
        elif expected not in _portable_source_sha256(resolved):
            errors.append(f"{path}: stale source SHA-256 for {raw_path}")

    if report.get("audible_change") is True:
        if int(report.get("schema_version", 0)) < 2:
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
            if not isinstance(runtime, dict) or "max_p99_frame_seconds" not in runtime:
                errors.append(
                    f"{path}: evaluation_contract.runtime lacks max_p99_frame_seconds"
                )
        if not declared_source_hashes:
            errors.append(
                f"{path}: audible-change report lacks verifiable source SHA-256 hashes"
            )
    if path.name.startswith("hardware-validation"):
        errors.extend(_validate_hardware_report_privacy(path, report))
    return errors


def _validate_hardware_report_privacy(
    path: Path | str,
    report: dict[str, Any],
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
    if int(report.get("schema_version", 0)) < 3:
        redaction = report.get("privacy_redaction")
        if not isinstance(redaction, dict) or redaction.get("applied") is not True:
            errors.append(
                f"{path}: historical hardware report lacks privacy-redaction provenance"
            )
    return errors


def validate_evaluation_tree(root: Path = DEFAULT_EVALUATION_ROOT) -> list[str]:
    errors: list[str] = []
    for path in sorted(root.glob("*.json")):
        errors.extend(validate_report(path))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    args = parser.parse_args()
    errors = validate_evaluation_tree(args.evaluation_root)
    if errors:
        print("Evaluation hygiene check failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Evaluation hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
