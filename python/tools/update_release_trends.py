"""Record comparable AudioForge release hardening metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRENDS = REPO_ROOT / "evaluation" / "release-trends.json"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_metrics(path: Path) -> dict[str, int]:
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    return {
        "bytes": sum(candidate.stat().st_size for candidate in files),
        "file_count": len(files),
    }


def _measurement(value: Any) -> dict[str, Any]:
    return {"status": "measured", "value": value}


def _not_measured(reason: str) -> dict[str, str]:
    return {"status": "not_measured", "reason": reason}


def _git_commit() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return f"{head}+uncommitted" if dirty else head


def _project_version() -> str:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def _report_archive_sha256(report: dict[str, Any]) -> str | None:
    artifact = report.get("artifact")
    if isinstance(artifact, dict):
        value = artifact.get("archive_sha256", artifact.get("sha256"))
        if isinstance(value, str) and SHA256_PATTERN.fullmatch(value.casefold()):
            return value.casefold()
    value = report.get("artifact_sha256")
    if isinstance(value, str) and SHA256_PATTERN.fullmatch(value.casefold()):
        return value.casefold()
    return None


def _deepfilter_metrics(
    path: Path | None,
    *,
    published: bool,
    archive_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path is None or not path.is_file():
        reason = "No release-matched DeepFilter hardening report was supplied."
        return _not_measured(reason), _not_measured(reason)
    report = json.loads(path.read_text(encoding="utf-8"))
    if published and (
        archive_sha256 is None
        or _report_archive_sha256(report) != archive_sha256
    ):
        reason = (
            "The supplied DeepFilter report is source-level evidence and is not "
            "bound to this exact published archive."
        )
        return _not_measured(reason), _not_measured(reason)
    contract = report.get("evaluation_contract")
    if not isinstance(contract, dict):
        raise ValueError("DeepFilter report lacks an evaluation_contract object")
    runtime = contract.get("runtime")
    clean_preservation = contract.get("clean_preservation")
    if not isinstance(runtime, dict) or not isinstance(clean_preservation, dict):
        raise ValueError(
            "DeepFilter evaluation_contract requires runtime and "
            "clean_preservation objects"
        )

    selected_config = report.get("selected_runtime_config")
    if not isinstance(selected_config, dict):
        selected_config = contract.get("configuration")
    if not isinstance(selected_config, dict):
        raise ValueError("DeepFilter report lacks a selected runtime configuration")

    quality: dict[str, Any] = {
        "selected_runtime_config": selected_config,
        "clean_preservation": clean_preservation,
    }
    alignment_gate = report.get("alignment_gate")
    if isinstance(alignment_gate, dict) and isinstance(
        alignment_gate.get("passes"), bool
    ):
        quality["alignment_gate_passed"] = alignment_gate["passes"]
    else:
        quality["alignment_gate_status"] = "not_measured_by_report"
    return _measurement(runtime), _measurement(quality)


def build_entry(
    *,
    version: str,
    status: str,
    commit: str,
    bundle: Path | None,
    archive: Path | None,
    deepfilter_report: Path | None,
    hardware_report: Path | None,
) -> dict[str, Any]:
    if status == "published" and COMMIT_PATTERN.fullmatch(commit.casefold()) is None:
        raise ValueError("published trend rows require an exact 40-character commit")
    archive_hash: str | None = None
    hardware: dict[str, Any] | None = None
    if hardware_report is not None and hardware_report.is_file():
        raw_hardware: Any = json.loads(hardware_report.read_text(encoding="utf-8"))
        if not isinstance(raw_hardware, dict):
            raise ValueError("hardware report root must be an object")
        hardware = raw_hardware
        if hardware.get("passed") is not True or hardware.get("status") != "passed":
            raise ValueError("hardware report must record status=passed and passed=true")
    if bundle is not None and bundle.is_dir():
        bundle_metric = _measurement(_directory_metrics(bundle))
    else:
        bundle_metric = _not_measured("No release-matched portable bundle was supplied.")
    if archive is not None and archive.is_file():
        archive_hash = _sha256(archive)
        archive_metric = _measurement(
            {
                "bytes": archive.stat().st_size,
                "sha256": archive_hash,
                "format": archive.suffix.removeprefix("."),
            }
        )
    else:
        archive_metric = _not_measured("No release-matched archive was supplied.")
    if hardware is not None:
        hardware_hash = _report_archive_sha256(hardware)
        if archive_hash is not None and hardware_hash != archive_hash:
            raise ValueError("hardware report does not match the supplied archive")
        artifact = hardware.get("artifact")
        artifact_bundle = artifact.get("bundle") if isinstance(artifact, dict) else None
        if bundle is None and isinstance(artifact_bundle, dict):
            bundle_bytes = artifact_bundle.get("total_bytes")
            file_count = artifact_bundle.get("file_count")
            if (
                isinstance(bundle_bytes, int)
                and bundle_bytes >= 0
                and isinstance(file_count, int)
                and file_count >= 0
            ):
                bundle_metric = _measurement(
                    {"bytes": bundle_bytes, "file_count": file_count}
                )
        hardware_metric = _measurement(hardware)
    else:
        hardware_metric = _not_measured(
            "No release-matched hardware health report was supplied."
        )
    runtime, quality = _deepfilter_metrics(
        deepfilter_report,
        published=status == "published",
        archive_sha256=archive_hash,
    )
    return {
        "version": version,
        "status": status,
        "commit": commit,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "package": {
            "bundle": bundle_metric,
            "archive": archive_metric,
        },
        "runtime": runtime,
        "quality": quality,
        "hardware": hardware_metric,
    }


def update_trends(path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    if path.is_file():
        trends = json.loads(path.read_text(encoding="utf-8"))
    else:
        trends = {
            "schema_version": 1,
            "policy": (
                "Only compare values with status=measured and release-matched inputs; "
                "missing historical evidence stays explicit rather than inferred."
            ),
            "releases": [],
        }
    releases = [
        release
        for release in trends["releases"]
        if release.get("version") != entry["version"]
    ]
    releases.append(entry)
    releases.sort(
        key=lambda release: tuple(int(part) for part in release["version"].split("."))
    )
    trends["releases"] = releases
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trends, indent=2) + "\n", encoding="utf-8")
    return trends


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_TRENDS)
    parser.add_argument("--version", default=_project_version())
    parser.add_argument(
        "--status", choices=("candidate", "published"), default="candidate"
    )
    parser.add_argument("--commit", default=None)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--deepfilter-report",
        type=Path,
        default=None,
        help=(
            "Release-matched DeepFilter evidence. Published rows accept it only "
            "when the report is bound to the supplied archive SHA-256."
        ),
    )
    parser.add_argument("--hardware-report", type=Path)
    args = parser.parse_args()
    entry = build_entry(
        version=args.version,
        status=args.status,
        commit=args.commit or _git_commit(),
        bundle=args.bundle,
        archive=args.archive,
        deepfilter_report=args.deepfilter_report,
        hardware_report=args.hardware_report,
    )
    update_trends(args.output, entry)
    print(f"Recorded AudioForge {args.version} release trends in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
