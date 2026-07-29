"""Record comparable AudioForge release hardening metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRENDS = REPO_ROOT / "evaluation" / "release-trends.json"
DEFAULT_DEEPFILTER_REPORT = (
    REPO_ROOT / "evaluation" / "deepfilter-hardening-report.json"
)


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


def _deepfilter_metrics(path: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if path is None or not path.is_file():
        reason = "No release-matched DeepFilter hardening report was supplied."
        return _not_measured(reason), _not_measured(reason)
    report = json.loads(path.read_text(encoding="utf-8"))
    contract = report["evaluation_contract"]
    return _measurement(contract["runtime"]), _measurement(
        {
            "selected_runtime_config": report["selected_runtime_config"],
            "clean_preservation": contract["clean_preservation"],
            "alignment_gate_passed": report["alignment_gate"]["passes"],
        }
    )


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
    runtime, quality = _deepfilter_metrics(deepfilter_report)
    if bundle is not None and bundle.is_dir():
        bundle_metric = _measurement(_directory_metrics(bundle))
    else:
        bundle_metric = _not_measured(
            "No release-matched portable bundle was supplied."
        )
    if archive is not None and archive.is_file():
        archive_metric = _measurement(
            {
                "bytes": archive.stat().st_size,
                "sha256": _sha256(archive),
                "format": archive.suffix.removeprefix("."),
            }
        )
    else:
        archive_metric = _not_measured("No release-matched archive was supplied.")
    if hardware_report is not None and hardware_report.is_file():
        hardware_metric = _measurement(
            json.loads(hardware_report.read_text(encoding="utf-8"))
        )
    else:
        hardware_metric = _not_measured(
            "No release-matched hardware health report was supplied."
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
        "--deepfilter-report", type=Path, default=DEFAULT_DEEPFILTER_REPORT
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
