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


REPO_ROOT = Path(__file__).resolve().parents[2]
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


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
    configured = os.environ.get("GITHUB_SHA")
    if configured is not None:
        workflow_commit = configured.strip().casefold()
        if GIT_COMMIT_PATTERN.fullmatch(workflow_commit) is None:
            raise RuntimeError("GITHUB_SHA is not a complete Git commit ID")
        if workflow_commit != head:
            raise RuntimeError("GITHUB_SHA does not match the checked-out source commit")
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


def create_sidecars(
    bundle: Path,
    archive: Path,
    output_dir: Path,
    *,
    baseline_path: Path | None = None,
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
    reports: Sequence[Path] = (),
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

        for report_path in reports:
            report = _load_json(report_path.resolve())
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

    verify = subparsers.add_parser("verify", help="verify release sidecars")
    _add_common_paths(verify)
    verify.add_argument("--checksum", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--metadata", type=Path, required=True)
    verify.add_argument("--expected-archive-sha256")
    verify.add_argument("--expected-commit")
    verify.add_argument("--report", type=Path, action="append", default=[])

    baseline = subparsers.add_parser(
        "write-baseline", help="write a reviewed bundle path baseline"
    )
    baseline.add_argument("--bundle", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
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
            reports=args.report,
        )
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Release provenance failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
