"""Collect build dependency identities and license texts for the shipped bundle."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tomllib
from typing import Any
import zipfile

from source_distribution import (
    SourceDistributionError,
    _archive_target,
    load_manifest,
    verify_sources,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR_ENV = "AUDIOFORGE_SOURCE_DIR"
SOURCE_REVISION_ENV = "AUDIOFORGE_SOURCE_REVISION"


def _native_source_components(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for entry in manifest.get("entries", []):
        if not isinstance(entry, dict) or entry.get("kind") not in {
            "native-build-source",
            "cargo-source-patch",
        }:
            continue
        components.append(
            {
                "id": str(entry["id"]),
                "name": str(entry["name"]),
                "version": str(entry["version"]),
                "license": str(entry.get("license") or "unknown"),
                "source": str(entry.get("source_of_truth") or entry["url"]),
                "notices": [],
            }
        )
    return components


def _copy_notice_bytes(
    data: bytes,
    basename: str,
    destination: Path,
    seen: set[str],
) -> dict[str, str] | None:
    digest = hashlib.sha256(data).hexdigest()
    if digest in seen:
        return None
    seen.add(digest)
    destination.mkdir(parents=True, exist_ok=True)
    name = f"{digest[:16]}-{Path(basename).name}"
    (destination / name).write_bytes(data)
    return {"file": f"{destination.name}/{name}", "sha256": digest}


def _archive_member_name(name: str) -> str:
    """Normalize an archive member without permitting traversal."""
    normalized = name.replace("\\", "/")
    if not normalized or normalized.startswith("/") or ".." in Path(normalized).parts:
        raise SourceDistributionError(f"Unsafe source archive member {name!r}")
    return normalized


def _read_declared_archive_members(
    archive_path: Path,
    entry_id: str,
    license_paths: list[Any],
) -> list[tuple[str, bytes]]:
    """Read declared notices from either a tar archive or a ZIP archive."""
    if zipfile.is_zipfile(archive_path):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = {
                    _archive_member_name(info.filename): info
                    for info in archive.infolist()
                    if not info.is_dir()
                }
                result: list[tuple[str, bytes]] = []
                for relative in license_paths:
                    if not isinstance(relative, str) or not relative:
                        raise SourceDistributionError(
                            f"Source archive {entry_id} lacks declared license {relative!r}"
                        )
                    relative = _archive_member_name(relative)
                    info = members.get(relative)
                    if info is None:
                        raise SourceDistributionError(
                            f"Source archive {entry_id} lacks declared license {relative!r}"
                        )
                    result.append((relative, archive.read(info)))
                return result
        except (OSError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
            if isinstance(exc, SourceDistributionError):
                raise
            raise SourceDistributionError(
                f"Could not read source license archive {entry_id}: {exc}"
            ) from exc

    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            members = {
                _archive_member_name(str(member.name)): member
                for member in archive.getmembers()
                if member.isfile()
            }
            result = []
            for relative in license_paths:
                if not isinstance(relative, str) or not relative:
                    raise SourceDistributionError(
                        f"Source archive {entry_id} lacks declared license {relative!r}"
                    )
                relative = _archive_member_name(relative)
                member = members.get(relative)
                if member is None:
                    raise SourceDistributionError(
                        f"Source archive {entry_id} lacks declared license {relative!r}"
                    )
                handle = archive.extractfile(member)
                if handle is None:
                    raise SourceDistributionError(
                        f"Source archive {entry_id} license is unreadable: {relative}"
                    )
                result.append((relative, handle.read()))
            return result
    except (OSError, tarfile.TarError) as exc:
        raise SourceDistributionError(
            f"Could not read source license archive {entry_id}: {exc}"
        ) from exc


def _source_archive_notices(
    manifest: dict[str, Any],
    source_dir: Path,
    output: Path,
) -> dict[str, list[dict[str, str]]]:
    """Copy only explicitly named license members from verified source archives."""
    notices: dict[str, list[dict[str, str]]] = {}
    for entry in manifest.get("entries", []):
        if not isinstance(entry, dict):
            continue
        license_paths = entry.get("license_paths")
        if not isinstance(license_paths, list) or not license_paths:
            continue
        archive_path = _archive_target(source_dir, entry)
        component_notices: list[dict[str, str]] = []
        seen: set[str] = set()
        for relative, data in _read_declared_archive_members(
            archive_path,
            str(entry["id"]),
            license_paths,
        ):
            record = _copy_notice_bytes(
                data,
                Path(relative).name,
                output / f"source-{entry['id']}",
                seen,
            )
            if record is not None:
                component_notices.append(record)
        notices[str(entry["id"])] = component_notices
    return notices


def _deepfilter_rust_components(
    output: Path,
    existing: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Collect notices from the exact C API workspace used for df.dll."""
    provenance_path = ROOT / "build-support" / "deepfilter" / "provenance.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        commit = provenance["upstream"]["commit"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"DeepFilter provenance is unreadable: {exc}") from exc
    workspace = ROOT / "target" / "deepfilter-build" / f"workspace-{commit}"
    manifest_path = workspace / "Cargo.toml"
    if not manifest_path.is_file():
        raise ValueError(
            "DeepFilter build workspace is missing; run build_deepfilter.ps1 before packaging"
        )
    try:
        cargo = json.loads(
            subprocess.run(
                [
                    "cargo",
                    "metadata",
                    "--locked",
                    "--format-version",
                    "1",
                    "--filter-platform",
                    "x86_64-pc-windows-msvc",
                    "--manifest-path",
                    str(manifest_path),
                    "--no-default-features",
                    "--features",
                    "capi",
                ],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not resolve the DeepFilter Cargo graph: {exc}") from exc
    resolved = {node["id"] for node in cargo["resolve"]["nodes"]}
    components: list[dict[str, Any]] = []
    for package in sorted(cargo["packages"], key=lambda item: (item["name"], item["version"])):
        key = (str(package["name"]).casefold(), str(package["version"]))
        if package["id"] not in resolved or key in existing:
            continue
        directory = Path(package["manifest_path"]).parent
        paths = [path for path in directory.rglob("*") if _notice_name(path) and path.is_file()]
        if package["name"] == "deep_filter":
            source = f"https://github.com/Rikorose/DeepFilterNet/commit/{commit}"
        else:
            source = f"https://crates.io/crates/{package['name']}/{package['version']}"
        components.append(
            {
                "name": package["name"],
                "version": package["version"],
                "license": package.get("license"),
                "source": source,
                "notices": copy_notices(
                    paths,
                    output / f"rust-deepfilter-{package['name']}-{package['version']}",
                ),
            }
        )
        existing.add(key)
    return components


def source_distribution_status(notice_output: Path | None = None) -> dict[str, Any]:
    """Expose the checked-in source manifest without hiding blockers."""
    manifest_path = ROOT / "licenses" / "source-manifest.json"
    if not manifest_path.is_file():
        return {
            "status": "pending",
            "manifest_status": "incomplete",
            "blockers": ["licenses/source-manifest.json is missing or cannot be loaded"],
            "manifest": "licenses/source-manifest.json",
            "native_components": [],
        }
    try:
        manifest = load_manifest(manifest_path)
    except (OSError, SourceDistributionError, ValueError) as exc:
        return {
            "status": "pending",
            "manifest_status": "incomplete",
            "blockers": [f"source manifest validation failed: {exc}"],
            "manifest": "licenses/source-manifest.json",
            "native_components": [],
        }
    status = manifest.get("status")
    blockers = manifest.get("blockers")
    if status not in {"complete", "incomplete"} or not isinstance(blockers, list):
        return {
            "status": "pending",
            "manifest_status": "incomplete",
            "blockers": ["source manifest has no truthful completion status"],
            "manifest": "licenses/source-manifest.json",
            "native_components": [],
        }
    native_components = _native_source_components(manifest)
    source_dir_value = os.environ.get(SOURCE_DIR_ENV)
    revision = os.environ.get(SOURCE_REVISION_ENV)
    if not source_dir_value or not revision:
        blockers = [
            *[str(blocker) for blocker in blockers],
            f"{SOURCE_DIR_ENV} and {SOURCE_REVISION_ENV} are required for a verified source receipt",
        ]
        return {
            "status": "pending",
            "manifest_status": status,
            "blockers": blockers,
            "manifest": "licenses/source-manifest.json",
            "native_components": native_components,
        }
    source_dir = Path(source_dir_value)
    if not source_dir.is_absolute():
        source_dir = ROOT / source_dir
    try:
        verify_sources(
            manifest,
            source_dir.resolve(),
            include_runtime_assets=True,
            require_receipt=True,
            revision=revision,
        )
    except SourceDistributionError as exc:
        return {
            "status": "pending",
            "manifest_status": status,
            "blockers": [*map(str, blockers), f"source hydration verification failed: {exc}"],
            "manifest": "licenses/source-manifest.json",
            "source_dir": "build/source-distribution"
            if source_dir.resolve().is_relative_to((ROOT / "build").resolve())
            else "external",
            "native_components": native_components,
        }
    if notice_output is not None:
        try:
            source_notices = _source_archive_notices(manifest, source_dir.resolve(), notice_output)
        except SourceDistributionError as exc:
            return {
                "status": "pending",
                "manifest_status": status,
                "blockers": [*map(str, blockers), f"source license extraction failed: {exc}"],
                "manifest": "licenses/source-manifest.json",
                "source_dir": "build/source-distribution"
                if source_dir.resolve().is_relative_to((ROOT / "build").resolve())
                else "external",
                "native_components": native_components,
            }
        for component in native_components:
            component["notices"] = source_notices.get(str(component["id"]), [])
    return {
        # release_provenance uses ``pending`` for a candidate whose
        # corresponding-source manifest is still incomplete.  Preserve the
        # manifest's exact state alongside it so the inventory cannot imply
        # that blockers were cleared.
        "status": "complete" if status == "complete" else "pending",
        "manifest_status": status,
        "blockers": [str(blocker) for blocker in blockers],
        "manifest": "licenses/source-manifest.json",
        "source_dir": "build/source-distribution"
        if source_dir.resolve().is_relative_to((ROOT / "build").resolve())
        else "external",
        "revision": revision,
        "native_components": native_components,
    }


def copy_notices(paths: list[Path], destination: Path) -> list[dict[str, str]]:
    """Retain every distinct notice, including equal basenames in subdirectories."""
    notices = []
    seen: set[str] = set()
    for path in sorted(paths):
        if not path.is_file():
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        destination.mkdir(parents=True, exist_ok=True)
        name = f"{digest[:16]}-{path.name}"
        (destination / name).write_bytes(data)
        notices.append({"file": f"{destination.name}/{name}", "sha256": digest})
    return notices


def _notice_name(path: Path) -> bool:
    return path.name.lower().startswith(("license", "copying", "copyright", "notice"))


def build_inventory(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    locked = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)",
                            (ROOT / "requirements/runtime.txt").read_text(encoding="utf-8"),
                            re.MULTILINE))
    names = set(locked)
    # These supply the frozen bootloader and Windows taskbar extension payloads.
    names.update(("pyinstaller", "pywin32"))
    python_components = []
    for name in sorted(names, key=str.casefold):
        dist = metadata.distribution(name)
        if name in locked and dist.version != locked[name]:
            raise ValueError(f"{name}: installed version does not match runtime lock")
        component = f"python-{dist.metadata['Name']}-{dist.version}"
        files = [Path(str(dist.locate_file(file))) for file in (dist.files or [])
                 if _notice_name(Path(str(file)))]
        python_components.append({
            "name": dist.metadata["Name"], "version": dist.version,
            "license": dist.metadata.get("License-Expression") or dist.metadata.get("License"),
            "source": f"https://pypi.org/project/{name}/{dist.version}/#files",
            "notices": copy_notices(files, output / component),
        })

    cargo = json.loads(subprocess.run(
        ["cargo", "metadata", "--locked", "--format-version", "1",
         "--filter-platform", "x86_64-pc-windows-msvc", "--features", "extension-module"],
        cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout)
    resolved = {node["id"] for node in cargo["resolve"]["nodes"]}
    rust_components = []
    for package in sorted(cargo["packages"], key=lambda item: (item["name"], item["version"])):
        if package["id"] not in resolved or package["name"] == "mic_eq_core":
            continue
        directory = Path(package["manifest_path"]).parent
        paths = [path for path in directory.rglob("*") if _notice_name(path) and path.is_file()]
        rust_components.append({
            "name": package["name"], "version": package["version"],
            "license": package["license"],
            "source": f"https://crates.io/crates/{package['name']}/{package['version']}",
            "notices": copy_notices(paths, output / f"rust-{package['name']}-{package['version']}"),
        })
    existing_rust = {
        (str(component["name"]).casefold(), str(component["version"]))
        for component in rust_components
    }
    rust_components.extend(_deepfilter_rust_components(output, existing_rust))
    runtime = json.loads((ROOT / "release-assets.json").read_text(encoding="utf-8"))
    source_status = source_distribution_status(output)
    inventory = {
        "schema_version": 1, "version": project["project"]["version"],
        "distribution_license": "GPL-3.0-only", "original_source_license": "MIT",
        "scope": "Locked Python runtime, frozen bootloader, Windows extension, and resolved Windows Rust build graph; Rust entries can include build-only dependencies.",
        "python": {"version": sys.version.split()[0], "source": f"https://www.python.org/downloads/release/python-{sys.version_info.major}{sys.version_info.minor}{sys.version_info.micro}/",
                   "notices": copy_notices([Path(sys.base_prefix) / "LICENSE.txt"], output / "cpython")},
        "python_components": python_components, "rust_components": rust_components,
        "native_assets": runtime["assets"],
        "source_distribution": source_status,
        "native_components": source_status.get("native_components", []),
    }
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "dependency-licenses")
    args = parser.parse_args()
    output = args.output.resolve()
    # Generated notices are build inputs; refuse to overwrite a source directory.
    if not output.is_relative_to(ROOT / "build") or output == ROOT / "build":
        parser.error("--output must be a subdirectory of the repository build directory")
    inventory = build_inventory(output)
    print(f"Collected {len(inventory['python_components'])} Python and "
          f"{len(inventory['rust_components'])} Rust dependency records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
