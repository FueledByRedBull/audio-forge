"""Download verified release runtime assets into the local workspace."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
from typing import Mapping, TypedDict

from verify_release_assets import (
    PINNED_ARCHIVE_HOSTS,
    PINNED_ARCHIVE_STATUS,
    SOURCE_BUILD_STATUS,
    AssetManifest,
    load_asset_manifest,
    verify_assets,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "release-assets.json"


class AssetPlan(TypedDict):
    name: str
    destination: Path
    archive_path: Path
    pinned_archive: bool
    direct_url: str | None


def _default_asset_source_tag() -> str:
    try:
        manifest = load_asset_manifest(MANIFEST_PATH, require_assets=False)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    tag = manifest.fallback_release_tag
    if tag is None:
        raise RuntimeError(
            "release-assets.json must define a non-empty fallback_release_tag"
        )
    return tag


def _manifest_entries() -> dict[str, dict[str, object]]:
    try:
        return load_asset_manifest(MANIFEST_PATH).entries
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _manifest_assets(
    manifest: AssetManifest, *, only_cpu_runtime: bool
) -> list[AssetPlan]:
    assets: list[AssetPlan] = []
    for entry in manifest.assets:
        raw_path = entry["path"]
        assert isinstance(raw_path, str)
        origin = entry.get("origin")
        status = origin.get("status") if isinstance(origin, dict) else None
        is_cpu_runtime = raw_path.startswith("target/onnxruntime-cpu/lib/")
        if only_cpu_runtime and not is_cpu_runtime:
            continue
        destination = Path(raw_path)
        assets.append(
            {
                "name": destination.name,
                "destination": destination,
                "archive_path": Path("_internal") / destination,
                "pinned_archive": status == PINNED_ARCHIVE_STATUS,
                "direct_url": (
                    entry.get("source")
                    if status == "pinned-upstream-model"
                    else None
                ),
            }
        )
    if only_cpu_runtime and not assets:
        raise RuntimeError(
            "release-assets.json contains no pinned CPU ONNX Runtime assets"
        )
    return assets


def _source_build_entry(
    asset: Mapping[str, object], entries: dict[str, dict[str, object]]
) -> dict[str, object] | None:
    manifest_entry = entries.get(str(asset["destination"]).replace("\\", "/"))
    if not manifest_entry:
        return None
    origin = manifest_entry.get("origin")
    if isinstance(origin, dict) and origin.get("status") == SOURCE_BUILD_STATUS:
        return manifest_entry
    return None


def _run(command: list[str], *, capture: bool = False) -> str:
    kwargs = {
        "cwd": REPO_ROOT,
        "check": True,
        "text": True,
    }
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    completed = subprocess.run(command, **kwargs)
    return completed.stdout if capture else ""


def _find_7z() -> str:
    candidate = Path("C:/Program Files/7-Zip/7z.exe")
    if candidate.exists():
        return str(candidate)
    seven_zip = shutil.which("7z")
    if seven_zip:
        return seven_zip
    raise RuntimeError("7-Zip was not found. Install 7-Zip or add 7z to PATH.")


def _release_asset_names(tag: str, repo: str) -> set[str]:
    output = _run(
        ["gh", "release", "view", tag, "--repo", repo, "--json", "assets", "--jq", ".assets[].name"],
        capture=True,
    )
    return {line.strip() for line in output.splitlines() if line.strip()}


def _download_asset(tag: str, repo: str, pattern: str, destination_dir: Path) -> None:
    _run(
        [
            "gh",
            "release",
            "download",
            tag,
            "--repo",
            repo,
            "--pattern",
            pattern,
            "--dir",
            str(destination_dir),
            "--clobber",
        ]
    )


def _download_direct_url(url: str, destination: Path) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "raw.githubusercontent.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.fragment
    ):
        raise ValueError(
            "direct release assets must use trusted raw.githubusercontent.com HTTPS"
        )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AudioForge-release-assets"},
    )
    with (
        urllib.request.urlopen(request, timeout=60) as response,
        destination.open("wb") as output,
    ):
        shutil.copyfileobj(response, output)


def _validate_pinned_archive_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in PINNED_ARCHIVE_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.fragment
    ):
        raise ValueError(
            "pinned runtime archives must use trusted GitHub HTTPS hosts"
        )


class _TrustedArchiveRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        target = urllib.parse.urljoin(req.full_url, newurl)
        _validate_pinned_archive_url(target)
        return super().redirect_request(req, fp, code, msg, headers, target)


def _manifest_pinned_archive_entry(
    asset: Mapping[str, object], entries: dict[str, dict[str, object]]
) -> dict[str, object] | None:
    if not asset.get("pinned_archive"):
        return None
    raw_destination = str(asset["destination"]).replace("\\", "/")
    entry = entries.get(raw_destination)
    if entry is None:
        raise RuntimeError(f"Pinned archive manifest entry is missing: {raw_destination}")
    origin = entry.get("origin")
    if not isinstance(origin, dict) or origin.get("status") != PINNED_ARCHIVE_STATUS:
        raise RuntimeError(
            f"{raw_destination} must declare origin.status={PINNED_ARCHIVE_STATUS}"
        )
    return entry


def _download_pinned_archive(
    manifest_entry: dict[str, object], temporary: Path, cache: dict[str, Path]
) -> Path:
    origin = manifest_entry.get("origin")
    if not isinstance(origin, dict):
        raise RuntimeError("Pinned archive entry has no origin object")
    raw_url = origin.get("archive_url") or manifest_entry.get("source")
    expected_sha = origin.get("archive_sha256")
    expected_size = origin.get("archive_size")
    if (
        not isinstance(raw_url, str)
        or not raw_url
        or not isinstance(expected_sha, str)
        or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha)
        or type(expected_size) is not int
        or expected_size <= 0
    ):
        raise RuntimeError("Pinned archive entry has invalid URL, SHA-256, or byte count")
    _validate_pinned_archive_url(raw_url)
    cache_key = f"{raw_url}|{expected_sha.lower()}|{expected_size}"
    if cache_key in cache:
        return cache[cache_key]

    filename = Path(urllib.parse.urlsplit(raw_url).path).name
    if not filename:
        raise RuntimeError("Pinned archive URL has no filename")
    archive_path = temporary / filename
    staged_path = temporary / f".{filename}.{os.getpid()}.part"
    opener = urllib.request.build_opener(_TrustedArchiveRedirectHandler)
    request = urllib.request.Request(
        raw_url,
        headers={"User-Agent": "AudioForge-release-assets"},
    )
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with opener.open(request, timeout=120) as response, staged_path.open("xb") as output:
            final_url = response.geturl()
            _validate_pinned_archive_url(final_url)
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                if byte_count + len(chunk) > expected_size:
                    raise RuntimeError(
                        f"Pinned archive exceeded its declared size of {expected_size} bytes"
                    )
                output.write(chunk)
                digest.update(chunk)
                byte_count += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        actual_sha = digest.hexdigest()
        if byte_count != expected_size or actual_sha.lower() != expected_sha.lower():
            raise RuntimeError(
                f"Pinned archive verification failed: expected {expected_size} bytes/{expected_sha}, "
                f"got {byte_count} bytes/{actual_sha}"
            )
        staged_path.replace(archive_path)
    finally:
        staged_path.unlink(missing_ok=True)
    cache[cache_key] = archive_path
    return archive_path


def _normalise_zip_member(name: str) -> str:
    if "\x00" in name:
        raise RuntimeError("ZIP archive contains a NUL byte in a member path")
    portable = name.replace("\\", "/")
    posix = PurePosixPath(portable)
    windows = PureWindowsPath(portable)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        raise RuntimeError(f"ZIP archive contains an unsafe member path: {name!r}")
    parts = tuple(part for part in portable.split("/") if part not in {"", "."})
    if not parts:
        raise RuntimeError("ZIP archive contains an empty member path")
    return "/".join(parts)


def _extract_pinned_zip_member(
    archive_path: Path,
    extracted_root: Path,
    relative_member: str,
) -> Path:
    expected_member = _normalise_zip_member(relative_member)
    extracted_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            normalised = _normalise_zip_member(info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise RuntimeError(
                    f"ZIP archive contains a symlink member: {info.filename!r}"
                )
            if normalised in members:
                raise RuntimeError(f"ZIP archive contains duplicate member: {normalised}")
            members[normalised] = info
        info = members.get(expected_member)
        if info is None or info.is_dir():
            raise RuntimeError(
                f"Archive '{archive_path.name}' did not contain file '{expected_member}'."
            )
        target = extracted_root.joinpath(*expected_member.split("/"))
        if not target.resolve().is_relative_to(extracted_root.resolve()):
            raise RuntimeError(f"ZIP member escaped extraction root: {expected_member}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)
    return target


def _extract_archive_asset(
    archive_path: Path,
    extracted_root: Path,
    relative_asset_path: Path,
) -> Path:
    seven_zip = _find_7z()
    if not extracted_root.exists():
        extracted_root.mkdir(parents=True, exist_ok=True)
        _run([seven_zip, "x", str(archive_path), f"-o{extracted_root}", "-y"])

    extracted_asset = extracted_root / relative_asset_path
    if not extracted_asset.exists():
        raise RuntimeError(
            f"Archive '{archive_path.name}' did not contain '{relative_asset_path.as_posix()}'."
        )
    return extracted_asset


def _build_source_asset(
    asset: Mapping[str, object], manifest_entry: dict[str, object], temporary: Path
) -> Path:
    origin = manifest_entry.get("origin")
    if not isinstance(origin, dict):
        raise RuntimeError(f"{asset['name']} source-build manifest entry has no origin object")
    raw_attestation = origin.get("attestation_path")
    if not isinstance(raw_attestation, str) or not raw_attestation:
        raise RuntimeError(
            f"{asset['name']} source-build manifest entry has no attestation_path"
        )
    attestation_relative = Path(raw_attestation.replace("\\", "/"))
    if attestation_relative.is_absolute() or ".." in attestation_relative.parts:
        raise RuntimeError("source-build attestation_path must stay inside the repository")
    attestation = REPO_ROOT / attestation_relative
    output = temporary / str(asset["name"])
    output.parent.mkdir(parents=True, exist_ok=True)
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if not powershell:
        raise RuntimeError("PowerShell is required to build the pinned DeepFilter source asset.")
    _run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(REPO_ROOT / "build_deepfilter.ps1"),
            "-OutputPath",
            str(output),
            "-AttestationPath",
            str(attestation),
        ]
    )
    if not output.is_file():
        raise RuntimeError(f"Source build completed without producing {output}")
    if not attestation.is_file():
        raise RuntimeError(f"Source build completed without producing {raw_attestation}")
    return output


def _atomic_copy(source: Path, destination: Path) -> None:
    staged = destination.with_name(f".{destination.name}.copy-{os.getpid()}")
    try:
        if staged.exists():
            staged.unlink()
        shutil.copy2(source, staged)
        os.replace(staged, destination)
    finally:
        if staged.exists():
            staged.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download AudioForge runtime assets from a GitHub release."
    )
    parser.add_argument(
        "--release-tag",
        default=_default_asset_source_tag(),
        help=(
            "Published release used as a fallback for pinned assets. "
            "Defaults to release-assets.json fallback_release_tag."
        ),
    )
    parser.add_argument(
        "--repo",
        default="FueledByRedBull/audio-forge",
        help="GitHub repository in owner/name form.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite local assets even if destination files already exist.",
    )
    parser.add_argument(
        "--only-cpu-runtime",
        "--only-onnxruntime",
        dest="only_cpu_runtime",
        action="store_true",
        help="Hydrate and verify only the pinned CPU ONNX Runtime files.",
    )
    args = parser.parse_args()

    try:
        manifest = load_asset_manifest(MANIFEST_PATH)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    assets = _manifest_assets(manifest, only_cpu_runtime=args.only_cpu_runtime)
    manifest_entries = manifest.entries
    if shutil.which("gh") is None:
        pending_release_assets = [
            asset
            for asset in assets
            if not (
                (REPO_ROOT / asset["destination"]).exists() and not args.force
            )
            and not asset.get("direct_url")
            and _source_build_entry(asset, manifest_entries) is None
            and _manifest_pinned_archive_entry(asset, manifest_entries) is None
        ]
        if pending_release_assets:
            raise RuntimeError("GitHub CLI 'gh' is required for release asset hydration.")

    pending_release_assets = [
        asset
        for asset in assets
        if not ((REPO_ROOT / asset["destination"]).exists() and not args.force)
        and not asset.get("direct_url")
        and _source_build_entry(asset, manifest_entries) is None
        and _manifest_pinned_archive_entry(asset, manifest_entries) is None
    ]
    asset_names = (
        _release_asset_names(args.release_tag, args.repo) if pending_release_assets else set()
    )
    archive_name = next(
        (name for name in sorted(asset_names) if name.startswith("AudioForge-") and name.endswith("-win64-ultra.7z")),
        None,
    )

    with tempfile.TemporaryDirectory(prefix="audioforge-release-assets-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        extracted_root = temp_dir / "archive-extract"
        archive_path: Path | None = None
        pinned_archive_paths: dict[str, Path] = {}
        pinned_extracted_root = temp_dir / "pinned-archive-extract"

        ordered_assets = [
            asset for asset in assets if _source_build_entry(asset, manifest_entries) is None
        ] + [asset for asset in assets if _source_build_entry(asset, manifest_entries) is not None]
        for asset in ordered_assets:
            destination = REPO_ROOT / asset["destination"]
            if destination.exists() and not args.force:
                print(f"Skipping existing {destination.relative_to(REPO_ROOT)}")
                continue

            source_build_entry = _source_build_entry(asset, manifest_entries)
            if source_build_entry is not None:
                source = _build_source_asset(asset, source_build_entry, temp_dir)
                destination.parent.mkdir(parents=True, exist_ok=True)
                _atomic_copy(source, destination)
                print(f"Built {asset['name']} from pinned source -> {destination.relative_to(REPO_ROOT)}")
                continue

            pinned_archive_entry = _manifest_pinned_archive_entry(asset, manifest_entries)
            if pinned_archive_entry is not None:
                origin = pinned_archive_entry["origin"]
                assert isinstance(origin, dict)
                archive_member = origin.get("archive_member")
                if not isinstance(archive_member, str) or not archive_member:
                    raise RuntimeError(
                        f"{asset['name']} pinned archive entry has no archive_member"
                    )
                archive = _download_pinned_archive(
                    pinned_archive_entry, temp_dir, pinned_archive_paths
                )
                source = _extract_pinned_zip_member(
                    archive, pinned_extracted_root, archive_member
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                _atomic_copy(source, destination)
                print(
                    f"Installed {asset['name']} from pinned CPU ONNX Runtime archive -> "
                    f"{destination.relative_to(REPO_ROOT)}"
                )
                continue

            direct_url = asset.get("direct_url")
            if isinstance(direct_url, str) and direct_url:
                source = temp_dir / asset["name"]
                _download_direct_url(direct_url, source)
            elif asset["name"] in asset_names:
                _download_asset(args.release_tag, args.repo, asset["name"], temp_dir)
                source = temp_dir / asset["name"]
            else:
                if archive_name is None:
                    raise RuntimeError(
                        f"Release '{args.release_tag}' is missing raw asset '{asset['name']}' "
                        "and no release archive fallback is available."
                    )
                if archive_path is None:
                    _download_asset(args.release_tag, args.repo, archive_name, temp_dir)
                    archive_path = temp_dir / archive_name
                source = _extract_archive_asset(archive_path, extracted_root, asset["archive_path"])

            destination.parent.mkdir(parents=True, exist_ok=True)
            _atomic_copy(source, destination)
            print(f"Installed {asset['name']} -> {destination.relative_to(REPO_ROOT)}")

    selected_paths = (
        {str(asset["destination"]).replace("\\", "/") for asset in assets}
        if args.only_cpu_runtime
        else None
    )
    verification_errors = verify_assets(MANIFEST_PATH, selected_paths=selected_paths)
    if verification_errors:
        formatted_errors = "\n  ".join(verification_errors)
        raise RuntimeError(f"Downloaded assets failed verification:\n  {formatted_errors}")

    print("Release assets downloaded successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"fetch_release_assets.py failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
