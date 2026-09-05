"""Download verified release runtime assets into the local workspace."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from verify_release_assets import verify_assets


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "release-assets.json"

ASSETS = [
    {
        "name": "df.dll",
        "destination": Path("df.dll"),
        "archive_path": Path("_internal/df.dll"),
    },
    {
        "name": "DirectML.dll",
        "destination": Path("target/release/DirectML.dll"),
        "archive_path": Path("_internal/DirectML.dll"),
    },
    {
        "name": "DeepFilterNet3_ll_onnx.tar.gz",
        "destination": Path("models/DeepFilterNet3_ll_onnx.tar.gz"),
        "archive_path": Path("_internal/models/DeepFilterNet3_ll_onnx.tar.gz"),
    },
    {
        "name": "DeepFilterNet3_onnx.tar.gz",
        "destination": Path("models/DeepFilterNet3_onnx.tar.gz"),
        "archive_path": Path("_internal/models/DeepFilterNet3_onnx.tar.gz"),
    },
    {
        "name": "silero_vad.onnx",
        "destination": Path("models/silero_vad.onnx"),
        "archive_path": Path("_internal/models/silero_vad.onnx"),
        "direct_url": (
            "https://raw.githubusercontent.com/snakers4/silero-vad/"
            "v6.2.1/src/silero_vad/data/silero_vad.onnx"
        ),
    },
]
SOURCE_BUILD_STATUS = "verified-source-build"


def _default_asset_source_tag() -> str:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    tag = manifest.get("fallback_release_tag")
    if not isinstance(tag, str) or not tag.startswith("v") or not tag[1:]:
        raise RuntimeError(
            "release-assets.json must define a non-empty fallback_release_tag"
        )
    return tag


def _manifest_entries() -> dict[str, dict[str, object]]:
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assets = raw.get("assets")
    if not isinstance(assets, list):
        raise RuntimeError("release-assets.json must contain an assets list")
    entries: dict[str, dict[str, object]] = {}
    for entry in assets:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            continue
        entries[entry["path"].replace("\\", "/")] = entry
    return entries


def _source_build_entry(
    asset: dict[str, object], entries: dict[str, dict[str, object]]
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
    asset: dict[str, object], manifest_entry: dict[str, object], temporary: Path
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
    args = parser.parse_args()

    if shutil.which("gh") is None:
        pending_release_assets = [
            asset
            for asset in ASSETS
            if not (
                (REPO_ROOT / asset["destination"]).exists() and not args.force
            )
            and not asset.get("direct_url")
            and _source_build_entry(asset, _manifest_entries()) is None
        ]
        if pending_release_assets:
            raise RuntimeError("GitHub CLI 'gh' is required for release asset hydration.")

    manifest_entries = _manifest_entries()
    pending_release_assets = [
        asset
        for asset in ASSETS
        if not ((REPO_ROOT / asset["destination"]).exists() and not args.force)
        and not asset.get("direct_url")
        and _source_build_entry(asset, manifest_entries) is None
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

        ordered_assets = [
            asset for asset in ASSETS if _source_build_entry(asset, manifest_entries) is None
        ] + [asset for asset in ASSETS if _source_build_entry(asset, manifest_entries) is not None]
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

    verification_errors = verify_assets()
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
