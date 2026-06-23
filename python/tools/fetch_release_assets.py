"""Download verified release runtime assets into the local workspace."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

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
    },
]


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


def _resolve_default_tag() -> str:
    import tomllib

    pyproject = REPO_ROOT / "pyproject.toml"
    with pyproject.open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]
    return f"v{version}"


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download AudioForge runtime assets from a GitHub release."
    )
    parser.add_argument(
        "--release-tag",
        default=_resolve_default_tag(),
        help="Release tag to read assets from. Defaults to v<pyproject version>.",
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
        raise RuntimeError("GitHub CLI 'gh' is required for fetch_release_assets.py.")

    asset_names = _release_asset_names(args.release_tag, args.repo)
    archive_name = next(
        (name for name in sorted(asset_names) if name.startswith("AudioForge-") and name.endswith("-win64-ultra.7z")),
        None,
    )

    with tempfile.TemporaryDirectory(prefix="audioforge-release-assets-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        extracted_root = temp_dir / "archive-extract"
        archive_path: Path | None = None

        for asset in ASSETS:
            destination = REPO_ROOT / asset["destination"]
            if destination.exists() and not args.force:
                print(f"Skipping existing {destination.relative_to(REPO_ROOT)}")
                continue

            if asset["name"] in asset_names:
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
            shutil.copy2(source, destination)
            print(f"Installed {asset['name']} -> {destination.relative_to(REPO_ROOT)}")

    print("Release assets downloaded successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"fetch_release_assets.py failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
