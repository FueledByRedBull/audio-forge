"""Create and hydrate the corresponding-source manifest for a release build.

The manifest is deliberately separate from ``license_inventory.py``.  The
inventory records notices found in the build environment; this tool records
the exact source archives and hashes needed to reconstruct the distributed
runtime.  A manifest can remain incomplete while a native binary still lacks
reproducible source provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse
import urllib.error
import urllib.request
import tomllib


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "licenses" / "source-manifest.json"
DEFAULT_SOURCE_DIR = ROOT / "build" / "source-distribution"
PYPI_JSON = "https://pypi.org/pypi/{name}/{version}/json"
TRUSTED_SOURCE_HOSTS = frozenset(
    {
        "archive.mesa3d.org",
        "api.nuget.org",
        "codeload.github.com",
        "crates.io",
        "download.qt.io",
        "files.pythonhosted.org",
        "github.com",
        "release-assets.githubusercontent.com",
        "raw.githubusercontent.com",
        "static.crates.io",
        "qt.mirror.constant.com",
        "wiki.qt.io",
        "www.openssl.org",
        "www.python.org",
        "pypi.org",
    }
)
EXPECTED_CPYTHON_VERSION = "3.12.10"
PYPI_SDIST_PACKAGES = (
    ("pyqt6", "PyQt6"),
    ("pyqt6-sip", "PyQt6-sip"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("pyinstaller", "PyInstaller"),
)

# PyQt6 6.11.0's sdist declares these build requirements.  They are build
# tools rather than shipped runtime libraries, but retaining their exact
# sdists makes the PyQt binding rebuild instructions actionable.
PYQT_BUILD_SOURCES = (
    ("sip", "6.16.1"),
    ("PyQt-builder", "1.19.1"),
)

PYWIN32_SOURCE = {
    "name": "pywin32",
    "version": "311",
    "filename": "pywin32-b311.tar.gz",
    "url": "https://github.com/mhammond/pywin32/archive/refs/tags/b311.tar.gz",
    "sha256": "4ae3f0d4adacc331733fe1204f7f93b8654615ca325adfc9eb6c67198246a91b",
    "source_of_truth": "https://github.com/mhammond/pywin32/tree/b311",
}

PYTHON_SOURCE = {
    "name": "CPython",
    "filename_template": "Python-{version}.tar.xz",
    "url_template": "https://www.python.org/ftp/python/{version}/Python-{version}.tar.xz",
    "sha256_by_version": {
        "3.12.10": "07ab697474595e06f06647417d3c7fa97ded07afc1a7e4454c5639919b46eaea",
    },
}

QT_MODULE_SOURCES = (
    {
        "module": "qtbase",
        "description": "Qt base (Core, Gui, Network, Widgets, Windows platform plugin)",
        "sha256": "d9594a31228aa23ad6b531719a29b45f0f3989fe6c136d45767ea179f233c1ac",
    },
    {
        "module": "qtimageformats",
        "description": "Qt image format plugins shipped in the Windows bundle",
        "sha256": "b2bf6c6845ac175ed7f819145483ba4676f617aaa6a5012c8efee63c8bbac413",
    },
)
QT_SOURCE = {
    "filename_template": "qtbase-everywhere-src-{version}.tar.xz",
    "url_template": (
        "https://download.qt.io/official_releases/qt/{minor}/{version}/"
        "submodules/qtbase-everywhere-src-{version}.tar.xz"
    ),
}

DEEPFILTER_SOURCE = {
    "name": "DeepFilterNet",
    "version": "d375b2d8309e0935d165700c91da9de862a99c31",
    "filename": "DeepFilterNet-d375b2d8309e0935d165700c91da9de862a99c31.tar.gz",
    "url": (
        "https://codeload.github.com/Rikorose/DeepFilterNet/"
        "tar.gz/d375b2d8309e0935d165700c91da9de862a99c31"
    ),
    "sha256": "49471f3633a24c097d82f3b0d2dbd83a0c1bac3e2f6f6c9a675ef0020ebe5c51",
    "source_of_truth": "https://github.com/Rikorose/DeepFilterNet/commit/d375b2d8309e0935d165700c91da9de862a99c31",
    "license_paths": [
        "DeepFilterNet-d375b2d8309e0935d165700c91da9de862a99c31/LICENSE",
        "DeepFilterNet-d375b2d8309e0935d165700c91da9de862a99c31/LICENSE-APACHE",
        "DeepFilterNet-d375b2d8309e0935d165700c91da9de862a99c31/LICENSE-MIT",
    ],
}

DEEPFILTER_TRACT_SOURCE = {
    "name": "tract-linalg (patched build input)",
    "version": "0.21.17",
    "filename": "tract-linalg-0.21.17.crate",
    "url": "https://crates.io/api/v1/crates/tract-linalg/0.21.17/download",
    "sha256": "5a4f0f9c134fa f99e3cf78b46d797398982e04ad46a3ad5223bcf8c0bd4af360".replace(" ", ""),
    "source_of_truth": "https://crates.io/crates/tract-linalg/0.21.17",
    "license_paths": [
        "tract-linalg-0.21.17/LICENSE",
        "tract-linalg-0.21.17/LICENSE-APACHE",
        "tract-linalg-0.21.17/LICENSE-MIT",
    ],
}

CPYTHON_EXTERNAL_SOURCES = (
    ("bzip2", "1.0.8", "ab8d1b0cc087c20d4c32c0e4fcf7d0c733a95da12cedc6d63b3f0a9af07427e2"),
    ("libffi", "3.4.4", "9d802681adfea27d84cae0487a785fb9caa925bdad44c401b364c59ab2b8edda"),
    ("openssl", "3.0.16", "6bb739ecddbd2cfb6d255eb5898437a9b5739277dee931338d3275bac5d96ba2"),
    ("sqlite", "3.49.1.0", "e335aeb44fa36cde60ecbb6a9f8be6f5d449d645ce9b0199ee53a7e6728d19d2"),
    ("xz", "5.2.5", "a15c168e39e87d750c3dc766edc7f19bdda57dacf01e509678467eace91ad282"),
    ("zlib", "1.3.1", "e3f3fb32564952006eb18b091ca8464740e5eca29d328cfb0b2da22768e0b638"),
)

OPENSSL_UPSTREAM_SOURCE = {
    "name": "OpenSSL",
    "version": "3.0.16",
    "filename": "openssl-3.0.16.tar.gz",
    "url": "https://www.openssl.org/source/openssl-3.0.16.tar.gz",
    "sha256": "57e03c50feab5d31b152af2b764f10379aecd8ee92f16c985983ce4a99f7ef86",
    "source_of_truth": "https://www.openssl.org/source/openssl-3.0.16.tar.gz.sha256",
    "license_paths": ["openssl-3.0.16/LICENSE.txt"],
}

OPENBLAS_SOURCES = (
    {
        "id": "openblas-recipe-scipy-0.3.31.22.0",
        "name": "SciPy OpenBLAS build recipe",
        "version": "0.3.31.22.0",
        "filename": "openblas-libs-0.3.31.22.0.tar.gz",
        "url": "https://github.com/MacPython/openblas-libs/archive/refs/tags/v0.3.31.22.0.tar.gz",
        "sha256": "cc4b804fa064c543fd129a21ee8d7379447e18bb24c9a039005096e867187a51",
        "source_of_truth": "https://github.com/MacPython/openblas-libs/tree/v0.3.31.22.0",
        "build_role": "exact SciPy 1.18.0 Windows OpenBLAS source and patch recipe",
        "license_paths": ["openblas-libs-0.3.31.22.0/LICENSE.txt"],
    },
    {
        "id": "openblas-upstream-9bdf051b",
        "name": "OpenBLAS upstream (NumPy ILP64)",
        "version": "9bdf051b96e956f848dfcef89c23e1993b0e1b3e",
        "filename": "OpenBLAS-9bdf051b96e956f848dfcef89c23e1993b0e1b3e.tar.gz",
        "url": "https://github.com/OpenMathLib/OpenBLAS/archive/9bdf051b96e956f848dfcef89c23e1993b0e1b3e.tar.gz",
        "sha256": "c2cb4ccf65724f363b58bc96e9b68f661a56ea6da048fdb5dbe65997f48d2ac4",
        "source_of_truth": "https://github.com/OpenMathLib/OpenBLAS/commit/9bdf051b96e956f848dfcef89c23e1993b0e1b3e",
        "build_role": "OpenBLAS source used by NumPy 2.5.1's scipy-openblas64 build",
        "license_paths": ["OpenBLAS-9bdf051b96e956f848dfcef89c23e1993b0e1b3e/LICENSE"],
    },
    {
        "id": "openblas-recipe-numpy-0.3.33.112.0",
        "name": "NumPy OpenBLAS build recipe",
        "version": "0.3.33.112.0",
        "filename": "openblas-libs-0.3.33.112.0.tar.gz",
        "url": "https://github.com/MacPython/openblas-libs/archive/refs/tags/v0.3.33.112.0.tar.gz",
        "sha256": "557f08e84c0ea58c5020453c101ba5c8dea71ce923650a9d57d9c49e3d844e8c",
        "source_of_truth": "https://github.com/MacPython/openblas-libs/tree/v0.3.33.112.0",
        "build_role": "exact NumPy 2.5.1 Windows OpenBLAS source and patch recipe",
        "license_paths": ["openblas-libs-0.3.33.112.0/LICENSE.txt"],
    },
    {
        "id": "openblas-upstream-5ffbf38b",
        "name": "OpenBLAS upstream (SciPy LP64)",
        "version": "5ffbf38b41a1fe494d038efcb09cf22f4d527c22",
        "filename": "OpenBLAS-5ffbf38b41a1fe494d038efcb09cf22f4d527c22.tar.gz",
        "url": "https://github.com/OpenMathLib/OpenBLAS/archive/5ffbf38b41a1fe494d038efcb09cf22f4d527c22.tar.gz",
        "sha256": "51257fb2f0aa7b4c23d9cf0302500b817dc82c0022ee681b24f9ab82dfd10312",
        "source_of_truth": "https://github.com/OpenMathLib/OpenBLAS/commit/5ffbf38b41a1fe494d038efcb09cf22f4d527c22",
        "build_role": "OpenBLAS source used by SciPy 1.18.0's scipy-openblas32 build",
        "license_paths": ["OpenBLAS-5ffbf38b41a1fe494d038efcb09cf22f4d527c22/LICENSE"],
    },
)

MESA_SOURCE = {
    "name": "Mesa llvmpipe",
    "version": "11.2.2",
    "filename": "mesa-11.2.2.tar.xz",
    "url": "https://archive.mesa3d.org/older-versions/11.x/11.2.2/mesa-11.2.2.tar.xz",
    "sha256": "40e148812388ec7c6d7b6657d5a16e2e8dabba8b97ddfceea5197947647bdfb4",
    "source_of_truth": "https://wiki.qt.io/MesaLlvmpipe",
    "build_role": "source for Qt's bundled opengl32sw.dll Mesa software rasterizer",
    "license_paths": ["mesa-11.2.2/docs/COPYING"],
}

DEEPFILTER_RECIPE_FILES = (
    "build_deepfilter.ps1",
    "build-support/deepfilter/Cargo.toml",
    "build-support/deepfilter/Cargo.lock",
    "build-support/deepfilter/provenance.json",
)


class SourceDistributionError(RuntimeError):
    """Raised when a source archive cannot be trusted or reproduced."""


def _validate_source_url(url: str) -> None:
    """Allow only HTTPS URLs on hosts whose content is pinned below."""
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError as exc:
        raise SourceDistributionError(f"Invalid source URL: {url!r}") from exc
    hostname = parsed.hostname.casefold() if parsed.hostname else ""
    if (
        parsed.scheme.casefold() != "https"
        or parsed.username
        or parsed.password
        or port is not None
        or hostname not in TRUSTED_SOURCE_HOSTS
    ):
        raise SourceDistributionError(
            f"Source URL must use trusted HTTPS: {url!r}"
        )


class _TrustedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject an untrusted redirect before urllib opens the next hop."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        target = urljoin(req.full_url, newurl)
        _validate_source_url(target)
        return super().redirect_request(req, fp, code, msg, headers, target)


def _open_source_url(request: urllib.request.Request, *, timeout: int):
    """Open a pinned source URL while validating every redirect hop."""
    opener = urllib.request.build_opener(_TrustedRedirectHandler)
    return opener.open(request, timeout=timeout)


def _read_locked_versions(path: Path) -> dict[str, str]:
    """Read exact ``name==version`` pins from a pip-compile lock file."""
    versions: dict[str, str] = {}
    pattern = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            versions[match.group(1).casefold()] = match.group(2)
    return versions


def _read_locked_hashes(path: Path) -> dict[str, set[str]]:
    """Read all pip hash pins grouped by normalized package name."""
    hashes: dict[str, set[str]] = {}
    current: str | None = None
    package_pattern = re.compile(r"^([A-Za-z0-9_.-]+)==")
    hash_pattern = re.compile(r"--hash=sha256:([0-9a-fA-F]{64})")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = package_pattern.match(line)
        if match:
            package_name = match.group(1).casefold()
            current = package_name
            hashes.setdefault(package_name, set())
        if current is not None:
            hashes[current].update(hash_pattern.findall(line))
    return hashes


def _fetch_json(url: str) -> dict[str, Any]:
    _validate_source_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "AudioForge source manifest"})
    try:
        with _open_source_url(request, timeout=60) as response:
            final_url = response.geturl() if hasattr(response, "geturl") else url
            if not isinstance(final_url, str):
                raise SourceDistributionError("Metadata redirect did not provide a URL")
            _validate_source_url(final_url)
            payload = json.load(response)
    except (OSError, ValueError) as exc:
        raise SourceDistributionError(f"Could not read metadata from {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SourceDistributionError(f"Metadata at {url} was not an object")
    return payload


def _pypi_sdist(package: str, version: str) -> dict[str, str]:
    """Resolve the immutable PyPI sdist URL and digest for one pin."""
    metadata = _fetch_json(PYPI_JSON.format(name=package, version=version))
    files = metadata.get("urls")
    if not isinstance(files, list):
        raise SourceDistributionError(f"PyPI metadata for {package} {version} has no files")
    candidates = [
        item
        for item in files
        if isinstance(item, dict) and item.get("packagetype") == "sdist"
    ]
    if len(candidates) != 1:
        raise SourceDistributionError(
            f"Expected one sdist for {package} {version}, found {len(candidates)}"
        )
    item = candidates[0]
    filename = item.get("filename")
    url = item.get("url")
    digest = (item.get("digests") or {}).get("sha256")
    if not (
        isinstance(filename, str)
        and filename
        and isinstance(url, str)
        and url
        and isinstance(digest, str)
        and digest
    ):
        raise SourceDistributionError(f"PyPI metadata for {package} {version} is incomplete")
    return {"filename": filename, "url": url, "sha256": digest}


def _crate_entries() -> list[dict[str, Any]]:
    """Return exact registry source archives resolved for the Windows build."""
    command = [
        "cargo",
        "metadata",
        "--locked",
        "--format-version",
        "1",
        "--filter-platform",
        "x86_64-pc-windows-msvc",
        "--features",
        "extension-module",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        metadata = json.loads(completed.stdout)
        lock = tomllib.load((ROOT / "Cargo.lock").open("rb"))["package"]
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as exc:
        raise SourceDistributionError(f"Could not resolve Cargo.lock sources: {exc}") from exc

    resolved_ids = {
        node["id"] for node in metadata.get("resolve", {}).get("nodes", [])
    }
    packages = [
        package
        for package in metadata.get("packages", [])
        if package.get("id") in resolved_ids and package.get("name") != "mic_eq_core"
    ]
    entries: list[dict[str, Any]] = []
    for package in sorted(packages, key=lambda item: (item["name"], item["version"])):
        matches = [
            item
            for item in lock
            if item.get("name") == package.get("name")
            and item.get("version") == package.get("version")
            and item.get("source") == package.get("source")
        ]
        if len(matches) != 1 or not matches[0].get("checksum"):
            raise SourceDistributionError(
                f"Cargo.lock checksum missing for {package.get('name')} {package.get('version')}"
            )
        name = str(package["name"])
        version = str(package["version"])
        entries.append(
            {
                "id": f"cargo-{name}-{version}",
                "kind": "cargo-crate",
                "name": name,
                "version": version,
                "filename": f"{name}-{version}.crate",
                "url": f"https://crates.io/api/v1/crates/{name}/{version}/download",
                "sha256": matches[0]["checksum"],
                "license": package.get("license"),
                "source_of_truth": f"https://crates.io/crates/{name}/{version}",
                "build_role": "resolved Rust dependency",
                "lock_file": "Cargo.lock",
            }
        )
    return entries


def _deepfilter_crate_entries() -> list[dict[str, Any]]:
    """Return registry archives in the pinned DeepFilter build lockfile."""
    lock_path = ROOT / "build-support" / "deepfilter" / "Cargo.lock"
    try:
        packages = tomllib.loads(lock_path.read_text(encoding="utf-8"))["package"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SourceDistributionError(
            f"Could not read DeepFilter Cargo.lock sources: {exc}"
        ) from exc

    entries: list[dict[str, Any]] = []
    for package in sorted(packages, key=lambda item: (item.get("name", ""), item.get("version", ""))):
        source = package.get("source")
        checksum = package.get("checksum")
        if not (
            isinstance(source, str)
            and source.startswith("registry+")
            and isinstance(checksum, str)
            and checksum
        ):
            # The lock retains inactive optional git packages and the local
            # patched tract-linalg package.  The production metadata graph
            # does not resolve the former; the latter is represented below by
            # its immutable crate archive and checked-in patch recipe.
            continue
        name = str(package["name"])
        version = str(package["version"])
        entries.append(
            {
                "id": f"deepfilter-cargo-{name}-{version}",
                "kind": "cargo-crate",
                "name": name,
                "version": version,
                "filename": f"{name}-{version}.crate",
                "url": f"https://crates.io/api/v1/crates/{name}/{version}/download",
                "sha256": checksum,
                "source_of_truth": f"https://crates.io/crates/{name}/{version}",
                "build_role": "resolved DeepFilter C API dependency",
                "lock_file": "build-support/deepfilter/Cargo.lock",
            }
        )
    return entries


def _source_entry(
    *,
    identifier: str,
    kind: str,
    name: str,
    version: str,
    source: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    entry = {
        "id": identifier,
        "kind": kind,
        "name": name,
        "version": version,
        "filename": source["filename"],
        "url": source["url"],
        "sha256": source["sha256"],
        **extra,
    }
    if "license_paths" in source:
        entry["license_paths"] = source["license_paths"]
    return entry


def _native_asset_entries() -> tuple[list[dict[str, Any]], list[str]]:
    """Describe non-source runtime assets without overstating source coverage."""
    asset_file = ROOT / "release-assets.json"
    if not asset_file.exists():
        return [], ["release-assets.json is missing"]
    try:
        assets = json.loads(asset_file.read_text(encoding="utf-8"))["assets"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return [], [f"Could not read release-assets.json: {exc}"]

    entries: list[dict[str, Any]] = []
    blockers: list[str] = []
    for asset in assets:
        path = str(asset.get("path", ""))
        origin = asset.get("origin") if isinstance(asset.get("origin"), dict) else {}
        unresolved = origin.get("status") == "inherited-binary-build-identity-unresolved"
        license_text = str(asset.get("license") or "").casefold()
        restricted_license = "microsoft software license" in license_text or "proprietary" in license_text
        restricted = origin.get("status") in {
            "license-restricted",
            "proprietary",
            "distribution-blocked",
        } or asset.get("distribution_status") in {
            "license-restricted",
            "proprietary",
            "distribution-blocked",
        } or restricted_license
        source_url = asset.get("source")
        source_filename = (
            Path(urlparse(source_url).path).name
            if isinstance(source_url, str)
            else ""
        ) or Path(path).name or "asset"
        entry = {
            "id": f"runtime-{Path(path).name or 'asset'}",
            "kind": "runtime-asset",
            "name": path,
            "version": str(origin.get("version") or origin.get("commit") or "pinned"),
            "filename": source_filename,
            "url": source_url,
            # A package URL (for example DirectML's NuGet archive) hashes the
            # package, while release-assets.json also records the extracted
            # file hash.  Verify the bytes fetched from the URL here and keep
            # the extracted identity separately.
            "sha256": origin.get("package_sha256") or asset.get("sha256"),
            "asset_sha256": asset.get("sha256"),
            "status": "blocked" if unresolved or restricted else "available",
            "source_role": "runtime asset, not application source",
            "license": asset.get("license"),
        }
        if origin.get("status") == "verified-source-build":
            # The binary is a derived build output.  Its source inputs and
            # recipe are distributed separately and the build attestation
            # verifies the candidate output; do not put a workstation-specific
            # binary digest into the corresponding-source receipt.
            entry["source_build"] = True
        entries.append(entry)
        if unresolved:
            blockers.append(
                f"{path}: inherited binary build identity is unresolved; recover the exact source revision, compiler, and recipe"
            )
        if restricted:
            blockers.append(
                f"{path}: upstream distribution terms are not cleared for the GPLv3 binary; remove it or record an approved license"
            )
    return entries, blockers


def build_manifest() -> dict[str, Any]:
    """Resolve the current locked build into a deterministic source manifest."""
    runtime = _read_locked_versions(ROOT / "requirements" / "runtime.txt")
    dev = _read_locked_versions(ROOT / "requirements" / "dev.txt")
    required = {
        "pyqt6": runtime.get("pyqt6"),
        "pyqt6-sip": runtime.get("pyqt6-sip"),
        "pyqt6-qt6": runtime.get("pyqt6-qt6"),
        "numpy": runtime.get("numpy"),
        "scipy": runtime.get("scipy"),
        "pywin32": runtime.get("pywin32"),
        "pyinstaller": dev.get("pyinstaller"),
    }
    missing = [name for name, version in required.items() if not version]
    if missing:
        raise SourceDistributionError(
            "Required pins are missing from the lock files: " + ", ".join(missing)
        )

    entries: list[dict[str, Any]] = []
    for lock_name, display_name in PYPI_SDIST_PACKAGES:
        version = required.get(lock_name)
        if not isinstance(version, str):
            raise SourceDistributionError(f"Required pin is missing for {lock_name}")
        entries.append(
            _source_entry(
                identifier=f"python-{lock_name}-{version}",
                kind="python-sdist",
                name=display_name,
                version=version,
                source=_pypi_sdist(display_name, version),
                source_of_truth=f"https://pypi.org/project/{display_name}/{version}/#files",
                build_role="locked Python runtime or packaging tool",
            )
        )

    pywin32_version = required.get("pywin32")
    if not isinstance(pywin32_version, str):
        raise SourceDistributionError("Required pin is missing for pywin32")
    if pywin32_version != PYWIN32_SOURCE["version"]:
        raise SourceDistributionError(
            f"pywin32 {pywin32_version} has no verified source mapping; update PYWIN32_SOURCE"
        )
    entries.append(
        _source_entry(
            identifier=f"python-pywin32-{pywin32_version}",
            kind="python-source",
            name=PYWIN32_SOURCE["name"],
            version=pywin32_version,
            source=PYWIN32_SOURCE,
            source_of_truth=PYWIN32_SOURCE["source_of_truth"],
            build_role="locked Windows runtime dependency; PyPI publishes wheels only",
        )
    )

    qt_version = required["pyqt6-qt6"]
    if not isinstance(qt_version, str):
        raise SourceDistributionError("Required pin is missing for pyqt6-qt6")
    for module in QT_MODULE_SOURCES:
        module_name = str(module["module"])
        source = {
            "filename": f"{module_name}-everywhere-src-{qt_version}.tar.xz",
            "url": (
                "https://download.qt.io/official_releases/qt/"
                f"{'.'.join(qt_version.split('.')[:2])}/{qt_version}/submodules/"
                f"{module_name}-everywhere-src-{qt_version}.tar.xz"
            ),
            "sha256": str(module["sha256"]),
        }
        entries.append(
            _source_entry(
                identifier=f"qt-{module_name}-{qt_version}",
                kind="qt-source",
                name=str(module["description"]),
                version=qt_version,
                source=source,
                source_of_truth="https://download.qt.io/official_releases/qt/",
                build_role="corresponding source for the Qt libraries and plugins used by the Windows bundle",
            )
        )

    for name, version in PYQT_BUILD_SOURCES:
        source = _pypi_sdist(name, version)
        entries.append(
            _source_entry(
                identifier=f"python-build-{name.casefold()}-{version}",
                kind="python-build-source",
                name=name,
                version=version,
                source=source,
                source_of_truth=f"https://pypi.org/project/{name}/{version}/#files",
                build_role="PyQt6 binding build requirement declared by the PyQt6 sdist",
            )
        )

    for name, version, digest in CPYTHON_EXTERNAL_SOURCES:
        source = {
            "filename": f"cpython-source-deps-{name}-{version}.tar.gz",
            "url": f"https://github.com/python/cpython-source-deps/archive/refs/tags/{name}-{version}.tar.gz",
            "sha256": digest,
        }
        entries.append(
            _source_entry(
                identifier=f"cpython-external-{name}-{version}",
                kind="cpython-build-source",
                name=f"CPython external dependency: {name}",
                version=version,
                source=source,
                source_of_truth=f"https://github.com/python/cpython-source-deps/tree/{name}-{version}",
                build_role="source dependency used by CPython 3.12.10 Windows build",
            )
        )
    entries.append(
        _source_entry(
            identifier="openssl-upstream-3.0.16",
            kind="cpython-build-source",
            name=OPENSSL_UPSTREAM_SOURCE["name"],
            version=OPENSSL_UPSTREAM_SOURCE["version"],
            source=OPENSSL_UPSTREAM_SOURCE,
            source_of_truth=OPENSSL_UPSTREAM_SOURCE["source_of_truth"],
            build_role="upstream OpenSSL source corresponding to Python's libcrypto/libssl",
        )
    )
    for source in OPENBLAS_SOURCES:
        entries.append(
            {
                **source,
                "kind": "native-build-source",
                "license": "BSD-3-Clause",
            }
        )
    entries.append(
        {
            **MESA_SOURCE,
            "id": "mesa-llvmpipe-11.2.2",
            "kind": "native-build-source",
            "license": "MIT",
        }
    )

    entries.extend(
        [
            {
                **DEEPFILTER_SOURCE,
                "id": f"deepfilter-upstream-{DEEPFILTER_SOURCE['version'][:12]}",
                "kind": "native-build-source",
                "license": "MIT",
                "build_role": "pinned DeepFilterNet source for the C API DLL recipe",
            },
            {
                **DEEPFILTER_TRACT_SOURCE,
                "id": "deepfilter-tract-linalg-0.21.17",
                "kind": "cargo-source-patch",
                "license": "MIT OR Apache-2.0",
                "build_role": "immutable input for the checked-in tract-linalg patch",
            },
        ]
    )

    entries.extend(_crate_entries())
    entries.extend(_deepfilter_crate_entries())
    runtime_entries, blockers = _native_asset_entries()
    entries.extend(runtime_entries)
    recipe_files, recipe_blockers = _recipe_files()
    blockers.extend(recipe_blockers)

    python_version = ".".join(str(part) for part in sys.version_info[:3])
    python_hash = PYTHON_SOURCE["sha256_by_version"].get(python_version)
    if python_version != EXPECTED_CPYTHON_VERSION:
        blockers.append(
            f"CPython {python_version}: release source manifest requires {EXPECTED_CPYTHON_VERSION}"
        )
    elif python_hash is None:
        blockers.append(f"CPython {python_version}: no verified source archive hash is recorded")
    else:
        entries.append(
            {
                "id": f"cpython-{python_version}",
                "kind": "cpython-source",
                "name": "CPython",
                "version": python_version,
                "filename": PYTHON_SOURCE["filename_template"].format(version=python_version),
                "url": PYTHON_SOURCE["url_template"].format(version=python_version),
                "sha256": python_hash,
                "source_of_truth": f"https://www.python.org/downloads/release/python-{python_version.replace('.', '')}/",
                "build_role": "locked interpreter used by the release workflow",
            }
        )

    entries.sort(key=lambda entry: str(entry["id"]).casefold())
    try:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project_version = project["project"]["version"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SourceDistributionError(f"Could not read project version: {exc}") from exc

    return {
        "schema_version": 1,
        "project": "AudioForge",
        "project_version": project_version,
        "distribution_license": "GPL-3.0-only",
        "original_source_license": "MIT",
        "status": "complete" if not blockers else "incomplete",
        "blockers": sorted(set(blockers)),
        "scope": (
            "Exact sources for the release Python runtime, PyQt/Qt binding stack, "
            "PyInstaller bootloader, CPython, and resolved Windows Cargo graph. "
            "Runtime models and redistributable binaries are recorded separately."
        ),
        "recipes": [
            {
                "id": "deepfilter-build",
                "name": "DeepFilter C API Windows build recipe",
                "files": recipe_files,
            }
        ],
        "entries": entries,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_project_version() -> str:
    try:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        version = project["project"]["version"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SourceDistributionError(f"Could not read project version: {exc}") from exc
    if not isinstance(version, str) or not version:
        raise SourceDistributionError("pyproject.toml has no project version")
    return version


def _cargo_lock_hashes(path: Path) -> dict[tuple[str, str], set[str]]:
    try:
        packages = tomllib.loads(path.read_text(encoding="utf-8"))["package"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SourceDistributionError(f"Could not read Cargo lockfile {path}: {exc}") from exc
    hashes: dict[tuple[str, str], set[str]] = {}
    for package in packages:
        if not isinstance(package, dict) or not str(package.get("source", "")).startswith("registry+"):
            continue
        name = package.get("name")
        version = package.get("version")
        checksum = package.get("checksum")
        if (
            isinstance(name, str)
            and name
            and isinstance(version, str)
            and version
            and isinstance(checksum, str)
            and checksum
        ):
            hashes.setdefault((name.casefold(), version), set()).add(checksum)
    return hashes


def _recipe_files() -> tuple[list[dict[str, str]], list[str]]:
    """Hash the checked-in build recipe that is part of the project source."""
    files: list[dict[str, str]] = []
    blockers: list[str] = []
    for relative in DEEPFILTER_RECIPE_FILES:
        path = ROOT / relative
        if not path.is_file():
            blockers.append(f"DeepFilter recipe file is missing: {relative}")
            continue
        files.append({"path": relative, "sha256": _recipe_sha256(path)})
    return files, blockers


def _manifest_entry_ids(manifest: dict[str, Any], *, kind: str | None = None) -> set[str]:
    return {
        str(entry["id"])
        for entry in manifest.get("entries", [])
        if isinstance(entry, dict)
        and isinstance(entry.get("id"), str)
        and (kind is None or entry.get("kind") == kind)
    }


def _expected_runtime_entries() -> dict[str, dict[str, Any]]:
    """Read the current runtime asset identities without resolving networks."""
    asset_file = ROOT / "release-assets.json"
    try:
        assets = json.loads(asset_file.read_text(encoding="utf-8"))["assets"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SourceDistributionError(f"Could not read release-assets.json: {exc}") from exc
    expected: dict[str, dict[str, Any]] = {}
    for asset in assets:
        if not isinstance(asset, dict):
            raise SourceDistributionError("release-assets.json contains a non-object asset")
        path = asset.get("path")
        if not isinstance(path, str) or not path:
            raise SourceDistributionError("release-assets.json contains an asset without a path")
        identifier = f"runtime-{Path(path).name or 'asset'}"
        if identifier in expected:
            raise SourceDistributionError(f"release-assets.json repeats asset identity {identifier}")
        expected[identifier] = asset
    return expected


def _expected_static_source_entries(runtime: dict[str, str]) -> dict[str, dict[str, str]]:
    """Return immutable source identities that do not require network metadata."""
    expected: dict[str, dict[str, str]] = {}

    pywin32_version = runtime.get("pywin32")
    if pywin32_version == PYWIN32_SOURCE["version"]:
        expected[f"python-pywin32-{pywin32_version}"] = {
            key: str(PYWIN32_SOURCE[key])
            for key in ("filename", "url", "sha256", "source_of_truth")
        }

    qt_version = runtime.get("pyqt6-qt6")
    if qt_version:
        minor = ".".join(qt_version.split(".")[:2])
        for module in QT_MODULE_SOURCES:
            module_name = str(module["module"])
            expected[f"qt-{module_name}-{qt_version}"] = {
                "filename": f"{module_name}-everywhere-src-{qt_version}.tar.xz",
                "url": (
                    "https://download.qt.io/official_releases/qt/"
                    f"{minor}/{qt_version}/submodules/"
                    f"{module_name}-everywhere-src-{qt_version}.tar.xz"
                ),
                "sha256": str(module["sha256"]),
                "source_of_truth": "https://download.qt.io/official_releases/qt/",
            }

    for name, version, digest in CPYTHON_EXTERNAL_SOURCES:
        expected[f"cpython-external-{name}-{version}"] = {
            "filename": f"cpython-source-deps-{name}-{version}.tar.gz",
            "url": f"https://github.com/python/cpython-source-deps/archive/refs/tags/{name}-{version}.tar.gz",
            "sha256": digest,
            "source_of_truth": f"https://github.com/python/cpython-source-deps/tree/{name}-{version}",
        }

    static_sources = [
        (
            "openssl-upstream-3.0.16",
            OPENSSL_UPSTREAM_SOURCE,
        ),
        ("mesa-llvmpipe-11.2.2", MESA_SOURCE),
        (f"deepfilter-upstream-{DEEPFILTER_SOURCE['version'][:12]}", DEEPFILTER_SOURCE),
        ("deepfilter-tract-linalg-0.21.17", DEEPFILTER_TRACT_SOURCE),
        *[(str(source["id"]), source) for source in OPENBLAS_SOURCES],
    ]
    for identifier, source in static_sources:
        expected[identifier] = {
            key: str(source[key])
            for key in ("filename", "url", "sha256", "source_of_truth")
        }
    return expected


def _validate_manifest(manifest: dict[str, Any], *, release: bool = False) -> None:
    """Reject malformed or untrusted manifests before touching the network."""
    if not isinstance(manifest, dict):
        raise SourceDistributionError("Source manifest is not an object")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise SourceDistributionError("Source manifest must contain entries")
    blockers = manifest.get("blockers")
    if not isinstance(blockers, list) or not all(isinstance(item, str) and item for item in blockers):
        raise SourceDistributionError("Source manifest blockers must be nonempty strings")
    status = manifest.get("status")
    if status is not None and status not in {"complete", "incomplete"}:
        raise SourceDistributionError("Source manifest has an invalid status")
    if release and status not in {"complete", "incomplete"}:
        raise SourceDistributionError("Release source manifest must declare complete or incomplete status")
    if status == "complete" and blockers:
        raise SourceDistributionError("Complete source manifest cannot have blockers")
    if status == "incomplete" and not blockers:
        raise SourceDistributionError("Incomplete source manifest must have blockers")

    identifiers: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise SourceDistributionError("Source manifest contains a non-object entry")
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise SourceDistributionError(f"Source manifest has duplicate or invalid id: {identifier!r}")
        identifiers.add(identifier)
        for key in ("kind", "name", "version", "filename", "url", "sha256"):
            if not isinstance(entry.get(key), str) or not entry[key]:
                raise SourceDistributionError(f"Manifest entry {identifier!r} lacks {key}")
        _archive_target(Path("."), entry)
        if not re.fullmatch(r"[0-9a-fA-F]{64}", entry["sha256"]):
            raise SourceDistributionError(f"Manifest entry {identifier!r} has an invalid SHA-256")
        asset_sha256 = entry.get("asset_sha256")
        if asset_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", str(asset_sha256)):
            raise SourceDistributionError(f"Manifest entry {identifier!r} has an invalid asset SHA-256")
        entry_status = entry.get("status", "available")
        if entry_status not in {"available", "blocked"}:
            raise SourceDistributionError(f"Manifest entry {identifier!r} has an invalid status")
        license_paths = entry.get("license_paths")
        if license_paths is not None:
            if (
                not isinstance(license_paths, list)
                or not license_paths
                or any(
                    not isinstance(path, str)
                    or not path
                    or Path(path).is_absolute()
                    or ".." in Path(path).parts
                    for path in license_paths
                )
            ):
                raise SourceDistributionError(
                    f"Manifest entry {identifier!r} has unsafe license paths"
                )
        source_build = entry.get("source_build")
        if source_build is not None:
            if (
                entry.get("kind") != "runtime-asset"
                or not isinstance(source_build, bool)
            ):
                raise SourceDistributionError(
                    f"Manifest entry {identifier!r} has an invalid source-build marker"
                )
        if release and entry.get("kind") == "cargo-crate":
            lock_file = entry.get("lock_file")
            lock_paths = {
                "Cargo.lock": ROOT / "Cargo.lock",
                "build-support/deepfilter/Cargo.lock": ROOT
                / "build-support"
                / "deepfilter"
                / "Cargo.lock",
            }
            if lock_file not in lock_paths:
                raise SourceDistributionError(
                    f"Manifest entry {identifier!r} has no supported Cargo lock_file"
                )
            lock_hashes = _cargo_lock_hashes(lock_paths[lock_file])
            key = (entry["name"].casefold(), entry["version"])
            if entry["sha256"] not in lock_hashes.get(key, set()):
                raise SourceDistributionError(
                    f"Manifest entry {identifier!r} does not match {lock_file}"
                )
        _validate_source_url(entry["url"])
        source_of_truth = entry.get("source_of_truth")
        if source_of_truth is not None:
            if not isinstance(source_of_truth, str):
                raise SourceDistributionError(f"Manifest entry {identifier!r} has invalid source_of_truth")
            _validate_source_url(source_of_truth)

    if not release:
        return
    blocked_entries = [
        entry for entry in entries if entry.get("status", "available") == "blocked"
    ]
    if status == "complete" and blocked_entries:
        raise SourceDistributionError("Complete source manifest cannot contain blocked entries")
    if manifest.get("schema_version") != 1 or manifest.get("project") != "AudioForge":
        raise SourceDistributionError("Source manifest has an unsupported project or schema")
    project_version = manifest.get("project_version")
    if project_version != _read_project_version():
        raise SourceDistributionError(
            f"Source manifest project version {project_version!r} does not match pyproject.toml"
        )

    recipes = manifest.get("recipes")
    if not isinstance(recipes, list) or not recipes:
        raise SourceDistributionError("Release source manifest must record build recipe files")
    recipe_files: dict[str, str] = {}
    for recipe in recipes:
        if not isinstance(recipe, dict) or not isinstance(recipe.get("files"), list):
            raise SourceDistributionError("Source manifest has an invalid recipe record")
        for file_record in recipe["files"]:
            if not isinstance(file_record, dict):
                raise SourceDistributionError("Source manifest has an invalid recipe file record")
            relative = file_record.get("path")
            digest = file_record.get("sha256")
            if (
                not isinstance(relative, str)
                or not relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)
            ):
                raise SourceDistributionError("Source manifest has an unsafe recipe file record")
            if relative in recipe_files:
                raise SourceDistributionError(f"Source manifest repeats recipe file {relative!r}")
            path = (ROOT / relative).resolve()
            try:
                path.relative_to(ROOT.resolve())
            except ValueError as exc:
                raise SourceDistributionError(f"Recipe file escapes the project: {relative!r}") from exc
            if not path.is_file():
                raise SourceDistributionError(f"Release recipe file is missing: {relative}")
            actual = _recipe_sha256(path)
            if actual.casefold() != digest.casefold():
                raise SourceDistributionError(f"Release recipe digest mismatch: {relative}")
            recipe_files[relative] = digest
    missing_recipe_files = set(DEEPFILTER_RECIPE_FILES) - set(recipe_files)
    if missing_recipe_files:
        if status == "complete" or not any(
            "DeepFilter recipe file is missing" in blocker for blocker in blockers
        ):
            raise SourceDistributionError(
                "Release source manifest omits recipe files: "
                + ", ".join(sorted(missing_recipe_files))
            )

    runtime = _read_locked_versions(ROOT / "requirements" / "runtime.txt")
    dev = _read_locked_versions(ROOT / "requirements" / "dev.txt")
    locked_hashes: dict[str, set[str]] = {}
    for lock_path in (ROOT / "requirements" / "runtime.txt", ROOT / "requirements" / "dev.txt"):
        for name, hashes in _read_locked_hashes(lock_path).items():
            locked_hashes.setdefault(name, set()).update(hashes)
    expected = {
        "PyQt6": runtime.get("pyqt6"),
        "PyQt6-sip": runtime.get("pyqt6-sip"),
        "NumPy": runtime.get("numpy"),
        "SciPy": runtime.get("scipy"),
        "pywin32": runtime.get("pywin32"),
        "PyInstaller": dev.get("pyinstaller"),
    }
    for name, version in expected.items():
        matches = [
            entry.get("name", "").casefold() == name.casefold()
            and entry.get("version") == version
            for entry in entries
        ]
        if not isinstance(version, str) or not any(matches):
            raise SourceDistributionError(f"Manifest lacks locked source for {name} {version}")
        if name.casefold() != "pywin32" and not any(
            entry.get("sha256") in locked_hashes.get(name.casefold(), set())
            for entry in entries
            if entry.get("name", "").casefold() == name.casefold()
            and entry.get("version") == version
        ):
            raise SourceDistributionError(f"Manifest source hash for {name} {version} is not pinned in the lock")
    qt_version = runtime.get("pyqt6-qt6")
    if not isinstance(qt_version, str) or not any(
        entry.get("kind") == "qt-source" and entry.get("version") == qt_version
        for entry in entries
    ):
        raise SourceDistributionError(f"Manifest lacks Qt source for {qt_version}")
    cpython_entries = [entry for entry in entries if entry.get("kind") == "cpython-source"]
    if len(cpython_entries) != 1 or cpython_entries[0].get("version") != EXPECTED_CPYTHON_VERSION:
        raise SourceDistributionError(
            f"Manifest must contain exactly CPython {EXPECTED_CPYTHON_VERSION} source"
        )
    expected_cpython_hash = PYTHON_SOURCE["sha256_by_version"].get(EXPECTED_CPYTHON_VERSION)
    if cpython_entries[0].get("sha256") != expected_cpython_hash:
        raise SourceDistributionError("Manifest CPython source hash is not the verified release archive")
    actual_python = ".".join(str(part) for part in sys.version_info[:3])
    if actual_python != EXPECTED_CPYTHON_VERSION:
        raise SourceDistributionError(
            f"Release source verification requires CPython {EXPECTED_CPYTHON_VERSION}, got {actual_python}"
        )
    pywin32_version = runtime.get("pywin32")
    if pywin32_version != PYWIN32_SOURCE["version"]:
        raise SourceDistributionError(
            f"pywin32 lock is {pywin32_version}; update the verified source mapping before release"
        )
    static_sources = _expected_static_source_entries(runtime)
    actual_by_id = {
        str(entry["id"]): entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    for identifier, expected_source in static_sources.items():
        actual = actual_by_id.get(identifier)
        if actual is None:
            raise SourceDistributionError(f"Manifest lacks current source identity for {identifier}")
        for key, expected_value in expected_source.items():
            if actual.get(key) != expected_value:
                raise SourceDistributionError(f"Manifest source identity is stale: {identifier} ({key})")

    expected_runtime = _expected_runtime_entries()
    actual_runtime = {
        str(entry["id"]): entry
        for entry in entries
        if entry.get("kind") == "runtime-asset"
    }
    if set(actual_runtime) != set(expected_runtime):
        raise SourceDistributionError("Manifest runtime assets are stale or incomplete")
    for identifier, expected_asset in expected_runtime.items():
        actual = actual_runtime[identifier]
        origin: dict[str, Any] = (
            expected_asset["origin"]
            if isinstance(expected_asset.get("origin"), dict)
            else {}
        )
        expected_sha = origin.get("package_sha256") or expected_asset.get("sha256")
        expected_status = (
            "blocked"
            if origin.get("status") in {
                "inherited-binary-build-identity-unresolved",
                "license-restricted",
                "proprietary",
                "distribution-blocked",
            }
            or expected_asset.get("distribution_status")
            in {"license-restricted", "proprietary", "distribution-blocked"}
            or "microsoft software license" in str(expected_asset.get("license") or "").casefold()
            or "proprietary" in str(expected_asset.get("license") or "").casefold()
            else "available"
        )
        if actual.get("url") != expected_asset.get("source") or actual.get("sha256") != expected_sha:
            raise SourceDistributionError(f"Manifest runtime asset is stale: {identifier}")
        if actual.get("asset_sha256") != expected_asset.get("sha256"):
            raise SourceDistributionError(f"Manifest runtime asset digest is stale: {identifier}")
        if actual.get("status", "available") != expected_status:
            raise SourceDistributionError(f"Manifest runtime asset status is stale: {identifier}")
        expected_source_build = (
            isinstance(origin, dict) and origin.get("status") == "verified-source-build"
        )
        if actual.get("source_build", False) != expected_source_build:
            raise SourceDistributionError(f"Manifest runtime asset source-build marker is stale: {identifier}")

    expected_cargo_entries = _crate_entries() + _deepfilter_crate_entries()
    expected_cargo = {entry["id"] for entry in expected_cargo_entries}
    actual_cargo = {
        entry["id"] for entry in entries if entry.get("kind") == "cargo-crate"
    }
    if actual_cargo != expected_cargo:
        raise SourceDistributionError("Manifest Cargo graph is stale or incomplete")
    for expected_entry in expected_cargo_entries:
        actual = actual_by_id.get(str(expected_entry["id"]))
        if actual is None:
            raise SourceDistributionError(
                f"Manifest lacks current Cargo source identity for {expected_entry['id']}"
            )
        for key in ("filename", "url", "sha256", "lock_file"):
            if actual.get(key) != expected_entry.get(key):
                raise SourceDistributionError(
                    f"Manifest Cargo source identity is stale: {expected_entry['id']} ({key})"
                )

    required_ids = {
        *(f"qt-{module['module']}-{qt_version}" for module in QT_MODULE_SOURCES),
        *(f"cpython-external-{name}-{version}" for name, version, _ in CPYTHON_EXTERNAL_SOURCES),
        "openssl-upstream-3.0.16",
        "mesa-llvmpipe-11.2.2",
        f"deepfilter-upstream-{DEEPFILTER_SOURCE['version'][:12]}",
        "deepfilter-tract-linalg-0.21.17",
        *(source["id"] for source in OPENBLAS_SOURCES),
    }
    actual_ids = _manifest_entry_ids(manifest)
    missing_ids = required_ids - actual_ids
    if missing_ids:
        raise SourceDistributionError(
            "Manifest omits corresponding native sources: " + ", ".join(sorted(missing_ids))
        )


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceDistributionError(f"Could not read source manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceDistributionError(f"Source manifest {path} has an invalid shape")
    _validate_manifest(value, release=True)
    return value


def _archive_target(root: Path, entry: dict[str, Any]) -> Path:
    filename = entry.get("filename")
    identifier = entry.get("id")
    if (
        not isinstance(filename, str)
        or not filename
        or not isinstance(identifier, str)
        or not identifier
    ):
        raise SourceDistributionError("Manifest entry has no safe id or filename")
    if Path(filename).name != filename or Path(identifier).name != identifier:
        raise SourceDistributionError(f"Unsafe source archive path in entry {identifier!r}")
    return root / "archives" / f"{identifier}--{filename}"


def _is_derived_runtime_entry(entry: dict[str, Any]) -> bool:
    """Return whether an asset is verified by its source build attestation."""
    return entry.get("kind") == "runtime-asset" and entry.get("source_build") is True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recipe_sha256(path: Path) -> str:
    """Hash recipe text with checkout line endings normalized to LF."""
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _manifest_digest(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_archive_bytes(revision: str, version: str) -> bytes:
    resolved = _git_text("rev-parse", "--verify", f"{revision}^{{commit}}")
    short_revision = _git_text("rev-parse", "--short=12", resolved)
    prefix = f"AudioForge-{version}-{short_revision}/"
    try:
        completed = subprocess.run(
            ["git", "archive", "--format=tar", f"--prefix={prefix}", resolved],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceDistributionError(f"Could not recreate project source archive: {exc}") from exc
    return completed.stdout


def _validate_project_archive(path: Path, manifest: dict[str, Any], revision: str) -> str:
    """Prove the project tar is the git archive for the recorded revision."""
    if not path.is_file():
        raise SourceDistributionError(f"Project source archive is missing: {path.name}")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise SourceDistributionError("Source receipt has an invalid resolved revision")
    actual_bytes = path.read_bytes()
    expected_bytes = _git_archive_bytes(revision, str(manifest["project_version"]))
    actual_digest = hashlib.sha256(actual_bytes).hexdigest()
    if actual_bytes != expected_bytes:
        raise SourceDistributionError("Project source archive does not match the recorded git revision")
    try:
        with tarfile.open(fileobj=io.BytesIO(actual_bytes), mode="r:") as archive:
            members = archive.getmembers()
            if not members:
                raise SourceDistributionError("Project source archive is empty")
            prefix = f"AudioForge-{manifest['project_version']}-{revision[:12]}/"
            root_name = prefix.rstrip("/")
            names = [member.name for member in members]
            if any(
                (name == root_name and not member.isdir())
                or (name != root_name and not name.startswith(prefix))
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                for member, name in zip(members, names, strict=True)
            ):
                raise SourceDistributionError("Project source archive contains an unsafe member")
            pyproject_name = prefix + "pyproject.toml"
            pyproject = archive.extractfile(pyproject_name)
            if pyproject is None:
                raise SourceDistributionError("Project source archive has no pyproject.toml")
            project = tomllib.loads(pyproject.read().decode("utf-8"))
            if project.get("project", {}).get("version") != manifest["project_version"]:
                raise SourceDistributionError("Project source archive version does not match the manifest")
    except (KeyError, OSError, tarfile.TarError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise SourceDistributionError(f"Could not inspect project source archive: {exc}") from exc
    return actual_digest


def _download(entry: dict[str, Any], destination: Path) -> Path:
    url = entry.get("url")
    expected = entry.get("sha256")
    if not isinstance(url, str) or not url or not isinstance(expected, str) or not expected:
        raise SourceDistributionError(f"Entry {entry.get('id')} is not a downloadable archive")
    _validate_source_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = _sha256(destination)
        if actual == expected:
            return destination
        raise SourceDistributionError(
            f"Existing source archive hash mismatch for {entry.get('id')}: {actual} != {expected}"
        )

    temporary = destination.with_name(destination.name + f".{os.getpid()}.part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "AudioForge source distribution"})
        with _open_source_url(request, timeout=120) as response:
            final_url = response.geturl() if hasattr(response, "geturl") else url
            if not isinstance(final_url, str):
                raise SourceDistributionError(
                    f"Source URL redirect did not provide a URL for {entry.get('id')}"
                )
            _validate_source_url(final_url)
            with temporary.open("xb") as handle:
                digest = hashlib.sha256()
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        actual = digest.hexdigest()
        if actual != expected:
            raise SourceDistributionError(
                f"Downloaded source hash mismatch for {entry.get('id')}: {actual} != {expected}"
            )
        temporary.replace(destination)
        return destination
    except SourceDistributionError:
        temporary.unlink(missing_ok=True)
        raise
    except (OSError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise SourceDistributionError(
            f"Could not download source for {entry.get('id')}: {exc}"
        ) from exc
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def download_sources(
    manifest: dict[str, Any],
    destination: Path,
    *,
    include_runtime_assets: bool = False,
) -> list[Path]:
    """Download and verify all source entries, preserving blocked assets."""
    _validate_manifest(manifest)
    downloaded: list[Path] = []
    for entry in manifest["entries"]:
        if not isinstance(entry, dict):
            raise SourceDistributionError("Source manifest contains a non-object entry")
        if entry.get("kind") == "runtime-asset" and not include_runtime_assets:
            continue
        if _is_derived_runtime_entry(entry):
            continue
        if entry.get("status") == "blocked":
            continue
        target = _archive_target(destination, entry)
        downloaded.append(_download(entry, target))
    return downloaded


def verify_sources(
    manifest: dict[str, Any],
    destination: Path,
    *,
    allow_incomplete: bool = False,
    include_runtime_assets: bool = False,
    require_receipt: bool = False,
    revision: str | None = None,
) -> None:
    """Verify every downloaded source archive and enforce blocker visibility."""
    _validate_manifest(manifest)
    failures: list[str] = []
    expected_paths: set[Path] = set()
    for entry in manifest["entries"]:
        if not isinstance(entry, dict):
            raise SourceDistributionError("Source manifest contains a non-object entry")
        if entry.get("status") == "blocked":
            continue
        if entry.get("kind") == "runtime-asset" and not include_runtime_assets:
            continue
        if _is_derived_runtime_entry(entry):
            continue
        try:
            path = _archive_target(destination, entry)
            expected_paths.add(path.relative_to(destination))
            if not path.is_file():
                failures.append(f"missing {entry.get('id')}")
            elif _sha256(path) != entry.get("sha256"):
                failures.append(f"hash mismatch {entry.get('id')}")
        except SourceDistributionError as exc:
            failures.append(str(exc))
    archive_root = destination / "archives"
    if archive_root.is_dir():
        actual_paths = {
            path.relative_to(destination)
            for path in archive_root.rglob("*")
            if path.is_file()
        }
        for stale in sorted(actual_paths - expected_paths):
            failures.append(f"unexpected source archive {stale.as_posix()}")
    receipt_path = destination / "source-receipt.json"
    if require_receipt or receipt_path.exists():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"invalid source receipt: {exc}")
        else:
            if not isinstance(receipt, dict):
                failures.append("invalid source receipt: expected an object")
            else:
                if receipt.get("schema_version") != 1 or receipt.get("project") != manifest.get("project"):
                    failures.append("source receipt project or schema mismatch")
                if receipt.get("project_version") != manifest.get("project_version"):
                    failures.append("source receipt project version mismatch")
                if receipt.get("manifest_sha256") != _manifest_digest(manifest):
                    failures.append("source receipt manifest digest mismatch")
                receipt_revision = receipt.get("revision")
                if not isinstance(receipt_revision, str):
                    failures.append("source receipt revision is missing")
                else:
                    if revision is not None:
                        try:
                            resolved_revision = _git_text("rev-parse", "--verify", f"{revision}^{{commit}}")
                        except SourceDistributionError as exc:
                            failures.append(str(exc))
                        else:
                            if receipt_revision != resolved_revision:
                                failures.append("source receipt revision mismatch")
                    project_record = receipt.get("project_source")
                    if not isinstance(project_record, dict):
                        failures.append("source receipt project source record is missing")
                    elif project_record.get("path") != "AudioForge-project-source.tar":
                        failures.append("source receipt project source path is invalid")
                    else:
                        project_path = destination / "AudioForge-project-source.tar"
                        try:
                            project_digest = _validate_project_archive(
                                project_path, manifest, receipt_revision
                            )
                        except SourceDistributionError as exc:
                            failures.append(str(exc))
                        else:
                            if project_record.get("sha256") != project_digest:
                                failures.append("source receipt project source digest mismatch")
                            if project_record.get("bytes") != project_path.stat().st_size:
                                failures.append("source receipt project source size mismatch")
                records = receipt.get("archives")
                if not isinstance(records, list):
                    failures.append("source receipt archives are missing")
                else:
                    expected_records = []
                    for entry in manifest["entries"]:
                        if not isinstance(entry, dict):
                            continue
                        if entry.get("status") == "blocked":
                            continue
                        if entry.get("kind") == "runtime-asset" and not include_runtime_assets:
                            continue
                        if _is_derived_runtime_entry(entry):
                            continue
                        path = _archive_target(destination, entry)
                        expected_records.append(
                            {
                                "id": entry["id"],
                                "path": path.relative_to(destination).as_posix(),
                                "sha256": entry["sha256"],
                                "bytes": path.stat().st_size if path.is_file() else -1,
                            }
                        )
                    actual_records = sorted(
                        [record for record in records if isinstance(record, dict)],
                        key=lambda record: str(record.get("id")),
                    )
                    if actual_records != sorted(expected_records, key=lambda record: str(record["id"])):
                        failures.append("source receipt archives do not match the hydrated manifest")
    blockers = [str(item) for item in manifest.get("blockers", [])]
    if failures:
        raise SourceDistributionError("; ".join(failures))
    if blockers and not allow_incomplete:
        raise SourceDistributionError(
            "Corresponding-source manifest remains incomplete: " + "; ".join(blockers)
        )


def _git_text(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceDistributionError(f"Could not inspect git revision: {exc}") from exc
    return completed.stdout.strip()


def _bundle_project_source(
    destination: Path,
    *,
    revision: str = "HEAD",
    expected_version: str | None = None,
) -> Path:
    """Archive a clean, version-matched project revision."""
    if _git_text("status", "--porcelain", "--untracked-files=all"):
        raise SourceDistributionError(
            "Refusing source bundle from a dirty worktree; commit the release first"
        )
    resolved = _git_text("rev-parse", "--verify", f"{revision}^{{commit}}")
    short_revision = _git_text("rev-parse", "--short=12", resolved)
    try:
        project = tomllib.loads(_git_text("show", f"{resolved}:pyproject.toml"))
        revision_version = project["project"]["version"]
    except (KeyError, TypeError, ValueError) as exc:
        raise SourceDistributionError(
            f"Release revision {resolved} has no readable project version"
        ) from exc
    if not isinstance(revision_version, str) or not revision_version:
        raise SourceDistributionError(f"Release revision {resolved} has no project version")
    expected_version = expected_version or _read_project_version()
    if revision_version != expected_version:
        raise SourceDistributionError(
            f"Revision {resolved} has version {revision_version}, expected {expected_version}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "AudioForge-project-source.tar"
    prefix = f"AudioForge-{revision_version}-{short_revision}/"
    command = ["git", "archive", "--format=tar", f"--prefix={prefix}", resolved]
    try:
        with output.open("wb") as handle:
            subprocess.run(command, cwd=ROOT, check=True, stdout=handle)
    except (OSError, subprocess.CalledProcessError) as exc:
        output.unlink(missing_ok=True)
        raise SourceDistributionError(f"Could not archive tracked project source: {exc}") from exc
    return output


def _write_receipt(
    destination: Path,
    manifest: dict[str, Any],
    *,
    revision: str,
    include_runtime_assets: bool,
) -> Path:
    """Record the exact hydrated inputs used by a source bundle."""
    resolved = _git_text("rev-parse", "--verify", f"{revision}^{{commit}}")
    archives: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        if not isinstance(entry, dict):
            continue
        if entry.get("status") == "blocked":
            continue
        if entry.get("kind") == "runtime-asset" and not include_runtime_assets:
            continue
        if _is_derived_runtime_entry(entry):
            continue
        path = _archive_target(destination, entry)
        archives.append(
            {
                "id": entry["id"],
                "path": path.relative_to(destination).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    receipt = {
        "schema_version": 1,
        "project": manifest["project"],
        "project_version": manifest["project_version"],
        "revision": resolved,
        "manifest_sha256": _manifest_digest(manifest),
        "project_source": {
            "path": "AudioForge-project-source.tar",
            "sha256": _sha256(destination / "AudioForge-project-source.tar"),
            "bytes": (destination / "AudioForge-project-source.tar").stat().st_size,
        },
        "archives": archives,
    }
    output = destination / "source-receipt.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="resolve and write the source manifest")
    manifest.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)

    download = subparsers.add_parser("download", help="download and verify source archives")
    download.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    download.add_argument("--output", type=Path, default=DEFAULT_SOURCE_DIR)
    download.add_argument("--include-runtime-assets", action="store_true")

    verify = subparsers.add_parser("verify", help="verify a hydrated source directory")
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    verify.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    verify.add_argument("--include-runtime-assets", action="store_true")
    verify.add_argument("--allow-incomplete", action="store_true")
    verify.add_argument("--require-receipt", action="store_true")
    verify.add_argument("--revision", help="revision recorded by a source receipt")

    bundle = subparsers.add_parser("bundle", help="hydrate sources and archive the project")
    bundle.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    bundle.add_argument("--output", type=Path, default=DEFAULT_SOURCE_DIR)
    bundle.add_argument(
        "--revision",
        default="HEAD",
        help="clean git tag or commit to archive (default: HEAD)",
    )
    bundle.add_argument("--include-runtime-assets", action="store_true")
    bundle.add_argument("--allow-incomplete", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "manifest":
            write_manifest(args.output.resolve(), build_manifest())
            print(f"Wrote source manifest: {args.output.resolve()}")
            return 0

        manifest = load_manifest(args.manifest.resolve())
        if args.command == "download":
            downloaded = download_sources(
                manifest,
                args.output.resolve(),
                include_runtime_assets=args.include_runtime_assets,
            )
            print(f"Downloaded and verified {len(downloaded)} source archives")
            if manifest.get("blockers"):
                print("Manifest remains incomplete:")
                for blocker in manifest["blockers"]:
                    print(f"- {blocker}")
            return 0

        if args.command == "verify":
            verify_sources(
                manifest,
                args.source_dir.resolve(),
                allow_incomplete=args.allow_incomplete,
                include_runtime_assets=args.include_runtime_assets,
                require_receipt=args.require_receipt,
                revision=args.revision,
            )
            print("Verified source archives")
            return 0

        if args.command == "bundle":
            download_sources(
                manifest,
                args.output.resolve(),
                include_runtime_assets=args.include_runtime_assets,
            )
            verify_sources(
                manifest,
                args.output.resolve(),
                allow_incomplete=args.allow_incomplete,
                include_runtime_assets=args.include_runtime_assets,
            )
            _bundle_project_source(
                args.output.resolve(),
                revision=args.revision,
                expected_version=manifest["project_version"],
            )
            receipt = _write_receipt(
                args.output.resolve(),
                manifest,
                revision=args.revision,
                include_runtime_assets=args.include_runtime_assets,
            )
            verify_sources(
                manifest,
                args.output.resolve(),
                allow_incomplete=args.allow_incomplete,
                include_runtime_assets=args.include_runtime_assets,
                require_receipt=True,
                revision=args.revision,
            )
            print(f"Wrote source bundle: {args.output.resolve()}")
            print(f"Wrote source receipt: {receipt}")
            return 0
    except SourceDistributionError as exc:
        print(f"source-distribution: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
