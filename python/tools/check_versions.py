"""Verify release version strings stay in sync."""

from __future__ import annotations

import ast
import json
import re
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _single_match(path: str, pattern: str, label: str) -> str:
    match = re.search(pattern, _read(path), re.MULTILINE)
    if not match:
        raise ValueError(f"{label}: version string not found in {path}")
    return match.group(1)


def _require_pattern(path: str, pattern: str, label: str) -> None:
    if not re.search(pattern, _read(path), re.MULTILINE):
        raise ValueError(f"{label}: version reference not found in {path}")


def _parse_python(path: str) -> ast.AST:
    return ast.parse(_read(path), filename=path)


def _extract_catalog_version_reference(path: str) -> str:
    tree = _parse_python(path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "preset_cls":
            continue
        for keyword in node.keywords:
            if keyword.arg != "version":
                continue
            value = keyword.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
            if isinstance(value, ast.Name) and value.id == "CURRENT_VERSION":
                return "__CURRENT_VERSION__"
    raise ValueError(f"built-in preset default: version reference not found in {path}")


def _extract_main_window_preset_version(path: str) -> str:
    tree = _parse_python(path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Attribute) or target.attr != "version":
                continue
            if not isinstance(target.value, ast.Name) or target.value.id != "preset":
                continue
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
            if isinstance(value, ast.Name) and value.id == "__version__":
                return "__PACKAGE_VERSION__"
    raise ValueError(f"auto-eq preset default: version assignment not found in {path}")


def _extract_cargo_lock_version(package_name: str) -> str:
    with (REPO_ROOT / "Cargo.lock").open("rb") as handle:
        lock = tomllib.load(handle)
    for package in lock.get("package", []):
        if package.get("name") == package_name:
            return str(package["version"])
    raise ValueError(f"Cargo.lock: package {package_name!r} not found")


def _check_release_asset_hydration() -> None:
    manifest = json.loads(_read("release-assets.json"))
    fallback_tag = manifest.get("fallback_release_tag")
    if not isinstance(fallback_tag, str) or re.fullmatch(
        r"v[0-9]+\.[0-9]+\.[0-9]+", fallback_tag
    ) is None:
        raise ValueError(
            "release-assets.json: fallback_release_tag must be a vMAJOR.MINOR.PATCH tag"
        )

    docs = {name: _read(name) for name in ("README.md", "RELEASING.md")}
    for name, contents in docs.items():
        explicit_tags = re.findall(
            r"fetch_release_assets\.py\s+--release-tag\s+(v[0-9]+\.[0-9]+\.[0-9]+)",
            contents,
        )
        mismatched = sorted({tag for tag in explicit_tags if tag != fallback_tag})
        if mismatched:
            raise ValueError(
                f"{name}: fetch_release_assets.py uses stale fallback tag(s): "
                + ", ".join(mismatched)
            )
    readme = docs["README.md"]
    if "fallback release is pinned once in `release-assets.json`" not in readme:
        raise ValueError(
            "README.md: release-asset hydration must document the manifest-owned fallback"
        )

    fetch_tool = _read("python/tools/fetch_release_assets.py")
    if "default=_default_asset_source_tag()" not in fetch_tool:
        raise ValueError(
            "fetch_release_assets.py: --release-tag default must come from release-assets.json"
        )

    workflow = _read(".github/workflows/release-package.yml")
    if ").fallback_release_tag" not in workflow:
        raise ValueError(
            "release-package.yml: asset fallback must come from release-assets.json"
        )


def _check_no_static_current_archive_claims(version: str) -> None:
    paths = (
        "README.md",
        f"release-notes/release-notes-v{version}.md",
    )
    archive_term = r"(?:archive|portable\s+folder|bundle)"
    exact_size = re.compile(
        rf"(?is){archive_term}.{{0,160}}\b\d[\d,]*\s+bytes\b"
    )
    exact_hash = re.compile(
        rf"(?is)(?:{archive_term}.{{0,200}}\b[0-9a-f]{{64}}\b|"
        rf"\b[0-9a-f]{{64}}\b.{{0,200}}{archive_term})"
    )
    for path in paths:
        contents = _read(path)
        if exact_size.search(contents) or exact_hash.search(contents):
            raise ValueError(
                f"{path}: pre-tag prose must not contain exact release archive "
                "size/hash claims; use generated provenance sidecars"
            )


def main() -> int:
    _check_release_asset_hydration()
    expected = _single_match("pyproject.toml", r'^version\s*=\s*"([^"]+)"', "pyproject")
    _check_no_static_current_archive_claims(expected)
    package_version = _single_match(
        "python/mic_eq/__init__.py",
        r'^__version__\s*=\s*"([^"]+)"',
        "python package",
    )
    current_version = _single_match(
        "python/mic_eq/config_parts/shared.py",
        r'^CURRENT_VERSION\s*=\s*"([^"]+)"',
        "shared config version",
    )
    catalog_version = _extract_catalog_version_reference("python/mic_eq/config_parts/catalogs.py")
    main_window_version = _extract_main_window_preset_version("python/mic_eq/ui/main_window.py")

    checks = {
        "rust-core/Cargo.toml": _single_match(
            "rust-core/Cargo.toml",
            r'^version\s*=\s*"([^"]+)"',
            "rust core",
        ),
        "Cargo.lock mic_eq_core": _extract_cargo_lock_version("mic_eq_core"),
        "python/mic_eq/__init__.py": package_version,
        "python/mic_eq/config_parts/shared.py CURRENT_VERSION": current_version,
        "README.md": _single_match(
            "README.md",
            r"Current version:\s*`v([^`]+)`",
            "readme",
        ),
        "python/mic_eq/config_parts/presets.py Preset.version": current_version,
        "python/mic_eq/config_parts/catalogs.py built-ins": (
            current_version if catalog_version == "__CURRENT_VERSION__" else catalog_version
        ),
        "python/mic_eq/ui/main_window.py auto-eq preset": (
            package_version if main_window_version == "__PACKAGE_VERSION__" else main_window_version
        ),
    }

    release_versions = set(
        re.findall(r"AudioForge-v([0-9]+\.[0-9]+\.[0-9]+)", _read("RELEASING.md"))
    )
    _require_pattern(
        "python/mic_eq/config_parts/presets.py",
        r"version:\s*str\s*=\s*CURRENT_VERSION",
        "preset default",
    )
    if not release_versions:
        raise ValueError("RELEASING.md: release archive version string not found")
    if release_versions != {expected}:
        checks["RELEASING.md"] = ", ".join(sorted(release_versions))

    release_notes_path = REPO_ROOT / "release-notes" / f"release-notes-v{expected}.md"
    if not release_notes_path.is_file():
        raise ValueError(
            f"release notes: expected file is missing: {release_notes_path.relative_to(REPO_ROOT)}"
        )

    mismatches = {path: version for path, version in checks.items() if version != expected}
    if mismatches:
        print(f"Version mismatch: pyproject.toml is {expected}")
        for path, version in mismatches.items():
            print(f"  {path}: {version}")
        return 1

    print(f"Version strings are in sync: {expected}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Version check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
