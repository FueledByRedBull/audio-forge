"""Verify release version strings stay in sync."""

from __future__ import annotations

import ast
import json
import re
import sys
import tomllib
from pathlib import Path

from release_version import ReleaseVersion, parse_tag, parse_version

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PYTHON_MINOR = (3, 13)


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _check_python_runtime() -> None:
    actual = (sys.version_info.major, sys.version_info.minor)
    if sys.implementation.name != "cpython" or actual != EXPECTED_PYTHON_MINOR:
        rendered = ".".join(str(part) for part in actual)
        raise ValueError(
            "release checks require CPython 3.13, "
            f"got {sys.implementation.name} {rendered}"
        )


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


def _version(path: str, value: str, label: str) -> ReleaseVersion:
    try:
        return parse_version(value)
    except ValueError as exc:
        raise ValueError(f"{label}: invalid version {value!r} in {path}") from exc


def _tag(path: str, value: str, label: str) -> ReleaseVersion:
    try:
        return parse_tag(value)
    except ValueError as exc:
        raise ValueError(f"{label}: invalid release tag {value!r} in {path}") from exc


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


def _check_no_static_current_archive_claims(version: ReleaseVersion | str) -> None:
    parsed = version if isinstance(version, ReleaseVersion) else parse_version(version)
    paths = (
        "README.md",
        f"release-notes/release-notes-{parsed.tag}.md",
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
    _check_python_runtime()
    _check_release_asset_hydration()
    expected_text = _single_match(
        "pyproject.toml", r'^version\s*=\s*"([^"]+)"', "pyproject"
    )
    expected = _version("pyproject.toml", expected_text, "pyproject")
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

    rust_version = _single_match(
        "rust-core/Cargo.toml",
        r'^version\s*=\s*"([^"]+)"',
        "rust core",
    )
    lock_version = _extract_cargo_lock_version("mic_eq_core")
    readme_tag = _single_match(
        "README.md",
        r"Current version:\s*`(v[^`]+)`",
        "readme",
    )
    catalog_text = current_version if catalog_version == "__CURRENT_VERSION__" else catalog_version
    window_text = package_version if main_window_version == "__PACKAGE_VERSION__" else main_window_version
    checks = {
        "licenses/source-manifest.json": json.loads(_read("licenses/source-manifest.json"))["project_version"],
        "rust-core/Cargo.toml": _version("rust-core/Cargo.toml", rust_version, "rust core").cargo,
        "Cargo.lock mic_eq_core": _version("Cargo.lock", lock_version, "Cargo.lock").cargo,
        "python/mic_eq/__init__.py": _version("python/mic_eq/__init__.py", package_version, "python package").pep440,
        "python/mic_eq/config_parts/shared.py CURRENT_VERSION": _version("python/mic_eq/config_parts/shared.py", current_version, "shared config").pep440,
        "README.md": _tag("README.md", readme_tag, "readme").tag,
        "python/mic_eq/config_parts/presets.py Preset.version": _version("python/mic_eq/config_parts/shared.py", current_version, "preset default").pep440,
        "python/mic_eq/config_parts/catalogs.py built-ins": _version("python/mic_eq/config_parts/catalogs.py", catalog_text, "built-in preset").pep440,
        "python/mic_eq/ui/main_window.py auto-eq preset": _version("python/mic_eq/ui/main_window.py", window_text, "auto-eq preset").pep440,
    }
    expected_by_path = {
        "licenses/source-manifest.json": expected.pep440,
        "rust-core/Cargo.toml": expected.cargo,
        "Cargo.lock mic_eq_core": expected.cargo,
        "python/mic_eq/__init__.py": expected.pep440,
        "python/mic_eq/config_parts/shared.py CURRENT_VERSION": expected.pep440,
        "README.md": expected.tag,
        "python/mic_eq/config_parts/presets.py Preset.version": expected.pep440,
        "python/mic_eq/config_parts/catalogs.py built-ins": expected.pep440,
        "python/mic_eq/ui/main_window.py auto-eq preset": expected.pep440,
    }

    release_versions = set(
        re.findall(
            r"AudioForge-(v[0-9]+\.[0-9]+\.[0-9]+(?:-rc\.[0-9]+)?)",
            _read("RELEASING.md"),
        )
    )
    _require_pattern(
        "python/mic_eq/config_parts/presets.py",
        r"version:\s*str\s*=\s*CURRENT_VERSION",
        "preset default",
    )
    if not release_versions:
        raise ValueError("RELEASING.md: release archive version string not found")
    if release_versions != {expected.tag}:
        checks["RELEASING.md"] = ", ".join(sorted(release_versions))
        expected_by_path["RELEASING.md"] = expected.tag

    release_notes_path = REPO_ROOT / "release-notes" / f"release-notes-{expected.tag}.md"
    if not release_notes_path.is_file():
        raise ValueError(
            f"release notes: expected file is missing: {release_notes_path.relative_to(REPO_ROOT)}"
        )

    mismatches = {
        path: version
        for path, version in checks.items()
        if version != expected_by_path.get(path, expected.tag)
    }
    if mismatches:
        print(f"Version mismatch: pyproject.toml is {expected.pep440}")
        for path, version in mismatches.items():
            print(f"  {path}: {version}")
        return 1

    print(f"Version strings are in sync: {expected.pep440}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Version check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
