"""Lightweight packaging sanity checks for the portable Windows bundle."""

from __future__ import annotations

import argparse
import json
import re
import sys
import sysconfig
import tomllib
from pathlib import Path

from prune_bundle import is_app_local_system_ucrt


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_BUNDLE_FILES = (
    "AudioForge.exe",
    "_internal/df.dll",
    "_internal/onnxruntime.dll",
    "_internal/onnxruntime_providers_shared.dll",
    "_internal/models/DeepFilterNet3_ll_onnx.tar.gz",
    "_internal/models/DeepFilterNet3_onnx.tar.gz",
    "_internal/models/silero_vad.onnx",
    "_internal/audioforge-build.json",
    "_internal/licenses/LICENSE",
    "_internal/licenses/GPL-3.0.txt",
    "_internal/licenses/dependencies/inventory.json",
    "_internal/licenses/DeepFilterNet-LICENSE.txt",
    "_internal/licenses/ONNXRuntime-LICENSE.txt",
    "_internal/licenses/ONNXRuntime-ThirdPartyNotices.txt",
    "_internal/licenses/Silero-VAD-LICENSE.txt",
    "_internal/licenses/THIRD_PARTY_NOTICES.md",
)
REQUIRED_MANIFEST_ASSETS = (
    "df.dll",
    "target/onnxruntime-cpu/lib/onnxruntime.dll",
    "target/onnxruntime-cpu/lib/onnxruntime.lib",
    "target/onnxruntime-cpu/lib/onnxruntime_providers_shared.dll",
    "models/DeepFilterNet3_ll_onnx.tar.gz",
    "models/DeepFilterNet3_onnx.tar.gz",
    "models/silero_vad.onnx",
)


def _contains(path: str, needle: str) -> bool:
    return needle in (REPO_ROOT / path).read_text(encoding="utf-8")


def _load_asset_manifest() -> tuple[list[dict[str, object]], list[str]]:
    manifest_path = REPO_ROOT / "release-assets.json"
    errors: list[str] = []
    if not manifest_path.is_file():
        return [], ["release-assets.json is missing"]

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [], [f"release-assets.json is invalid JSON: {exc}"]

    assets = raw.get("assets")
    if not isinstance(assets, list) or not assets:
        return [], ["release-assets.json must contain a non-empty assets list"]

    seen_paths: set[str] = set()
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict):
            errors.append(f"release-assets.json assets[{index}] must be an object")
            continue
        path = asset.get("path")
        sha256 = asset.get("sha256")
        source = asset.get("source")
        license_note = asset.get("license")
        if not isinstance(path, str) or not path:
            errors.append(f"release-assets.json assets[{index}].path is required")
            continue
        seen_paths.add(path.replace("\\", "/"))
        bundle_path = asset.get("bundle_path")
        if isinstance(bundle_path, str) and bundle_path:
            seen_paths.add(bundle_path.replace("\\", "/"))
        if not isinstance(sha256, str) or len(sha256) != 64:
            errors.append(f"release-assets.json asset {path} must have a 64-character sha256")
        if not isinstance(source, str) or not source:
            errors.append(f"release-assets.json asset {path} must document source")
        if not isinstance(license_note, str) or not license_note:
            errors.append(f"release-assets.json asset {path} must document license")

    for required in REQUIRED_MANIFEST_ASSETS:
        if required not in seen_paths:
            errors.append(f"release-assets.json is missing required asset {required}")

    return assets, errors


def check_source_packaging() -> list[str]:
    errors: list[str] = []

    spec_text = (REPO_ROOT / "AudioForge.spec").read_text(encoding="utf-8")
    for scipy_module in ("scipy.integrate", "scipy.interpolate", "scipy.stats"):
        if re.search(rf'^\s*"{re.escape(scipy_module)}",\s*$', spec_text, re.MULTILINE):
            errors.append(
                f"AudioForge.spec must not exclude runtime SciPy dependency {scipy_module}"
            )

    spec_expectations = [
        ("AudioForge.spec", "onnxruntime.dll"),
        ("AudioForge.spec", "df.dll"),
        ("AudioForge.spec", "models"),
        ("AudioForge.spec", "licenses"),
        ("AudioForge.spec", "AudioForge.ico"),
        ("AudioForge.spec", '"ssl",'),
        ("AudioForge.spec", '"_ssl",'),
        ("AudioForge.spec", '"_hashlib",'),
    ]
    script_expectations = [
        ("build_exe.ps1", "$PSScriptRoot"),
        ("build_exe.ps1", "[string]$PythonPath"),
        ("build_exe.ps1", "VIRTUAL_ENV"),
        ("build_exe.ps1", "EXT_SUFFIX"),
        ("build_exe.ps1", "onnxruntime.dll"),
        ("build_exe.ps1", "ORT_PREFER_DYNAMIC_LINK"),
        ("build_exe.ps1", "DeepFilterNet3_ll_onnx.tar.gz"),
        ("build_exe.ps1", "DeepFilterNet3_onnx.tar.gz"),
        ("build_exe.ps1", "silero_vad.onnx"),
        ("build_exe.ps1", "verify_release_assets.py"),
        ("build_exe.ps1", "license_inventory.py"),
        ("build_exe.ps1", "prune_bundle.py"),
        ("build_exe.ps1", "audioforge-build.json"),
        ("python/tools/prune_bundle.py", "is_app_local_system_ucrt"),
        ("python/tools/prune_bundle.py", "is_unused_openssl_payload"),
        ("python/tools/release_provenance.py", "build_bundle_manifest"),
        ("build_msi.ps1", "[string]$PythonPath"),
        ("python/tools/run_semgrep.py", '"--exclude=.venv*"'),
        ("python/tools/run_semgrep.py", '"--exclude=credentials.*"'),
    ]
    runtime_expectations = [
        (
            "python/mic_eq/ui/app_bootstrap.py",
            "configure_deepfilter_runtime_paths(str(lib_path), str(model_dir_with_model))",
        ),
        (
            "rust-core/src/dsp/deepfilter_ffi.rs",
            'env::var("AUDIOFORGE_ALLOW_EXTERNAL_DF")',
        ),
        ("launcher.py", "os.add_dll_directory(str(dll_dir))"),
    ]
    workflow_expectations = [
        (".github/workflows/release-package.yml", "maturin develop --release"),
        (".github/workflows/release-package.yml", "python/tools/verify_release_assets.py"),
        (".github/workflows/release-package.yml", "powershell -ExecutionPolicy Bypass -File .\\build_exe.ps1"),
        (".github/workflows/release-package.yml", "-PythonPath .\\.venv\\Scripts\\python.exe"),
        (".github/workflows/release-package.yml", "ORT_LIB_LOCATION=$ortLib"),
        (".github/workflows/release-package.yml", "python/tools/package_smoke.py"),
        (".github/workflows/release-package.yml", "actions/upload-artifact@"),
        (".github/workflows/release-package.yml", "AudioForge-$expectedTag-win64-ultra.7z"),
        (".github/workflows/release-package.yml", "fetch_release_assets.py"),
        (".github/workflows/release-package.yml", "release_provenance.py create"),
        (".github/workflows/release-package.yml", "release_provenance.py verify"),
        (".github/workflows/release-promote.yml", "actions/download-artifact@"),
        (".github/workflows/release-promote.yml", "release_provenance.py verify"),
        (".github/workflows/release-promote.yml", "gh release upload"),
        (
            ".github/workflows/release-hardware-qualify.yml",
            "evaluate_hardware_validation.py",
        ),
        (
            ".github/workflows/release-hardware-qualify.yml",
            "audioforge-release-hardware-validation-",
        ),
        (
            ".github/workflows/release-hardware-matrix.yml",
            "evaluate_hardware_matrix.py",
        ),
        (
            ".github/workflows/release-hardware-matrix.yml",
            "audioforge-release-hardware-matrix-",
        ),
    ]

    for path, needle in [
        *spec_expectations,
        *script_expectations,
        *runtime_expectations,
        *workflow_expectations,
    ]:
        if not _contains(path, needle):
            errors.append(f"{path}: missing expected packaging reference {needle!r}")

    bootstrap_source = (REPO_ROOT / "python/mic_eq/ui/app_bootstrap.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        'os.environ["DEEPFILTER_LIB_PATH"]',
        'os.environ["DEEPFILTER_MODEL_PATH"]',
        'os.environ.setdefault("DEEPFILTER_LIB_PATH"',
        'os.environ.setdefault("DEEPFILTER_MODEL_PATH"',
    ):
        if forbidden in bootstrap_source:
            errors.append(
                "python/mic_eq/ui/app_bootstrap.py must register bundled DeepFilter "
                f"paths directly instead of writing {forbidden!r}"
            )

    if "dist-info" in (REPO_ROOT / "python/tools/prune_bundle.py").read_text(encoding="utf-8"):
        errors.append("python/tools/prune_bundle.py must not prune dependency dist-info metadata")

    _assets, manifest_errors = _load_asset_manifest()
    errors.extend(manifest_errors)

    for notice in REQUIRED_BUNDLE_FILES:
        if not notice.startswith("_internal/licenses/"):
            continue
        source_name = notice.removeprefix("_internal/licenses/")
        if source_name.startswith("dependencies/"):
            continue  # Generated from the locked build environment.
        source_path = (
            REPO_ROOT / "LICENSE"
            if source_name == "LICENSE"
            else REPO_ROOT / "licenses" / source_name
        )
        if not source_path.is_file():
            errors.append(f"required bundle notice source is missing: {source_path}")

    return errors


def _has_bundle_file(dist: Path, relative_path: str) -> bool:
    return (dist / Path(relative_path)).is_file()


def _expected_extension_suffix() -> str:
    return str(sysconfig.get_config_var("EXT_SUFFIX") or ".pyd")


def _packaged_extensions(dist: Path) -> list[Path]:
    extension_dir = dist / "_internal" / "mic_eq"
    return sorted(
        path
        for path in extension_dir.glob("mic_eq_core*.pyd")
        if path.is_file()
    )


def _has_native_extension(dist: Path) -> bool:
    expected = dist / "_internal" / "mic_eq" / (
        "mic_eq_core" + _expected_extension_suffix()
    )
    return expected.is_file()


def _foreign_native_extensions(dist: Path) -> list[Path]:
    expected_name = "mic_eq_core" + _expected_extension_suffix()
    return [path for path in _packaged_extensions(dist) if path.name != expected_name]


def _has_duplicate_native_extension(dist: Path) -> bool:
    duplicate_dir = dist / "_internal" / "mic_eq_core"
    return any(
        path.is_file() and path.name.startswith("mic_eq_core") and path.suffix == ".pyd"
        for path in duplicate_dir.glob("mic_eq_core*.pyd")
    )


def _expected_version() -> str:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def _bundle_version(dist: Path) -> str | None:
    build_info_path = dist / "_internal" / "audioforge-build.json"
    if not build_info_path.is_file():
        return None
    try:
        build_info = json.loads(build_info_path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    actual = build_info.get("version")
    return actual if isinstance(actual, str) else None


def _check_bundle_identity(dist: Path) -> list[str]:
    build_info_path = dist / "_internal" / "audioforge-build.json"
    if not build_info_path.is_file():
        return []
    actual = _bundle_version(dist)
    if actual is None:
        return [f"{build_info_path} is invalid or lacks a string version"]
    expected = _expected_version()
    if actual != expected:
        return [
            f"{build_info_path} reports version {actual!r}; expected {expected!r}"
        ]
    return []


def check_dist_bundle(
    dist: Path | None = None,
    *,
    allow_historical_ucrt_for_version: str | None = None,
) -> list[str]:
    if dist is None:
        dist = REPO_ROOT / "dist" / "AudioForge"
    errors: list[str] = []
    if not dist.is_dir():
        errors.append(f"{dist} is missing")
        return errors

    for relative_path in REQUIRED_BUNDLE_FILES:
        if not _has_bundle_file(dist, relative_path):
            errors.append(f"{dist} does not contain {relative_path}")

    if not _has_native_extension(dist):
        errors.append(
            f"{dist} does not contain _internal/mic_eq/mic_eq_core{_expected_extension_suffix()}"
        )

    foreign_extensions = _foreign_native_extensions(dist)
    if foreign_extensions:
        names = ", ".join(path.relative_to(dist).as_posix() for path in foreign_extensions)
        errors.append(f"{dist} contains foreign Python ABI extensions: {names}")

    if _has_duplicate_native_extension(dist):
        errors.append(f"{dist} contains duplicate _internal/mic_eq_core/mic_eq_core*.pyd")

    directml_payloads = sorted(
        path.relative_to(dist).as_posix()
        for path in dist.rglob("*")
        if path.is_file() and path.name.casefold() == "directml.dll"
    )
    if directml_payloads:
        errors.append(
            f"{dist} contains retired DirectML payload(s): " + ", ".join(directml_payloads)
        )
    retired_notice = dist / "_internal" / "licenses" / "DirectML-LICENSE.txt"
    if retired_notice.is_file():
        errors.append(f"{dist} contains the retired DirectML license notice")

    openssl_payloads = sorted(
        path.relative_to(dist).as_posix()
        for path in dist.rglob("*")
        if path.is_file()
        and (
            path.name.casefold() in {"_ssl.pyd", "_hashlib.pyd"}
            or path.name.casefold().startswith("libssl-")
            or path.name.casefold().startswith("libcrypto-")
        )
    )
    if openssl_payloads:
        errors.append(
            f"{dist} contains excluded OpenSSL payload(s): "
            + ", ".join(openssl_payloads)
        )

    forbidden_ucrt = sorted(
        path.relative_to(dist).as_posix()
        for path in dist.rglob("*")
        if path.is_file() and is_app_local_system_ucrt(path)
    )
    if forbidden_ucrt:
        historical_exception = (
            allow_historical_ucrt_for_version == "1.10.1"
            and _bundle_version(dist) == allow_historical_ucrt_for_version
            and len(forbidden_ucrt) == 46
        )
        if not historical_exception:
            errors.append(
                f"{dist} contains OS-provided app-local UCRT/API-set payloads: "
                + ", ".join(forbidden_ucrt)
            )

    # Qt uses Windows' ICU C ABI. An unrelated ICU collected from build PATH
    # can export different symbols under the same DLL name and break startup.
    bundled_system_icu = sorted(
        path.relative_to(dist).as_posix()
        for path in dist.rglob("*")
        if path.is_file() and path.name.casefold() in {"icu.dll", "icuuc.dll", "icuin.dll"}
    )
    if bundled_system_icu:
        errors.append(f"{dist} contains app-local Windows ICU: " + ", ".join(bundled_system_icu))

    for plugin in ("qpdf.dll", "qsvg.dll"):
        if (dist / "_internal/PyQt6/Qt6/plugins/imageformats" / plugin).exists():
            errors.append(f"{dist} contains unused image plugin without its Qt module: {plugin}")

    errors.extend(_check_bundle_identity(dist))

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Check packaging source files without requiring a built dist/AudioForge bundle.",
    )
    parser.add_argument(
        "--dist",
        type=Path,
        help="validate this extracted bundle instead of dist/AudioForge",
    )
    parser.add_argument(
        "--allow-historical-ucrt-for-version",
        choices=("1.10.1",),
        help=(
            "qualification-only exception for the exact historical v1.10.1 "
            "bundle's 46 inert app-local UCRT/API-set files"
        ),
    )
    args = parser.parse_args()

    errors = check_source_packaging()
    if not args.source_only:
        errors.extend(
            check_dist_bundle(
                args.dist.resolve() if args.dist else None,
                allow_historical_ucrt_for_version=(
                    args.allow_historical_ucrt_for_version
                ),
            )
        )

    if errors:
        print("Package smoke check failed:")
        for error in errors:
            print(f"  {error}")
        return 1

    print("Package smoke check passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Package smoke check crashed: {exc}", file=sys.stderr)
        raise SystemExit(1)
