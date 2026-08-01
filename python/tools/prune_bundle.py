from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def is_app_local_system_ucrt(path: Path) -> bool:
    """Return whether *path* is an OS-provided UCRT/API-set forwarder."""

    name = path.name.casefold()
    return name == "ucrtbase.dll" or (
        name.startswith("api-ms-win-") and name.endswith(".dll")
    )


def prune_bundle(bundle_root: Path) -> list[Path]:
    removed: list[Path] = []
    translations_dir = bundle_root / "_internal" / "PyQt6" / "Qt6" / "translations"
    if translations_dir.exists():
        shutil.rmtree(translations_dir)
        print(f"Pruned Qt translations: {translations_dir}")

    packaged_extension_dir = bundle_root / "_internal" / "mic_eq"
    duplicate_extension_dir = bundle_root / "_internal" / "mic_eq_core"
    has_packaged_extension = any(packaged_extension_dir.glob("mic_eq_core*.pyd"))
    if has_packaged_extension and duplicate_extension_dir.exists():
        shutil.rmtree(duplicate_extension_dir)
        print(f"Removed duplicate native extension payload: {duplicate_extension_dir}")

    # AudioForge supports Windows 10/11. On those systems the UCRT in the
    # system directory is always used, even when an application-local copy is
    # present. PyInstaller collection can vary by runner image, so remove this
    # OS payload deterministically rather than allowing host-dependent bundles.
    for candidate in sorted(
        (path for path in bundle_root.rglob("*") if path.is_file()),
        key=lambda path: path.as_posix().casefold(),
    ):
        if is_app_local_system_ucrt(candidate):
            candidate.unlink()
            removed.append(candidate.relative_to(bundle_root))
            print(f"Removed system UCRT/API-set payload: {candidate}")

    for relative_path in (
        Path("_internal/PyQt6/Qt6/bin/Qt6Pdf.dll"),
        Path("_internal/PyQt6/Qt6/bin/Qt6Svg.dll"),
        Path("_internal/PyQt6/QtPdf.pyd"),
        Path("_internal/PyQt6/QtPdfWidgets.pyd"),
        Path("_internal/PyQt6/Qt6/plugins/iconengines/qsvgicon.dll"),
    ):
        candidate = bundle_root / relative_path
        if candidate.exists():
            if candidate.is_dir():
                shutil.rmtree(candidate)
            else:
                candidate.unlink()
            print(f"Removed unused bundled payload: {candidate}")
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune unused payload from a bundled AudioForge app.")
    parser.add_argument("bundle_root", type=Path, help="Path to dist/AudioForge")
    args = parser.parse_args()
    prune_bundle(args.bundle_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
