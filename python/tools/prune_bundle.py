from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def prune_bundle(bundle_root: Path) -> None:
    translations_dir = bundle_root / "_internal" / "PyQt6" / "Qt6" / "translations"
    if translations_dir.exists():
        shutil.rmtree(translations_dir)
        print(f"Pruned Qt translations: {translations_dir}")

    for dist_info_dir in (bundle_root / "_internal").glob("*.dist-info"):
        if dist_info_dir.is_dir():
            shutil.rmtree(dist_info_dir)
            print(f"Removed wheel metadata: {dist_info_dir}")

    for relative_path in (
        Path("_internal/PyQt6/Qt6/bin/Qt6Pdf.dll"),
        Path("_internal/PyQt6/QtPdf.pyd"),
        Path("_internal/PyQt6/QtPdfWidgets.pyd"),
    ):
        candidate = bundle_root / relative_path
        if candidate.exists():
            if candidate.is_dir():
                shutil.rmtree(candidate)
            else:
                candidate.unlink()
            print(f"Removed unused bundled payload: {candidate}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune unused payload from a bundled AudioForge app.")
    parser.add_argument("bundle_root", type=Path, help="Path to dist/AudioForge")
    args = parser.parse_args()
    prune_bundle(args.bundle_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
