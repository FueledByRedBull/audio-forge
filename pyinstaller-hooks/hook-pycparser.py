"""Local PyInstaller hook for pycparser.

Avoids noisy hidden-import warnings when wheels do not ship the generated
parser tables as real modules.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path


def _shipped_parser_tables() -> list[str]:
    spec = find_spec("pycparser")
    if spec is None or spec.submodule_search_locations is None:
        return []

    package_dir = Path(next(iter(spec.submodule_search_locations)))
    hiddenimports: list[str] = []
    for module_name in ("lextab", "yacctab"):
        if (package_dir / f"{module_name}.py").exists():
            hiddenimports.append(f"pycparser.{module_name}")
    return hiddenimports


hiddenimports = _shipped_parser_tables()
