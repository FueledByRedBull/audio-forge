"""What every Qt Quick scene of the app shares: its theme tokens and how it is loaded."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtQuickWidgets import QQuickWidget

from .components import icon_font
from .theme import PALETTE, RADIUS_CARD, RADIUS_CONTROL, prefers_reduced_motion, qcolor

QML_DIR = Path(__file__).parent / "qml"

# QML import paths taken out of the environment so child processes do not inherit them; each
# scene's engine gets them directly instead (see app_bootstrap).
_import_paths: list[str] = []


def keep_import_paths(paths: list[str]) -> None:
    """Have every scene loaded from now on search ``paths`` for QML modules."""
    _import_paths[:] = paths


def quick_theme() -> dict:
    """The theme tokens a scene reads as ``Look.t``."""

    font = icon_font()
    return {
        **{k: v for k, v in asdict(PALETTE).items() if isinstance(v, str)},
        "bandColors": list(PALETTE.eq_band_colors),
        "iconFont": font.family() if font is not None else "",
        "radiusCard": RADIUS_CARD,
        "radiusControl": RADIUS_CONTROL,
        "motion": 0 if prefers_reduced_motion() else 140,
    }


def load_scene(view: QQuickWidget, source: Path, theme: dict, properties: dict) -> str | None:
    """Load ``source`` into ``view``; return why it did not load, or ``None`` once it is ready."""

    engine = view.engine()
    for path in _import_paths:
        engine.addImportPath(path)
    view.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
    view.setClearColor(qcolor(PALETTE.app_surface))
    view.rootContext().setContextProperty("theme", theme)
    view.setInitialProperties(properties)
    view.setSource(QUrl.fromLocalFile(str(source)))
    if view.status() == QQuickWidget.Status.Ready:
        return None
    return "; ".join(error.toString() for error in view.errors()) or "no details were reported"
