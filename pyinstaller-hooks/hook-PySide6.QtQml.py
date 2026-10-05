"""Collect only the QML modules the main window imports.

The stock hook copies every QML module in the PySide6 wheel (3D, WebEngine,
charts and so on) and, with them, Qt libraries that are in neither the source
manifest nor the license notices.
"""

from pathlib import Path, PurePath

from PyInstaller.utils.hooks.qt import add_qt6_dependencies, pyside6_library_info

QML_MODULES = (
    "QtQml",
    "QtQml/Models",
    "QtQml/WorkerScript",
    "QtQuick",
    "QtQuick/Controls",
    "QtQuick/Controls/impl",
    "QtQuick/Controls/Basic",
    "QtQuick/Controls/Basic/impl",
    "QtQuick/Templates",
    "QtQuick/Layouts",
    "QtQuick/Effects",
    "QtQuick/Window",
)

hiddenimports, binaries, datas = add_qt6_dependencies(__file__)
_source = Path(pyside6_library_info.location["QmlImportsPath"])
_destination = PurePath(pyside6_library_info.qt_rel_dir) / "qml"
for _module in QML_MODULES:
    for _path in sorted((_source / _module).iterdir()):
        if _path.is_file():
            _target = binaries if _path.suffix == ".dll" else datas
            _target.append((str(_path), str(_destination / _module)))
