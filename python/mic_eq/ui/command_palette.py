"""The command palette: one searchable list over the window's own commands.

Every row runs through the proxy the toolbar, a stage's page or a menu already uses, so its
behaviour, enabled state and confirmations are the same. Filtering, selection and the draft
of a numeric setting live here, so they are testable without the scene.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Property, QObject, Qt, Signal, Slot
from PySide6.QtGui import QAccessible, QAccessibleAnnouncementEvent, QAction, QKeySequence
from PySide6.QtWidgets import QWidget

from .processing_controls import SettingControl
from .shell_state import ActionControl, PageState

SHORTCUT = "Ctrl+K"
CALLERS_LINE = "Callers hear this."
NO_MATCH_TEXT = "No matching commands"
# The numeric settings a row can change: each one sets a level or an amount callers hear, and
# the setting's own range keeps every step safe. (section, key, title)
NUMBER_SETTINGS = (
    ("gate", "threshold_db", "Gate threshold"),
    ("compressor", "threshold_db", "Compressor threshold"),
    ("compressor", "ratio", "Compressor ratio"),
    ("compressor", "makeup_gain_db", "Compressor makeup gain"),
    ("limiter", "ceiling_db", "Limiter ceiling"),
)


@dataclass(frozen=True)
class Command:
    """One row: what it is called, what it says, and how it runs."""

    key: str
    group: str
    kind: str  # "action", "toggle", "number" or "page"
    title: Callable[[], str]
    detail: Callable[[], str]
    enabled: Callable[[], bool]
    run: Callable[[], None]
    callers: bool
    value: Callable[[], str] = lambda: ""
    setting: SettingControl | None = None


def action_command(key: str, title: str | Callable[[], str], detail: str, proxy: ActionControl,
                   *, callers: bool, group: str = "Action") -> Command:
    return Command(
        key, group, "action", title if callable(title) else (lambda: title), lambda: detail,
        lambda: bool(proxy.property("enabled")), proxy.click, callers,
    )


def preset_command(proxy: ActionControl) -> Command:
    """A built-in preset, loaded through its entry in the Presets menu."""

    name = str(proxy.property("text"))
    return action_command("preset:" + name, name, proxy.action.toolTip(), proxy, callers=True, group="Preset")


def toggle_command(name: str, proxy) -> Command:
    """A stage's on/off switch, through the proxy its page uses."""

    return Command(
        "stage:" + name, "Stage", "toggle", lambda: name,
        lambda: "Enter turns it off." if proxy.property("checked") else "Enter turns it on.",
        lambda: bool(proxy.property("enabled")), proxy.click, True,
        value=lambda: "On" if proxy.property("checked") else "Off",
    )


def number_command(key: str, title: str, proxy: SettingControl) -> Command:
    """A numeric setting; the palette keeps a draft of it until Enter."""

    return Command(
        "setting:" + key, "Setting", "number", lambda: title,
        lambda: "Left and Right change it; Enter applies it.",
        lambda: bool(proxy.property("enabled")), lambda: None, True, value=lambda: str(proxy.property("text")),
        setting=proxy,
    )


def page_command(name: str, index: int, pages: PageState) -> Command:
    return Command(
        "page:" + name, "Page", "page", lambda: "Open " + name, lambda: f"Show the {name} page.",
        lambda: True, lambda: pages.setIndex(index), False,
    )


def window_commands(
    *, top: dict, presets: dict, chain: list[dict], settings: dict[tuple[str, str], SettingControl],
    preset_actions: list[ActionControl], pages: list[tuple[str, int]], page_state: PageState,
) -> list[Command]:
    """The rows, built from the proxies the scene's toolbar, stage rows, menus and pages use."""

    mute: ActionControl = top["mute"]
    return [
        action_command("start", "Start processing", "Start processing your voice.", top["start"], callers=True),
        action_command("stop", "Stop processing", "Stop processing your voice.", top["stop"], callers=True),
        action_command(
            "mute", lambda: "Unmute" if mute.property("checked") else "Mute",
            "Keep transmission muted until you turn it off.", mute, callers=True,
        ),
        action_command(
            "voice_setup", "Run Auto Voice Setup",
            "Record room noise and speech, then set up the voice chain.", presets["voiceSetup"], callers=False,
        ),
        action_command(
            "test_sound", "Test my sound",
            "Record five seconds, then hear the recording and the current settings.",
            presets["testSound"], callers=False,
        ),
        *(toggle_command(row["name"], row["toggle"]) for row in chain if row["toggle"] is not None),
        *(preset_command(proxy) for proxy in preset_actions),
        *(number_command(f"{section}.{key}", title, settings[(section, key)])
          for section, key, title in NUMBER_SETTINGS),
        *(page_command(name, index, page_state) for name, index in pages),
    ]


class CommandPalette(QObject):
    """The palette's open state, filter, selection and numeric drafts, for the scene to render."""

    changed = Signal()

    def __init__(self, parent: QObject, window: QWidget, commands: list[Command], sources) -> None:
        super().__init__(parent)
        self._window = window
        self._commands = commands
        self._shown, self._query, self._current = False, "", -1
        self._drafts: dict[str, float] = {}
        self._rows: list[dict] = []
        self._matches: list[Command] = []
        for signal in sources:
            signal.connect(self._sources_changed)
        # A window shortcut, never a global hotkey; it belongs to the palette so it goes with it.
        self.action = QAction("Command palette", self)
        self.action.setShortcut(QKeySequence(SHORTCUT))
        self.action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.action.setAutoRepeat(False)
        self.action.triggered.connect(self.open)
        window.addAction(self.action)

    shown = Property(bool, lambda self: self._shown, notify=changed)
    query = Property(str, lambda self: self._query, notify=changed)
    rows = Property(list, lambda self: self._rows, notify=changed)
    current = Property(int, lambda self: self._current, notify=changed)
    noMatchText = Property(str, lambda self: NO_MATCH_TEXT, constant=True)
    shortcut = Property(str, lambda self: SHORTCUT, constant=True)

    def _row(self, command: Command) -> dict:
        detail, value = command.detail(), command.value()
        draft = self._drafts.get(command.key)
        if draft is not None and command.setting is not None:
            value = command.setting.format_value(draft)
            detail = f"Was {command.setting.property('text')}. Enter applies it; Esc discards it."
        callers = CALLERS_LINE if command.callers else ""
        return {
            "key": command.key, "group": command.group, "kind": command.kind,
            "title": command.title(), "detail": detail, "callers": callers,
            "description": f"{detail} {callers}".strip(), "enabled": bool(command.enabled()),
            "value": value, "draft": draft is not None,
        }

    def _refresh(self, *, reselect: bool = False) -> None:
        terms = self._query.casefold().split()
        matches = []
        rows = []
        for command in self._commands:
            row = self._row(command)
            words = f"{row['title']} {row['group']}".casefold()
            if all(term in words for term in terms):
                matches.append(command)
                rows.append(row)
        current = min(max(0 if reselect else self._current, 0), len(rows) - 1)
        if (rows, current) != (self._rows, self._current):
            self._matches, self._rows, self._current = matches, rows, current
            self.changed.emit()
        else:
            self._matches = matches

    @Slot()
    def _sources_changed(self) -> None:
        if self._shown:
            self._refresh()

    @Slot()
    def open(self) -> None:
        if self._shown:
            return
        self._shown, self._query = True, ""
        self._drafts.clear()
        self._refresh(reselect=True)
        self.changed.emit()

    @Slot()
    def close(self) -> None:
        """Close without changing anything: a draft is dropped."""
        if not self._shown:
            return
        self._shown = False
        self._drafts.clear()
        self._refresh()
        self.changed.emit()

    @Slot(str)
    def setQuery(self, query: str) -> None:
        if query != self._query:
            self._query = query
            self._refresh(reselect=True)
            self.changed.emit()

    @Slot(int)
    def move(self, step: int) -> None:
        """Move the selection by ``step`` rows, stopping at either end."""
        if self._rows:
            self.select(max(0, min(len(self._rows) - 1, self._current + step)))
            self._announce()

    @Slot(int)
    def select(self, index: int) -> None:
        if 0 <= index < len(self._rows) and index != self._current:
            self._current = index
            self.changed.emit()

    def _selected(self) -> Command | None:
        return self._matches[self._current] if 0 <= self._current < len(self._matches) else None

    @Slot(int, result=bool)
    def adjust(self, direction: int) -> bool:
        """Step the selected numeric setting's draft; False when the selected row is not one."""
        command = self._selected()
        proxy = command.setting if command is not None else None
        if command is None or proxy is None:
            return False
        current, step = float(proxy.property("value")), float(proxy.property("step"))
        if proxy.property("enabled") and direction:
            base = self._drafts.get(command.key, current)
            low, high = float(proxy.property("minimum")), float(proxy.property("maximum"))
            draft = round(min(high, max(low, base + direction * step)), 6)
            if draft == current:
                self._drafts.pop(command.key, None)
            else:
                self._drafts[command.key] = draft
            self._refresh()
            self._announce()
        return True

    @Slot(int)
    def adjustAt(self, index: int, direction: int) -> None:
        self.select(index)
        self.adjust(direction)

    @Slot()
    def run(self) -> None:
        """Run the selected row; a disabled one does nothing and the palette stays open."""
        command = self._selected()
        if command is None or not command.enabled():
            return
        draft = self._drafts.get(command.key)
        # Closed first, so focus is back where it was before a command opens a dialog.
        self.close()
        if command.setting is not None:
            if draft is not None:
                # As a page edit does: the value, then the release that ends the edit in history.
                command.setting.setValue(draft)
                command.setting.released()
        else:
            command.run()

    @Slot(int)
    def runAt(self, index: int) -> None:
        self.select(index)
        self.run()

    def _announce(self) -> None:
        # The search field keeps focus, so the selected row is told to assistive technology.
        if 0 <= self._current < len(self._rows) and QAccessible.isActive():
            row = self._rows[self._current]
            text = ", ".join(part for part in (row["title"], row["value"], row["description"]) if part)
            QAccessible.updateAccessibility(QAccessibleAnnouncementEvent(self._window, text))
