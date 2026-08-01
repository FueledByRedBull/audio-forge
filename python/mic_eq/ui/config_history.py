"""Bounded, transactional history for immutable processing snapshots."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..config import Preset


DEFAULT_HISTORY_LIMIT = 50
_CONFIGURATION_ROOTS = frozenset(
    {
        "gate",
        "eq",
        "rnnoise",
        "deesser",
        "compressor",
        "limiter",
        "bypass",
    }
)


@dataclass(frozen=True, slots=True)
class ConfigurationSnapshot:
    """One canonical, immutable, validated preset payload."""

    payload_json: str
    label: str
    source: str

    @classmethod
    def from_preset(
        cls,
        preset: Preset,
        *,
        label: str,
        source: str,
    ) -> "ConfigurationSnapshot":
        payload_json = json.dumps(
            preset.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        snapshot = cls(payload_json, str(label), str(source))
        snapshot.to_preset()
        return snapshot

    def payload(self) -> dict[str, Any]:
        parsed = json.loads(self.payload_json)
        if not isinstance(parsed, dict):
            raise ValueError("configuration snapshot root must be an object")
        return parsed

    def to_preset(self) -> Preset:
        return Preset.from_dict(self.payload())


class BoundedConfigurationHistory:
    """A bounded history whose cursor moves only after successful restore."""

    def __init__(self, limit: int = DEFAULT_HISTORY_LIMIT) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 2:
            raise ValueError("history limit must be an integer of at least two")
        self._limit = limit
        self._entries: list[ConfigurationSnapshot] = []
        self._cursor = -1

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def current(self) -> ConfigurationSnapshot | None:
        if not 0 <= self._cursor < len(self._entries):
            return None
        return self._entries[self._cursor]

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return 0 <= self._cursor < len(self._entries) - 1

    @property
    def undo_label(self) -> str | None:
        return self.current.label if self.can_undo and self.current else None

    @property
    def redo_label(self) -> str | None:
        if not self.can_redo:
            return None
        return self._entries[self._cursor + 1].label

    def initialize(self, snapshot: ConfigurationSnapshot) -> None:
        snapshot.to_preset()
        self._entries = [snapshot]
        self._cursor = 0

    def record(self, snapshot: ConfigurationSnapshot) -> bool:
        snapshot.to_preset()
        current = self.current
        if current is not None and current.payload_json == snapshot.payload_json:
            return False
        if self._cursor < len(self._entries) - 1:
            del self._entries[self._cursor + 1 :]
        self._entries.append(snapshot)
        self._cursor = len(self._entries) - 1
        overflow = len(self._entries) - self._limit
        if overflow > 0:
            del self._entries[:overflow]
            self._cursor -= overflow
        return True

    def undo(
        self,
        restore: Callable[[ConfigurationSnapshot], None],
    ) -> ConfigurationSnapshot | None:
        if not self.can_undo:
            return None
        target_index = self._cursor - 1
        target = self._entries[target_index]
        target.to_preset()
        restore(target)
        self._cursor = target_index
        return target

    def redo(
        self,
        restore: Callable[[ConfigurationSnapshot], None],
    ) -> ConfigurationSnapshot | None:
        if not self.can_redo:
            return None
        target_index = self._cursor + 1
        target = self._entries[target_index]
        target.to_preset()
        restore(target)
        self._cursor = target_index
        return target


def changed_configuration_paths(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> set[str]:
    """Return changed preset value paths, excluding metadata/provenance."""

    changed: set[str] = set()

    def visit(before: object, after: object, path: str) -> None:
        if isinstance(before, Mapping) and isinstance(after, Mapping):
            for key in sorted(set(before) | set(after)):
                child = f"{path}.{key}" if path else str(key)
                visit(before.get(key), after.get(key), child)
            return
        if isinstance(before, list) and isinstance(after, list):
            for index in range(max(len(before), len(after))):
                child = f"{path}.{index}"
                left = before[index] if index < len(before) else None
                right = after[index] if index < len(after) else None
                visit(left, right, child)
            return
        if before != after:
            changed.add(path)

    for root in sorted(_CONFIGURATION_ROOTS):
        visit(previous.get(root), current.get(root), root)
    return changed


def explicit_provenance_after_edit(
    previous: ConfigurationSnapshot,
    current_preset: Preset,
) -> dict[str, str]:
    """Preserve old provenance and mark only changed value paths explicit."""

    previous_payload = previous.payload()
    current_payload = current_preset.to_dict()
    provenance_raw = previous_payload.get("value_provenance", {})
    if not isinstance(provenance_raw, dict):
        raise ValueError("snapshot value_provenance must be an object")
    provenance = {str(key): str(value) for key, value in provenance_raw.items()}
    for path in changed_configuration_paths(previous_payload, current_payload):
        provenance[path] = "explicit"
    return provenance


__all__ = [
    "BoundedConfigurationHistory",
    "ConfigurationSnapshot",
    "DEFAULT_HISTORY_LIMIT",
    "changed_configuration_paths",
    "explicit_provenance_after_edit",
]
