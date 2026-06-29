"""Pure device-selection policy for AudioForge UI controllers."""

from __future__ import annotations

from ..config import DeviceIdentity


VIRTUAL_OUTPUT_MARKERS = ("cable", "vb-audio", "virtual")


def find_identity_index(
    identities: list[DeviceIdentity | None],
    identity: DeviceIdentity | None,
) -> int:
    """Find an exact device identity, then fall back by stable name."""
    if identity is None:
        return -1

    for index, item in enumerate(identities):
        if item == identity:
            return index

    for index, item in enumerate(identities):
        if isinstance(item, DeviceIdentity) and item.name == identity.name:
            return index

    return -1


def default_device_index(identities: list[DeviceIdentity | None]) -> int:
    """Return the default device index, or the first item when no default exists."""
    for index, item in enumerate(identities):
        if isinstance(item, DeviceIdentity) and item.is_default:
            return index
    return 0 if identities else -1


def preferred_output_index(identities: list[DeviceIdentity | None]) -> int:
    """Prefer Windows virtual routing outputs, then the default, then first item."""
    for index, item in enumerate(identities):
        if not isinstance(item, DeviceIdentity):
            continue
        name_lower = item.name.lower()
        if any(marker in name_lower for marker in VIRTUAL_OUTPUT_MARKERS):
            return index
    return default_device_index(identities)


__all__ = [
    "VIRTUAL_OUTPUT_MARKERS",
    "default_device_index",
    "find_identity_index",
    "preferred_output_index",
]
