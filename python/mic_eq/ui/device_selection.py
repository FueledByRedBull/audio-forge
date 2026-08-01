"""Pure device-selection policy for AudioForge UI controllers."""

from __future__ import annotations

from ..config import DeviceIdentity, coerce_device_identity


VIRTUAL_OUTPUT_MARKERS = ("cable", "vb-audio", "virtual")


def find_identity_index(
    identities: list[DeviceIdentity | None],
    identity: DeviceIdentity | None,
) -> int:
    """Resolve a persisted identity without guessing among duplicate names."""
    if identity is None:
        return -1

    if identity.endpoint_id:
        endpoint_matches = [
            index
            for index, item in enumerate(identities)
            if isinstance(item, DeviceIdentity)
            and item.endpoint_id == identity.endpoint_id
            and (
                not identity.direction
                or not item.direction
                or item.direction == identity.direction
            )
        ]
        if len(endpoint_matches) == 1:
            return endpoint_matches[0]
        # An endpoint-backed identity is authoritative. Falling through to a
        # friendly-name match could select replacement hardware and apply the
        # old endpoint's latency/preset state to the wrong route.
        return -1

    normalized_name = " ".join(identity.name.casefold().split())
    name_matches = [
        (index, item)
        for index, item in enumerate(identities)
        if isinstance(item, DeviceIdentity)
        and " ".join(item.name.casefold().split()) == normalized_name
    ]
    if len(name_matches) == 1:
        return name_matches[0][0]

    # Enumeration ordinals can select a duplicate in the current snapshot, but
    # they are not stable identity across reconnect/restart. Fail closed when
    # no endpoint ID disambiguates duplicate names.
    return -1


def identity_is_persistable(
    identities: list[DeviceIdentity | None],
    identity: DeviceIdentity | None,
) -> bool:
    """Return whether an identity can safely own persisted route state."""
    if identity is None:
        return False
    if identity.endpoint_id:
        return (
            sum(
                isinstance(item, DeviceIdentity)
                and item.endpoint_id == identity.endpoint_id
                for item in identities
            )
            == 1
        )
    normalized_name = " ".join(identity.name.casefold().split())
    return (
        sum(
            isinstance(item, DeviceIdentity)
            and " ".join(item.name.casefold().split()) == normalized_name
            for item in identities
        )
        == 1
    )


def device_name_ordinal(identity: DeviceIdentity | None) -> int:
    """Return the native duplicate-name selector for an identity."""
    if identity is None or identity.name_ordinal is None:
        return 0
    return identity.name_ordinal


def start_processor_for_route(
    processor: object,
    input_device: object,
    output_device: object,
) -> object:
    """Start the native processor on the exact selected duplicate-name occurrences."""
    input_identity = coerce_device_identity(input_device)
    output_identity = coerce_device_identity(output_device)
    start = getattr(processor, "start")
    return start(
        input_identity.name if input_identity is not None else None,
        output_identity.name if output_identity is not None else None,
        device_name_ordinal(input_identity),
        device_name_ordinal(output_identity),
    )


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
    "device_name_ordinal",
    "find_identity_index",
    "identity_is_persistable",
    "preferred_output_index",
    "start_processor_for_route",
]
