"""Pure device-selection policy for AudioForge UI controllers."""

from __future__ import annotations

import sys
import threading

from ..config import DeviceIdentity, coerce_device_identity


VIRTUAL_OUTPUT_MARKERS = ("cable", "vb-audio", "virtual")
# The sink of AudioForge's own PipeWire virtual microphone (audio/virtual_mic.rs).
AUDIOFORGE_MIC_ENDPOINT = "pipewire:audioforge_mic_input"
# The input the PipeWire host lists as "Follow system default" (audio/device.rs).
FOLLOW_DEFAULT_INPUT_ENDPOINT = "pipewire:input_default"
OWN_MICROPHONE_MESSAGE = (
    "No microphone is available: the system's default input is AudioForge Mic, which carries "
    "AudioForge's own output, so recording from it would feed AudioForge back into itself. "
    "Connect a microphone or choose one in the Input list, then press Start."
)


class OwnMicrophoneError(RuntimeError):
    """The input would record AudioForge Mic, AudioForge's own output."""


def records_own_microphone(input_device: object) -> bool:
    """Linux: whether this input would record AudioForge Mic.

    That is "Follow system default" while the session manager has made AudioForge
    Mic the default microphone, as it does when no other microphone exists. The
    answer is a cached flag from the native core, so it never waits.
    """
    if not sys.platform.startswith("linux"):
        return False
    identity = coerce_device_identity(input_device)
    if identity is not None and identity.endpoint_id != FOLLOW_DEFAULT_INPUT_ENDPOINT:
        return False
    from .. import mic_eq_core

    return bool(mic_eq_core.default_source_is_virtual_microphone())


# Starts refused and running inputs stopped this session: a start can be refused on the
# engine's worker thread while the window stops one, so the count is locked.
_refusals_lock = threading.Lock()
_refusals = 0


def note_own_microphone_refusal() -> None:
    global _refusals
    with _refusals_lock:
        _refusals += 1


def own_microphone_diagnostics() -> dict[str, object]:
    """Linux: what the support export says about the guard; nothing on other platforms."""
    if not sys.platform.startswith("linux"):
        return {}
    from .. import mic_eq_core

    with _refusals_lock:
        refusals = _refusals
    return {
        "own_microphone_refusal_count": refusals,
        # Without the metadata watch the guard sees no default microphone and stays off.
        "default_source_watch_failed": bool(mic_eq_core.default_source_watch_failed()),
    }


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
    """Start the native processor on the exact selected route identities."""
    input_identity = coerce_device_identity(input_device)
    output_identity = coerce_device_identity(output_device)
    if output_identity is None:
        raise ValueError("Select a destination device before starting audio processing")
    if records_own_microphone(input_identity):
        note_own_microphone_refusal()
        raise OwnMicrophoneError(OWN_MICROPHONE_MESSAGE)
    start = getattr(processor, "start")
    return start(
        input_identity.name if input_identity is not None else None,
        output_identity.name if output_identity is not None else None,
        device_name_ordinal(input_identity),
        device_name_ordinal(output_identity),
        input_device_endpoint_id=(
            input_identity.endpoint_id or None
            if input_identity is not None
            else None
        ),
        output_device_endpoint_id=(
            output_identity.endpoint_id or None
            if output_identity is not None
            else None
        ),
    )


def default_device_index(identities: list[DeviceIdentity | None]) -> int:
    """Return the default device index, or the first item when no default exists."""
    for index, item in enumerate(identities):
        if isinstance(item, DeviceIdentity) and item.is_default:
            return index
    return 0 if identities else -1


def preferred_output_index(identities: list[DeviceIdentity | None]) -> int:
    """Prefer AudioForge's own virtual mic, then virtual routing outputs, then
    the default, then the first item."""
    for index, item in enumerate(identities):
        if isinstance(item, DeviceIdentity) and item.endpoint_id == AUDIOFORGE_MIC_ENDPOINT:
            return index
    for index, item in enumerate(identities):
        if not isinstance(item, DeviceIdentity):
            continue
        name_lower = item.name.lower()
        if any(marker in name_lower for marker in VIRTUAL_OUTPUT_MARKERS):
            return index
    return default_device_index(identities)


__all__ = [
    "AUDIOFORGE_MIC_ENDPOINT",
    "FOLLOW_DEFAULT_INPUT_ENDPOINT",
    "OWN_MICROPHONE_MESSAGE",
    "VIRTUAL_OUTPUT_MARKERS",
    "OwnMicrophoneError",
    "default_device_index",
    "device_name_ordinal",
    "find_identity_index",
    "identity_is_persistable",
    "note_own_microphone_refusal",
    "own_microphone_diagnostics",
    "preferred_output_index",
    "records_own_microphone",
    "start_processor_for_route",
]
