"""Small ownership helpers for microphone captures used by calibration dialogs."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import time
from typing import Any

from ..config import DeviceIdentity, coerce_device_identity
from .device_selection import start_processor_for_route

logger = logging.getLogger(__name__)

CAPTURE_NO_PROGRESS_TIMEOUT_S = 3.0
CAPTURE_START_GRACE_S = 2.0
CAPTURE_COMPLETION_GRACE_S = 5.0


@dataclass
class CaptureProgressDeadline:
    """Bound a capture by elapsed time and by time since input last advanced."""

    duration_s: float
    no_progress_timeout_s: float = CAPTURE_NO_PROGRESS_TIMEOUT_S
    start_grace_s: float = CAPTURE_START_GRACE_S
    completion_grace_s: float = CAPTURE_COMPLETION_GRACE_S
    started_at: float | None = None
    last_progress: float = 0.0
    last_progress_at: float | None = None

    def __post_init__(self) -> None:
        values = (
            self.duration_s,
            self.no_progress_timeout_s,
            self.start_grace_s,
            self.completion_grace_s,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Capture deadlines must be finite and non-negative")
        if self.duration_s <= 0.0 or self.no_progress_timeout_s <= 0.0:
            raise ValueError("Capture duration and no-progress timeout must be positive")
        now = time.monotonic() if self.started_at is None else self.started_at
        self.started_at = now
        self.last_progress_at = now

    def failure_reason(self, progress: float, *, now: float | None = None) -> str | None:
        current_time = time.monotonic() if now is None else now
        normalized = float(progress)
        if math.isfinite(normalized):
            normalized = min(1.0, max(0.0, normalized))
            if normalized > self.last_progress + 1e-6:
                self.last_progress = normalized
                self.last_progress_at = current_time

        started_at = self.started_at if self.started_at is not None else 0.0
        last_progress_at = (
            self.last_progress_at
            if self.last_progress_at is not None
            else started_at
        )
        elapsed = current_time - started_at
        if elapsed >= self.duration_s + self.completion_grace_s:
            limit = f"{self.duration_s + self.completion_grace_s:g}"
            return f"Audio capture exceeded its {limit}-second time limit."
        if (
            elapsed >= self.start_grace_s
            and current_time - last_progress_at >= self.no_progress_timeout_s
        ):
            return "No audio capture progress was received."
        return None


class CaptureSession:
    """Own one raw capture's mute and recovery-suppression state."""

    def __init__(self, owner: Any, mute_reason: str) -> None:
        self.owner = owner
        self.mute_reason = mute_reason
        self.deadline: CaptureProgressDeadline | None = None
        self._recording_started = False
        self._mute_owned = False
        self._recovery_owned = False
        self._previous_recovery_suppressed = False

    @property
    def active(self) -> bool:
        return self._recording_started

    def start(self, duration_s: float, **recording_options: Any) -> None:
        if self._recording_started:
            raise RuntimeError("A capture is already active in this session")
        processor = self.owner.processor
        self.deadline = CaptureProgressDeadline(duration_s)
        try:
            self._mute_owned = True
            set_temporary_mute(self.owner, self.mute_reason, True)
            get_suppressed = getattr(processor, "is_recovery_suppressed", None)
            if callable(get_suppressed):
                self._previous_recovery_suppressed = bool(get_suppressed())
            self._recovery_owned = True
            processor.set_recovery_suppressed(True)
            self._recording_started = True
            processor.start_raw_recording(duration_s, **recording_options)
        except Exception:
            self.cleanup()
            raise

    def failure_reason(self, progress: float) -> str | None:
        if not self._recording_started or self.deadline is None:
            return None
        return self.deadline.failure_reason(progress)

    def stop_recording(self) -> Any:
        if not self._recording_started:
            return None
        try:
            audio = self.owner.processor.stop_raw_recording()
        except Exception:
            self.cleanup()
            raise
        self._recording_started = False
        self._release_ownership()
        return audio

    def cleanup(self) -> None:
        if self._recording_started:
            try:
                self.owner.processor.stop_raw_recording()
            except RuntimeError as exc:
                if "No recording in progress" not in str(exc):
                    logger.warning("Failed to stop raw capture during cleanup: %s", exc)
            except Exception as exc:
                logger.warning("Failed to stop raw capture during cleanup: %s", exc)
            self._recording_started = False
        self._release_ownership()

    def _release_ownership(self) -> None:
        if self._recovery_owned:
            try:
                self.owner.processor.set_recovery_suppressed(
                    self._previous_recovery_suppressed
                )
            except Exception as exc:
                logger.warning("Failed to restore audio recovery state: %s", exc)
            else:
                self._recovery_owned = False
        if self._mute_owned:
            try:
                set_temporary_mute(self.owner, self.mute_reason, False)
            except Exception as exc:
                logger.warning("Failed to release temporary capture mute: %s", exc)
            else:
                self._mute_owned = False


def find_processor_owner(widget: object) -> Any | None:
    parent: Any = widget
    while parent and not hasattr(parent, "processor"):
        parent = parent.parent()
    return parent


def find_eq_state_owner(widget: object) -> Any | None:
    parent: Any = widget
    while parent and not hasattr(parent, "eq_state"):
        parent = parent.parent()
    return parent


def processor_sample_rate(owner: Any) -> int:
    if owner is None or not hasattr(owner, "processor"):
        raise RuntimeError("Could not find audio processor")
    sample_rate = int(owner.processor.sample_rate())
    if sample_rate <= 0:
        raise RuntimeError("Processor sample rate is unavailable")
    return sample_rate


def processor_output_sample_rate(owner: Any) -> int:
    if owner is None or not hasattr(owner, "processor"):
        raise RuntimeError("Could not find audio processor.")
    sample_rate = int(owner.processor.output_sample_rate())
    if sample_rate <= 0:
        raise RuntimeError("Output sample rate is unavailable.")
    return sample_rate


def selected_device_identities(
    owner: Any,
) -> tuple[DeviceIdentity | None, DeviceIdentity | None]:
    if owner is None:
        return None, None
    input_device = owner.input_choice.currentData() if hasattr(owner, "input_choice") else None
    output_device = owner.output_choice.currentData() if hasattr(owner, "output_choice") else None
    return coerce_device_identity(input_device), coerce_device_identity(output_device)


def active_device_identities(
    processor: Any,
) -> tuple[DeviceIdentity | None, DeviceIdentity | None]:
    """Read the exact route used by the running native stream."""

    def read_identity(direction: str) -> DeviceIdentity | None:
        get_name = getattr(processor, f"get_active_{direction}_device", None)
        get_endpoint_id = getattr(
            processor, f"get_active_{direction}_device_endpoint_id", None
        )
        get_name_ordinal = getattr(
            processor, f"get_active_{direction}_device_name_ordinal", None
        )
        return coerce_device_identity(
            {
                "name": get_name() if callable(get_name) else None,
                "endpoint_id": get_endpoint_id() if callable(get_endpoint_id) else None,
                "name_ordinal": get_name_ordinal() if callable(get_name_ordinal) else None,
                "direction": direction,
            }
        )

    return read_identity("input"), read_identity("output")


def route_identities_match(
    selected: tuple[DeviceIdentity | None, DeviceIdentity | None],
    active: tuple[DeviceIdentity | None, DeviceIdentity | None],
) -> bool:
    return _route_identity_matches(selected[0], active[0]) and _route_identity_matches(
        selected[1], active[1]
    )


def _route_identity_matches(
    selected: DeviceIdentity | None,
    active: DeviceIdentity | None,
) -> bool:
    if selected is None or active is None:
        return selected is None and active is None
    if selected.endpoint_id:
        return bool(active.endpoint_id) and (
            selected.endpoint_id.casefold() == active.endpoint_id.casefold()
        )
    return (
        " ".join(selected.name.casefold().split())
        == " ".join(active.name.casefold().split())
        and (selected.name_ordinal or 0) == (active.name_ordinal or 0)
    )


def device_name(device: object) -> str | None:
    identity = coerce_device_identity(device)
    if identity is not None:
        return identity.name
    return device if isinstance(device, str) and device else None


def device_label(device: str | None, default_label: str) -> str:
    return device if device else default_label


def start_selected_route(owner: Any) -> object:
    input_device = owner.input_choice.currentData() if hasattr(owner, "input_choice") else None
    output_device = owner.output_choice.currentData() if hasattr(owner, "output_choice") else None
    return start_processor_for_route(owner.processor, input_device, output_device)


def restart_processor_for_route(
    processor: Any,
    selected: tuple[DeviceIdentity | None, DeviceIdentity | None],
    previous: tuple[DeviceIdentity | None, DeviceIdentity | None],
) -> object:
    if any(device is None for device in previous):
        raise RuntimeError(
            "Cannot identify the current route; stop processing before switching devices"
        )
    processor.stop()
    try:
        return start_processor_for_route(processor, selected[0], selected[1])
    except Exception as switch_error:
        try:
            start_processor_for_route(processor, previous[0], previous[1])
        except Exception as restore_error:
            raise RuntimeError(
                f"{switch_error}; failed to restore previous route: {restore_error}"
            ) from switch_error
        raise RuntimeError(f"{switch_error}; previous route restored") from switch_error


def sync_owner_processing_controls(owner: Any) -> None:
    sync = getattr(owner, "_sync_processing_controls", None)
    if callable(sync):
        sync()


def set_temporary_mute(owner: Any, reason: str, enabled: bool) -> None:
    """Set one dialog-owned mute without changing the user's mute preference."""
    setter = getattr(owner, "set_temporary_output_mute", None)
    if callable(setter):
        setter(bool(enabled), reason)
        return
    processor = getattr(owner, "processor", None)
    setter = getattr(processor, "set_output_mute", None)
    if callable(setter):
        setter(bool(enabled or getattr(owner, "user_muted", False)))


def owner_calibration_context_key(owner: Any) -> str | None:
    getter = getattr(owner, "_calibration_context_key", None)
    if not callable(getter):
        return None
    value = getter()
    return value if isinstance(value, str) and value else None


@dataclass(frozen=True)
class CaptureRouteContext:
    route_key: str | None
    format_context: tuple[int, int, int, int] | None
    context_key: str | None


def known_capture_format_context(
    value: object,
) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in value
    ):
        return None
    return tuple(value)  # type: ignore[return-value]


def capture_route_context(owner: Any) -> CaptureRouteContext:
    route_getter = getattr(owner, "_current_device_route_key", None)
    format_getter = getattr(owner, "_current_capture_format_context", None)
    try:
        context_key = owner_calibration_context_key(owner)
    except Exception:
        context_key = None
    if not callable(route_getter) or not callable(format_getter):
        return CaptureRouteContext(None, None, context_key)
    try:
        route_key = route_getter()
        format_context = known_capture_format_context(format_getter())
    except Exception:
        return CaptureRouteContext(None, None, context_key)
    if not isinstance(route_key, str) or not route_key:
        route_key = None
    return CaptureRouteContext(route_key, format_context, context_key)
