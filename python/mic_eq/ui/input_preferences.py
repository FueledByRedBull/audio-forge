"""Input channel and cleanup preferences: which one a route or device has, and how one is stored.

Nothing here touches Qt. The window finds the current route and device keys and
applies the result to its controls and the engine; this module reads and writes
the preferences in the configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

from ..config import InputDevicePreference
from .shell_specs import INPUT_CHANNEL_MODE_OPTIONS, INPUT_CLEANUP_MODE_OPTIONS


def is_valid_channel_mode(mode: object) -> bool:
    return isinstance(mode, str) and any(
        mode == option_mode for _label, option_mode in INPUT_CHANNEL_MODE_OPTIONS
    )


def is_valid_cleanup_mode(mode: object) -> bool:
    return isinstance(mode, str) and any(
        mode == option_mode for _label, option_mode in INPUT_CLEANUP_MODE_OPTIONS
    )


class RouteSave(Enum):
    # No stable route, or a mode that is not an option: nothing was written.
    REFUSED = "refused"
    # The override was written but the configuration could not be saved; it is rolled back.
    NOT_SAVED = "not_saved"
    SAVED = "saved"


class InputPreferences:
    def __init__(self, config: Any) -> None:
        self._config = config

    def lookup(self, route_key: str | None, device_key: str | None) -> InputDevicePreference:
        """The route's preference, else the input device's, else the defaults."""
        route_preferences = getattr(self._config, "route_input_preferences", {})
        if route_key is not None and route_key in route_preferences:
            return route_preferences[route_key]

        device_preferences = getattr(self._config, "input_device_preferences", {})
        if device_key is not None and device_key in device_preferences:
            return device_preferences[device_key]

        # An unmatched route or device is unbound; global settings never
        # transfer to a different endpoint.
        return InputDevicePreference()

    def store(
        self,
        route_key: str | None,
        device_key: str | None,
        *,
        channel_mode: str,
        cleanup_mode: str,
    ) -> None:
        """Record a choice on the route that already has an override, else on the device."""
        preference = InputDevicePreference(
            channel_mode=channel_mode,
            cleanup_mode=cleanup_mode,
            provenance="explicit_user",
        )
        route_preferences = getattr(self._config, "route_input_preferences", None)
        device_preferences = getattr(self._config, "input_device_preferences", None)
        if route_preferences is not None and device_preferences is not None:
            if route_key in route_preferences:
                route_preferences[route_key] = preference
            elif device_key is not None:
                device_preferences[device_key] = preference
        self._config.input_channel_mode = channel_mode
        self._config.input_cleanup_mode = cleanup_mode

    def set_for_route(
        self,
        route_key: str | None,
        *,
        channel_mode: str,
        cleanup_mode: str,
        save: Callable[[], bool],
    ) -> RouteSave:
        """Persist an exact-route override, restoring the previous one if ``save`` fails."""
        route_preferences = getattr(self._config, "route_input_preferences", None)
        if route_key is None or route_preferences is None:
            return RouteSave.REFUSED
        if not (is_valid_channel_mode(channel_mode) and is_valid_cleanup_mode(cleanup_mode)):
            return RouteSave.REFUSED
        previous = route_preferences.get(route_key)
        route_preferences[route_key] = InputDevicePreference(
            channel_mode=channel_mode,
            cleanup_mode=cleanup_mode,
            provenance="explicit_user",
        )
        if save():
            return RouteSave.SAVED
        if previous is None:
            del route_preferences[route_key]
        else:
            route_preferences[route_key] = previous
        return RouteSave.NOT_SAVED
