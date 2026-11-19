from __future__ import annotations

from unittest.mock import Mock

import pytest

from mic_eq.config import AppConfig, InputDevicePreference
from mic_eq.ui.input_preferences import (
    InputPreferences,
    RouteSave,
    is_valid_channel_mode,
    is_valid_cleanup_mode,
)

LEFT_GENTLE = InputDevicePreference("left", "gentle")


def test_the_validity_checks_accept_only_listed_modes() -> None:
    assert is_valid_channel_mode("left")
    assert not is_valid_channel_mode("sideways")
    assert not is_valid_channel_mode(None)
    assert is_valid_cleanup_mode("gentle")
    assert not is_valid_cleanup_mode("loud")
    assert not is_valid_cleanup_mode(3)


def test_lookup_prefers_the_route_then_the_device_then_the_defaults() -> None:
    config = AppConfig(
        route_input_preferences={"route": InputDevicePreference("right", "strong")},
        input_device_preferences={"device": LEFT_GENTLE},
    )
    preferences = InputPreferences(config)

    assert preferences.lookup("route", "device") == InputDevicePreference("right", "strong")
    assert preferences.lookup("other-route", "device") == LEFT_GENTLE
    assert preferences.lookup(None, "device") == LEFT_GENTLE
    assert preferences.lookup("other-route", "other-device") == InputDevicePreference()
    assert preferences.lookup(None, None) == InputDevicePreference()


def test_lookup_never_falls_back_to_the_global_modes() -> None:
    config = AppConfig(input_channel_mode="left", input_cleanup_mode="gentle")

    assert InputPreferences(config).lookup("route", "device") == InputDevicePreference()


def test_store_updates_the_route_that_has_an_override() -> None:
    config = AppConfig(route_input_preferences={"route": InputDevicePreference("right", "off")})

    InputPreferences(config).store("route", "device", channel_mode="left", cleanup_mode="gentle")

    assert config.route_input_preferences["route"] == InputDevicePreference(
        "left", "gentle", "explicit_user"
    )
    assert config.input_device_preferences == {}
    assert (config.input_channel_mode, config.input_cleanup_mode) == ("left", "gentle")


def test_store_falls_back_to_the_device_without_a_route_override() -> None:
    config = AppConfig()

    InputPreferences(config).store("route", "device", channel_mode="left", cleanup_mode="gentle")

    assert config.input_device_preferences["device"] == InputDevicePreference(
        "left", "gentle", "explicit_user"
    )
    assert config.route_input_preferences == {}
    assert (config.input_channel_mode, config.input_cleanup_mode) == ("left", "gentle")


def test_store_without_a_stable_device_only_sets_the_global_modes() -> None:
    config = AppConfig()

    InputPreferences(config).store(None, None, channel_mode="left", cleanup_mode="gentle")

    assert config.route_input_preferences == {}
    assert config.input_device_preferences == {}
    assert (config.input_channel_mode, config.input_cleanup_mode) == ("left", "gentle")


def test_set_for_route_saves_the_override() -> None:
    config = AppConfig()
    save = Mock(return_value=True)

    result = InputPreferences(config).set_for_route(
        "route", channel_mode="left", cleanup_mode="gentle", save=save
    )

    assert result is RouteSave.SAVED
    assert config.route_input_preferences["route"] == InputDevicePreference(
        "left", "gentle", "explicit_user"
    )
    save.assert_called_once_with()


@pytest.mark.parametrize("previous", [None, InputDevicePreference("right", "off")])
def test_set_for_route_rolls_back_when_the_save_fails(previous) -> None:
    config = AppConfig()
    if previous is not None:
        config.route_input_preferences["route"] = previous

    result = InputPreferences(config).set_for_route(
        "route", channel_mode="left", cleanup_mode="gentle", save=lambda: False
    )

    assert result is RouteSave.NOT_SAVED
    assert config.route_input_preferences.get("route") == previous


@pytest.mark.parametrize(
    ("route_key", "channel_mode", "cleanup_mode"),
    [(None, "left", "gentle"), ("route", "sideways", "gentle"), ("route", "left", "loud")],
    ids=["no-route", "bad-channel-mode", "bad-cleanup-mode"],
)
def test_set_for_route_refuses_without_writing_or_saving(route_key, channel_mode, cleanup_mode) -> None:
    config = AppConfig()
    save = Mock(return_value=True)

    result = InputPreferences(config).set_for_route(
        route_key, channel_mode=channel_mode, cleanup_mode=cleanup_mode, save=save
    )

    assert result is RouteSave.REFUSED
    assert config.route_input_preferences == {}
    save.assert_not_called()
