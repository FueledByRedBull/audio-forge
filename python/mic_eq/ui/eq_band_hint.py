"""One plain sentence about an EQ band: its type and frequency, and what it does to the voice there."""

from __future__ import annotations

from .eq_graph import FILTER_NAMES, EQGraphGeometry, format_frequency_hz
from .overview_presentation import DOT

BAND_OFF = f"Band off {DOT} switch it on to hear it."
STAGE_OFF = "The equalizer is off, so this band changes nothing."
# Where a shelf's change reaches; a bell's stays near its frequency.
_REACH = {"bell": " around here", "low_shelf": " from here down", "high_shelf": " from here up"}


def shown_gain_tenths(gain_db: float) -> int:
    """The gain as the band's gain box shows it, in tenths of a dB."""
    return int(round(gain_db * 10))


def region_word(frequency_hz: float) -> str:
    """What a change at ``frequency_hz`` is heard as, in the words of the graph's region strip."""
    for _name, _low_hz, high_hz, word in EQGraphGeometry.REGIONS:
        if frequency_hz < high_hz:
            return word
    return EQGraphGeometry.REGIONS[-1][3]


def band_hint(
    filter_type: str, frequency_hz: float, gain_db: float, *, band_on: bool = True, stage_on: bool = True
) -> str:
    if not stage_on:
        return STAGE_OFF
    if not band_on:
        return BAND_OFF
    lead = f"{FILTER_NAMES[filter_type]} at {format_frequency_hz(frequency_hz)} {DOT} "
    # A pass filter is heard an octave past its corner, where it has taken most away.
    if filter_type == "high_pass":
        word = region_word(frequency_hz / 2)
        return lead + f"it removes everything below this frequency, starting with the {word}."
    if filter_type == "low_pass":
        word = region_word(frequency_hz * 2)
        return lead + f"it removes everything above this frequency, starting with the {word}."
    word = region_word(frequency_hz)
    if filter_type == "notch":
        return lead + f"it takes out a narrow slice of the {word}, for a hum or a ringing tone."
    tenths = shown_gain_tenths(gain_db)
    if tenths > 0:
        return lead + f"boosted, so it adds {word}{_REACH[filter_type]}."
    if tenths < 0:
        return lead + f"cut, so it reduces {word}{_REACH[filter_type]}."
    return lead + f"flat, so it changes nothing yet; raise it for more {word}, lower it for less."
