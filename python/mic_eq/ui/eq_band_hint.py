"""One plain sentence about an EQ band: its type and frequency, and what it does to the voice there."""

from __future__ import annotations

from .eq_graph import FILTER_NAMES, EQGraphGeometry, format_frequency_hz
from .overview_presentation import DOT

# A gain the band's gain box shows as 0 is flat.
FLAT_GAIN_DB = 0.05
# Where a shelf's change reaches; a bell's stays near its frequency.
_REACH = {"bell": " around here", "low_shelf": " from here down", "high_shelf": " from here up"}


def region_word(frequency_hz: float) -> str:
    """What a change at ``frequency_hz`` is heard as, in the words of the graph's region strip."""
    for _name, _low_hz, high_hz, word in EQGraphGeometry.REGIONS:
        if frequency_hz < high_hz:
            return word
    return EQGraphGeometry.REGIONS[-1][3]


def band_hint(filter_type: str, frequency_hz: float, gain_db: float) -> str:
    word = region_word(frequency_hz)
    lead = f"{FILTER_NAMES[filter_type]} at {format_frequency_hz(frequency_hz)} {DOT} "
    if filter_type == "high_pass":
        return lead + f"it removes everything below this frequency, starting with the {word}."
    if filter_type == "low_pass":
        return lead + f"it removes everything above this frequency, starting with the {word}."
    if filter_type == "notch":
        return lead + f"it takes out a narrow slice of the {word}, for a hum or a ringing tone."
    if gain_db >= FLAT_GAIN_DB:
        return lead + f"boosted, so it adds {word}{_REACH[filter_type]}."
    if gain_db <= -FLAT_GAIN_DB:
        return lead + f"cut, so it reduces {word}{_REACH[filter_type]}."
    return lead + f"flat, so it changes nothing yet; raise it for more {word}, lower it for less."
