"""The one-line state under each signal-chain row in the sidebar.

Pure: the window gathers :class:`ChainInputs` and each row shows its
:class:`ChainLine`. While audio is stopped a meter row reads "Idle" and its
activity bar is empty, so nothing on screen looks live when it is not.
"""

from __future__ import annotations

from dataclasses import dataclass

DOT = "·"
# A reduction this deep fills the row's activity bar.
FULL_ACTIVITY_DB = 12.0


@dataclass(frozen=True, slots=True)
class ChainInputs:
    running: bool = False
    cleanup_mode: str = "Off"
    gate_enabled: bool = True
    gate_threshold_db: float = -40.0
    suppression_enabled: bool = True
    backend: str = "RNNoise"
    strength: float = 1.0
    deesser_enabled: bool = True
    deesser_reduction_db: float | None = None
    eq_enabled: bool = True
    eq_bands: int = 10
    compressor_enabled: bool = True
    compressor_reduction_db: float | None = None
    limiter_enabled: bool = True
    limiter_reduction_db: float | None = None


@dataclass(frozen=True, slots=True)
class ChainLine:
    line: str
    # 0..1, only for a row with a reduction reading.
    activity: float = 0.0
    metered: bool = False


def _number(value: float) -> str:
    text = f"{value:.0f}" if value == round(value) else f"{value:.1f}"
    return "0" if text == "-0" else text


def _reduction(enabled: bool, running: bool, reduction_db: float | None) -> ChainLine:
    if not enabled:
        return ChainLine("Off", metered=True)
    if not running or reduction_db is None:
        return ChainLine("Idle", metered=True)
    activity = max(0.0, min(1.0, reduction_db / FULL_ACTIVITY_DB))
    return ChainLine(f"Reduction {reduction_db:.1f} dB", activity, metered=True)


def describe_chain(inputs: ChainInputs) -> dict[str, ChainLine]:
    """Each row's line, keyed as the sidebar's chain rows are."""
    cleanup = "HP 80 Hz" if inputs.cleanup_mode == "Off" else f"HP 80 Hz {DOT} {inputs.cleanup_mode}"
    bands = f"{inputs.eq_bands} band{'s' if inputs.eq_bands != 1 else ''}"
    return {
        "cleanup": ChainLine(cleanup),
        "gate": ChainLine(f"{_number(inputs.gate_threshold_db)} dB" if inputs.gate_enabled else "Off"),
        "suppression": ChainLine(
            f"{inputs.backend} {DOT} {round(inputs.strength * 100)}%" if inputs.suppression_enabled else "Off"
        ),
        "deesser": _reduction(inputs.deesser_enabled, inputs.running, inputs.deesser_reduction_db),
        "eq": ChainLine(bands if inputs.eq_enabled else "Off"),
        "compressor": _reduction(inputs.compressor_enabled, inputs.running, inputs.compressor_reduction_db),
        "limiter": _reduction(inputs.limiter_enabled, inputs.running, inputs.limiter_reduction_db),
    }
