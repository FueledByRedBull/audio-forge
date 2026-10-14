from typing import Any
from unittest.mock import Mock

from mic_eq.config import EQ_SLOPES_DB_PER_OCTAVE
from mic_eq.ui.eq_controls import EQControl
from mic_eq.ui.eq_state import EQ_FILTER_OPTIONS, EQState

TYPES = [choice for _name, choice in EQ_FILTER_OPTIONS]
SLOPES = sorted(EQ_SLOPES_DB_PER_OCTAVE)


def _bands() -> list[tuple[float, float, float]]:
    return [
        (80.0 * (1.6**index), 1.0 if index == 4 else 0.0, 1.41)
        for index in range(10)
    ]


def _diagnostics() -> dict[str, float | str]:
    return {
        "analysis_confidence": 0.82,
        "eq_confidence": 0.76,
        "capture_confidence": 0.88,
        "validation_confidence": 0.79,
        "validation_before_error_db": 4.2,
        "validation_after_error_db": 2.1,
        "validation_gain_scale": 0.85,
        "target_profile": "broadcast:adaptive",
        "recommendation_status": "apply",
    }


def test_manual_eq_edits_invalidate_diagnostics_and_fresh_results_restore_them(qapp):
    state = EQState(Mock())
    shown: Any = EQControl(state, "diagnostics")
    diagnostics = _diagnostics()

    def band(key: str) -> Any:
        return EQControl(state, key, 4)

    def edit(control: EQControl, value: float) -> None:
        control.setValue(value)
        control.released()

    def apply_candidate() -> None:
        state.apply_auto_eq_results(_bands(), diagnostics=diagnostics)
        assert state.auto_eq_diagnostics == diagnostics

    def assert_cleared() -> None:
        assert state.auto_eq_diagnostics is None
        assert shown.text == "Auto-EQ: no calibration diagnostics"

    def edit_gain() -> None:
        edit(band("gain"), band("gain").value + 1)

    def edit_slope() -> None:
        band("type").setIndex(TYPES.index("high_pass"))
        state.set_auto_eq_diagnostics(diagnostics)
        band("slope").setIndex(SLOPES.index(24))

    edits = (
        edit_gain,
        lambda: edit(band("q"), 1.5),
        lambda: band("type").setIndex(TYPES.index("notch")),
        edit_slope,
        lambda: band("enabled").click(),
        lambda: edit(band("frequency"), 1100.0),
        lambda: (state.edit_curve_band(4, 2100.0, 2.0), state.flush()),
        lambda: EQControl(state, "enabled").click(),
    )

    for change in edits:
        apply_candidate()
        change()
        qapp.processEvents()
        assert_cleared()

    apply_candidate()
    assert shown.text.startswith("Auto-EQ: ")
