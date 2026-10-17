"""The read-only EQ comparison on the review pages."""

from __future__ import annotations

import re

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest

from mic_eq.config import EQSettings, build_eq_candidate_settings
from mic_eq.ui.eq_compare import EQComparisonWidget, eq_response_db
from mic_eq.ui.theme import PALETTE, qcolor

FREQUENCIES = [60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0, 3200.0, 6300.0, 10000.0, 16000.0]


def _proposal(base: EQSettings, gains: list[float], *, enabled: bool = True) -> EQSettings:
    return build_eq_candidate_settings(
        base, FREQUENCIES, gains, [1.41] * 10, layer="correction", enabled=enabled
    )


def _widget(now: EQSettings, proposed: EQSettings) -> EQComparisonWidget:
    widget = EQComparisonWidget()
    widget.resize(520, 180)
    widget.set_curves(now, proposed, 48_000.0)
    return widget


def test_the_largest_change_names_the_size_and_the_place(qapp) -> None:
    flat = EQSettings(enabled=True)
    gains = [0.0] * 10
    gains[6] = 3.0  # a bell at 3.2 kHz
    widget = _widget(flat, _proposal(flat, gains))

    change, frequency = widget.largest_change()

    assert change == pytest.approx(3.0, abs=0.2)
    assert frequency == pytest.approx(3200.0, rel=0.05)
    description = widget.description()
    assert f"{change:+.1f} dB" in description and "3.2 kHz" in description
    assert re.search(r"solid[^.]*proposed", description, re.I) and re.search(r"dashed[^.]*(now|current)", description, re.I)
    assert widget.accessibleDescription() == widget.description()


def test_a_cut_is_signed_and_the_same_eq_says_so(qapp) -> None:
    flat = EQSettings(enabled=True)
    gains = [0.0] * 10
    gains[2] = -4.0
    cut = _widget(flat, _proposal(flat, gains))
    assert cut.largest_change()[0] < -3.5 and "-4.0 dB at 250 Hz" in cut.description()

    same = _widget(flat, flat)
    assert re.search(r"\bsame\b", same.description()) and "not the same" not in same.description()
    assert " dB" not in same.description()


def test_an_eq_that_is_switched_off_is_flat(qapp) -> None:
    off = EQSettings(enabled=False)
    gains = [3.0] * 10

    assert np.allclose(eq_response_db(_proposal(off, gains, enabled=False), np.array([100.0, 1000.0]), 48_000.0), 0.0)
    widget = _widget(_proposal(off, gains, enabled=False), _proposal(off, gains, enabled=True))
    assert widget.largest_change()[0] > 2.5  # switching it on is a change


def test_the_two_curves_differ_by_shape_not_only_by_colour(qapp) -> None:
    flat = EQSettings(enabled=True)
    widget = _widget(flat, _proposal(flat, [1.0] * 10))

    assert [(label, dashed) for label, _response, dashed in widget.lines()] == [
        ("Now", True),
        ("Proposed", False),
    ]


def test_the_widget_takes_no_input(qapp) -> None:
    widget = _widget(EQSettings(enabled=True), _proposal(EQSettings(enabled=True), [2.0] * 10))
    widget.show()
    before = widget.largest_change()

    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=QPoint(100, 90))
    QTest.keyClick(widget, Qt.Key.Key_Up)

    assert widget.largest_change() == before
    assert widget.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert widget.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    widget.hide()


def test_it_paints_the_proposed_line_solid_and_the_line_now_dashed(qapp, monkeypatch) -> None:
    flat = EQSettings(enabled=True)
    proposed = _proposal(flat, [3.0] * 10)
    # Two level lines: +3 dB for the proposal and -3 dB for the EQ now.
    monkeypatch.setattr(
        "mic_eq.ui.eq_compare.eq_response_db",
        lambda eq, frequencies, _rate: np.full(frequencies.size, 3.0 if eq is proposed else -3.0),
    )
    widget = _widget(flat, proposed)
    image: QImage = widget.grab().toImage()

    # The plot is 468 by 150 pixels from (40, 8), and the range is +-6 dB: +3 dB is a quarter
    # of the plot height above its middle (y 83), so 45.5, and -3 dB is 120.5.
    solid_row, dashed_row = 45, 120
    columns = range(100, 400)
    accent, muted = qcolor(PALETTE.accent).rgb(), qcolor(PALETTE.text_muted).rgb()

    solid = [image.pixelColor(x, solid_row).rgb() == accent for x in columns]
    dashed = [image.pixelColor(x, dashed_row).rgb() == muted for x in columns]
    assert all(solid)
    assert 0.4 < sum(dashed) / len(dashed) < 0.85
    # Neither colour is anywhere else on those columns' lines.
    assert not any(image.pixelColor(x, solid_row).rgb() == muted for x in columns)
    assert not any(image.pixelColor(x, dashed_row).rgb() == accent for x in columns)


def test_it_says_what_now_is_when_the_eq_is_not_in_use(qapp) -> None:
    flat = EQSettings(enabled=True)
    widget = EQComparisonWidget()
    proposal = _proposal(flat, [3.0] * 10)
    widget.set_curves(flat, proposal, 48_000.0, unused_mode="raw")

    assert [label for label, _response, _dashed in widget.lines()] == [
        "Now (not used in Raw monitor)",
        "Proposed",
    ]
    assert "the EQ now, which Raw monitor does not use" in widget.description()
    widget.set_curves(flat, proposal, 48_000.0)
    assert "Raw monitor" not in widget.description()
