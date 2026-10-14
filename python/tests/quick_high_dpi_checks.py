"""The Qt Quick scene checks that matter at a scale factor above 1.

Not collected by the normal run: ``test_quick_high_dpi.py`` runs this file in a
child pytest with ``QT_SCALE_FACTOR`` set, because Qt reads the factor once, at
start-up. Everything imported here is a test or fixture that already holds at
scale 1; it is repeated at the scale the child was started with.
"""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

import os

import pytest

from test_quick_layout import (  # noqa: F401  (tests and fixtures repeated at this scale)
    minimum_window,
    segoe_font,
    test_a_long_preset_name_is_elided_and_whole_in_the_tooltip,
    test_an_elided_status_caption_has_its_full_text_as_its_tip,
    test_nothing_in_the_content_column_runs_under_the_meters,
    test_the_mic_page_rows_fit_the_page,
    test_the_scene_is_built_at_its_minimum_width,
    test_the_status_strip_texts_do_not_overlap,
)
from test_quick_shell import (  # noqa: F401  (tests and fixtures repeated at this scale)
    mic_page,
    quick_window,
    test_a_band_drag_is_not_taken_over_by_the_page,
    test_a_drag_moves_the_band_by_the_pointers_travel,
    test_live_data_changes_repaint_existing_qml_items,
    test_scene_loads_cleanly,
)

EXPECTED = float(os.environ["AUDIOFORGE_EXPECT_SCALE"])


def test_the_window_really_runs_at_the_requested_scale(quick_window, qapp) -> None:  # noqa: F811
    view = quick_window.centralWidget()
    assert view.devicePixelRatio() == pytest.approx(EXPECTED)
    quick_window.resize(1200, 800)
    qapp.processEvents()
    frame = view.grabFramebuffer()
    # Sizes in the scene stay logical; only the pixels behind them multiply.
    assert frame.width() == pytest.approx(view.width() * EXPECTED, abs=1)
    assert frame.height() == pytest.approx(view.height() * EXPECTED, abs=1)
