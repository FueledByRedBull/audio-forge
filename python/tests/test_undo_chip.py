"""What sits beside Undo on the Equalizer page: the edits since the last save and the latest one in words."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QObject, QPointF, Qt
from PySide6.QtQuick import QQuickItem
from PySide6.QtTest import QTest

from mic_eq.config import Preset
from mic_eq.ui.config_history import ConfigurationSnapshot
from mic_eq.ui.main_window import MINIMUM_WINDOW_HEIGHT, MINIMUM_WINDOW_WIDTH
from mic_eq.ui.processing_configuration import PresetOrigin, ProcessingConfiguration
from mic_eq.ui.undo_chip import UndoChip, changes_phrase, describe_change, format_age
from test_processing_configuration import Harness, harness, started  # noqa: F401  (fixtures)
from test_quick_accessibility import (  # noqa: F401  (fixtures)
    EQUALIZER,
    Text,
    _AccessibleScene,
    _iface,
    _items,
    _kind,
    _show,
    _with_proxy,
    either_window,
    icon_window,
    no_qml_warnings,
    quick_window,
    segoe_font,
    shell_window,
)


def _payload(**changes) -> dict:
    """A preset's payload with dotted paths replaced, for the formatter."""
    payload = Preset().to_dict()
    for path, value in changes.items():
        node = payload
        *parents, leaf = path.split(".")
        for part in parents:
            node = node[int(part)] if isinstance(node, list) else node[part]
        node[leaf] = value
    return payload


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


# The formatter


@pytest.mark.parametrize(
    ("changes", "words"),
    [
        ({"eq.bands.0.gain_db": -3.0}, "Band 1 gain 0.0 → -3.0 dB"),
        ({"eq.bands.4.frequency_hz": 437.5}, "Band 5 frequency 1280 → 437.5 Hz"),
        ({"eq.bands.2.q": 2.0}, "Band 3 Q 1.41 → 2.00"),
        ({"gate.enabled": False}, "Noise gate off"),
        ({"rnnoise.enabled": False}, "Noise suppression off"),
        ({"deesser.enabled": True}, "De-esser on"),
        ({"eq.enabled": False}, "Equalizer off"),
        ({"compressor.enabled": False}, "Compressor off"),
        ({"limiter.enabled": False}, "Limiter off"),
    ],
)
def test_one_changed_band_value_or_stage_switch_is_described_in_words(changes, words) -> None:
    assert describe_change(_payload(), _payload(**changes)) == words


def test_a_gain_is_shown_to_a_tenth_and_a_zero_has_no_sign() -> None:
    before = _payload(**{"eq.bands.0.gain_db": -1.0})
    assert describe_change(before, _payload(**{"eq.bands.0.gain_db": -3.0})) == "Band 1 gain -1.0 → -3.0 dB"
    assert describe_change(before, _payload(**{"eq.bands.0.gain_db": -0.04})) == "Band 1 gain -1.0 → 0.0 dB"


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"gate.threshold_db": -35.0},
        {"eq.bands.0.gain_db": -3.0, "eq.bands.1.gain_db": 2.0},
        {"eq.bands.0.gain_db": -3.0, "eq.bands.0.frequency_hz": 90.0},
        {"eq.bands.0.filter_type": "peaking"},
        {"eq.bands.0.enabled": False},
        {"gate.enabled": False, "rnnoise.enabled": False},
        {"bypass": True},
    ],
    ids=["nothing", "other-number", "two-bands", "two-fields", "filter-type", "band-switch", "two-stages", "bypass"],
)
def test_a_change_the_formatter_does_not_cover_has_no_words(changes) -> None:
    assert describe_change(_payload(), _payload(**changes)) is None


@pytest.mark.parametrize(
    ("seconds", "words"),
    [(0, "0 s ago"), (0.9, "0 s ago"), (12, "12 s ago"), (59.9, "59 s ago"), (60, "1 min ago"),
     (150, "2 min ago"), (3599, "59 min ago"), (7300, "2 h ago"), (-5, "0 s ago")],
)
def test_the_age_is_in_seconds_then_minutes_then_hours(seconds, words) -> None:
    assert format_age(seconds) == words


def test_the_chip_reads_one_change_or_many_and_nothing_at_zero() -> None:
    assert [changes_phrase(n) for n in (0, 1, 2, 11)] == [
        "", "1 change since saved", "2 changes since saved", "11 changes since saved",
    ]


# The count


def _edit(harness, threshold_db: float) -> None:  # noqa: F811
    harness.states.gate.set_settings({"threshold_db": threshold_db})
    assert harness.configuration.commit_edit()


def test_edits_raise_the_count_and_undo_and_redo_move_it(started) -> None:  # noqa: F811
    configuration = started.configuration
    assert configuration.changes_since_saved == 0
    _edit(started, -35.0)
    assert configuration.changes_since_saved == 1
    _edit(started, -30.0)
    assert configuration.changes_since_saved == 2
    configuration.undo()
    assert configuration.changes_since_saved == 1
    configuration.undo()
    assert configuration.changes_since_saved == 0
    configuration.redo()
    assert configuration.changes_since_saved == 1


def test_saving_zeroes_the_count_and_it_counts_again_from_there(started) -> None:  # noqa: F811
    configuration = started.configuration
    _edit(started, -35.0)
    _edit(started, -30.0)
    assert configuration.mark_saved(started.preset("Saved"), Path("Saved.json"))
    assert configuration.changes_since_saved == 0
    _edit(started, -25.0)
    assert configuration.changes_since_saved == 1
    configuration.undo()
    assert configuration.changes_since_saved == 0
    # Further back than the save is a change from it too.
    configuration.undo()
    assert configuration.changes_since_saved == 1
    configuration.redo()
    configuration.redo()
    assert configuration.changes_since_saved == 1


def test_editing_back_to_the_saved_sound_leaves_nothing_to_save(started) -> None:  # noqa: F811
    configuration = started.configuration
    original = started.threshold()
    _edit(started, original + 5)
    assert configuration.changes_since_saved == 1
    _edit(started, original)
    assert configuration.changes_since_saved == 0 and not configuration.identity.modified


def test_a_loaded_preset_starts_at_zero_and_undoing_it_counts_from_the_one_before(started) -> None:  # noqa: F811
    configuration = started.configuration
    _edit(started, -35.0)
    started.prompts.append(True)
    assert configuration.load(started.preset("Loud", threshold_db=-30.0), PresetOrigin(key="loud"), remember=False).loaded
    assert configuration.changes_since_saved == 0
    configuration.undo()
    # The edited Default again: one step from its own saved sound.
    assert configuration.changes_since_saved == 1
    configuration.redo()
    assert configuration.changes_since_saved == 0


def _bounded(started: Harness, limit: int) -> ProcessingConfiguration:  # noqa: F811
    """A configuration over the same stages whose history holds only ``limit`` entries."""
    configuration = ProcessingConfiguration(
        started.states, started.config, host=started.host, library=started.library,
        confirm_unsaved=started.prompts.pop, debouncer=started.debouncer, save_config=lambda: True,
        history_limit=limit,
    )
    configuration.start_history()
    return configuration


def test_once_the_saved_point_has_left_the_history_every_undo_step_counts(harness) -> None:  # noqa: F811
    configuration = _bounded(harness, 3)
    for index, threshold in enumerate((-35.0, -34.0), start=1):
        harness.states.gate.set_settings({"threshold_db": threshold})
        assert configuration.commit_edit()
        assert configuration.changes_since_saved == index
    # The third edit pushes the saved baseline out: the two entries behind this one remain.
    harness.states.gate.set_settings({"threshold_db": -33.0})
    assert configuration.commit_edit()
    assert configuration.history.size == 3 and configuration.changes_since_saved == 2
    harness.states.gate.set_settings({"threshold_db": -32.0})
    assert configuration.commit_edit()
    assert configuration.changes_since_saved == 2
    configuration.undo()
    assert configuration.changes_since_saved == 1
    # The oldest entry left is still not the saved sound, so the chip stays up.
    configuration.undo()
    assert configuration.history.cursor == 0 and configuration.changes_since_saved == 1


# The caption and its age


@pytest.fixture
def chip(started):  # noqa: F811
    clock = Clock()
    started.configuration.clock = clock
    owner = QObject()
    yield UndoChip(owner, started.configuration), clock


def _gain(started, value: float) -> None:  # noqa: F811
    started.states.eq.set_band_value(0, "gain_db", value)
    started.states.eq.flush()
    assert started.configuration.commit_edit()


def test_the_caption_names_the_latest_change_and_how_long_ago(started, chip) -> None:  # noqa: F811
    model, clock = chip
    model.refresh()
    assert (model.count, model.text, model.caption) == (0, "", "")
    _gain(started, -3.0)
    model.refresh()
    assert (model.count, model.text) == (1, "1 change since saved")
    assert model.caption == "Band 1 gain 0.0 → -3.0 dB · 0 s ago"
    clock.now += 12
    model.refresh()
    assert model.caption == "Band 1 gain 0.0 → -3.0 dB · 12 s ago"
    clock.now += 138
    model.refresh()
    assert model.caption == "Band 1 gain 0.0 → -3.0 dB · 2 min ago"


def test_the_age_is_not_a_change_so_it_announces_only_when_it_reads_differently(started, chip) -> None:  # noqa: F811
    model, clock = chip
    _gain(started, -3.0)
    seen: list[str] = []
    model.changed.connect(lambda: seen.append(model.caption))
    model.refresh()
    model.refresh()
    clock.now += 0.4
    model.refresh()
    assert len(seen) == 1
    clock.now += 1
    model.refresh()
    assert len(seen) == 2


def test_the_caption_follows_undo_and_redo_and_is_left_out_when_nothing_can_be_said(started, chip) -> None:  # noqa: F811
    model, clock = chip
    _gain(started, -3.0)
    clock.now += 10
    _gain(started, -5.0)
    model.refresh()
    assert model.text == "2 changes since saved"
    assert model.caption == "Band 1 gain -3.0 → -5.0 dB · 0 s ago"
    started.configuration.undo()
    model.refresh()
    assert model.text == "1 change since saved"
    assert model.caption == "Band 1 gain 0.0 → -3.0 dB · 10 s ago"
    started.configuration.redo()
    started.states.gate.set_settings({"threshold_db": -35.0})
    assert started.configuration.commit_edit()
    model.refresh()
    assert model.text == "3 changes since saved" and model.caption == ""


def test_a_save_empties_the_chip(started, chip) -> None:  # noqa: F811
    model, _clock = chip
    _gain(started, -3.0)
    model.refresh()
    assert model.count == 1
    started.configuration.mark_saved(started.preset("Saved"), Path("Saved.json"))
    model.refresh()
    assert (model.count, model.text, model.caption) == (0, "", "")


def test_a_snapshot_made_without_a_clock_has_no_time_and_equals_one_made_with(started) -> None:  # noqa: F811
    preset = Preset()
    plain = ConfigurationSnapshot.from_preset(preset, label="a", source="test")
    stamped = ConfigurationSnapshot.from_preset(preset, label="a", source="test", taken_at=42.0)
    assert (plain.taken_at, stamped.taken_at) == (0.0, 42.0) and plain == stamped


# The scene


def _find(window, name: str) -> QQuickItem:
    item = window.centralWidget().rootObject().findChild(QObject, name)
    assert isinstance(item, QQuickItem), name
    return item


def _chip_text(window) -> str:
    chip = _find(window, "undoChip")
    return next(item.property("text") for item in _items(chip) if _kind(item) == "QQuickText")


def _click(window, qapp, item) -> None:
    centre = item.mapToScene(QPointF(item.width() / 2, item.height() / 2)).toPoint()
    QTest.mouseClick(window.centralWidget(), Qt.MouseButton.LeftButton, pos=centre)
    qapp.processEvents()


def _scene_edit(window, qapp, band: int, gain: float) -> None:
    window.eq_state.set_band_value(band, "gain_db", gain)
    window.eq_state.flush()
    assert window.configuration.commit_edit()
    qapp.processEvents()


@pytest.fixture
def equalizer(quick_window, qapp):  # noqa: F811
    clock = Clock()
    quick_window.configuration.clock = clock
    _show(quick_window, qapp, EQUALIZER)
    return quick_window, clock


def test_the_chip_is_hidden_with_nothing_to_save_and_counts_edits_when_there_are_some(equalizer, qapp) -> None:
    window, _clock = equalizer
    assert not _find(window, "undoChip").isVisible()
    assert not _find(window, "undoCaption").isVisible()
    _scene_edit(window, qapp, 0, -3.0)
    assert _find(window, "undoChip").isVisible() and _chip_text(window) == "1 change since saved"
    _scene_edit(window, qapp, 1, 2.0)
    assert _chip_text(window) == "2 changes since saved"
    window.configuration.mark_saved(window.configuration.current, Path("Saved.json"))
    qapp.processEvents()
    assert not _find(window, "undoChip").isVisible()


def test_the_graph_stays_put_when_the_chip_and_caption_appear(equalizer, qapp) -> None:
    window, _clock = equalizer
    well = _find(window, "eqGraphWell")
    before = well.mapToScene(QPointF(0, 0))
    _scene_edit(window, qapp, 0, -3.0)
    assert _find(window, "undoCaption").isVisible()
    assert well.mapToScene(QPointF(0, 0)) == before


def test_the_caption_shows_only_for_a_change_that_has_words(equalizer, qapp) -> None:
    window, clock = equalizer
    caption = _find(window, "undoCaption")
    _scene_edit(window, qapp, 0, -3.0)
    assert caption.isVisible()
    assert caption.property("text") == "Band 1 gain 0.0 → -3.0 dB · 0 s ago"
    clock.now += 12
    window.history_chip.refresh()
    assert caption.property("text") == "Band 1 gain 0.0 → -3.0 dB · 12 s ago"
    window.gate_state.set_settings({"threshold_db": window.gate_state.get_settings()["threshold_db"] + 1})
    assert window.configuration.commit_edit()
    qapp.processEvents()
    assert _chip_text(window) == "2 changes since saved"
    assert not caption.isVisible() and caption.property("text") == ""


def test_the_undo_button_is_named_disabled_with_nothing_to_undo_and_undoes_on_a_click(
    either_window, qapp  # noqa: F811
) -> None:
    window = either_window
    window.configuration.clock = Clock()
    _show(window, qapp, EQUALIZER)
    button = _find(window, "equalizerUndo")
    original = window.eq_state.get_band(0).gain_db
    with _AccessibleScene():
        assert _iface(button).text(Text.Name) == "Undo"
        assert _iface(button).state().disabled
    assert button.property("hint") == "No processing-configuration edit to undo"
    _scene_edit(window, qapp, 0, -3.0)
    _scene_edit(window, qapp, 0, -5.0)
    assert _chip_text(window) == "2 changes since saved"
    assert button.property("hint").startswith("Undo ") and button.property("hint").endswith("(Ctrl+Z)")
    with _AccessibleScene():
        assert not _iface(button).state().disabled

    _click(window, qapp, button)
    assert window.eq_state.get_band(0).gain_db == pytest.approx(-3.0)
    assert _chip_text(window) == "1 change since saved"
    _click(window, qapp, button)
    assert window.eq_state.get_band(0).gain_db == pytest.approx(original)
    assert not _find(window, "undoChip").isVisible()
    with _AccessibleScene():
        assert _iface(button).state().disabled


# Fonts on other platforms run wider than here; 60 px under the minimum stands in for them.
@pytest.mark.parametrize("width", [MINIMUM_WINDOW_WIDTH, MINIMUM_WINDOW_WIDTH - 60], ids=["minimum", "wider-fonts"])
def test_the_undo_group_keeps_clear_of_the_header_buttons_at_the_minimum_width(
    quick_window, qapp, width  # noqa: F811
) -> None:
    window = quick_window
    window.configuration.clock = Clock()
    window.setMinimumSize(width, MINIMUM_WINDOW_HEIGHT)
    window.resize(width, MINIMUM_WINDOW_HEIGHT)
    window.shell_page.setIndex(EQUALIZER)
    qapp.processEvents()
    window.eq_state.set_band_value(9, "frequency_hz", 15000.5)
    window.eq_state.flush()
    assert window.configuration.commit_edit()
    window.configuration.clock.now += 700
    window.history_chip.refresh()
    qapp.processEvents()
    assert _find(window, "undoCaption").isVisible()
    model = window.quick_bridge.model
    group = _find(window, "undoHistory")
    others = [
        _with_proxy(window, "Btn", model["eq"]["tone"]),
        _with_proxy(window, "Btn", model["presets"]["autoEq"]),
        _with_proxy(window, "IconBtn", model["eq"]["menu"]),
    ]
    rect = group.mapRectToScene(group.boundingRect())
    for other in others:
        assert not rect.intersects(other.mapRectToScene(other.boundingRect()))
    assert rect.right() <= window.centralWidget().width()
