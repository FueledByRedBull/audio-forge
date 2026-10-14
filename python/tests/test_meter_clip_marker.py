"""Peak hold and the self-clearing clip marker on the level meters."""

import math

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QPainter, QRegion
from PySide6.QtWidgets import QWidget

from mic_eq.ui.level_meter import CLIP_FLAT_TOP_SAMPLES, CLIP_HOLD_SECONDS, LevelMeter, MeterWidget


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _meter(clock: _Clock, label: str = "IN") -> LevelMeter:
    meter = LevelMeter(label, name="Input level meter", clock=clock)
    meter.set_levels(-30.0, -40.0)
    meter.note_peak(None, None, full_scale_count=0)  # the first count is only a baseline
    return meter


def _clip(meter: LevelMeter, samples: int = CLIP_FLAT_TOP_SAMPLES) -> None:
    """Read a block in which the engine counted ``samples`` more full-scale samples."""
    count = (meter._last_full_scale_count or 0) + samples
    meter.note_peak(1.0, 1.0, full_scale_count=count)


def _db(linear: float) -> float:
    return 20.0 * math.log10(linear)


def test_hold_line_follows_a_peak_the_block_meter_never_showed(qapp):
    meter = _meter(_Clock())
    meter.note_peak(0.5, 0.5)
    assert abs(meter.peak_hold_db - _db(0.5)) < 1e-6
    assert not meter.clip_marker


def test_marker_lights_for_a_flat_top_and_clears_by_itself(qapp):
    clock = _Clock()
    meter = _meter(clock)
    _clip(meter)
    assert meter.clip_marker and meter.decay_timer.isActive()

    clock.now += CLIP_HOLD_SECONDS - 0.1
    meter._decay_peak_hold()
    assert meter.clip_marker

    clock.now += 0.2
    meter._decay_peak_hold()
    assert not meter.clip_marker


def test_a_new_clip_restarts_the_hold(qapp):
    clock = _Clock()
    meter = _meter(clock)
    _clip(meter)
    clock.now += 2.0
    _clip(meter)
    clock.now += 2.0
    meter._decay_peak_hold()
    assert meter.clip_marker
    clock.now += 1.1
    meter._decay_peak_hold()
    assert not meter.clip_marker


def test_a_peak_alone_never_lights_the_marker_because_it_cannot_say_how_long_it_lasted(qapp):
    meter = _meter(_Clock())
    for peak in (0.998, 0.999, 1.0, 1.5):
        meter.note_peak(peak, peak)
    assert not meter.clip_marker
    assert meter.peak_hold_db == meter.DB_MAX


def test_a_lone_full_scale_sample_is_a_high_peak_not_a_clip(qapp):
    meter = _meter(_Clock())
    _clip(meter, CLIP_FLAT_TOP_SAMPLES - 1)
    assert not meter.clip_marker and not meter.clipped_within(10.0)
    assert meter.property("description") == ""
    recent = meter.recent_peak_db(10.0)
    assert recent is not None and abs(recent) < 1e-6  # Health still sees a peak at 0 dB.


def test_samples_counted_across_blocks_within_the_hold_window_add_up(qapp):
    clock = _Clock()
    meter = _meter(clock)
    _clip(meter, 3)
    clock.now += 1.0
    assert not meter.clip_marker
    _clip(meter, CLIP_FLAT_TOP_SAMPLES - 3)
    assert meter.clip_marker


def test_samples_spread_beyond_the_hold_window_do_not_add_up(qapp):
    clock = _Clock()
    meter = _meter(clock)
    _clip(meter, 3)
    clock.now += CLIP_HOLD_SECONDS + 0.1
    _clip(meter, 3)
    assert not meter.clip_marker


def test_the_first_count_is_a_baseline_and_an_unchanged_count_adds_nothing(qapp):
    meter = LevelMeter("IN", clock=_Clock())
    meter.set_levels(-30.0, -40.0)
    meter.note_peak(0.2, 0.2, full_scale_count=500)
    assert not meter.clip_marker
    meter.note_peak(0.2, 0.2, full_scale_count=500)
    assert not meter.clip_marker
    meter.note_peak(0.2, 0.2, full_scale_count=500 + CLIP_FLAT_TOP_SAMPLES)
    assert meter.clip_marker


def test_a_count_that_restarts_at_zero_after_a_new_stream_still_counts(qapp):
    meter = _meter(_Clock())
    meter.note_peak(0.2, 0.2, full_scale_count=50)
    meter.note_peak(0.2, 0.2, full_scale_count=CLIP_FLAT_TOP_SAMPLES)
    assert meter.clip_marker


def test_out_bar_hold_uses_the_pre_limiter_peak_but_the_marker_uses_what_leaves_the_app(qapp):
    meter = _meter(_Clock(), "OUT")
    meter.note_peak(1.6, 0.9)
    assert meter.peak_hold_db == meter.DB_MAX
    assert not meter.clip_marker
    meter.note_peak(0.5, 1.0, full_scale_count=CLIP_FLAT_TOP_SAMPLES)
    assert meter.clip_marker and meter.clip_peak_db == 0.0


def test_peaks_are_ignored_while_nothing_is_measured(qapp):
    meter = LevelMeter("IN", clock=_Clock())
    meter.note_peak(1.0, 1.0, full_scale_count=5)
    assert not meter.clip_marker and meter.peak_hold_db <= meter.DB_MIN
    meter.set_levels(-20.0, -10.0)
    meter.note_peak(None, None, full_scale_count=0)
    _clip(meter)
    assert meter.clip_marker
    meter.set_unavailable()
    assert not meter.clip_marker and meter.recent_peak_db(10.0) is None
    assert not meter.clipped_within(10.0)


def test_unreadable_peaks_change_nothing(qapp):
    meter = _meter(_Clock())
    for bad in (None, float("nan"), float("inf"), -1.0, 0.0):
        meter.note_peak(bad, bad)
    assert not meter.clip_marker and meter.recent_peak_db(10.0) is None


def test_recent_peak_and_clip_age_cover_a_window_longer_than_the_marker(qapp):
    clock = _Clock()
    meter = _meter(clock)
    meter.note_peak(0.5, 0.5)
    clock.now += 4.0
    _clip(meter)
    meter.note_peak(1.2, 1.2)
    clock.now += 4.0
    meter.note_peak(0.1, 0.1)
    recent = meter.recent_peak_db(10.0)
    assert recent is not None and abs(recent - _db(1.2)) < 1e-6
    assert meter.recent_peak_db(3.0) == _db(0.1)
    assert meter.clipped_within(5.0) and not meter.clipped_within(3.0)
    clock.now += 20.0
    assert meter.recent_peak_db(10.0) is None


def test_a_meter_without_a_peak_reading_keeps_its_clip_for_the_whole_hold(qapp):
    clock = _Clock()
    meter = _meter(clock)
    _clip(meter)
    meter.set_levels(-30.0, None)  # the next tick has an average but no block peak
    assert meter.clip_marker and meter.clipped_within(CLIP_HOLD_SECONDS)
    assert meter.recent_peak_db(10.0) is not None
    assert meter.decay_timer.isActive()
    clock.now += CLIP_HOLD_SECONDS - 0.1
    meter.set_levels(-30.0, None)
    meter._decay_peak_hold()
    assert meter.clip_marker
    clock.now += 0.2
    meter._decay_peak_hold()
    assert not meter.clip_marker


def test_marker_is_described_to_assistive_technology_only_while_lit(qapp):
    clock = _Clock()
    meter = _meter(clock)
    widget = MeterWidget(meter)
    assert meter.property("description") == "" and widget.accessibleDescription() == ""
    _clip(meter)
    assert "Clipping" in meter.property("description")
    assert widget.accessibleDescription() == meter.property("description")
    clock.now += CLIP_HOLD_SECONDS + 0.1
    meter._decay_peak_hold()
    assert widget.accessibleDescription() == ""
    widget.deleteLater()


def _paint(meter: LevelMeter) -> QImage:
    image = QImage(60, 160, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    meter.paint_meter(painter, 60, 160)
    painter.end()
    return image


def test_lit_marker_draws_a_clip_tag_and_the_widget_paints_the_same_picture(qapp):
    clock = _Clock()
    meter = _meter(clock)
    quiet = _paint(meter)
    _clip(meter)
    lit = _paint(meter)
    assert lit != quiet
    # The tag is a word on a filled shape, not just a colour change.
    label_strip = (0, 160 - 18, 60, 18)
    assert lit.copy(*label_strip) != quiet.copy(*label_strip)

    view = MeterWidget(meter)
    view.resize(60, 160)
    rendered = QImage(60, 160, QImage.Format.Format_ARGB32_Premultiplied)
    rendered.fill(Qt.GlobalColor.transparent)
    view.render(rendered, QPoint(), QRegion(), QWidget.RenderFlag.DrawChildren)
    assert rendered == lit
    view.deleteLater()


class _CountingWidget(MeterWidget):
    def __init__(self, model):
        self.description_sets = 0
        super().__init__(model)

    def setAccessibleDescription(self, description):
        self.description_sets += 1
        super().setAccessibleDescription(description)


def test_level_updates_leave_the_accessible_description_alone_while_it_is_unchanged(qapp):
    clock = _Clock()
    meter = _meter(clock)
    widget = _CountingWidget(meter)
    widget.description_sets = 0
    for step in range(20):
        meter.set_levels(-30.0 + step, -20.0 + step)
    assert widget.description_sets == 0
    _clip(meter)
    assert widget.description_sets == 1 and "Clipping" in widget.accessibleDescription()
    widget.deleteLater()


def test_isolated_touches_of_full_scale_are_counted_as_times_not_as_a_flat_top(qapp):
    clock = _Clock()
    meter = _meter(clock)
    for _ in range(CLIP_FLAT_TOP_SAMPLES):
        clock.now += 0.4
        _clip(meter, 1)
    assert meter.clip_marker
    assert meter.full_scale_samples_within(8.0) == CLIP_FLAT_TOP_SAMPLES
    assert f"full scale {CLIP_FLAT_TOP_SAMPLES} times" in meter.property("description")
    clock.now += 9.0
    assert meter.full_scale_samples_within(8.0) == 0


def test_a_later_clip_after_the_marker_cleared_counts_from_its_own_rise(qapp):
    clock = _Clock()
    meter = _meter(clock)
    _clip(meter, 40)
    clock.now += CLIP_HOLD_SECONDS + 0.1
    meter._decay_peak_hold()
    assert meter.full_scale_samples_within(8.0) == 40  # still inside the longer Health window
    _clip(meter, CLIP_FLAT_TOP_SAMPLES)
    assert meter.full_scale_samples_within(8.0) == CLIP_FLAT_TOP_SAMPLES


def test_a_muted_meter_labels_itself_and_says_its_level_is_from_before_the_mute(qapp):
    def label_strip(meter: LevelMeter) -> QImage:
        image = QImage(60, 160, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.black)
        painter = QPainter(image)
        meter.paint_meter(painter, 60, 160)
        painter.end()
        return image.copy(0, 140, 60, 20)

    meter = _meter(_Clock(), "OUT")
    changes: list[None] = []
    meter.changed.connect(lambda: changes.append(None))
    unmuted = label_strip(meter)
    assert meter.property("description") == ""

    meter.set_muted(True)
    meter.set_muted(True)
    assert len(changes) == 1
    assert "before the mute" in meter.property("description")
    assert label_strip(meter) == label_strip(_meter(_Clock(), "MUTE"))

    meter.set_muted(False)
    assert meter.property("description") == "" and label_strip(meter) == unmuted
