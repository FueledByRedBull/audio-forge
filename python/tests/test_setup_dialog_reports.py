"""What the two setup dialogs report to the main window, and the words they use for it."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pytest
from PySide6.QtWidgets import QComboBox, QGroupBox, QLabel, QMessageBox, QPushButton

from mic_eq.ui.calibration_dialog import CalibrationDialog
from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog
from test_auto_eq_candidate_lifecycle import (
    _candidate,
    _capture,
    _clipped,
    _complete,
    _no_correction,
    _over_the_rule,
    _Owner,
    _owner_with,
)
from test_voice_setup_integration import (
    _bind_capture_context,
    _comparison_dialog,
    _legacy_eq,
    _review_dialog,
    _short_voice_captures,
    _verdict_with_dulling_advisory,
    _verifying_dialog,
    _voice_setup_result,
)
from test_voice_setup_integration import real_main_window as _real_main_window

real_main_window = _real_main_window  # the fixture, under the name the tests ask for

OUTCOMES = {
    "applied", "verified", "rolled_back", "refused", "advisory", "cancelled", "failed",
}


def _check_record(record: dict[str, Any], kind: str) -> None:
    assert set(record) == {
        "kind", "outcome", "reason_codes", "messages", "numbers", "recorded_at",
    }
    assert record["kind"] == kind and record["outcome"] in OUTCOMES
    assert all(
        isinstance(code, str) and code and code == code.lower() and " " not in code
        for code in record["reason_codes"]
    )
    assert all(isinstance(message, str) and message for message in record["messages"])
    assert all(
        isinstance(value, (bool, int, float, str)) for value in record["numbers"].values()
    )
    stamp = datetime.fromisoformat(record["recorded_at"])
    assert stamp.utcoffset() == timedelta(0)


def _voice(window, monkeypatch=None):
    records: list[dict[str, Any]] = []
    window.record_setup_result = records.append
    return records


# --- Voice Setup -------------------------------------------------------------------------


def test_an_incomplete_voice_result_is_reported_as_refused(real_main_window) -> None:
    records = _voice(real_main_window)
    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    result["eq_settings"] = {"band_freqs": [1000.0]}
    try:
        dialog._on_analysis_complete(result)

        assert [r["outcome"] for r in records] == ["refused"]
        _check_record(records[0], "voice_setup")
        assert records[0]["reason_codes"] == ["incomplete_result"]
        assert records[0]["messages"] == [dialog.warning_label.text()]
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_an_advisory_voice_result_is_reported_with_its_reasons(real_main_window) -> None:
    records = _voice(real_main_window)
    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    result["diagnostics"]["apply_recommended"] = False
    result["diagnostics"]["uncertainty_reasons"] = [
        "too little VAD-active speech",
        "something the analysis learned later",
    ]
    result["diagnostics"]["noise_referenced_snr_db"] = 12.5
    try:
        dialog._on_analysis_complete(result)

        assert [r["outcome"] for r in records] == ["advisory"]
        _check_record(records[0], "voice_setup")
        assert records[0]["reason_codes"] == ["too_little_vad_active_speech", "other"]
        assert records[0]["numbers"]["noise_referenced_snr_db"] == 12.5
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_a_verified_voice_setup_is_reported_with_the_checked_numbers(
    real_main_window, monkeypatch
) -> None:
    records = _voice(real_main_window)
    dialog, _speech = _verifying_dialog(real_main_window)
    monkeypatch.setattr(
        "mic_eq.ui.voice_setup_dialog.QMessageBox.information", lambda *_a, **_k: None
    )
    monkeypatch.setattr(real_main_window, "_prompt_save_current_preset", lambda **_k: None)
    _comparison_dialog(monkeypatch, {}, keep=True)
    try:
        dialog._on_verification_complete(_verdict_with_dulling_advisory("accept", advised=False))
        dialog._on_start_clicked()

        verified = [r for r in records if r["outcome"] == "verified"]
        assert len(verified) == 1
        _check_record(verified[0], "voice_setup")
        assert verified[0]["numbers"]["snr_change_db"] == -5.0
        assert verified[0]["numbers"]["spectral_target_error_after_db"] == 0.9
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_a_rolled_back_verification_is_reported_with_its_codes(real_main_window) -> None:
    records = _voice(real_main_window)
    dialog, _speech = _verifying_dialog(real_main_window)
    try:
        dialog._on_verification_complete(_verdict_with_dulling_advisory("rollback", advised=False))

        assert records[-1]["outcome"] == "rolled_back"
        _check_record(records[-1], "voice_setup")
        assert records[-1]["reason_codes"] == ["rollback_reason"]
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_a_declined_final_comparison_is_reported_as_rolled_back(
    real_main_window, monkeypatch
) -> None:
    records = _voice(real_main_window)
    dialog, _speech = _verifying_dialog(real_main_window)
    _comparison_dialog(monkeypatch, {}, keep=False)
    try:
        dialog._on_verification_complete(_verdict_with_dulling_advisory("accept", advised=False))
        dialog._compare_recording()

        assert records[-1]["outcome"] == "rolled_back"
        assert records[-1]["reason_codes"] == ["final_comparison_declined"]
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_failures_are_reported_as_failed(real_main_window) -> None:
    records = _voice(real_main_window)
    dialog, _speech = _verifying_dialog(real_main_window)
    try:
        dialog._on_verification_failed("The second recording is gone.")
        dialog._on_analysis_failed("Setup lost track of its recordings.")
        dialog.setup_state = "noise_recording"
        dialog._on_recording_failed("The microphone stopped sending sound.")

        assert [r["outcome"] for r in records] == ["failed", "failed", "failed"]
        assert [r["reason_codes"] for r in records] == [
            ["verification_failed"], ["analysis_failed"], ["recording_failed"],
        ]
        for record in records:
            _check_record(record, "voice_setup")
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_a_refused_take_is_reported_as_refused(real_main_window) -> None:
    records = _voice(real_main_window)
    dialog = VoiceSetupDialog(parent=real_main_window)
    try:
        dialog.setup_state = "noise_recording"
        dialog._on_recording_complete(np.full(48_000, np.nan, dtype=np.float32))

        assert [r["outcome"] for r in records] == ["refused"]
        assert records[0]["reason_codes"] == ["recording_rejected"]
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_closing_after_a_result_is_cancelled_and_before_one_is_silent(real_main_window) -> None:
    records = _voice(real_main_window)
    empty = VoiceSetupDialog(parent=real_main_window)
    empty.reject()
    empty.deleteLater()
    assert records == []

    dialog, _speech = _review_dialog(real_main_window)
    dialog._on_analysis_complete(_voice_setup_result(real_main_window, dialog))
    dialog.reject()
    dialog.deleteLater()

    assert [r["outcome"] for r in records] == ["cancelled"]
    _check_record(records[0], "voice_setup")


# --- The review: one line for each reason the capture was weak ---------------------------


def _issue_lines(dialog: VoiceSetupDialog) -> list[str]:
    layout = dialog.capture_issues_layout
    labels = [item.widget() for item in map(layout.itemAt, range(layout.count())) if item]
    return [label.text() for label in labels if isinstance(label, QLabel)]


def test_the_review_shows_each_weak_capture_clause_as_its_own_line(real_main_window) -> None:
    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    result["diagnostics"].update(
        weak_capture=True,
        apply_recommended=False,
        noise_referenced_snr_db=4.0,
        vad_active_duration_s=1.2,
        uncertainty_reasons=["too few voiced frames for a stable spectrum"],
        noise_reference_quality={
            "status": "questionable",
            "reasons": ["room-noise capture is not stationary"],
        },
    )
    try:
        dialog._on_analysis_complete(result)

        lines = _issue_lines(dialog)
        assert len(lines) == 4
        assert "4.0 dB louder" in lines[0] and "1.2 seconds of speech" in lines[1]
        assert "too few clear voiced sounds" in lines[2]
        assert "not steady" in lines[3] and "without talking" in lines[3]
        assert all("suggestions only" in line for line in lines)
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_a_reliable_review_has_no_capture_lines_and_a_new_result_replaces_old_ones(
    real_main_window,
) -> None:
    dialog, _speech = _review_dialog(real_main_window)
    weak = _voice_setup_result(real_main_window, dialog)
    weak["diagnostics"].update(weak_capture=True, vad_active_duration_s=1.0)
    try:
        dialog._show_summary(weak)
        assert len(_issue_lines(dialog)) == 1

        dialog._show_summary(_voice_setup_result(real_main_window, dialog))
        assert _issue_lines(dialog) == []
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_the_review_calls_the_score_a_capture_quality_never_confidence(real_main_window) -> None:
    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    try:
        for apply_recommended, weak in ((True, False), (False, False), (False, True)):
            result["diagnostics"].update(apply_recommended=apply_recommended, weak_capture=weak)
            dialog._show_summary(result)
            text = dialog.overall_label.text() + dialog.overall_label.toolTip()
            assert "onfidence" not in text
            assert text.startswith("Weak capture" if weak else "Good capture")
        assert "92%" in dialog.overall_label.toolTip()
        assert "92%" not in dialog.overall_label.text()
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_a_suggestions_only_result_makes_apply_the_secondary_action(real_main_window) -> None:
    from mic_eq.ui.theme import PRIMARY_ACTION_BUTTON_STYLE

    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    result["diagnostics"]["apply_recommended"] = False
    try:
        dialog._on_analysis_complete(result)

        assert dialog.phase_label.text() == "Suggestions only: not every check passed"
        assert dialog.start_button.text() == "Apply Suggestions"
        assert dialog.start_button.styleSheet() == ""

        dialog._reset_setup_ui()
        _bind_capture_context(dialog, *_short_voice_captures(48_000))
        dialog._on_analysis_complete(_voice_setup_result(real_main_window, dialog))
        assert dialog.phase_label.text() == "Recommendations ready"
        assert dialog.start_button.styleSheet() == PRIMARY_ACTION_BUTTON_STYLE
    finally:
        dialog.reject()
        dialog.deleteLater()


# --- Auto-EQ -----------------------------------------------------------------------------


def _auto_eq(monkeypatch=None, owner=None):
    owner = owner or _Owner()
    records: list[dict[str, Any]] = []
    owner.record_setup_result = records.append  # type: ignore[attr-defined]
    return owner, CalibrationDialog(owner), records


def test_an_applied_auto_eq_is_reported_as_applied(qapp, monkeypatch) -> None:
    owner, dialog, records = _auto_eq()
    try:
        _complete(dialog, monkeypatch, _candidate(2.0, owner.preset.eq))
        dialog._on_start_clicked()

        assert [r["outcome"] for r in records] == ["applied"]
        _check_record(records[0], "auto_eq")
        assert records[0]["reason_codes"] == ["correction_ready"]
        assert records[0]["numbers"]["apply_recommended"] is True
    finally:
        if not dialog._close_requested:
            dialog.reject()
        owner.close()


def test_a_correction_that_was_not_offered_is_reported_as_refused(qapp, monkeypatch) -> None:
    owner, dialog, records = _auto_eq(owner=_owner_with())
    try:
        _complete(dialog, monkeypatch, _no_correction(_over_the_rule(owner.preset.eq)))

        assert [r["outcome"] for r in records] == ["refused"]
        _check_record(records[0], "auto_eq")
        assert records[0]["reason_codes"][0] == "headroom_limited"
        assert "headroom_validation_removed_the_usable_correction" in records[0]["reason_codes"]
    finally:
        dialog.reject()
        owner.close()


def test_auto_eq_failures_and_refused_takes_are_reported(qapp, monkeypatch) -> None:
    owner, dialog, records = _auto_eq()
    try:
        _capture(dialog, monkeypatch, "broadcast")
        dialog._on_analysis_failed("Setup lost track.", generation=dialog._analysis_generation)
        dialog.recording_state = "recording"
        dialog._on_recording_complete(_clipped())
        dialog.recording_state = "recording"
        dialog._on_recording_failed("The microphone stopped sending sound.")

        assert [r["outcome"] for r in records] == ["failed", "refused", "failed"]
        for record in records:
            _check_record(record, "auto_eq")
    finally:
        dialog.reject()
        owner.close()


def test_auto_eq_closed_after_a_result_is_cancelled_and_after_a_retake_is_not(
    qapp, monkeypatch
) -> None:
    owner, dialog, records = _auto_eq()
    _complete(dialog, monkeypatch, _candidate(1.0, owner.preset.eq))
    dialog.reject()
    assert [r["outcome"] for r in records] == ["cancelled"]
    _check_record(records[0], "auto_eq")
    owner.close()

    owner, dialog, records = _auto_eq()
    monkeypatch.setattr(QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.Yes)
    _complete(dialog, monkeypatch, _candidate(1.0, owner.preset.eq))
    dialog._on_retake_clicked()
    dialog.reject()
    assert records == []
    owner.close()


# --- Auto-EQ does not claim to measure or calibrate --------------------------------------

_CLAIMS = ("calibrat", "confidence", "measurement", "measured", "your microphone's")


def _shown_text(dialog: CalibrationDialog) -> str:
    texts: list[str] = [dialog.warning_label.text(), dialog.warning_label.toolTip()]
    for label in dialog.findChildren(QLabel):
        texts += [label.text(), label.toolTip(), label.accessibleName()]
    for button in dialog.findChildren(QPushButton):
        texts += [button.text(), button.toolTip(), button.accessibleName()]
    for group in dialog.findChildren(QGroupBox):
        texts.append(group.title())
    for combo in dialog.findChildren(QComboBox):
        texts += [combo.toolTip()] + [combo.itemText(i) for i in range(combo.count())]
    return " ".join(texts).lower()


def test_auto_eq_text_never_claims_to_measure_or_calibrate(qapp, monkeypatch) -> None:
    owner, dialog, _records = _auto_eq(owner=_owner_with())
    try:
        idle = _shown_text(dialog)
        _complete(dialog, monkeypatch, _over_the_rule(owner.preset.eq))
        ready = _shown_text(dialog)
        for text in (idle, ready):
            for phrase in _CLAIMS:
                assert phrase not in text, phrase
        # Auto-EQ fits toward a tone preset, and the screen says so.
        assert "tone preset" in ready
    finally:
        dialog.reject()
        owner.close()


# --- The proposed EQ drawn over the one now ----------------------------------------------


def test_the_voice_review_draws_the_proposed_eq_over_the_one_now(real_main_window) -> None:
    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    result["eq_settings"] = _legacy_eq(real_main_window, [3.0] + [0.0] * 9)
    try:
        dialog._on_analysis_complete(result)

        compare = dialog.eq_compare
        assert compare.isHidden() is False
        change, _frequency = compare.largest_change()
        assert change == pytest.approx(3.0, abs=1.0)
        assert compare.accessibleDescription().startswith("Largest change: +")
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_the_comparison_keeps_the_original_eq_as_now_after_apply(real_main_window) -> None:
    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    result["eq_settings"] = _legacy_eq(real_main_window, [3.0] + [0.0] * 9)
    try:
        dialog._on_analysis_complete(result)
        before = dialog.eq_compare.largest_change()
        assert before[0] == pytest.approx(3.0, abs=1.0)

        dialog._apply_setup()
        assert dialog.setup_state == "verification_ready"

        assert dialog.eq_compare.largest_change() == pytest.approx(before)
        assert dialog.eq_compare.isHidden() is False
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_a_review_without_a_tone_result_draws_no_comparison(real_main_window) -> None:
    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    try:
        dialog._show_summary(result)
        assert dialog.eq_compare.isHidden() is False

        result["eq_settings"] = None
        result["eq_error"] = "Recording too unclear. Please try again."
        dialog._show_summary(result)

        assert dialog.eq_compare.isHidden()
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_the_auto_eq_result_draws_the_proposed_eq_only_when_it_offers_one(qapp, monkeypatch) -> None:
    owner, dialog, _records = _auto_eq(owner=_owner_with())
    try:
        _complete(dialog, monkeypatch, _over_the_rule(owner.preset.eq))
        assert dialog.eq_compare.isHidden() is False
        assert "Largest change" in dialog.eq_compare.accessibleDescription()

        _complete(dialog, monkeypatch, _no_correction(_over_the_rule(owner.preset.eq)))
        assert dialog.eq_compare.isHidden()
    finally:
        dialog.reject()
        owner.close()


# --- The polish after a rejected take and on the review ----------------------------------


def _current_step(dialog) -> int:
    from mic_eq.ui.components import STEP_CURRENT_STYLE

    labels = dialog.step_header._labels
    return next(i for i, label in enumerate(labels) if label.styleSheet() == STEP_CURRENT_STYLE)


def test_a_rejected_room_noise_take_stays_on_its_own_step_with_a_clean_meter(
    real_main_window,
) -> None:
    dialog = VoiceSetupDialog(parent=real_main_window)
    try:
        dialog.setup_state = "noise_recording"
        dialog.level_meter.set_levels(-12.0, -0.2)
        dialog.time_label.setText("Time remaining: 1s")

        dialog._on_recording_complete(np.full(48_000, np.nan, dtype=np.float32))

        assert dialog.setup_state == "idle"
        assert dialog.STEPS[_current_step(dialog)] == "Room noise"
        assert dialog.noise_group.isHidden() is False and dialog.curve_group.isHidden()
        assert dialog.recording_group.title() == "Recording"
        assert dialog.level_meter.measurement_available is False
        assert dialog.level_meter.is_clipping is False
        assert dialog.time_label.text() == "Ready to record again"
        assert "Time remaining" not in dialog.time_label.text()
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_a_failed_level_check_stays_on_the_level_step(real_main_window) -> None:
    dialog = VoiceSetupDialog(parent=real_main_window)
    try:
        dialog.setup_state = "level_check"
        dialog._on_recording_failed("The microphone stopped sending sound.")

        assert dialog.STEPS[_current_step(dialog)] == "Level"

        dialog._reset_setup_ui()
        assert dialog.STEPS[_current_step(dialog)] == "Target"
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_compare_belongs_to_the_review_alone(real_main_window) -> None:
    dialog, _speech = _review_dialog(real_main_window)
    try:
        assert dialog.compare_button.isHidden()

        dialog._on_analysis_complete(_voice_setup_result(real_main_window, dialog))
        assert dialog.STEPS[_current_step(dialog)] == "Review"
        assert dialog.recording_group.title() == "Review"
        assert dialog.compare_button.isHidden() is False and dialog.compare_button.isEnabled()
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_the_auto_eq_review_is_titled_review_and_shows_compare_only_there(qapp, monkeypatch) -> None:
    owner, dialog, _records = _auto_eq(owner=_owner_with())
    try:
        assert dialog.compare_button.isHidden()
        assert dialog.recording_group.title() == "Recording"

        _complete(dialog, monkeypatch, _over_the_rule(owner.preset.eq))

        assert dialog.recording_group.title() == "Review"
        assert dialog.compare_button.isHidden() is False
    finally:
        dialog.reject()
        owner.close()


def test_the_passage_is_never_cut_or_scrolled_inside_the_page(real_main_window, qapp) -> None:
    from mic_eq.analysis import voice_setup  # noqa: F401  (the dialogs import it already)
    from PySide6.QtWidgets import QTextEdit

    from mic_eq.ui.calibration_support import RAINBOW_PASSAGE

    owner = _Owner()
    dialogs = (VoiceSetupDialog(parent=real_main_window), CalibrationDialog(owner))
    try:
        for dialog in dialogs:
            dialog.resize(520, 700)
            dialog.show()
            qapp.processEvents()
            passage = [
                label for label in dialog.findChildren(QLabel) if label.text() == RAINBOW_PASSAGE
            ]
            assert len(passage) == 1 and not dialog.findChildren(QTextEdit)
            label = passage[0]
            assert label.wordWrap()
            # Tall enough for every wrapped line.
            assert label.height() >= label.heightForWidth(label.width()) - 1
    finally:
        for dialog in dialogs:
            dialog.reject()
        owner.close()


@pytest.mark.parametrize("restored", [True, False], ids=["restored", "not-restored"])
def test_a_rollback_says_only_what_is_true_about_the_previous_settings(
    real_main_window, monkeypatch, restored
) -> None:
    dialog, _speech = _verifying_dialog(real_main_window)
    monkeypatch.setattr(dialog, "_restore_pre_setup_snapshot", lambda: restored)
    try:
        dialog._on_verification_complete(_verdict_with_dulling_advisory("rollback", advised=False))
        text = dialog.warning_label.text()
    finally:
        dialog.reject()
        dialog.deleteLater()

    assert ("Your previous settings were put back." in text) is restored
    assert ("could not be fully put back" in text) is not restored
    assert text.count("put back") == 1
    assert text.startswith("Your voice stood out from the background noise by 5.0 dB less than before.")


# --- A result that was reported is not overwritten by closing the window ------------------


def _ended(dialog) -> None:
    dialog.reject()
    dialog.deleteLater()


def test_voice_closing_after_an_advisory_result_keeps_the_advisory_record(real_main_window) -> None:
    records = _voice(real_main_window)
    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    result["diagnostics"]["apply_recommended"] = False
    result["diagnostics"]["uncertainty_reasons"] = ["too little VAD-active speech"]
    dialog._on_analysis_complete(result)

    _ended(dialog)

    assert [r["outcome"] for r in records] == ["advisory"]


def test_voice_cancelling_after_a_refused_result_keeps_the_refused_record(real_main_window) -> None:
    records = _voice(real_main_window)
    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    result["eq_settings"] = {"band_freqs": [1000.0]}
    dialog._on_analysis_complete(result)

    dialog._on_cancel_clicked()
    dialog.deleteLater()

    assert [r["outcome"] for r in records] == ["refused"]


def test_voice_closing_after_a_good_result_is_reported_as_cancelled(real_main_window) -> None:
    records = _voice(real_main_window)
    dialog, _speech = _review_dialog(real_main_window)
    dialog._on_analysis_complete(_voice_setup_result(real_main_window, dialog))

    _ended(dialog)

    assert [r["outcome"] for r in records] == ["cancelled"]


def test_voice_closing_after_a_failure_or_a_rollback_keeps_that_record(real_main_window) -> None:
    for outcome in ("failed", "rolled_back"):
        records = _voice(real_main_window)
        dialog, _speech = _verifying_dialog(real_main_window)
        if outcome == "failed":
            dialog._on_verification_failed("The second recording is gone.")
        else:
            dialog._on_verification_complete(_verdict_with_dulling_advisory("rollback", advised=False))

        _ended(dialog)

        assert [r["outcome"] for r in records] == [outcome]


def test_voice_a_retake_starts_a_new_run_that_can_be_cancelled(real_main_window) -> None:
    records = _voice(real_main_window)
    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    result["diagnostics"]["apply_recommended"] = False
    dialog._on_analysis_complete(result)

    dialog._reset_setup_ui()
    _bind_capture_context(dialog, *_short_voice_captures(48_000))
    dialog._on_analysis_complete(_voice_setup_result(real_main_window, dialog))
    _ended(dialog)

    assert [r["outcome"] for r in records] == ["advisory", "cancelled"]


def test_auto_eq_cancelling_after_a_refused_result_keeps_the_refused_record(qapp, monkeypatch) -> None:
    owner, dialog, records = _auto_eq(owner=_owner_with())
    _complete(dialog, monkeypatch, _no_correction(_over_the_rule(owner.preset.eq)))

    dialog._on_cancel_clicked()

    assert [r["outcome"] for r in records] == ["refused"]
    owner.close()


def test_auto_eq_a_new_result_after_a_refused_one_can_be_cancelled(qapp, monkeypatch) -> None:
    owner, dialog, records = _auto_eq(owner=_owner_with())
    _complete(dialog, monkeypatch, _no_correction(_over_the_rule(owner.preset.eq)))
    _complete(dialog, monkeypatch, _candidate(1.0, owner.preset.eq))

    dialog.reject()

    assert [r["outcome"] for r in records] == ["refused", "cancelled"]
    owner.close()


def test_auto_eq_closing_after_a_failed_analysis_keeps_the_failed_record(qapp, monkeypatch) -> None:
    owner, dialog, records = _auto_eq(owner=_owner_with())
    _complete(dialog, monkeypatch, _candidate(1.0, owner.preset.eq))
    dialog.recording_state = "analyzing"
    dialog._on_analysis_failed("Setup lost track.", generation=dialog._analysis_generation)

    dialog.reject()

    assert [r["outcome"] for r in records] == ["failed"]
    owner.close()


# --- The words on the review ---------------------------------------------------------------


def test_the_capture_problem_lines_are_read_out_as_their_own_sentences(real_main_window) -> None:
    from PySide6.QtGui import QAccessible

    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    result["diagnostics"].update(weak_capture=True, vad_active_duration_s=1.2)
    try:
        dialog._show_summary(result)

        layout = dialog.capture_issues_layout
        items = [layout.itemAt(i) for i in range(layout.count())]
        labels = [item.widget() for item in items if item is not None]
        labels = [label for label in labels if isinstance(label, QLabel)]
        assert labels and len(labels) == layout.count()
        for label in labels:
            interface = QAccessible.queryAccessibleInterface(label)
            assert interface is not None
            spoken = " ".join(
                interface.text(part) for part in (QAccessible.Text.Name, QAccessible.Text.Description)
            )
            assert label.text() in spoken
    finally:
        _ended(dialog)


def test_an_advisory_result_says_each_capture_problem_once(real_main_window) -> None:
    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    result["diagnostics"].update(
        weak_capture=True,
        apply_recommended=False,
        noise_referenced_snr_db=4.0,
        vad_active_duration_s=1.2,
        uncertainty_reasons=[
            "too little VAD-active speech",
            "speech-to-noise ratio is weak",
            "offline DSP validation did not pass",
        ],
    )
    try:
        dialog._on_analysis_complete(result)

        assert dialog.warning_label.text() == (
            "These settings did not pass every check, so treat them as suggestions. "
            "The final safety check on your recording did not pass, so these settings "
            "are suggestions only."
        )
        assert len(_issue_lines(dialog)) == 2
    finally:
        _ended(dialog)


def test_an_advisory_result_whose_only_causes_are_capture_problems_keeps_just_the_headline(
    real_main_window,
) -> None:
    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    result["diagnostics"].update(
        weak_capture=True,
        apply_recommended=False,
        vad_active_duration_s=1.2,
        uncertainty_reasons=["too little VAD-active speech"],
    )
    try:
        dialog._on_analysis_complete(result)

        assert dialog.warning_label.text() == (
            "These settings did not pass every check, so treat them as suggestions."
        )
    finally:
        _ended(dialog)


def test_the_auto_eq_tooltip_says_what_the_validation_score_rises_with(qapp, monkeypatch) -> None:
    owner, dialog, _records = _auto_eq(owner=_owner_with())
    try:
        _complete(dialog, monkeypatch, _over_the_rule(owner.preset.eq))

        tooltip = dialog.confidence_label.toolTip()
        assert "rises as the correction gets closer to the tone preset" in tooltip
        assert "size of the correction" not in tooltip
    finally:
        dialog.reject()
        owner.close()


# --- The comparison says what "now" is when the EQ is not in use ---------------------------


def test_the_voice_review_says_when_the_eq_now_is_not_in_use(real_main_window, monkeypatch) -> None:
    monkeypatch.setattr(real_main_window, "_processing_mode", lambda: "bypass")
    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    result["eq_settings"] = _legacy_eq(real_main_window, [3.0] + [0.0] * 9)
    try:
        dialog._on_analysis_complete(result)

        assert dialog.eq_compare.lines()[0][0] == "Now (not used in Bypass)"
        assert "does not use" in dialog.eq_compare.accessibleDescription()
    finally:
        _ended(dialog)


def test_auto_eq_says_when_the_eq_now_is_not_in_use(qapp, monkeypatch) -> None:
    from test_auto_eq_candidate_lifecycle import _ModeOwner

    owner = _ModeOwner("raw")
    dialog = CalibrationDialog(owner)
    try:
        _complete(dialog, monkeypatch, _over_the_rule(owner.preset.eq))

        assert dialog.eq_compare.lines()[0][0] == "Now (not used in Raw monitor)"
    finally:
        dialog.reject()
        owner.close()


def test_a_comparison_that_cannot_be_drawn_is_logged_and_left_out(
    real_main_window, monkeypatch, caplog
) -> None:
    import logging

    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    result["eq_settings"] = _legacy_eq(real_main_window, [3.0] + [0.0] * 9)

    def refuse(*_args, **_kwargs):
        raise RuntimeError("no response")

    monkeypatch.setattr(dialog.eq_compare, "set_curves", refuse)
    try:
        with caplog.at_level(logging.WARNING, logger="mic_eq.ui.voice_setup_dialog"):
            dialog._show_summary(result)

        assert dialog.eq_compare.isHidden()
        assert any(
            record.levelno == logging.WARNING and "EQ comparison" in record.getMessage()
            for record in caplog.records
        )
    finally:
        _ended(dialog)
