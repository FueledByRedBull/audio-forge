"""What the two setup dialogs report to the main window, and the words they use for it."""
# pyright: reportMissingImports=false
# (the sibling test modules are found by pytest at run time)

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from mic_eq.ui.calibration_dialog import CalibrationDialog
from mic_eq.ui.voice_setup_dialog import VoiceSetupDialog
from mic_eq.ui.voice_setup_view import STEPS
from test_auto_eq_candidate_lifecycle import (
    _candidate,
    _capture,
    _clipped,
    _complete,
    _no_correction,
    _over_the_rule,
    _Owner,
    _owner_with,
    _scene_texts,
)
from test_voice_setup_integration import (
    _bind_capture_context,
    _comparison_dialog,
    _dismissed,
    _legacy_eq,
    _review_dialog,
    _short_voice_captures,
    _verdict_with_dulling_advisory,
    _verifying_dialog,
    _voice_setup_result,
)
from test_voice_setup_integration import real_main_window as _real_main_window
from test_first_run_setup import scene_item
from test_quick_shell import _AccessibleScene

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
    with _dismissed(dialog):
        dialog._on_analysis_complete(result)

        assert [r["outcome"] for r in records] == ["refused"]
        _check_record(records[0], "voice_setup")
        assert records[0]["reason_codes"] == ["incomplete_result"]
        assert records[0]["messages"] == [dialog.view_model.warning_text]


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
    with _dismissed(dialog):
        dialog._on_analysis_complete(result)

        assert [r["outcome"] for r in records] == ["advisory"]
        _check_record(records[0], "voice_setup")
        assert records[0]["reason_codes"] == ["too_little_vad_active_speech", "other"]
        assert records[0]["numbers"]["noise_referenced_snr_db"] == 12.5


# The cause-to-code mapping is test_setup_review.py::test_each_weak_capture_cause_has_a_stable_code;
# one cause proves the dialog passes the codes on.
@pytest.mark.parametrize(
    ("cause", "code"),
    [({"noise_referenced_snr_db": 4.0}, "snr_under_6_db")],
    ids=["snr-under-6"],
)
def test_a_weak_capture_reports_which_check_failed(real_main_window, cause, code) -> None:
    records = _voice(real_main_window)
    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    result["diagnostics"].update(
        {"weak_capture": True, "apply_recommended": False, "noise_referenced_snr_db": 12.5, **cause}
    )
    with _dismissed(dialog):
        dialog._on_analysis_complete(result)

        assert code in records[0]["reason_codes"]


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
    with _dismissed(dialog):
        dialog._on_verification_complete(_verdict_with_dulling_advisory("accept", advised=False))
        dialog._on_start_clicked()

        verified = [r for r in records if r["outcome"] == "verified"]
        assert len(verified) == 1
        _check_record(verified[0], "voice_setup")
        assert verified[0]["numbers"]["snr_change_db"] == -5.0
        assert verified[0]["numbers"]["spectral_target_error_after_db"] == 0.9


def test_a_rolled_back_verification_is_reported_with_its_codes(real_main_window) -> None:
    records = _voice(real_main_window)
    dialog, _speech = _verifying_dialog(real_main_window)
    with _dismissed(dialog):
        dialog._on_verification_complete(_verdict_with_dulling_advisory("rollback", advised=False))

        assert records[-1]["outcome"] == "rolled_back"
        _check_record(records[-1], "voice_setup")
        assert records[-1]["reason_codes"] == ["rollback_reason"]


def test_a_declined_final_comparison_is_reported_as_rolled_back(
    real_main_window, monkeypatch
) -> None:
    records = _voice(real_main_window)
    dialog, _speech = _verifying_dialog(real_main_window)
    _comparison_dialog(monkeypatch, {}, keep=False)
    with _dismissed(dialog):
        dialog._on_verification_complete(_verdict_with_dulling_advisory("accept", advised=False))
        dialog._compare_recording()

        assert records[-1]["outcome"] == "rolled_back"
        assert records[-1]["reason_codes"] == ["final_comparison_declined"]


def test_a_final_comparison_that_cannot_open_is_not_reported_as_declined(
    real_main_window, monkeypatch
) -> None:
    records = _voice(real_main_window)
    dialog, _speech = _verifying_dialog(real_main_window)
    _comparison_dialog(monkeypatch, {}, load_error="no such file")
    with _dismissed(dialog):
        dialog._on_verification_complete(_verdict_with_dulling_advisory("accept", advised=False))
        dialog._compare_recording()

        assert dialog.view_model.shown["phaseText"] == "Final comparison could not open; previous settings put back"
        warning = dialog.view_model.warning_text
        assert warning.startswith("The final comparison could not open, so your old settings were kept.")
        assert "Your old settings are active." in warning
        assert "declined" not in dialog.view_model.shown["phaseText"] and "You kept" not in warning
        assert dialog.setup_state == "completed" and dialog._final_comparison_pending is False
        assert records[-1]["outcome"] == "rolled_back"
        assert records[-1]["reason_codes"] == ["final_comparison_unavailable"]
        assert records[-1]["messages"] == [warning]


def test_failures_are_reported_as_failed(real_main_window) -> None:
    records = _voice(real_main_window)
    dialog, _speech = _verifying_dialog(real_main_window)
    with _dismissed(dialog):
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


def test_closing_after_a_result_is_cancelled_and_before_one_is_silent(real_main_window) -> None:
    records = _voice(real_main_window)
    empty = VoiceSetupDialog(parent=real_main_window, configuration=real_main_window.configuration)
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
    return dialog.view_model.issue_lines


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
    with _dismissed(dialog):
        dialog._on_analysis_complete(result)

        lines = _issue_lines(dialog)
        assert len(lines) == 4
        assert "4.0 dB louder" in lines[0] and "1.2 seconds of speech" in lines[1]
        assert "too few clear voiced sounds" in lines[2]
        assert "not steady" in lines[3] and "without talking" in lines[3]


def test_a_reliable_review_has_no_capture_lines_and_a_new_result_replaces_old_ones(
    real_main_window,
) -> None:
    dialog, _speech = _review_dialog(real_main_window)
    weak = _voice_setup_result(real_main_window, dialog)
    weak["diagnostics"].update(weak_capture=True, vad_active_duration_s=1.0)
    with _dismissed(dialog):
        dialog._show_summary(weak)
        assert len(_issue_lines(dialog)) == 1

        # Each line is read out as its own sentence.
        with _AccessibleScene() as scene:
            chip = scene_item(dialog, "issueChip0")
            spoken = " ".join((scene.name(chip), scene.description(chip)))
            assert _issue_lines(dialog)[0] in spoken

        dialog._show_summary(_voice_setup_result(real_main_window, dialog))
        assert _issue_lines(dialog) == []


def test_the_review_calls_the_score_a_capture_quality_never_confidence(real_main_window) -> None:
    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    with _dismissed(dialog):
        for apply_recommended, weak in ((True, False), (False, False), (False, True)):
            result["diagnostics"].update(apply_recommended=apply_recommended, weak_capture=weak)
            dialog._show_summary(result)
            text = dialog.view_model.chip("overall")["text"] + dialog.view_model.chip("overall")["tip"]
            assert "onfidence" not in text
            assert text.startswith("Weak capture" if weak else "Good capture")
        assert "92%" in dialog.view_model.chip("overall")["tip"]
        assert "92%" not in dialog.view_model.chip("overall")["text"]


def test_the_review_says_why_noise_suppression_and_the_gate_were_not_tuned(real_main_window) -> None:
    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    with _dismissed(dialog):
        result["diagnostics"].update(apply_recommended=True, weak_capture=False)
        result["diagnostics"]["joint_tuning"] = {
            "status": "withheld", "withheld_for": "questionable_noise_reference", "reasons": [],
        }
        dialog._show_summary(result)
        assert dialog.view_model.chip("overall")["text"].endswith(
            "\nNoise suppression and the gate were not tuned because the room-noise "
            "recording was not steady. Your current settings are kept."
        )
        result["diagnostics"]["joint_tuning"] = {"apply_recommended": True, "reasons": []}
        dialog._show_summary(result)
        assert dialog.view_model.chip("overall")["text"].endswith(
            "\nNew noise suppression and gate settings passed the checks."
        )


def test_a_suggestions_only_result_makes_apply_the_secondary_action(real_main_window) -> None:
    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    result["diagnostics"]["apply_recommended"] = False
    with _dismissed(dialog):
        dialog._on_analysis_complete(result)

        assert dialog.view_model.shown["phaseText"] == "Suggestions only: not every check passed"
        assert dialog.view_model.shown["startText"] == "Apply Suggestions"
        assert dialog.view_model.shown["startLook"] == "plain"

        dialog._reset_setup_ui()
        _bind_capture_context(dialog, *_short_voice_captures(48_000))
        dialog._on_analysis_complete(_voice_setup_result(real_main_window, dialog))
        assert dialog.view_model.shown["phaseText"] == "Recommendations ready"
        assert dialog.view_model.shown["startLook"] == "primary"


# --- Auto-EQ -----------------------------------------------------------------------------


def _auto_eq(monkeypatch=None, owner=None):
    owner = owner or _Owner()
    records: list[dict[str, Any]] = []
    owner.record_setup_result = records.append  # type: ignore[attr-defined]
    return owner, CalibrationDialog(owner, configuration=owner.configuration), records


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
    view = dialog.view_model
    texts = [view.warning_text, view.warning_tip, str(view.shown["startText"]), str(view.shown["curveDescription"])]
    texts += [chip["text"] + " " + chip["tip"] for chip in view.chip_list]
    return " ".join([*texts, _scene_texts(dialog)]).lower()


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


def test_the_comparison_keeps_the_original_eq_as_now_after_apply(real_main_window) -> None:
    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    result["eq_settings"] = _legacy_eq(real_main_window, [3.0] + [0.0] * 9)
    with _dismissed(dialog):
        dialog._on_analysis_complete(result)
        assert dialog.view_model.shown["eqCompareShown"]
        before = dialog.eq_compare.largest_change()
        assert before[0] == pytest.approx(3.0, abs=1.0)
        assert dialog.eq_compare.description().startswith("Largest change: +")

        dialog._apply_setup()
        assert dialog.setup_state == "verification_ready"

        assert dialog.eq_compare.largest_change() == pytest.approx(before)
        assert dialog.view_model.shown["eqCompareShown"]


def test_a_review_without_a_tone_result_draws_no_comparison(real_main_window) -> None:
    dialog, _speech = _review_dialog(real_main_window)
    result = _voice_setup_result(real_main_window, dialog)
    with _dismissed(dialog):
        dialog._show_summary(result)
        assert dialog.view_model.shown["eqCompareShown"]

        result["eq_settings"] = None
        result["eq_error"] = "Recording too unclear. Please try again."
        dialog._show_summary(result)

        assert not dialog.view_model.shown["eqCompareShown"]


def test_the_auto_eq_result_draws_the_proposed_eq_only_when_it_offers_one(qapp, monkeypatch) -> None:
    owner, dialog, _records = _auto_eq(owner=_owner_with())
    try:
        _complete(dialog, monkeypatch, _over_the_rule(owner.preset.eq))
        assert dialog.view_model.shown["eqCompareShown"] is True
        assert "Largest change" in dialog.eq_compare.description()

        _complete(dialog, monkeypatch, _no_correction(_over_the_rule(owner.preset.eq)))
        assert not dialog.view_model.shown["eqCompareShown"]
    finally:
        dialog.reject()
        owner.close()


# --- The polish after a rejected take and on the review ----------------------------------


def _current_step(dialog) -> int:
    return dialog.view_model.shown["step"]


def test_a_rejected_room_noise_take_stays_on_its_own_step_with_a_clean_meter(
    real_main_window,
) -> None:
    dialog = VoiceSetupDialog(parent=real_main_window, configuration=real_main_window.configuration)
    with _dismissed(dialog):
        dialog.setup_state = "noise_recording"
        dialog.level_meter.set_levels(-12.0, -0.2)
        dialog.view_model.set_page(timeText="Time remaining: 1s")

        dialog._on_recording_complete(np.full(48_000, np.nan, dtype=np.float32))

        assert dialog.setup_state == "idle"
        assert STEPS[_current_step(dialog)] == "Room noise"
        assert dialog.view_model.shown["noiseShown"] and not dialog.view_model.shown["targetShown"]
        assert dialog.view_model.shown["recordingTitle"] == "Recording"
        assert dialog.level_meter.measurement_available is False
        assert dialog.level_meter.is_clipping is False
        assert dialog.view_model.shown["timeText"] == "Ready to record again"
        assert "Time remaining" not in dialog.view_model.shown["timeText"]


def test_a_failed_level_check_stays_on_the_level_step(real_main_window) -> None:
    dialog = VoiceSetupDialog(parent=real_main_window, configuration=real_main_window.configuration)
    with _dismissed(dialog):
        dialog.setup_state = "level_check"
        dialog._on_recording_failed("The microphone stopped sending sound.")

        assert STEPS[_current_step(dialog)] == "Level"

        dialog._reset_setup_ui()
        assert STEPS[_current_step(dialog)] == "Target"


def test_compare_belongs_to_the_review_alone(real_main_window) -> None:
    dialog, _speech = _review_dialog(real_main_window)
    with _dismissed(dialog):
        assert not dialog.view_model.shown["compareShown"]

        dialog._on_analysis_complete(_voice_setup_result(real_main_window, dialog))
        assert STEPS[_current_step(dialog)] == "Review"
        assert dialog.view_model.shown["recordingTitle"] == "Review"
        assert dialog.view_model.shown["compareShown"] and dialog.view_model.shown["compareEnabled"]


def test_the_auto_eq_review_is_titled_review_and_shows_compare_only_there(qapp, monkeypatch) -> None:
    owner, dialog, _records = _auto_eq(owner=_owner_with())
    try:
        assert not dialog.view_model.shown["compareShown"]
        assert dialog.view_model.shown["recordingTitle"] == "Recording"

        _complete(dialog, monkeypatch, _over_the_rule(owner.preset.eq))

        assert dialog.view_model.shown["recordingTitle"] == "Review"
        assert dialog.view_model.shown["compareShown"] is True
    finally:
        dialog.reject()
        owner.close()


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

        warning = dialog.view_model.warning_text
        assert "These settings did not pass every check, so treat them as suggestions" in warning
        assert "final safety check on your recording did not pass" in warning
        assert warning.endswith("suggestions only.")
        # The capture problems have their own lines; the headline does not repeat their figures.
        assert "4.0 dB" not in warning and "1.2" not in warning
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

        warning = dialog.view_model.warning_text
        assert "These settings did not pass every check, so treat them as suggestions" in warning
        assert "final safety check" not in warning
    finally:
        _ended(dialog)


def test_the_auto_eq_tooltip_says_what_the_validation_score_rises_with(qapp, monkeypatch) -> None:
    owner, dialog, _records = _auto_eq(owner=_owner_with())
    try:
        _complete(dialog, monkeypatch, _over_the_rule(owner.preset.eq))

        tooltip = dialog.view_model.chip("confidence")["tip"]
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
        assert "does not use" in dialog.eq_compare.description()
    finally:
        _ended(dialog)


def test_auto_eq_says_when_the_eq_now_is_not_in_use(qapp, monkeypatch) -> None:
    from test_auto_eq_candidate_lifecycle import _ModeOwner

    owner = _ModeOwner("raw")
    dialog = CalibrationDialog(owner, configuration=owner.configuration)
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

        assert not dialog.view_model.shown["eqCompareShown"]
        assert any(
            record.levelno == logging.WARNING and "EQ comparison" in record.getMessage()
            for record in caplog.records
        )
    finally:
        _ended(dialog)


# --- A refused retake does not replace the run's outcome ---------------------------------


def _say_yes(monkeypatch) -> None:
    monkeypatch.setattr(QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.Yes)


def _refuse_voice_take(dialog: VoiceSetupDialog) -> None:
    dialog.setup_state = "noise_recording"
    dialog._on_recording_complete(np.full(48_000, np.nan, dtype=np.float32))


def _advisory_voice_result(window, dialog) -> dict[str, Any]:
    result = _voice_setup_result(window, dialog)
    result["diagnostics"]["apply_recommended"] = False
    return result


def test_voice_setup_reports_a_refused_verification_take_after_an_advisory_result(
    real_main_window,
) -> None:
    records = _voice(real_main_window)
    dialog, _speech = _review_dialog(real_main_window)
    dialog._on_analysis_complete(_advisory_voice_result(real_main_window, dialog))
    with _dismissed(dialog):
        dialog.setup_state = "verification_recording"
        dialog._reject_capture("The second take was not usable.")

        assert [(r["outcome"], r["reason_codes"]) for r in records] == [
            ("advisory", records[0]["reason_codes"]),
            ("refused", ["recording_rejected"]),
        ]
        assert dialog.setup_state == "completed"


def test_voice_setup_reports_a_refused_take_when_nothing_better_exists(
    real_main_window, monkeypatch
) -> None:
    records = _voice(real_main_window)
    _say_yes(monkeypatch)
    dialog, _speech = _review_dialog(real_main_window)
    # A result still waiting to be applied is dropped by a retake without being reported.
    dialog._on_analysis_complete(_voice_setup_result(real_main_window, dialog))
    try:
        dialog._on_retake_clicked()
        _refuse_voice_take(dialog)
        dialog.reject()

        assert [(r["outcome"], r["reason_codes"]) for r in records] == [
            ("refused", ["recording_rejected"])
        ]
    finally:
        dialog.deleteLater()


def test_a_new_result_after_a_refused_retake_is_reported_and_a_later_refusal_is_not(
    real_main_window, monkeypatch
) -> None:
    records = _voice(real_main_window)
    _say_yes(monkeypatch)
    dialog, _speech = _review_dialog(real_main_window)
    dialog._on_analysis_complete(_advisory_voice_result(real_main_window, dialog))
    with _dismissed(dialog):
        dialog._on_retake_clicked()
        _refuse_voice_take(dialog)
        _bind_capture_context(dialog, *_short_voice_captures(48_000))
        second = _voice_setup_result(real_main_window, dialog)
        second["eq_settings"] = {"band_freqs": [1000.0]}
        dialog._on_analysis_complete(second)
        dialog._on_retake_clicked()
        _refuse_voice_take(dialog)
        dialog.reject()  # the retake dropped the result, so closing adds no "cancelled"

        assert [r["outcome"] for r in records] == ["advisory", "refused"]
        assert records[-1]["reason_codes"] == ["incomplete_result"]


def _refuse_auto_eq_take(dialog: CalibrationDialog) -> None:
    dialog.recording_state = "recording"
    dialog._on_recording_complete(_clipped())


def test_auto_eq_keeps_its_result_when_a_retake_is_refused(qapp, monkeypatch) -> None:
    owner, dialog, records = _auto_eq(owner=_owner_with())
    _say_yes(monkeypatch)
    try:
        _complete(dialog, monkeypatch, _no_correction(_over_the_rule(owner.preset.eq)))
        dialog._on_retake_clicked()
        _refuse_auto_eq_take(dialog)
        dialog.reject()

        assert [r["outcome"] for r in records] == ["refused"]
        assert records[0]["reason_codes"][0] == "headroom_limited"
    finally:
        owner.close()


def test_auto_eq_reports_a_refused_take_when_nothing_better_exists(qapp, monkeypatch) -> None:
    owner, dialog, records = _auto_eq()
    _say_yes(monkeypatch)
    try:
        _complete(dialog, monkeypatch, _candidate(1.0, owner.preset.eq))  # waiting to be applied
        dialog._on_retake_clicked()
        _refuse_auto_eq_take(dialog)
        dialog.reject()

        assert [(r["outcome"], r["reason_codes"]) for r in records] == [
            ("refused", ["recording_rejected"])
        ]
    finally:
        owner.close()


def test_auto_eq_reports_the_new_result_after_a_retake(qapp, monkeypatch) -> None:
    owner, dialog, records = _auto_eq(owner=_owner_with())
    _say_yes(monkeypatch)
    try:
        _complete(dialog, monkeypatch, _no_correction(_over_the_rule(owner.preset.eq)))
        dialog._on_retake_clicked()
        _complete(dialog, monkeypatch, _candidate(2.0, owner.preset.eq))
        dialog._on_start_clicked()

        assert [r["outcome"] for r in records] == ["refused", "applied"]
    finally:
        if not dialog._close_requested:
            dialog.reject()
        owner.close()
