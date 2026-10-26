"""What the Voice Setup review says about a weak capture and how good it was."""
# pyright: reportMissingImports=false
# (the sibling test module is found by pytest at run time)

from __future__ import annotations

from dataclasses import replace

import pytest

from mic_eq.ui.setup_messages import capture_issue_codes, capture_issue_lines, quality_tooltip, quality_word
from test_voice_setup import _analyze_clean_capture

WITHHELD = "Automatic loudness was not switched on"


def _lines(result):
    return capture_issue_lines(result["diagnostics"], result["compressor_settings"])


def test_a_reliable_capture_has_nothing_to_report(monkeypatch):
    assert _lines(_analyze_clean_capture(monkeypatch)) == []


@pytest.mark.parametrize(
    ("analysis_input", "alter", "code", "figure", "todo"),
    [
        (
            "analyze_voice_spectrum",
            lambda spectrum: replace(spectrum, snr_db=4.2),
            "snr_under_6_db",
            "4.2 dB louder",
            "closer to the microphone, or in a quieter room",
        ),
        (
            "_vad_masked_speech_features",
            lambda features: {**features, "active_duration_s": 1.5},
            "speech_under_2_s",
            "heard 1.5 seconds of speech",
            "speak for the whole take",
        ),
        (
            "analyze_voice_spectrum",
            lambda spectrum: replace(spectrum, used_single_spectrum_fallback=True),
            "spectrum_fallback",
            None,
            "speak for the whole take",
        ),
        (
            "analyze_noise_reference",
            lambda reference: replace(
                reference,
                status="invalid",
                reasons=["room-noise capture is dominated by changing events"],
            ),
            "noise_reference_not_usable",
            None,
            "Record the room noise again without talking.",
        ),
    ],
    ids=["snr-under-6", "under-2s-of-speech", "single-spectrum", "room-noise-not-usable"],
)
def test_each_clause_that_vetoes_the_capture_gets_its_own_line(
    monkeypatch, analysis_input, alter, code, figure, todo
):
    result = _analyze_clean_capture(monkeypatch, (analysis_input, alter))
    lines = _lines(result)

    assert result["diagnostics"]["weak_capture"] is True
    assert result["compressor_settings"]["auto_makeup_enabled"] is False
    # A weak SNR also drags the overall score under 0.5, which is a clause of its own.
    codes = capture_issue_codes(result["diagnostics"])
    assert len(codes) == len(lines)
    assert codes.count(code) == 1, codes
    own = lines[codes.index(code)]
    if figure:
        assert figure in own
    assert todo in own
    # What was held back because of it.
    assert WITHHELD in own and "suggestions only" in own


def test_a_voice_between_6_and_8_db_over_the_room_is_weak_but_keeps_automatic_loudness(monkeypatch):
    result = _analyze_clean_capture(
        monkeypatch, ("analyze_voice_spectrum", lambda spectrum: replace(spectrum, snr_db=7.0))
    )
    lines = _lines(result)

    assert result["diagnostics"]["weak_capture"] is True
    assert capture_issue_codes(result["diagnostics"]) == ["snr_under_8_db"]
    assert len(lines) == 1 and "7.0 dB louder" in lines[0]
    assert WITHHELD not in lines[0]  # only the four clauses of the reliability check withhold it


def test_a_low_overall_score_alone_is_reported_with_the_score(monkeypatch):
    result = _analyze_clean_capture(
        monkeypatch,
        ("analyze_voice_spectrum", lambda spectrum: replace(spectrum, residual_confidence=0.0)),
    )
    lines = _lines(result)

    assert capture_issue_codes(result["diagnostics"]) == ["capture_quality_low"]
    assert len(lines) == 1 and "unclear overall" in lines[0] and "%" in lines[0]


def test_two_clauses_that_fire_together_each_keep_their_own_line(monkeypatch):
    result = _analyze_clean_capture(
        monkeypatch,
        (
            "analyze_voice_spectrum",
            lambda spectrum: replace(spectrum, snr_db=4.0, used_single_spectrum_fallback=True),
        ),
    )

    lines = _lines(result)

    assert len(lines) == 2
    assert any("4.0 dB louder" in line for line in lines)
    assert any("too few clear voiced sounds" in line for line in lines)


@pytest.mark.parametrize(
    "diagnostics",
    [
        {"weak_capture": True},
        {"weak_capture": True, "noise_referenced_snr_db": float("nan")},
        {"weak_capture": True, "noise_reference_quality": {"status": "usable"}},
    ],
)
def test_a_weak_capture_always_shows_at_least_one_line(diagnostics):
    assert capture_issue_lines(diagnostics)


def test_the_makeup_that_replaced_automatic_loudness_is_named():
    lines = capture_issue_lines(
        {"weak_capture": True, "vad_active_duration_s": 1.0}, {"makeup_gain_db": 4.0}
    )

    assert "(makeup is fixed at 4.0 dB)" in lines[0]


def test_the_quality_is_said_in_words_and_never_as_confidence():
    assert quality_word(0.9) == ("Good capture", "ok")
    assert quality_word(0.6) == ("Usable capture", "warn")
    assert quality_word(0.2) == ("Weak capture", "bad")
    # A weak capture is never "Good" or "Usable", whatever its score.
    assert quality_word(0.49, weak=True) == ("Weak capture", "warn")
    tooltip = quality_tooltip(0.91)
    assert "91%" in tooltip and "not the chance that the settings are right" in tooltip
    assert "confidence" not in tooltip.lower()


@pytest.mark.parametrize(
    ("snr", "shown", "rule"),
    [
        (5.6, "only 5.6 dB louder", "needs at least 6 dB"),
        (5.99, "only 5.9 dB louder", "needs at least 6 dB"),
        (6.0, "6.0 dB louder", "8 dB or more"),
        (7.99, "7.9 dB louder", "8 dB or more"),
    ],
)
def test_a_voice_level_under_a_threshold_never_prints_as_the_threshold(snr, shown, rule):
    (line,) = capture_issue_lines({"weak_capture": True, "noise_referenced_snr_db": snr})

    assert shown in line and rule in line


@pytest.mark.parametrize(("seconds", "shown"), [(1.99, "1.9"), (1.96, "1.9"), (0.04, "0.0"), (1.5, "1.5")])
def test_speech_just_under_two_seconds_never_prints_as_two(seconds, shown):
    (line,) = capture_issue_lines({"weak_capture": True, "vad_active_duration_s": seconds})

    assert f"heard {shown} seconds of speech; it needs at least 2." in line


@pytest.mark.parametrize(("quality", "shown"), [(0.496, "49%"), (0.4999, "49%"), (0.29, "29%"), (0.1, "10%")])
def test_a_quality_under_half_never_prints_as_half(quality, shown):
    (line,) = capture_issue_lines(
        {"weak_capture": True, "capture_confidence": quality, "noise_referenced_snr_db": 12.0}
    )

    assert f"(capture quality {shown})" in line


def test_a_silent_room_noise_take_is_not_told_to_stop_talking():
    from mic_eq.analysis.noise_reference import SILENT_REFERENCE_REASON

    (line,) = capture_issue_lines(
        {
            "weak_capture": True,
            "noise_reference_quality": {"status": "questionable", "reasons": [SILENT_REFERENCE_REASON]},
        }
    )

    assert "without talking" not in line
    assert "Check that the microphone is not muted" in line
    assert line.endswith("then record the room noise again.")


@pytest.mark.parametrize("snr", [float("-inf"), float("inf"), float("nan")])
def test_a_figure_that_is_not_a_number_never_breaks_the_lines(snr):
    assert capture_issue_lines({"weak_capture": True, "noise_referenced_snr_db": snr})


@pytest.mark.parametrize(
    ("diagnostics", "codes"),
    [
        ({"noise_referenced_snr_db": 4.0}, ["snr_under_6_db"]),
        ({"noise_referenced_snr_db": 7.0}, ["snr_under_8_db"]),
        ({"vad_active_duration_s": 1.2}, ["speech_under_2_s"]),
        ({"uncertainty_reasons": ["too few voiced frames for a stable spectrum"]}, ["spectrum_fallback"]),
        ({"noise_reference_quality": {"status": "questionable", "reasons": ["x"]}}, ["noise_reference_not_usable"]),
        ({"capture_confidence": 0.3}, ["capture_quality_low"]),
        (
            {"noise_referenced_snr_db": 4.0, "vad_active_duration_s": 1.2, "capture_confidence": 0.3},
            ["snr_under_6_db", "speech_under_2_s", "capture_quality_low"],
        ),
        ({}, ["weak_capture"]),
    ],
)
def test_each_weak_capture_cause_has_a_stable_code(diagnostics, codes):
    weak = {"weak_capture": True, **diagnostics}

    assert capture_issue_codes(weak) == codes
    assert len(capture_issue_codes(weak)) == len(capture_issue_lines(weak))


def test_a_reliable_capture_has_no_codes():
    assert capture_issue_codes({"noise_referenced_snr_db": 12.0, "vad_active_duration_s": 5.0}) == []
