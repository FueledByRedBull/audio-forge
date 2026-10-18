"""Headroom validation of Auto-EQ candidates: native-render inputs, failed renders, the correction split, and when the validation gain scale abstains or reduces."""

from typing import Any

import numpy as np
import pytest

from mic_eq.analysis.auto_eq_parts import headroom
from mic_eq.analysis.auto_eq_parts.optimizer import calculate_eq_bands
from mic_eq.config import EQSettings

BAND_FREQS = [80.0, 160.0, 320.0, 640.0, 1280.0, 2500.0, 5000.0, 8000.0, 12000.0, 16000.0]


def _render(excess_db: float, **extra):
    """Native metrics whose headroom excess is exactly ``excess_db`` (> -0.5)."""
    return {
        "pre_limiter_true_peak_headroom_db": headroom.HEADROOM_TARGET_DB - excess_db,
        "limiter_gain_reduction_db": 0.0,
        "true_peak_limiter_gain_reduction_db": 0.0,
        **extra,
    }


def _script_native(monkeypatch, outcomes):
    """Answer successive native renders: a dict is a result, a string a failure message."""
    queue = list(outcomes)
    calls: list[dict] = []

    def simulate(_audio, _rate, _bands, settings):
        calls.append(settings)
        outcome = queue.pop(0)
        if isinstance(outcome, dict):
            return outcome, None
        return None, {"kind": "runtime_error", "message": outcome}

    monkeypatch.setattr(headroom, "_native_simulate", simulate)
    return calls


def _validate(*, typed: bool, gains: float = 1.0, **eq_extra):
    eq = {
        "band_freqs": list(BAND_FREQS),
        "band_gains": [gains] * 10,
        "band_qs": [1.41] * 10,
        "enabled": True,
        "apply_recommended": True,
        **eq_extra,
    }
    extra = {}
    if typed:
        extra = {
            "chain_settings": {"full_chain": True, "input_pre_filtered": False},
            "candidate_base_eq_settings": EQSettings().to_dict(),
        }
    return headroom.apply_headroom_validation(
        np.zeros(480, dtype=np.float32), 48_000, eq, **extra
    )


def test_chain_defaults_sent_to_the_native_simulator():
    expected = {
        "return_output_audio": False,
        "deesser_enabled": False,
        "deesser_auto_enabled": True,
        "deesser_auto_amount": 0.5,
        "deesser_low_cut_hz": 4000.0,
        "deesser_high_cut_hz": 11000.0,
        "deesser_threshold_db": -28.0,
        "deesser_ratio": 4.0,
        "deesser_attack_ms": 2.0,
        "deesser_release_ms": 80.0,
        "deesser_max_reduction_db": 6.0,
        "compressor_enabled": True,
        "compressor_threshold_db": -20.0,
        "compressor_ratio": 4.0,
        "compressor_attack_ms": 10.0,
        "compressor_release_ms": 200.0,
        "compressor_makeup_gain_db": 0.0,
        "compressor_adaptive_release": False,
        "compressor_base_release_ms": 50.0,
        "compressor_auto_makeup_enabled": False,
        "compressor_target_lufs": -18.0,
        "compressor_noise_reference_reliability": 0.0,
        "compressor_sidechain_highpass_enabled": True,
        "limiter_enabled": True,
        "limiter_ceiling_db": -0.5,
        "limiter_release_ms": 50.0,
        "limiter_careful_output_enabled": True,
        "gate_enabled": True,
        "gate_threshold_db": -40.0,
        "gate_attack_ms": 10.0,
        "gate_release_ms": 100.0,
        "gate_mode": 0,
        "gate_vad_threshold": 0.48,
        "gate_vad_hold_time_ms": 200.0,
        "gate_vad_pre_gain": 1.0,
        "gate_auto_threshold_enabled": True,
        "gate_margin_db": 10.0,
        "suppressor_enabled": True,
        "suppressor_strength": 1.0,
        "noise_model": "rnnoise",
    }
    assert headroom._flatten_chain_settings(None) == expected
    assert headroom._flatten_chain_settings({}) == expected


def test_a_band_value_that_is_not_a_number_falls_back_per_field():
    bands = headroom._bands_from_settings(
        {
            "band_freqs": ["loud", None, float("nan"), float("inf"), *BAND_FREQS[4:]],
            "band_gains": ["x", None, float("nan"), float("inf"), *[1.0] * 6],
            "band_qs": ["x", None, float("nan"), float("inf"), *[2.0] * 6],
        }
    )
    assert bands[:4] == [(1000.0, 0.0, 1.41)] * 4
    assert bands[4] == (1280.0, 1.0, 2.0)


def test_split_is_none_when_the_full_correction_already_fits():
    def validation(excess_db, flat):
        return {"before": _render(excess_db), "flat_excess_db": flat}

    # Exactly at the rule is a fit too; the split is only for a correction that breaks it.
    for excess in (-0.5, 0.0):
        assert headroom._headroom_excess_db(_render(excess)) <= 0.0
        assert headroom.headroom_excess_split_db(validation(excess, -3.0)) is None
    assert headroom.headroom_excess_split_db(validation(0.5, -3.0)) == (0.0, 0.5)


def test_split_separates_the_recordings_part_from_what_the_correction_adds():
    def split(excess_db, flat):
        return headroom.headroom_excess_split_db(
            {"before": _render(excess_db), "flat_excess_db": flat}
        )

    assert split(3.0, 1.0) == (1.0, 2.0)
    assert split(3.0, -2.0) == (0.0, 3.0)  # a recording that fits adds no part
    assert split(3.0, 3.0) is None  # the correction adds nothing
    assert split(3.0, 4.0) is None
    for flat in (None, True, "1", float("nan"), float("inf")):
        assert split(3.0, flat) is None
    assert headroom.headroom_excess_split_db({"flat_excess_db": 1.0}) is None


def test_adjacent_slope_is_gain_step_per_octave_of_the_widest_slope():
    freqs = [100.0, 200.0, 400.0, 800.0, 1600.0, 1700.0, 3400.0, 6800.0, 9000.0, 12000.0]
    gains = [0.0, -3.0, -3.0, -3.0, -3.0, -3.5, -3.5, -3.5, -3.5, -3.5]
    result = {"band_gains": gains, "band_freqs": freqs, "band_qs": [1.41] * 10}

    headroom._refresh_scaled_eq_metadata(result, None, None, None)

    assert result["max_adjacent_gain_difference_db"] == pytest.approx(3.0)
    # A cut counts like a boost. The 0.5 dB step across 1600-1700 Hz is steeper per octave
    # than the 3 dB step over one.
    assert result["max_adjacent_gain_slope_db_per_octave"] == pytest.approx(
        0.5 / np.log2(1700.0 / 1600.0)
    )


def _fit_result() -> dict[str, Any]:
    return {
        "band_gains": [1.0] * 10,
        "band_freqs": list(BAND_FREQS),
        "band_qs": [1.41] * 10,
    }


@pytest.mark.parametrize("odd", ["freqs", "measured_db", "target_db"])
def test_fit_context_arrays_of_different_shapes_are_not_scored(odd):
    arrays = {
        "freqs": np.geomspace(80.0, 16_000.0, 8),
        "measured_db": np.full(8, -40.0),
        "target_db": np.full(8, -30.0),
    }
    arrays[odd] = arrays[odd][:7]
    result = _fit_result()

    headroom._refresh_scaled_eq_metadata(result, None, None, None, arrays)

    assert "validation_before_error_db" not in result
    assert "validation_after_error_db" not in result


def test_fit_context_with_matching_shapes_is_scored_and_an_empty_one_is_not():
    result = _fit_result()
    freqs = np.geomspace(80.0, 16_000.0, 8)
    context = {"freqs": freqs, "measured_db": np.full(8, -40.0), "target_db": np.full(8, -30.0)}
    headroom._refresh_scaled_eq_metadata(result, None, None, None, context)
    assert result["validation_after_error_db"] < result["validation_before_error_db"]

    empty = _fit_result()
    context = {key: np.empty(0) for key in ("freqs", "measured_db", "target_db")}
    headroom._refresh_scaled_eq_metadata(empty, None, None, None, context)
    assert "validation_before_error_db" not in empty


def _validate_with(**kwargs):
    return headroom.apply_headroom_validation(
        np.zeros(480, dtype=np.float32),
        48_000,
        {
            "band_freqs": list(BAND_FREQS),
            "band_gains": [1.0] * 10,
            "band_qs": [1.41] * 10,
            "enabled": True,
            "apply_recommended": True,
        },
        **kwargs,
    )


def _chain_unavailable(chain_settings):
    return _validate_with(
        candidate_base_eq_settings=EQSettings().to_dict(), chain_settings=chain_settings
    )


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(
            lambda: _validate_with(candidate_base_eq_settings={"bands": []}),
            id="incumbent-invalid",
        ),
        pytest.param(lambda: _chain_unavailable(None), id="no-chain-settings"),
        pytest.param(
            lambda: _chain_unavailable({"full_chain": True, "input_pre_filtered": "no"}),
            id="pre-filter-not-a-bool",
        ),
    ],
)
def test_an_unvalidated_candidate_is_neither_safe_nor_advisory(run):
    result = run()

    validation = result["headroom_validation"]
    assert validation["status"] == "unavailable"
    assert validation["safe"] is False
    assert validation["authoritative"] is False
    assert validation["advisory"] is False
    assert result["headroom_safe"] is False
    assert result["headroom_advisory"] is False
    assert result["validated_candidate_eq"] is None
    assert result["apply_recommended"] is False


def test_a_render_that_fails_midway_is_neither_safe_nor_advisory(monkeypatch):
    _script_native(monkeypatch, [_render(2.0), "boom"])
    result = _validate(typed=True)

    validation = result["headroom_validation"]
    assert validation["status"] == "unavailable"
    assert "boom" in validation["reason"]
    assert validation["safe"] is False
    assert validation["authoritative"] is False
    assert validation["advisory"] is False
    assert result["headroom_safe"] is False
    assert result["headroom_advisory"] is False


def test_a_later_render_failure_raises_for_legacy_and_is_unavailable_for_typed(monkeypatch):
    calls = _script_native(monkeypatch, [_render(2.0), "boom"])
    with pytest.raises(RuntimeError, match="boom"):
        _validate(typed=False)
    assert len(calls) == 2

    calls = _script_native(monkeypatch, [_render(2.0), "boom"])
    result = _validate(typed=True)
    assert len(calls) == 2
    assert result["headroom_validation"]["status"] == "unavailable"
    assert result["validated_candidate_eq"] is None


def test_a_failed_first_render_raises_for_legacy_and_is_unavailable_for_typed(monkeypatch):
    _script_native(monkeypatch, ["boom"])
    with pytest.raises(RuntimeError, match="boom"):
        _validate(typed=False)

    _script_native(monkeypatch, ["boom"])
    result = _validate(typed=True)
    assert result["headroom_validation"]["status"] == "unavailable"


def test_a_failed_incumbent_render_is_unavailable(monkeypatch):
    # Every scale breaks the rule, so the user's current EQ is rendered for comparison.
    calls = _script_native(
        monkeypatch, [_render(2.0)] * len(headroom.HEADROOM_SCALES) + ["boom"]
    )
    result = _validate(typed=True)

    assert len(calls) == len(headroom.HEADROOM_SCALES) + 1
    assert result["headroom_validation"]["status"] == "unavailable"
    assert result["headroom_safe"] is False
    assert result["validated_candidate_eq"] is None


def test_a_failed_flat_render_only_costs_the_split(monkeypatch):
    # Safe at 0.55 (under 0.70), so the recording alone is rendered next; that render fails.
    calls = _script_native(
        monkeypatch, [_render(2.0), _render(1.5), _render(1.0), _render(0.0), "boom"]
    )
    result = _validate(typed=True)

    assert len(calls) == 5
    validation = result["headroom_validation"]
    assert validation["status"] == "safe"
    assert validation["gain_scale"] == 0.55
    assert "flat_excess_db" not in validation
    assert result["headroom_safe"] is True


@pytest.mark.parametrize(
    "peak, expected",
    [
        (-0.3, -0.3),
        (np.float32(-1.5), -1.5),
        (float("nan"), None),
        (float("inf"), None),
        ("loud", None),
        (None, None),
    ],
)
def test_incumbent_true_peak_is_reported_only_when_it_is_a_finite_number(
    monkeypatch, peak, expected
):
    extra = {} if peak is None else {"output_true_peak_db": peak}
    _script_native(
        monkeypatch,
        [_render(2.0)] * len(headroom.HEADROOM_SCALES) + [_render(1.0, **extra)],
    )
    result = _validate(typed=True)

    validation = result["headroom_validation"]
    assert validation["status"] == "risk_worse"
    assert validation["incumbent_excess_db"] == 1.0
    if expected is None:
        assert "incumbent_output_true_peak_db" not in validation
    else:
        assert validation["incumbent_output_true_peak_db"] == pytest.approx(expected)
        assert isinstance(validation["incumbent_output_true_peak_db"], float)


def _scaled_down_by_headroom(monkeypatch, *, typed):
    # Over the rule at 1.0 and 0.85, meets it at 0.70.
    calls = _script_native(monkeypatch, [_render(2.0), _render(1.0), _render(0.0)])
    result = _validate(typed=typed, validation_gain_scale=0.8)
    assert len(calls) == 3
    return result


@pytest.mark.parametrize("typed", [False, True])
def test_headroom_scale_multiplies_the_optimizers_validation_gain_scale(monkeypatch, typed):
    result = _scaled_down_by_headroom(monkeypatch, typed=typed)

    assert result["headroom_gain_scale"] == 0.70
    assert result["validation_gain_scale"] == pytest.approx(0.8 * 0.70)
    assert result["validation_confidence"] == pytest.approx(0.72)


@pytest.mark.parametrize("typed", [False, True])
def test_a_correction_that_fits_leaves_the_validation_gain_scale_alone(monkeypatch, typed):
    _script_native(monkeypatch, [_render(0.0)])
    result = _validate(typed=typed, validation_gain_scale=0.8)

    assert result["headroom_gain_scale"] == 1.0
    assert result["validation_gain_scale"] == pytest.approx(0.8)
    assert "validation_confidence" not in result


@pytest.mark.parametrize(
    "headroom_scale, expected_status",
    [(1.0, "apply"), (0.85, "abstain"), (0.0, "abstain")],
)
def test_only_a_headroom_scale_below_one_abstains_a_neutral_candidate(
    headroom_scale, expected_status
):
    result = _fit_result()
    result["band_gains"] = [0.1] * 10
    result["headroom_gain_scale"] = headroom_scale

    headroom._refresh_scaled_eq_metadata(result, None, None, None)

    assert result["recommendation_status"] == expected_status
    assert result["apply_recommended"] is (expected_status != "abstain")
    removed = "headroom validation removed the usable correction"
    assert (removed in result["abstention_reasons"]) is (expected_status == "abstain")


def test_the_optimizers_validation_gain_scale_alone_never_abstains_a_candidate():
    result = _fit_result()
    result["band_gains"] = [0.1] * 10
    result["validation_gain_scale"] = 0.5

    headroom._refresh_scaled_eq_metadata(result, None, None, None)

    assert result["recommendation_status"] == "apply"
    assert result["apply_recommended"] is True
    assert result["abstention_reasons"] == []


@pytest.mark.parametrize(
    "headroom_scale, expected_status",
    [(1.0, "apply"), (0.70, "apply"), (0.69, "reduced"), (0.55, "reduced")],
)
def test_a_usable_correction_scaled_below_seventy_percent_is_reduced(
    headroom_scale, expected_status
):
    result = _fit_result()
    result["headroom_gain_scale"] = headroom_scale

    headroom._refresh_scaled_eq_metadata(result, None, None, None)

    assert result["recommendation_status"] == expected_status
    assert result["apply_recommended"] is True


def test_validation_confidence_follows_the_optimizers_validation_gain_scale():
    freqs = np.geomspace(80.0, 16_000.0, 128)
    measured = -40.0 - 6.0 * np.log2(freqs / 1000.0)
    target = np.exp(-np.log2(freqs / 2500.0) ** 2)
    context: dict = {}
    base = calculate_eq_bands(freqs, measured, target, fit_context=context)

    confidences = {}
    for scale in (1.0, 0.5):
        result = dict(base)
        result["validation_gain_scale"] = scale
        headroom._refresh_scaled_eq_metadata(result, freqs, measured, target, context)
        confidences[scale] = result["validation_confidence"]

    assert confidences[1.0] - confidences[0.5] == pytest.approx(0.30 * 0.5)
