"""Invariants that every bench mix keeps through the factory chain, and the refusals of Voice Setup.

These are the invariant tests CONTRIBUTING.md names for a tuning change. Already proven elsewhere, not repeated:
sustained clipping is refused by the setup analysis (test_capture_clipping.py::
test_the_setup_analysis_refuses_sustained_clipping_but_not_a_stray_sample), a refused or incomplete Voice Setup
result changes nothing when applied (test_voice_setup_integration.py::
test_a_result_without_an_eq_says_nothing_still_applies), and the released headroom rule holds where the limiter
allowance does not apply (test_voice_setup.py::
test_the_released_headroom_rule_holds_where_the_limiter_allowance_does_not_apply).
"""

from __future__ import annotations

import numpy as np
import pytest

import bench_metrics as bm
from mic_eq import CORE_AVAILABLE
from mic_eq.analysis import joint_tuning, voice_setup
from mic_eq.analysis.auto_eq_parts import headroom

pytestmark = pytest.mark.skipif(not CORE_AVAILABLE, reason="native extension is not built")

SR = bm.SAMPLE_RATE
# The latency budget the code states: the joint tuner's limit on a suppressor's latency (its comment: 10 ms for
# RNNoise, 30 ms for the standard DeepFilterNet, plus 5 ms of allowance). The whole factory chain has to fit inside it.
LATENCY_BUDGET_MS = joint_tuning._DEFAULT_MAX_SUPPRESSOR_LATENCY_MS
# A pin, not a product budget: the factory chain measures 17.2 ms (RNNoise's 10 ms frame plus the limiters'
# lookahead), so this catches a doubling that the 35 ms budget would let through. Raise it with a stated reason.
LATENCY_PIN_MS = 20.0


@pytest.fixture(scope="module")
def bench() -> bm.Bench:
    return bm.load_bench()


def test_every_bench_mix_keeps_the_factory_chain_invariants(bench):
    settings = bm.current_settings(False)
    ceiling_db = float(settings["limiter_settings"]["ceiling_db"])
    assert settings["limiter_settings"]["enabled"]
    for mix_id in bench.manifest["mixes"]:
        raw = bm.build_mix(bench, mix_id)["raw"]
        result = bm.render(settings, raw)
        output = np.asarray(result["output_audio"], np.float32)
        assert output.shape == raw.shape and np.isfinite(output).all() and not result["non_finite_output"], mix_id
        # The ceiling the chain reports is the configured one, or lower when careful output is on.
        assert result["limiter_effective_ceiling_db"] <= ceiling_db + 1e-9, mix_id
        assert np.max(np.abs(output)) <= 10 ** (result["limiter_effective_ceiling_db"] / 20) + 1e-6, mix_id
        # No clipping: with the limiter on, not even where the input (the clipped take) already clipped.
        assert np.count_nonzero(np.abs(output) >= 1.0) == 0, mix_id
        latency_ms = result["chain_latency_samples"] * 1000.0 / SR
        assert latency_ms <= LATENCY_PIN_MS <= LATENCY_BUDGET_MS, mix_id


def test_the_limiter_holds_the_ceiling_when_driven(bench):
    """No bench mix reaches the limiter at factory settings (the loudest peaks at -2.7 dBFS), so drive one."""
    settings = bm.current_settings(False)
    settings["compressor_settings"].update(auto_makeup_enabled=False, makeup_gain_db=12.0)
    raw = bm.build_mix(bench, "hot-t05")["raw"]
    result = bm.render(settings, raw)
    output = np.asarray(result["output_audio"], np.float32)
    assert result["pre_limiter_true_peak_headroom_db"] < 0.0  # without the limiter this would clip
    assert result["limiter_gain_reduction_db"] > 0.0
    assert np.max(np.abs(output)) <= 10 ** (result["limiter_effective_ceiling_db"] / 20) + 1e-6
    assert np.isfinite(output).all() and np.count_nonzero(np.abs(output) >= 1.0) == 0


def _crop(mix: dict[str, np.ndarray], seconds: float, noise_seconds: float = 1.6) -> dict[str, np.ndarray]:
    size = int(seconds * SR)
    return {**mix, "raw": mix["raw"][:size], "clean": mix["clean"][:size], "labels": mix["labels"][:size],
            "noise_capture": mix["noise_capture"][: int(noise_seconds * SR)]}


def test_voice_setup_refuses_silence_the_clipped_take_and_under_two_seconds_of_speech(bench):
    silence = np.zeros(3 * SR + SR // 5, np.float32)
    # Each case with the veto that refuses it: B1 the unclear recording, B2 the weak capture, "capture" the take itself.
    cases = {
        "silence": ({"raw": silence, "clean": silence.astype(np.float64), "labels": np.ones(silence.size, bool),
                     "noise_capture": np.zeros(int(1.6 * SR), np.float32)}, "B1"),
        # One phrase (about 1.9 s of speech) and its pause.
        "under two seconds of speech": (_crop(bm.build_mix(bench, "room-t02"), 3.2), "B2"),
        "the clipped take": (bm.build_mix(bench, "clipped-t01"), "capture"),
    }
    for name, (mix, veto) in cases.items():
        outcome = bm.voice_setup(bm.make_context(mix), False)
        assert (outcome["outcome"], outcome["veto"]) == ("refused", veto), (name, outcome["reason"])


def test_an_applied_voice_setup_proposal_meets_the_final_headroom_check(bench):
    # The released thresholds, pinned: loosening one is a tuning change that this test should make visible.
    assert (headroom.HEADROOM_TARGET_DB, headroom.LIMITER_GAIN_REDUCTION_WARN_DB,
            headroom.TRUE_PEAK_GAIN_REDUCTION_WARN_DB) == (1.0, 1.0, 0.5)
    assert voice_setup.LIMITER_ALLOWANCE_DB == 3.0

    outcome = bm.voice_setup(bm.make_context(_crop(bm.build_mix(bench, "room-t02"), 6.0)), False)
    assert outcome["outcome"] == "applied", outcome["reason"]
    # The check is the released rule (allowance 0) or the narrow limiter allowance; nothing in between.
    allowance = float(outcome["limiter_allowance_db"])
    assert allowance in (0.0, voice_setup.LIMITER_ALLOWANCE_DB)
    final = outcome["final"]
    readings = {name: final[name] for name in headroom._NATIVE_SAFETY_METRICS}
    assert outcome["offline_headroom_safe"] and voice_setup._headroom_safe(readings, allowance)
    if allowance == 0.0:
        assert headroom._is_headroom_safe(readings)
    assert not final["clipped"] and not final["limiter_overload"]
