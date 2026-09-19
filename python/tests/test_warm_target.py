"""Focused checks for the warm Auto-EQ target curve."""

from __future__ import annotations

import numpy as np

from mic_eq.analysis.auto_eq_parts.target import get_target_curve
from mic_eq.config import EQ_FREQUENCIES, TARGET_CURVES


def test_warm_target_has_a_broad_body_and_restrained_presence() -> None:
    static = get_target_curve(EQ_FREQUENCIES, "warm", target_mode="static")
    adaptive = get_target_curve(
        EQ_FREQUENCIES,
        "warm",
        measured_db=np.linspace(-8.0, 8.0, len(EQ_FREQUENCIES)),
        target_mode="adaptive",
    )

    body = np.mean(static[1:4])
    presence = np.mean(static[4:8])
    assert body > presence
    assert np.max(np.abs(static)) <= 3.0
    assert np.max(np.abs(adaptive)) <= 2.0
    assert np.mean(adaptive[1:4]) > np.mean(adaptive[4:8])


def test_warm_static_mode_matches_catalog_and_adaptive_stays_warm() -> None:
    curve = TARGET_CURVES["warm"]
    static = get_target_curve(EQ_FREQUENCIES, "warm", target_mode="static")
    adaptive = get_target_curve(
        EQ_FREQUENCIES,
        "warm",
        measured_db=np.full(len(EQ_FREQUENCIES), 4.0),
        target_mode="adaptive",
    )

    np.testing.assert_allclose(static, curve.band_targets)
    np.testing.assert_allclose(adaptive, 0.4 * np.asarray(curve.band_targets))


def test_flat_target_remains_neutral_with_measured_audio() -> None:
    measured = np.linspace(-20.0, 20.0, len(EQ_FREQUENCIES))

    np.testing.assert_allclose(
        get_target_curve(
            EQ_FREQUENCIES,
            "flat",
            measured_db=measured,
            target_mode="adaptive",
        ),
        np.zeros(len(EQ_FREQUENCIES)),
    )
