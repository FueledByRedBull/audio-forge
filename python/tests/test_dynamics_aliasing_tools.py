"""Tests for dynamics aliasing diagnostics."""

from __future__ import annotations

import numpy as np

import evaluate_dynamics_aliasing as aliasing


def test_alignment_recovers_small_positive_lag():
    rng = np.random.default_rng(42)
    reference = rng.normal(0.0, 0.1, 4_000)
    candidate = np.concatenate((np.zeros(17), reference))

    aligned_reference, aligned_candidate, lag = aliasing._align(reference, candidate)

    assert lag == 17
    assert np.allclose(aligned_reference, aligned_candidate)


def test_relative_error_is_very_low_for_identical_signals():
    signal = np.sin(np.linspace(0.0, 20.0, 4_000))

    assert aliasing._relative_error_db(signal, signal) < -200.0
