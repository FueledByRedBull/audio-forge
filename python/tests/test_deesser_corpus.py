"""Corpus, model provenance, and soft-fusion regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mic_eq.analysis.deesser_corpus import (
    CORPUS_CASES,
    CORPUS_LICENSE,
    CORPUS_VERSION,
    generate_deesser_case,
    labels_for_analysis_frames,
)
from mic_eq.analysis.deesser_fusion import (
    CLIP_COEFFICIENTS,
    CLIP_INTERCEPT,
    ENABLE_PROBABILITY_THRESHOLD,
    FRAME_COEFFICIENTS,
    FRAME_INTERCEPT,
    MODEL_VERSION,
    predict_clip_probability,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "evaluation" / "deesser-corpus-v1-report.json"


def test_generated_corpus_has_required_coverage_and_explicit_license():
    assert CORPUS_LICENSE == "CC0-1.0"
    assert CORPUS_VERSION
    assert len(CORPUS_CASES) == 96
    assert {case.sample_rate for case in CORPUS_CASES} == {44_100, 48_000}
    assert {case.condition for case in CORPUS_CASES} >= {
        "bright",
        "clean",
        "fricative_f",
        "hiss",
        "hvac",
        "transient",
    }
    assert {case.sibilant_kind for case in CORPUS_CASES} >= {"s", "sh", "f", None}
    assert {case.voice_hz for case in CORPUS_CASES} == {105.0, 155.0, 220.0}
    assert {case.distance_scale for case in CORPUS_CASES} == {0.55, 1.0}


def test_generated_case_frame_labels_align_with_sibilant_events():
    positive_case = next(case for case in CORPUS_CASES if case.sibilant_kind == "s")
    negative_case = next(case for case in CORPUS_CASES if case.condition == "bright")
    positive = generate_deesser_case(positive_case)
    negative = generate_deesser_case(negative_case)
    frame_indices = np.arange(250, dtype=int)

    assert np.count_nonzero(labels_for_analysis_frames(positive, frame_indices)) > 0
    assert np.count_nonzero(labels_for_analysis_frames(negative, frame_indices)) == 0


def test_report_matches_versioned_runtime_model_and_is_honest_about_scope():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    model = report["model"]
    corpus = report["corpus"]

    assert model["version"] == MODEL_VERSION
    assert corpus["version"] == CORPUS_VERSION
    assert corpus["license"] == CORPUS_LICENSE
    assert corpus["perceptual_validation"] is False
    assert corpus["real_recordings"] is False
    assert np.isclose(model["frame_intercept"], FRAME_INTERCEPT, atol=5e-4)
    assert np.allclose(
        model["frame_coefficients"],
        FRAME_COEFFICIENTS,
        atol=5e-4,
    )
    assert np.isclose(model["clip_intercept"], CLIP_INTERCEPT, atol=5e-4)
    assert np.allclose(
        model["clip_coefficients"],
        CLIP_COEFFICIENTS,
        atol=5e-4,
    )
    assert np.isclose(
        model["clip_operating_threshold"],
        ENABLE_PROBABILITY_THRESHOLD,
        atol=5e-4,
    )


def test_cross_validated_report_prioritizes_low_false_activation():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    frame = report["metrics"]["frame"]
    clip = report["metrics"]["clip"]

    assert "out-of-fold" in report["corpus"]["evaluation_method"]
    assert frame["pr_auc_average_precision"] >= 0.60
    assert frame["false_positive_rate"] <= 0.03
    assert clip["pr_auc_average_precision"] >= 0.95
    assert clip["false_positive_rate"] <= 0.03
    assert clip["recall"] >= 0.90
    assert clip["brier_score"] <= 0.08


def test_clip_fusion_is_continuous_around_operating_threshold():
    low = np.asarray([0.40, 0.45, 0.12, 0.45, 0.25, 0.70], dtype=float)
    probabilities = []
    for delta in np.linspace(-0.02, 0.02, 17):
        candidate = low.copy()
        candidate[0] += delta
        probabilities.append(predict_clip_probability(candidate))

    differences = np.diff(probabilities)
    assert np.all(differences > 0.0)
    assert float(np.max(differences)) < 0.02
    assert np.all(FRAME_COEFFICIENTS >= 0.0)
    assert np.all(CLIP_COEFFICIENTS >= 0.0)
