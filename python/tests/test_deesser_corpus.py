"""Corpus, model provenance, and soft-fusion regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import evaluate_deesser_corpus
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


def test_threshold_search_takes_the_best_recall_within_the_false_positive_limit():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    threshold = evaluate_deesser_corpus._low_false_activation_threshold(
        labels, probabilities
    )

    assert 0.4 < threshold <= 0.6


def test_threshold_search_refuses_to_substitute_a_constant():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.9, 0.95, 0.4, 0.5])

    with pytest.raises(ValueError, match="false-positive rate"):
        evaluate_deesser_corpus._low_false_activation_threshold(
            labels, probabilities, maximum_false_positive_rate=0.0
        )


def test_out_of_fold_split_holds_out_whole_clips_and_balances_labels(monkeypatch):
    class StopAfterFrameFolds(Exception):
        pass

    clips = []
    next_row = 0
    for index in range(18):
        row_count = 1 + index % 3
        clips.append(
            {
                "name": f"clip-{index}",
                "label": int(index % 3 != 0),
                "row_start": next_row,
                "row_count": row_count,
            }
        )
        next_row += row_count
    features = np.arange(next_row, dtype=float).reshape(-1, 1)
    labels = np.zeros(next_row, dtype=int)
    monkeypatch.setattr(
        evaluate_deesser_corpus,
        "_extract_corpus",
        lambda: (features, labels, clips),
    )
    training_rows: list[set[int]] = []

    def record_fit(train_features, train_labels, **kwargs):
        training_rows.append({int(row) for row in train_features[:, 0]})
        if len(training_rows) == 6:
            raise StopAfterFrameFolds
        return 0.0, np.zeros(1)

    monkeypatch.setattr(evaluate_deesser_corpus, "_fit_logistic", record_fit)

    with pytest.raises(StopAfterFrameFolds):
        evaluate_deesser_corpus.train_and_evaluate()

    all_rows = set(range(next_row))
    held_out = [all_rows - rows for rows in training_rows]
    assert sum(len(rows) for rows in held_out) == next_row
    assert set().union(*held_out) == all_rows
    for rows in held_out:
        held_clips = [
            clip
            for clip in clips
            if set(range(clip["row_start"], clip["row_start"] + clip["row_count"])) <= rows
        ]
        assert sum(clip["row_count"] for clip in held_clips) == len(rows)
        assert sorted(int(clip["label"]) for clip in held_clips) == [0, 1, 1]


def test_the_shipped_clip_stage_trains_on_the_out_of_fold_scores_it_is_thresholded_on(monkeypatch):
    clips = []
    next_row = 0
    for index in range(18):
        row_count = 2 + index % 3
        clips.append(
            {
                "name": f"clip-{index}",
                "label": int(index % 3 != 0),
                "condition": "generated",
                "temporal_score": 0.1 * (index % 4),
                "row_start": next_row,
                "row_count": row_count,
            }
        )
        next_row += row_count
    rng = np.random.default_rng(3)
    features = rng.random((next_row, 3))
    labels = (rng.random(next_row) < 0.3).astype(int)
    monkeypatch.setattr(
        evaluate_deesser_corpus, "_extract_corpus", lambda: (features, labels, clips)
    )
    monkeypatch.setattr(
        evaluate_deesser_corpus, "_low_false_activation_threshold", lambda *_a, **_k: 0.5
    )
    fits: list[tuple[float, np.ndarray]] = []

    def record_fit(train_features, train_labels, *, regularization, **kwargs):
        fits.append((regularization, np.array(train_features)))
        # A frame model that depends on the rows it was fitted on, so its scores differ
        # between a fold's model and the model fitted on every row.
        if regularization == 2.0:
            return 0.0, np.asarray(train_features).mean(axis=0)
        return 0.0, np.zeros(np.asarray(train_features).shape[1])

    monkeypatch.setattr(evaluate_deesser_corpus, "_fit_logistic", record_fit)

    evaluate_deesser_corpus.train_and_evaluate()

    # Six folds, then the clip stage on every clip's out-of-fold scores.
    *fold_fits, shipped_fit = [rows for regularization, rows in fits if regularization == 0.4]
    assert len(fold_fits) == 6
    assert {tuple(row) for row in shipped_fit} == {tuple(row) for rows in fold_fits for row in rows}
