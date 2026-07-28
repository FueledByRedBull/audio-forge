from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_tool(name: str):
    path = Path(__file__).parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


vad_eval = _load_tool("evaluate_vad_models")
vad_corpus = _load_tool("build_vad_evaluation_corpus")
vad_child_corpus = _load_tool("fetch_vad_child_validation_corpus")
vad_selection = _load_tool("evaluate_vad_model_selection")


def test_frame_labels_use_majority_overlap_at_model_window_rate(tmp_path: Path):
    capture = vad_eval.Capture(
        path=tmp_path / "fixture.wav",
        sample_rate=16_000,
        audio=np.zeros(1_536, dtype=np.float32),
        speech_intervals_samples=((256, 1_024),),
    )

    labels = vad_eval._frame_labels(capture, 3)

    assert labels.tolist() == [True, True, False]


def test_binary_metrics_match_known_confusion_matrix():
    probabilities = np.asarray([0.9, 0.7, 0.6, 0.1])
    labels = np.asarray([True, False, True, False])

    metrics = vad_eval._binary_metrics(probabilities, labels, 0.5)

    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_negative"] == 0
    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == 0.5
    assert abs(metrics["f1"] - 0.8) < 1e-12


def test_event_metrics_reject_short_false_burst_and_measure_onset(tmp_path: Path):
    capture = vad_eval.Capture(
        path=tmp_path / "events.wav",
        sample_rate=16_000,
        audio=np.zeros(32 * 512, dtype=np.float32),
        speech_intervals_samples=((10 * 512, 25 * 512),),
    )
    labels = vad_eval._frame_labels(capture, 32)
    assert labels is not None
    probabilities = np.zeros(32, dtype=float)
    probabilities[1:5] = 0.9  # Shorter than the 250 ms event floor.
    probabilities[12:25] = 0.9

    metrics = vad_eval._event_metrics(
        [(capture, probabilities, labels)],
        threshold=0.5,
    )

    assert metrics["true_event_count"] == 1
    assert metrics["detected_event_count"] == 1
    assert metrics["false_opening_count"] == 0
    assert metrics["median_onset_delay_ms"] == 64.0


def test_logit_calibration_is_monotonic_and_improves_shifted_fixture():
    probabilities = np.asarray([0.05, 0.10, 0.20, 0.35, 0.45, 0.55, 0.65, 0.8])
    labels = np.asarray([False, False, False, False, True, True, True, True])

    a, b = vad_eval._fit_logit_calibration(probabilities, labels)
    calibrated = vad_eval._apply_logit_calibration(probabilities, a, b)

    assert a > 0.0
    assert np.all(np.diff(calibrated) > 0.0)
    assert np.mean((calibrated - labels.astype(float)) ** 2) < np.mean(
        (probabilities - labels.astype(float)) ** 2
    )


def test_deterministic_noise_segment_repeats_and_removes_dc():
    noise = np.linspace(-1.0, 2.0, 10_000)

    first = vad_corpus._deterministic_noise_segment(noise, 2_000, "fixture")
    second = vad_corpus._deterministic_noise_segment(noise, 2_000, "fixture")

    np.testing.assert_array_equal(first, second)
    assert abs(float(np.mean(first))) < 1e-12


def test_pitch_up_shortens_waveform_without_non_finite_samples():
    source = np.sin(np.linspace(0.0, 100.0, 1_300))

    pitched = vad_corpus._pitch_up(source)

    assert pitched.size == 1_000
    assert np.isfinite(pitched).all()


def test_native_calibration_inversion_round_trips_probabilities():
    raw = np.asarray([0.01, 0.1, 0.5, 0.9, 0.99])
    calibrated = vad_eval._apply_logit_calibration(
        raw,
        vad_selection.NATIVE_CALIBRATION_A,
        vad_selection.NATIVE_CALIBRATION_B,
    )

    recovered = vad_selection._invert_native_calibration(calibrated)

    np.testing.assert_allclose(recovered, raw, atol=1e-10, rtol=1e-8)


def test_threshold_independent_metrics_rank_perfect_ordering():
    labels = np.asarray([False, False, True, True])
    perfect = np.asarray([0.1, 0.2, 0.8, 0.9])
    reversed_scores = perfect[::-1]

    assert vad_selection._roc_auc(perfect, labels) == 1.0
    assert vad_selection._roc_auc(reversed_scores, labels) == 0.0
    assert vad_selection._average_precision(perfect, labels) == 1.0


def test_child_selection_is_deterministic_and_speaker_balanced():
    rows = []
    for gender in ("female", "male"):
        for speaker_index in range(3):
            for utterance_index in range(3):
                rows.append(
                    {
                        "age": "6",
                        "gender": gender,
                        "speaker_id": f"{gender}-{speaker_index}",
                        "filename": f"{speaker_index}-{utterance_index}.flac",
                    }
                )

    first = vad_child_corpus._select_rows(rows)
    second = vad_child_corpus._select_rows(list(reversed(rows)))

    assert first == second
    assert len(first) == 8
    speakers = {row["speaker_id"] for row in first}
    assert len(speakers) == 4
    assert all(sum(row["speaker_id"] == speaker for row in first) == 2 for speaker in speakers)
