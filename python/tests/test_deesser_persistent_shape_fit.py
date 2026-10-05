"""The reproducible shape fit cannot consume held-out or substituted data."""

import hashlib
import json

import numpy as np
import pytest

import evaluate_deesser_real_speech as evaluator


def _rows():
    features = np.random.default_rng(813).normal(size=(90, 6))
    features[:, 5] = 2.0  # A constant feature must not penalize the intercept.
    return [{"index": index, "speaker": speaker, "condition": condition, "source": "evaluation",
             "features": features[index * 3 + offset].tolist(), "label": 1 if condition == "harsh" else -1}
            for index, speaker in enumerate(evaluator.PERSISTENT_SHAPE_FIT_SPEAKERS)
            for offset, condition in enumerate(evaluator.CONDITIONS)]


def test_ridge_matches_independent_augmented_least_squares_and_raw_scores():
    rows = _rows()
    model = evaluator._fit_persistent_shape(rows)
    x = np.array([row["features"] for row in rows])
    y = np.array([row["label"] for row in rows])
    mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    design = np.column_stack([np.ones(90), (x - mean) / scale])
    penalty = np.diag([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    expected, *_ = np.linalg.lstsq(np.vstack([design, penalty]), np.r_[y, np.zeros(7)], rcond=None)
    np.testing.assert_allclose(model["coefficients"], expected, atol=1e-12)
    np.testing.assert_allclose(model["mean"], mean, atol=1e-12)
    np.testing.assert_allclose(model["scale"], scale, atol=1e-12)
    np.testing.assert_allclose(model["raw_intercept"] + x @ model["raw_coefficients"],
                               design @ expected, atol=1e-12)
    assert model["coefficients"][0] == pytest.approx(y.mean(), abs=1e-12)
    assert model["positive_rule"] == "score >= 0"
    assert model["threshold"] == 0.0
    assert model["ridge"] == 1.0
    assert model["standardization_ddof"] == 0
    assert model["penalize_intercept"] is False


@pytest.mark.parametrize("change", ["missing", "duplicate", "setup", "ears", "label", "nonfinite", "width"])
def test_fit_refuses_missing_substituted_or_invalid_evidence(change):
    rows = _rows()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    elif change == "setup":
        rows[0]["source"] = "setup"
    elif change == "ears":
        rows[0]["speaker"] = "ears-p001"
    elif change == "label":
        rows[0]["label"] = 1
    elif change == "nonfinite":
        rows[0]["features"][0] = np.nan
    else:
        for row in rows:
            row["features"].pop()
    with pytest.raises(ValueError):
        evaluator._fit_persistent_shape(rows)


def test_non_voicebank_manifest_is_rejected_before_audio_read(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text(json.dumps({"description": "EARS", "speakers": {}}), encoding="utf-8")
    monkeypatch.setattr(evaluator.harness, "splits", lambda _: pytest.fail("must reject the corpus identity first"))
    with pytest.raises(ValueError, match="frozen VoiceBank manifest; EARS is evaluation-only"):
        evaluator._persistent_shape_corpus(tmp_path)


def test_changed_corpus_input_cannot_be_fit(tmp_path, monkeypatch):
    source = f"{evaluator.harness.DATASHARE.format(next(iter(evaluator.harness.VOICEBANK.values())))}#clean_trainset_28spk_wav.zip"
    manifest = {
        "licenses": {"VoiceBank (VCTK subset)": "CC BY 4.0"},
        "speakers": {speaker: {"source": source, "files": [{"path": "input.wav", "sha256": "0" * 64}]}
                     for speaker in evaluator.PERSISTENT_SHAPE_FIT_SPEAKERS},
        "noise": {},
    }
    content = json.dumps(manifest).encode()
    (tmp_path / "manifest.json").write_bytes(content)
    (tmp_path / "input.wav").write_bytes(b"changed audio")
    monkeypatch.setattr(evaluator, "PERSISTENT_SHAPE_MANIFEST_SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(evaluator.harness, "splits", lambda _: {
        "fit": {"speakers": list(evaluator.PERSISTENT_SHAPE_FIT_SPEAKERS), "environments": []},
    })
    with pytest.raises(ValueError, match="fit input hash mismatch: input.wav"):
        evaluator._persistent_shape_corpus(tmp_path)


@pytest.mark.parametrize("option,value", [
    ("--split", "test"), ("--split", "calibrate"), ("--first", "1"),
    ("--users", "29"), ("--workers", "2"), ("--rows-output", "rows.json"),
])
def test_cli_refuses_fit_partition_or_case_selection(tmp_path, monkeypatch, option, value):
    monkeypatch.setattr(evaluator, "fit_persistent_shape", lambda *_: pytest.fail("invalid fit must not start"))
    with pytest.raises(SystemExit) as failure:
        evaluator.main(["--fit-persistent-shape", "--output", str(tmp_path / "fit.json"), option, value])
    assert failure.value.code == 2


def test_fit_cli_writes_report_once_and_never_renders_or_overwrites(tmp_path, monkeypatch):
    report = {"model": {"raw_intercept": 0.125}, "scope": "test report"}
    monkeypatch.setattr(evaluator, "fit_persistent_shape", lambda *_: report)
    monkeypatch.setattr(evaluator, "_render", lambda *_: pytest.fail("fit cannot render the de-esser"))
    output = tmp_path / "fit.json"
    arguments = ["--fit-persistent-shape", "--output", str(output)]
    assert evaluator.main(arguments) == 0
    original = output.read_bytes()
    assert json.loads(original) == report
    monkeypatch.setattr(evaluator, "fit_persistent_shape", lambda *_: pytest.fail("must preserve existing output"))
    with pytest.raises(SystemExit) as failure:
        evaluator.main(arguments)
    assert failure.value.code == 2
    assert output.read_bytes() == original
