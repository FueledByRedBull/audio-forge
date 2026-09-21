"""Joint selection must keep exact incumbent settings when evidence is weak."""

from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mic_eq.analysis import joint_tuning
from mic_eq.analysis.cancellation import AnalysisCancelled
from mic_eq.config import Preset
from mic_eq.mic_eq_core import simulate_gate_suppressor_order
from tools import evaluate_product_tuning
from tools.evaluate_product_tuning import (
    REPO_ROOT,
    _finalize_report,
    _git_revision,
    _selected_settings,
)
from tools.check_evaluation_hygiene import validate_report


def _tuning_args() -> tuple[Any, ...]:
    preset = Preset()
    return (
        np.full(48_000, 0.001, dtype=np.float32),
        np.full(48_000, 0.1, dtype=np.float32),
        48_000,
        asdict(preset.gate),
        asdict(preset.compressor),
        {
            "band_freqs": preset.eq.band_freqs,
            "band_gains": preset.eq.band_gains,
            "band_qs": preset.eq.band_qs,
        },
        asdict(preset.deesser),
        asdict(preset.limiter),
    )


def _downstream_metrics(audio: np.ndarray, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
    del audio
    return {
        "simulation_backend": "rust",
        "compressor_gain_reduction_p95_db": 3.5,
        "compressor_gain_reduction_median_db": 1.47,
        "compressor_gain_reduction_db": 4.0,
        "compressor_pumping_score_db": 0.0,
        "output_true_peak_db": -1.0,
        "limiter_effective_ceiling_db": -0.5,
        "pre_limiter_true_peak_headroom_db": 1.0,
        "non_finite_output": False,
        "candidate_runtime_ms": 0.0,
    }


def _activity_tape(audio: np.ndarray, settings: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    probability = 0.2 if float(np.mean(np.abs(audio))) < 0.01 else 0.8
    reliability = float(bool(settings.get("vad_available", True)))
    frame_count = (audio.size + 479) // 480
    return [(probability, reliability, -60.0, 0.0)] * frame_count


def test_native_gate_enabled_matches_requested_control():
    audio = np.full(4800, 0.01, dtype=np.float32)
    muted = simulate_gate_suppressor_order(audio, [0.0] * 10, False, 0.0,
                                          {"gate_enabled": True, "gate_threshold_db": -10.0,
                                           "gate_auto_threshold_enabled": False})
    open_gate = simulate_gate_suppressor_order(audio, [0.0] * 10, False, 0.0,
                                             {"gate_enabled": False})
    assert np.max(np.abs(muted["output_audio"])) < 0.001
    assert np.max(np.abs(open_gate["output_audio"])) > 0.005


def test_vad_alignment_waits_for_a_complete_window_and_holds_last_probability():
    aligned = joint_tuning._align_probabilities(
        np.asarray([0.2, 0.8], dtype=np.float32),
        4_800,
        4_800,
    )

    np.testing.assert_allclose(
        aligned,
        [0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8],
    )


def test_joint_unavailable_keeps_incumbent_and_cancellation_propagates(monkeypatch):
    preset = Preset()
    incumbent_gate = {**asdict(preset.gate), "threshold_db": -51.0}
    kwargs: dict[str, Any] = dict(incumbent_settings={"gate": incumbent_gate,
                  "suppressor": {"model": "rnnoise", "enabled": True, "strength": 0.62}})
    args = (np.ones(48000, dtype=np.float32) * 0.001,
            np.ones(96000, dtype=np.float32) * 0.1, 48000,
            asdict(preset.gate), asdict(preset.compressor),
            {"band_freqs": preset.eq.band_freqs, "band_gains": preset.eq.band_gains,
             "band_qs": preset.eq.band_qs}, asdict(preset.deesser), asdict(preset.limiter))
    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: None)
    result = joint_tuning.tune_gate_suppression_dynamics(*args, **kwargs)
    assert not result["apply_recommended"]
    assert result["gate_settings"] == incumbent_gate
    assert result["suppressor_settings"]["strength"] == 0.62
    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate_gate_suppressor_order)
    with pytest.raises(AnalysisCancelled):
        joint_tuning.tune_gate_suppression_dynamics(*args, **kwargs, cancel_check=lambda: True)


def test_joint_selection_evaluates_each_model_and_selects_lower_noise_model(
    monkeypatch,
):
    observed_releases: list[float] = []

    def simulate(audio, _probabilities, _before_gate, _strength, settings):
        model = settings["noise_model"]
        observed_releases.append(float(settings["gate_release_ms"]))
        latency = {"rnnoise": 480, "deepfilter-ll": 480, "deepfilter": 1_440}[model]
        is_noise = float(np.mean(np.abs(audio))) < 0.01
        attenuation = {"rnnoise": 0.5, "deepfilter-ll": 0.1, "deepfilter": 0.05}[model]
        output = np.asarray(audio, dtype=np.float32) * (
            attenuation if is_noise else 0.98
        )
        return {
            "output_audio": output,
            "dry_audio": output,
            "suppressor_latency_samples": latency,
            "runtime_ms": 0.0,
            "auto_makeup_activity": _activity_tape(audio, settings),
        }

    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate)
    monkeypatch.setattr(joint_tuning, "simulate_candidate_chain", _downstream_metrics)
    args = _tuning_args()
    frame_count = 100
    incumbent = {
        "gate": dict(args[3]),
        "suppressor": {"model": "rnnoise", "enabled": True, "strength": 1.0},
    }
    result = joint_tuning.tune_gate_suppression_dynamics(
        *args,
        vad_probabilities=np.ones(frame_count, dtype=np.float32),
        noise_vad_probabilities=np.zeros(frame_count, dtype=np.float32),
        incumbent_settings=incumbent,
        max_suppressor_latency_ms=15.0,
    )

    assert result["apply_recommended"]
    assert result["suppressor_settings"]["model"] == "deepfilter-ll"
    assert result["selected_candidate"]["noise_model"] == "deepfilter-ll"
    assert result["selected_candidate"]["metrics"]["suppressor_latency_ms"] == 10.0
    assert result["max_suppressor_latency_ms"] == 15.0
    assert [entry["model"] for entry in result["model_evaluations"]] == [
        "rnnoise",
        "deepfilter-ll",
        "deepfilter",
    ]
    assert {
        candidate["noise_model"] for candidate in result["candidates"]
    } == {"rnnoise", "deepfilter-ll", "deepfilter"}
    deepfilter_candidates = [
        candidate
        for candidate in result["candidates"]
        if candidate["noise_model"] == "deepfilter"
    ]
    assert deepfilter_candidates
    assert all(not candidate["gates"]["suppressor_latency"] for candidate in deepfilter_candidates)
    assert float(args[3]["release_ms"]) + 70.0 in observed_releases


def test_joint_selection_reports_unavailable_models_and_retains_incumbent(monkeypatch):
    def simulate(audio, _probabilities, _before_gate, _strength, settings):
        model = settings["noise_model"]
        if model != "rnnoise":
            raise RuntimeError(f"{model} backend unavailable")
        output = np.asarray(audio, dtype=np.float32)
        return {
            "output_audio": output,
            "dry_audio": output,
            "suppressor_latency_samples": 480,
            "runtime_ms": 0.0,
            "auto_makeup_activity": _activity_tape(audio, settings),
        }

    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate)
    monkeypatch.setattr(joint_tuning, "simulate_candidate_chain", _downstream_metrics)
    args = _tuning_args()
    result = joint_tuning.tune_gate_suppression_dynamics(
        *args,
        vad_probabilities=np.ones(100, dtype=np.float32),
        noise_vad_probabilities=np.zeros(100, dtype=np.float32),
        incumbent_settings={
            "gate": dict(args[3]),
            "suppressor": {"model": "rnnoise", "enabled": True, "strength": 0.62},
        },
    )

    assert not result["apply_recommended"]
    assert result["suppressor_settings"]["model"] == "rnnoise"
    assert result["suppressor_settings"]["strength"] == 0.62
    statuses = {entry["model"]: entry["status"] for entry in result["model_evaluations"]}
    assert statuses["rnnoise"] == "available"
    assert statuses["deepfilter-ll"] == "unavailable"
    assert statuses["deepfilter"] == "unavailable"
    assert any("backend unavailable" in entry.get("reason", "") for entry in result["model_evaluations"])


@pytest.mark.parametrize("noise_available", [True, False])
def test_joint_tuning_refreshes_probabilities_for_incumbent_pre_gain(monkeypatch, noise_available):
    observed_gains: list[float] = []

    def analyze(_audio, _sample_rate, *, threshold, pre_gain):
        del threshold
        observed_gains.append(float(pre_gain))
        if not noise_available and len(observed_gains) == 2:
            return None, "energy_fallback"
        return np.ones(100, dtype=np.float32), "silero"

    def simulate(audio, _probabilities, _before_gate, _strength, settings):
        output = np.asarray(audio, dtype=np.float32)
        return {
            "output_audio": output,
            "dry_audio": output,
            "suppressor_latency_samples": 480,
            "runtime_ms": 0.0,
            "auto_makeup_activity": _activity_tape(audio, settings),
        }

    monkeypatch.setattr(joint_tuning, "analyze_offline_vad", analyze)
    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate)
    monkeypatch.setattr(joint_tuning, "simulate_candidate_chain", _downstream_metrics)
    args = list(_tuning_args())
    args[3] = {**args[3], "vad_pre_gain": 1.0}
    incumbent_gate = {**args[3], "vad_pre_gain": 2.0}

    result = joint_tuning.tune_gate_suppression_dynamics(
        *args,
        vad_probabilities=np.ones(100, dtype=np.float32),
        noise_vad_probabilities=np.zeros(100, dtype=np.float32),
        incumbent_settings={
            "gate": incumbent_gate,
            "suppressor": {"enabled": True, "strength": 1.0, "model": "rnnoise"},
        },
    )

    assert observed_gains == [2.0, 2.0]
    if not noise_available:
        assert not result["apply_recommended"]
        assert result["gate_settings"] == incumbent_gate


@pytest.mark.parametrize("with_neural_vad", [True, False])
def test_joint_downstream_uses_neural_vad_separately_from_energy_fallback(
    monkeypatch, with_neural_vad
):
    monkeypatch.setattr(joint_tuning, "_model_order", lambda _model: ["rnnoise"])
    args = list(_tuning_args())
    args[4] = {**args[4], "auto_makeup_enabled": True}
    observed: list[tuple[float, dict[str, Any]]] = []

    def simulate(audio, _probabilities, _before_gate, _strength, _settings):
        values = np.asarray(audio, dtype=np.float32)
        output = values * (0.5 if float(np.mean(np.abs(values))) < 0.01 else 1.0)
        return {
            "output_audio": output,
            "dry_audio": values,
            "suppressor_latency_samples": 480,
            "runtime_ms": 0.0,
            "auto_makeup_activity": _activity_tape(audio, _settings),
        }

    def downstream(audio, _sample_rate, _eq_settings, chain_settings):
        observed.append((float(np.mean(np.abs(audio))), dict(chain_settings)))
        return _downstream_metrics(audio)

    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate)
    monkeypatch.setattr(joint_tuning, "simulate_candidate_chain", downstream)
    speech_vad = np.linspace(0.1, 0.9, 100, dtype=np.float32)
    noise_vad = np.linspace(0.9, 0.1, 100, dtype=np.float32)
    kwargs: dict[str, Any] = {
        "incumbent_settings": {
            "gate": dict(args[3]),
            "suppressor": {"model": "rnnoise", "enabled": True, "strength": 1.0},
        },
    }
    if with_neural_vad:
        kwargs.update(
            vad_probabilities=speech_vad,
            noise_vad_probabilities=noise_vad,
        )

    joint_tuning.tune_gate_suppression_dynamics(*args, **kwargs)

    assert observed
    if not with_neural_vad:
        assert all("vad_probabilities" not in chain for _, chain in observed)
        assert all(
            all(row[1] == 0.0 for row in chain["auto_makeup_activity"])
            for _, chain in observed
        )
        return

    for mean, chain in observed:
        assert "vad_probabilities" not in chain
        assert all(row[1] == 1.0 for row in chain["auto_makeup_activity"])
        kind = "noise" if mean < 0.01 else "speech"
        expected_probability = 0.2 if kind == "noise" else 0.8
        assert all(
            row[0] == expected_probability
            for row in chain["auto_makeup_activity"]
        )


def test_joint_tuning_rejects_missing_auto_makeup_activity(monkeypatch):
    monkeypatch.setattr(joint_tuning, "_model_order", lambda _model: ["rnnoise"])
    args = _tuning_args()

    def simulate(audio, _probabilities, _before_gate, _strength, _settings):
        values = np.asarray(audio, dtype=np.float32)
        return {
            "output_audio": values,
            "dry_audio": values,
            "suppressor_latency_samples": 480,
            "runtime_ms": 0.0,
        }

    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate)
    monkeypatch.setattr(
        joint_tuning,
        "simulate_candidate_chain",
        lambda *_args, **_kwargs: pytest.fail("downstream must not run"),
    )
    result = joint_tuning.tune_gate_suppression_dynamics(
        *args,
        vad_probabilities=np.ones(100, dtype=np.float32),
        noise_vad_probabilities=np.zeros(100, dtype=np.float32),
        incumbent_settings={
            "gate": dict(args[3]),
            "suppressor": {"model": "rnnoise", "enabled": True, "strength": 1.0},
        },
    )

    assert not result["apply_recommended"]
    assert "auto_makeup_activity" in result["reason"]


def test_deepfilter_incumbent_keeps_selection_inside_deepfilter_family(monkeypatch):
    preset = Preset()
    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: None)
    result = joint_tuning.tune_gate_suppression_dynamics(
        np.full(48_000, 0.001, dtype=np.float32),
        np.full(96_000, 0.1, dtype=np.float32),
        48_000,
        asdict(preset.gate),
        asdict(preset.compressor),
        {
            "band_freqs": preset.eq.band_freqs,
            "band_gains": preset.eq.band_gains,
            "band_qs": preset.eq.band_qs,
        },
        asdict(preset.deesser),
        asdict(preset.limiter),
        incumbent_settings={
            "gate": asdict(preset.gate),
            "suppressor": {
                "enabled": True,
                "strength": 0.62,
                "model": "deepfilter-ll",
            },
        },
        noise_model="deepfilter-ll",
    )
    assert [entry["model"] for entry in result["model_evaluations"]] == [
        "deepfilter-ll",
        "deepfilter",
    ]
    policy = result["model_selection_policy"]
    assert policy["id"] == "retain_deepfilter_family_v1"
    assert policy["excluded_models"] == ["rnnoise"]


def test_latency_gate_accepts_the_configured_35_ms_boundary():
    metrics = {
        "speech_retained_ratio": 0.90,
        "false_closure_rate": 0.0,
        "tail_retained_ratio": 0.60,
        "noise_attenuation_db": 6.0,
        "chatter_events": 0.0,
        "compressor_p95_db": 3.5,
        "compressor_target_p95_db": 3.5,
        "compressor_peak_db": 4.0,
        "compressor_peak_cap_db": 8.0,
        "output_true_peak_db": -1.0,
        "limiter_ceiling_db": -0.5,
        "pre_limiter_headroom_db": 1.0,
        "non_finite_output": 0.0,
        "runtime_factor": 0.5,
        "suppressor_latency_ms": 35.0,
        "max_suppressor_latency_ms": 35.0,
    }
    assert joint_tuning._gates(metrics)["suppressor_latency"]


def test_evaluator_uses_actual_selected_model_and_canonicalizes_alias():
    gate = {"threshold_db": -40.0}
    selected_gate, model, strength = _selected_settings(
        {
            "gate_settings": gate,
            "suppressor_settings": {
                "model": "deepfilternet-ll",
                "strength": 0.47,
            },
        },
        incumbent_model="rnnoise",
        incumbent_strength=1.0,
    )
    assert selected_gate == gate
    assert model == "deepfilter-ll"
    assert strength == 0.47


def _synthetic_product_report() -> dict[str, Any]:
    return {
        "corpus_manifest_sha256": "e" * 64,
        "implementation_sha256": {
            path: hashlib.sha256((REPO_ROOT / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for path in ("python/tools/evaluate_product_tuning.py", "release-assets.json")
        },
        "selection_split": "per-capture first half selects among eligible models",
        "incumbent_model": "rnnoise",
        "candidate_models": ["rnnoise", "deepfilter-ll"],
        "model_selection_policy": {"id": "all_supported_models_v1"},
        "max_suppressor_latency_ms": 35.0,
        "case_count": 1,
        "all_gates_passed": True,
        "cases": [
            {
                "id": "capture-1",
                "threshold_db": -40.0,
                "selected_latency_ms": 10.0,
                "runtime_ms": 12.5,
                "clean_error_increase": -0.01,
                "gates": {"clean_speech_preserved": True, "selected_latency_within_35ms": True},
            }
        ],
    }


def test_evaluator_clean_comparison_obeys_native_activity_contract(tmp_path, monkeypatch):
    from scipy.io.wavfile import write

    evidence_paths = (
        "python/tools/evaluate_product_tuning.py",
        "python/mic_eq/analysis/joint_tuning.py",
        "rust-core/src/audio/processor/python_api.rs",
        "release-assets.json",
    )
    for relative in evidence_paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"native-contract fixture\n")
    (tmp_path / "df.dll").write_bytes(b"native-contract fixture\n")
    monkeypatch.setattr(evaluate_product_tuning, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        evaluate_product_tuning,
        "__file__",
        str(tmp_path / evidence_paths[0]),
    )

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    clean = (0.05 * np.sin(2 * np.pi * 220 * np.arange(192_000) / 48_000)).astype(np.float32)
    write(corpus / "clean.wav", 48_000, clean)
    write(corpus / "noisy.wav", 48_000, clean + np.float32(0.001))
    (corpus / "manifest.json").write_text(json.dumps({"captures": [{
        "id": "native-contract", "duration_seconds": 4.0,
        "clean": {"path": "clean.wav"}, "noisy": {"path": "noisy.wav"},
    }]}), encoding="utf-8")

    def retain_incumbent(*_args, incumbent_settings, **_kwargs):
        return {
            "gate_settings": incumbent_settings["gate"],
            "suppressor_settings": incumbent_settings["suppressor"],
            "apply_recommended": False, "decision": "retain_incumbent", "runtime_ms": 0.0,
        }

    monkeypatch.setattr(evaluate_product_tuning, "tune_gate_suppression_dynamics", retain_incumbent)
    # Keep the real helper and native renderer: omitted activity requests used
    # to fail here only after the expensive candidate search had completed.
    report = evaluate_product_tuning.evaluate(corpus, "rnnoise")

    assert report["case_count"] == 2
    assert all(case["gates"]["clean_speech_preserved"] for case in report["cases"])
    assert all(case["gates"]["heldout_or_exact_retention"] for case in report["cases"])


def test_evaluator_hashes_mixed_line_endings_portably(tmp_path, monkeypatch):
    from mic_eq.ui import app_bootstrap

    paths = (
        "python/tools/evaluate_product_tuning.py",
        "python/mic_eq/analysis/joint_tuning.py",
        "rust-core/src/audio/processor/python_api.rs",
        "release-assets.json",
    )
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"first\r\nsecond\n")
    (tmp_path / "df.dll").write_bytes(b"binary\r\nidentity\n")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "manifest.json").write_text('{"captures": []}', encoding="utf-8")
    monkeypatch.setattr(evaluate_product_tuning, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(evaluate_product_tuning, "__file__", str(tmp_path / paths[0]))
    monkeypatch.setattr(app_bootstrap, "configure_deepfilter_env", lambda: None)

    report = evaluate_product_tuning.evaluate(corpus, "rnnoise")

    expected = hashlib.sha256(b"first\nsecond\n").hexdigest()
    assert all(report["implementation_sha256"][path] == expected for path in paths)
    assert report["implementation_sha256"]["df.dll"] == hashlib.sha256(
        b"binary\r\nidentity\n"
    ).hexdigest()


@pytest.mark.parametrize("revision", [None, "f" * 40])
def test_evaluator_report_finalizer_emits_auditable_contract_without_changing_metrics(
    monkeypatch, revision
):
    monkeypatch.setattr(evaluate_product_tuning, "_git_revision", lambda _root: revision)
    report = _synthetic_product_report()
    implementation_hashes = report["implementation_sha256"]
    finalized = _finalize_report(
        report,
        corpus=REPO_ROOT / "models" / "test-report-corpus",
    )

    assert finalized["schema_version"] == 2
    assert finalized["audible_change"] is True
    assert finalized["implementation_sha256"] == implementation_hashes
    assert finalized["measurement_implementation_sha256"] == implementation_hashes
    assert finalized["report_formatting"]["metadata_only"] is False
    assert finalized["evaluation_contract"]["configuration"]["case_count"] == 1
    assert finalized["evaluation_contract"]["configuration"]["capture_count"] == 1
    assert finalized["evaluation_contract"]["latency"]["max_selected_suppressor_latency_ms"] == 10.0
    assert "source_revision" not in finalized
    if revision is None:
        assert "measurement_source_revision" not in finalized
    else:
        assert finalized["measurement_source_revision"] == revision


def test_evaluator_revision_is_omitted_for_dirty_worktree(monkeypatch):
    def fake_run(command, **_kwargs):
        assert command[1:] == ["status", "--porcelain", "--untracked-files=no"]
        return SimpleNamespace(returncode=0, stdout=" M python/tools/evaluate_product_tuning.py")

    monkeypatch.setattr(evaluate_product_tuning.subprocess, "run", fake_run)

    assert _git_revision(REPO_ROOT) is None


@pytest.mark.parametrize(
    ("model", "explicit_output", "keep_details"),
    [("rnnoise", True, True), ("rnnoise", False, False),
     ("deepfilter", False, True), ("deepfilter-ll", False, False)],
)
def test_evaluator_cli_writes_the_auditable_contract(
    tmp_path, monkeypatch, model, explicit_output, keep_details
):
    monkeypatch.chdir(tmp_path)

    def evaluate(_corpus, selected_model):
        assert selected_model == model
        return _synthetic_product_report()

    monkeypatch.setattr(
        evaluate_product_tuning,
        "evaluate",
        evaluate,
    )
    suffix = "" if model == "rnnoise" else f"-{model}"
    output = tmp_path / (
        "product-report.json" if explicit_output
        else f"evaluation/product-joint-tuning{suffix}.json"
    )
    details = tmp_path / "details" / "cases.json"
    arguments = ["--corpus", str(REPO_ROOT / "models" / "test-report-corpus"),
                 "--model", model]
    if explicit_output:
        arguments.extend(["--output", str(output)])
    if keep_details:
        arguments.extend(["--details-output", str(details)])

    assert evaluate_product_tuning.main(arguments) == 0
    written = json.loads(output.read_text(encoding="utf-8"))

    assert "cases" not in written
    assert details.exists() == keep_details
    if keep_details:
        detailed = json.loads(details.read_text(encoding="utf-8"))
        assert detailed.pop("cases") == _synthetic_product_report()["cases"]
        assert detailed == written
    assert list(tmp_path.glob("evaluation/*.json")) == ([] if explicit_output else [output])
    assert written["audible_change"] is True
    assert written["evaluation_contract"]["runtime"]["max_p99_frame_seconds"] is None
    assert written["report_formatting"]["metadata_only"] is False
    assert validate_report(output) == []


def test_evaluator_rejects_overlapping_report_outputs_before_evaluation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evaluate_product_tuning, "evaluate", lambda *_: pytest.fail("must not evaluate"))
    with pytest.raises(SystemExit) as error:
        evaluate_product_tuning.main(
            ["--output", "report.json", "--details-output", "./report.json"]
        )
    assert error.value.code == 2
    assert not (tmp_path / "report.json").exists()


def test_training_noise_floor_does_not_use_heldout_noise(monkeypatch):
    preset = Preset()
    gate = {
        **asdict(preset.gate),
        "gate_mode": 1,
        "auto_threshold_enabled": True,
        "gate_margin_db": 10.0,
    }
    speech = np.full(48_000, 0.1, dtype=np.float32)
    training_noise = np.full(24_000, 0.001, dtype=np.float32)
    noise_a = np.concatenate((training_noise, np.full(24_000, 0.002, dtype=np.float32)))
    noise_b = np.concatenate((training_noise, np.full(24_000, 0.2, dtype=np.float32)))
    observed_thresholds: list[float] = []

    def simulate(audio, _probabilities, _before_gate, _strength, settings):
        observed_thresholds.append(float(settings["gate_threshold_db"]))
        output = np.asarray(audio, dtype=np.float32)
        return {
            "output_audio": output,
            "dry_audio": output,
            "suppressor_latency_samples": 480,
            "runtime_ms": 0.0,
            "auto_makeup_activity": _activity_tape(audio, settings),
        }

    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate)
    monkeypatch.setattr(joint_tuning, "simulate_candidate_chain", _downstream_metrics)

    def run(noise: np.ndarray) -> list[float]:
        observed_thresholds.clear()
        joint_tuning.tune_gate_suppression_dynamics(
            noise,
            speech,
            48_000,
            gate,
            asdict(preset.compressor),
            {
                "band_freqs": preset.eq.band_freqs,
                "band_gains": preset.eq.band_gains,
                "band_qs": preset.eq.band_qs,
            },
            asdict(preset.deesser),
            asdict(preset.limiter),
            vad_probabilities=np.ones(100, dtype=np.float32),
            noise_vad_probabilities=np.zeros(100, dtype=np.float32),
            incumbent_settings={
                "gate": gate,
                "suppressor": {"enabled": True, "strength": 1.0, "model": "rnnoise"},
            },
        )
        return list(observed_thresholds)

    thresholds_a = run(noise_a)
    thresholds_b = run(noise_b)
    assert thresholds_a
    assert thresholds_a == thresholds_b
    assert {round(threshold, 5) for threshold in thresholds_a} <= {-53.0, -50.0, -47.0}
