"""Evaluate the joint tuner on existing held-out clean/noisy speech pairs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from mic_eq.analysis.joint_tuning import (
    _DEFAULT_MAX_SUPPRESSOR_LATENCY_MS,
    _SUPPORTED_NOISE_MODELS,
    _candidate_settings,
    _canonical_noise_model,
    _energy_probabilities,
    _model_selection_metadata,
    _run_gate_suppressor,
    tune_gate_suppression_dynamics,
)
from mic_eq.analysis.wav_io import read_mono_wav
from mic_eq.config import Preset
from mic_eq.mic_eq_core import simulate_gate_suppressor_order

_MAX_SUPPRESSOR_LATENCY_MS = _DEFAULT_MAX_SUPPRESSOR_LATENCY_MS


def _selected_settings(
    result: Mapping[str, object],
    *,
    incumbent_model: str,
    incumbent_strength: float,
) -> tuple[dict[str, object], str, float]:
    selected_gate = result.get("gate_settings")
    selected_suppressor = result.get("suppressor_settings")
    if not isinstance(selected_gate, Mapping) or not isinstance(
        selected_suppressor, Mapping
    ):
        raise ValueError("joint tuner returned incomplete selected settings")
    selected_model = _canonical_noise_model(
        selected_suppressor.get("model", incumbent_model)
    )
    if selected_model not in _SUPPORTED_NOISE_MODELS:
        raise ValueError(f"joint tuner returned unsupported model: {selected_model}")
    try:
        selected_strength = float(
            selected_suppressor.get("strength", incumbent_strength)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("joint tuner returned invalid suppressor strength") from exc
    if not np.isfinite(selected_strength) or not 0.0 <= selected_strength <= 1.0:
        raise ValueError("joint tuner returned invalid suppressor strength")
    return dict(selected_gate), selected_model, selected_strength


def evaluate(corpus: Path, model: str) -> dict:
    incumbent_model = _canonical_noise_model(model)
    if incumbent_model not in _SUPPORTED_NOISE_MODELS:
        raise ValueError(f"unsupported incumbent model: {model}")
    selection_policy = _model_selection_metadata(incumbent_model)
    # The tuner evaluates every available backend, so DeepFilter assets must be
    # configured even when RNNoise is the incumbent being measured.
    from mic_eq.ui.app_bootstrap import configure_deepfilter_env

    configure_deepfilter_env()
    manifest = json.loads((corpus / "manifest.json").read_text())
    captures = [case for case in manifest["captures"] if case["duration_seconds"] >= 4.0]
    # Predefined gates: clean speech preservation, finite/non-clipping output,
    # selected-model latency <=35 ms, sub-realtime cost, and exact incumbent
    # retention whenever the held-out evidence is inconclusive.
    cases = []
    preset = Preset()
    eq = {"band_freqs": preset.eq.band_freqs, "band_gains": preset.eq.band_gains,
          "band_qs": preset.eq.band_qs}
    for capture in captures[:12]:
        rate, clean = read_mono_wav(corpus / capture["clean"]["path"])
        _, noisy = read_mono_wav(corpus / capture["noisy"]["path"])
        if rate != 48000:
            raise ValueError("Evaluation requires the pinned 48 kHz corpus")
        noise = noisy - clean
        # Use a deliberately restrictive incumbent to exercise corrective choices,
        # and the normal default to verify conservative retention.
        for threshold in (-40.0, -22.0):
            gate = {**asdict(preset.gate), "threshold_db": threshold}
            incumbent = {
                "gate": gate,
                "suppressor": {
                    "enabled": True,
                    "strength": 1.0,
                    "model": incumbent_model,
                },
            }
            result = tune_gate_suppression_dynamics(
                noise, noisy, rate, asdict(preset.gate), asdict(preset.compressor),
                eq,
                asdict(preset.deesser),
                asdict(preset.limiter),
                incumbent_settings=incumbent,
                noise_model=incumbent_model,
                max_suppressor_latency_ms=_MAX_SUPPRESSOR_LATENCY_MS,
            )
            selected_gate, selected_model, selected_strength = _selected_settings(
                result,
                incumbent_model=incumbent_model,
                incumbent_strength=1.0,
            )
            noise_floor = float(20 * np.log10(max(float(np.sqrt(np.mean(noise**2))), 1e-12)))
            probabilities = _energy_probabilities(clean, noise_floor)
            outputs = []
            for render_gate, strength, render_model in (
                (gate, 1.0, incumbent_model),
                (selected_gate, selected_strength, selected_model),
            ):
                _, controls = _candidate_settings(
                    render_gate,
                    noise_floor_db=noise_floor,
                    strength=strength,
                    threshold_delta_db=0.0,
                    release_delta_ms=0.0,
                )
                outputs.append(_run_gate_suppressor(simulate_gate_suppressor_order,
                    clean,
                    probabilities,
                    controls,
                    suppressor_strength=strength,
                    noise_model=render_model,
                ))
            before, after = (np.asarray(output["output_audio"]) for output in outputs)
            reference_energy = max(float(np.mean(clean**2)), 1e-12)
            degradation = float((np.mean((after-clean)**2) - np.mean((before-clean)**2)) / reference_energy)
            selected_latency_ms = (
                float(outputs[1]["suppressor_latency_samples"]) * 1000.0 / rate
            )
            exact_incumbent = bool(
                not result.get("apply_recommended")
                and dict(result.get("gate_settings") or {}) == gate
                and selected_model == incumbent_model
                and abs(selected_strength - 1.0) <= 1.0e-9
            )
            gates = {
                "clean_speech_preserved": degradation <= 0.01,
                "finite_output": bool(np.isfinite(before).all() and np.isfinite(after).all()),
                "no_clipping": bool(
                    np.max(np.abs(before)) <= 1.0 and np.max(np.abs(after)) <= 1.0
                ),
                "selected_latency_within_35ms": selected_latency_ms <= _MAX_SUPPRESSOR_LATENCY_MS,
                "sub_realtime": float(outputs[1]["runtime_ms"]) < len(clean) / rate * 1000,
                "heldout_or_exact_retention": bool(result.get("holdout_passed"))
                if result.get("apply_recommended")
                else exact_incumbent,
            }
            cases.append(
                {
                    "id": capture["id"],
                    "threshold_db": threshold,
                    "incumbent_model": incumbent_model,
                    "selected_model": selected_model,
                    "selected_strength": selected_strength,
                    "decision": result["decision"],
                    "model_selection_policy": selection_policy["id"],
                    "selected_latency_ms": selected_latency_ms,
                    "gates": gates,
                    "clean_error_increase": degradation,
                    "runtime_ms": result["runtime_ms"],
                }
            )
            print(capture["id"], threshold, result["decision"], all(gates.values()), flush=True)
    root = Path(__file__).resolve().parents[2]
    baseline_path = (
        root
        / "models"
        / "evaluation-details"
        / f"product-joint-tuning-{incumbent_model}-unrestricted.json"
    )
    previous_unrestricted_baseline = None
    if baseline_path.is_file():
        previous = json.loads(baseline_path.read_text())
        failed_cases = [
            case
            for case in previous.get("cases", [])
            if not all(bool(value) for value in case.get("gates", {}).values())
        ]
        failed_gates = sorted(
            {
                str(gate_name)
                for case in failed_cases
                for gate_name, passed in case.get("gates", {}).items()
                if not passed
            }
        )
        previous_unrestricted_baseline = {
            "report": baseline_path.relative_to(root).as_posix(),
            "all_gates_passed": bool(previous.get("all_gates_passed", False)),
            "failed_case_count": len(failed_cases),
            "failed_gates": failed_gates,
            "historical_source_hashes": dict(
                previous.get("implementation_sha256", {})
            ),
            "source_snapshot_available": False,
            "reason": (
                "preserved before applying the directional DeepFilter-family "
                "retention policy; the historical source snapshot is unavailable"
            ),
        }
    evidence_files = [Path(__file__), root / "python/mic_eq/analysis/joint_tuning.py",
                      root / "rust-core/src/audio/processor/python_api.rs",
                      *sorted((root / "python/mic_eq").glob("mic_eq_core*.pyd"))]
    evidence_files.extend((root / "df.dll", root / "release-assets.json"))
    return {"corpus_manifest_sha256": hashlib.sha256((corpus / "manifest.json").read_bytes()).hexdigest(),
            "implementation_sha256": {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in evidence_files},
            "selection_split": "first half selects among all available models; second half must pass all gates and improve on incumbent; incumbent settings are retained exactly when evidence is inconclusive; selected suppressor latency must be <=35 ms; fallback noise floor comes from training noise only, while explicit noise_floor_db is precomputed context",
            "model": incumbent_model,
            "incumbent_model": incumbent_model,
            "candidate_models": list(selection_policy["candidate_models"]),
            "model_selection_policy": selection_policy,
            "max_suppressor_latency_ms": _MAX_SUPPRESSOR_LATENCY_MS,
            "case_count": len(cases),
            "all_gates_passed": bool(cases) and all(all(case["gates"].values()) for case in cases),
            "applied_candidates": sum(case["decision"] == "apply_candidate" for case in cases),
            "cases": cases,
            **(
                {"previous_unrestricted_baseline": previous_unrestricted_baseline}
                if previous_unrestricted_baseline is not None
                else {}
            )}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("models/deepfilter_fullband_eval"))
    parser.add_argument("--model", default="rnnoise", choices=("rnnoise", "deepfilter", "deepfilter-ll"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/product-joint-tuning.json"))
    args = parser.parse_args()
    report = evaluate(args.corpus, args.model)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["all_gates_passed"] else 1)
