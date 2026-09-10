"""Pure checks for the isolated ONNX Runtime probe evaluator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import evaluate_onnxruntime_probe as probe


def test_equal_posteriors_pass_all_decision_checks() -> None:
    values = np.asarray([0.1, 0.48, 0.9], dtype=np.float64)

    result = probe._compare_posteriors({"arr_0": values}, {"arr_0": values.copy()})

    assert result["frame_count"] == 3
    assert result["exact_equal_frames"] == 3
    assert result["max_abs_delta"] == 0.0
    assert all(change == 0 for change in result["threshold_changes"].values())
    assert result["hysteresis_surrogate"]["decision_changes"] == 0


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError, match="no captures"):
        probe._compare_posteriors({}, {})


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape differs"):
        probe._compare_posteriors(
            {"arr_0": np.zeros(2)},
            {"arr_0": np.zeros(3)},
        )


def test_probe_records_and_accepts_distinct_interpreters(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline-python.exe"
    candidate = tmp_path / "candidate-python.exe"
    arguments = probe._parser().parse_args(
        [
            "--baseline-python",
            str(baseline),
            "--candidate-python",
            str(candidate),
            "--manifest",
            "manifest.json",
            "--model",
            "model.onnx",
            "--throughput-audio",
            "audio.wav",
        ]
    )

    assert arguments.baseline_python == baseline
    assert arguments.candidate_python == candidate
    identity = probe._runtime_identity()
    assert identity["executable"]
    assert identity["implementation"]
    assert identity["version_info"] == [
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro,
    ]
    assert not Path(identity["executable"]).is_absolute()


def test_reproduction_command_uses_current_portable_inputs() -> None:
    arguments = probe._parser().parse_args(
        [
            "--baseline-python-root",
            "python",
            "--candidate-python-root",
            "python",
            "--baseline-dll-root",
            "target/release",
            "--candidate-dll-root",
            "target/onnxruntime-cpu/lib",
            "--manifest",
            "models/vad_eval_corpus/manifest.json",
            "--model",
            "models/silero_vad.onnx",
            "--throughput-audio",
            "models/vad_eval_silero_test.wav",
            "--cpu-archive",
            "target/onnxruntime-cpu-probe/onnxruntime-win-x64-1.23.2.zip",
            "--candidate-runtime-file",
            "target/onnxruntime-cpu/lib/onnxruntime.dll",
            "--report",
            "evaluation/onnxruntime-cpu-probe.json",
        ]
    )

    command = probe._reproduction_command(
        arguments,
        baseline_python=Path(".venv/Scripts/python.exe"),
        candidate_python=Path(".venv313/Scripts/python.exe"),
        manifest=Path("models/vad_eval_corpus/manifest.json"),
        model=Path("models/silero_vad.onnx"),
        throughput_audio=Path("models/vad_eval_silero_test.wav"),
    )

    assert "--candidate-python-root python" in command
    assert "target/onnxruntime-cpu-probe/candidate313" not in command
    assert "C:\\Users\\" not in command


def test_hardware_and_jitter_flags_do_not_mask_software_gate_failure() -> None:
    gates = {name: True for name in probe.SOFTWARE_GATE_NAMES}
    gates.update(
        {
            "per_frame_jitter_or_deadline_measured": False,
            "hardware_qualification_measured": False,
        }
    )
    assert probe._software_gates_pass(gates)

    gates["max_abs_delta_le_1e-5"] = False
    assert not probe._software_gates_pass(gates)


def test_candidate_runtime_files_are_bound_to_pinned_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    paths: list[Path] = []
    assets: list[dict[str, object]] = []
    archive_sha256 = "a" * 64
    for index, name in enumerate(
        (
            "onnxruntime.dll",
            "onnxruntime.lib",
            "onnxruntime_providers_shared.dll",
        ),
        start=1,
    ):
        path = tmp_path / name
        payload = b"x" * index
        path.write_bytes(payload)
        digest = probe.hashlib.sha256(path.read_bytes()).hexdigest()
        assets.append(
            {
                "path": f"target/onnxruntime-cpu/lib/{name}",
                "size": len(payload),
                "sha256": digest,
                "source": "https://github.com/example/runtime.zip",
                "origin": {
                    "status": "verified-upstream-archive",
                    "archive_sha256": archive_sha256,
                    "archive_size": 123,
                    "archive_member": f"package/{name}",
                    "runtime": "CPU-only Windows x64",
                },
            }
        )
        paths.append(path)
    (tmp_path / "release-assets.json").write_text(
        json.dumps({"assets": assets}), encoding="utf-8"
    )
    monkeypatch.setattr(probe, "REPO_ROOT", tmp_path)

    assert len(probe._record_cpu_ort_runtime_files(paths)) == 3
    with pytest.raises(ValueError, match="missing pinned CPU ORT assets"):
        probe._record_cpu_ort_runtime_files(paths[:2])
