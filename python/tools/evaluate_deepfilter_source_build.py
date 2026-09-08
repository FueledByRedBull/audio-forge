"""Compare a source-built DeepFilter DLL with the inherited runtime artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "models" / "deepfilter_fullband_eval"
DEFAULT_BENCHMARK = REPO_ROOT / "target" / "release" / "deepfilter_benchmark.exe"
DEFAULT_REFERENCE = REPO_ROOT / "target" / "deepfilter" / "df-inherited-before-adoption.dll"
DEFAULT_CANDIDATE = REPO_ROOT / "df.dll"
DEFAULT_MODEL_DIR = REPO_ROOT / "models"
DEFAULT_ATTESTATION = REPO_ROOT / "target" / "deepfilter" / "df.dll.provenance.json"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "deepfilter-source-build-report.json"
PROVENANCE_PATH = REPO_ROOT / "build-support" / "deepfilter" / "provenance.json"
REFERENCE_SHA256 = "270d14b92143c18c5f7e14e9eba3192d7843ec52fb9d1ade2fd9173d03a71656"
SAMPLE_RATE = 48_000
SLICE_FRAMES = 48_000
REPEATS = 5
SEGMENTS = (
    ("p232_001", 0),
    ("p232_006", 24_000),
    ("p257_001", 0),
    ("p257_006", 24_000),
)
MODELS = (("ll", 480), ("standard", 1_440))
P99_GATE_MS = 10.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _read_manifest(corpus: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    captures = raw.get("captures")
    if not isinstance(captures, list):
        raise ValueError("full-band corpus manifest must contain captures")
    indexed: dict[str, dict[str, Any]] = {}
    for capture in captures:
        if not isinstance(capture, dict) or not isinstance(capture.get("id"), str):
            raise ValueError("full-band corpus manifest has an invalid capture")
        indexed[capture["id"]] = capture
    return indexed


def _read_provenance() -> dict[str, Any]:
    raw = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("DeepFilter provenance must be a JSON object")
    return raw


def _validate_attestation(
    attestation: Path,
    candidate_library: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Bind the measured candidate to the source identity claimed in its attestation."""
    if not attestation.is_file():
        raise FileNotFoundError(f"DeepFilter build attestation is missing: {attestation}")
    raw = json.loads(attestation.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("DeepFilter build attestation must be a JSON object")
    if raw.get("schema_version") != 1 or raw.get("kind") != "audioforge.deepfilter.build":
        raise ValueError("DeepFilter build attestation has an unsupported schema or kind")
    output = raw.get("output")
    if not isinstance(output, dict):
        raise ValueError("DeepFilter build attestation has no output object")
    output_sha256 = output.get("sha256")
    output_bytes = output.get("bytes")
    if not isinstance(output_sha256, str) or len(output_sha256) != 64:
        raise ValueError("DeepFilter build attestation output.sha256 is invalid")
    if type(output_bytes) is not int or output_bytes <= 0:
        raise ValueError("DeepFilter build attestation output.bytes is invalid")
    candidate_actual_sha256 = _sha256(candidate_library)
    if output_sha256.lower() != candidate_actual_sha256.lower():
        raise ValueError(
            "candidate DLL does not match attestation output SHA-256: "
            f"expected {output_sha256}, got {candidate_actual_sha256}"
        )
    candidate_bytes = candidate_library.stat().st_size
    if output_bytes != candidate_bytes:
        raise ValueError(
            "candidate DLL does not match attestation output size: "
            f"expected {output_bytes}, got {candidate_bytes}"
        )

    source = raw.get("source")
    upstream = provenance.get("upstream")
    if not isinstance(source, dict) or not isinstance(upstream, dict):
        raise ValueError("DeepFilter attestation/provenance source identity is incomplete")
    if source.get("repository") != upstream.get("repository"):
        raise ValueError("DeepFilter attestation repository does not match pinned provenance")
    if source.get("commit") != upstream.get("commit"):
        raise ValueError("DeepFilter attestation commit does not match pinned provenance")
    return raw


def _load_slice(corpus: Path, captures: dict[str, dict[str, Any]], identifier: str, offset: int) -> tuple[Path, np.ndarray, dict[str, Any]]:
    capture = captures.get(identifier)
    if capture is None:
        raise ValueError(f"corpus manifest is missing {identifier}")
    noisy = capture.get("noisy")
    if not isinstance(noisy, dict) or not isinstance(noisy.get("path"), str):
        raise ValueError(f"corpus manifest has no noisy path for {identifier}")
    relative = Path(noisy["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe noisy corpus path: {relative}")
    path = (corpus / relative).resolve(strict=True)
    if not path.is_relative_to(corpus.resolve()):
        raise ValueError(f"noisy corpus path escapes corpus: {relative}")
    expected_hash = noisy.get("sha256")
    if not isinstance(expected_hash, str) or _sha256(path).lower() != expected_hash.lower():
        raise ValueError(f"noisy corpus hash mismatch: {relative}")
    sample_rate, audio = wavfile.read(path)
    if sample_rate != SAMPLE_RATE or audio.ndim != 1 or audio.dtype.kind not in "iu":
        raise ValueError(f"{relative} must be mono integer PCM at 48 kHz")
    if audio.size < offset + SLICE_FRAMES:
        raise ValueError(f"{relative} is too short for offset {offset}")
    info = np.iinfo(audio.dtype)
    scale = float(max(abs(int(info.min)), int(info.max)))
    samples = np.asarray(audio[offset : offset + SLICE_FRAMES], dtype=np.float64) / scale
    samples = np.asarray(samples, dtype="<f4")
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"{relative} produced non-finite input samples")
    return path, samples, {"path": relative.as_posix(), "sha256": expected_hash, "offset": offset}


def _run_backend(
    benchmark: Path,
    library: Path,
    model_dir: Path,
    model: str,
    input_path: Path,
    output_path: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    command = [
        str(benchmark),
        "--model",
        model,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--library",
        str(library),
        "--model-dir",
        str(model_dir),
        "--attenuation-db",
        "80",
        "--post-filter-beta",
        "0",
        "--strength",
        "1",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    json_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    if not json_lines:
        raise RuntimeError(f"benchmark emitted no JSON for {model}")
    metadata = json.loads(json_lines[-1])
    output = np.fromfile(output_path, dtype="<f4")
    return metadata, output


def _comparison(reference: dict[str, Any], candidate: dict[str, Any], reference_output: np.ndarray, candidate_output: np.ndarray) -> dict[str, Any]:
    if reference_output.size != candidate_output.size:
        raise RuntimeError("reference and candidate output lengths differ")
    finite = bool(np.all(np.isfinite(reference_output)) and np.all(np.isfinite(candidate_output)))
    if finite:
        error = candidate_output.astype(np.float64) - reference_output.astype(np.float64)
        max_abs = float(np.max(np.abs(error)))
        rmse = float(np.sqrt(np.mean(np.square(error))))
    else:
        max_abs = float("inf")
        rmse = float("inf")
    return {
        "samples": int(candidate_output.size),
        "finite": finite,
        "reference_latency_samples": int(reference["latency_samples"]),
        "candidate_latency_samples": int(candidate["latency_samples"]),
        "max_abs": max_abs,
        "rmse": rmse,
        "reference_p99_frame_ms": float(reference["p99_frame_seconds"]) * 1000.0,
        "candidate_p99_frame_ms": float(candidate["p99_frame_seconds"]) * 1000.0,
    }


def evaluate(
    *,
    corpus: Path,
    benchmark: Path,
    reference_library: Path,
    candidate_library: Path,
    model_dir: Path,
    attestation: Path,
    reference_sha256: str,
    candidate_sha256: str | None,
    output: Path,
) -> dict[str, Any]:
    for path in (benchmark, reference_library, candidate_library):
        if not path.is_file():
            raise FileNotFoundError(path)
    provenance = _read_provenance()
    upstream = provenance.get("upstream")
    if not isinstance(upstream, dict):
        raise ValueError("DeepFilter provenance has no upstream source identity")
    attestation_record = _validate_attestation(attestation, candidate_library, provenance)
    reference_actual_sha256 = _sha256(reference_library)
    if reference_actual_sha256.lower() != reference_sha256.lower():
        raise ValueError(
            f"reference DLL SHA-256 mismatch: expected {reference_sha256}, got {reference_actual_sha256}"
        )
    candidate_actual_sha256 = _sha256(candidate_library)
    if candidate_sha256 and candidate_actual_sha256.lower() != candidate_sha256.lower():
        raise ValueError(
            f"candidate DLL SHA-256 mismatch: expected {candidate_sha256}, got {candidate_actual_sha256}"
        )
    corpus = corpus.resolve(strict=True)
    benchmark = benchmark.resolve(strict=True)
    reference_library = reference_library.resolve(strict=True)
    candidate_library = candidate_library.resolve(strict=True)
    model_dir = model_dir.resolve(strict=True)
    captures = _read_manifest(corpus)
    manifest_path = corpus / "manifest.json"
    source_hashes = {
        _relative(path): _sha256(path)
        for path in (
            Path(__file__).resolve(),
            REPO_ROOT / "rust-core/src/bin/deepfilter_benchmark.rs",
            REPO_ROOT / "rust-core/src/dsp/deepfilter_ffi.rs",
        )
    }
    benchmark_hashes = {
        "binary": {_relative(benchmark): _sha256(benchmark)},
        "corpus_manifest": {_relative(manifest_path): _sha256(manifest_path)},
    }
    model_hashes = {
        name: _sha256(model_dir / name)
        for name in ("DeepFilterNet3_ll_onnx.tar.gz", "DeepFilterNet3_onnx.tar.gz")
    }
    expected_models = provenance.get("models")
    if not isinstance(expected_models, dict):
        raise ValueError("DeepFilter provenance has no model hashes")
    for name, expected in expected_models.items():
        if not isinstance(expected, dict) or not isinstance(expected.get("sha256"), str):
            raise ValueError(f"DeepFilter provenance has no valid hash for {name}")
        model_path = model_dir / Path(name).name
        if model_path.name not in model_hashes:
            raise ValueError(f"DeepFilter evaluator does not cover model {name}")
        if model_hashes[model_path.name].lower() != expected["sha256"].lower():
            raise ValueError(f"DeepFilter model hash mismatch: {model_path.name}")
    comparisons: list[dict[str, Any]] = []
    runtime: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="audioforge-deepfilter-source-build-") as temporary_name:
        temporary = Path(temporary_name)
        for model, expected_latency in MODELS:
            for identifier, offset in SEGMENTS:
                source_path, samples, source = _load_slice(corpus, captures, identifier, offset)
                input_path = temporary / f"{model}-{identifier}-{offset}.f32"
                input_path.write_bytes(samples.tobytes())
                source["input_sha256"] = _sha256(input_path)
                reference_output_path = temporary / f"reference-{model}-{identifier}-{offset}.f32"
                candidate_output_path = temporary / f"candidate-{model}-{identifier}-{offset}.f32"
                reference_metadata, reference_output = _run_backend(
                    benchmark, reference_library, model_dir, model, input_path, reference_output_path
                )
                candidate_metadata, candidate_output = _run_backend(
                    benchmark, candidate_library, model_dir, model, input_path, candidate_output_path
                )
                if reference_metadata["latency_samples"] != expected_latency or candidate_metadata["latency_samples"] != expected_latency:
                    raise RuntimeError(f"unexpected {model} latency for {identifier}@{offset}")
                comparison = _comparison(reference_metadata, candidate_metadata, reference_output, candidate_output)
                comparison.update(
                    {
                        "model": model,
                        "segment": f"{identifier}@{offset}",
                        "expected_latency_samples": expected_latency,
                        "input": source,
                    }
                )
                comparisons.append(comparison)

            _, repeat_samples, _ = _load_slice(corpus, captures, "p257_006", 24_000)
            repeat_input = temporary / f"{model}-runtime-repeat.f32"
            repeat_input.write_bytes(repeat_samples.tobytes())
            for repeat in range(1, REPEATS + 1):
                repeat_output = temporary / f"candidate-{model}-runtime-{repeat}.f32"
                metadata, samples = _run_backend(
                    benchmark, candidate_library, model_dir, model, repeat_input, repeat_output
                )
                if metadata["latency_samples"] != expected_latency:
                    raise RuntimeError(
                        f"unexpected {model} runtime-repeat latency on repeat {repeat}: "
                        f"expected {expected_latency}, got {metadata['latency_samples']}"
                    )
                runtime.append(
                    {
                        "model": model,
                        "repeat": repeat,
                        "finite": bool(np.all(np.isfinite(samples))),
                        "latency_samples": int(metadata["latency_samples"]),
                        "expected_latency_samples": expected_latency,
                        "p99_frame_ms": float(metadata["p99_frame_seconds"]) * 1000.0,
                        "rtf": float(metadata["rtf"]),
                    }
                )

    attestation_hash = _sha256(attestation) if attestation.is_file() else None
    report = {
        "schema_version": 2,
        "purpose": "Compact evidence for the pinned DeepFilter source build used by the AudioForge release candidate.",
        "command": "python/tools/evaluate_deepfilter_source_build.py --reference-library target/deepfilter/df-inherited-before-adoption.dll --candidate-library df.dll --benchmark target/release/deepfilter_benchmark.exe --corpus models/deepfilter_fullband_eval --model-dir models --output evaluation/deepfilter-source-build-report.json",
        "source": {
            "repository": upstream["repository"],
            "commit": upstream["commit"],
            "models": model_hashes,
        },
        "recipe": {
            "script": "build_deepfilter.ps1",
            "manifest": "build-support/deepfilter/Cargo.toml",
            "lockfile": "build-support/deepfilter/Cargo.lock",
            "provenance": "build-support/deepfilter/provenance.json",
            "tract_linalg": "0.21.17; crates.io archive SHA-256 5A4F0F9C134FAF99E3CF78B46D797398982E04AD46A3AD5223BCF8C0BD4AF360",
            "toolchain": "Rust 1.94.0 x86_64-pc-windows-msvc; release-lto; capi feature; default features disabled",
            "recipe_hashing": "Recipe text hashes are UTF-8 with CRLF/CR normalized to LF; model and archive hashes cover raw bytes.",
            "validation": "Full pinned libDF tree and tract-linalg patch tree are checked before build; required C exports are checked before atomic publication.",
        },
        "output": {
            "path": _relative(candidate_library),
            "bytes": candidate_library.stat().st_size,
            "sha256": candidate_actual_sha256,
            "attestation_path": _relative(attestation),
            "attestation_sha256": attestation_hash,
            "attestation_output_sha256": attestation_record["output"]["sha256"],
            "attestation_output_bytes": attestation_record["output"]["bytes"],
            "manifest_sha256_is_reference_only": True,
            "hash_note": "Native linker metadata can vary between approved toolchains; each build publishes its own attestation.",
        },
        "reference": {
            "path": _relative(reference_library),
            "bytes": reference_library.stat().st_size,
            "sha256": reference_actual_sha256,
            "expected_sha256": reference_sha256,
        },
        "source_sha256": source_hashes,
        "benchmark": {
            "hashes": benchmark_hashes,
            "args": "--attenuation-db 80 --post-filter-beta 0 --strength 1",
        },
        "parity": {
            "cases": len(comparisons),
            "models": [model for model, _ in MODELS],
            "finite_cases": sum(bool(row["finite"]) for row in comparisons),
            "latency_samples": {model: latency for model, latency in MODELS},
            "max_abs": max(float(row["max_abs"]) for row in comparisons),
            "rmse": max(float(row["rmse"]) for row in comparisons),
            "comparisons": comparisons,
        },
        "runtime": {
            "deterministic_gate_cases": len(comparisons) + len(runtime),
            "finite_cases": sum(bool(row["finite"]) for row in comparisons) + sum(bool(row["finite"]) for row in runtime),
            "latency_cases": sum(
                row["candidate_latency_samples"] == row["expected_latency_samples"]
                for row in comparisons
            )
            + sum(
                row["latency_samples"] == row["expected_latency_samples"]
                for row in runtime
            ),
            "repeats": f"{REPEATS} per model plus the {len(comparisons)} parity cases",
            "max_p99_frame_ms": max(float(row["candidate_p99_frame_ms"]) for row in comparisons + [{"candidate_p99_frame_ms": row["p99_frame_ms"]} for row in runtime]),
            "max_rtf": max(float(row["rtf"]) for row in runtime),
            "gate_p99_frame_ms": P99_GATE_MS,
            "runs": runtime,
        },
        "limitations": [
            "Microphone and physical output qualification remain pending by maintainer direction.",
            "The original inherited DLL's build identity was not recoverable; parity establishes behavior on this held-out corpus, not provenance.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--reference-library", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--candidate-library", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--attestation", type=Path, default=DEFAULT_ATTESTATION)
    parser.add_argument("--reference-sha256", default=REFERENCE_SHA256)
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = evaluate(
        corpus=args.corpus.resolve(),
        benchmark=args.benchmark.resolve(),
        reference_library=args.reference_library.resolve(),
        candidate_library=args.candidate_library.resolve(),
        model_dir=args.model_dir.resolve(),
        attestation=args.attestation.resolve(),
        reference_sha256=args.reference_sha256,
        candidate_sha256=args.candidate_sha256,
        output=args.output.resolve(),
    )
    parity = report["parity"]
    runtime = report["runtime"]
    passed = (
        parity["finite_cases"] == parity["cases"]
        and parity["max_abs"] == 0.0
        and parity["rmse"] == 0.0
        and runtime["finite_cases"] == runtime["deterministic_gate_cases"]
        and runtime["latency_cases"] == runtime["deterministic_gate_cases"]
        and runtime["max_p99_frame_ms"] <= P99_GATE_MS
    )
    print(
        f"DeepFilter source-build evaluation {'passed' if passed else 'failed'}: "
        f"{parity['cases']} parity cases, {runtime['deterministic_gate_cases']} total runtime cases, "
        f"max p99 {runtime['max_p99_frame_ms']:.4f} ms"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
