"""Evaluate pinned official DPDFNet EvalSet outputs on a stratified subset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pesq import pesq  # type: ignore[reportMissingImports]
from pystoi import stoi  # type: ignore[reportMissingImports]
from scipy.io import wavfile

OUTPUTS = ("Noisy", "DeepFilterNet3", "DPDFNet2", "DPDFNet4", "DPDFNet8")
SAMPLE_RATE = 16_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> np.ndarray:
    sample_rate, raw = wavfile.read(path)
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"{path} is {sample_rate} Hz, expected {SAMPLE_RATE}")
    audio = np.asarray(raw)
    if audio.ndim == 2:
        audio = np.mean(audio.astype(np.float64), axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype.name)
        audio = audio.astype(np.float64) / float(max(abs(info.min), info.max))
    return np.asarray(np.nan_to_num(audio), dtype=np.float64)


def _si_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference - float(np.mean(reference))
    estimate = estimate - float(np.mean(estimate))
    energy = float(np.dot(reference, reference))
    if energy <= 1.0e-15:
        return -120.0
    target = reference * (float(np.dot(estimate, reference)) / energy)
    noise = estimate - target
    return float(
        10.0
        * np.log10(
            max(float(np.dot(target, target)), 1.0e-15)
            / max(float(np.dot(noise, noise)), 1.0e-15)
        )
    )


def _metric_row(clean: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    common = min(clean.size, estimate.size)
    reference = np.ascontiguousarray(clean[:common], dtype=np.float64)
    enhanced = np.ascontiguousarray(estimate[:common], dtype=np.float64)
    return {
        "pesq_wb": float(pesq(SAMPLE_RATE, reference, enhanced, "wb")),
        "stoi": float(stoi(reference, enhanced, SAMPLE_RATE, extended=False)),
        "si_snr_db": _si_snr(reference, enhanced),
    }


def evaluate(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    grouped: dict[tuple[str, str, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in manifest["files"]:
        path = root / row["path"]
        if _sha256(path) != row["sha256"]:
            raise ValueError(f"hash mismatch: {path}")
        key = (row["language"], row["noise_type"], float(row["snr_db"]))
        grouped[key][row["model_name"]] = row

    rows: list[dict[str, Any]] = []
    for condition, files in sorted(grouped.items()):
        missing = {"Clean", *OUTPUTS} - set(files)
        if missing:
            raise ValueError(f"{condition} is missing {sorted(missing)}")
        clean = _load(root / files["Clean"]["path"])
        for output_name in OUTPUTS:
            estimate = _load(root / files[output_name]["path"])
            rows.append(
                {
                    "language": condition[0],
                    "noise_type": condition[1],
                    "snr_db": condition[2],
                    "model": output_name,
                    **_metric_row(clean, estimate),
                }
            )

    aggregate: dict[str, dict[str, float | int]] = {}
    for output_name in OUTPUTS:
        selected = [row for row in rows if row["model"] == output_name]
        aggregate[output_name] = {
            "condition_count": len(selected),
            "mean_pesq_wb": float(np.mean([row["pesq_wb"] for row in selected])),
            "mean_stoi": float(np.mean([row["stoi"] for row in selected])),
            "mean_si_snr_db": float(np.mean([row["si_snr_db"] for row in selected])),
            "median_pesq_wb": float(np.median([row["pesq_wb"] for row in selected])),
            "median_stoi": float(np.median([row["stoi"] for row in selected])),
            "median_si_snr_db": float(
                np.median([row["si_snr_db"] for row in selected])
            ),
        }

    deepfilter = aggregate["DeepFilterNet3"]
    component_comparisons = {
        model: {
            "pesq_above_deepfilternet3": (
                float(aggregate[model]["mean_pesq_wb"])
                > float(deepfilter["mean_pesq_wb"])
            ),
            "stoi_above_deepfilternet3": (
                float(aggregate[model]["mean_stoi"]) > float(deepfilter["mean_stoi"])
            ),
            "si_snr_above_deepfilternet3": (
                float(aggregate[model]["mean_si_snr_db"])
                > float(deepfilter["mean_si_snr_db"])
            ),
        }
        for model in ("DPDFNet2", "DPDFNet4", "DPDFNet8")
    }
    return {
        "schema_version": 1,
        "dataset": {
            "source": manifest["source"],
            "revision": manifest["revision"],
            "license": manifest["license"],
            "manifest_sha256": _sha256(manifest_path),
            "selection": manifest["selection"],
        },
        "metric_packages": {
            "pesq": importlib.metadata.version("pesq"),
            "pystoi": importlib.metadata.version("pystoi"),
        },
        "method": {
            "sample_rate": SAMPLE_RATE,
            "pesq_mode": "wideband",
            "stoi_extended": False,
            "aggregation": "unweighted mean and median across 36 stratified conditions",
            "alignment": "official EvalSet files used as supplied; equal-length common prefix",
        },
        "aggregate": aggregate,
        "component_comparisons": component_comparisons,
        "claim_labels": {
            "locally_reproduced": [
                "PESQ, STOI, and SI-SNR ordering on the pinned 36-condition stratified EvalSet subset."
            ],
            "author_reported_not_independently_reproduced": [
                "The paper's full 324-condition averages.",
                "DNSMOS, NISQA, PRISM, DNS4 blind-test scores, and Ceva NPU realtime factors.",
            ],
            "scope_limit": (
                "Subset reproduction validates direction on selected official outputs; "
                "it does not reproduce the paper's full-table values."
            ),
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("models/dpdfnet_eval_subset/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/dpdfnet-official-evalset-report.json"),
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        help="Optional path for per-condition rows; the main report stays concise.",
    )
    args = parser.parse_args()
    report = evaluate(args.manifest)
    if args.details_output:
        args.details_output.parent.mkdir(parents=True, exist_ok=True)
        args.details_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    report.pop("rows", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    print(json.dumps(report["component_comparisons"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
