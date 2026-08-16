"""Fetch pinned DPDFNet evaluation assets without adding runtime dependencies."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from release_provenance import sha256_file as _sha256

DATASET_ID = "Ceva-IP/DPDFNet_EvalSet"
DATASET_REVISION = "24866d8ae065f0518aef0f5f0c6200f31166af98"
MODEL_ID = "Ceva-IP/DPDFNet"
MODEL_REVISION = "dd6818d00f50c836fed43a6243ebe49116de5964"
MODEL_OUTPUTS = ("Clean", "Noisy", "DeepFilterNet3", "DPDFNet2", "DPDFNet4", "DPDFNet8")
BENCHMARK_MODELS = (
    "onnx/baseline.onnx",
    "onnx/dpdfnet2.onnx",
    "onnx/dpdfnet4.onnx",
    "onnx/dpdfnet8.onnx",
    "onnx/dpdfnet2_48khz_hr.onnx",
)
LICENSE = "Apache-2.0"


def _resolve_url(kind: str, repo_id: str, revision: str, path: str) -> str:
    quoted_path = "/".join(urllib.parse.quote(part) for part in path.split("/"))
    prefix = "datasets/" if kind == "datasets" else ""
    return f"https://huggingface.co/{prefix}{repo_id}/resolve/{revision}/{quoted_path}"


def _validated_hugging_face_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "huggingface.co"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.fragment
    ):
        raise ValueError("evaluation assets must use trusted Hugging Face HTTPS")
    return url


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        _validated_hugging_face_url(url),
        headers={"User-Agent": "AudioForge-evaluation/1"},
    )
    for attempt in range(8):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            os.replace(temporary, destination)
            time.sleep(0.5)
            return
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 7:
                temporary.unlink(missing_ok=True)
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else 2.0 ** (attempt + 1)
            time.sleep(min(max(delay, 2.0), 30.0))
    raise RuntimeError(f"download retry loop exhausted for {url}")


def _load_metadata() -> tuple[str, list[dict[str, str]]]:
    url = _resolve_url(
        "datasets",
        DATASET_ID,
        DATASET_REVISION,
        "metadata.csv",
    )
    request = urllib.request.Request(
        _validated_hugging_face_url(url),
        headers={"User-Agent": "AudioForge-evaluation/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read().decode("utf-8")
    return payload, list(csv.DictReader(io.StringIO(payload)))


def _selected_conditions(rows: list[dict[str, str]]) -> list[tuple[str, str, int]]:
    condition_set = {
        (row["language"], row["noise_type"], int(float(row["snr_db"])))
        for row in rows
        if row["model_name"] == "Clean"
    }
    languages = sorted({language for language, _noise, _snr in condition_set})
    noise_types = sorted({noise for _language, noise, _snr in condition_set})
    snrs = sorted({snr for _language, _noise, snr in condition_set})
    selected: list[tuple[str, str, int]] = []
    for language_index, language in enumerate(languages):
        for snr_index, snr in enumerate(snrs):
            noise = noise_types[(language_index * len(snrs) + snr_index) % len(noise_types)]
            condition = (language, noise, snr)
            if condition not in condition_set:
                raise RuntimeError(f"missing expected EvalSet condition: {condition}")
            selected.append(condition)
    return selected


def fetch_dataset_subset(root: Path) -> dict[str, Any]:
    metadata_payload, rows = _load_metadata()
    conditions = _selected_conditions(rows)
    selected_keys = set(conditions)
    selected_rows = [
        row
        for row in rows
        if row["model_name"] in MODEL_OUTPUTS
        and (
            row["language"],
            row["noise_type"],
            int(float(row["snr_db"])),
        )
        in selected_keys
    ]
    expected_count = len(conditions) * len(MODEL_OUTPUTS)
    if len(selected_rows) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} selected rows, found {len(selected_rows)}"
        )

    files: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows, start=1):
        relative_path = Path(row["file_name"])
        destination = root / relative_path
        if not destination.is_file():
            print(f"[{index}/{len(selected_rows)}] {relative_path}", flush=True)
            _download(
                _resolve_url(
                    "datasets",
                    DATASET_ID,
                    DATASET_REVISION,
                    row["file_name"],
                ),
                destination,
            )
        files.append(
            {
                "path": relative_path.as_posix(),
                "sha256": _sha256(destination),
                "size_bytes": destination.stat().st_size,
                "duration_s": float(row["duration_s"]),
                "sample_rate": int(row["sample_rate"]),
                "language": row["language"],
                "noise_type": row["noise_type"],
                "snr_db": float(row["snr_db"]),
                "model_name": row["model_name"],
            }
        )

    metadata_path = root / "metadata.csv"
    metadata_path.write_text(metadata_payload, encoding="utf-8", newline="")
    manifest = {
        "schema_version": 1,
        "source": f"https://huggingface.co/datasets/{DATASET_ID}",
        "revision": DATASET_REVISION,
        "license": LICENSE,
        "selection": {
            "strategy": "all 12 languages x all 3 SNRs, rotating across all 9 noise types",
            "condition_count": len(conditions),
            "outputs": list(MODEL_OUTPUTS),
        },
        "files": files,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def fetch_benchmark_models(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for index, model_path in enumerate(BENCHMARK_MODELS, start=1):
        relative_path = Path(Path(model_path).name)
        destination = root / relative_path
        if not destination.is_file():
            print(f"[model {index}/{len(BENCHMARK_MODELS)}] {relative_path}", flush=True)
            _download(
                _resolve_url("models", MODEL_ID, MODEL_REVISION, model_path),
                destination,
            )
        files.append(
            {
                "path": relative_path.as_posix(),
                "sha256": _sha256(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
    manifest = {
        "schema_version": 1,
        "source": f"https://huggingface.co/{MODEL_ID}",
        "revision": MODEL_REVISION,
        "license": LICENSE,
        "files": files,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("models/dpdfnet_eval_subset"),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("models/dpdfnet_benchmark_models"),
    )
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args()

    if not args.skip_dataset:
        dataset = fetch_dataset_subset(args.dataset_root)
        print(
            f"dataset: {len(dataset['files'])} files at revision "
            f"{dataset['revision']}"
        )
    if not args.skip_models:
        models = fetch_benchmark_models(args.model_root)
        print(f"models: {len(models['files'])} files at revision {models['revision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
