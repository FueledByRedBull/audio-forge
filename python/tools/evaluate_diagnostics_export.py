"""Evaluate the privacy and determinism contract of diagnostics export."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from release_provenance import sha256_file as _sha256

from mic_eq.diagnostics_export import (
    MAX_SERIALIZED_BYTES,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    build_diagnostics_snapshot,
    serialize_diagnostics_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "diagnostics-export-report.json"
FIXED_TIME = datetime(2026, 7, 31, 12, 0, 0, tzinfo=timezone.utc)
FORBIDDEN_TOKENS = (
    "Private USB Microphone",
    "Private Virtual Cable",
    "C:\\Users\\private",
    "/home/private",
    "secret-value",
    '"raw_audio":',
    '"API_TOKEN":',
    '"unknown_path":',
)


def _fixture(key: bytes) -> dict[str, Any]:
    config = SimpleNamespace(
        input_channel_mode="average",
        input_cleanup_mode="gentle",
        use_measured_latency=True,
        main_control_tab_index=1,
        voice_setup_dynamics_intensity="balanced",
        voice_setup_custom_p95_db=3.5,
        voice_setup_custom_peak_cap_db=8.0,
        latency_calibration_profiles={
            "C:\\Users\\private\\route": object(),
        },
        last_preset="C:\\Users\\private\\preset.json",
    )
    processing = {
        "gate": {"enabled": True, "threshold_db": -42.0},
        "eq": {
            "schema_version": 2,
            "enabled": True,
            "bands": [
                {
                    "filter_type": "low_shelf",
                    "frequency_hz": 80.0,
                    "gain_db": float("inf"),
                    "q": 1.41,
                    "bandwidth_mode": "q",
                    "bandwidth_octaves": None,
                    "slope_db_per_octave": 12,
                    "stage": "combined",
                    "enabled": True,
                }
            ],
        },
        "rnnoise": {
            "enabled": True,
            "strength": 0.75,
            "model": "deepfilter-ll",
            "model_path": "C:\\Users\\private\\model.tar.gz",
        },
        "name": "Private preset",
    }
    runtime = {
        "input_dropped_samples": 0,
        "output_true_peak_db": -2.0,
        "output_short_term_lufs": float("nan"),
        "noise_backend_available": True,
        "noise_backend_error": "C:\\Users\\private\\model failed",
        "last_stream_error": "secret-value",
        "last_restart_reason": "/home/private/device changed",
        "raw_audio": [0.1, 0.2],
        "API_TOKEN": "secret-value",
        "unknown_path": "C:\\Users\\private\\audio.wav",
    }
    return build_diagnostics_snapshot(
        app_version="1.10.1",
        runtime_diagnostics=runtime,
        config=config,
        processing_settings=processing,
        input_device={
            "name": "Private USB Microphone",
            "is_default": True,
        },
        output_device={"name": "Private Virtual Cable"},
        processing_sample_rate_hz=48_000,
        output_sample_rate_hz=48_000,
        running=True,
        generated_at=FIXED_TIME,
        pseudonym_key=key,
        system_info={
            "operating_system": "Windows",
            "os_release": "11",
            "os_version": "10.0.26100",
            "architecture": "AMD64",
            "python_version": "3.12.10",
            "python_implementation": "CPython",
        },
    )


def evaluate() -> dict[str, Any]:
    first = _fixture(b"A" * 32)
    repeated = _fixture(b"A" * 32)
    alternate_key = _fixture(b"B" * 32)
    payload = serialize_diagnostics_snapshot(first)
    text = payload.decode("utf-8")
    parsed = json.loads(text)
    checks = {
        "schema": parsed["schema"]
        == {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "deterministic_for_fixed_inputs": first == repeated,
        "report_local_pseudonyms": (
            first["audio_engine"]["input_device"]["pseudonym"]
            != alternate_key["audio_engine"]["input_device"]["pseudonym"]
        ),
        "forbidden_values_absent": not any(
            token in text for token in FORBIDDEN_TOKENS
        ),
        "non_finite_values_removed": (
            first["runtime"]["output_short_term_lufs"] is None
            and "gain_db" not in first["processing"]["eq"]["bands"][0]
            and "NaN" not in text
            and "Infinity" not in text
        ),
        "raw_errors_reduced_to_presence": (
            first["runtime"]["backend_error_present"] is True
            and first["runtime"]["stream_error_present"] is True
            and first["runtime"]["restart_reason_present"] is True
        ),
        "size_bounded": len(payload) <= MAX_SERIALIZED_BYTES,
        "privacy_contract_declared": all(
            first["privacy"][key] is False
            for key in (
                "raw_audio_included",
                "environment_variables_included",
                "arbitrary_paths_included",
                "raw_device_names_included",
                "secrets_included",
            )
        ),
    }
    source_paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "python/mic_eq/diagnostics_export.py",
        REPO_ROOT / "python/mic_eq/ui/main_window.py",
        REPO_ROOT / "python/tests/test_diagnostics_export.py",
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "metrics": {
            "serialized_bytes": len(payload),
            "maximum_bytes": MAX_SERIALIZED_BYTES,
            "runtime_allowlisted_field_count": len(first["runtime"]),
            "processing_section_count": len(first["processing"]),
        },
        "privacy_contract": first["privacy"],
        "provenance": {
            "source_hashes": {
                path.relative_to(REPO_ROOT).as_posix(): _sha256(path)
                for path in source_paths
            },
            "fixture_contains_only_synthetic_identifiers": True,
        },
        "limitations": [
            "The snapshot is intentionally a coherent allowlisted state sample, not a raw application log.",
            "Report-local device pseudonyms cannot correlate devices across exports.",
            "Automated privacy checks cannot prove support completeness for every future diagnostic field; new fields require explicit allowlisting and tests.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = evaluate()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Diagnostics export evaluation status={report['status']} "
        f"bytes={report['metrics']['serialized_bytes']}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
