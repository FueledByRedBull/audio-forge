"""
AudioForge headless health check.

Runs the processor for a specified duration and validates callback health.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from mic_eq import AudioProcessor


# Normal clock-drift retiming counters are observational, not failures. True
# recovery, callback, overflow, short-write, and backlog-loss counters stay strict.
_ZERO_REQUIRED_DIAGNOSTICS = (
    "input_dropped_samples",
    "input_backlog_dropped_samples",
    "input_backlog_recovery_count",
    "input_callback_error_count",
    "lock_contention_count",
    "output_callback_error_count",
    "output_recovery_count",
    "output_recovery_event_count",
    "output_short_write_dropped_samples",
    "output_underrun_streak",
    "rt_buffer_overflow_count",
    "rt_error_code",
    "stream_restart_count",
    "suppressor_non_finite_count",
)


def _critical_diagnostic_failures(
    diagnostics: dict, *, output_underrun_baseline: int
) -> list[str]:
    failures: list[str] = []
    for key in _ZERO_REQUIRED_DIAGNOSTICS:
        if key not in diagnostics:
            failures.append(f"{key}=missing")
            continue
        try:
            value = int(diagnostics[key] or 0)
        except (TypeError, ValueError):
            failures.append(f"{key}=invalid")
            continue
        if value != 0:
            failures.append(f"{key}={value}")

    if not bool(diagnostics.get("noise_backend_available", False)):
        failures.append("noise_backend_available=false")
    if bool(diagnostics.get("noise_backend_failed", False)):
        failures.append("noise_backend_failed=true")
    if diagnostics.get("last_stream_error"):
        failures.append("last_stream_error=set")
    final_underruns = diagnostics.get("output_underrun_total")
    if not isinstance(final_underruns, (int, float)):
        failures.append("output_underrun_total=missing_or_invalid")
    else:
        final_underrun_count = int(final_underruns)
        if final_underrun_count != output_underrun_baseline:
            failures.append(
                "output_underrun_total="
                f"{final_underrun_count} (baseline {output_underrun_baseline})"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="AudioForge headless health check.")
    parser.add_argument(
        "--duration",
        type=float,
        default=600.0,
        help="Total runtime in seconds (default 600).",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=0.5,
        help="Polling interval in seconds (default 0.5).",
    )
    parser.add_argument(
        "--max-callback-age",
        type=int,
        default=2000,
        help="Max allowed callback age in ms (default 2000).",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=5.0,
        help="Warmup grace in seconds for callbacks to appear (default 5).",
    )
    parser.add_argument(
        "--allow-recovery",
        action="store_true",
        help="Allow auto-recovery events without failing.",
    )
    parser.add_argument(
        "--input-device", type=str, default=None, help="Input device name."
    )
    parser.add_argument(
        "--output-device", type=str, default=None, help="Output device name."
    )
    args = parser.parse_args()

    processor = AudioProcessor()
    try:
        result = processor.start(args.input_device, args.output_device)
        print(f"Started processor: {result}")

        start = time.monotonic()
        warmup_start = start
        last_restart_count = 0
        max_input_age = 0
        max_output_age = 0
        output_underrun_baseline: int | None = None
        try:
            last_restart_count = processor.get_stream_restart_count()
        except Exception:
            last_restart_count = 0

        while time.monotonic() - start < args.duration:
            try:
                input_age = processor.get_input_callback_age_ms()
                output_age = processor.get_output_callback_age_ms()
            except Exception as exc:
                print(f"Health check error: {type(exc).__name__}: {exc}")
                return 3

            try:
                recovery_result = processor.service_recovery()
            except Exception:
                recovery_result = None

            if recovery_result is False:
                err_msg = ""
                try:
                    err_msg = processor.get_last_stream_error() or ""
                except Exception:
                    err_msg = ""
                if err_msg:
                    print(f"Health check failed: auto-recovery failed ({err_msg}).")
                else:
                    print("Health check failed: auto-recovery failed.")
                return 4

            try:
                current_restart_count = processor.get_stream_restart_count()
            except Exception:
                current_restart_count = last_restart_count

            if current_restart_count > last_restart_count:
                warmup_start = time.monotonic()

            now = time.monotonic()
            in_warmup = (now - warmup_start) < args.warmup
            unknown_age_threshold = 1 << 63
            input_unknown = input_age >= unknown_age_threshold
            output_unknown = output_age >= unknown_age_threshold

            if in_warmup and (input_unknown or output_unknown):
                last_restart_count = current_restart_count
                time.sleep(args.poll)
                continue

            if not in_warmup and output_underrun_baseline is None:
                warm_diagnostics = dict(processor.get_runtime_diagnostics())
                output_underrun_baseline = int(
                    warm_diagnostics.get("output_underrun_total", 0) or 0
                )

            if not in_warmup and (input_unknown or output_unknown):
                unknown_parts = []
                if input_unknown:
                    unknown_parts.append("input")
                if output_unknown:
                    unknown_parts.append("output")
                missing = "/".join(unknown_parts)
                print(
                    "Health check failed: callback never observed "
                    f"({missing}) after {args.warmup:.1f}s warmup."
                )
                return 5

            if not input_unknown:
                max_input_age = max(max_input_age, int(input_age))
            if not output_unknown:
                max_output_age = max(max_output_age, int(output_age))

            if input_age > args.max_callback_age or output_age > args.max_callback_age:
                print(
                    "Health check failed: callback age exceeded "
                    f"(input={input_age}ms, output={output_age}ms)."
                )
                return 1

            if not args.allow_recovery and current_restart_count > last_restart_count:
                print(
                    "Health check failed: auto-recovery triggered "
                    f"(restarts={current_restart_count})."
                )
                return 2

            last_restart_count = current_restart_count
            time.sleep(args.poll)

        diagnostics = dict(processor.get_runtime_diagnostics())
        if output_underrun_baseline is None:
            print("Health check failed: no post-warmup underrun baseline was recorded.")
            return 7
        print(
            "Health summary: "
            f"max_input_age_ms={max_input_age} "
            f"max_output_age_ms={max_output_age} "
            f"restarts={last_restart_count} "
            f"underrun_baseline={output_underrun_baseline} "
            f"diagnostics={json.dumps(diagnostics, sort_keys=True, separators=(',', ':'))}"
        )
        diagnostic_failures = _critical_diagnostic_failures(
            diagnostics,
            output_underrun_baseline=output_underrun_baseline,
        )
        if diagnostic_failures:
            print(
                "Health check failed: critical runtime diagnostics were not clean "
                f"({', '.join(diagnostic_failures)})."
            )
            return 6
        print("Health check passed.")
        return 0
    except Exception as exc:
        print(f"Health check error: {type(exc).__name__}: {exc}")
        return 3
    finally:
        try:
            processor.stop()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
