"""
AudioForge headless health check.

Runs the processor for a specified duration and validates callback health.
"""

from __future__ import annotations

import argparse
import sys
import time

from mic_eq import AudioProcessor


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
    parser.add_argument("--input-device", type=str, default=None, help="Input device name.")
    parser.add_argument("--output-device", type=str, default=None, help="Output device name.")
    args = parser.parse_args()

    processor = AudioProcessor()
    try:
        result = processor.start(args.input_device, args.output_device)
        print(f"Started processor: {result}")

        start = time.monotonic()
        warmup_start = start
        last_restart_count = 0
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
