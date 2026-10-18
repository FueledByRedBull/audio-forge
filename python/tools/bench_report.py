"""Render the bench clip set through the offline chain and Voice Setup and write a report.

The set is `python/tests/fixtures/bench/`: ten clean takes of two speakers and sixteen mixes built from
them (fan, keyboard, cafe, traffic, hum, reverb, far mic, hot, quiet, clipped, TV, music, babble and
two quiet rooms). It is fixed, public and free to reuse. Nothing about it is held out or "spent": a
tuning change is expected to be run against it as often as needed. The numbers inform the maintainer's
decision and are not a gate. Tuning against the set can overfit it; that is accepted, and the report
says what it covers, not what a stranger's microphone would do.

Each mix goes through two paths:

* the factory chain, rendered offline with the factory `Preset()`;
* Voice Setup with Auto-EQ as the app's offline flow runs it (analysis, then the dialog's verification
  with its automatic reductions), read on the proposal and, when the setup is applied, on the settings
  the user ends with. The mix is both the speech recording and the verification take, and its own
  first two seconds of room noise are the noise recording.

Measures against the clean take are ESTOI, SI-SDR, tone error, and onset level. Sibilant loss, pause,
deep-pause and tail level are read against the input. Each is defined as at v1.18.0; see
`bench_metrics.py`. CPU time is the process CPU time of the render, or of the analysis and verification.

    python python/tools/bench_report.py --output bench.md --json-output bench.json
    python python/tools/bench_report.py --output after.md --json-output after.json --compare bench.json

Silero is used when `VAD_MODEL_PATH` or `models/silero_vad.onnx` is available; without it the VAD is
reported as unavailable and the gate and Voice Setup run on their energy path. DeepFilter is not
needed; RNNoise is the suppressor. The full set takes about ten minutes. The Voice Setup tuner reads
wall-clock time, so its choices can move with machine load; run on an idle machine.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import bench_metrics as bm

SCHEMA_VERSION = 1
# (key, heading, decimals)
MEASURES = (
    ("estoi", "ESTOI", 3), ("si_sdr", "SI-SDR dB", 1), ("tone_error_db", "tone err dB", 2),
    ("sibilant_loss_db", "sib loss dB", 1), ("deep_pause_db", "deep pause dB", 1), ("tail_db", "tail dB", 1),
    ("onset_db", "onset dB", 1), ("limiter_gain_reduction_db", "limiter GR dB", 1),
)


def git_revision(root: Path = bm.REPO_ROOT) -> str | None:
    """HEAD when the working tree is clean, otherwise None (a report of unsaved code names no commit)."""
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return run("rev-parse", "HEAD") if not run("status", "--porcelain") else None
    except (OSError, subprocess.CalledProcessError):
        return None


def build_identity(root: Path, vad: bool) -> dict[str, Any]:
    import scipy
    from mic_eq import mic_eq_core

    return {
        "native_module": Path(mic_eq_core.__file__).name,
        "native_module_sha256": bm.file_sha256(Path(mic_eq_core.__file__)),
        "git_revision": git_revision(),
        "manifest_sha256": bm.file_sha256(root / "manifest.json"),
        "vad": "silero" if vad else "unavailable",
        # The tuner tries DeepFilter models only when this is set, so reports from different setups differ visibly.
        "deepfilter_enabled": os.environ.get("AUDIOFORGE_ENABLE_DEEPFILTER"),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
    }


def check_files(bench: bm.Bench, root: Path) -> None:
    """The committed inputs are the files the manifest names; a report of other audio compares with nothing."""
    entries = {**bench.manifest["takes"], **bench.manifest["noises"]}
    differ = [e["file"] for e in entries.values() if bm.file_sha256(root / e["file"]) != e["sha256"]]
    if differ:
        raise SystemExit(f"bench files differ from the manifest: {', '.join(differ)}")


def run(bench: bm.Bench, mix_ids: list[str], vad: bool, *, with_voice_setup: bool = True,
        partial: Path | None = None) -> dict[str, Any]:
    """Render each mix; ``partial`` gets one JSON line per finished mix, so a crash keeps the earlier ones."""
    mixes: dict[str, Any] = {}
    if partial:
        partial.write_text("", encoding="utf-8")
    for mix_id in mix_ids:
        mix = bm.build_mix(bench, mix_id)
        ctx = bm.make_context(mix)
        settings = bm.current_settings(vad)
        started = time.process_time()
        factory = bm.measure_settings(settings, ctx)
        entry: dict[str, Any] = {
            "condition": bench.manifest["mixes"][mix_id]["condition"], "take": bench.manifest["mixes"][mix_id]["take"],
            "factory": {**factory, "cpu_s": time.process_time() - started,
                        "suppressor": {k: settings["suppressor_settings"].get(k) for k in ("enabled", "model", "strength")}},
        }
        if with_voice_setup:
            entry["voice_setup"] = bm.voice_setup(ctx, vad)
        mixes[mix_id] = entry
        if partial:
            with partial.open("a", encoding="utf-8", newline="\n") as lines:
                lines.write(json.dumps({mix_id: entry}, default=float) + "\n")
        print(f"{mix_id} done", file=sys.stderr)
    return mixes


def number(value: Any, decimals: int) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        return "-"
    return f"{value:.{decimals}f}"


def signed(value: float | None, decimals: int) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:+.{decimals}f}"


def _mean(values: list[Any]) -> float | None:
    finite = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def _table(headings: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headings) + " |", "|" + "|".join("---" for _ in headings) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _voice_measures(entry: dict[str, Any]) -> dict[str, Any]:
    vs = entry.get("voice_setup") or {}
    return vs.get("proposal") or {}


def _outcome(vs: dict[str, Any]) -> str:
    return vs["outcome"] + (f" ({vs['veto']}: {bm.VETO_NAMES.get(vs['veto'], vs.get('reason', ''))})" if vs["veto"] else "")


def _clip(measures: dict[str, Any]) -> str:
    if "error" in measures:
        return "render failed"
    flags = [name for name, key in (("clip", "clipped"), ("limiter", "limiter_overload")) if measures.get(key)]
    return ",".join(flags) if flags else "no"


def _rows(mixes: dict[str, Any], measures_of: Any, lead: Any, tail: Any) -> list[list[str]]:
    rows = []
    for mix_id, entry in mixes.items():
        measures = measures_of(entry)
        rows.append([mix_id, *lead(entry), *(number(measures.get(key), d) for key, _, d in MEASURES),
                     _clip(measures) if measures else "-", *tail(entry)])
    means = [_mean([measures_of(e).get(key) for e in mixes.values()]) for key, _, _ in MEASURES]
    rows.append(["**mean**", *["" for _ in lead(next(iter(mixes.values())))],
                 *(number(m, d) for m, (_, _, d) in zip(means, MEASURES)), "", *["" for _ in tail(next(iter(mixes.values())))]])
    return rows


def _suppressors(mixes: dict[str, Any]) -> str:
    """The factory chain's model and the models Voice Setup's tuner chose, with how many mixes each."""
    def count(models: list[str]) -> str:
        return ", ".join(f"{name} x{models.count(name)}" for name in sorted(set(models))) or "none"

    factory = [str(e["factory"].get("suppressor", {}).get("model")) for e in mixes.values()]
    chosen = [str(e["voice_setup"]["suppressor"]["model"]) for e in mixes.values()
              if (e.get("voice_setup") or {}).get("suppressor")]
    return f"factory chain {count(factory)}; Voice Setup chose {count(chosen)}"


def markdown(result: dict[str, Any]) -> str:
    build, mixes = result["build"], result["mixes"]
    head = [h for _, h, _ in MEASURES]
    out = [
        "# Bench report", "",
        f"{len(mixes)} mixes of the fixed bench set. The numbers are information for the maintainer, not a gate.",
        "",
        f"- Build: native module `{build['native_module_sha256'][:16]}`, git revision "
        f"`{build['git_revision'] or 'not clean'}`, manifest `{build['manifest_sha256'][:16]}`.",
        f"- VAD: {'Silero' if build['vad'] == 'silero' else 'unavailable (energy path)'}.",
        f"- Suppressor: {_suppressors(mixes)}. AUDIOFORGE_ENABLE_DEEPFILTER: {build.get('deepfilter_enabled') or 'unset'}.",
        f"- Run time: {result['run_seconds']:.0f} s wall, {sum(e['factory']['cpu_s'] for e in mixes.values()):.1f} s "
        f"CPU in the factory renders.",
        "",
        "## Factory chain", "",
        _table(["mix", *head, "overload", "CPU s"],
               _rows(mixes, lambda e: e["factory"], lambda e: [], lambda e: [number(e["factory"]["cpu_s"], 1)])),
    ]
    if any("voice_setup" in e for e in mixes.values()):
        voiced = {k: e for k, e in mixes.items() if "voice_setup" in e}
        out += ["", "## Voice Setup with Auto-EQ", "",
                "Readings of the proposal; a refused setup is not applied, and the user keeps the factory chain.", "",
                _table(["mix", "outcome", "vetoes fired", "steps", *head, "overload", "CPU s"],
                       _rows(voiced, _voice_measures,
                             lambda e: [_outcome(e["voice_setup"]), ",".join(e["voice_setup"]["vetoes"]) or "-",
                                        ",".join(e["voice_setup"]["steps"]) or "-"],
                             lambda e: [number(e["voice_setup"]["cpu_s"], 1)]))]
    return "\n".join(out) + "\n"


def _delta(after: Any, before: Any) -> float | None:
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in (after, before)):
        return None
    return float(after - before)


def compare_markdown(result: dict[str, Any], earlier: dict[str, Any]) -> str:
    """After minus before for every measure of every mix the two reports share."""
    notes = []
    for key, label in (("vad", "VAD"), ("manifest_sha256", "bench manifest"),
                       ("deepfilter_enabled", "AUDIOFORGE_ENABLE_DEEPFILTER setting")):
        if result["build"].get(key) != earlier["build"].get(key):
            notes.append(f"- The {label} differs between the two reports, so the numbers are not comparable.")
    shared = [k for k in result["mixes"] if k in earlier["mixes"]]
    skipped = sorted(set(result["mixes"]) ^ set(earlier["mixes"]))
    if skipped:
        notes.append(f"- Not in both reports: {', '.join(skipped)}.")
    head = [h for _, h, _ in MEASURES]

    def deltas(getter: Any) -> list[list[str]]:
        rows, columns = [], [[] for _ in MEASURES]
        for mix_id in shared:
            after, before = getter(result["mixes"][mix_id]), getter(earlier["mixes"][mix_id])
            values = [_delta(after.get(key), before.get(key)) for key, _, _ in MEASURES]
            for column, value in zip(columns, values):
                column.append(value)
            rows.append([mix_id, *(signed(v, d) for v, (_, _, d) in zip(values, MEASURES))])
        rows.append(["**mean**", *(signed(_mean(c), d) for c, (_, _, d) in zip(columns, MEASURES))])
        return rows

    out = ["# Bench comparison (after minus before)", "", *notes, "", "## Factory chain", "",
           _table(["mix", *head], deltas(lambda e: e["factory"])), ""]
    if all("voice_setup" in result["mixes"][k] and "voice_setup" in earlier["mixes"][k] for k in shared):
        changes = []
        for mix_id in shared:
            a, b = (m[mix_id]["voice_setup"] for m in (result["mixes"], earlier["mixes"]))
            if (a["outcome"], a["veto"], a["steps"]) != (b["outcome"], b["veto"], b["steps"]):
                changes.append(f"- {mix_id}: {_outcome(b)} [{','.join(b['steps'])}] -> {_outcome(a)} [{','.join(a['steps'])}]")
        out += ["## Voice Setup with Auto-EQ", "", *(changes or ["Outcomes and steps are unchanged."]), "",
                _table(["mix", *head], deltas(_voice_measures))]
    cpu = [(k, result["mixes"][k]["factory"]["cpu_s"] - earlier["mixes"][k]["factory"]["cpu_s"]) for k in shared]
    out += ["", f"Factory render CPU: {signed(sum(v for _, v in cpu), 1)} s in total over the shared mixes."]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench-dir", type=Path, default=bm.BENCH_DIR)
    parser.add_argument("--mixes", help="comma-separated mix ids (default: all)")
    parser.add_argument("--skip-voice-setup", action="store_true", help="factory chain only")
    parser.add_argument("--output", type=Path, help="write the Markdown report here (default: stdout)")
    parser.add_argument("--json-output", type=Path, help="write the numbers and the build identity here; each "
                        "finished mix is also appended to <file>.partial.jsonl until the run ends")
    parser.add_argument("--compare", type=Path, help="an earlier report's JSON; print after-minus-before deltas")
    args = parser.parse_args(argv)
    earlier = None
    if args.compare:  # fail before the run, not after ten minutes
        try:
            earlier = json.loads(args.compare.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            parser.error(f"cannot read {args.compare}: {error}")
        if not isinstance(earlier, dict) or earlier.get("schema_version") != SCHEMA_VERSION or not all(
                key in earlier for key in ("build", "mixes")):
            parser.error(f"{args.compare} is not a schema {SCHEMA_VERSION} bench report")
    bench = bm.load_bench(args.bench_dir)
    check_files(bench, args.bench_dir)
    mix_ids = args.mixes.split(",") if args.mixes else list(bench.manifest["mixes"])
    unknown = [m for m in mix_ids if m not in bench.manifest["mixes"]]
    if unknown:
        parser.error(f"unknown mix ids: {', '.join(unknown)}")
    vad = bm.silero_available()
    started = time.perf_counter()
    partial = args.json_output.with_name(args.json_output.name + ".partial.jsonl") if args.json_output else None
    mixes = run(bench, mix_ids, vad, with_voice_setup=not args.skip_voice_setup, partial=partial)
    result = {"schema_version": SCHEMA_VERSION, "build": build_identity(args.bench_dir, vad),
              "run_seconds": time.perf_counter() - started, "mixes": mixes}
    text = markdown(result)
    if earlier:
        text += "\n" + compare_markdown(result, earlier)
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
        if partial:
            partial.unlink(missing_ok=True)
    if args.output:
        args.output.write_text(text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
