"""Summarise bench reports into one evaluation report: what a Voice Setup change did, before and after.

``bench_report.py --json-output`` writes the numbers of one run. This reads several of them and writes the compact
report that goes in ``evaluation/``, with the common decision block (``measured``: the maintainer decides). It answers
three questions from the same fixed bench set:

* **Noise reference.** When the room-noise recording is questionable (unsteady, partly matching the voice, or with a
  distractor in it), how often does Voice Setup refuse before and after the change that withholds only the gate and
  suppressor, what a retake would bring, and whether the setups it now applies are worse than the current settings.
* **Compressor.** Whether the onset and tail clauses fairly judge a compressor: how many setups, and how many
  compressor steps on their own, fail a clause read plain and read against their own compression, with a squashing
  compressor as the control that must still fail.
* **De-esser.** Whether changing the detector's confidence terms (``--deesser NAME=report.json``, runs on an
  extension built with the change) moves any bench reading beyond the margins of the earlier ablation, or any Voice
  Setup outcome. The margins (0.1 dB, 0.002 ESTOI) are borrowed from that rule as a yardstick, not as a gate.

    python python/tools/bench_study.py --before before.json --after after.json --deesser both=both.json \\
        --output evaluation/<name>-report.json --details-output models/evaluation-details/<name>

``--output`` is required and must not name a tracked report by accident; ``--details-output`` receives the inputs and
the per-mix rows, which stay out of Git. The bench set is fixed and public and nothing in it is held out: a sweep of
the distractor mixes over more SNRs (``bench_report.py --sweep-snr``) is separate, labelled, and only a sensitivity
check of the same recipes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import bench_metrics as bm
from evaluation_decision import decision_block

SCHEMA_VERSION = 1
# Margins of the retired de-esser ablation rule (0.1 dB, 0.002 ESTOI), used as a yardstick for "no change".
MARGINS = {"estoi": 0.002, "si_sdr": 0.1, "tone_error_db": 0.1, "sibilant_loss_db": 0.1, "deep_pause_db": 0.1,
           "tail_db": 0.1, "onset_db": 0.1, "pause_db": 0.1}
ARMS = ("step", "off", "squash")
LIMITATIONS = (
    "Fixed public bench set: two speakers, synthetic and public-recording noise, one realization per mix. Nothing is "
    "held out, and the set can be overfitted; the numbers say what changed on it, not what a stranger's microphone does.",
    "The distractors (TV, music, babble) are synthetic or built from the same two speakers. The sweep varies only their "
    "SNR; it is not a cohort.",
    "The current settings are always the factory Preset(). A retake is a stretch of the same noise that played during "
    "the take, so it measures a continuous distractor, not a passing one.",
    "The clauses are those of the retired refusal study; the compression-aware reading is new and is checked here only "
    "against its own controls (no compressor, a squashing compressor), not against listening.",
    "The Voice Setup tuner reads wall-clock time, so a run on a loaded machine can differ in its choices.",
)


def load(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema_version") != 1 or not {"build", "mixes"} <= report.keys():
        raise SystemExit(f"{path} is not a bench report (bench_report.py --json-output)")
    return report


def _rounded(value: Any) -> float | None:
    return round(float(value), 4) if isinstance(value, (int, float)) and np.isfinite(value) else None


def _ends(vs: dict[str, Any]) -> str | None:
    return "final" if vs.get("final") else "proposal" if vs.get("proposal") else None


def setup_row(entry: dict[str, Any]) -> dict[str, Any]:
    """What Voice Setup did with a mix and how the setup the user ends with compares with the current settings."""
    vs, current = entry["voice_setup"], entry["factory"]
    which = _ends(vs)
    row: dict[str, Any] = {"outcome": vs["outcome"], "veto": vs["veto"], "tuner": vs.get("tuner"), "steps": vs["steps"]}
    if which:
        readings = vs[which]
        row["reads"] = {key: _rounded(readings.get(key)) for key in ("estoi", "si_sdr", "tone_error_db", "onset_db")}
        row["fails_plain"] = bm.failed_clauses(readings, current)
        row["fails_aware"] = bm.failed_clauses(vs["fair"][which], current["fair"])
    return row


def noise_reference_study(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    shared = [k for k, e in before["mixes"].items() if "voice_setup" in e and "voice_setup" in after["mixes"].get(k, {})]
    rows, groups = [], {"usable": [], "questionable": [], "invalid": [], "none": []}
    for mix_id in shared:
        old, new = before["mixes"][mix_id], after["mixes"][mix_id]
        reference = old["voice_setup"].get("noise_reference") or {}
        # "none": refused before the room noise was judged (the clipped take).
        row = {"mix": mix_id, "sweep": "@" in mix_id, "reference": reference.get("status") or "none",
               "reasons": reference.get("reasons", []), "retakes": reference.get("retakes"),
               "before": setup_row(old), "after": setup_row(new)}
        rows.append(row)
        groups.setdefault(row["reference"], []).append(row)

    def tally(group: list[dict[str, Any]], side: str) -> dict[str, int]:
        count = Counter(f"refused:{r[side]['veto']}" if r[side]["outcome"] == "refused" else r[side]["outcome"]
                        for r in group)
        return dict(sorted(count.items()))

    def study(group: list[dict[str, Any]]) -> dict[str, Any]:
        retakes = [status for r in group for status in (r["retakes"] or [])]
        applied_now = [r for r in group if r["after"]["outcome"] == "applied"]
        return {
            "mixes": len(group), "before": tally(group, "before"), "after": tally(group, "after"),
            "retakes_usable": f"{retakes.count('usable')} of {len(retakes)}" if retakes else None,
            "applied_fail_plain": sum(bool(r["after"].get("fails_plain")) for r in applied_now),
            "applied_fail_aware": sum(bool(r["after"].get("fails_aware")) for r in applied_now),
            "applied": len(applied_now),
        }

    changed = [r for r in rows if r["before"]["outcome"] != r["after"]["outcome"]]
    return {
        "groups": {name: {"fixed": study([r for r in g if not r["sweep"]]), "sweep": study([r for r in g if r["sweep"]])}
                   for name, g in groups.items() if g},
        "changed": [{"mix": r["mix"], "reference": r["reference"], "before": f"{r['before']['outcome']} {r['before']['veto'] or ''}".strip(),
                     "after": f"{r['after']['outcome']} {r['after']['veto'] or ''}".strip(),
                     "tuner": r["after"]["tuner"], "fails_plain": r["after"].get("fails_plain"),
                     "fails_aware": r["after"].get("fails_aware"), "reads": r["after"].get("reads")} for r in changed],
        "rows": rows,
    }


def compressor_study(report: dict[str, Any]) -> dict[str, Any]:
    mixes = [(k, e) for k, e in report["mixes"].items() if (e.get("voice_setup") or {}).get("compressor_step")]
    out: dict[str, Any] = {"mixes": len(mixes)}
    for arm in ARMS:
        failing = {"plain": Counter(), "aware": Counter()}
        mixes_failing = {"plain": 0, "aware": 0}
        for _, entry in mixes:
            readings = entry["voice_setup"]["compressor_step"][arm]
            for kind, current in (("plain", entry["factory"]), ("aware", entry["factory"]["fair"])):
                fails = bm.failed_clauses(readings["plain" if kind == "plain" else "fair"], current)
                failing[kind].update(fails)
                mixes_failing[kind] += bool(fails)
        out[arm] = {kind: {"mixes_failing": mixes_failing[kind], "by_clause": dict(sorted(failing[kind].items()))} for kind in failing}
    ended = [(k, e) for k, e in report["mixes"].items() if (e.get("voice_setup") or {}).get("proposal")]
    out["whole_setups"] = {
        "setups": len(ended),
        "failing_plain": sum(bool(setup_row(e).get("fails_plain")) for _, e in ended),
        "failing_aware": sum(bool(setup_row(e).get("fails_aware")) for _, e in ended),
    }
    return out


def _arm_readings(entry: dict[str, Any], arm: str) -> dict[str, Any] | None:
    if arm == "deesser_on":
        return entry.get("deesser_on")
    vs = entry.get("voice_setup") or {}
    return vs.get(arm)


def deesser_study(control: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    """Paired differences (variant minus control) of every reading, per arm, against the margins."""
    shared = [k for k in control["mixes"] if k in variant["mixes"]]
    arms: dict[str, Any] = {}
    for arm in ("deesser_on", "proposal", "final"):
        deltas: dict[str, list[float]] = {key: [] for key in MARGINS}
        for mix_id in shared:
            a, b = (_arm_readings(r["mixes"][mix_id], arm) for r in (variant, control))
            if not a or not b or "error" in a or "error" in b:
                continue
            for key in MARGINS:
                x, y = a.get(key), b.get(key)
                if isinstance(x, (int, float)) and isinstance(y, (int, float)) and np.isfinite(x) and np.isfinite(y):
                    deltas[key].append(float(x - y))
        if any(deltas.values()):
            arms[arm] = {key: {"mixes": len(v), "mean": _rounded(float(np.mean(v))), "max_abs": _rounded(float(np.max(np.abs(v)))),
                               "beyond_margin": int(np.sum(np.abs(v) > MARGINS[key])), "identical": int(np.sum(np.asarray(v) == 0.0))}
                         for key, v in deltas.items() if v}
    outcome_changes = [k for k in shared if "voice_setup" in control["mixes"][k] and "voice_setup" in variant["mixes"][k]
                       and any(control["mixes"][k]["voice_setup"][f] != variant["mixes"][k]["voice_setup"][f]
                               for f in ("outcome", "veto", "steps"))]
    beyond = sum(m["beyond_margin"] for arm in arms.values() for m in arm.values())
    return {"mixes": len(shared), "arms": arms, "readings_beyond_margin": beyond, "outcome_or_step_changes": outcome_changes,
            "measurable_change": bool(beyond or outcome_changes)}


def build_report(before: dict[str, Any], after: dict[str, Any], deesser: dict[str, dict[str, Any]]) -> dict[str, Any]:
    study = noise_reference_study(before, after)
    details_rows = study.pop("rows")
    report = {
        "schema_version": SCHEMA_VERSION,
        "title": "Voice Setup on the fixed bench set: the questionable room-noise reference, the compressor clauses, the de-esser confidence terms",
        "decision": decision_block("measured", "The numbers are information; the maintainer decides."),
        "builds": {"before": before["build"], "after": after["build"], **{f"deesser_{n}": r["build"] for n, r in deesser.items()}},
        "bench": {"mixes": len(after["mixes"]), "sweep_mixes": sum("@" in k for k in after["mixes"])},
        "noise_reference": study,
        "compressor": compressor_study(after),
        "deesser": {name: deesser_study(after, run) for name, run in deesser.items()},
        "limitations": list(LIMITATIONS),
    }
    report["_rows"] = details_rows
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", type=Path, required=True, help="bench_report.py --json-output of the base build")
    parser.add_argument("--after", type=Path, required=True, help="the same run with the change")
    parser.add_argument("--deesser", action="append", default=[], metavar="NAME=REPORT",
                        help="a run on an extension built with a de-esser change, compared with --after")
    parser.add_argument("--output", type=Path, required=True, help="the evaluation report to write")
    parser.add_argument("--details-output", type=Path, help="a folder for the inputs and the per-mix rows (ignored by Git)")
    args = parser.parse_args(argv)
    variants = {}
    for item in args.deesser:
        name, _, path = item.partition("=")
        if not name or not path:
            parser.error(f"--deesser takes NAME=REPORT, not {item!r}")
        variants[name] = load(Path(path))
    inputs = {args.before, args.after, *(Path(i.partition("=")[2]) for i in args.deesser)}
    copies = {path: args.details_output / path.name for path in inputs} if args.details_output else {}
    for path, target in copies.items():
        if target.exists() and not target.samefile(path):
            raise SystemExit(f"{target} already exists; not overwriting an earlier input")
    report = build_report(load(args.before), load(args.after), variants)
    rows = report.pop("_rows")
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    if args.details_output:
        args.details_output.mkdir(parents=True, exist_ok=True)
        (args.details_output / "rows.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8", newline="\n")
        for path, target in copies.items():
            if not target.exists():
                shutil.copy(path, target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
