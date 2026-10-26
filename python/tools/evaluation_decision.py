"""The decision block of an evaluation report, and a reader for the reports that predate it.

A new report carries ``"decision": {"outcome", "reason", "decided", "by"}``:

* ``outcome``: ``adopt`` (the change the report evaluated is in the product), ``retain`` (the incumbent stays and
  the change is not in) or ``measured`` (the report records measurements and no decision);
* ``reason``: why, in a sentence;
* ``decided``: the ISO date of the decision, or null for ``measured``;
* ``by``: who made the decision, or null for ``measured``.

``read_decision`` gives the same four fields for any tracked report, whichever way it states its decision. For
a report released before the block, ``reason`` and ``decided`` are null where it records none, and ``by`` is null.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

OUTCOMES = ("adopt", "retain", "measured")
FIELDS = frozenset({"outcome", "reason", "decided", "by"})

# The outcome of each report released before the block. The old fields cannot be read by rule: `retained` and
# `retain` name the candidate in some reports and the incumbent in others, and several reports state the
# decision only in a sentence. Each entry is the report's own statement, in the block's terms; the reports
# themselves stay as released. A report that states no decision is `measured`.
LEGACY_OUTCOMES = {
    "archive-format-benchmark.json": "retain",
    "audio-followups-2026-10-report.json": "adopt",
    "auto-eq-confidence-calibration.json": "retain",
    "auto-eq-fixed-grid-2026-10-report.json": "retain",
    "auto-eq-solver-port-2026-10-report.json": "adopt",
    "auto-eq-tone-layer-2026-10-report.json": "retain",
    "auto-makeup-real-speech-2026-09-report.json": "adopt",
    "compressor-control-report.json": "adopt",
    "compressor-expanded-search-2026-09-report.json": "retain",
    # The first run of the expanded search; the 2026-09 run failed and the product searches threshold only.
    "compressor-search-report.json": "retain",
    "correction-tone-product-report.json": "adopt",
    "cross-take-auto-eq-report.json": "retain",
    "deepfilter-fullband-report.json": "retain",
    "deepfilter-hardening-report.json": "adopt",
    "deepfilter-source-build-report.json": "measured",
    "deesser-corpus-v1-2026-09-report.json": "measured",
    "deesser-corpus-v1-report.json": "measured",
    "deesser-real-speech-2026-10-report.json": "retain",
    "dpdfnet-official-evalset-report.json": "measured",
    "dpdfnet-vs-deepfilternet3-report.json": "retain",
    "dynamics-aliasing-2026-09-report.json": "retain",
    "eq-candidate-pool-report.json": "retain",
    "eq-filter-types-report.json": "adopt",
    "gate-policy-2026-10-report.json": "retain",
    "gate-speech-presence-2026-10-report.json": "retain",
    "limiter-lookahead-2026-10-report.json": "adopt",
    "mic-coloration-prior-2026-10-report.json": "retain",
    "onnxruntime-cpu-probe.json": "adopt",
    "processing-order-2026-09-report.json": "retain",
    "processing-order-2026-10-report.json": "retain",
    "product-joint-tuning-2026-09-report.json": "adopt",
    "product-joint-tuning-deepfilter-2026-09-report.json": "adopt",
    "product-joint-tuning-deepfilter-ll-2026-09-report.json": "adopt",
    "release-bundle-path-baseline.json": "measured",
    "resampler-quality-2026-10-report.json": "adopt",
    "resampler-quality-report.json": "adopt",
    "rnnoise-backend-comparison.json": "retain",
    "sparse-auto-eq-filter-report.json": "retain",
    "vad-model-selection-report.json": "adopt",
    "vad-v6.2.1-report.json": "adopt",
    "voice-preservation-2026-09-report.json": "adopt",
    "voice-preservation-report.json": "adopt",
    "voice-setup-arms-2026-10-report.json": "measured",
    "voice-setup-limiter-allowance-narrow-2026-10-report.json": "adopt",
    "voice-setup-loudness-fit-2026-10-report.json": "adopt",
    "voice-setup-refusals-2026-10-report.json": "measured",
    "voice-setup-tuner-2026-10-report.json": "retain",
    "voice-setup-verification-tonal-check-2026-10-report.json": "adopt",
    "warm-auto-eq-report.json": "adopt",
}


def block_errors(block: Any) -> list[str]:
    """Everything wrong with a decision block; empty when it is well formed."""
    if not isinstance(block, dict):
        return ["decision must be an object"]
    errors = [f"decision has no {name}" for name in sorted(FIELDS - block.keys())]
    errors += [f"decision has unknown field {name}" for name in sorted(block.keys() - FIELDS)]
    if errors:
        return errors
    if block["outcome"] not in OUTCOMES:
        errors.append(f"decision outcome must be one of {', '.join(OUTCOMES)}")
    if not isinstance(block["reason"], str) or not block["reason"].strip():
        errors.append("decision reason must be a sentence")
    decided, by = block["decided"], block["by"]
    if block["outcome"] == "measured":
        if decided is not None or by is not None:
            errors.append("a measured report records no decided date or decider")
        return errors
    try:
        date.fromisoformat(decided)
    except (TypeError, ValueError):
        errors.append("decision decided must be an ISO date (YYYY-MM-DD)")
    if not isinstance(by, str) or not by.strip():
        errors.append("decision by must name who decided")
    return errors


def decision_block(outcome: str, reason: str, decided: str | None = None, by: str | None = None) -> dict[str, Any]:
    """A well-formed decision block, or ValueError saying what is wrong with the arguments."""
    block = {"outcome": outcome, "reason": reason, "decided": decided, "by": by}
    if errors := block_errors(block):
        raise ValueError("; ".join(errors))
    return block


def _legacy_reason(report: dict[str, Any]) -> str | None:
    decision = report.get("decision")
    if isinstance(decision, dict):
        for key in ("reason", "product_action", "next_step"):
            if isinstance(decision.get(key), str):
                return decision[key]
    elif isinstance(decision, str) and " " in decision:  # a sentence; a bare token carries no reason
        return decision
    for key in ("summary", "reason"):
        if isinstance(report.get(key), str):
            return report[key]
    return None


def _legacy_date(report: dict[str, Any]) -> str | None:
    for key in ("generated_at_utc", "generated_at", "evaluated_at"):
        if isinstance(report.get(key), str) and len(report[key]) >= 10:
            return report[key][:10]
    return None


def read_decision(path: Path) -> dict[str, Any]:
    """The decision block of a tracked report, whichever way the report states it.

    A report with the block gives it as written. One released before it gets its outcome from
    ``LEGACY_OUTCOMES``, its reason from whichever field holds one (``decision`` as an object or a sentence,
    ``summary``, ``reason``), and its date from when it was generated; it records no decider, so ``by`` is null.
    A report with neither is a ValueError: a new report needs the block.
    """
    report = json.loads(path.read_text(encoding="utf-8"))
    decision = report.get("decision") if isinstance(report, dict) else None
    if isinstance(decision, dict) and "outcome" in decision:
        if errors := block_errors(decision):
            raise ValueError(f"{path.name}: {'; '.join(errors)}")
        return dict(decision)
    if path.name not in LEGACY_OUTCOMES:
        raise ValueError(f'{path.name} has no decision block; add "decision" as evaluation/README.md describes')
    outcome = LEGACY_OUTCOMES[path.name]
    return {
        "outcome": outcome,
        "reason": _legacy_reason(report),
        "decided": None if outcome == "measured" else _legacy_date(report),
        "by": None,
    }
