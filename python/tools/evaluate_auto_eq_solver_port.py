"""Compare two Auto-EQ solver builds on the 2,084 cases of the fixed-grid study.

Question (fixed 2026-10-14, before the new solvers were run): do the Auto-EQ
fits computed with NumPy solvers match the fits computed with SciPy's
``least_squares`` and SLSQP closely enough that removing SciPy from the runtime
changes no recommendation and nothing audible?

Cases are those of ``evaluation/auto-eq-fixed-grid-2026-10-report.json``: 44
cross-take passages x 5 targets x 2 modes x 4 smoothing strengths, the same
passages raised to -1 dBFS peaks (5 targets, adaptive, conservative), 4
invalid captures, and 20 fit-split harness captures x 5 targets (with their
room-noise references). The four development passages (indices 0, 13, 26, 39)
are excluded from judging; ``--dev-only`` runs them for debugging and the
report then makes no decision.

Both arms are source roots (``python/mic_eq``) that differ only in the solver.
Each is staged into a temporary directory with the one native module given by
``--native-module`` and the CPU ONNX Runtime, so a root without a built
extension works and no root is written. Workers verify the module's SHA-256.

Amendment 1 (2026-10-14 01:48 +0300, before any judged case was run) added a
control arm: the incumbent source with SciPy, with the start vector of every
``least_squares`` and ``minimize`` call moved one unit in the last place
(``numpy.nextafter`` toward +infinity, clipped into the bounds) by an
interposed wrapper. SciPy is not stable against itself under that change, so
the port is judged against the control. "Control difference" is control versus
incumbent; "port difference" is candidate versus incumbent.

Gates, all of which must pass:
  S1 same_status: failure status is identical on every case of the port
     difference. A recommendation-status change is an allowed exception when
     the case moves from "reduced" to "apply", the incumbent's only reason is
     the constraint solver, its constraint solve failed (iteration cap or exit
     failure) or was rejected by the feasibility check, and the candidate's
     result is feasible. Other recommendation-status changes in the port
     difference number at most the changes in the control difference.
  S2 same_response: outside the S1 exceptions, on cases with an EQ in both
     arms, at least 90 % of the port difference is within 0.05 dB at every
     frequency, and for each T in {0.05, 0.25, 1.0} dB the port cases over T
     number at most twice the control cases over T, plus 5.
  S3 same_solver_verdicts: ``constraint_solver_success`` never goes from true
     to false in the port difference, and goes from false to true only with a
     feasible candidate result; cases where ``nonlinear_solver_success`` differs
     number at most twice those of the control difference, plus 2.
  S4 exceptions_are_small: each S1 exception's final response is within 1.0 dB
     of the SciPy arm's response had its SLSQP result not been rejected. That
     response is measured by replaying the SciPy arm with the rejected result
     accepted; if the replay is not available, of the SciPy arm's final
     ("reduced") response scaled back to full strength. If neither exists the
     case is reported and fails the gate.

Percentages use integer arithmetic. Run times, the response-difference
distribution and solver effort are reported, not gated. Interrupted runs
resume from ``<details-output stem>.<arm>.<n>.partial.jsonl``.

The report is meant to be committed under evaluation/. Its ``source_sha256``
pins ``PINNED_SOURCES``, which must be identical in the candidate root and in
the working tree (checked before any case runs); the incumbent's own files are
listed in ``measurement_identity``. Per-case detail goes only to
``--details-output``, which belongs under the ignored models/evaluation-details/.
"""

from __future__ import annotations

import argparse
import functools
import glob
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "models" / "cross_take_eval"
DEFAULT_HARNESS = REPO_ROOT / "models" / "voice_setup_eval"
SAMPLE_RATE = 48_000
DEV_PASSAGES = (0, 13, 26, 39)
TARGETS = ("broadcast", "podcast", "streaming", "warm", "flat")
MODES = ("adaptive", "static")
SMOOTHING = ("conservative", "balanced", "broad", "off")
INVALID = ("silence", "noise", "clipped", "short")
HARNESS_USERS = 20
RESPONSE_FREQS = np.geomspace(80.0, 16_000.0, 257)
MAX_WORKERS = 4
TYPICAL_TOLERANCE_DB = 0.05
TYPICAL_PERCENT = 90
EXCEED_THRESHOLDS_DB = (0.05, 0.25, 1.0)
EXCEED_FACTOR, EXCEED_ALLOWANCE = 2, 5
NONLINEAR_ALLOWANCE = 2
EXCEPTION_TOLERANCE_DB = 1.0
LIST_LIMIT = 100  # entries listed per list in the report; the details file holds every case
MAX_REPORT_BYTES = 100_000  # check_evaluation_hygiene.MAX_TRACKED_REPORT_BYTES (the worker cannot import it)
AUDIBLE_COUNTS = ("port_status_changes", "port_cases_over_0.05_db")
EVALUATOR_PATH = "python/tools/evaluate_auto_eq_solver_port.py"
PINNED_SOURCES = (  # check_evaluation_hygiene compares these with the working tree
    "python/mic_eq/analysis/auto_eq_parts/eq_fit.py",
    "python/mic_eq/analysis/auto_eq_parts/optimizer.py",
    "python/mic_eq/analysis/wav_io.py",
    EVALUATOR_PATH,
)
FEASIBILITY_TOLERANCE = 1.0e-9  # the existing check in _constrained_gain_refinement
CONSTRAINT_REASON = "constrained gain solve used its feasible projection"
SOURCE_GLOBS = (
    "python/mic_eq/analysis/**/*.py",
    "python/mic_eq/config.py",
    "python/mic_eq/config_parts/*.py",
    "python/mic_eq/ui/calibration_support.py",
)
AMENDMENT = {
    "number": 1,
    "fixed_at": "2026-10-14 01:48 +0300",
    "predates_any_judged_case": True,
    "reason": (
        "A parity study on 360 synthetic seeded fits showed the incumbent is not stable against itself: with "
        "every solver start value moved by one unit in the last place, SciPy's fitted gains move by up to 5.2 dB "
        "in the worst fit, constraint_solver_success flips in 20 of 360 fits, and SLSQP fails outright in 14 "
        "calls. Gates S1 to S3 as first written (rule fixed 2026-10-14 00:55 +0300) would measure SciPy's "
        "noise, not the port, so they are replaced; S4 stands."
    ),
}
RULES = {
    "question": (
        "Do the Auto-EQ fits computed with the NumPy solvers (bounded least squares, gain projection, "
        "constrained gain refinement) match the fits computed with SciPy's least_squares and SLSQP closely "
        "enough that removing SciPy from the runtime changes no recommendation and nothing audible?"
    ),
    "control arm": (
        "A third arm runs the incumbent source with SciPy, with the start vector of every least_squares and "
        "minimize call moved by one unit in the last place (numpy.nextafter toward +infinity, clipped into "
        "the bounds). \"Control difference\" means control arm versus incumbent arm; \"port difference\" "
        "means candidate arm versus incumbent arm."
    ),
    "S1 same_status": (
        "Failure status is identical on every case in the port difference. A recommendation-status change in "
        "the port difference is an allowed exception when the case moves from \"reduced\" to \"apply\", its "
        "only recommendation reason in the incumbent is the constraint-solver one, the incumbent's constraint "
        "solve failed (iteration cap or solver exit failure) or its result was rejected by the feasibility "
        "check, and the candidate's result is feasible. Every exception is listed. The number of other "
        "recommendation-status changes in the port difference is at most the number of recommendation-status "
        "changes in the control difference."
    ),
    "S2 same_response": (
        "On cases outside the S1 exceptions that produce an EQ in both arms compared: at least 90 % of cases "
        "in the port difference are within 0.05 dB at every frequency; and for each threshold T in "
        "{0.05, 0.25, 1.0} dB, the number of cases exceeding T in the port difference is at most twice the "
        "number exceeding T in the control difference plus 5."
    ),
    "S3 same_solver_verdicts": (
        "constraint_solver_success never changes from true to false in the port difference; a change from "
        "false to true requires a feasible candidate result. The number of cases where "
        "nonlinear_solver_success differs in the port difference is at most twice the number in the control "
        "difference plus 2."
    ),
    "S4 exceptions_are_small": (
        "For each S1 exception the new arm's final response is within 1.0 dB of the SciPy arm's response "
        "before SciPy's result was rejected... if that intermediate is not observable, within 1.0 dB of "
        "the SciPy arm's final (\"reduced\") response scaled back to full strength; if neither can be "
        "computed the case is reported and counts as a failure of this gate."
    ),
    "decision": (
        "SciPy leaves the runtime only if S1 to S4 all pass. Otherwise SciPy stays for this release and "
        "the evidence is filed under models/evaluation-details/auto-eq-solver-port-2026-10/. A failed "
        "gate is not reinterpreted after the run; a solver bug found on the development passages may be "
        "fixed and the whole comparison rerun, and each rerun is recorded."
    ),
}
REPORTED_ARM = "scipy_numpy_svd"
REPORTED_PAIRS = (("candidate", REPORTED_ARM), (REPORTED_ARM, "incumbent"))
TARGET_ERROR_THRESHOLDS_DB = (0.05, 0.25)
GATE_RULES = ("S1 same_status", "S2 same_response", "S3 same_solver_verdicts", "S4 exceptions_are_small")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percent(part: int, whole: int) -> int:
    return 100 * part // max(whole, 1)


def nearest_rank(values: list[float], percent: int) -> float:
    ordered = sorted(values)
    return ordered[max(0, -(-percent * len(ordered) // 100) - 1)]


# ---------------------------------------------------------------- cases

def passage_names(corpus: Path) -> list[str]:
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    return sorted({f"{pair['speaker']}:{pair['statement_id']}" for pair in manifest["pairs"]})


def _case(audio: str, target: str, mode: str, smoothing: str) -> dict[str, str]:
    return {"id": f"{audio}|{target}|{mode}|{smoothing}", "audio": audio,
            "target": target, "mode": mode, "smoothing": smoothing}


def build_cases(names: list[str], indices: list[int], with_fixed_cases: bool) -> list[dict[str, str]]:
    cases = []
    for index in indices:
        for target in TARGETS:
            for mode in MODES:
                for smoothing in SMOOTHING:
                    cases.append(_case(names[index], target, mode, smoothing))
            cases.append(_case(f"{names[index]}:hot", target, "adaptive", "conservative"))
    if with_fixed_cases:
        cases += [_case(f"invalid:{kind}", "broadcast", "adaptive", "conservative") for kind in INVALID]
        cases += [_case(f"harness:{user}", target, "adaptive", "conservative")
                  for user in range(HARNESS_USERS) for target in TARGETS]
    return cases


def verify_corpora(corpus: Path, harness: Path) -> dict[str, str]:
    """Check every file of both manifests against its recorded SHA-256."""
    cross = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    setup = json.loads((harness / "manifest.json").read_text(encoding="utf-8"))
    entries = [(corpus, take) for pair in cross["pairs"] for take in pair["takes"].values()]
    entries += [(harness, item) for speaker in setup["speakers"].values() for item in speaker["files"]]
    entries += [(harness, item) for item in setup["noise"].values()]
    for root, entry in entries:
        path = root / entry["path"]
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise SystemExit(f"Corpus file is missing or its hash differs: {entry['path']}")
    return {"cross_take_manifest_sha256": sha256_file(corpus / "manifest.json"),
            "harness_manifest_sha256": sha256_file(harness / "manifest.json")}


# ---------------------------------------------------------------- worker (one arm, one process)

_TRACE: dict[str, Any] = {}


def _rounded(values: Any, digits: int) -> list[float]:
    return np.round(np.asarray(values, float), digits).tolist()


def _violations(gains: np.ndarray, lower: np.ndarray, upper: np.ndarray, limits: np.ndarray) -> dict[str, float | None]:
    if not np.all(np.isfinite(gains)):
        return {"bound_excess_db": None, "adjacency_excess_db": None}
    return {"bound_excess_db": float(max(0.0, np.max(lower - gains), np.max(gains - upper))),
            "adjacency_excess_db": float(max(0.0, np.max(np.abs(np.diff(gains)) - limits)))}


def _summary(result: Any, names: tuple[str, ...]) -> dict[str, Any]:
    """SciPy leaves some fields out of trivial results (SLSQP with no free variable has no ``status``)."""
    found = {name: getattr(result, name, None) for name in names}
    return {"success": bool(result.success), "message": str(getattr(result, "message", "")),
            **{name: None if value is None else int(value) for name, value in found.items()}}


def _bound_arrays(bounds: Any, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Limits from a least_squares ``(lower, upper)``, a minimize list of pairs, or ``Bounds``."""
    if bounds is None:
        return np.full(size, -np.inf), np.full(size, np.inf)
    if hasattr(bounds, "lb"):
        low, high = bounds.lb, bounds.ub
    elif len(bounds) == size and all(len(pair) == 2 for pair in bounds):  # ponytail: a 2-variable fit is ambiguous
        low = [-np.inf if pair[0] is None else pair[0] for pair in bounds]
        high = [np.inf if pair[1] is None else pair[1] for pair in bounds]
    else:
        low, high = bounds
    return np.broadcast_to(np.asarray(low, float), size), np.broadcast_to(np.asarray(high, float), size)


def _moved_start(x0: Any, bounds: Any) -> np.ndarray:
    """One unit in the last place up, clipped into the bounds (the control arm's perturbation)."""
    start = np.asarray(x0, float)
    low, high = _bound_arrays(bounds, start.size)
    return np.clip(np.nextafter(start, np.inf), low, high)


def _install_probes(accept_rejected: bool, perturb_start: bool = False, numpy_svd: bool = False) -> dict[str, bool]:
    """Record what each arm's solvers do without changing what they return.

    With ``perturb_start`` (the control arm) every ``least_squares`` and ``minimize`` call starts one unit
    in the last place higher than the source asked for.

    With ``numpy_svd`` (the reported ``scipy_numpy_svd`` arm) the ``svd`` that SciPy's ``trf`` solver calls
    (``from scipy.linalg import svd``; its ``qr`` is only used by the unused ``lsmr`` solver) is replaced by
    ``numpy.linalg.svd``, so the arm differs from the incumbent only in the LAPACK build behind the SVD.

    ``scipy.optimize`` is probed when present, so the SciPy arm exposes the raw SLSQP result that
    ``_constrained_gain_refinement`` discards when it rejects it. With ``accept_rejected`` the
    refinement returns that raw result instead, which replays the arm as if it had not been rejected.
    """
    from mic_eq.analysis.auto_eq_parts import optimizer, pipeline
    from mic_eq.analysis.auto_eq_parts.response import _predict_eq_response

    hooks = {"scipy": False, "refinement": False, "residuals": False, "solver_time": False}
    try:
        import scipy.optimize as scipy_optimize
    except ImportError:
        scipy_optimize = None
    if scipy_optimize is not None:
        least_squares, minimize = scipy_optimize.least_squares, scipy_optimize.minimize

        def moved(x0: Any, kwargs: dict[str, Any]) -> Any:
            if not perturb_start:
                return x0
            start = _moved_start(x0, kwargs.get("bounds"))
            _TRACE["starts"][0] += int(not np.array_equal(start, np.asarray(x0, float)))
            _TRACE["starts"][1] += 1
            return start

        def probed_least_squares(fun: Any, x0: Any, *args: Any, **kwargs: Any) -> Any:
            result = least_squares(fun, moved(x0, kwargs), *args, **kwargs)
            _TRACE["least_squares"].append(_summary(result, ("status", "nfev", "njev")))
            return result

        def probed_minimize(fun: Any, x0: Any, *args: Any, **kwargs: Any) -> Any:
            result = minimize(fun, moved(x0, kwargs), *args, **kwargs)
            _TRACE["minimize"].append(_summary(result, ("status", "nit", "nfev")) | {"x": np.asarray(result.x, float).tolist()})
            return result

        scipy_optimize.least_squares, scipy_optimize.minimize = probed_least_squares, probed_minimize
        hooks["scipy"] = True
        if numpy_svd:
            from scipy.optimize._lsq import trf

            def svd(matrix: Any, full_matrices: bool = True) -> Any:
                return np.linalg.svd(matrix, full_matrices=full_matrices)

            trf.svd = svd

    if hasattr(optimizer, "bounded_least_squares"):
        bounded_least_squares = optimizer.bounded_least_squares

        @functools.wraps(bounded_least_squares)
        def probed_bounded_least_squares(*args: Any, **kwargs: Any) -> Any:
            result = bounded_least_squares(*args, **kwargs)
            _TRACE["least_squares"].append(_summary(result, ("status", "nfev", "njev")))
            return result

        optimizer.bounded_least_squares = probed_bounded_least_squares

    def counting(function: Any, stage: str) -> Any:
        @functools.wraps(function)
        def counted(*args: Any, **kwargs: Any) -> Any:
            _TRACE["residual_calls"][stage] += 1
            return function(*args, **kwargs)

        return counted

    for name, stage in (("_gain_only_residuals", "stage1"), ("_joint_gain_q_residuals", "stage2")):
        if hasattr(optimizer, name):
            setattr(optimizer, name, counting(getattr(optimizer, name), stage))
            hooks["residuals"] = True

    if hasattr(optimizer, "_constrained_gain_refinement"):
        refinement = optimizer._constrained_gain_refinement
        signature = inspect.signature(refinement)
        if not {"gains", "qs", "centers_hz", "gain_lower", "gain_upper"} <= set(signature.parameters):
            raise SystemExit("_constrained_gain_refinement changed its parameters; update the probe")

        @functools.wraps(refinement)
        def probed_refinement(*args: Any, **kwargs: Any) -> Any:
            before = len(_TRACE["minimize"])
            gains, success = refinement(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            given = bound.arguments
            centers = np.asarray(given["centers_hz"], float)
            limits = optimizer._adjacent_gain_limits(centers)
            lower = given["gain_lower"]
            upper = given["gain_upper"]
            lower = np.full(10, optimizer.GAIN_MIN_DB) if lower is None else np.asarray(lower, float)
            upper = np.full(10, optimizer.GAIN_MAX_DB) if upper is None else np.asarray(upper, float)
            qs = np.asarray(given["qs"], float)

            def response(values: Any) -> list[float]:
                return _rounded(_predict_eq_response(RESPONSE_FREQS, values, qs, centers), 4)

            entry: dict[str, Any] = {"success": bool(success), **_violations(gains, lower, upper, limits),
                                     "response": response(gains), "raw": None}
            calls = _TRACE["minimize"][before:]
            if calls:  # the last SLSQP call is the refinement; the first is the feasible projection
                last = calls[-1]
                raw = np.asarray(last["x"], float)
                finite = bool(np.all(np.isfinite(raw)))
                violation = _violations(raw, lower, upper, limits)
                if not last["success"]:
                    reason = "solver_failure"
                elif not finite:
                    reason = "non_finite"
                elif (violation["adjacency_excess_db"] or 0.0) > FEASIBILITY_TOLERANCE:
                    reason = "adjacency_limit"
                else:
                    reason = "accepted"
                entry["raw"] = {k: last[k] for k in ("success", "status", "nit", "nfev", "message")}
                entry["raw"] |= {"reason": reason, **violation}
                if reason != "accepted":
                    entry["raw"]["response"] = response(raw) if finite else None
                if accept_rejected and not success and finite:
                    entry["counterfactual_accepted"] = True
                    _TRACE["refine"].append(entry)
                    return raw, True
            _TRACE["refine"].append(entry)
            return gains, success

        optimizer._constrained_gain_refinement = probed_refinement
        hooks["refinement"] = True

    calculate = pipeline.calculate_eq_bands

    @functools.wraps(calculate)
    def timed(*args: Any, **kwargs: Any) -> Any:
        wall, cpu = time.perf_counter(), time.process_time()
        result = calculate(*args, **kwargs)
        _TRACE["solver_seconds"] = time.perf_counter() - wall
        _TRACE["solver_cpu_seconds"] = time.process_time() - cpu
        _TRACE["pre_headroom"] = {key: result.get(key) for key in
                                  ("nonlinear_solver_success", "constraint_solver_success")}
        return result

    pipeline.calculate_eq_bands = timed
    hooks["solver_time"] = True
    return hooks


@functools.cache
def _passages(corpus: str) -> dict[str, np.ndarray]:
    from mic_eq.analysis.wav_io import read_mono_wav

    root = Path(corpus)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    parts: dict[str, list[np.ndarray]] = {}
    for pair in manifest["pairs"]:
        for take in pair["takes"].values():
            parts.setdefault(f"{pair['speaker']}:{pair['statement_id']}", []).append(
                read_mono_wav(root / take["path"])[1])
    return {name: np.concatenate(audio)[:480_000].astype(np.float32) for name, audio in sorted(parts.items())}


@functools.lru_cache(maxsize=4)
def case_audio(corpus: str, harness_root: str, name: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Capture and room noise for a passage, ``:hot``, ``invalid:`` or ``harness:`` case."""
    if name.startswith("harness:"):
        import evaluate_voice_setup as harness
        from mic_eq.ui.calibration_support import filtered_capture_for_analysis

        root = Path(harness_root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        user = harness.user(root, manifest, "fit", int(name.split(":")[1]))
        return (filtered_capture_for_analysis(user["setup_capture"], SAMPLE_RATE),
                filtered_capture_for_analysis(user["noise_capture"], SAMPLE_RATE))
    if name.startswith("invalid:"):
        speech = next(iter(_passages(corpus).values()))
        kind = name.split(":", 1)[1]
        if kind == "silence":
            return np.zeros(480_000, np.float32), None
        if kind == "noise":
            return (0.01 * np.random.default_rng(0).standard_normal(480_000)).astype(np.float32), None
        if kind == "clipped":
            return np.clip(speech * 40.0, -1.0, 1.0).astype(np.float32), None
        return speech[:24_000], None
    audio = _passages(corpus)[name.removesuffix(":hot")]
    if name.endswith(":hot"):
        audio = audio * np.float32(10 ** (-1 / 20) / np.max(np.abs(audio)))
    return audio, None


def run_case(case: dict[str, str], corpus: str, harness_root: str) -> dict[str, Any]:
    from mic_eq.analysis.auto_eq import analyze_auto_eq
    from mic_eq.analysis.auto_eq_parts import optimizer
    from mic_eq.analysis.auto_eq_parts.response import _predict_eq_response

    audio, noise = case_audio(corpus, harness_root, case["audio"])
    _TRACE.update(least_squares=[], minimize=[], refine=[], starts=[0, 0], residual_calls={"stage1": 0, "stage2": 0},
                  solver_seconds=None, solver_cpu_seconds=None, pre_headroom={})
    failure = None
    eq: dict[str, Any] = {}
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        eq, _validation = analyze_auto_eq(audio, SAMPLE_RATE, case["target"], target_mode=case["mode"],
                                          smoothing_strength=case["smoothing"], noise_audio=noise)
    except ValueError as error:
        failure = str(error)
    except Exception as error:  # a crashed arm is a result, not a reason to stop the comparison
        failure = f"{type(error).__name__}: {error}"
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    record: dict[str, Any] = {
        "id": case["id"], "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
        "noise_sha256": None if noise is None else hashlib.sha256(noise.tobytes()).hexdigest(),
        "failure": failure, "status": None, "response": None, "full_strength_response": None,
        "target_error_after_db": None,
        "fit_seconds": wall, "fit_cpu_seconds": cpu,
        "solver_seconds": _TRACE["solver_seconds"], "solver_cpu_seconds": _TRACE["solver_cpu_seconds"],
        "nonlinear_solver_success": _TRACE["pre_headroom"].get("nonlinear_solver_success"),
        "constraint_solver_success": _TRACE["pre_headroom"].get("constraint_solver_success"),
        "least_squares": _TRACE["least_squares"], "residual_calls": _TRACE["residual_calls"],
        "moved_starts": _TRACE["starts"],
        "refine": _TRACE["refine"],
    }
    if failure is None:
        gains, qs, centers = eq["band_gains"], eq["band_qs"], eq["band_freqs"]
        scale = float(eq["validation_gain_scale"])
        limits = optimizer._adjacent_gain_limits(np.asarray(centers, float))
        record |= {
            "status": eq["recommendation_status"], "gains": gains, "qs": qs, "centers": centers,
            "nonlinear_solver_success": eq["nonlinear_solver_success"],
            "constraint_solver_success": eq["constraint_solver_success"],
            "target_error_after_db": eq.get("validation_after_error_db"),
            "validation_gain_scale": scale, "headroom_gain_scale": float(eq["headroom_gain_scale"]),
            "recommendation_reasons": list(eq["recommendation_reasons"]),
            "abstention_reasons": list(eq["abstention_reasons"]),
            "final_adjacency_excess_db": _violations(
                np.asarray(gains, float), np.full(10, optimizer.GAIN_MIN_DB), np.full(10, optimizer.GAIN_MAX_DB),
                limits)["adjacency_excess_db"],
            "response": _rounded(_predict_eq_response(RESPONSE_FREQS, gains, qs, centers), 5),
        }
        if 0.0 < scale < 1.0:
            record["full_strength_response"] = _rounded(
                _predict_eq_response(RESPONSE_FREQS, np.asarray(gains) / scale, qs, centers), 5)
    return record


def run_worker(args: argparse.Namespace) -> int:
    import mic_eq
    from mic_eq.config import TARGET_CURVES

    stage = Path(args.stage).resolve()
    core = sys.modules.get("mic_eq.mic_eq_core")
    if not mic_eq.CORE_AVAILABLE or core is None:
        raise SystemExit("The native module did not load")
    core_path = Path(core.__file__ or "").resolve()
    if stage not in core_path.parents or sha256_file(core_path) != args.expect_native_sha256:
        raise SystemExit("The loaded native module is not the one given with --native-module")
    if stage not in Path(mic_eq.__file__).resolve().parents:
        raise SystemExit("mic_eq was not imported from the staged source root")
    if tuple(TARGET_CURVES) != TARGETS:
        raise SystemExit(f"Target curves changed: {list(TARGET_CURVES)}")
    sys.path.append(args.tools_dir)
    _install_probes(args.accept_rejected, args.perturb_start, args.numpy_svd)
    cases = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    with open(args.out, "a", encoding="utf-8") as out:
        for done, case in enumerate(cases, 1):
            record = run_case(case, args.corpus, args.harness_root) | {"run_key": args.run_key}
            out.write(json.dumps(record, allow_nan=False) + "\n")
            out.flush()
            if done % 50 == 0:
                print(f"  worker {Path(args.out).name}: {done}/{len(cases)}", file=sys.stderr, flush=True)
    return 0


# ---------------------------------------------------------------- orchestration

def tree_hashes(root: Path) -> dict[str, str]:
    files = {path for pattern in SOURCE_GLOBS for path in root.glob(pattern) if path.is_file()}
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(files)}


def pinned_sources(candidate_root: Path, repo_root: Path = REPO_ROOT, evaluator: Path = Path(__file__).resolve()
                   ) -> dict[str, str]:
    """SHA-256 of ``PINNED_SOURCES`` in the working tree, each checked against the file the arm used."""
    pins = {}
    for relative in PINNED_SOURCES:
        used = evaluator if relative == EVALUATOR_PATH else candidate_root / relative
        tracked = repo_root / relative
        if not used.is_file() or not tracked.is_file() or sha256_file(used) != sha256_file(tracked):
            raise SystemExit(f"{relative} in the measurement is not the working tree's file, so the report "
                             "could not be verified; copy the working tree's file into the candidate root")
        pins[relative] = sha256_file(tracked)
    return pins


def stage_arm(root: Path, native: Path, runtime_root: Path, destination: Path) -> Path:
    """Copy ``root``'s package beside the shared native module and CPU ONNX Runtime."""
    shutil.copytree(root / "python" / "mic_eq", destination / "python" / "mic_eq",
                    ignore=shutil.ignore_patterns("__pycache__", "mic_eq_core*"))
    shutil.copy2(native, destination / "python" / "mic_eq" / native.name)
    shutil.copytree(runtime_root / "target" / "onnxruntime-cpu" / "lib",
                    destination / "target" / "onnxruntime-cpu" / "lib")
    return destination


def load_partials(prefix: Path, run_key: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name in sorted(glob.glob(f"{prefix}.*.partial.jsonl")):
        for line in Path(name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue  # an interrupted write leaves at most the last line incomplete
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record["run_key"] != run_key:
                raise SystemExit(f"{name} was written by a different measurement; delete it to start over")
            records[record["id"]] = record
    return records


def shard(cases: list[dict[str, str]], count: int) -> list[list[dict[str, str]]]:
    """Keep cases that share a capture together (fits reuse its audio), balanced by size."""
    groups: dict[str, list[dict[str, str]]] = {}
    for case in cases:
        groups.setdefault(case["audio"].removesuffix(":hot"), []).append(case)
    shards: list[list[dict[str, str]]] = [[] for _ in range(count)]
    for group in sorted(groups.values(), key=len, reverse=True):
        min(shards, key=len).extend(group)
    return [item for item in shards if item]


def run_arm(label: str, stage: Path, cases: list[dict[str, str]], prefix: Path, run_key: str,
            args: argparse.Namespace, native_sha256: str, scratch: Path, accept_rejected: bool = False,
            perturb_start: bool = False, numpy_svd: bool = False) -> dict[str, dict[str, Any]]:
    done = load_partials(prefix, run_key)
    pending = [case for case in cases if case["id"] not in done]
    print(f"{label}: {len(done)} cases resumed, {len(pending)} to run", file=sys.stderr, flush=True)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("AUDIOFORGE_", "DEEPFILTER_"))}
    environment |= {"PYTHONPATH": str(stage / "python"), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1", "VAD_MODEL_PATH": str(args.runtime_root / "models" / "silero_vad.onnx")}
    processes = []
    try:
        for index, jobs in enumerate(shard(pending, args.workers)):
            jobs_file = scratch / f"{label}.{index}.jobs.json"
            jobs_file.write_text(json.dumps(jobs), encoding="utf-8")
            command = [sys.executable, "-P", str(Path(__file__).resolve()), "--worker", "--stage", str(stage),
                       "--jobs", str(jobs_file), "--out", f"{prefix}.{index}.partial.jsonl",
                       "--expect-native-sha256", native_sha256, "--corpus", str(args.corpus),
                       "--harness-root", str(args.harness_root),
                       "--tools-dir", str(Path(__file__).resolve().parent), "--run-key", run_key]
            command += ["--accept-rejected"] * accept_rejected + ["--perturb-start"] * perturb_start
            command += ["--numpy-svd"] * numpy_svd
            processes.append(subprocess.Popen(command, env=environment))
        codes = [process.wait() for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
    if any(codes):
        raise SystemExit(f"{label} worker failed with exit codes {codes}; rerun the same command to resume")
    done = load_partials(prefix, run_key)
    missing = [case["id"] for case in cases if case["id"] not in done]
    if missing:
        raise SystemExit(f"{label} produced no result for {len(missing)} cases, first {missing[0]}")
    return done


# ---------------------------------------------------------------- gates

def peak_difference(first: list[float], second: list[float]) -> float:
    return float(np.max(np.abs(np.asarray(first) - np.asarray(second))))


def feasible(record: dict[str, Any]) -> bool:
    """The record's constrained-gain result meets the bounds and adjacency limits (False if unknown)."""
    if record["refine"]:
        return all((call["bound_excess_db"] or 0.0) <= FEASIBILITY_TOLERANCE
                   and (call["adjacency_excess_db"] or 0.0) <= FEASIBILITY_TOLERANCE for call in record["refine"])
    excess = record.get("final_adjacency_excess_db")
    return excess is not None and excess <= FEASIBILITY_TOLERANCE


def s1_exception(incumbent: dict[str, Any], other: dict[str, Any]) -> dict[str, Any] | None:
    """The allowed status change, with SciPy's rejected or failed result, or None."""
    if (incumbent["failure"] or other["failure"] or incumbent["status"] != "reduced" or other["status"] != "apply"
            or set(incumbent["recommendation_reasons"]) != {CONSTRAINT_REASON}):
        return None
    rejected = [call for call in incumbent["refine"] if not call["success"] and call["raw"]]
    if not rejected or not feasible(other):
        return None
    raw = rejected[0]["raw"]  # the first rejection; a second refinement may be rejected too
    kind = raw["reason"]
    if kind == "solver_failure":
        kind = "iteration_cap" if raw["status"] == 9 else "exit_failure"
    return {"scipy_failure": kind, "scipy_adjacency_excess_db": raw["adjacency_excess_db"],
            "scipy_bound_excess_db": raw["bound_excess_db"], "scipy_status": raw["status"],
            "scipy_message": raw["message"], "rejected_refinements": len(rejected),
            "rejected_call_index": incumbent["refine"].index(rejected[0])}


def s4_distance(incumbent: dict[str, Any], candidate: dict[str, Any], replay: dict[str, Any] | None,
                info: dict[str, Any]) -> dict[str, Any]:
    """Distance of the new arm's final response from SciPy's response had its result not been rejected."""
    final = candidate["response"]
    result: dict[str, Any] = {"form": None, "distance_db": None}
    if replay is not None and not replay["failure"] and replay["response"] is not None:
        result |= {"form": "replay_final", "distance_db": peak_difference(final, replay["response"]),
                   "replay_status": replay["status"]}
    else:
        scaled_back = incumbent["full_strength_response"] or incumbent["response"]
        if scaled_back is not None:
            result |= {"form": "scaled_back_final", "distance_db": peak_difference(final, scaled_back)}
    # Reported with the gate value: like for like, both before validation, and both at full strength.
    raw = incumbent["refine"][info["rejected_call_index"]]["raw"].get("response")
    index = min(info["rejected_call_index"], len(candidate["refine"]) - 1)
    if raw is not None and index >= 0:
        result["refinement_output_distance_db"] = peak_difference(candidate["refine"][index]["response"], raw)
    scaled_candidate = candidate["full_strength_response"] or candidate["response"]
    scaled_incumbent = incumbent["full_strength_response"] or incumbent["response"]
    result["full_strength_distance_db"] = peak_difference(scaled_candidate, scaled_incumbent)
    return result


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": sum(values) / len(values), "median": nearest_rank(values, 50),
            "p95": nearest_rank(values, 95), "max": max(values)}


FLAGS = ("constraint_solver_success", "nonlinear_solver_success")


def direction(before: Any, after: Any) -> str:
    return f"{str(before).lower()}->{str(after).lower()}"


def pair_difference(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    """Every case behind the gate counts for ``arm`` versus the incumbent."""
    result: dict[str, Any] = {"failure_changes": [], "status_changes": [], "exceptions": {}, "flag_changes": [],
                              "infeasible_repairs": [], "differences": {}}
    for row in rows:
        case_id, first, second = row["case"]["id"], row["incumbent"], row[arm]
        if first["failure"] != second["failure"]:
            result["failure_changes"].append({"case": case_id, "from": first["failure"], "to": second["failure"]})
        elif first["status"] != second["status"]:
            result["status_changes"].append({"case": case_id, "from": first["status"], "to": second["status"]})
        info = s1_exception(first, second)
        if info is not None:
            result["exceptions"][case_id] = info
        for flag in FLAGS:
            if first[flag] != second[flag]:
                result["flag_changes"].append(
                    {"case": case_id, "flag": flag, "direction": direction(first[flag], second[flag])})
                if flag == FLAGS[0] and second[flag] is True and not feasible(second):
                    result["infeasible_repairs"].append(case_id)
        if info is None and not first["failure"] and not second["failure"]:
            result["differences"][case_id] = peak_difference(first["response"], second["response"])
    result["other_status_changes"] = [change for change in result["status_changes"]
                                      if change["case"] not in result["exceptions"]]
    return result


def pair_counts(pair: dict[str, Any]) -> dict[str, Any]:
    differences = list(pair["differences"].values())
    constraint = [change["direction"] for change in pair["flag_changes"] if change["flag"] == FLAGS[0]]
    nonlinear = [change["direction"] for change in pair["flag_changes"] if change["flag"] == FLAGS[1]]
    return {
        "failure_changes": len(pair["failure_changes"]), "status_changes": len(pair["status_changes"]),
        "allowed_exceptions": len(pair["exceptions"]), "other_status_changes": len(pair["other_status_changes"]),
        "response_cases": len(differences), "within_0.05_db": sum(d <= TYPICAL_TOLERANCE_DB for d in differences),
        **{f"over_{limit}_db": sum(d > limit for d in differences) for limit in EXCEED_THRESHOLDS_DB},
        "max_difference_db": max(differences, default=None),
        "constraint_true_to_false": constraint.count("true->false"),
        "constraint_false_to_true": constraint.count("false->true"),
        "constraint_other_changes": len(constraint) - constraint.count("true->false") - constraint.count("false->true"),
        "constraint_repairs_not_feasible": len(pair["infeasible_repairs"]),
        "nonlinear_differ": len(nonlinear), "nonlinear_true_to_false": nonlinear.count("true->false"),
        "nonlinear_false_to_true": nonlinear.count("false->true"),
    }


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the amended gates to ``rows`` of {case, incumbent, candidate, control, replay}."""
    port, control = pair_difference(rows, "candidate"), pair_difference(rows, "control")
    mine, theirs = pair_counts(port), pair_counts(control)
    exceptions = port["exceptions"]
    distances = {}
    for row in rows:
        case_id = row["case"]["id"]
        if case_id in exceptions:
            distances[case_id] = s4_distance(row["incumbent"], row["candidate"], row["replay"], exceptions[case_id])
    unmeasured = [case_id for case_id, item in distances.items() if item["distance_db"] is None]
    too_far = [case_id for case_id, item in distances.items()
               if item["distance_db"] is not None and item["distance_db"] > EXCEPTION_TOLERANCE_DB]
    exceeding = {str(limit): {"port": mine[f"over_{limit}_db"], "control": theirs[f"over_{limit}_db"],
                              "allowed": EXCEED_FACTOR * theirs[f"over_{limit}_db"] + EXCEED_ALLOWANCE}
                 for limit in EXCEED_THRESHOLDS_DB}
    nonlinear_allowed = EXCEED_FACTOR * theirs["nonlinear_differ"] + NONLINEAR_ALLOWANCE
    gates = {
        "S1": {"passed": not mine["failure_changes"] and mine["other_status_changes"] <= theirs["status_changes"],
               "cases": len(rows), "port": mine, "control": theirs,
               "other_status_changes_allowed": theirs["status_changes"]},
        "S2": {"passed": 100 * mine["within_0.05_db"] >= TYPICAL_PERCENT * mine["response_cases"]
               and all(item["port"] <= item["allowed"] for item in exceeding.values()),
               "port_cases": mine["response_cases"], "control_cases": theirs["response_cases"],
               "port_within_0.05_db": mine["within_0.05_db"],
               "port_within_0.05_db_percent": percent(mine["within_0.05_db"], mine["response_cases"]),
               "control_within_0.05_db": theirs["within_0.05_db"],
               "control_within_0.05_db_percent": percent(theirs["within_0.05_db"], theirs["response_cases"]),
               "port_max_difference_db": mine["max_difference_db"],
               "control_max_difference_db": theirs["max_difference_db"], "exceeding": exceeding},
        "S3": {"passed": not mine["constraint_true_to_false"] and not mine["constraint_other_changes"]
               and not mine["constraint_repairs_not_feasible"] and mine["nonlinear_differ"] <= nonlinear_allowed,
               "port_constraint_true_to_false": mine["constraint_true_to_false"],
               "port_constraint_false_to_true": mine["constraint_false_to_true"],
               "port_constraint_repairs_not_feasible": mine["constraint_repairs_not_feasible"],
               "control_constraint_true_to_false": theirs["constraint_true_to_false"],
               "control_constraint_false_to_true": theirs["constraint_false_to_true"],
               "port_nonlinear_differ": mine["nonlinear_differ"],
               "port_nonlinear_true_to_false": mine["nonlinear_true_to_false"],
               "port_nonlinear_false_to_true": mine["nonlinear_false_to_true"],
               "control_nonlinear_differ": theirs["nonlinear_differ"],
               "nonlinear_differ_allowed": nonlinear_allowed},
        "S4": {"passed": not unmeasured and not too_far, "exceptions": len(exceptions),
               "measured_by_replay": sum(item["form"] == "replay_final" for item in distances.values()),
               "measured_scaled_back": sum(item["form"] == "scaled_back_final" for item in distances.values()),
               "not_measurable": unmeasured, "over_1.0_db": too_far,
               "max_distance_db": max((item["distance_db"] for item in distances.values()
                                       if item["distance_db"] is not None), default=None)},
    }
    return {"gates": gates, "port": port, "control": control, "exceptions": exceptions, "distances": distances}


def _nfev(record: dict[str, Any]) -> list[int | None]:
    return [call["nfev"] for call in record["least_squares"]]


def reported_pair(rows: list[dict[str, Any]], first: str, second: str, capped: bool) -> dict[str, Any]:
    """Response differences, status changes and SciPy-style effort between two arms (never gated)."""
    differences: dict[str, float] = {}
    status_changes = 0
    same_nfev = [0, 0, 0]  # stage 1, stage 2, every stage
    for row in rows:
        one, other = row[first], row[second]
        status_changes += (one["failure"], one["status"]) != (other["failure"], other["status"])
        if not one["failure"] and not other["failure"]:
            differences[row["case"]["id"]] = peak_difference(one["response"], other["response"])
        efforts = _nfev(one), _nfev(other)
        same_nfev[0] += efforts[0][:1] == efforts[1][:1]
        same_nfev[1] += efforts[0][1:2] == efforts[1][1:2]
        same_nfev[2] += efforts[0] == efforts[1]
    values = list(differences.values())
    over = sorted((case for case, value in differences.items() if value > TYPICAL_TOLERANCE_DB),
                  key=lambda case: -differences[case])
    return {
        "cases": len(rows), "response_cases": len(values), "status_changes": status_changes,
        "within_0.05_db": sum(value <= TYPICAL_TOLERANCE_DB for value in values),
        **{f"over_{limit}_db": sum(value > limit for value in values) for limit in EXCEED_THRESHOLDS_DB},
        "max_difference_db": max(values, default=None),
        "nfev_identical_stage1": same_nfev[0], "nfev_identical_stage2": same_nfev[1],
        "nfev_identical_all_stages": same_nfev[2],
        "cases_over_0.05_db": listed([f"{case}: {differences[case]:.4g} dB" for case in over])
        if capped else {case: differences[case] for case in over},
    }


def reported_target_error(rows: list[dict[str, Any]], arms: list[str]) -> dict[str, Any]:
    """Final weighted error to the target curve (``validation_after_error_db``) of each arm versus the incumbent."""
    result: dict[str, Any] = {}
    for arm in arms:
        pairs = [(row["incumbent"]["target_error_after_db"], row[arm]["target_error_after_db"]) for row in rows
                 if row["incumbent"]["target_error_after_db"] is not None
                 and row[arm]["target_error_after_db"] is not None]
        change = [after - before for before, after in pairs]
        result[arm] = {
            "cases": len(pairs),
            "mean_db": sum(after for _, after in pairs) / len(pairs) if pairs else None,
            "incumbent_mean_db": sum(before for before, _ in pairs) / len(pairs) if pairs else None,
            **{f"worse_by_over_{limit}_db": sum(value > limit for value in change)
               for limit in TARGET_ERROR_THRESHOLDS_DB},
            **{f"better_by_over_{limit}_db": sum(value < -limit for value in change)
               for limit in TARGET_ERROR_THRESHOLDS_DB},
        }
    return result


def reported(rows: list[dict[str, Any]], capped: bool) -> dict[str, Any]:
    """The reported-only block: neither it nor its arm enters a gate or the decision text."""
    if REPORTED_ARM not in rows[0]:
        return {"skipped": True}
    return {
        "note": "Reported only; not gated and not part of the decision.",
        "inputs_identical": all(row["incumbent"][key] == row[REPORTED_ARM][key] for row in rows
                                for key in ("audio_sha256", "noise_sha256")),
        **{f"{first}_vs_{second}": reported_pair(rows, first, second, capped) for first, second in REPORTED_PAIRS},
        "target_error_after_vs_incumbent": reported_target_error(rows, ["candidate", "control", REPORTED_ARM]),
    }


def decision_text(scope: str, inputs_identical: bool, evaluation: dict[str, Any]) -> tuple[str, str]:
    gates = evaluation["gates"]
    failed = [name for name, gate in gates.items() if not gate["passed"]]
    port, control = gates["S1"]["port"], gates["S1"]["control"]
    counts = (f"port difference: {port['status_changes']} status changes ({port['allowed_exceptions']} allowed "
              f"exceptions), {port['failure_changes']} failure changes; control difference: "
              f"{control['status_changes']} status changes; cases over 0.25 dB: {port['over_0.25_db']} port, "
              f"{control['over_0.25_db']} control; constraint verdict true to false: "
              f"{port['constraint_true_to_false']}; nonlinear verdicts differing: {port['nonlinear_differ']} "
              f"port, {control['nonlinear_differ']} control")
    if not inputs_identical:
        return "comparison_invalid", "The arms did not receive identical input audio, so no gate was judged."
    if scope != "judged":
        return (f"not_judged_{scope}_run",
                f"A {scope} run is not judged ({counts}); gates failed: {', '.join(failed) or 'none'}.")
    if failed:
        return ("retain_scipy",
                f"Gates {', '.join(failed)} failed ({counts}), so SciPy stays in the runtime for this release.")
    exceptions = port["allowed_exceptions"]
    note = f" The {exceptions} exceptions are listed for individual confirmation." if exceptions else ""
    return ("remove_scipy", f"Gates S1 to S4 all passed ({counts}).{note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--incumbent-root", type=Path, help="source root with the SciPy solvers")
    parser.add_argument("--candidate-root", type=Path, help="source root with the NumPy solvers")
    parser.add_argument("--native-module", type=Path, help="mic_eq_core*.pyd both arms load")
    parser.add_argument("--runtime-root", type=Path, default=REPO_ROOT,
                        help="holds target/onnxruntime-cpu/lib and models/silero_vad.onnx")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--harness-root", type=Path, default=DEFAULT_HARNESS)
    parser.add_argument("--workers", type=int, default=3, help=f"per arm, at most {MAX_WORKERS}")
    parser.add_argument("--indices", type=lambda text: [int(item) for item in text.split(",")],
                        help="run only these passage indices (no invalid or harness cases)")
    parser.add_argument("--dev-only", action="store_true", help="the four development passages")
    parser.add_argument("--with-fixed-cases", action="store_true",
                        help="add the invalid and harness cases to a run with --indices or --dev-only")
    parser.add_argument("--match", type=lambda text: text.split(","),
                        help="run only cases whose id contains one of these texts (smoke tests)")
    parser.add_argument("--limit", type=int, help="run only the first N cases (smoke tests)")
    parser.add_argument("--skip-reported-arms", action="store_true",
                        help="do not run the reported-only scipy_numpy_svd arm")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--details-output", type=Path)
    for name in ("--worker", "--accept-rejected", "--perturb-start", "--numpy-svd"):
        parser.add_argument(name, action="store_true", help=argparse.SUPPRESS)
    for name in ("--stage", "--jobs", "--out", "--expect-native-sha256", "--tools-dir", "--run-key"):
        parser.add_argument(name, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return run_worker(args)
    for name in ("incumbent_root", "candidate_root", "native_module", "report", "details_output"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be between 1 and {MAX_WORKERS}")
    if not args.native_module.name.startswith("mic_eq_core") or args.native_module.suffix != ".pyd":
        parser.error("--native-module must be a mic_eq_core*.pyd file")
    for root in (args.incumbent_root, args.candidate_root):
        if not (root / "python" / "mic_eq").is_dir():
            parser.error(f"{root} has no python/mic_eq")
    if args.details_output.resolve().is_relative_to(REPO_ROOT / "evaluation"):
        parser.error("--details-output holds every case and must stay out of evaluation/; "
                     "use models/evaluation-details/")
    return compare(args)


def compare(args: argparse.Namespace) -> int:
    names = passage_names(args.corpus)
    judged = [index for index in range(len(names)) if index not in DEV_PASSAGES]
    indices: list[int] = list(DEV_PASSAGES) if args.dev_only else args.indices or judged
    if len(names) != 48 or not set(indices) <= set(range(len(names))):
        raise SystemExit(f"Expected 48 cross-take passages with indices {indices} among them, found {len(names)}")
    explicit = args.dev_only or args.indices is not None
    cases = build_cases(names, indices, with_fixed_cases=not explicit or args.with_fixed_cases)
    if not explicit and len(cases) != 2084:
        raise SystemExit(f"Expected 2,084 cases, built {len(cases)}")
    scope = ("judged" if sorted(indices) == judged and not explicit
             else "development" if set(indices) <= set(DEV_PASSAGES) else "partial")
    if args.match is not None:
        cases = [case for case in cases if any(text in case["id"] for text in args.match)]
    if args.limit is not None:
        cases = cases[: args.limit]
    if args.match is not None or args.limit is not None:
        scope = "partial"
    if not cases:
        raise SystemExit("No cases selected")
    pins = pinned_sources(args.candidate_root)
    manifests = verify_corpora(args.corpus, args.harness_root)
    native_sha256 = sha256_file(args.native_module)
    arm_hashes = {"incumbent": tree_hashes(args.incumbent_root), "candidate": tree_hashes(args.candidate_root)}
    evaluator_sha256 = sha256_file(Path(__file__).resolve())
    args.details_output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results: dict[str, dict[str, dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory(prefix="solver-port-") as temporary:
        scratch = Path(temporary)
        stages = {arm: stage_arm(getattr(args, f"{arm}_root"), args.native_module, args.runtime_root,
                                 scratch / arm) for arm in ("incumbent", "candidate")}
        keys = {arm: hashlib.sha256(json.dumps([arm_hashes[arm], native_sha256, manifests, evaluator_sha256,
                                                args.limit, args.match]).encode()).hexdigest() for arm in stages}
        for arm, stage in stages.items():
            prefix = args.details_output.parent / f"{args.details_output.stem}.{arm}"
            results[arm] = run_arm(arm, stage, cases, prefix, keys[arm], args, native_sha256, scratch)
        # The control arm is the incumbent source with every solver start moved one unit in the last place.
        prefix = args.details_output.parent / f"{args.details_output.stem}.control"
        results["control"] = run_arm("control", stages["incumbent"], cases, prefix, keys["incumbent"] + "c", args,
                                     native_sha256, scratch, perturb_start=True)
        # Replay the SciPy arm with its rejected result accepted, on the cases that look like S1 exceptions.
        looks_allowed = [case for case in cases
                         if s1_exception(results["incumbent"][case["id"]], results["candidate"][case["id"]])]
        prefix = args.details_output.parent / f"{args.details_output.stem}.replay"
        replay = (run_arm("replay", stages["incumbent"], looks_allowed, prefix, keys["incumbent"] + "r", args,
                          native_sha256, scratch, accept_rejected=True) if looks_allowed else {})
        # Reported only: the incumbent source with numpy.linalg.svd behind SciPy's trust-region solver.
        if not args.skip_reported_arms:
            prefix = args.details_output.parent / f"{args.details_output.stem}.{REPORTED_ARM}"
            results[REPORTED_ARM] = run_arm(REPORTED_ARM, stages["incumbent"], cases, prefix,
                                            keys["incumbent"] + "n", args, native_sha256, scratch, numpy_svd=True)
    rows = [{"case": case, "incumbent": results["incumbent"][case["id"]],
             "candidate": results["candidate"][case["id"]], "control": results["control"][case["id"]],
             "replay": replay.get(case["id"])}
            | ({REPORTED_ARM: results[REPORTED_ARM][case["id"]]} if REPORTED_ARM in results else {})
            for case in cases]
    inputs_identical = all(row["incumbent"][key] == row[arm][key] for row in rows
                           for arm in ("candidate", "control") for key in ("audio_sha256", "noise_sha256"))
    evaluation = evaluate(rows)
    decision, summary = decision_text(scope, inputs_identical, evaluation)
    report = build_report(args, cases, scope, indices, rows, evaluation, decision, summary, arm_hashes,
                          native_sha256, manifests, evaluator_sha256, pins, time.perf_counter() - started)
    write_json(args.details_output, {"report": args.report.name, "decision": decision, "rows": rows,
                                     "s1_exceptions": evaluation["exceptions"], "s4": evaluation["distances"],
                                     "reported": reported(rows, capped=False),
                                     "port_difference": {k: v for k, v in evaluation["port"].items() if k != "differences"}
                                     | {"response_differences_db": evaluation["port"]["differences"]},
                                     "control_difference": {k: v for k, v in evaluation["control"].items()
                                                            if k != "differences"}
                                     | {"response_differences_db": evaluation["control"]["differences"]}},
               indent=None)
    write_json(args.report, report, limit=MAX_REPORT_BYTES)
    for name in glob.glob(f"{args.details_output.parent / args.details_output.stem}.*.partial.jsonl"):
        os.remove(name)
    sys.stdout.write(json.dumps({"decision": decision, "summary": summary,
                                 "gates": {name: gate["passed"] for name, gate in evaluation["gates"].items()}},
                                indent=1) + "\n")
    return 0


def write_json(path: Path, payload: dict[str, Any], indent: int | None = 1, limit: int | None = None) -> None:
    text = json.dumps(payload, indent=indent, allow_nan=False) + "\n"
    if limit is not None and len(text.encode()) > limit:
        raise SystemExit(f"{path.name} would be {len(text.encode())} bytes, over the {limit} that a tracked report "
                         "may have; lower LIST_LIMIT (the details file is already written)")
    path.write_text(text, encoding="utf-8")


def distribution(differences: list[float]) -> dict[str, Any]:
    edges = (0.001, 0.01, 0.05, 0.1, 0.25, 1.0)
    return {"cases": len(differences), "max": max(differences, default=None),
            "median": nearest_rank(differences, 50) if differences else None,
            "p99": nearest_rank(differences, 99) if differences else None,
            "histogram": {f"<= {edge}": sum(d <= edge for d in differences) for edge in edges}
            | {f"> {edges[-1]}": sum(d > edges[-1] for d in differences)}}


def listed(entries: list[str]) -> dict[str, Any]:
    return {"count": len(entries), "listed": entries[:LIST_LIMIT], "not_listed": max(len(entries) - LIST_LIMIT, 0)}


def status_entries(pair: dict[str, Any]) -> list[str]:
    return [f"{change['case']}: {change['from']} -> {change['to']}" for change in pair["status_changes"]]


def flag_entries(pair: dict[str, Any]) -> list[str]:
    return [f"{change['case']}: {change['flag'].removesuffix('_solver_success')} {change['direction']}"
            for change in pair["flag_changes"]]


def evaluation_contract(evaluation: dict[str, Any], args: argparse.Namespace, cases: int, native_sha256: str,
                        manifests: dict[str, str], pins: dict[str, str]) -> dict[str, Any]:
    """The audible-change contract of check_evaluation_hygiene, for an offline fit (no audio path)."""
    mine, theirs = pair_counts(evaluation["port"]), pair_counts(evaluation["control"])
    return {
        "configuration": {
            "scope": "offline Auto-EQ fit (analyze_auto_eq), SciPy solvers against NumPy solvers, one native module",
            "sample_rate": SAMPLE_RATE, "cases": cases, "targets": list(TARGETS), "modes": list(MODES),
            "smoothing": list(SMOOTHING),
        },
        "asset_hashes": {"native_module": {args.native_module.name: native_sha256}, "corpora": manifests,
                         "source": pins},
        "runtime": {
            "max_p99_frame_seconds": None,
            "max_p99_frame_seconds_reason": "The fit is an offline whole-capture analysis, not a realtime "
                                            "callback; fit times per arm are in results.run_time.",
        },
        "latency": {"added_realtime_latency_samples": 0,
                    "reason": "Only the offline analysis changes; the audio path and its latency are untouched."},
        "clean_preservation": {
            "audible_change_basis": f"true when a case's recommendation status changes or its final response "
                                    f"differs from the incumbent's by more than {TYPICAL_TOLERANCE_DB} dB at some "
                                    "frequency (the S2 tolerance)",
            "port_status_changes": mine["status_changes"], "port_cases_over_0.05_db": mine["over_0.05_db"],
            "port_max_difference_db": mine["max_difference_db"],
            "control_cases_over_0.05_db": theirs["over_0.05_db"],
            "control_max_difference_db": theirs["max_difference_db"],
            "note": "The control arm is the incumbent with every solver start moved one unit in the last place; "
                    "its difference from the incumbent is the incumbent's own sensitivity. Whether the port "
                    "stays within it is decided by gates S1 and S2, not by this field.",
        },
    }


def build_report(args: argparse.Namespace, cases: list[dict[str, str]], scope: str, indices: list[int],
                 rows: list[dict[str, Any]], evaluation: dict[str, Any], decision: str, summary: str,
                 arm_hashes: dict[str, dict[str, str]], native_sha256: str, manifests: dict[str, str],
                 evaluator_sha256: str, pins: dict[str, str], elapsed_seconds: float) -> dict[str, Any]:
    timing = {}
    effort = {}
    for arm in ("incumbent", "candidate", "control", *([REPORTED_ARM] if REPORTED_ARM in rows[0] else [])):
        records = [row[arm] for row in rows]
        timing[arm] = {
            "fit_seconds": summarize([record["fit_seconds"] for record in records]),
            "fit_cpu_seconds": summarize([record["fit_cpu_seconds"] for record in records]),
            "solver_seconds": summarize([record["solver_seconds"] for record in records
                                         if record["solver_seconds"] is not None]),
        }
        first = [r["least_squares"][0]["nfev"] for r in records if r["least_squares"]]
        second = [r["least_squares"][1]["nfev"] for r in records if len(r["least_squares"]) > 1]
        effort[arm] = {
            "residual_calls_stage1": summarize([r["residual_calls"]["stage1"] for r in records if r["residual_calls"]["stage1"]]),
            "residual_calls_stage2": summarize([r["residual_calls"]["stage2"] for r in records if r["residual_calls"]["stage2"]]),
            "least_squares_nfev_stage1": summarize([value for value in first if value is not None]),
            "least_squares_nfev_stage2": summarize([value for value in second if value is not None]),
            "slsqp_results_rejected": sum(1 for r in records for call in r["refine"]
                                          if call["raw"] and call["raw"]["reason"] != "accepted"),
        }
    moved = [row["control"]["moved_starts"] for row in rows]
    differing = {path: {"incumbent": arm_hashes["incumbent"].get(path), "candidate": arm_hashes["candidate"].get(path)}
                 for path in sorted(set(arm_hashes["incumbent"]) | set(arm_hashes["candidate"]))
                 if arm_hashes["incumbent"].get(path) != arm_hashes["candidate"].get(path)}
    port, control = evaluation["port"], evaluation["control"]
    exceptions = [
        f"{case_id}: {info['scipy_failure']}, SciPy adjacency excess {info['scipy_adjacency_excess_db']:.3g} dB, "
        f"bound excess {info['scipy_bound_excess_db']:.3g} dB, S4 distance "
        f"{evaluation['distances'][case_id]['distance_db']} dB ({evaluation['distances'][case_id]['form']})"
        for case_id, info in evaluation["exceptions"].items()]
    contract = evaluation_contract(evaluation, args, len(cases), native_sha256, manifests, pins)
    return {
        "schema_version": 2,
        "audible_change": any(contract["clean_preservation"][key] for key in AUDIBLE_COUNTS),
        "evaluation_contract": contract,
        "source_sha256": pins,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "decision": decision,
        "summary": summary,
        "predefined_rules": {
            "status": "The rule as amended by Amendment 1; gates S1 to S3 of the first version "
                      "(fixed 2026-10-14 00:55 +0300) are superseded, S4 stands.",
            "amendment_1": AMENDMENT,
            **RULES,
        },
        "cases": {"total": len(cases), "passage_indices": sorted(indices),
                  "development_passages_excluded": list(DEV_PASSAGES),
                  "composition": "44 cross-take passages x 5 targets x 2 modes x 4 smoothing; their -1 dBFS "
                                 "versions (5 targets, adaptive, conservative); 4 invalid captures; 20 fit-split "
                                 "harness captures x 5 targets"},
        "results": {
            "gates": {name: gate | {"rule": GATE_RULES[index]}
                      for index, (name, gate) in enumerate(evaluation["gates"].items())},
            "s1_exceptions": listed(exceptions),
            "port_difference": {
                "failure_changes": listed([f"{c['case']}: {c['from']} -> {c['to']}" for c in port["failure_changes"]]),
                "other_status_changes": listed(status_entries({"status_changes": port["other_status_changes"]})),
                "flag_changes": listed(flag_entries(port)),
                "response_difference_db": distribution(list(port["differences"].values())),
            },
            "control_difference": {
                "failure_changes": listed([f"{c['case']}: {c['from']} -> {c['to']}" for c in control["failure_changes"]]),
                "status_changes": listed(status_entries(control)),
                "allowed_exception_cases": len(control["exceptions"]),
                "flag_changes": listed(flag_entries(control)),
                "response_difference_db": distribution(list(control["differences"].values())),
            },
            "control_arm": {"solver_calls_with_moved_start": sum(item[0] for item in moved),
                            "solver_calls": sum(item[1] for item in moved)},
            "reported": reported(rows, capped=True),
            "run_time": timing,
            "solver_effort": effort,
            "elapsed_seconds": elapsed_seconds,
        },
        "measurement_identity": {
            "evaluator_sha256": evaluator_sha256,
            "native_module": args.native_module.name,
            "native_module_sha256": native_sha256,
            "corpora": manifests,
            "response_grid": "257 log-spaced frequencies, 80 Hz to 16 kHz, _predict_eq_response",
            "incumbent_source_tree_sha256": hashlib.sha256(json.dumps(arm_hashes["incumbent"]).encode()).hexdigest(),
            "candidate_source_tree_sha256": hashlib.sha256(json.dumps(arm_hashes["candidate"]).encode()).hexdigest(),
            "control_arm": "incumbent source tree; every least_squares and minimize start moved by "
                           "numpy.nextafter toward +infinity, clipped into the bounds, by an interposed wrapper",
            "scipy_numpy_svd_arm": "reported only; incumbent source tree, no start nudge, with the svd that "
                                   "scipy.optimize._lsq.trf calls replaced by numpy.linalg.svd",
            "files_that_differ_between_arms": differing,
            "source_sha256_pins": "the candidate arm's files, each identical in the working tree; the incumbent "
                                  "arm's optimizer.py and its lack of eq_fit.py are in "
                                  "files_that_differ_between_arms",
            "workers_per_arm": args.workers,
        },
        "limitations": [
            "The SciPy arm's rejected or failed SLSQP result is not in the fit diagnostics; it is read from an "
            "interposed scipy.optimize.minimize and replayed by accepting it, so S4 is measured on a replay of "
            "the SciPy arm.",
            "An S1 exception needs 'reduced' to 'apply', the constraint solver as the only reason in the "
            "incumbent, a failed or rejected incumbent refinement, and a candidate result within 1e-9 of the "
            "bounds and adjacency limits. The existing feasibility check does not test bounds; bound excess is "
            "recorded.",
            "The control difference is analysed with the same definitions as the port difference: its own "
            "exceptions are left out of its response counts, and the S1 allowance counts every "
            "recommendation-status change in it.",
            "Solver effort of the NumPy arm is counted in residual evaluations; SciPy's own nfev is recorded "
            "for the SciPy arms only. Run times are wall and CPU seconds on a shared PC and not gated.",
            "Responses are compared on the rounded stored values (5 decimals).",
            *([] if evaluation["exceptions"] else [
                f"No case exercised the path where the SciPy arm's constraint solve failed or was rejected and "
                f"the NumPy arm's succeeded (0 of {len(cases)} cases are S1 exceptions). Gate S4 and the S1 "
                "exception rule therefore measured nothing in this run; that behaviour is covered only by the "
                "seeded parity study of 360 synthetic fits, which is not part of this report."]),
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
