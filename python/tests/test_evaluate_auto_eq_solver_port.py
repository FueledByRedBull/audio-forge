"""Synthetic check of the gate logic in evaluate_auto_eq_solver_port (no audio, no native module)."""

import argparse
from pathlib import Path

import numpy as np
import pytest

import check_evaluation_hygiene as hygiene
import evaluate_auto_eq_solver_port as cs

FLAT = [0.0] * 257


def record(status="apply", constraint=True, nonlinear=True, response=FLAT, reasons=(), refine=(), full=None):
    return {"failure": None, "status": status, "nonlinear_solver_success": nonlinear,
            "constraint_solver_success": constraint, "response": response, "full_strength_response": full,
            "recommendation_reasons": list(reasons), "refine": list(refine), "final_adjacency_excess_db": 0.0,
            "audio_sha256": "a", "noise_sha256": None, "moved_starts": [1, 2]}


def feasible_call():
    return {"success": True, "bound_excess_db": 0.0, "adjacency_excess_db": 0.0, "response": FLAT, "raw": None}


def rejected(reason="adjacency_limit", status=0):
    return {"success": False, "bound_excess_db": 0.0, "adjacency_excess_db": 1.0, "response": FLAT,
            "raw": {"reason": reason, "adjacency_excess_db": 1.0, "bound_excess_db": 0.0, "status": status,
                    "message": "m", "response": [0.5] * 257}}


def reduced(call):
    return record("reduced", False, reasons=[cs.CONSTRAINT_REASON], refine=[call])


def row(name, incumbent, candidate=None, control=None, replay=None):
    return {"case": {"id": name}, "incumbent": incumbent, "candidate": candidate or incumbent,
            "control": control or incumbent, "replay": replay}


def gates(rows):
    return cs.evaluate(rows)["gates"]


def test_identical_arms_pass() -> None:
    assert all(gate["passed"] for gate in gates([row("a", record()), row("b", reduced(rejected()))]).values())


def test_s1_exceptions_and_allowance() -> None:
    ok = record(refine=[feasible_call()])
    for failure in (rejected("adjacency_limit"), rejected("solver_failure", 9), rejected("solver_failure", 8)):
        result = cs.evaluate([row("flip", reduced(failure), ok, replay=record(response=[0.9] * 257))])
        assert result["gates"]["S1"]["port"]["allowed_exceptions"] == 1 and result["gates"]["S4"]["passed"]
    assert (cs.s1_exception(reduced(rejected("solver_failure", 9)), ok) or {})["scipy_failure"] == "iteration_cap"
    assert (cs.s1_exception(reduced(rejected("solver_failure", 8)), ok) or {})["scipy_failure"] == "exit_failure"
    infeasible = record(refine=[rejected()])
    assert cs.s1_exception(reduced(rejected()), infeasible) is None
    # other status changes are allowed up to the control's count of status changes
    worse = record("reduced", False, reasons=["other"])
    assert not gates([row("x", record(), worse)])["S1"]["passed"]
    assert gates([row("x", record(), worse, control=worse)])["S1"]["passed"]
    assert not gates([row("x", record(), worse, control=worse), row("y", record(), worse)])["S1"]["passed"]
    # a failure change in the port difference always fails
    crashed = record() | {"failure": "Recording too unclear.", "status": None}
    assert not gates([row("x", record(), crashed, control=crashed)])["S1"]["passed"]


def test_s2_counts() -> None:
    def many(port_over, control_over, size=100):
        rows = []
        for index in range(size):
            far = [0.3] * 257
            rows.append(row(f"c{index}", record(), record(response=far if index < port_over else FLAT),
                            record(response=far if index < control_over else FLAT)))
        return gates(rows)["S2"]

    assert many(5, 0)["passed"] and not many(6, 0)["passed"] and many(7, 1)["passed"] and not many(8, 1)["passed"]
    assert many(5, 0)["exceeding"]["0.25"] == {"port": 5, "control": 0, "allowed": 5}
    slight = [row(f"c{i}", record(), record(response=[0.06] * 257 if i < 11 else FLAT)) for i in range(100)]
    assert not gates(slight)["S2"]["passed"]  # 89 % within 0.05 dB


def test_s3_flags() -> None:
    assert not gates([row("x", record(), record(constraint=False))])["S3"]["passed"]
    repair = reduced(rejected())
    assert gates([row("x", repair, record(refine=[feasible_call()]))])["S3"]["passed"]
    assert not gates([row("x", repair, record(refine=[rejected()]))])["S3"]["passed"]
    off = record(nonlinear=False)
    three = [row(f"c{i}", record(), off) for i in range(3)]
    assert not gates(three)["S3"]["passed"]  # allowed 2
    assert gates([row(f"c{i}", record(), off, control=off if i == 0 else None) for i in range(4)])["S3"]["passed"]


def test_s4_fallback_and_failure() -> None:
    ok = record(refine=[feasible_call()])
    far = reduced(rejected())
    far["full_strength_response"] = [2.0] * 257
    result = gates([row("flip", far, ok)])["S4"]
    assert result["measured_scaled_back"] == 1 and not result["passed"]


def test_moved_start() -> None:
    start = np.array([0.0, 1.0, 5.0])
    moved = cs._moved_start(start, ([-1.0, -1.0, -1.0], [1.0, 2.0, 5.0]))  # least_squares form
    assert moved[0] == np.nextafter(0.0, 1) and moved[1] == np.nextafter(1.0, 2) and moved[2] == 5.0
    pairs = cs._moved_start(start, [(-1.0, 1.0), (None, None), (0.0, 5.0)])  # minimize form
    assert pairs[0] == np.nextafter(0.0, 1) and pairs[2] == 5.0
    assert cs.nearest_rank([1.0, 2.0, 3.0, 4.0], 50) == 2.0


def test_reported_block_counts_and_is_not_gated() -> None:
    def solved(response, nfev=(10, 5), error=1.0):
        return record(response=response) | {
            "least_squares": [{"nfev": n} for n in nfev], "target_error_after_db": error}

    far, near = [0.3] * 257, [0.04] * 257
    rows = []
    for index in range(4):
        incumbent = solved(FLAT)
        candidate = solved(far if index == 0 else near if index == 1 else FLAT, nfev=(10, 6 if index == 0 else 5),
                           error=1.5 if index == 0 else 1.0)
        rows.append(row(f"c{index}", incumbent, candidate) | {"scipy_numpy_svd": solved(FLAT)})

    block = cs.reported(rows, capped=True)
    versus = block["candidate_vs_scipy_numpy_svd"]
    assert (versus["within_0.05_db"], versus["over_0.05_db"], versus["over_0.25_db"], versus["over_1.0_db"]) == (3, 1, 1, 0)
    assert versus["max_difference_db"] == pytest.approx(0.3)
    assert (versus["nfev_identical_stage1"], versus["nfev_identical_stage2"], versus["nfev_identical_all_stages"]) == (4, 3, 3)
    assert versus["cases_over_0.05_db"]["listed"] == ["c0: 0.3 dB"]
    assert block["scipy_numpy_svd_vs_incumbent"]["over_0.05_db"] == 0 and block["inputs_identical"]
    errors = block["target_error_after_vs_incumbent"]["candidate"]
    assert errors["cases"] == 4 and errors["worse_by_over_0.25_db"] == 1 and errors["better_by_over_0.05_db"] == 0
    assert cs.reported(rows, capped=False)["candidate_vs_scipy_numpy_svd"]["cases_over_0.05_db"] == {"c0": pytest.approx(0.3)}
    assert cs.reported([row("a", record())], capped=True) == {"skipped": True}
    assert all(gate["passed"] for gate in cs.evaluate(rows)["gates"].values()) == all(
        gate["passed"] for gate in cs.evaluate([{k: v for k, v in r.items() if k != "scipy_numpy_svd"} for r in rows])["gates"].values())


def timed(entry):
    return entry | {"fit_seconds": 1.0, "fit_cpu_seconds": 1.0, "solver_seconds": 0.5,
                    "least_squares": [{"nfev": 3}, {"nfev": 4}], "residual_calls": {"stage1": 3, "stage2": 4}}


def report_for(rows, tmp_path):
    """The tracked report the tool would write for ``rows``, with the real working-tree pins."""
    cases: list[dict[str, str]] = [item["case"] for item in rows]
    rows = [{key: timed(value) if key != "case" and value is not None else value for key, value in item.items()}
            for item in rows]
    args = argparse.Namespace(native_module=Path("mic_eq_core.pyd"), workers=3)
    pins = {path: cs.sha256_file(cs.REPO_ROOT / path) for path in cs.PINNED_SOURCES}
    report = cs.build_report(args, cases, "judged", [1], rows, cs.evaluate(rows),
                             "remove_scipy", "summary", {"incumbent": {"a.py": "1"}, "candidate": {"a.py": "2"}},
                             "n" * 64, {"cross_take_manifest_sha256": "0" * 64}, "e" * 64, pins, 1.0)
    path = tmp_path / "report.json"
    cs.write_json(path, report, limit=cs.MAX_REPORT_BYTES)
    return report, path


def test_report_is_a_valid_tracked_report_and_says_whether_the_change_is_audible(tmp_path) -> None:
    same, path = report_for([row("a", record()), row("b", reduced(rejected()))], tmp_path)
    assert hygiene.validate_report(path) == []
    assert same["audible_change"] is False and same["schema_version"] == 2
    moved, path = report_for([row("a", record(), record(response=[0.3] * 257)), row("b", record())], tmp_path)
    assert hygiene.validate_report(path) == []
    assert moved["audible_change"] is True
    contract = moved["evaluation_contract"]
    assert hygiene.AUDIBLE_CONTRACT_FIELDS <= contract.keys()
    assert contract["clean_preservation"]["port_max_difference_db"] == pytest.approx(0.3)
    assert contract["clean_preservation"]["port_cases_over_0.05_db"] == 1
    assert set(moved["source_sha256"]) == set(cs.PINNED_SOURCES)


def test_report_states_when_no_case_reached_the_constraint_solver_exception(tmp_path) -> None:
    none, _ = report_for([row("a", record())], tmp_path)
    assert any("0 of 1 cases are S1 exceptions" in text for text in none["limitations"])
    ok = record(refine=[feasible_call()])
    flipped, _ = report_for([row("flip", reduced(rejected()), ok, replay=record(response=[0.9] * 257))], tmp_path)
    assert not any("S1 exceptions" in text for text in flipped["limitations"])


def test_pinned_sources_must_match_the_working_tree(tmp_path) -> None:
    tree, arm = tmp_path / "tree", tmp_path / "arm"
    for root in (tree, arm):
        for relative in cs.PINNED_SOURCES:
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_text(relative, encoding="utf-8")
    evaluator = arm / cs.EVALUATOR_PATH
    pins = cs.pinned_sources(arm, tree, evaluator)
    assert pins == {path: cs.sha256_file(tree / path) for path in cs.PINNED_SOURCES}
    (arm / cs.PINNED_SOURCES[0]).write_text("edited", encoding="utf-8")
    with pytest.raises(SystemExit, match="eq_fit.py"):
        cs.pinned_sources(arm, tree, evaluator)


def test_report_size_limit_matches_hygiene_and_is_enforced(tmp_path) -> None:
    assert cs.MAX_REPORT_BYTES == hygiene.MAX_TRACKED_REPORT_BYTES
    with pytest.raises(SystemExit, match="over the"):
        cs.write_json(tmp_path / "big.json", {"text": "x" * 200}, limit=100)
    assert not (tmp_path / "big.json").exists()


def test_details_output_may_not_be_under_the_tracked_evaluation_folder(tmp_path, capsys) -> None:
    for name in ("incumbent", "candidate"):
        (tmp_path / name / "python" / "mic_eq").mkdir(parents=True)
    command = ["--incumbent-root", str(tmp_path / "incumbent"), "--candidate-root", str(tmp_path / "candidate"),
               "--native-module", str(tmp_path / "mic_eq_core.pyd"), "--report", str(tmp_path / "report.json"),
               "--details-output", str(cs.REPO_ROOT / "evaluation" / "details.json")]
    with pytest.raises(SystemExit):
        cs.main(command)
    assert "stay out of evaluation/" in capsys.readouterr().err
