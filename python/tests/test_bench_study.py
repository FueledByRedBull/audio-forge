"""The study summary built from bench reports: its tallies, its controls and the report it writes."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import bench_study as study
import check_evaluation_hygiene as hygiene


def reading(**changes: Any) -> dict[str, Any]:
    return {"estoi": 0.95, "si_sdr": 12.0, "tone_error_db": 1.0, "sibilant_loss_db": 2.0, "deep_pause_db": -20.0,
            "tail_db": -15.0, "onset_db": -5.0, "pause_db": -18.0, "clipped": False, "limiter_overload": False, **changes}


def arms(plain: dict[str, Any], aware: dict[str, Any]) -> dict[str, Any]:
    return {"plain": plain, "fair": aware}


def entry(*, outcome="applied", veto=None, status="usable", retakes=None, tuner="retained", steps=("tone",),
          plain=None, aware=None, squash_aware=None) -> dict[str, Any]:
    """A mix of a bench report. ``plain`` and ``aware`` change the readings of the setup against the factory ones."""
    plain, aware = plain or {}, aware or {}
    current = reading() | {"fair": reading()}
    ends = reading(**plain)
    setup = {
        "outcome": outcome, "veto": veto, "vetoes": [], "steps": list(steps), "tuner": tuner,
        "noise_reference": {"status": status, "reasons": [] if status == "usable" else ["room noise drifts"],
                            **({"retakes": retakes} if retakes else {})},
        "proposal": ends, "final": ends if outcome == "applied" else None,
        "fair": {"proposal": reading(**aware), "final": reading(**aware) if outcome == "applied" else None},
        "compressor_step": {
            "step": arms(reading(**plain), reading(**aware)),
            "off": arms(reading(onset_db=-8.0), reading()),
            "squash": arms(reading(estoi=0.8), reading(**(squash_aware or {"estoi": 0.7}))),
        },
    }
    return {"factory": current, "deesser_on": reading(), "voice_setup": setup}


def report(**mixes: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "build": {"native_module_sha256": "0" * 64, "vad": "silero"}, "mixes": mixes}


BEFORE = report(**{
    "room": entry(plain={"tail_db": -12.0}),
    "cafe": entry(outcome="refused", veto="B2", status="questionable", retakes=["questionable", "usable", "usable"],
                  plain={"tail_db": -12.0}),
    "tv@14": entry(outcome="refused", veto="B2", status="questionable", retakes=["questionable"] * 3, tuner="retained"),
    "babble": entry(outcome="refused", veto="B2", status="invalid", retakes=["invalid"] * 3),
})
AFTER = deepcopy(BEFORE)
AFTER["mixes"]["cafe"]["voice_setup"] |= {"outcome": "applied", "veto": None, "tuner": "withheld"}
AFTER["mixes"]["cafe"]["voice_setup"]["final"] = AFTER["mixes"]["cafe"]["voice_setup"]["proposal"]
AFTER["mixes"]["cafe"]["voice_setup"]["fair"]["final"] = AFTER["mixes"]["cafe"]["voice_setup"]["fair"]["proposal"]


def test_the_noise_reference_tallies_follow_each_status_before_and_after():
    result = study.noise_reference_study(BEFORE, AFTER)
    groups = result["groups"]

    assert groups["usable"]["fixed"]["before"] == groups["usable"]["fixed"]["after"] == {"applied": 1}
    # The questionable fixed mix is now applied; the swept one stays refused for its weak speech-to-noise ratio.
    assert groups["questionable"]["fixed"]["before"] == {"refused:B2": 1} and groups["questionable"]["fixed"]["after"] == {"applied": 1}
    assert groups["questionable"]["sweep"]["after"] == {"refused:B2": 1}
    assert groups["invalid"]["fixed"]["after"] == {"refused:B2": 1}  # an invalid reference still ends the setup
    # A retake of a continuous distractor comes back questionable again.
    assert groups["questionable"]["fixed"]["retakes_usable"] == "2 of 3" and groups["questionable"]["sweep"]["retakes_usable"] == "0 of 3"
    (changed,) = result["changed"]
    assert (changed["mix"], changed["before"], changed["after"], changed["tuner"]) == ("cafe", "refused B2", "applied", "withheld")
    assert changed["fails_plain"] == ["tail_db"] and changed["fails_aware"] == []
    assert groups["questionable"]["fixed"]["applied_fail_plain"] == 1 and groups["questionable"]["fixed"]["applied_fail_aware"] == 0


def test_the_compressor_study_counts_what_each_reading_fails_and_keeps_the_squash_control():
    result = study.compressor_study(AFTER)

    assert result["mixes"] == 4
    assert result["step"]["plain"]["mixes_failing"] == 2 and result["step"]["aware"]["mixes_failing"] == 0
    assert result["off"]["plain"] == {"mixes_failing": 4, "by_clause": {"onset_db": 4}}
    assert result["off"]["aware"]["mixes_failing"] == 0
    # The squashing compressor must still fail when read against its own compression.
    assert result["squash"]["aware"] == {"mixes_failing": 4, "by_clause": {"estoi": 4}}
    assert result["whole_setups"] == {"setups": 4, "failing_plain": 2, "failing_aware": 0}


def test_a_deesser_run_that_moves_nothing_is_no_change_and_one_that_moves_a_reading_is():
    same = study.deesser_study(AFTER, deepcopy(AFTER))
    assert same["measurable_change"] is False and same["readings_beyond_margin"] == 0
    assert same["arms"]["deesser_on"]["estoi"] == {"mixes": 4, "mean": 0.0, "max_abs": 0.0, "beyond_margin": 0, "identical": 4}

    inside = deepcopy(AFTER)
    inside["mixes"]["room"]["deesser_on"]["estoi"] += 0.0019
    assert study.deesser_study(AFTER, inside)["measurable_change"] is False

    outside = deepcopy(AFTER)
    outside["mixes"]["room"]["deesser_on"]["estoi"] += 0.003
    result = study.deesser_study(AFTER, outside)
    assert result["measurable_change"] is True and result["arms"]["deesser_on"]["estoi"]["beyond_margin"] == 1

    other_steps = deepcopy(AFTER)
    other_steps["mixes"]["room"]["voice_setup"]["steps"] = ["tone", "deesser"]
    result = study.deesser_study(AFTER, other_steps)
    assert result["outcome_or_step_changes"] == ["room"] and result["measurable_change"] is True


def test_the_report_carries_the_decision_block_passes_the_hygiene_check_and_stays_small(tmp_path: Path):
    paths = {}
    for name, data in (("before", BEFORE), ("after", AFTER), ("both", AFTER)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(data), encoding="utf-8")
    output, details = tmp_path / "study-report.json", tmp_path / "details"

    assert study.main(["--before", str(paths["before"]), "--after", str(paths["after"]), "--deesser", f"both={paths['both']}",
                       "--output", str(output), "--details-output", str(details)]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))

    assert written["decision"] == {"outcome": "measured", "reason": "The numbers are information; the maintainer decides.",
                                   "decided": None, "by": None}
    assert "_rows" not in written and set(written["deesser"]) == {"both"} and written["limitations"]
    assert hygiene.validate_report(output) == [] and output.stat().st_size < hygiene.MAX_TRACKED_REPORT_BYTES
    assert {p.name for p in details.iterdir()} == {"rows.json", "before.json", "after.json", "both.json"}
    assert len(json.loads((details / "rows.json").read_text(encoding="utf-8"))) == 4


def test_the_tool_refuses_a_file_that_is_not_a_bench_report_and_a_bad_variant_name(tmp_path: Path):
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    good = tmp_path / "good.json"
    good.write_text(json.dumps(AFTER), encoding="utf-8")

    with pytest.raises(SystemExit, match="not a bench report"):
        study.main(["--before", str(other), "--after", str(good), "--output", str(tmp_path / "out.json")])
    with pytest.raises(SystemExit):
        study.main(["--before", str(good), "--after", str(good), "--deesser", "no-equals-sign",
                    "--output", str(tmp_path / "out.json")])
    with pytest.raises(SystemExit):  # --output is required
        study.main(["--before", str(good), "--after", str(good)])


def test_unverified_and_failed_setups_and_mixes_without_a_reference_are_tallied_as_themselves():
    before, after = deepcopy(BEFORE), deepcopy(AFTER)
    clipped = entry(outcome="refused", veto="capture")
    clipped["voice_setup"]["noise_reference"] = None
    for side in (before, after):
        side["mixes"]["clipped"] = deepcopy(clipped)
    after["mixes"]["room"]["voice_setup"]["outcome"] = "unverified"
    before["mixes"]["babble"]["voice_setup"] |= {"outcome": "failed", "veto": None}

    groups = study.noise_reference_study(before, after)["groups"]

    assert groups["none"]["fixed"]["before"] == {"refused:capture": 1}  # not counted as usable
    assert groups["usable"]["fixed"]["mixes"] == 1
    assert groups["usable"]["fixed"]["after"] == {"unverified": 1}
    assert groups["invalid"]["fixed"]["before"] == {"failed": 1}


def test_the_details_folder_never_overwrites_an_earlier_input(tmp_path: Path):
    good = tmp_path / "after.json"
    good.write_text(json.dumps(AFTER), encoding="utf-8")
    details = tmp_path / "details"
    details.mkdir()
    (details / "after.json").write_text("an earlier run", encoding="utf-8")

    with pytest.raises(SystemExit, match="already exists"):
        study.main(["--before", str(good), "--after", str(good), "--output", str(tmp_path / "out.json"),
                    "--details-output", str(details)])
    assert (details / "after.json").read_text(encoding="utf-8") == "an earlier run"
