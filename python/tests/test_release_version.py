"""Tests for release-version normalization across packaging tools."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


TOOL = Path(__file__).parents[1] / "tools" / "release_version.py"
SPEC = importlib.util.spec_from_file_location("release_version", TOOL)
assert SPEC is not None and SPEC.loader is not None
release_version = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release_version
SPEC.loader.exec_module(release_version)


@pytest.mark.parametrize("value", ("2.0.0rc1", "2.0.0-rc.1"))
def test_rc_spellings_share_canonical_outputs(value: str) -> None:
    version = release_version.parse_version(value)

    assert version.pep440 == "2.0.0rc1"
    assert version.cargo == "2.0.0-rc.1"
    assert version.tag == "v2.0.0-rc.1"
    assert version.msi == "2.0.1"
    assert version.is_prerelease


def test_final_msi_version_is_after_all_rcs() -> None:
    candidate = release_version.parse_version("2.0.0rc98")
    final = release_version.parse_version("2.0.0")

    assert tuple(map(int, candidate.msi.split("."))) < tuple(
        map(int, final.msi.split("."))
    )
    assert final.msi == "2.0.99"


def test_tags_are_canonical_and_reject_ambiguous_rc_tags() -> None:
    assert release_version.parse_tag("v2.0.0").tag == "v2.0.0"
    assert release_version.parse_tag("v2.0.0-rc.1").pep440 == "2.0.0rc1"
    with pytest.raises(ValueError, match="canonical|use v2.0.0-rc.1"):
        release_version.parse_tag("v2.0.0rc1")


@pytest.mark.parametrize(
    "value",
    ("2.0", "2.0.0rc0", "2.0.0-rc1", "2.0.0-RC1", "v2.0.0", "2.0.0+local"),
)
def test_invalid_release_versions_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        release_version.parse_version(value)


def test_rc_sequence_is_bounded_for_msi() -> None:
    with pytest.raises(ValueError, match="MSI limit"):
        release_version.parse_version("2.0.0rc99").msi


@pytest.mark.parametrize("value", ("256.0.0", "2.256.0", "2.0.655"))
def test_msi_mapping_rejects_out_of_range_fields(value: str) -> None:
    with pytest.raises(ValueError, match="MSI limit|too large"):
        release_version.parse_version(value).msi
