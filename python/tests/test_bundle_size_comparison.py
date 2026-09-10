"""Size reports reuse release manifests without changing their contract."""

import json

import pytest

import release_provenance


def test_size_comparison_reports_growth_additions_removals_and_cli(tmp_path, capsys):
    old = {"files": [{"path": "a.dll", "size": 100}, {"path": "removed.dll", "size": 40}]}
    new = {"files": [{"path": "a.dll", "size": 150}, {"path": "new.dll", "size": 10}]}
    previous, current = tmp_path / "old.json", tmp_path / "new.json"
    previous.write_text(json.dumps(old), encoding="utf-8")
    current.write_text(json.dumps(new), encoding="utf-8")
    assert release_provenance.main([
        "compare-sizes", "--manifest", str(current), "--previous-manifest", str(previous),
    ]) == 0
    output = capsys.readouterr().out
    assert "140 -> 160 bytes (+20)" in output
    assert "`a.dll` | 100 | 150 | +50" in output
    assert "`removed.dll` | 40 | 0 | -40" in output
    assert "`new.dll` | 0 | 10 | +10" in output
    assert "No payload size changes" in release_provenance.compare_bundle_sizes(old, old)


@pytest.mark.parametrize("files", [
    [{"path": "bad", "size": -1}],
    [{"path": "bad", "size": True}],
    [{"path": "a", "size": 1}, {"path": "A", "size": 1}],
])
def test_size_comparison_rejects_invalid_manifest(files):
    with pytest.raises(ValueError):
        release_provenance.compare_bundle_sizes({"files": files}, {"files": []})
