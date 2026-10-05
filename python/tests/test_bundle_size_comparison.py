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


def test_size_comparison_attributes_components_without_double_counting():
    old = {"files": [
        {"path": "_internal/PySide6/Qt6/bin/Qt6Core.dll", "size": 200},
        {"path": "_internal/scipy/signal/_signal.pyd", "size": 80},
        {"path": "_internal/scipy.libs/libscipy_openblas.dll", "size": 120},
    ]}
    new = {"files": [
        {"path": "_internal/PySide6/Qt6Core.dll", "size": 180},
        {"path": "_internal/shiboken6/Shiboken.pyd", "size": 20},
        {"path": "_internal/numpy/_core/_multiarray_umath.pyd", "size": 40},
        {"path": "_internal/numpy.libs/libopenblas.dll", "size": 60},
        {"path": "_internal/models/DeepFilterNet3_ll_onnx/encoder.onnx", "size": 50},
        {"path": "_internal/models/DeepFilterNet3_onnx.tar.gz", "size": 70},
        {"path": "_internal/models/silero_vad.onnx", "size": 10},
        {"path": "_internal/df.dll", "size": 15},
        {"path": "_internal/onnxruntime.dll", "size": 25},
        {"path": "_internal/onnxruntime_providers_shared.dll", "size": 5},
        {"path": "AudioForge.exe", "size": 60},
        {"path": "_internal/mic_eq/mic_eq_core.cp313-win_amd64.pyd", "size": 20},
        {"path": "_internal/python313.dll", "size": 30},
        {"path": "_internal/licenses/dependencies/scipy/LICENSE", "size": 3},
        {"path": "unrecognized.dll", "size": 7},
    ]}
    output = release_provenance.compare_bundle_sizes(new, old)
    assert "400 -> 595 bytes (+195)" in output
    component_section = output.split("| Component |", 1)[1].split("| Largest file", 1)[0]
    expected = {
        "Qt and bindings": (200, 200, 0),
        "SciPy and bundled BLAS": (200, 0, -200),
        "NumPy and bundled BLAS": (0, 100, 100),
        "DeepFilter LL model": (0, 50, 50),
        "DeepFilter Standard model": (0, 70, 70),
        "Silero model": (0, 10, 10),
        "DeepFilter runtime": (0, 15, 15),
        "ONNX Runtime": (0, 30, 30),
        "Application executable and native core": (0, 80, 80),
        "Licenses": (0, 3, 3),
        "Python and other runtime files": (0, 37, 37),
    }
    for name, (before, after, delta) in expected.items():
        assert f"| {name} | {before} | {after} | {delta:+} |" in component_section
    assert sum(after for _, after, _ in expected.values()) == 595
    assert "Compressed component sizes are not additive in a solid archive" in output
