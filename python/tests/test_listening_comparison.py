"""Regression checks for same-capture listening comparison rendering."""

from __future__ import annotations

import numpy as np
import pytest

from mic_eq.analysis import listening_comparison as comparison
from PyQt6.QtMultimedia import QAudioFormat
from mic_eq.ui.listening_comparison_dialog import (
    _pcm_bytes,
    _preview_samples,
    _preview_safety_gain,
)
from mic_eq.analysis.listening_comparison import RenderedComparisonClip


def _settings(gain: float = 0.0) -> dict:
    return {
        "enabled": True,
        "band_freqs": [100.0, 200.0, 400.0, 800.0, 1200.0, 2400.0, 4800.0, 8000.0, 12000.0, 16000.0],
        "band_gains": [gain] * 10,
        "band_qs": [1.41] * 10,
    }


def test_native_comparison_renders_aligned_finite_audio():
    from mic_eq.config import Preset

    capture = (np.sin(np.arange(48000) * (2 * np.pi * 220 / 48000)) * 0.1).astype(np.float32)
    settings = Preset().eq.to_dict()
    result = comparison.render_comparison(capture, 48000, settings, settings)
    assert result.native_authoritative
    assert all(clip.samples.shape == capture.shape for clip in result.clips)
    np.testing.assert_array_equal(result.current.samples, result.proposed.samples)
    assert np.isfinite(result.current.samples).all()


@pytest.mark.parametrize("current_scope,proposed_scope", [
    ("downstream", "downstream"), ("normal", "normal"),
    ("downstream", "normal"), ("raw", "normal"), ("bypass", "normal"),
])
def test_comparison_describes_each_native_render_scope(current_scope, proposed_scope, qapp, monkeypatch):
    def chain(scope):
        return {
            "full_chain": scope != "downstream",
            "processing_mode": "normal" if scope == "downstream" else scope,
            "vad_available": False, "gate_enabled": False,
            "suppressor_enabled": False,
            "deesser_enabled": True,
        }

    capture = (0.01 * np.sin(np.arange(4800) * 2 * np.pi / 48)).astype(np.float32)
    result = comparison.render_comparison(
        capture, 48000, _settings(), _settings(),
        current_chain_settings=chain(current_scope),
        proposed_chain_settings=chain(proposed_scope),
    )
    assert result.original.rendered_stages == ()
    for clip, scope in [(result.current, current_scope), (result.proposed, proposed_scope)]:
        assert scope in clip.render_scope.lower()
        if scope == "raw":
            assert clip.rendered_stages == ("output safety",)
        elif scope == "bypass":
            assert clip.rendered_stages == ("input pre-filter", "output safety")
        else:
            assert clip.rendered_stages.index("deesser") < clip.rendered_stages.index("tone EQ")
            assert ("input cleanup" in clip.rendered_stages) is (scope == "normal")
            assert "noise suppression" not in clip.rendered_stages
    assert result.current.render_scope in result.scope_label
    assert result.proposed.render_scope in result.scope_label
    if current_scope != proposed_scope:
        assert "current" in result.scope_label and "proposed" in result.scope_label

    from mic_eq.ui.listening_comparison_dialog import ListeningComparisonDialog

    monkeypatch.setattr(ListeningComparisonDialog, "_start_render", lambda self: None)
    dialog = ListeningComparisonDialog(
        audio_data=capture, sample_rate=48000,
        current_settings=_settings(), proposed_settings=_settings(),
    )
    dialog._on_render_ready(result)
    assert result.scope_label in dialog.status_label.text()
    for clip in (result.current, result.proposed):
        assert clip.render_scope in dialog.scope_warning.text()
        assert ", ".join(clip.rendered_stages) in dialog.scope_warning.text()
    dialog.close()


@pytest.mark.parametrize("sample_format,dtype", [
    (QAudioFormat.SampleFormat.Int16, "<i2"),
    (QAudioFormat.SampleFormat.Int32, "<i4"),
    (QAudioFormat.SampleFormat.Float, "<f4"),
    (QAudioFormat.SampleFormat.UInt8, "u1"),
])
def test_preview_pcm_fades_duplicates_channels_and_preserves_positive_peak(sample_format, dtype):
    audio_format = QAudioFormat()
    audio_format.setSampleRate(48000)
    audio_format.setChannelCount(2)
    audio_format.setSampleFormat(sample_format)
    source = np.ones(4800, dtype=np.float32)
    output = np.frombuffer(_pcm_bytes(source, audio_format, 48000), dtype=dtype).reshape(-1, 2)
    np.testing.assert_array_equal(output[:, 0], output[:, 1])
    assert output[2400, 0] > 0
    assert output[0, 0] < output[2400, 0]
    assert output[-1, 0] < output[2400, 0]
    assert np.all(source == 1.0)


def test_render_comparison_keeps_one_capture_and_reports_native_delay(monkeypatch):
    def fake_simulation(audio, sample_rate, eq, chain):
        gain = float(eq["band_gains"][0])
        output = np.asarray(audio, dtype=np.float32) * (10.0 ** (gain / 20.0))
        return {
            "output_audio": output.tolist(),
            "simulation_backend": "rust",
            "chain_latency_samples": 240,
        }

    monkeypatch.setattr(comparison, "simulate_candidate_chain", fake_simulation)
    monkeypatch.setattr(
        comparison,
        "_speech_mask",
        lambda audio, sample_rate: (np.ones(audio.size, dtype=bool), "test"),
    )
    capture = np.sin(np.linspace(0.0, 50.0, 4096, dtype=np.float32)) * 0.1

    result = comparison.render_comparison(
        capture,
        48_000,
        _settings(),
        _settings(3.0),
    )

    assert [clip.label for clip in result.clips] == ["original", "current", "proposed"]
    assert all(clip.samples.size == capture.size for clip in result.clips)
    assert all(np.isfinite(clip.samples).all() for clip in result.clips)
    assert result.alignment_samples == 240
    assert result.alignment_ms == pytest.approx(5.0)
    assert result.native_authoritative is True
    assert result.excluded_stages == ("input cleanup", "gate/VAD", "noise suppression", "deesser")


def test_level_match_preserves_actual_difference_and_matches_preview_level(monkeypatch):
    def fake_simulation(audio, sample_rate, eq, chain):
        gain = float(eq["band_gains"][0])
        output = np.asarray(audio, dtype=np.float32) * (10.0 ** (gain / 20.0))
        return {
            "output_audio": output.tolist(),
            "simulation_backend": "rust",
            "chain_latency_samples": 0,
        }

    monkeypatch.setattr(comparison, "simulate_candidate_chain", fake_simulation)
    monkeypatch.setattr(
        comparison,
        "_speech_mask",
        lambda audio, sample_rate: (np.ones(audio.size, dtype=bool), "test"),
    )
    capture = np.full(1024, 0.1, dtype=np.float32)

    result = comparison.render_comparison(
        capture,
        48_000,
        _settings(),
        _settings(6.0),
        level_match=True,
    )

    assert result.proposed.actual_level_delta_db == pytest.approx(6.0, abs=0.05)
    assert result.proposed.level_match_gain_db == pytest.approx(-6.0, abs=0.05)
    assert result.proposed.playback_level_db == pytest.approx(
        result.original.playback_level_db, abs=0.05
    )


def test_level_match_is_applied_at_playback_without_rerendering():
    clip = RenderedComparisonClip(
        label="proposed",
        samples=np.full(128, 0.1, dtype=np.float32),
        actual_level_db=-20.0,
        actual_level_delta_db=6.0,
        playback_level_db=-20.0,
        peak_db=-20.0,
        level_match_gain_db=0.0,
        safety_gain_db=0.0,
        simulation_backend="rust",
    )

    matched = _preview_samples(clip, level_match=True)
    np.testing.assert_allclose(matched, 0.0501187, rtol=1e-4)
    np.testing.assert_array_equal(
        _preview_samples(clip, level_match=False), clip.samples
    )


def test_dialog_uses_one_peak_safety_gain_for_level_matched_playback():
    def clip(label: str, value: float, delta: float) -> RenderedComparisonClip:
        return RenderedComparisonClip(
            label=label,
            samples=np.full(128, value, dtype=np.float32),
            actual_level_db=-20.0,
            actual_level_delta_db=delta,
            playback_level_db=-20.0,
            peak_db=-20.0,
            level_match_gain_db=0.0,
            safety_gain_db=0.0,
            simulation_backend="rust",
        )

    result = comparison.ComparisonRenderResult(
        sample_rate=48_000,
        original=clip("original", 0.2, 0.0),
        current=clip("current", 0.2, 0.0),
        proposed=clip("proposed", 0.9, -12.0),
        level_match=False,
        level_match_reference_db=-20.0,
        speech_detection="test",
        alignment_samples=0,
        alignment_ms=0.0,
    )
    safety_gain = _preview_safety_gain(result, level_match=True)
    previews = [
        _preview_samples(
            clip,
            level_match=True,
            common_safety_gain_db=safety_gain,
            rendered_level_matched=result.level_match,
        )
        for clip in result.clips
    ]

    assert safety_gain < 0.0
    assert max(float(np.max(np.abs(samples))) for samples in previews) <= (
        10.0 ** (comparison.PLAYBACK_PEAK_DB / 20.0) + 1.0e-6
    )
    assert np.max(np.abs(previews[0])) == pytest.approx(
        np.max(np.abs(previews[1]))
    )


def test_peak_safety_uses_one_attenuation_for_all_level_matched_clips(monkeypatch):
    def fake_simulation(audio, sample_rate, eq, chain):
        return {
            "output_audio": np.asarray(audio, dtype=np.float32).tolist(),
            "simulation_backend": "rust",
            "chain_latency_samples": 0,
        }

    monkeypatch.setattr(comparison, "simulate_candidate_chain", fake_simulation)
    monkeypatch.setattr(
        comparison,
        "_speech_mask",
        lambda audio, sample_rate: (np.ones(audio.size, dtype=bool), "test"),
    )
    capture = np.full(1024, 0.95, dtype=np.float32)

    result = comparison.render_comparison(
        capture,
        48_000,
        _settings(),
        _settings(),
        level_match=True,
    )

    gains = [clip.safety_gain_db for clip in result.clips]
    assert gains[0] < 0.0
    assert gains == pytest.approx([gains[0]] * 3)
    assert max(clip.peak_db for clip in result.clips) <= comparison.PLAYBACK_PEAK_DB


def test_typed_eq_keeps_custom_filter_and_slope_for_native_renderer(monkeypatch):
    from mic_eq.config import EQSettings

    typed = EQSettings().to_dict()
    typed["bands"][5]["filter_type"] = "notch"
    typed["bands"][5]["slope_db_per_octave"] = 24
    seen: dict[str, object] = {}

    def fake_simulation(audio, sample_rate, eq, chain):
        seen["eq"] = eq
        seen["chain"] = chain
        return {
            "output_audio": np.asarray(audio, dtype=np.float32).tolist(),
            "simulation_backend": "rust",
            "chain_latency_samples": 0,
        }

    monkeypatch.setattr(comparison, "simulate_candidate_chain", fake_simulation)
    capture = np.full(1024, 0.1, dtype=np.float32)
    comparison.render_comparison(capture, 48_000, typed, typed)

    rendered_eq = seen["eq"]
    assert isinstance(rendered_eq, dict)
    assert rendered_eq["bands"][5]["filter_type"] == "notch"
    assert rendered_eq["bands"][5]["slope_db_per_octave"] == 24


def test_full_chain_leaves_vad_to_native_renderer(monkeypatch):
    seen: list[dict[str, object]] = []

    def fake_simulation(audio, sample_rate, eq, chain):
        seen.append(dict(chain))
        return {
            "output_audio": np.asarray(audio, dtype=np.float32).tolist(),
            "simulation_backend": "rust",
            "chain_latency_samples": 480,
        }

    monkeypatch.setattr(comparison, "simulate_candidate_chain", fake_simulation)
    capture = np.full(1024, 0.1, dtype=np.float32)
    settings = _settings()
    chain = {
        "full_chain": True,
        "input_pre_filtered": True,
        "input_cleanup_mode": "gentle",
        "processing_mode": "normal",
        "gate": {
            "enabled": True,
            "gate_mode": 1,
            "vad_threshold": 0.48,
        },
        "rnnoise": {"enabled": True, "strength": 0.7, "model": "rnnoise"},
    }
    result = comparison.render_comparison(
        capture,
        48_000,
        settings,
        settings,
        current_chain_settings=chain,
        proposed_chain_settings=chain,
    )

    assert len(seen) == 2
    assert all(item["full_chain"] is True for item in seen)
    assert all(item["input_pre_filtered"] is True for item in seen)
    assert all(item["input_cleanup_mode"] == "gentle" for item in seen)
    assert all("vad_probabilities" not in item for item in seen)
    for item in seen:
        gate = item["gate"]
        suppressor = item["suppressor"]
        assert isinstance(gate, dict)
        assert isinstance(suppressor, dict)
        assert gate["gate_mode"] == 1
        assert gate["vad_threshold"] == 0.48
        assert suppressor["strength"] == 0.7
        assert suppressor["model"] == "rnnoise"
    assert result.excluded_stages == ("deesser",)


def test_full_chain_native_failure_is_visible(monkeypatch):
    monkeypatch.setattr(
        comparison,
        "simulate_candidate_chain",
        lambda audio, sample_rate, eq, chain: {
            "simulation_backend": "python",
            "native_simulation_failure": {
                "message": "deepfilter model is unavailable"
            },
            "output_audio": np.asarray(audio, dtype=np.float32).tolist(),
        },
    )
    with pytest.raises(RuntimeError, match="deepfilter model is unavailable"):
        comparison.render_comparison(
            np.ones(1024, dtype=np.float32) * 0.1,
            48_000,
            _settings(),
            _settings(),
            current_chain_settings={"full_chain": True},
            proposed_chain_settings={"full_chain": True},
        )


@pytest.mark.parametrize(
    "capture, sample_rate, message",
    [
        (np.ones((2, 4), dtype=np.float32), 48_000, "mono"),
        (np.array([0.0, np.nan], dtype=np.float32), 48_000, "non-finite"),
        (np.ones(9, dtype=np.float32), 8_000, "longer"),
    ],
)
def test_capture_validation_rejects_unsafe_or_unbounded_input(
    capture, sample_rate, message, monkeypatch
):
    monkeypatch.setattr(comparison, "MAX_COMPARISON_SECONDS", 0.001)
    with pytest.raises(ValueError, match=message):
        comparison.render_comparison(capture, sample_rate, _settings(), _settings())
