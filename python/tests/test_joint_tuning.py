"""Joint selection must keep exact incumbent settings when evidence is weak."""

from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from mic_eq.analysis import joint_tuning
from mic_eq.analysis.cancellation import AnalysisCancelled
from mic_eq.config import Preset
from mic_eq.mic_eq_core import simulate_gate_suppressor_order


def _tuning_args() -> tuple[Any, ...]:
    preset = Preset()
    return (
        np.full(96_000, 0.001, dtype=np.float32),
        np.full(48_000, 0.1, dtype=np.float32),
        48_000,
        asdict(preset.gate),
        asdict(preset.compressor),
        {
            "band_freqs": preset.eq.band_freqs,
            "band_gains": preset.eq.band_gains,
            "band_qs": preset.eq.band_qs,
        },
        asdict(preset.deesser),
        asdict(preset.limiter),
    )


def _downstream_metrics(audio: np.ndarray, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
    del audio
    return {
        "simulation_backend": "rust",
        "compressor_gain_reduction_p95_db": 3.5,
        "compressor_gain_reduction_median_db": 1.47,
        "compressor_gain_reduction_db": 4.0,
        "compressor_pumping_score_db": 0.0,
        "output_true_peak_db": -1.0,
        "limiter_effective_ceiling_db": -0.5,
        "pre_limiter_true_peak_headroom_db": 1.0,
        "non_finite_output": False,
        "candidate_runtime_ms": 0.0,
    }


def _activity_tape(audio: np.ndarray, settings: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    reliability = float(bool(settings.get("vad_available", True)))
    return [
        (0.2 if rms < 0.01 else 0.8, reliability, -60.0, 0.0)
        for rms in joint_tuning._frame_rms(audio)
    ]


def test_native_gate_enabled_matches_requested_control():
    audio = np.full(4800, 0.01, dtype=np.float32)
    muted = simulate_gate_suppressor_order(audio, [0.0] * 10, False, 0.0,
                                          {"gate_enabled": True, "gate_threshold_db": -10.0,
                                           "gate_auto_threshold_enabled": False})
    open_gate = simulate_gate_suppressor_order(audio, [0.0] * 10, False, 0.0,
                                             {"gate_enabled": False})
    assert np.max(np.abs(muted["output_audio"])) < 0.001
    assert np.max(np.abs(open_gate["output_audio"])) > 0.005


def test_vad_alignment_waits_for_a_window_finished_before_the_block():
    aligned = joint_tuning._align_probabilities(
        np.asarray([0.2, 0.8], dtype=np.float32),
        4_800,
        4_800,
    )

    np.testing.assert_allclose(
        aligned,
        [0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8],
    )


def test_joint_unavailable_keeps_incumbent_and_cancellation_propagates(monkeypatch):
    preset = Preset()
    incumbent_gate = {**asdict(preset.gate), "threshold_db": -51.0}
    kwargs: dict[str, Any] = dict(incumbent_settings={"gate": incumbent_gate,
                  "suppressor": {"model": "rnnoise", "enabled": True, "strength": 0.62}})
    args = (np.ones(48000, dtype=np.float32) * 0.001,
            np.ones(96000, dtype=np.float32) * 0.1, 48000,
            asdict(preset.gate), asdict(preset.compressor),
            {"band_freqs": preset.eq.band_freqs, "band_gains": preset.eq.band_gains,
             "band_qs": preset.eq.band_qs}, asdict(preset.deesser), asdict(preset.limiter))
    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: None)
    result = joint_tuning.tune_gate_suppression_dynamics(*args, **kwargs)
    assert not result["apply_recommended"]
    assert result["gate_settings"] == incumbent_gate
    assert result["suppressor_settings"]["strength"] == 0.62
    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate_gate_suppressor_order)
    with pytest.raises(AnalysisCancelled):
        joint_tuning.tune_gate_suppression_dynamics(*args, **kwargs, cancel_check=lambda: True)


@pytest.mark.parametrize("with_neural_vad", [True, False])
def test_joint_selection_evaluates_each_model_and_selects_lower_noise_model(
    monkeypatch, with_neural_vad,
):
    observed_releases: list[float] = []

    def simulate(audio, _probabilities, _before_gate, _strength, settings):
        model = settings["noise_model"]
        observed_releases.append(float(settings["gate_release_ms"]))
        latency = {"rnnoise": 480, "deepfilter-ll": 480, "deepfilter": 1_440}[model]
        attenuation = {"rnnoise": 0.5, "deepfilter-ll": 0.1, "deepfilter": 0.05}[model]
        values = np.asarray(audio, dtype=np.float32)
        output = values * np.where(np.abs(values) < 0.01, attenuation, 0.98)
        return {
            "output_audio": output,
            "dry_audio": output,
            "suppressor_latency_samples": latency,
            "runtime_ms": 0.0,
            "auto_makeup_activity": _activity_tape(audio, settings),
        }

    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate)
    monkeypatch.setattr(joint_tuning, "simulate_candidate_chain", _downstream_metrics)
    args = _tuning_args()
    frame_count = 100
    incumbent = {
        "gate": dict(args[3]),
        "suppressor": {"model": "rnnoise", "enabled": True, "strength": 1.0},
    }
    result = joint_tuning.tune_gate_suppression_dynamics(
        *args,
        vad_probabilities=np.ones(frame_count, dtype=np.float32) if with_neural_vad else None,
        noise_vad_probabilities=np.zeros(frame_count, dtype=np.float32) if with_neural_vad else None,
        incumbent_settings=incumbent,
        max_suppressor_latency_ms=15.0,
    )

    assert result["apply_recommended"]
    assert result["suppressor_settings"]["model"] == "deepfilter-ll"
    assert result["selected_candidate"]["noise_model"] == "deepfilter-ll"
    assert result["selected_candidate"]["metrics"]["suppressor_latency_ms"] == 10.0
    assert result["max_suppressor_latency_ms"] == 15.0
    assert [entry["model"] for entry in result["model_evaluations"]] == [
        "rnnoise",
        "deepfilter-ll",
        "deepfilter",
    ]
    assert {
        candidate["noise_model"] for candidate in result["candidates"]
    } == {"rnnoise", "deepfilter-ll", "deepfilter"}
    deepfilter_candidates = [
        candidate
        for candidate in result["candidates"]
        if candidate["noise_model"] == "deepfilter"
    ]
    assert deepfilter_candidates
    assert all(not candidate["gates"]["suppressor_latency"] for candidate in deepfilter_candidates)
    assert float(args[3]["release_ms"]) + 70.0 in observed_releases


def test_frontend_selection_is_independent_of_uncalibrated_compression(monkeypatch):
    def simulate(audio, _probabilities, _before_gate, _strength, settings):
        model = settings["noise_model"]
        attenuation = {"rnnoise": 0.5, "deepfilter-ll": 0.1, "deepfilter": 0.05}
        values = np.asarray(audio, dtype=np.float32)
        output = values * np.where(np.abs(values) < 0.01, attenuation[model], 0.98)
        return {
            "output_audio": output,
            "dry_audio": output,
            "suppressor_latency_samples": 1_440 if model == "deepfilter" else 480,
            "runtime_ms": 0.0,
            "auto_makeup_activity": _activity_tape(audio, settings),
        }

    # Stub only the neural backend; the downstream compressor and selection
    # run normally, including the second-half candidate verification.
    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate)
    args = list(_tuning_args())
    args[1] = (0.1 * np.sin(2 * np.pi * 180 * np.arange(48_000) / 48_000)).astype(np.float32)
    args[5] = {**args[5], "enabled": False}
    args[6] = {"enabled": False}
    compressor = dict(args[4])
    results = []
    for threshold, ratio, target in ((-55.0, 10.0, 3.5), (-12.0, 1.5, 9.0)):
        args[4] = {
            **compressor,
            "enabled": True,
            "threshold_db": threshold,
            "ratio": ratio,
            "target_p95_reduction_db": target,
            "target_median_reduction_db": target / 2.0,
            "auto_makeup_enabled": False,
            "makeup_gain_db": 12.0,
        }
        result = joint_tuning.tune_gate_suppression_dynamics(
            *args,
            vad_probabilities=np.ones(100, dtype=np.float32),
            noise_vad_probabilities=np.zeros(100, dtype=np.float32),
            max_suppressor_latency_ms=15.0,
        )
        assert result["apply_recommended"], [
            name for name, passed in result["gates"].items() if not passed
        ]
        assert result["holdout_passed"]
        assert result["suppressor_settings"]["model"] == "deepfilter-ll"
        results.append(result)

    assert results[0]["gate_settings"] == results[1]["gate_settings"]
    assert results[0]["suppressor_settings"] == results[1]["suppressor_settings"]
    assert [c["score"] for c in results[0]["candidates"]] == [
        c["score"] for c in results[1]["candidates"]
    ]


@pytest.mark.parametrize("leaking_split", ["training", "holdout", None])
def test_joint_tuning_preserves_noise_after_speech(monkeypatch, leaking_split):
    def simulate(audio, _probabilities, _before_gate, _strength, settings):
        values = np.asarray(audio, dtype=np.float32)
        quiet = np.abs(values) < 0.01
        model = settings["noise_model"]
        contains_speech = bool(np.any(~quiet))
        split = "training" if values[0] < 0.15 else "holdout"
        # Reset noise-only probes favor the candidate. After speech it can
        # leave more noise, while still meeting the absolute attenuation gate.
        noise_gain = 0.5 if model == "rnnoise" else 0.1
        if model != "rnnoise" and contains_speech and split == leaking_split:
            noise_gain = 0.7
        speech_gain = 0.6 if model == "rnnoise" else 0.98
        output = values * np.where(quiet, noise_gain, speech_gain)
        return {
            "output_audio": output,
            "dry_audio": output,
            "suppressor_latency_samples": 1_440 if model == "deepfilter" else 480,
            "runtime_ms": 0.0,
            "auto_makeup_activity": _activity_tape(audio, settings),
        }

    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate)
    monkeypatch.setattr(joint_tuning, "simulate_candidate_chain", _downstream_metrics)
    args = list(_tuning_args())
    args[0] = np.full(96_000, 0.001, dtype=np.float32)
    args[1] = np.concatenate((
        np.full(48_000, 0.1, dtype=np.float32),
        np.full(48_000, 0.2, dtype=np.float32),
    ))
    result = joint_tuning.tune_gate_suppression_dynamics(
        *args,
        vad_probabilities=np.ones(100, dtype=np.float32),
        noise_vad_probabilities=np.zeros(100, dtype=np.float32),
        max_suppressor_latency_ms=15.0,
    )

    assert result["apply_recommended"] is (leaking_split is None)
    if leaking_split == "training":
        candidates = [c for c in result["candidates"] if c["noise_model"] == "deepfilter-ll"]
        assert candidates and all(not c["gates"]["post_speech_noise"] for c in candidates)
    elif leaking_split == "holdout":
        assert result["holdout_evaluated"]
        assert not result["holdout_gates"]["post_speech_noise"]
    else:
        assert result["suppressor_settings"]["model"] == "deepfilter-ll"
        assert result["holdout_passed"]


def test_joined_captures_keep_partial_frame_padding_and_causal_control():
    speech = np.full(481, 0.1, dtype=np.float32)
    noise = np.full(477, 0.001, dtype=np.float32)
    audio, probabilities, noise_start = joint_tuning._join_captures(
        speech, noise,
        np.array([0.2, 0.8], dtype=np.float32),
        np.array([0.1], dtype=np.float32),
    )

    assert noise_start == 960
    np.testing.assert_array_equal(audio[:speech.size], speech)
    np.testing.assert_array_equal(audio[speech.size:noise_start], np.zeros(479))
    np.testing.assert_array_equal(audio[noise_start:], noise)
    np.testing.assert_allclose(probabilities, [0.2, 0.8, 0.1])


def test_continuous_render_splits_aligned_audio_and_activity_after_settling(monkeypatch):
    observed_segments: set[tuple[str, int]] = set()
    speech_sizes = (24_000, 24_023)
    noise_sizes = (48_000, 48_017)

    def simulate(audio, probabilities, _before_gate, _strength, settings):
        values = np.asarray(audio, dtype=np.float32)
        noise_start = int(np.flatnonzero(values == np.float32(0.001))[0])
        assert noise_start % 480 == 0
        frame_count = (values.size + 479) // 480
        assert len(probabilities) == frame_count
        assert all(p == 0.0 for p in probabilities[noise_start // 480:])
        model = settings["noise_model"]
        gain = {"rnnoise": 0.5, "deepfilter-ll": 0.1, "deepfilter": 0.05}[model]
        output = values * np.where(np.abs(values) < 0.01, gain, 0.98)
        # The first 600 ms may still contain a detector/gate/model tail. It
        # must not be counted as settled noise, nor change the speech slice.
        output[noise_start:noise_start + 28_800] = 0.05
        return {
            "output_audio": output,
            "dry_audio": output,
            "suppressor_latency_samples": 1_440 if model == "deepfilter" else 480,
            "runtime_ms": 0.0,
            "auto_makeup_activity": [
                (index / frame_count, 1.0, -60.0, 0.0) for index in range(frame_count)
            ],
        }

    def downstream(audio, _rate, _eq, chain):
        activity = chain["auto_makeup_activity"]
        assert len(activity) == (len(audio) + 479) // 480
        if len(audio) in speech_sizes:
            part = speech_sizes.index(len(audio))
            kind, first_frame = "speech", 0
            np.testing.assert_allclose(audio, 0.098, atol=1.0e-7)
        else:
            part = noise_sizes.index(len(audio))
            kind = "noise"
            first_frame = (speech_sizes[part] + 479) // 480
            np.testing.assert_allclose(audio[:28_800], 0.05, atol=1.0e-7)
        frames = (speech_sizes[part] + 479) // 480 + (noise_sizes[part] + 479) // 480
        np.testing.assert_allclose(
            [row[0] for row in activity],
            np.arange(first_frame, first_frame + len(activity)) / frames,
        )
        observed_segments.add((kind, len(audio)))
        return _downstream_metrics(audio)

    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate)
    monkeypatch.setattr(joint_tuning, "simulate_candidate_chain", downstream)
    args = list(_tuning_args())
    args[0] = np.full(sum(noise_sizes), 0.001, dtype=np.float32)
    args[1] = np.full(sum(speech_sizes), 0.1, dtype=np.float32)
    result = joint_tuning.tune_gate_suppression_dynamics(
        *args,
        vad_probabilities=np.ones(100, dtype=np.float32),
        noise_vad_probabilities=np.zeros(100, dtype=np.float32),
    )

    assert result["apply_recommended"], result.get("failure_reasons")
    assert result["holdout_passed"]
    assert observed_segments == {
        ("speech", speech_sizes[0]), ("speech", speech_sizes[1]),
        ("noise", noise_sizes[0]), ("noise", noise_sizes[1]),
    }
    selected = result["selected_candidate"]["metrics"]
    assert selected["post_speech_noise_frame_count"] == 40
    assert selected["noise_attenuation_db"] == pytest.approx(20.0, abs=1.0e-5)


@pytest.mark.parametrize("noise_samples", [57_600, 57_617])
def test_joint_tuning_retains_incumbent_without_settled_noise_support(monkeypatch, noise_samples):
    def simulate(audio, _probabilities, _before_gate, _strength, settings):
        return {
            "output_audio": audio,
            "dry_audio": audio,
            "suppressor_latency_samples": 480,
            "runtime_ms": 0.0,
            "auto_makeup_activity": _activity_tape(audio, settings),
        }

    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate)
    monkeypatch.setattr(joint_tuning, "simulate_candidate_chain", _downstream_metrics)
    args = list(_tuning_args())
    args[0] = np.full(noise_samples, 0.001, dtype=np.float32)
    result = joint_tuning.tune_gate_suppression_dynamics(*args)

    assert not result["apply_recommended"]
    assert result["gate_settings"] == args[3]
    assert not result["gates"]["post_speech_noise_evidence"]
    assert any("no complete frame after the 600 ms" in reason for reason in result["reasons"])


def test_joint_selection_reports_unavailable_models_and_retains_incumbent(monkeypatch):
    def simulate(audio, _probabilities, _before_gate, _strength, settings):
        model = settings["noise_model"]
        if model != "rnnoise":
            raise RuntimeError(f"{model} backend unavailable")
        output = np.asarray(audio, dtype=np.float32)
        return {
            "output_audio": output,
            "dry_audio": output,
            "suppressor_latency_samples": 480,
            "runtime_ms": 0.0,
            "auto_makeup_activity": _activity_tape(audio, settings),
        }

    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate)
    monkeypatch.setattr(joint_tuning, "simulate_candidate_chain", _downstream_metrics)
    args = _tuning_args()
    result = joint_tuning.tune_gate_suppression_dynamics(
        *args,
        vad_probabilities=np.ones(100, dtype=np.float32),
        noise_vad_probabilities=np.zeros(100, dtype=np.float32),
        incumbent_settings={
            "gate": dict(args[3]),
            "suppressor": {"model": "rnnoise", "enabled": True, "strength": 0.62},
        },
    )

    assert not result["apply_recommended"]
    assert result["suppressor_settings"]["model"] == "rnnoise"
    assert result["suppressor_settings"]["strength"] == 0.62
    statuses = {entry["model"]: entry["status"] for entry in result["model_evaluations"]}
    assert statuses["rnnoise"] == "available"
    assert statuses["deepfilter-ll"] == "unavailable"
    assert statuses["deepfilter"] == "unavailable"
    assert any("backend unavailable" in entry.get("reason", "") for entry in result["model_evaluations"])


@pytest.mark.parametrize("noise_available", [True, False])
def test_joint_tuning_refreshes_probabilities_for_incumbent_pre_gain(monkeypatch, noise_available):
    observed_gains: list[float] = []

    def analyze(_audio, _sample_rate, *, threshold, pre_gain):
        del threshold
        observed_gains.append(float(pre_gain))
        if not noise_available and len(observed_gains) == 2:
            return None, "energy_fallback"
        return np.ones(100, dtype=np.float32), "silero"

    def simulate(audio, _probabilities, _before_gate, _strength, settings):
        output = np.asarray(audio, dtype=np.float32)
        return {
            "output_audio": output,
            "dry_audio": output,
            "suppressor_latency_samples": 480,
            "runtime_ms": 0.0,
            "auto_makeup_activity": _activity_tape(audio, settings),
        }

    monkeypatch.setattr(joint_tuning, "analyze_offline_vad", analyze)
    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate)
    monkeypatch.setattr(joint_tuning, "simulate_candidate_chain", _downstream_metrics)
    args = list(_tuning_args())
    args[3] = {**args[3], "vad_pre_gain": 1.0}
    incumbent_gate = {**args[3], "vad_pre_gain": 2.0}

    result = joint_tuning.tune_gate_suppression_dynamics(
        *args,
        vad_probabilities=np.ones(100, dtype=np.float32),
        noise_vad_probabilities=np.zeros(100, dtype=np.float32),
        incumbent_settings={
            "gate": incumbent_gate,
            "suppressor": {"enabled": True, "strength": 1.0, "model": "rnnoise"},
        },
    )

    assert observed_gains == [2.0, 2.0]
    if not noise_available:
        assert not result["apply_recommended"]
        assert result["gate_settings"] == incumbent_gate


@pytest.mark.parametrize("speech_neural_vad,noise_neural_vad", [
    (True, True), (False, False), (True, False), (False, True),
])
def test_joint_downstream_uses_neural_vad_separately_from_energy_fallback(
    monkeypatch, speech_neural_vad, noise_neural_vad,
):
    monkeypatch.setattr(joint_tuning, "_model_order", lambda _model: ["rnnoise"])
    args = list(_tuning_args())
    args[4] = {**args[4], "auto_makeup_enabled": True}
    observed: list[tuple[float, dict[str, Any]]] = []

    def simulate(audio, _probabilities, _before_gate, _strength, _settings):
        values = np.asarray(audio, dtype=np.float32)
        output = values * (0.5 if float(np.mean(np.abs(values))) < 0.01 else 1.0)
        return {
            "output_audio": output,
            "dry_audio": values,
            "suppressor_latency_samples": 480,
            "runtime_ms": 0.0,
            "auto_makeup_activity": _activity_tape(audio, _settings),
        }

    def downstream(audio, _sample_rate, _eq_settings, chain_settings):
        observed.append((float(np.mean(np.abs(audio))), dict(chain_settings)))
        return _downstream_metrics(audio)

    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate)
    monkeypatch.setattr(joint_tuning, "simulate_candidate_chain", downstream)
    speech_vad = np.linspace(0.1, 0.9, 100, dtype=np.float32)
    noise_vad = np.linspace(0.9, 0.1, 100, dtype=np.float32)
    kwargs: dict[str, Any] = {
        "incumbent_settings": {
            "gate": dict(args[3]),
            "suppressor": {"model": "rnnoise", "enabled": True, "strength": 1.0},
        },
    }
    if speech_neural_vad:
        kwargs["vad_probabilities"] = speech_vad
    if noise_neural_vad:
        kwargs["noise_vad_probabilities"] = noise_vad

    result = joint_tuning.tune_gate_suppression_dynamics(*args, **kwargs)

    if speech_neural_vad != noise_neural_vad:
        assert not observed
        assert not result["apply_recommended"]
        assert result["gate_settings"] == args[3]
        assert result["status"] == "unavailable"
        assert "VAD availability differs" in result["reason"]
        return
    assert observed
    assert result["candidates"]
    assert all(
        bool(candidate["metrics"]["continuous_vad_available"])
        == (speech_neural_vad and noise_neural_vad)
        for candidate in result["candidates"]
    )
    if not (speech_neural_vad and noise_neural_vad):
        assert all("vad_probabilities" not in chain for _, chain in observed)
        assert all(
            all(row[1] == 0.0 for row in chain["auto_makeup_activity"])
            for _, chain in observed
        )
        return

    for mean, chain in observed:
        assert "vad_probabilities" not in chain
        assert all(row[1] == 1.0 for row in chain["auto_makeup_activity"])
        kind = "noise" if mean < 0.01 else "speech"
        expected_probability = 0.2 if kind == "noise" else 0.8
        assert all(
            row[0] == expected_probability
            for row in chain["auto_makeup_activity"]
        )


def test_joint_tuning_rejects_missing_auto_makeup_activity(monkeypatch):
    monkeypatch.setattr(joint_tuning, "_model_order", lambda _model: ["rnnoise"])
    args = _tuning_args()

    def simulate(audio, _probabilities, _before_gate, _strength, _settings):
        values = np.asarray(audio, dtype=np.float32)
        return {
            "output_audio": values,
            "dry_audio": values,
            "suppressor_latency_samples": 480,
            "runtime_ms": 0.0,
        }

    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate)
    monkeypatch.setattr(
        joint_tuning,
        "simulate_candidate_chain",
        lambda *_args, **_kwargs: pytest.fail("downstream must not run"),
    )
    result = joint_tuning.tune_gate_suppression_dynamics(
        *args,
        vad_probabilities=np.ones(100, dtype=np.float32),
        noise_vad_probabilities=np.zeros(100, dtype=np.float32),
        incumbent_settings={
            "gate": dict(args[3]),
            "suppressor": {"model": "rnnoise", "enabled": True, "strength": 1.0},
        },
    )

    assert not result["apply_recommended"]
    assert "auto_makeup_activity" in result["reason"]


def test_deepfilter_incumbent_keeps_selection_inside_deepfilter_family(monkeypatch):
    preset = Preset()
    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: None)
    result = joint_tuning.tune_gate_suppression_dynamics(
        np.full(48_000, 0.001, dtype=np.float32),
        np.full(96_000, 0.1, dtype=np.float32),
        48_000,
        asdict(preset.gate),
        asdict(preset.compressor),
        {
            "band_freqs": preset.eq.band_freqs,
            "band_gains": preset.eq.band_gains,
            "band_qs": preset.eq.band_qs,
        },
        asdict(preset.deesser),
        asdict(preset.limiter),
        incumbent_settings={
            "gate": asdict(preset.gate),
            "suppressor": {
                "enabled": True,
                "strength": 0.62,
                "model": "deepfilter-ll",
            },
        },
        noise_model="deepfilter-ll",
    )
    assert [entry["model"] for entry in result["model_evaluations"]] == [
        "deepfilter-ll",
        "deepfilter",
    ]
    policy = result["model_selection_policy"]
    assert policy["id"] == "retain_deepfilter_family_v1"
    assert policy["excluded_models"] == ["rnnoise"]


def test_latency_gate_accepts_the_configured_35_ms_boundary():
    metrics = {
        "speech_retained_ratio": 0.90,
        "false_closure_rate": 0.0,
        "tail_retained_ratio": 0.60,
        "noise_attenuation_db": 6.0,
        "post_speech_noise_frame_count": 20.0,
        "chatter_events": 0.0,
        "compressor_p95_db": 3.5,
        "compressor_target_p95_db": 3.5,
        "compressor_peak_db": 4.0,
        "compressor_peak_cap_db": 8.0,
        "output_true_peak_db": -1.0,
        "limiter_ceiling_db": -0.5,
        "pre_limiter_headroom_db": 1.0,
        "non_finite_output": 0.0,
        "runtime_factor": 0.5,
        "suppressor_latency_ms": 35.0,
        "max_suppressor_latency_ms": 35.0,
    }
    assert joint_tuning._gates(metrics)["suppressor_latency"]


def test_training_noise_floor_does_not_use_heldout_noise(monkeypatch):
    preset = Preset()
    gate = {
        **asdict(preset.gate),
        "gate_mode": 1,
        "auto_threshold_enabled": True,
        "gate_margin_db": 10.0,
    }
    speech = np.full(48_000, 0.1, dtype=np.float32)
    training_noise = np.full(24_000, 0.001, dtype=np.float32)
    noise_a = np.concatenate((training_noise, np.full(24_000, 0.002, dtype=np.float32)))
    noise_b = np.concatenate((training_noise, np.full(24_000, 0.2, dtype=np.float32)))
    observed_thresholds: list[float] = []

    def simulate(audio, _probabilities, _before_gate, _strength, settings):
        observed_thresholds.append(float(settings["gate_threshold_db"]))
        output = np.asarray(audio, dtype=np.float32)
        return {
            "output_audio": output,
            "dry_audio": output,
            "suppressor_latency_samples": 480,
            "runtime_ms": 0.0,
            "auto_makeup_activity": _activity_tape(audio, settings),
        }

    monkeypatch.setattr(joint_tuning, "_load_native_simulator", lambda: simulate)
    monkeypatch.setattr(joint_tuning, "simulate_candidate_chain", _downstream_metrics)

    def run(noise: np.ndarray) -> list[float]:
        observed_thresholds.clear()
        joint_tuning.tune_gate_suppression_dynamics(
            noise,
            speech,
            48_000,
            gate,
            asdict(preset.compressor),
            {
                "band_freqs": preset.eq.band_freqs,
                "band_gains": preset.eq.band_gains,
                "band_qs": preset.eq.band_qs,
            },
            asdict(preset.deesser),
            asdict(preset.limiter),
            vad_probabilities=np.ones(100, dtype=np.float32),
            noise_vad_probabilities=np.zeros(100, dtype=np.float32),
            incumbent_settings={
                "gate": gate,
                "suppressor": {"enabled": True, "strength": 1.0, "model": "rnnoise"},
            },
        )
        return list(observed_thresholds)

    thresholds_a = run(noise_a)
    thresholds_b = run(noise_b)
    assert thresholds_a
    assert thresholds_a == thresholds_b
    assert {round(threshold, 5) for threshold in thresholds_a} <= {-53.0, -50.0, -47.0}
