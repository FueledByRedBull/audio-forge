"""Focused checks for Voice Setup's combined-chain verification."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from mic_eq.analysis import voice_setup
from mic_eq.analysis.cancellation import AnalysisCancelled
from mic_eq.config import EQ_FREQUENCIES, EQSettings, Preset, build_eq_candidate_settings


def _setup_result(*, model: str = "rnnoise", suppressor_enabled: bool = True) -> dict:
    preset = Preset()
    suppressor = {**asdict(preset.rnnoise), "model": model, "enabled": suppressor_enabled}
    return {
        # The dialog passes EQState.get_settings(): typed fields plus the legacy band arrays.
        "eq_settings": _eq_state_payload(preset.eq.to_dict()),
        "gate_settings": asdict(preset.gate),
        "suppressor_settings": suppressor,
        "deesser_settings": asdict(preset.deesser),
        "compressor_settings": {
            **asdict(preset.compressor),
            "target_p95_reduction_db": 3.5,
            "peak_reduction_cap_db": 8.0,
        },
        "limiter_settings": asdict(preset.limiter),
        "verification_chain_settings": {
            "full_chain": True,
            "input_pre_filtered": False,
            "input_cleanup_mode": "gentle",
            "processing_mode": "normal",
            "gate": asdict(preset.gate),
            "suppressor": suppressor,
        },
    }


def _patch_measurements(
    monkeypatch, *, drop_voice: bool, frontend=None, metrics=None, calls: list[dict] | None = None
) -> list[dict]:
    """Stub the renderer; ``frontend`` maps the take to what the frontend-only render returns.

    ``metrics`` overrides what every render reports about the dynamics stages. Renders are logged in ``calls``.
    """
    calls = [] if calls is None else calls

    class Spectrum:
        freqs = np.geomspace(80.0, 12_000.0, 16)
        median_spectrum_db = np.zeros(16)
        spectral_snr_db = np.zeros(16)
        snr_db = 0.0

    def simulate(audio, _sample_rate, eq_settings, chain):
        calls.append({"eq": eq_settings, "chain": dict(chain)})
        output = np.zeros_like(audio) if drop_voice and len(calls) == 1 else audio
        if frontend is not None and chain["limiter"] == {"enabled": False}:
            output = frontend(audio)
        return {
            "simulation_backend": "rust",
            "output_audio": output.tolist(),
            "compressor_gain_reduction_p95_db": 0.0,
            "compressor_gain_reduction_db": 0.0,
            "deesser_gain_reduction_p95_db": 0.0,
            "output_true_peak_db": -3.0,
            "limiter_effective_ceiling_db": -1.5,
            "true_peak_limited_events": 0,
            "limiter_gain_reduction_db": 0.0,
            "true_peak_limiter_gain_reduction_db": 0.0,
            "output_rms_db": -26.0,
            "input_rms_db": -26.0,
            **(metrics or {}),
        }

    monkeypatch.setattr(voice_setup, "simulate_candidate_chain", simulate)
    monkeypatch.setattr(voice_setup, "analyze_voice_spectrum", lambda *_args, **_kwargs: Spectrum())
    monkeypatch.setattr(voice_setup, "_shape_error_db", lambda *_args, **_kwargs: 0.0)
    def features(audio, *_args, **_kwargs):
        frame_db = -120.0 if not np.any(audio) else -26.0
        count = max(1, len(audio) // 960)
        return {
            "active_loudness_spread_db": 0.0,
            "frame_db": np.full(count, frame_db),
            "active_frame_mask": np.ones(count, dtype=bool),
        }

    monkeypatch.setattr(voice_setup, "_vad_masked_speech_features", features)
    return calls


def test_verification_renders_exact_full_chain_and_typed_eq(monkeypatch) -> None:
    calls = _patch_measurements(monkeypatch, drop_voice=False)
    sample_rate = 48_000
    noise = np.full(sample_rate * 2, 0.001, dtype=np.float32)
    original = np.full(sample_rate * 3, 0.04, dtype=np.float32)
    verification = np.full(sample_rate * 3, 0.04, dtype=np.float32)

    result = voice_setup.validate_voice_setup_verification(
        noise,
        original,
        verification,
        sample_rate,
        _setup_result(),
        "flat",
        raw_noise_audio=noise,
        raw_verification_speech_audio=verification,
    )

    assert result["decision"] == "accept"
    assert result["full_chain"] is True
    # The take, the room noise, the frontend alone for the dulling advice, then the take
    # and the room noise again with the suppressor off for the tonal check.
    assert len(calls) == 5
    assert calls[2]["chain"]["limiter"] == {"enabled": False}
    assert calls[2]["eq"]["enabled"] is False
    assert [call["chain"]["suppressor"]["enabled"] for call in calls] == [True, True, True, False, False]
    assert all(call["chain"]["full_chain"] is True for call in calls)
    assert all(call["chain"]["input_pre_filtered"] is False for call in calls)
    assert all(call["chain"]["input_cleanup_mode"] == "gentle" for call in calls)
    assert all(call["chain"]["processing_mode"] == "normal" for call in calls)
    assert all("bands" in call["eq"] for call in calls)


def test_verification_rejects_full_chain_voice_dropout(monkeypatch) -> None:
    _patch_measurements(monkeypatch, drop_voice=True)
    sample_rate = 48_000
    noise = np.full(sample_rate * 2, 0.001, dtype=np.float32)
    speech = np.full(sample_rate * 3, 0.04, dtype=np.float32)

    result = voice_setup.validate_voice_setup_verification(
        noise,
        speech,
        speech,
        sample_rate,
        _setup_result(),
        "flat",
        raw_noise_audio=noise,
        raw_verification_speech_audio=speech,
    )

    assert result["decision"] == "rollback"
    assert result["reason_codes"] == ["voice_gate_silencing"]
    assert result["full_chain_voice_dropout"] is True


SAMPLE_RATE = 48_000
SIBILANTS = [(0.5 + second, 0.65 + second) for second in range(6)]
DULLING_SENTENCE = (
    "noise suppression is dulling consonants by about 9 dB; "
    "lower its strength or use DeepFilterNet LL"
)


def _take(bursts, *, sibilant_gain_db=0.0, body_gain_db=0.0, noise=0.0005, seconds=6.0) -> np.ndarray:
    """A steady voiced body with 6.5 kHz bursts at the given (start, end) seconds, and some noise."""
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    body = sum(amplitude * np.sin(2 * np.pi * hz * t) for hz, amplitude in ((140.0, 0.11), (220.0, 0.07), (440.0, 0.05)))
    sibilance = 0.2 * np.sin(2 * np.pi * 6500.0 * t) * sum((t >= start) & (t < end) for start, end in bursts)
    hiss = noise * np.random.default_rng(5).normal(size=t.size)
    return (body * 10 ** (body_gain_db / 20) + sibilance * 10 ** (sibilant_gain_db / 20) + hiss).astype(np.float32)


def _room(seconds=2.0) -> np.ndarray:
    return (0.0025 * np.random.default_rng(6).normal(size=int(SAMPLE_RATE * seconds))).astype(np.float32)


def test_sibilant_change_reads_the_frontend_off_the_take_alone() -> None:
    voice = _take(SIBILANTS)
    for sibilant_gain_db, body_gain_db, expected in ((-9.0, 0.0, -9.0), (0.0, 0.0, 0.0), (0.0, -6.0, 6.0)):
        change, frames = voice_setup._sibilant_change_db(
            voice, _take(SIBILANTS, sibilant_gain_db=sibilant_gain_db, body_gain_db=body_gain_db), _room(), SAMPLE_RATE
        )
        assert change == pytest.approx(expected, abs=0.3)
        assert frames >= voice_setup.SUPPRESSOR_DULLING_MIN_FRAMES


def test_sibilant_change_abstains_without_enough_sibilant_frames() -> None:
    room = _room()
    # Three hundredths of a second of sibilant is a handful of frames, however much is lost.
    brief = [(1.0, 1.03)]
    change, frames = voice_setup._sibilant_change_db(
        _take(brief), _take(brief, sibilant_gain_db=-20.0), room, SAMPLE_RATE
    )
    assert change is None
    assert 0 < frames < voice_setup.SUPPRESSOR_DULLING_MIN_FRAMES
    # Energy up there that is only room noise is not sibilance, whatever the frontend does to it.
    hissy = _take([], noise=0.0025)
    assert voice_setup._sibilant_change_db(hissy, _take([], noise=0.0), room, SAMPLE_RATE) == (None, 0)


def _verify_full_chain(
    monkeypatch, voice_in, voice_out, *, drop_voice=False, model="rnnoise", metrics=None, original_gain_db=0.0,
    suppressor_enabled=True, calls=None,
) -> dict:
    """Verify ``voice_in`` through a renderer whose frontend-only render returns ``voice_out``.

    ``original_gain_db`` moves the setup passage away from the verification take.
    """
    scale = np.float32(0.05 / np.sqrt(np.mean(np.square(voice_in, dtype=np.float64))))
    take = voice_in * scale
    _patch_measurements(
        monkeypatch, drop_voice=drop_voice, frontend=lambda _audio: voice_out * scale, metrics=metrics, calls=calls
    )
    noise = _room()
    return voice_setup.validate_voice_setup_verification(
        noise, take * np.float32(10 ** (original_gain_db / 20)), take, SAMPLE_RATE,
        _setup_result(model=model, suppressor_enabled=suppressor_enabled),
        "flat", raw_noise_audio=noise, raw_verification_speech_audio=take,
    )


@pytest.mark.parametrize(
    ("decision", "conditions"),
    [
        ("accept", {}),
        ("rollback", {"drop_voice": True}),
        ("reduce", {"metrics": {"compressor_gain_reduction_p95_db": 6.0}}),
        ("retry", {"original_gain_db": -12.0}),
    ],
)
def test_dulling_advice_is_offered_beside_the_decision_and_changes_nothing_else(
    monkeypatch, decision, conditions
) -> None:
    voice_in, voice_out = _take(SIBILANTS), _take(SIBILANTS, sibilant_gain_db=-9.0)

    advised = _verify_full_chain(monkeypatch, voice_in, voice_out, **conditions)
    monkeypatch.setattr(voice_setup, "SUPPRESSOR_DULLING_THRESHOLD_DB", float("inf"))
    plain = _verify_full_chain(monkeypatch, voice_in, voice_out, **conditions)

    assert advised["decision"] == plain["decision"] == decision
    assert advised["reduction_targets"] == plain["reduction_targets"] == (["compressor"] if decision == "reduce" else [])
    assert advised["suppressor_dulling_db"] == pytest.approx(9.0, abs=0.3)
    assert advised["suppressor_dulling_abstained"] is False
    assert advised["reason_codes"] == [*plain["reason_codes"], "suppressor_dulling"]
    assert advised["suppressor_dulling_advice"] == DULLING_SENTENCE
    assert plain["suppressor_dulling_advice"] is None
    assert advised["reasons"] == plain["reasons"]
    assert "suppressor_dulling" not in plain["reason_codes"]
    assert {key: value for key, value in advised.items() if key not in ("reason_codes", "suppressor_dulling_advice")} == {
        key: value for key, value in plain.items() if key not in ("reason_codes", "suppressor_dulling_advice")
    }


@pytest.mark.parametrize(
    ("bursts", "sibilant_gain_db", "dulling_db"),
    [(SIBILANTS, 0.0, 0.0), ([(1.0, 1.03)], -20.0, None)],
    ids=["intact", "too-few-sibilant-frames"],
)
def test_dulling_advice_stays_quiet_when_the_take_is_intact_or_unmeasurable(
    monkeypatch, bursts, sibilant_gain_db, dulling_db
) -> None:
    result = _verify_full_chain(
        monkeypatch, _take(bursts), _take(bursts, sibilant_gain_db=sibilant_gain_db)
    )

    assert result["decision"] == "accept"
    assert result["reason_codes"] == ["accepted"]
    assert result["suppressor_dulling_abstained"] is (dulling_db is None)
    assert result["suppressor_dulling_db"] == (None if dulling_db is None else pytest.approx(dulling_db, abs=0.3))
    assert result["suppressor_dulling_advice"] is None


@pytest.mark.parametrize(
    ("model", "remedy"),
    [
        ("rnnoise", "lower its strength or use DeepFilterNet LL"),
        ("deepfilter", "lower its strength"),
        ("deepfilter-ll", "lower its strength"),
    ],
)
def test_dulling_advice_suggests_deepfilternet_ll_only_to_rnnoise(monkeypatch, model, remedy) -> None:
    result = _verify_full_chain(
        monkeypatch, _take(SIBILANTS), _take(SIBILANTS, sibilant_gain_db=-9.0), model=model
    )

    assert result["reason_codes"][-1] == "suppressor_dulling"
    assert result["suppressor_dulling_advice"] == f"noise suppression is dulling consonants by about 9 dB; {remedy}"
    assert result["reasons"] == ["repeatability and full-chain DSP constraints passed"]


def test_a_proposal_without_a_suppressor_is_not_measured_for_dulling(monkeypatch) -> None:
    calls: list[dict] = []
    # Were the frontend rendered, the dulled voice below would earn the advice.
    result = _verify_full_chain(
        monkeypatch, _take(SIBILANTS), _take(SIBILANTS, sibilant_gain_db=-9.0), suppressor_enabled=False, calls=calls
    )

    assert result["decision"] == "accept"
    assert result["reason_codes"] == ["accepted"]
    assert not [key for key in result if key.startswith("suppressor_dulling")]
    # Only the take and the room noise are rendered: no frontend alone, no bypass.
    assert len(calls) == 2


def test_downstream_only_verification_gives_no_dulling_advice(monkeypatch) -> None:
    calls = _patch_measurements(monkeypatch, drop_voice=False)
    setup = _setup_result()
    del setup["verification_chain_settings"]
    take = _take(SIBILANTS)

    result = voice_setup.validate_voice_setup_verification(_room(), take, take, SAMPLE_RATE, setup, "flat")

    assert result["full_chain"] is False
    assert len(calls) == 2
    assert not [key for key in result if key.startswith("suppressor_dulling")]


def _verify_raw_take(setup, **kwargs) -> dict:
    noise, take = _room(), _take(SIBILANTS)
    return voice_setup.validate_voice_setup_verification(
        noise, take, take, SAMPLE_RATE, setup, "flat",
        raw_noise_audio=noise, raw_verification_speech_audio=take, **kwargs,
    )


@pytest.mark.parametrize(
    ("renders_before_cancel", "renders_run"), [(3, 3), (4, 5)], ids=["frontend-render", "bypass-renders"]
)
def test_cancelling_during_the_extra_renders_stops_before_any_analysis(
    monkeypatch, renders_before_cancel, renders_run
) -> None:
    calls = _patch_measurements(monkeypatch, drop_voice=False)
    monkeypatch.setattr(
        voice_setup, "analyze_voice_spectrum", lambda *_a, **_k: pytest.fail("analysis went on after the cancel")
    )

    with pytest.raises(AnalysisCancelled):
        _verify_raw_take(_setup_result(), cancel_check=lambda: len(calls) >= renders_before_cancel)

    assert len(calls) == renders_run


def test_a_suppressor_under_the_rnnoise_key_is_still_switched_off_for_the_tonal_check(monkeypatch) -> None:
    calls = _patch_measurements(monkeypatch, drop_voice=False)
    setup = _setup_result(model="deepfilter")
    suppressor = setup.pop("suppressor_settings")
    # The dialog's chain snapshot names the suppressor "rnnoise"; it is the only place it appears here.
    chain = setup["verification_chain_settings"]
    chain["rnnoise"] = chain.pop("suppressor")

    _verify_raw_take(setup)

    assert [call["chain"]["suppressor"] for call in calls] == [suppressor] * 3 + [{**suppressor, "enabled": False}] * 2
    assert not [call for call in calls if "rnnoise" in call["chain"]]


def _broadband_voice() -> np.ndarray:
    """Harmonics of a wavering 140 Hz voice over hiss: enough spectrum for a tonal reading."""
    t = np.arange(SAMPLE_RATE * 4) / SAMPLE_RATE
    phase = 2 * np.pi * np.cumsum(140.0 * (1 + 0.1 * np.sin(2 * np.pi * 0.7 * t))) / SAMPLE_RATE
    amplitudes = {1: 0.08, 2: 0.06, 3: 0.05, 4: 0.05, 5: 0.04, 8: 0.03, 12: 0.02, 20: 0.01, 30: 0.006}
    body = sum(amplitude * np.sin(harmonic * phase) for harmonic, amplitude in amplitudes.items())
    voice = body * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t) ** 2) + 0.004 * np.random.default_rng(7).normal(size=t.size)
    return voice.astype(np.float32)


def _patch_spectral_renderer(monkeypatch, *, suppressor_cut_db: float, damage: dict | None = None) -> list[dict]:
    """Stub only the renderer, so that the spectrum and the tonal check are measured for real.

    A chain whose suppressor is on cuts everything above 3 kHz by ``suppressor_cut_db`` and is then
    ``damage``d (scaled voice or noise audio, overridden renderer metrics). The same chain with the
    suppressor off renders cleanly. EQ bands act as bells half an octave wide.
    """
    damage = damage or {}
    spectrum, shape_error = voice_setup.analyze_voice_spectrum, voice_setup._shape_error_db
    calls = _patch_measurements(monkeypatch, drop_voice=False)
    monkeypatch.setattr(voice_setup, "analyze_voice_spectrum", spectrum)
    monkeypatch.setattr(voice_setup, "_shape_error_db", shape_error)
    render = voice_setup.simulate_candidate_chain
    noise_size = _room().size

    def simulate(audio, sample_rate, eq_settings, chain):
        suppressed = chain["suppressor"]["enabled"]
        freqs = np.fft.rfftfreq(audio.size, 1 / sample_rate)
        gain_db = np.zeros_like(freqs)
        if suppressed:
            gain_db[freqs > 3_000.0] -= suppressor_cut_db
        if eq_settings["enabled"]:
            for band in eq_settings["bands"]:
                octaves = np.log2(np.maximum(freqs, 1.0) / band["frequency_hz"])
                gain_db += band["gain_db"] * np.exp(-0.5 * (octaves / 0.5) ** 2)
        output = np.fft.irfft(np.fft.rfft(audio) * 10 ** (gain_db / 20), audio.size).astype(np.float32)
        if suppressed:
            output *= damage.get("noise_gain" if audio.size == noise_size else "voice_gain", 1.0)
        rendered = render(output, sample_rate, eq_settings, chain)
        if suppressed:
            rendered.update(damage.get("metrics", {}))
        return rendered

    monkeypatch.setattr(voice_setup, "simulate_candidate_chain", simulate)
    return calls


def _verify_tonal(
    monkeypatch, *, voice=None, wrong_eq_db=0.0, suppressor_cut_db=9.0, damage=None, suppressor_enabled=True,
    wrong_eq_hz=2_500.0,
) -> tuple[dict, list[dict]]:
    """Verify a voice through a suppressor that reshapes it; ``wrong_eq_db`` is added at the EQ band ``wrong_eq_hz`` (default 2.5 kHz, nearest 3 kHz)."""
    voice = _broadband_voice() if voice is None else voice
    take = voice * np.float32(0.05 / np.sqrt(np.mean(np.square(voice, dtype=np.float64))))
    calls = _patch_spectral_renderer(monkeypatch, suppressor_cut_db=suppressor_cut_db, damage=damage)
    setup = _setup_result(suppressor_enabled=suppressor_enabled)
    next(band for band in setup["eq_settings"]["bands"] if band["frequency_hz"] == wrong_eq_hz)["gain_db"] = wrong_eq_db
    noise = _room()
    result = voice_setup.validate_voice_setup_verification(
        noise, take, take, SAMPLE_RATE, setup, "flat", raw_noise_audio=noise, raw_verification_speech_audio=take
    )
    return result, calls


@pytest.mark.parametrize("bypass", [True, False], ids=["bypassed", "released"])
def test_the_suppressor_reshaping_the_spectrum_is_a_tonal_error_only_when_the_check_is_released(
    monkeypatch, bypass
) -> None:
    monkeypatch.setattr(voice_setup, "TONAL_CHECK_BYPASSES_SUPPRESSOR", bypass)

    result, calls = _verify_tonal(monkeypatch)

    suppressor_flags = [call["chain"]["suppressor"]["enabled"] for call in calls]
    if bypass:
        assert result["decision"] == "accept"
        assert result["reason_codes"] == ["accepted"]
        assert result["spectral_target_error_after_db"] == pytest.approx(
            result["spectral_target_error_before_db"], abs=0.01
        )
        assert suppressor_flags == [True, True, True, False, False]
    else:
        assert result["decision"] == "rollback"
        assert result["reason_codes"] == ["target_error"]
        assert suppressor_flags == [True, True, True]


def test_a_low_band_error_the_by_band_reading_catches_passes_as_released(monkeypatch) -> None:
    assert voice_setup.TONAL_CHECK_COUNTS_EVERY_BAND is False

    result, _ = _verify_tonal(monkeypatch, wrong_eq_db=6.0, wrong_eq_hz=160.0)

    assert result["decision"] == "accept"
    assert result["reason_codes"] == ["accepted"]
    assert not any(key.startswith("tone_band_error") for key in result)


def test_counting_every_band_rolls_back_a_low_band_error_and_restores(monkeypatch) -> None:
    with monkeypatch.context() as scope:
        scope.setattr(voice_setup, "TONAL_CHECK_COUNTS_EVERY_BAND", True)
        result, _ = _verify_tonal(monkeypatch, wrong_eq_db=6.0, wrong_eq_hz=160.0)

    assert result["decision"] == "rollback"
    assert result["reason_codes"] == ["target_error"]
    assert result["tone_band_error_after_db"] > result["tone_band_error_before_db"] + 1.0
    assert result["spectral_target_error_after_db"] <= result["spectral_target_error_before_db"] + 1.0
    assert voice_setup.TONAL_CHECK_COUNTS_EVERY_BAND is False


def test_a_wrong_eq_is_still_a_tonal_error_with_a_suppressor_that_reshapes_the_spectrum(monkeypatch) -> None:
    result, _ = _verify_tonal(monkeypatch, wrong_eq_db=9.0)

    assert result["decision"] == "rollback"
    assert result["reason_codes"] == ["target_error"]
    assert result["spectral_target_error_after_db"] > result["spectral_target_error_before_db"] + 1.0


@pytest.mark.parametrize(
    ("damage", "reason_codes"),
    [
        (
            {"metrics": {"compressor_gain_reduction_p95_db": 6.0, "limiter_gain_reduction_db": 2.5,
                         "deesser_gain_reduction_p95_db": 5.9}},
            ["compressor_excess", "limiter_excess", "deesser_excess"],
        ),
        (
            {"metrics": {"compressor_gain_reduction_db": 9.0, "output_true_peak_db": 0.0}},
            ["compressor_peak_cap", "true_peak_ceiling"],
        ),
        ({"noise_gain": 5.0}, ["noise_safety_limit", "snr_safety_limit"]),
        ({"voice_gain": 0.0}, ["voice_gate_silencing", "snr_safety_limit"]),
    ],
    ids=["reductions", "peaks", "noise-and-snr", "dropout"],
)
def test_every_finding_but_the_tonal_check_reads_the_real_render(monkeypatch, damage, reason_codes) -> None:
    result, _ = _verify_tonal(monkeypatch, damage=damage)

    assert result["reason_codes"] == reason_codes


def test_the_dulling_advice_reads_the_real_render_not_the_tonal_bypass(monkeypatch) -> None:
    result, _ = _verify_tonal(monkeypatch, voice=_take(SIBILANTS))

    assert result["reason_codes"] == ["accepted", "suppressor_dulling"]
    assert result["suppressor_dulling_db"] == pytest.approx(9.0, abs=0.3)


def test_without_the_suppressor_in_the_proposal_the_tonal_check_is_as_released(monkeypatch) -> None:
    outcomes = {}
    for bypass in (True, False):
        monkeypatch.setattr(voice_setup, "TONAL_CHECK_BYPASSES_SUPPRESSOR", bypass)
        outcomes[bypass], calls = _verify_tonal(monkeypatch, suppressor_enabled=False)
        assert len(calls) == 2

    assert outcomes[True] == outcomes[False]
    assert outcomes[True]["decision"] == "accept"


def test_downstream_only_verification_is_as_released(monkeypatch) -> None:
    outcomes = {}
    for bypass in (True, False):
        monkeypatch.setattr(voice_setup, "TONAL_CHECK_BYPASSES_SUPPRESSOR", bypass)
        calls = _patch_measurements(monkeypatch, drop_voice=False)
        setup = _setup_result()
        del setup["verification_chain_settings"]
        take = _take(SIBILANTS)
        outcomes[bypass] = voice_setup.validate_voice_setup_verification(
            _room(), take, take, SAMPLE_RATE, setup, "flat"
        )
        assert len(calls) == 2

    assert outcomes[True] == outcomes[False]


def _eq_state_payload(eq: dict) -> dict:
    """What EQState.get_settings() hands the dialog: the typed payload plus the legacy band arrays."""
    settings = EQSettings.from_dict(eq)
    return eq | {"band_freqs": settings.band_freqs, "band_gains": settings.band_gains, "band_qs": settings.band_qs}


def _run_initial_analysis(
    monkeypatch, *, processing_mode: str, eq_payload=lambda eq: eq, auto_eq=None, incumbent: bool = True,
    compressor: dict | None = None, reading=lambda chain: {},
):
    sample_rate = 48_000
    noise = np.random.default_rng(42).normal(0, 0.001, sample_rate * 2).astype(np.float32)
    t = np.arange(sample_rate * 4) / sample_rate
    speech = ((0.08 + 0.06 * np.sin(2 * np.pi * 3 * t)) * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    preset = Preset()
    incumbent_eq = preset.eq.to_dict()
    incumbent_eq["bands"][0]["gain_db"] = 2.5
    candidate_eq = {
        "band_freqs": list(EQ_FREQUENCIES),
        "band_gains": [1.0] * len(EQ_FREQUENCIES),
        "band_qs": [1.41] * len(EQ_FREQUENCIES),
        "analysis_confidence": 0.9,
        "apply_recommended": True,
    }
    proposed_gate = {**asdict(preset.gate), "threshold_db": -36.0}
    proposed_suppressor = {"enabled": True, "strength": 0.65, "model": "deepfilter"}
    calls = []
    calibration = {}
    tuned = {}
    progress = []
    frontend_activity = [
        (0.85, 1.0, -46.0, 0.9)
    ] * ((speech.size + 479) // 480)

    def simulate(audio, _sample_rate, eq_settings, chain):
        calls.append((audio, eq_settings, chain))
        if chain.get("return_auto_makeup_activity"):
            return {
                "simulation_backend": "rust",
                "output_audio": audio,
                "auto_makeup_activity": frontend_activity,
            }
        return {
            "simulation_backend": "rust", "output_audio": audio,
            "output_true_peak_db": -3.0, "limiter_effective_ceiling_db": -1.5,
            "compressor_gain_reduction_db": 0.0, "compressor_gain_reduction_p95_db": 0.0,
            "deesser_gain_reduction_db": 0.0, "pre_limiter_true_peak_headroom_db": 2.0,
            "limiter_gain_reduction_db": 0.0, "true_peak_limiter_gain_reduction_db": 0.0,
            **reading(chain),
        }

    def calibrate(**kwargs):
        calibration["speech_audio"] = np.asarray(kwargs["speech_audio"]).copy()
        calibration["auto_makeup_activity"] = kwargs["auto_makeup_activity"]
        calibration["eq_settings"] = kwargs["eq_settings"]
        return kwargs["compressor_settings"], {"backend": "rust"}

    # Leave capture measurement real; stub only the searches and final renderer.
    def tune(*args, **_kwargs):
        tuned["eq_settings"] = args[5]
        return {
            "apply_recommended": True, "gate_settings": proposed_gate,
            "suppressor_settings": proposed_suppressor,
        }

    monkeypatch.setattr(voice_setup, "analyze_auto_eq", auto_eq or (lambda *_a, **_k: (candidate_eq, {})))
    monkeypatch.setattr(voice_setup, "_calibrate_compressor_threshold", calibrate)
    monkeypatch.setattr(voice_setup, "tune_gate_suppression_dynamics", tune)
    monkeypatch.setattr(voice_setup, "simulate_candidate_chain", simulate)
    result = voice_setup.analyze_voice_setup(
        noise * 0.9, speech * 0.9, sample_rate, vad_available=False,
        incumbent_settings={**({"eq": eq_payload(incumbent_eq)} if incumbent else {}),
                            "gate": asdict(preset.gate), "rnnoise": asdict(preset.rnnoise),
                            **({"compressor": compressor} if compressor is not None else {})},
        raw_noise_audio=noise, raw_speech_audio=speech,
        input_cleanup_mode="gentle", processing_mode=processing_mode,
        progress_callback=lambda _text, value: progress.append(value),
    )
    return {
        "result": result,
        "calls": calls,
        "calibration": calibration,
        "tuned": tuned,
        "current_eq": (EQSettings.from_dict(incumbent_eq) if incumbent else EQSettings()).to_dict(),
        "progress": progress,
        "speech": speech,
        "proposed_gate": proposed_gate,
        "proposed_suppressor": proposed_suppressor,
        "expected_eq": build_eq_candidate_settings(
            EQSettings.from_dict(incumbent_eq), candidate_eq["band_freqs"],
            candidate_eq["band_gains"], candidate_eq["band_qs"], layer="correction",
        ).to_dict(),
        "frontend_activity": frontend_activity,
    }


def test_initial_analysis_renders_raw_full_chain_with_the_applied_eq_layers(monkeypatch):
    observed = _run_initial_analysis(monkeypatch, processing_mode="bypass")
    result = observed["result"]
    calls = observed["calls"]
    calibration = observed["calibration"]
    audio, eq, chain = calls[0]
    np.testing.assert_array_equal(audio, observed["speech"])
    assert len(calls) == 2
    assert chain["full_chain"] is True
    assert chain["input_pre_filtered"] is False
    assert chain["input_cleanup_mode"] == "gentle"
    assert chain["processing_mode"] == "bypass"
    assert chain["return_auto_makeup_activity"] is True
    assert chain["gate"] == observed["proposed_gate"]
    assert chain["rnnoise"] == observed["proposed_suppressor"]
    assert chain["deesser"] == {"enabled": False}
    assert chain["compressor"]["enabled"] is False
    assert chain["limiter"] == {"enabled": False}
    assert calibration["auto_makeup_activity"] is observed["frontend_activity"]
    np.testing.assert_array_equal(calibration["speech_audio"], audio)
    assert calibration["eq_settings"] == observed["expected_eq"]
    expected_frontend_eq = {**observed["expected_eq"], "enabled": False}
    assert eq == expected_frontend_eq
    assert eq["layers"]["tone"][0]["gain_db"] == 2.5
    assert observed["progress"] == sorted(observed["progress"])
    assert 55 in observed["progress"]
    assert result["diagnostics"]["offline_validation_passed"] is True
    assert result["diagnostics"]["offline_validation_full_chain"] is True
    assert result["diagnostics"]["full_chain_raw_capture_used"] is True


def test_initial_analysis_accepts_the_eq_payload_the_eq_state_returns(monkeypatch):
    # Every other test here feeds the analysis a clean typed EQ; the app sends the legacy arrays beside it.
    observed = _run_initial_analysis(monkeypatch, processing_mode="normal", eq_payload=_eq_state_payload)

    assert observed["calibration"]["eq_settings"] == observed["expected_eq"]
    diagnostics = observed["result"]["diagnostics"]
    assert diagnostics["offline_validation_passed"] is True
    assert diagnostics["offline_validation"]["simulation_backend"] == "rust"
    assert not [reason for reason in diagnostics["uncertainty_reasons"] if "validation" in reason]


def test_a_malformed_incumbent_eq_is_named_in_the_failed_final_check(monkeypatch):
    observed = _run_initial_analysis(
        monkeypatch, processing_mode="normal", eq_payload=lambda eq: eq | {"bands": eq["bands"][:3]}
    )

    diagnostics = observed["result"]["diagnostics"]
    assert diagnostics["offline_validation"]["error"].startswith("typed candidate EQ settings are unavailable (ValueError: ")
    assert diagnostics["offline_validation_passed"] is False
    assert (
        "full-chain validation failed: " + diagnostics["offline_validation"]["error"]
    ) in diagnostics["uncertainty_reasons"]
    assert "offline DSP validation could not run, so these settings are advisory" in diagnostics["uncertainty_reasons"]
    assert not [reason for reason in diagnostics["uncertainty_reasons"] if "Rust" in reason]


def test_initial_analysis_renders_selected_frontend_before_compressor_for_normal_mode(
    monkeypatch,
):
    observed = _run_initial_analysis(monkeypatch, processing_mode="normal")
    assert len(observed["calls"]) == 2
    assert observed["calls"][0][2]["processing_mode"] == "normal"
    assert observed["calibration"]["auto_makeup_activity"] is observed[
        "frontend_activity"
    ]
    np.testing.assert_array_equal(
        observed["calibration"]["speech_audio"], observed["calls"][0][0]
    )
    assert observed["calibration"]["eq_settings"] == observed["expected_eq"]
    assert observed["progress"] == sorted(observed["progress"])
    assert 55 in observed["progress"]


def test_initial_analysis_raw_mode_keeps_full_chain_path_available(monkeypatch):
    observed = _run_initial_analysis(monkeypatch, processing_mode="raw")
    assert len(observed["calls"]) == 2
    assert observed["calls"][0][2]["processing_mode"] == "raw"
    assert observed["result"]["diagnostics"]["offline_validation_full_chain"] is True


def _abstention(*reasons):
    def analyze(*_args, **_kwargs):
        return {
            "band_freqs": list(EQ_FREQUENCIES),
            "band_gains": [0.0] * len(EQ_FREQUENCIES),
            "band_qs": [1.41] * len(EQ_FREQUENCIES),
            "analysis_confidence": 0.4,
            "apply_recommended": False,
            "recommendation_status": "abstain",
            "abstention_reasons": list(reasons),
        }, {}

    return analyze


def _failed_validation(*failures, tone_only):
    def analyze(*_args, **_kwargs):
        from mic_eq.analysis.failure_detection import AnalysisValidationError

        raise AnalysisValidationError("Recording too unclear. Please try again.", list(failures), tone_only)

    return analyze


_TONE_ONLY_ABSTENTIONS = [
    ["constrained gain solve produced no safe correction"],
    ["no validated correction improved the target safely"],
    ["headroom validation removed the usable correction"],
    [
        "constrained gain solve produced no safe correction",
        "no validated correction improved the target safely",
    ],
]


def _assert_computed_on_the_current_eq(observed):
    current_eq = observed["current_eq"]
    calls = observed["calls"]
    assert observed["tuned"]["eq_settings"] == current_eq
    assert observed["calibration"]["eq_settings"] == current_eq
    assert [eq for _audio, eq, _chain in calls] == [{**current_eq, "enabled": False}, current_eq]
    assert observed["result"]["eq_settings"] == {**current_eq, "apply_recommended": False}


@pytest.mark.parametrize(
    "auto_eq",
    [
        _failed_validation("clipped_gains (7 >= 6)", tone_only=True),
        *(_abstention(*reasons) for reasons in _TONE_ONLY_ABSTENTIONS),
    ],
    ids=["validation", "solve", "validated", "headroom", "solve-and-validated"],
)
@pytest.mark.usefixtures("partial_results")
def test_a_tone_only_refusal_withholds_the_tone_and_computes_the_rest_on_the_users_eq(monkeypatch, auto_eq):
    observed = _run_initial_analysis(monkeypatch, processing_mode="normal", auto_eq=auto_eq)
    diagnostics = observed["result"]["diagnostics"]

    _assert_computed_on_the_current_eq(observed)
    assert diagnostics["tone_withheld"]["reason_code"] == "tone_withheld"
    assert diagnostics["tone_withheld"]["cause"]
    assert diagnostics["offline_validation_passed"] is True
    assert "Auto-EQ abstained from this capture" not in diagnostics["uncertainty_reasons"]


@pytest.mark.usefixtures("partial_results")
def test_a_withheld_tone_leaves_the_tone_out_of_the_setup_confidence(monkeypatch):
    diagnostics = _run_initial_analysis(
        monkeypatch, processing_mode="normal", auto_eq=_abstention(*_TONE_ONLY_ABSTENTIONS[0])
    )["result"]["diagnostics"]

    quiet_room = float(np.clip((-32.0 - diagnostics["conservative_noise_rms_db"]) / 18.0, 0.0, 1.0))
    deesser = diagnostics["deesser_confidence"]
    assert diagnostics["setup_confidence"] == pytest.approx(
        voice_setup._bounded_quality_score([
            (diagnostics["gate_confidence"], 0.25),
            (deesser if diagnostics["deesser_enabled"] else max(deesser, 0.5), 0.15),
            (diagnostics["compressor_confidence"], 0.15),
            (quiet_room, 0.10),
        ])
    )


@pytest.mark.usefixtures("partial_results")
def test_a_withheld_tone_with_no_eq_in_the_incumbent_uses_the_default_eq(monkeypatch):
    observed = _run_initial_analysis(
        monkeypatch, processing_mode="normal", incumbent=False,
        auto_eq=_failed_validation("gain_rms (11.0 > 10.0 dB)", tone_only=True),
    )

    assert observed["current_eq"] == EQSettings().to_dict()
    _assert_computed_on_the_current_eq(observed)


@pytest.mark.parametrize(
    "auto_eq",
    [
        _failed_validation("peak_count (1 too low)", tone_only=False),
        _failed_validation("headroom risk after downstream simulation", tone_only=False),
    ],
    ids=["unclear-capture", "headroom-risk"],
)
@pytest.mark.usefixtures("partial_results")
def test_a_refusal_that_is_not_about_the_tone_still_refuses_the_whole_setup(monkeypatch, auto_eq):
    observed = _run_initial_analysis(monkeypatch, processing_mode="normal", auto_eq=auto_eq)
    result = observed["result"]

    assert result["eq_settings"] is None
    assert result["eq_error"] == "Recording too unclear. Please try again."
    assert observed["calibration"] == {}
    assert result["diagnostics"]["tone_withheld"] is None


@pytest.mark.parametrize(
    "reasons",
    [
        ["capture quality score is too low"],
        ["noise-referenced SNR is too low", "no validated correction improved the target safely"],
        ["room-noise reference is invalid"],
        ["insufficient repeatable voiced windows"],
    ],
)
@pytest.mark.usefixtures("partial_results")
def test_an_abstention_with_a_capture_reason_is_not_withheld(monkeypatch, reasons):
    observed = _run_initial_analysis(monkeypatch, processing_mode="normal", auto_eq=_abstention(*reasons))
    result = observed["result"]

    assert result["diagnostics"]["tone_withheld"] is None
    assert result["diagnostics"]["apply_recommended"] is False
    assert "Auto-EQ abstained from this capture" in result["diagnostics"]["uncertainty_reasons"]
    assert result["eq_settings"]["abstention_reasons"] == reasons


@pytest.mark.usefixtures("partial_results")
def test_a_tone_refusal_stays_whole_when_the_incumbent_eq_cannot_be_read(monkeypatch):
    observed = _run_initial_analysis(
        monkeypatch, processing_mode="normal",
        eq_payload=lambda eq: eq | {"bands": eq["bands"][:3]},
        auto_eq=_failed_validation("clipped_gains (7 >= 6)", tone_only=True),
    )

    assert observed["result"]["eq_settings"] is None
    assert observed["result"]["diagnostics"]["tone_withheld"] is None


_USERS_COMPRESSOR = {
    **asdict(Preset().compressor),
    "threshold_db": -17.25,
    "auto_makeup_enabled": False,
    "makeup_gain_db": 1.0,
}


def _too_hot_unless_users(chain):
    if chain["compressor"]["threshold_db"] == _USERS_COMPRESSOR["threshold_db"]:
        return {}
    return {"pre_limiter_true_peak_headroom_db": -4.0, "limiter_gain_reduction_db": 4.0}


@pytest.mark.usefixtures("compressor_fallback")
def test_a_withheld_compressor_is_checked_in_the_full_chain_as_the_user_has_it(monkeypatch):
    observed = _run_initial_analysis(
        monkeypatch, processing_mode="normal", compressor=_USERS_COMPRESSOR, reading=_too_hot_unless_users
    )
    result = observed["result"]
    diagnostics = result["diagnostics"]
    audio, eq, chain = observed["calls"][-1]

    # The last render is the final check with the user's compressor, on the proposal's other steps.
    np.testing.assert_array_equal(audio, observed["speech"])
    assert eq == observed["expected_eq"]
    assert chain["compressor"] == _USERS_COMPRESSOR
    assert chain["gate"] == observed["proposed_gate"]
    assert chain["rnnoise"] == observed["proposed_suppressor"]
    assert chain["full_chain"] is True and chain["input_pre_filtered"] is False
    assert chain["processing_mode"] == "normal" and chain["input_cleanup_mode"] == "gentle"
    assert result["compressor_settings"] == _USERS_COMPRESSOR
    assert result["gate_settings"] == observed["proposed_gate"]
    assert result["suppressor_settings"] == observed["proposed_suppressor"]
    # This synthetic capture is also a weak one, so the weak-capture cause comes with the headroom one.
    assert diagnostics["compressor_withheld"] == {
        "reason_code": "compressor_withheld",
        "cause": ["headroom", "weak_capture"],
    }
    assert diagnostics["offline_validation_passed"] is True
    assert diagnostics["offline_validation_full_chain"] is True
    assert "output_audio" not in diagnostics["offline_validation"]


@pytest.mark.usefixtures("compressor_fallback")
def test_a_headroom_failure_the_current_compressor_shares_is_still_refused(monkeypatch):
    observed = _run_initial_analysis(
        monkeypatch,
        processing_mode="normal",
        compressor=_USERS_COMPRESSOR,
        reading=lambda chain: {"pre_limiter_true_peak_headroom_db": -4.0, "limiter_gain_reduction_db": 4.0},
    )
    result = observed["result"]

    assert result["diagnostics"]["compressor_withheld"] is None
    assert result["diagnostics"]["offline_validation_passed"] is False
    assert result["compressor_settings"]["threshold_db"] != _USERS_COMPRESSOR["threshold_db"]
    assert observed["calls"][-1][2]["compressor"]["threshold_db"] == _USERS_COMPRESSOR["threshold_db"]


_WITHHELD_COMPRESSOR = {"reason_code": "compressor_withheld", "cause": ["headroom"]}


def _verify_with_compressor_metrics(monkeypatch, metrics, *, withheld):
    _patch_measurements(monkeypatch, drop_voice=False, metrics=metrics)
    setup = _setup_result()
    if withheld:
        setup["diagnostics"] = {"compressor_withheld": _WITHHELD_COMPRESSOR}
    sample_rate = 48_000
    noise = np.full(sample_rate * 2, 0.001, dtype=np.float32)
    speech = np.full(sample_rate * 3, 0.04, dtype=np.float32)
    return voice_setup.validate_voice_setup_verification(
        noise, speech, speech, sample_rate, setup, "flat",
        raw_noise_audio=noise, raw_verification_speech_audio=speech,
    )


@pytest.mark.parametrize(
    ("metrics", "kept", "code"),
    [
        ({"compressor_gain_reduction_p95_db": 6.0}, "reduce", "compressor_excess"),
        ({"compressor_gain_reduction_db": 12.0}, "rollback", "compressor_peak_cap"),
    ],
    ids=["p95-over-target", "peak-over-cap"],
)
def test_verification_leaves_the_compressor_targets_out_when_the_compressor_is_withheld(
    monkeypatch, metrics, kept, code
):
    set_by_us = _verify_with_compressor_metrics(monkeypatch, metrics, withheld=False)
    withheld = _verify_with_compressor_metrics(monkeypatch, metrics, withheld=True)

    assert set_by_us["decision"] == kept and code in set_by_us["reason_codes"]
    assert withheld["decision"] == "accept"
    assert withheld["reduction_targets"] == [] and code not in withheld["reason_codes"]


def test_verification_still_answers_limiter_pressure_when_the_compressor_is_withheld(monkeypatch):
    outcome = _verify_with_compressor_metrics(monkeypatch, {"limiter_gain_reduction_db": 2.0}, withheld=True)

    assert outcome["decision"] == "reduce"
    assert outcome["reduction_targets"] == ["limiter"]
