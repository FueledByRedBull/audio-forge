"""Bit-exact pins for ``mic_eq.analysis.spectrum``.

Every public result of the module is reduced to a SHA-256 digest of its float64
bytes and compared with digests recorded from the implementation before it was
optimised. Any change to a single bit of any output, including a different
floating-point summation order, fails the matching case.

The digests depend on the exact numpy build and CPU math library. ``_CANARY``
records a few elementary numpy results; when it differs the pinned digests say
nothing about the code, so the pinned cases are skipped instead of failing.
"""

import dataclasses
import hashlib
import struct
from typing import Any

import numpy as np
import pytest

from mic_eq.analysis import spectrum

FS = 48_000


def _speech(seed, seconds=4.0, fs=FS, noise_db=-55.0):
    """Speech-like signal built from numpy primitives only."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    out = np.zeros(n)
    pos = int(0.2 * fs)
    while pos < n - int(0.1 * fs):
        seg = min(int(rng.uniform(0.12, 0.35) * fs), n - pos)
        f = np.fft.rfftfreq(seg, 1.0 / fs)
        if rng.random() < 0.6:
            t = np.arange(seg) / fs
            f0 = rng.uniform(95, 190) * (1.0 + 0.08 * np.sin(2 * np.pi * rng.uniform(2, 6) * t))
            pulses = (np.diff(np.floor(np.cumsum(f0) / fs), prepend=0.0) > 0).astype(float)
            env = sum(
                gain / (1.0 + ((f - centre) / width) ** 2)
                for centre, width, gain in (
                    (rng.uniform(300, 800), 120, 1.0),
                    (rng.uniform(900, 2400), 220, 0.5),
                    (rng.uniform(2400, 3200), 300, 0.3),
                )
            )
            sig = np.fft.irfft(np.fft.rfft(pulses) * env, n=seg)
        else:
            env = 1.0 / (1.0 + ((f - 6500) / 1800) ** 2)
            sig = 0.4 * np.fft.irfft(np.fft.rfft(rng.standard_normal(seg)) * env, n=seg)
        sig *= np.hanning(seg)
        out[pos:pos + seg] += sig / (np.max(np.abs(sig)) + 1e-12) * rng.uniform(0.08, 0.4)
        pos += seg + int(rng.uniform(0.02, 0.45) * fs)
    return out + _room_noise(seed + 1000, seconds, fs, noise_db)


def _room_noise(seed, seconds, fs=FS, level_db=-55.0):
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    noise = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) / np.sqrt(np.maximum(f, 20.0)), n=n)
    return noise * 10 ** (level_db / 20) / np.std(noise)


def _posteriors(audio, fs=FS):
    """Deterministic stand-in for Silero posteriors (one per 512-sample@16 kHz window)."""
    window = int(np.ceil(fs * 512 / 16_000))
    count = audio.size // window
    rms = np.sqrt(np.mean(audio[:count * window].reshape(count, window) ** 2, axis=1) + 1e-14)
    return np.clip((20 * np.log10(rms) + 50.0) / 12.0, 0.0, 1.0)


def _burst_in_quiet():
    audio = _room_noise(7, 6.0, level_db=-60.0)
    burst = _speech(8, 1.2, noise_db=-90.0)
    audio[100_000:100_000 + burst.size] += burst
    return audio


def _grid(kind):
    if kind == "fft48k":
        return np.fft.rfftfreq(4096, 1.0 / FS)
    if kind == "fft44k":
        return np.fft.rfftfreq(2048, 1.0 / 44_100)
    if kind == "geometric":
        return np.geomspace(5.0, 22_000.0, 700)
    return np.fft.rfftfreq(4096, 1.0 / FS)[::-1].copy()  # "reversed"


def _curve(grid, seed=3):
    rng = np.random.default_rng(seed)
    return 6.0 * np.sin(np.log2(grid + 1.0)) - 40.0 + 3.0 * rng.standard_normal(grid.size)


def _noise_reference():
    freqs = np.linspace(0.0, 24_000.0, 513)
    return freqs, -70.0 - 8.0 * np.log2(1.0 + freqs / 200.0)


def _vad_flat(audio, level):
    return np.full(audio.size // 1536, level)


# Each case builds its own inputs, so collecting the names generates no audio.
_ANALYZE_CASES = {
    "voiced_vad_explicit_noise": lambda: (
        (_speech(11), FS),
        dict(vad_probabilities=_posteriors(_speech(11)), noise_audio=_room_noise(21, 2.0)),
    ),
    "voiced_energy_only_in_capture_noise": lambda: ((_speech(12), FS), {}),
    "override_with_source": lambda: (
        (_speech(13), FS),
        dict(
            vad_probabilities=_posteriors(_speech(13)),
            noise_spectrum_override=_noise_reference(),
            noise_reference_source_override="custom_reference",
        ),
    ),
    "override_default_source": lambda: (
        (_speech(13), FS),
        dict(noise_spectrum_override=_noise_reference()),
    ),
    "override_invalid_uses_noise_audio": lambda: (
        (_speech(13), FS),
        dict(
            noise_spectrum_override=(_noise_reference()[0], _noise_reference()[1][:-1]),
            noise_audio=_room_noise(22, 2.0),
        ),
    ),
    "unvoiced_only": lambda: ((0.3 * _room_noise(31, 3.0, level_db=-30.0), FS), {}),
    "silence": lambda: ((np.zeros(2 * FS), FS), {}),
    "short_one_frame": lambda: ((_speech(14)[:4096], FS), {}),
    "short_fallback_explicit_noise": lambda: (
        (_speech(14)[20_000:29_000], FS),
        dict(noise_audio=_room_noise(23, 1.0)),
    ),
    "sparse_burst_fallback": lambda: ((_burst_in_quiet(), FS), {}),
    "noisy": lambda: (
        (_speech(16, 4.0, noise_db=-28.0), FS),
        dict(noise_audio=_room_noise(24, 2.0, level_db=-28.0)),
    ),
    "fs44100_nperseg2048": lambda: (
        (_speech(15, 4.0, 44_100), 44_100, 2048),
        dict(
            vad_probabilities=_posteriors(_speech(15, 4.0, 44_100), 44_100),
            noise_audio=_room_noise(25, 2.0, 44_100),
        ),
    ),
    "nperseg8192": lambda: (
        (_speech(11), FS, 8192),
        dict(vad_probabilities=_posteriors(_speech(11))),
    ),
    "vad_all_low_uses_energy": lambda: (
        (_speech(12), FS),
        dict(vad_probabilities=_vad_flat(_speech(12), 0.05)),
    ),
    "vad_all_high": lambda: (
        (_speech(12), FS),
        dict(vad_probabilities=_vad_flat(_speech(12), 0.95)),
    ),
}


def _fingerprint_value(value):
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        payload = repr((array.shape, str(array.dtype))).encode() + array.tobytes()
    elif isinstance(value, float):
        payload = struct.pack("<d", value)
    else:
        payload = repr(value).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _field_digests(outputs):
    return {name: _fingerprint_value(value) for name, value in sorted(outputs.items())}


def _fingerprint(outputs):
    joined = "|".join(f"{name}={digest}" for name, digest in _field_digests(outputs).items())
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def _result_fields(result):
    return {f.name: getattr(result, f.name) for f in dataclasses.fields(result)}


def _run_analyze(name):
    args, kwargs = _ANALYZE_CASES[name]()
    return _result_fields(spectrum.analyze_voice_spectrum(*args, **kwargs))


def _run_compute(name):
    audio, fs, nperseg = {
        "voiced": (_speech(41), FS, 4096),
        "fs44100_nperseg2048": (_speech(42, 4.0, 44_100), 44_100, 2048),
        "quiet_flat": (_room_noise(43, 2.0, level_db=-70.0), FS, 4096),
    }[name]
    freqs, spectrum_db = spectrum.compute_voice_spectrum(audio, fs, nperseg)
    return {"freqs": freqs, "spectrum_db": spectrum_db}


def _run_octave_frequencies(name):
    args, kwargs = {
        "default": ((), {}),
        "third": ((3,), {}),
        "twelfth": ((12,), {}),
        "whole": ((1,), {}),
        "odd_fifth_voice_band": ((5, (100, 8000)), {}),
        "twenty_fourth_ref440": ((24, (20, 20_000), 440.0), {}),
    }[name]
    centre, lower, upper = spectrum.get_octave_frequencies(*args, **kwargs)
    return {"centre": centre, "lower": lower, "upper": upper}


def _run_smooth(name):
    kind, grid_kind, option = name.split(":")
    grid = _grid(grid_kind)
    values = _curve(grid)
    if kind == "octave":
        return {"out": spectrum.smooth_spectrum_octave(grid, values, int(option))}
    strength: Any = None if option == "None" else option
    return {"out": spectrum.smooth_spectrum_perceptual(grid, values, strength)}


def _run_peaks(name):
    seed, fraction = {"third": (51, 3), "sixth": (52, 6)}[name]
    freqs, spectrum_db = spectrum.compute_voice_spectrum(_speech(seed), FS)
    peak_freqs, peak_values = spectrum.find_octave_spaced_peaks(spectrum_db, freqs, fraction)
    return {"peak_freqs": peak_freqs, "peak_values": peak_values}


_LITERAL_FLOATS = (
    "snr_db",
    "residual_confidence",
    "measurement_coverage",
    "voiced_window_ratio",
    "vad_active_window_ratio",
)

RUNNERS = {
    **{f"analyze:{n}": (lambda n=n: _run_analyze(n)) for n in _ANALYZE_CASES},
    **{f"compute:{n}": (lambda n=n: _run_compute(n)) for n in ("voiced", "fs44100_nperseg2048", "quiet_flat")},
    **{
        f"octave_frequencies:{n}": (lambda n=n: _run_octave_frequencies(n))
        for n in ("default", "third", "twelfth", "whole", "odd_fifth_voice_band", "twenty_fourth_ref440")
    },
    **{
        f"smooth:octave:{g}:{f}": (lambda n=f"octave:{g}:{f}": _run_smooth(n))
        for g in ("fft48k", "fft44k", "geometric", "reversed")
        for f in (2, 3, 6, 12)
    },
    **{
        f"smooth:perceptual:{g}:{s}": (lambda n=f"perceptual:{g}:{s}": _run_smooth(n))
        for g in ("fft48k", "fft44k", "geometric", "reversed")
        for s in ("balanced", "conservative", "broad", "off", "None")
    },
    **{f"peaks:{n}": (lambda n=n: _run_peaks(n)) for n in ("third", "sixth")},
}


def _canary():
    ramp = np.linspace(-120.0, 3.0, 4097)
    probe = np.sin(ramp) * 40.0 - 50.0
    payload = b"".join(
        np.ascontiguousarray(part).tobytes()
        for part in (
            10.0 * np.log10(np.abs(probe) + 1e-12),
            10 ** (probe / 10),
            np.median(probe.reshape(-1, 17)[:, :16], axis=0),
            np.abs(np.fft.rfft(probe * np.hamming(probe.size))) ** 2,
            _speech(41, 0.5),
        )
    )
    return hashlib.sha256(payload).hexdigest()[:16]


# Recorded from the unoptimised implementation (numpy 2.5.1, Windows x64).
_CANARY = "90b3699f4f261d5c"
EXPECTED = {'analyze:fs44100_nperseg2048': '4d4dd3ac35ecaa95',
 'analyze:noisy': '12a762542b416c43',
 'analyze:nperseg8192': '21acabfaf239cfd9',
 'analyze:override_default_source': 'ae9ed7da4c459dd4',
 'analyze:override_invalid_uses_noise_audio': 'cc5d54fc426dd2bc',
 'analyze:override_with_source': '596a144f33404091',
 'analyze:short_fallback_explicit_noise': 'f92fc14b1ac860f9',
 'analyze:short_one_frame': '93f434118fe0c7df',
 'analyze:silence': '103af2c134396982',
 'analyze:sparse_burst_fallback': '2ba990f952995f93',
 'analyze:unvoiced_only': 'd1e916cd6cd785e1',
 'analyze:vad_all_high': 'a178fede1bf7d782',
 'analyze:vad_all_low_uses_energy': '2cc6b885f5488c07',
 'analyze:voiced_energy_only_in_capture_noise': 'd484763d1b693c2a',
 'analyze:voiced_vad_explicit_noise': 'bec743a18b0bfb2f',
 'compute:fs44100_nperseg2048': '0717844b665745f2',
 'compute:quiet_flat': 'b7ace57ee90955a5',
 'compute:voiced': '2383fa13e72d6cde',
 'octave_frequencies:default': '7c6f073153f8bcd6',
 'octave_frequencies:odd_fifth_voice_band': '384e921c97aa8cc2',
 'octave_frequencies:third': '2249d7e91165cf9e',
 'octave_frequencies:twelfth': '5d350c06c3543ee2',
 'octave_frequencies:twenty_fourth_ref440': '5fee47cf7258a5f1',
 'octave_frequencies:whole': '0fe4072d50f30de6',
 'peaks:sixth': '8c7959240aa397fa',
 'peaks:third': '66bfd13e7292695c',
 'smooth:octave:fft44k:12': '7ee33819f9ee51ae',
 'smooth:octave:fft44k:2': '45e3da59e56ea49e',
 'smooth:octave:fft44k:3': '595a2886b55d87de',
 'smooth:octave:fft44k:6': '317144a4372109c9',
 'smooth:octave:fft48k:12': '6599a64437868921',
 'smooth:octave:fft48k:2': '99fd29e6def92f92',
 'smooth:octave:fft48k:3': '764c85391ecf3582',
 'smooth:octave:fft48k:6': '86ca6bbf7a3f030c',
 'smooth:octave:geometric:12': '4fa185fcda5fcc5b',
 'smooth:octave:geometric:2': 'c7e5df7d14916692',
 'smooth:octave:geometric:3': '72db3a88ab41a677',
 'smooth:octave:geometric:6': '467f07489554cca6',
 'smooth:octave:reversed:12': '84992e5496bbc9c5',
 'smooth:octave:reversed:2': '7206ad3b4fefaa84',
 'smooth:octave:reversed:3': 'e1dfee76689c6b66',
 'smooth:octave:reversed:6': '4d3ab71b47911cb0',
 'smooth:perceptual:fft44k:None': '922f26b37bbe40af',
 'smooth:perceptual:fft44k:balanced': '922f26b37bbe40af',
 'smooth:perceptual:fft44k:broad': '253449be079cb976',
 'smooth:perceptual:fft44k:conservative': 'acfe88e16fd66e43',
 'smooth:perceptual:fft44k:off': '30adbffd60bdfdff',
 'smooth:perceptual:fft48k:None': '84872776722ce724',
 'smooth:perceptual:fft48k:balanced': '84872776722ce724',
 'smooth:perceptual:fft48k:broad': '4a79fc22bc3c3dcc',
 'smooth:perceptual:fft48k:conservative': '08315e30cb06fc06',
 'smooth:perceptual:fft48k:off': 'ee4735bd3e939c14',
 'smooth:perceptual:geometric:None': 'a7d5bef93f43888d',
 'smooth:perceptual:geometric:balanced': 'a7d5bef93f43888d',
 'smooth:perceptual:geometric:broad': '95a697a2fc8801d9',
 'smooth:perceptual:geometric:conservative': '3c6d05a1de2578ce',
 'smooth:perceptual:geometric:off': 'c951c50ca4f03ab3',
 'smooth:perceptual:reversed:None': '45e98b27f40ea8bf',
 'smooth:perceptual:reversed:balanced': '45e98b27f40ea8bf',
 'smooth:perceptual:reversed:broad': '15a02a0735ec2d6d',
 'smooth:perceptual:reversed:conservative': 'd9beaaef56321b06',
 'smooth:perceptual:reversed:off': 'cab51fb80be33f62'}
LITERALS = {'fs44100_nperseg2048': {'fallback': False,
                         'measurement_coverage': 0.7233222909104376,
                         'residual_confidence': 0.5729147670769094,
                         'snr_db': 14.200718116571878,
                         'source': 'explicit_capture',
                         'vad_active_window_ratio': 0.39766081871345027,
                         'voiced_window_ratio': 0.39766081871345027,
                         'windows': 68},
 'noisy': {'fallback': False,
           'measurement_coverage': 0.8270581351991315,
           'residual_confidence': 0.7240405066203934,
           'snr_db': -12.105026641359618,
           'source': 'explicit_capture',
           'vad_active_window_ratio': 0.0,
           'voiced_window_ratio': 1.0,
           'windows': 92},
 'nperseg8192': {'fallback': False,
                 'measurement_coverage': 0.45169161695021515,
                 'residual_confidence': 0.27975615575257273,
                 'snr_db': 15.842374359261385,
                 'source': 'in_capture_non_speech',
                 'vad_active_window_ratio': 0.4222222222222222,
                 'voiced_window_ratio': 0.4222222222222222,
                 'windows': 19},
 'override_default_source': {'fallback': False,
                             'measurement_coverage': 0.7552335822057573,
                             'residual_confidence': 0.8349317533912198,
                             'snr_db': 60.92772414312617,
                             'source': 'validated_conservative',
                             'vad_active_window_ratio': 0.0,
                             'voiced_window_ratio': 0.43478260869565216,
                             'windows': 40},
 'override_invalid_uses_noise_audio': {'fallback': False,
                                       'measurement_coverage': 0.7552335822057573,
                                       'residual_confidence': 0.8349317533912198,
                                       'snr_db': 24.728341700752164,
                                       'source': 'explicit_capture',
                                       'vad_active_window_ratio': 0.0,
                                       'voiced_window_ratio': 0.43478260869565216,
                                       'windows': 40},
 'override_with_source': {'fallback': False,
                          'measurement_coverage': 0.79084626709282,
                          'residual_confidence': 0.8716485058011376,
                          'snr_db': 59.37118877551471,
                          'source': 'custom_reference',
                          'vad_active_window_ratio': 0.5,
                          'voiced_window_ratio': 0.5,
                          'windows': 46},
 'short_fallback_explicit_noise': {'fallback': True,
                                   'measurement_coverage': 0.45,
                                   'residual_confidence': 0.0,
                                   'snr_db': 22.659463845341335,
                                   'source': 'explicit_capture',
                                   'vad_active_window_ratio': 0.0,
                                   'voiced_window_ratio': 0.3333333333333333,
                                   'windows': 1},
 'short_one_frame': {'fallback': True,
                     'measurement_coverage': 0.45,
                     'residual_confidence': 0.0,
                     'snr_db': 0.0,
                     'source': 'unavailable',
                     'vad_active_window_ratio': 0.0,
                     'voiced_window_ratio': 1.0,
                     'windows': 1},
 'silence': {'fallback': False,
             'measurement_coverage': 0.4666666666666667,
             'residual_confidence': 0.47803803021118385,
             'snr_db': 0.0,
             'source': 'unavailable',
             'vad_active_window_ratio': 0.0,
             'voiced_window_ratio': 1.0,
             'windows': 45},
 'sparse_burst_fallback': {'fallback': True,
                           'measurement_coverage': 0.45,
                           'residual_confidence': 0.0,
                           'snr_db': 15.32626766603955,
                           'source': 'in_capture_non_speech',
                           'vad_active_window_ratio': 0.0,
                           'voiced_window_ratio': 0.07194244604316546,
                           'windows': 1},
 'unvoiced_only': {'fallback': False,
                   'measurement_coverage': 0.848631851347754,
                   'residual_confidence': 0.6735105739858569,
                   'snr_db': 0.0,
                   'source': 'unavailable',
                   'vad_active_window_ratio': 0.0,
                   'voiced_window_ratio': 1.0,
                   'windows': 69},
 'vad_all_high': {'fallback': False,
                  'measurement_coverage': 0.7945652173913043,
                  'residual_confidence': 0.6640489130434782,
                  'snr_db': 0.0,
                  'source': 'unavailable',
                  'vad_active_window_ratio': 1.0,
                  'voiced_window_ratio': 1.0,
                  'windows': 92},
 'vad_all_low_uses_energy': {'fallback': False,
                             'measurement_coverage': 0.6351194308276166,
                             'residual_confidence': 0.6108323903361736,
                             'snr_db': 17.93001789506335,
                             'source': 'in_capture_non_speech',
                             'vad_active_window_ratio': 0.0,
                             'voiced_window_ratio': 0.31521739130434784,
                             'windows': 29},
 'voiced_energy_only_in_capture_noise': {'fallback': False,
                                         'measurement_coverage': 0.6351194308276166,
                                         'residual_confidence': 0.6108323903361736,
                                         'snr_db': 17.93001789506335,
                                         'source': 'in_capture_non_speech',
                                         'vad_active_window_ratio': 0.0,
                                         'voiced_window_ratio': 0.31521739130434784,
                                         'windows': 29},
 'voiced_vad_explicit_noise': {'fallback': False,
                               'measurement_coverage': 0.7902627340592807,
                               'residual_confidence': 0.385690922758039,
                               'snr_db': 15.160972012359453,
                               'source': 'explicit_capture',
                               'vad_active_window_ratio': 0.40217391304347827,
                               'voiced_window_ratio': 0.40217391304347827,
                               'windows': 37}}


@pytest.fixture(scope="module")
def pinned_platform():
    if _canary() != _CANARY:
        pytest.skip("numpy build or CPU math differs from the one the digests were recorded on")


def _literals(fields):
    return {
        "fallback": fields["used_single_spectrum_fallback"],
        "source": fields["noise_reference_source"],
        "windows": fields["window_spectra_db"].shape[0],
        **{key: fields[key] for key in _LITERAL_FLOATS},
    }


@pytest.mark.parametrize("name", sorted(RUNNERS))
def test_outputs_are_bit_identical_to_recorded_digests(pinned_platform, name):
    outputs = RUNNERS[name]()
    # The per-field digests only make the failure readable.
    assert _fingerprint(outputs) == EXPECTED[name], _field_digests(outputs)
    if name.startswith("analyze:"):
        assert _literals(outputs) == LITERALS[name.removeprefix("analyze:")]


def test_short_audio_is_rejected():
    with pytest.raises(ValueError, match="Audio too short for FFT"):
        spectrum.analyze_voice_spectrum(np.zeros(4095), FS)
    with pytest.raises(ValueError, match="Audio too short for FFT"):
        spectrum.compute_voice_spectrum(np.zeros(4095), FS)


@pytest.mark.parametrize("grid_kind", ["fft48k", "geometric"])
def test_returned_arrays_can_be_modified_without_affecting_later_calls(grid_kind):
    grid = _grid(grid_kind)
    values = _curve(grid)
    for call in (
        lambda: spectrum.get_octave_frequencies(6),
        lambda: (spectrum.smooth_spectrum_octave(grid, values, 6),),
        lambda: (spectrum.smooth_spectrum_perceptual(grid, values, "conservative"),),
    ):
        first = [np.array(part) for part in call()]
        for part in call():
            part[...] = -1.0
        assert all(np.array_equal(a, b) for a, b in zip(first, call()))
    assert np.array_equal(values, _curve(grid))
