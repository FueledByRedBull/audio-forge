"""
Comprehensive pytest suite for Auto-EQ behavior.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np


# Load modules directly to avoid importing mic_eq package init side effects.
CONFIG_PATH = Path(__file__).parent.parent / "mic_eq" / "config.py"
config_spec = importlib.util.spec_from_file_location("mic_eq.config", CONFIG_PATH)
config = importlib.util.module_from_spec(config_spec)
sys.modules["mic_eq.config"] = config
config_spec.loader.exec_module(config)

AUTO_EQ_PATH = Path(__file__).parent.parent / "mic_eq" / "analysis" / "auto_eq.py"
auto_eq_spec = importlib.util.spec_from_file_location(
    "mic_eq.analysis.auto_eq", AUTO_EQ_PATH
)
auto_eq = importlib.util.module_from_spec(auto_eq_spec)
auto_eq_spec.loader.exec_module(auto_eq)

_predict_eq_response = auto_eq._predict_eq_response
calculate_eq_bands = auto_eq.calculate_eq_bands
get_target_curve = auto_eq.get_target_curve
EQ_FREQUENCIES = config.EQ_FREQUENCIES


def _seed_for_response(response_type: str) -> int:
    seeds = {
        "noise": 1701,
        "quiet": 1702,
    }
    return seeds.get(response_type, 1700)


def generate_test_spectrum(freqs, response_type="flat"):
    """Generate synthetic spectra in dBFS."""
    base_level = -70.0
    rng = np.random.default_rng(_seed_for_response(response_type))

    if response_type == "flat":
        spectrum_db = np.full_like(freqs, base_level)
    elif response_type == "bassy":
        spectrum_db = base_level + 10.0 / (1 + (freqs / 200.0) ** 2)
    elif response_type == "bright":
        spectrum_db = base_level + 10.0 * (freqs / 4000.0) ** 2 / (
            1 + (freqs / 4000.0) ** 2
        )
    elif response_type == "midscooped":
        log_freq = np.log10(freqs)
        log_center = np.log10(1500.0)
        sigma = 0.18
        spectrum_db = base_level - 8.0 * np.exp(
            -((log_freq - log_center) ** 2) / (2 * sigma**2)
        )
    elif response_type == "dark":
        spectrum_db = base_level - 10.0 / (1 + (8000.0 / freqs) ** 2)
    elif response_type == "proximity":
        spectrum_db = base_level + 15.0 / (1 + (freqs / 100.0) ** 3)
    elif response_type == "harsh":
        spectrum_db = base_level + 12.0 * np.exp(
            -((freqs - 4000.0) ** 2) / (2 * 1500.0**2)
        )
    elif response_type == "noise":
        noise_level = -50.0
        spectrum_db = base_level + (noise_level - base_level) * rng.random(len(freqs))
    elif response_type == "extreme":
        spectrum_db = base_level + 20.0 * np.sin(3 * np.log10(freqs / 100.0))
    elif response_type == "quiet":
        spectrum_db = np.full_like(freqs, -85.0) + rng.normal(0.0, 2.0, len(freqs))
    else:
        spectrum_db = np.full_like(freqs, base_level)

    return np.clip(spectrum_db, -100.0, -30.0)


def _default_freqs():
    return np.logspace(np.log10(20), np.log10(20000), 1000)


def test_01_flat_response_to_flat_target():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "flat")
    target_db = get_target_curve(freqs, "flat")
    gains = calculate_eq_bands(freqs, spectrum_db, target_db)["band_gains"]
    max_gain = max(abs(g) for g in gains)
    assert max_gain < 1.0


def test_02_bassy_mic_to_broadcast_target():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "bassy")
    target_db = get_target_curve(freqs, "broadcast")
    gains = calculate_eq_bands(freqs, spectrum_db, target_db)["band_gains"]
    assert gains[0] < -2.0 or gains[1] < -2.0


def test_03_dark_mic_to_podcast_target():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "dark")
    target_db = get_target_curve(freqs, "podcast")
    gains = calculate_eq_bands(freqs, spectrum_db, target_db)["band_gains"]
    assert gains[7] > 2.0 or gains[8] > 2.0 or gains[9] > 2.0


def test_04_midscooped_to_streaming_target():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "midscooped")
    target_db = get_target_curve(freqs, "streaming")
    gains = calculate_eq_bands(freqs, spectrum_db, target_db)["band_gains"]
    assert any(g > 2.0 for g in [gains[3], gains[4], gains[5]])


def test_05_proximity_effect_correction():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "proximity")
    target_db = get_target_curve(freqs, "broadcast")
    gains = calculate_eq_bands(freqs, spectrum_db, target_db)["band_gains"]
    assert gains[0] < -5.0


def test_06_harsh_highs_correction():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "harsh")
    target_db = get_target_curve(freqs, "podcast")
    gains = calculate_eq_bands(freqs, spectrum_db, target_db)["band_gains"]
    assert any(g < -2.0 for g in [gains[5], gains[6], gains[7]])


def test_07_noisy_signal():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "noise")
    target_db = get_target_curve(freqs, "broadcast")
    gains = calculate_eq_bands(freqs, spectrum_db, target_db)["band_gains"]
    assert all(-12.0 <= g <= 12.0 for g in gains)


def test_08_extreme_uneven_response():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "extreme")
    target_db = get_target_curve(freqs, "flat")
    gains = calculate_eq_bands(freqs, spectrum_db, target_db)["band_gains"]
    # The solver applies a correction factor, so strong responses may not hit
    # hard +/-12 dB bounds in final output. Require a large correction instead.
    assert any(abs(g) >= 7.0 for g in gains)


def test_09_very_quiet_signal():
    freqs = _default_freqs()
    spectrum_db = generate_test_spectrum(freqs, "quiet")
    target_db = get_target_curve(freqs, "broadcast")
    gains = calculate_eq_bands(freqs, spectrum_db, target_db)["band_gains"]
    assert all(-12.0 <= g <= 12.0 for g in gains)


def test_10_predict_eq_response_linearity():
    center = float(EQ_FREQUENCIES[4])
    freqs = np.array([100.0, center, 10000.0])
    qs = [1.414] * 10

    gains_6db = np.zeros(10)
    gains_6db[4] = 6.0
    response_6db = _predict_eq_response(freqs, gains_6db, qs, EQ_FREQUENCIES)

    gains_12db = np.zeros(10)
    gains_12db[4] = 12.0
    response_12db = _predict_eq_response(freqs, gains_12db, qs, EQ_FREQUENCIES)

    linear_6db = 10 ** (response_6db[1] / 20.0)
    linear_12db = 10 ** (response_12db[1] / 20.0)
    ratio = linear_12db / linear_6db

    assert 1.8 <= ratio <= 2.2
