"""EQ response prediction for Auto-EQ optimization."""

import numpy as np

from .constants import SAMPLE_RATE

def _predict_eq_response(freqs, gains, qs, center_freqs):
    """
    Predict how EQ settings affect frequency response.

    Simulates the combined effect of all 10 parametric EQ bands.
    This accounts for band interaction - each band affects neighboring
    frequencies based on its Q factor.

    Args:
        freqs: Frequency array (Hz)
        gains: List of 10 gain values (dB)
        qs: List of 10 Q values
        center_freqs: List of 10 center frequencies (Hz)

    Returns:
        response_db: Combined EQ response in dB at each frequency
    """
    # Initialize response in linear domain (1.0 = unity gain = 0 dB)
    response_linear = np.ones_like(freqs, dtype=float)
    w = 2 * np.pi * freqs / SAMPLE_RATE
    z_inv = np.exp(-1j * w)
    z_inv_2 = z_inv * z_inv

    for gain_db, q, fc in zip(gains, qs, center_freqs):
        # Skip bands with no gain (optimization)
        if abs(gain_db) < 0.01:
            continue

        # Convert dB gain to linear amplitude
        # For biquad filters: A = 10^(dB/40)
        # gain_db = 0  -> A = 1 (no change)
        # gain_db = 6  -> A ≈ 1.995 (2x amplitude = 6dB boost)
        A = 10 ** (gain_db / 40.0)

        # Compute peaking EQ coefficients (Audio EQ Cookbook, RBJ).
        # At center frequency: magnitude response = A^2 = 10^(dB/20)
        w0 = 2 * np.pi * fc / SAMPLE_RATE
        alpha = np.sin(w0) / (2.0 * q)
        cos_w0 = np.cos(w0)

        # Biquad coefficients for peaking EQ
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A

        # Evaluate transfer function H(z) at each frequency
        # H(e^(j*w)) = (b0 + b1*e^(-j*w) + b2*e^(-j*2w)) / (a0 + a1*e^(-j*w) + a2*e^(-j*2w))

        # Numerator and denominator of transfer function
        numerator = b0 + b1 * z_inv + b2 * z_inv_2
        denominator = a0 + a1 * z_inv + a2 * z_inv_2

        # Magnitude response = |H(e^(j*w))|
        magnitude = np.abs(numerator / denominator)

        # Accumulate in linear domain (multiply responses)
        response_linear *= magnitude

    # Convert back to dB
    response_db = 20 * np.log10(np.maximum(response_linear, 1e-12))

    return response_db
