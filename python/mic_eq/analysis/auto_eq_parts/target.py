"""Target curve interpolation for Auto-EQ."""

import numpy as np

from mic_eq.config import EQ_FREQUENCIES, TARGET_CURVES


def _band_mean(freqs: np.ndarray, values: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return float(np.mean(values))
    return float(np.mean(values[mask]))


def _adaptive_voice_offsets(
    freqs: np.ndarray,
    measured_db: np.ndarray,
    target_preset: str,
) -> np.ndarray:
    """Derive bounded identity-preserving target offsets from measured voice balance."""
    if measured_db.size == 0:
        return np.zeros_like(freqs, dtype=float)

    body_db = _band_mean(freqs, measured_db, 180.0, 800.0)
    presence_db = _band_mean(freqs, measured_db, 1200.0, 3500.0)
    sibilance_db = _band_mean(freqs, measured_db, 5500.0, 8500.0)
    low_mid_balance = np.clip((body_db - presence_db) / 8.0, -1.0, 1.0)
    sibilance_ratio = np.clip((sibilance_db - presence_db) / 7.0, -1.0, 1.0)

    x = np.log10(np.clip(freqs, 20.0, None))
    voice_mask = (freqs >= 100.0) & (freqs <= 8000.0)
    tilt = 0.0
    if np.count_nonzero(voice_mask) >= 2:
        xv = x[voice_mask]
        yv = measured_db[voice_mask]
        xc = xv - float(np.mean(xv))
        denom = float(np.sum(xc * xc))
        if denom > 0.0:
            tilt = float(np.dot(xc, yv - float(np.mean(yv))) / denom)
    tilt_norm = np.clip(tilt / 12.0, -1.0, 1.0)

    offsets = np.zeros_like(freqs, dtype=float)
    # Keep flat closest to neutral: only gentle normalization of obvious tilt.
    if target_preset == "flat":
        offsets += np.clip(-0.60 * tilt_norm, -0.8, 0.8) * np.interp(
            freqs, [100.0, 1000.0, 8000.0], [-1.0, 0.0, 1.0]
        )
        return np.clip(offsets, -1.0, 1.0)

    # Preserve speaker identity by keeping adaptive offsets small and broad.
    warmth_offset = np.clip(-0.9 * low_mid_balance, -1.2, 1.2)
    presence_offset = np.clip(0.8 * low_mid_balance - 0.5 * tilt_norm, -1.5, 1.5)
    sibilance_offset = np.clip(-1.2 * sibilance_ratio, -1.8, 1.2)

    safe_freqs = np.clip(freqs, 20.0, None)
    offsets += warmth_offset * np.exp(-((np.log2(safe_freqs / 350.0)) ** 2) / (2 * 0.8**2))
    offsets += presence_offset * np.exp(-((np.log2(safe_freqs / 2200.0)) ** 2) / (2 * 0.9**2))
    offsets += sibilance_offset * np.exp(-((np.log2(safe_freqs / 7000.0)) ** 2) / (2 * 0.65**2))
    return np.clip(offsets, -2.0, 2.0)


def get_target_curve(freqs, target_preset='broadcast', measured_db=None):
    """
    Get target curve values at specified frequencies.

    Args:
        freqs: Frequency array (Hz)
        target_preset: Target curve name ('broadcast', 'podcast', 'streaming', 'flat')

    Returns:
        target_db: Target dB values at each frequency
    """
    if target_preset not in TARGET_CURVES:
        raise ValueError(f"Unknown target preset: {target_preset}")

    target_curve = TARGET_CURVES[target_preset]

    # Interpolate target curve to requested frequencies
    freqs = np.asarray(freqs, dtype=float)
    target_db = np.interp(
        freqs,
        EQ_FREQUENCIES,
        target_curve.band_targets,
        left=target_curve.band_targets[0],
        right=target_curve.band_targets[-1]
    )
    if measured_db is not None:
        measured_arr = np.asarray(measured_db, dtype=float)
        if measured_arr.shape == freqs.shape:
            target_db = target_db + _adaptive_voice_offsets(
                freqs,
                measured_arr,
                target_preset,
            )

    return target_db
