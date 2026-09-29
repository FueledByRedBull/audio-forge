"""EQ response prediction for Auto-EQ optimization."""

from __future__ import annotations

import numpy as np

from .constants import SAMPLE_RATE

FILTER_PEAK = "peak"
FILTER_LOW_SHELF = "low_shelf"
FILTER_HIGH_SHELF = "high_shelf"
FILTER_NOTCH = "notch"
FILTER_HIGH_PASS = "high_pass"
FILTER_LOW_PASS = "low_pass"
SUPPORTED_FILTER_TYPES = frozenset(
    {
        FILTER_PEAK,
        "bell",
        "peaking",
        FILTER_LOW_SHELF,
        FILTER_HIGH_SHELF,
        FILTER_NOTCH,
        FILTER_HIGH_PASS,
        FILTER_LOW_PASS,
    }
)
SUPPORTED_PASS_SLOPES_DB_PER_OCTAVE = (12, 24, 36, 48)


def _normalize_filter_type(filter_type: str) -> str:
    if filter_type in {"bell", "peaking"}:
        return FILTER_PEAK
    if filter_type not in SUPPORTED_FILTER_TYPES:
        raise ValueError(f"unsupported EQ filter type: {filter_type}")
    return filter_type


def _is_gain_filter(filter_type: str) -> bool:
    return _normalize_filter_type(filter_type) in {
        FILTER_PEAK,
        FILTER_LOW_SHELF,
        FILTER_HIGH_SHELF,
    }


def _default_filter_types(num_bands: int) -> list[str]:
    if num_bands <= 0:
        return []
    return [
        FILTER_LOW_SHELF if index == 0 else FILTER_HIGH_SHELF if index == num_bands - 1 else FILTER_PEAK
        for index in range(num_bands)
    ]


def _biquad_coefficients(
    gain_db: float,
    q: float,
    fc: float,
    filter_type: str,
    sample_rate: float = SAMPLE_RATE,
) -> tuple[float, float, float, float, float, float]:
    """Return RBJ biquad coefficients with un-normalized a0."""
    amplitude = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / sample_rate
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)

    filter_type = _normalize_filter_type(filter_type)
    if filter_type == FILTER_LOW_SHELF:
        two_sqrt_a_alpha = 2.0 * np.sqrt(amplitude) * alpha
        b0 = amplitude * ((amplitude + 1.0) - (amplitude - 1.0) * cos_w0 + two_sqrt_a_alpha)
        b1 = 2.0 * amplitude * ((amplitude - 1.0) - (amplitude + 1.0) * cos_w0)
        b2 = amplitude * ((amplitude + 1.0) - (amplitude - 1.0) * cos_w0 - two_sqrt_a_alpha)
        a0 = (amplitude + 1.0) + (amplitude - 1.0) * cos_w0 + two_sqrt_a_alpha
        a1 = -2.0 * ((amplitude - 1.0) + (amplitude + 1.0) * cos_w0)
        a2 = (amplitude + 1.0) + (amplitude - 1.0) * cos_w0 - two_sqrt_a_alpha
    elif filter_type == FILTER_HIGH_SHELF:
        two_sqrt_a_alpha = 2.0 * np.sqrt(amplitude) * alpha
        b0 = amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cos_w0 + two_sqrt_a_alpha)
        b1 = -2.0 * amplitude * ((amplitude - 1.0) + (amplitude + 1.0) * cos_w0)
        b2 = amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cos_w0 - two_sqrt_a_alpha)
        a0 = (amplitude + 1.0) - (amplitude - 1.0) * cos_w0 + two_sqrt_a_alpha
        a1 = 2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cos_w0)
        a2 = (amplitude + 1.0) - (amplitude - 1.0) * cos_w0 - two_sqrt_a_alpha
    elif filter_type == FILTER_NOTCH:
        b0 = 1.0
        b1 = -2.0 * cos_w0
        b2 = 1.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
    elif filter_type == FILTER_HIGH_PASS:
        b0 = (1.0 + cos_w0) / 2.0
        b1 = -(1.0 + cos_w0)
        b2 = (1.0 + cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
    elif filter_type == FILTER_LOW_PASS:
        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
    else:
        b0 = 1.0 + alpha * amplitude
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * amplitude
        a0 = 1.0 + alpha / amplitude
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / amplitude
    return b0, b1, b2, a0, a1, a2


def _native_typed_response(
    freqs: np.ndarray,
    bands: list[tuple[str, float, float, float, int, bool]],
    sample_rate: float,
) -> np.ndarray | None:
    """Use the runtime EQ renderer when its native extension is available."""
    try:
        from mic_eq import CORE_AVAILABLE, eq_magnitude_response_v2
    except ImportError:
        return None
    # The native entry point implements the persisted 10-band EQ contract.
    # Short arbitrary filter collections use the equivalent Python calculation.
    if not CORE_AVAILABLE or len(bands) != 10:
        return None
    native_bands = [
        (
            "bell" if filter_type == FILTER_PEAK else filter_type,
            frequency_hz,
            gain_db,
            q,
            slope_db_per_octave,
            is_enabled,
        )
        for (
            filter_type,
            frequency_hz,
            gain_db,
            q,
            slope_db_per_octave,
            is_enabled,
        ) in bands
    ]
    return np.asarray(
        eq_magnitude_response_v2(freqs.tolist(), native_bands, sample_rate),
        dtype=float,
    )


def _pass_section_count(slope_db_per_octave: int) -> int:
    if slope_db_per_octave not in SUPPORTED_PASS_SLOPES_DB_PER_OCTAVE:
        raise ValueError(
            "pass filter slope must be one of "
            f"{SUPPORTED_PASS_SLOPES_DB_PER_OCTAVE} dB/octave"
        )
    return slope_db_per_octave // 12


def _predict_eq_response(
    freqs,
    gains,
    qs,
    center_freqs,
    filter_types: list[str] | tuple[str, ...] | None = None,
    *,
    slopes_db_per_octave: list[int] | tuple[int, ...] | None = None,
    enabled: list[bool] | tuple[bool, ...] | None = None,
    sample_rate: float = SAMPLE_RATE,
):
    """
    Predict how EQ settings affect frequency response.

    With no filter metadata this retains the legacy 10-band optimizer layout:
    shelves on bands zero and nine, bells in between. Typed callers can supply
    filter types, slopes, and enable flags; those use the native renderer when
    available and the same steady-state biquad sections as an offline fallback.
    The calculation does not model the live coefficient crossfade during edits.
    """
    try:
        sample_rate = float(sample_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("EQ response sample rate must be finite and positive") from exc
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("EQ response sample rate must be finite and positive")
    freqs_arr = np.asarray(freqs, dtype=float)
    if freqs_arr.ndim != 1 or not np.all(np.isfinite(freqs_arr)):
        raise ValueError("EQ response frequencies must be finite and one-dimensional")
    if np.any(freqs_arr < 0.0) or np.any(freqs_arr > sample_rate / 2.0):
        raise ValueError("EQ response frequencies must be between zero and Nyquist")
    gains_arr = np.asarray(gains, dtype=float)
    qs_arr = np.asarray(qs, dtype=float)
    centers_arr = np.asarray(center_freqs, dtype=float)
    if not (
        gains_arr.ndim == qs_arr.ndim == centers_arr.ndim == 1
        and gains_arr.size == qs_arr.size == centers_arr.size
    ):
        raise ValueError("gain, Q, and center frequency arrays must have the same length")
    if not (
        np.isfinite(gains_arr).all()
        and np.isfinite(qs_arr).all()
        and np.isfinite(centers_arr).all()
        and np.all(qs_arr > 0.0)
    ):
        raise ValueError("EQ gains, Q values, and center frequencies must be finite")
    typed = filter_types is not None
    if typed:
        filter_types = [_normalize_filter_type(str(value)) for value in filter_types]
    else:
        filter_types = _default_filter_types(gains_arr.size)
    if len(filter_types) != gains_arr.size:
        raise ValueError("filter_types length must match gains")
    if slopes_db_per_octave is None:
        slopes_db_per_octave = [12] * gains_arr.size
    if len(slopes_db_per_octave) != gains_arr.size:
        raise ValueError("slope length must match gains")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in slopes_db_per_octave
    ):
        raise ValueError("typed EQ slopes must be integers")
    slopes = [int(value) for value in slopes_db_per_octave]
    if enabled is None:
        enabled = [True] * gains_arr.size
    if len(enabled) != gains_arr.size or any(
        not isinstance(value, (bool, np.bool_)) for value in enabled
    ):
        raise ValueError("enabled flags must match gains and contain booleans")
    enabled_flags = [bool(value) for value in enabled]
    if typed:
        if np.any(centers_arr < 20.0) or np.any(
            centers_arr > sample_rate / 2.0 - 1.0
        ):
            raise ValueError("typed EQ centers must be at least 20 Hz and below Nyquist by 1 Hz")
        if np.any(gains_arr < -12.0) or np.any(gains_arr > 12.0):
            raise ValueError("typed EQ gains must be between -12 and 12 dB")
        if np.any(qs_arr < 0.1) or np.any(qs_arr > 10.0):
            raise ValueError("typed EQ Q values must be between 0.1 and 10")
        if any(slope not in SUPPORTED_PASS_SLOPES_DB_PER_OCTAVE for slope in slopes):
            raise ValueError(
                "typed EQ slopes must be one of "
                f"{SUPPORTED_PASS_SLOPES_DB_PER_OCTAVE} dB/octave"
            )
    elif np.any(centers_arr <= 0.0) or np.any(centers_arr >= sample_rate / 2.0):
        raise ValueError("legacy EQ centers must be between zero and Nyquist")

    typed_bands = [
        (
            str(filter_type),
            float(center),
            float(gain),
            float(q),
            slope,
            enabled_flag,
        )
        for filter_type, center, gain, q, slope, enabled_flag in zip(
            filter_types,
            centers_arr,
            gains_arr,
            qs_arr,
            slopes,
            enabled_flags,
            strict=True,
        )
    ]
    if typed:
        native_response = _native_typed_response(freqs_arr, typed_bands, sample_rate)
        if native_response is not None:
            return native_response

    w = 2 * np.pi * freqs_arr / sample_rate
    cos_w = np.cos(w)
    sin_w = np.sin(w)
    cos_2w = np.cos(2.0 * w)
    sin_2w = np.sin(2.0 * w)
    response_db = np.zeros_like(freqs_arr, dtype=float)

    for gain_db, q, fc, filter_type, slope, is_enabled in zip(
        gains_arr,
        qs_arr,
        centers_arr,
        filter_types,
        slopes,
        enabled_flags,
        strict=True,
    ):
        if not is_enabled:
            continue
        filter_type = _normalize_filter_type(str(filter_type))
        is_pass = filter_type in {FILTER_HIGH_PASS, FILTER_LOW_PASS}
        if not is_pass and _is_gain_filter(filter_type) and gain_db == 0.0:
            continue
        sections = _pass_section_count(slope) if is_pass else 1
        for section_index in range(sections):
            section_q = float(q)
            section_gain = float(gain_db)
            if is_pass:
                angle = (2 * section_index + 1) * np.pi / (2 * 2 * sections)
                section_q = float(1.0 / (2.0 * np.cos(angle)))
                section_gain = 0.0
            elif filter_type == FILTER_NOTCH:
                section_gain = 0.0
            b0, b1, b2, a0, a1, a2 = _biquad_coefficients(
                section_gain,
                section_q,
                float(fc),
                str(filter_type),
                sample_rate,
            )
            b0 /= a0
            b1 /= a0
            b2 /= a0
            a1 /= a0
            a2 /= a0
            numerator_real = b0 + b1 * cos_w + b2 * cos_2w
            numerator_imag = -b1 * sin_w - b2 * sin_2w
            denominator_real = 1.0 + a1 * cos_w + a2 * cos_2w
            denominator_imag = -a1 * sin_w - a2 * sin_2w
            numerator_power = numerator_real**2 + numerator_imag**2
            denominator_power = denominator_real**2 + denominator_imag**2
            magnitude = np.sqrt(numerator_power / np.maximum(denominator_power, 1.0e-30))
            response_db += 20.0 * np.log10(np.maximum(magnitude, 1.0e-10))

    return response_db
