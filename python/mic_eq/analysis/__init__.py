"""
Audio analysis module for Auto-EQ calibration.

Provides FFT analysis, spectral smoothing, peak detection, and EQ calculation
algorithms for automatic microphone equalization.
"""
from .spectrum import (
    compute_voice_spectrum,
    get_octave_frequencies,
    smooth_spectrum_octave
)

__all__ = [
    'compute_voice_spectrum',
    'get_octave_frequencies',
    'smooth_spectrum_octave'
]
