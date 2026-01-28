"""
Audio analysis module for Auto-EQ calibration.

Provides FFT analysis, spectral smoothing, peak detection, and EQ calculation
algorithms for automatic microphone equalization.
"""
from .spectrum import (
    compute_voice_spectrum,
    get_octave_frequencies,
    smooth_spectrum_octave,
    find_octave_spaced_peaks
)
from .auto_eq import (
    get_target_curve,
    calculate_eq_bands,
    analyze_auto_eq
)

__all__ = [
    # Spectrum analysis
    'compute_voice_spectrum',
    'get_octave_frequencies',
    'smooth_spectrum_octave',
    'find_octave_spaced_peaks',
    # EQ calculation
    'get_target_curve',
    'calculate_eq_bands',
    'analyze_auto_eq'
]
