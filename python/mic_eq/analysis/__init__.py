"""
Audio analysis module for Auto-EQ calibration.

Provides FFT analysis, spectral smoothing, peak detection, EQ calculation,
failure detection, and analysis worker for automatic microphone equalization.
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
from .failure_detection import (
    validate_analysis,
    ValidationResult
)
from .noise_reference import CaptureMetadata, NoiseReferenceAnalysis, analyze_noise_reference
from .voice_setup import analyze_voice_setup, validate_voice_setup_verification

__all__ = [
    # Spectrum analysis
    'compute_voice_spectrum',
    'get_octave_frequencies',
    'smooth_spectrum_octave',
    'find_octave_spaced_peaks',
    # EQ calculation
    'get_target_curve',
    'calculate_eq_bands',
    'analyze_auto_eq',
    'analyze_voice_setup',
    'validate_voice_setup_verification',
    'CaptureMetadata',
    'NoiseReferenceAnalysis',
    'analyze_noise_reference',
    # Validation
    'validate_analysis',
    'ValidationResult'
]
