# AudioForge v1.8.9

## Measurement and Auto-EQ

- Auto-EQ now uses the recorded room-noise capture as a matched, frequency-dependent SNR reference during Auto Voice Setup.
- Standalone captures use valid non-speech frames when available and report SNR as unavailable when no honest reference exists.
- Broad microphone/voice tilt is preserved by default instead of being silently removed.
- EQ smoothing and adjacency constraints operate on actual log-frequency spacing, and the final confidence-scaled gains are re-optimized under those constraints.
- Low-quality captures abstain from applying EQ, with explicit reasons shown in the calibration UI.

## Voice Setup

- De-esser recommendations now use time-localized, noise-supported sibilant frames, including energy-supported unvoiced consonants that speech VAD can correctly score low.
- Steady microphone brightness no longer counts as transient sibilance.
- Compressor thresholds are calibrated against the exact native Rust detector and validated through the complete offline DSP chain.
- Loudness analysis now distinguishes 400 ms momentary loudness from 3 second short-term K-weighted loudness.

## Correctness and terminology

- Analysis frames have their DC component removed consistently before spectral estimation.
- Built-in EQ targets are described as AudioForge house curves; they no longer claim BS.1770 EQ compliance.
- Fractional-octave smoothing documentation now distinguishes IEC-derived spacing equations from a certified IEC filter bank.

The release includes focused algorithm, integration, native-chain, UI-abstention, packaging, and executable startup verification.
