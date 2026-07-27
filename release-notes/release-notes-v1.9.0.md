# AudioForge v1.9.0

## Closed-loop Auto Voice Setup

Auto Voice Setup now treats room tone as measured evidence rather than an always-valid input. It detects short, silent, clipped, unstable, transient-heavy, speech-contaminated, stale, or route-mismatched references and either uses a conservative per-frequency estimate, limits EQ boosts, requests a recapture, or abstains.

Dynamics intensity is now independent from target loudness. Gentle, Balanced, Dense, and Custom modes calibrate active-speech compressor median and p95 gain reduction against the native Rust chain while enforcing a peak-reduction cap.

After a candidate is applied temporarily, the wizard asks for a second passage. It compares repeatability and exact downstream native-chain measurements and returns an explicit accept, reduce, retry, or rollback decision. It reports spectral target error, frequency-dependent SNR, loudness variation, gain-reduction statistics, true peak, limiter activity, clipping, and noise-floor change.

## De-esser evidence

The de-esser recommendation uses a versioned, interpretable logistic fusion of time-local high-frequency strength, temporal contrast, noise reliability, unvoiced evidence, candidate support, and peak evidence. Coefficients and the low-false-activation threshold are reproduced by a clip-grouped evaluation tool over a generated CC0 fixture corpus.

The report includes frame- and clip-level precision, recall, false positives, false negatives, PR-AUC, Brier score, and expected calibration error. These generated fixtures validate detector behavior and regression safety; they do not establish listening quality or real-speaker perceptual performance.

## Compatibility

Existing presets migrate to v1.9.0 without changing their manual compressor settings. Auto Voice Setup defaults to Balanced intensity, and custom p95/peak-cap choices persist in application configuration.
