# AudioForge v1.11.4

This patch fixes voice coloration, recovery, and analysis defects found during
a whole-repository mathematical and realtime-audio audit.

## Voice continuity and quality

- RNNoise partial-strength mixing now aligns the delayed wet frame with its
  matching dry frame, removing the confirmed comb-filter path.
- Gate/VAD and suppressor failures recover from dead workers, returned backend
  errors, non-finite output, and consumed-without-output stalls without treating
  a soft reset as a model rebuild.
- Active microphone input paired with sustained silent output now triggers
  bounded recovery while intentional mute, calibration, and expected gate
  closure remain excluded.
- Multichannel phase-safe input preserves the strongest channel, correlation is
  mean-centered, de-esser reduction respects its configured cap, and compressor
  diagnostics retain the strongest reduction within each block.

## Analysis and validation

- Spectrum smoothing `off`, logarithmic octave interpolation, partial-SNR
  confidence, headroom-derived metadata, short-capture loudness labels, and
  fallback SNR power scaling now match their documented contracts.
- Offline simulations flush and trim stateful latency, preserve exact source
  length, and include tail safety events. Processing-order evidence was
  regenerated on aligned timelines and retains the shipping order.

## Qualification

The release native extension was rebuilt and the complete Python and Rust test
suites, static checks, source-bound evaluators, and evaluation-hygiene checks
passed. Exact package identities are published only in generated checksum,
metadata, and manifest sidecars for the qualified artifact.
