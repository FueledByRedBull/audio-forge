# AudioForge v1.10.1

This is a hardening release. It preserves the existing controls, model choices,
ten-band EQ contract, preset surface, and processing defaults while fixing
confirmed DSP, calibration, migration, evaluation, and release-pipeline defects.

## DeepFilter correctness and operating point

Partial-strength DeepFilter processing now aligns the dry signal with each
model's measured wet latency: 480 samples for Low Latency and 1440 samples for
Standard. The fixed-capacity delay is allocation-free in realtime and resets
across stream discontinuities, backend switches, and enable transitions, so a
50% wet/dry setting no longer creates latency-induced comb filtering.
When Standard is disabled or unavailable, diagnostics now report only its
actual 480-sample passthrough frame instead of model lookahead that did not run.

DeepFilter now applies an explicit internal 30 dB attenuation limit with zero
post-filter beta instead of relying on an implicit 80 dB construction value.
The selected setting was evaluated across 12 languages with paired clean/noisy
references. The report records model hashes, backend, strength, sample rate,
latency, CPU-tail timing, clean preservation, and the fact that human listening
was not run.

## Route-aware latency calibration

Calibration probes are queued on AudioForge's selected CPAL output rather than
the Windows default device. Capture and output sample rates are handled
separately, and profiles store measured route latency, current engine latency,
and their total. Engine terms refresh when suppressor, limiter, resampler, or
buffer configuration changes. Optional fixed CPAL buffers are used only when a
device accepts them; driver defaults remain the safe fallback.

## Hardening and regression evidence

- The EQ graph now uses the native Rust response calculation instead of a
  second Python RBJ implementation.
- Presets record whether values were explicit or inherited migration defaults,
  preventing future migrations from overwriting intentional choices.
- Corpus manifests are machine-portable and evaluation reports are checked for
  configuration, asset hashes, P99 timing, latency, clean-preservation gates,
  listening status, and local-path leakage.
- A deterministic full-chain golden regression covers de-essing, EQ,
  compression, limiting, true peak, finite output, and stage telemetry.
- Real multilingual speech validates auto-makeup activity/reliability fusion;
  Auto-EQ confidence thresholds were held or changed only from held-out
  evidence.
- A 192 kHz reference diagnostic found no material folded dynamics aliasing, so
  oversampling was not added.

## Evaluated changes not adopted

The current gate-before-suppressor order, de-esser-before-EQ order, 2 ms main
limiter lookahead, and Rust `nnnoiseless` backend remain unchanged. Proposed
alternatives failed predefined component gates:

- suppressor-before-gate increased chatter, pumping, and speech-tail loss;
- EQ-before-de-esser increased false reduction on bright non-sibilant cases;
- 0.5 ms and 1 ms limiter lookahead did not improve the protected output;
- current upstream Xiph RNNoise gained about 0.78 dB median noisy SI-SDR but
  worsened noisy spectral distortion and clean preservation, ran about 7.1
  times slower at P99, and implied roughly 13.6 MB of archive growth.

DPDFNet remains rejected and absent from production.

## Compatibility

Existing presets migrate to v1.10.1 without changing user-authored processing
values. No new panel, setting, model, filter type, or production dependency was
added.

## Release validation

The final Windows candidate passed 260 Python tests, 270 Rust tests with four
expected hardware-only ignores plus stress/doc tests, lint/type/static/dependency checks, a clean native and
PyInstaller build, package smoke, and actual EXE startup/model discovery. A
selected-route correlation measured 118.29 ms at 0.869 confidence. The
30-minute physical-microphone run recorded no stream restart, post-warmup
underrun, callback/drop/recovery error, RT overflow, backend failure, or
non-finite suppressor output.

The distinct published CI artifact was subsequently downloaded from the public
release and qualified without rebuilding. It passed its own package,
model-discovery and hidden-startup checks, measured 112.29 ms selected-route
correlation at 0.867 confidence, and completed a separate 30-minute
physical-microphone run with no post-warmup underrun, restart, drop, recovery,
backend, realtime-buffer, or non-finite regression. The exact archive/tree and
machine details are bound in
`evaluation/hardware-validation-v1.10.1-published.json`; use the checksum
sidecar attached to the GitHub Release for the archive identity. This remains
objective release-machine evidence, not a controlled listening claim.
