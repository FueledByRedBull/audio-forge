## v1.7.12 - 2026-03-20

- Completed DSP redesign updates with canonical compressor knee/detector semantics, sample-rate-aware limiter lookahead latency reporting, split-band de-esser recombination, and percentile-based VAD floor tracking.
- Fixed limiter lookahead peak planning to include the active output decision window and preserved gate gain smoothing during VAD force-close transitions.
- Upgraded Auto-EQ to a two-stage dense-grid optimizer (gain-only then gain+Q) with bounded Q regularization and gain-ripple penalties, and made Auto-EQ calibration follow the user-selected UI input/output devices.
- Added/updated regression coverage across gate/compressor/limiter/VAD/EQ/resampler paths and refreshed release packaging hooks for SciPy hidden-import handling.

SHA256 (`AudioForge-v1.7.12-win64-ultra.7z`):
`B2B7540829CF51123A6198BB73B9BA15030022D9E2C2A9AA8D7343A9974E7979`
