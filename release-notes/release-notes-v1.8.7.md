# AudioForge v1.8.7

## Highlights

- Smoothed Silero VAD posteriors now provide continuous gain control around uncertain speech while retaining the existing realtime gate state machine and bounded RT work.
- Adaptive cleanup refines drifting 49-61 Hz hum with log-power interpolation and phase-continuity alias selection before smoothly retuning the primary and harmonic notches.
- Auto-EQ and Auto Voice Setup share native Silero posterior analysis, fuse it with an energy floor, reject localized spectral-shape outliers, and expose measurement coverage/backend diagnostics.
- Latency calibration now treats the selected output-to-input path as the measured route. It applies that route delay directly and no longer reports a fabricated one-way estimate from `round_trip / 2`.
- The Windows portable bundle is 289,235,748 bytes in the verified build, down 12,997,224 bytes (4.30%) from the prior 302,232,972-byte build through native symbol stripping and removal of unused SciPy/Qt payloads. Required DSP, models, DirectML, df.dll, and software rendering remain bundled.

## Compatibility and migration

- Existing latency profiles without `route_latency_ms` migrate to their measured route delay. The old halved `applied_compensation_ms` is not reused as the new route compensation.
- If native Silero inference is unavailable in a reduced/source environment, analysis keeps the energy-based measurement path and reports `energy_fallback` explicitly.

## Validation

- Targeted Rust tests cover monotone continuous VAD reduction and fractional drifting-hum tracking.
- Python tests cover posterior/energy fusion, robust spectral outlier rejection, route compensation, and latency-profile migration.
- Full Rust tests, release native-extension build, Clippy with warnings denied, Ruff, Pyright, and the relevant Python test suites pass on Windows.
- The rebuilt bundle passed packaging checks, bundled SciPy signal/optimization imports, and an Auto-EQ analysis smoke test using the bundled Silero model.
- The development lock refreshes Click/Semgrep; the release audit explicitly records the three MCP advisories still pinned by Semgrep's unused optional MCP server until upstream ships a compatible pin.

## Artifact

The release workflow produces `AudioForge-v1.8.7-win64-ultra.7z` and the matching `AudioForge-v1.8.7-win64-ultra.7z.sha256` checksum sidecar.
