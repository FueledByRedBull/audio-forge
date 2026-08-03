# Changelog

## v1.11.2 - 2026-08-03

- Restored the one-row device, action, and health layouts at the default window
  width. The default 1280x850 window no longer needs an outer scrollbar.
- Kept the ten EQ bands in a responsive 5x2 layout at ordinary widths, with no
  horizontal scrolling or clipped controls.
- Separated level-meter ticks from their labels so the top mark reads `0`
  instead of looking like `-0`.
- Made the visible runtime-counter chip concise while retaining every detailed
  counter in its tooltip and accessibility description.
- Added regression coverage for the default viewport, minimum-size scrolling,
  meter geometry, and runtime-diagnostic presentation.

## v1.11.1 - 2026-08-03

- Clamped restored and default window geometry to one available display.
- Added responsive device, processing, status, splitter, EQ-band, and preset
  layouts. The UI no longer requires an ultra-wide or multi-monitor window.
- Removed hidden horizontal clipping from the Cleanup, Dynamics, Auto-EQ,
  Auto Voice Setup, latency, and first-run workflows.
- Sized numeric controls for their legal values and let long labels, diagnostics,
  and combo-box options wrap or elide safely.
- Applied one explicit dark application palette with automated contrast checks.
- Added layout regression coverage from 900 to 1920 pixels wide and refreshed
  the deterministic screenshot path to use the production palette.

## v1.11.0 - 2026-08-02

- Added versioned manual bell, notch, shelf, high-pass, and low-pass EQ bands.
  Added selectable 12-48 dB/octave pass slopes and exact native curve rendering.
- Added constrained mouse and keyboard graph editing with click-safe runtime
  updates and preset migration.
- Added route-bound presets using stable Windows endpoint identity,
  duplicate-name ordinals, reconnect-safe resolution, and migration provenance.
- Added a resumable first-run shell over route setup, latency calibration, and
  Auto Voice Setup.
- Added full-processing undo/redo, explainable Auto-EQ abstention, privacy-safe
  diagnostics export, accessibility contracts, and sanitized screenshots.
- Reworked release qualification around one hash-bound build, exact-artifact
  validation, hardware qualification, and promotion without rebuilding.
- Added system-UCRT enforcement, bundle manifests, archive provenance,
  44.1/48 kHz resampler gates, and native 48 kHz DeepFilter evidence.
- Revalidated processing order with reproducible objective benchmarks. Neither
  candidate passed its predefined gate, so the incumbent order remains.
- Revalidated limiter lookahead. Shorter settings did not provide the required
  material latency reduction, so the 2 ms limiter remains.
- Rejected cross-take confidence injection, sparse type-selecting Auto-EQ, and
  separate correction/tone stages after their held-out gates failed.
- Hardened evidence freshness, malformed-input handling, preset endpoint and
  backend resolution, monotonic latency timing, and setup cancellation.
- Increased output-queue reserve by 10 ms. The exact package then passed the
  strict 30-minute route test without post-warmup loss or recovery events.
- Re-ran source-bound Auto-EQ, DeepFilter, diagnostics, filter-type, and UI
  evidence. Rejected candidates stayed rejected, and DeepFilter retained its
  30 dB attenuation and zero-beta production setting.

## v1.10.1 - 2026-07-29

- Fixed DeepFilter partial-strength wet/dry comb filtering by delaying the dry path to the measured model latency (480 samples for Low Latency and 1440 for Standard), with allocation-free reset, switch, and 0/50/100% mix-contract tests. Disabled or unavailable Standard now reports only its actual one-frame passthrough latency instead of two nonexistent model-lookahead frames.
- Replaced the implicit 80 dB DeepFilter attenuation limit with an explicit internal 30 dB operating point and zero post-filter beta, selected on a 12-language paired clean/noisy corpus. Evaluation reports now record exact model/runtime configuration, assets, latency, P99 timing, clean preservation, and listening status.
- Routed latency-calibration probes through AudioForge's selected CPAL output, kept capture and output sample rates distinct, separated route/engine/total latency, refreshed profiles when engine configuration changes, and added optional fixed-buffer negotiation with safe driver-default fallback.
- Removed the duplicate Python RBJ EQ implementation; graph rendering now calls the exact native Rust filter topology, with parity coverage across bands, frequency, gain, Q, and sample rate.
- Added preset value provenance, portable corpus manifests, evaluation-contract validation, release trend records, full-chain golden regression, dynamics-aliasing diagnostics, real-speech auto-makeup evidence, and held-out Auto-EQ confidence-threshold calibration.
- Experimentally rejected suppressor-before-gate, EQ-before-de-esser, shorter limiter lookahead, and current upstream Xiph RNNoise as defaults under predefined quality, clean-preservation, runtime, and size gates. Existing processing order, 2 ms limiter lookahead, and `nnnoiseless` remain unchanged.
- Made release-asset hydration use one manifest-owned fallback tag, added CI drift detection, upgraded/pinned GitHub Actions to Node 24 revisions, and added sustained hardware, package, and executable validation gates.

## v1.10.0 - 2026-07-29

- Upgraded Silero VAD from v5.1.2 to v6.2.1 and corrected the ONNX adapter contract: each 512-sample frame is now preceded by the documented 64-sample rolling context, and the recurrent `stateN` output is required rather than optionally looked up under a wrong name. Shipped 1.9.0 builds had silently run the model with truncated input and a frozen recurrent state; the repair and the model upgrade were evaluated as independent decisions on a speaker-disjoint labeled corpus, and both passed their retention gates (held-out speech-event recall improved from roughly 62% to 96% with fewer noise-only false openings).
- Added startup model-contract validation so an incompatible VAD model fails loudly instead of silently degrading, plus Platt-scaled probability calibration fitted on the calibration split and verified on held-out captures.
- Recalibrated downstream VAD consumers against corrected inference and migrated an exact legacy gate VAD threshold from 0.4 to 0.48. Other saved thresholds are preserved; because the old format did not record value provenance, it cannot distinguish the old default from a deliberately re-entered exact 0.4.
- Replaced the compressor's RMS-only auto-makeup activity decision with a continuous fusion of the calibrated VAD posterior, VAD availability/freshness reliability, noise-floor-relative level, and noise-reference reliability, with a deterministic level-based fallback when evidence is stale or absent. Fixed a block-rate coefficient bug found during measurement: per-sample smoothing coefficients are now raised to the elapsed sample count when control updates once per block. Measured on controlled fixtures, false makeup activation on noise-only material dropped from 8.4 dB to 0 dB with unchanged speech convergence.
- Expanded Auto Voice Setup's closed-loop compressor search from threshold-only to threshold, ratio, attack, and release under a deterministic 68-simulation budget, scored by a normalized multi-objective covering loudness error, median/P95 gain reduction, peak headroom, 2-8 Hz pumping, and silence gain. Held-out median objective improved 10.44% with 83.33% of captures improving and no hard-safety regressions.
- Redesigned Auto-EQ confidence: measurement uncertainty is now estimated separately from ordinary phonetic variation, phonetic coverage is a distinct gate, confidence binds once inside solver bounds instead of through duplicated post-solve scaling and caps, constrained refinement uses symmetric bounds, and low-support bands abstain locally with a constraint-preserving re-solve instead of zeroing the whole recommendation. Post-solver gain projection now solves the nearest feasible constrained point rather than clipping.
- Evaluated DPDFNet-2 48 kHz high-resolution against both bundled DeepFilterNet3 backends on held-out noisy speech. It won every noisy-speech component gate and met CPU realtime targets but failed clean-speech preservation (dropout and spectral-distance gates), so the candidate was rejected and all production integration, bundled assets, and dependencies were removed. Evidence is retained in `evaluation/`.
- Added reproducible evaluation infrastructure: corpus builders with hashed manifests and speaker-disjoint splits, VAD/compressor/EQ/suppressor evaluation tools with machine-readable reports, a multi-speaker child-voice validation corpus, and a blind listening protocol. Corrected the DeepFilterNet standard-model latency description from ~40 ms to ~30 ms.

## v1.9.0 - 2026-07-28

- Added explicit room-noise-reference integrity analysis covering duration, finite/digital-silence/clipping failures, stationarity, RMS and octave stability, transients, speech contamination, capture age/identity, and consistency with credible non-speech voice frames.
- Propagated conservative frequency-dependent noise estimates, confidence reduction, boost restrictions, recapture guidance, and abstention through SNR, Auto-EQ, gate, de-esser, compressor, and overall setup decisions.
- Replaced brittle valid-evidence de-esser conjunctions with a versioned nonnegative logistic soft fusion trained and evaluated with clip-grouped folds on a reproducible CC0 generated fixture corpus. Reports include frame/clip precision, recall, false activation, PR-AUC, Brier score, and calibration error; generated fixtures are explicitly not perceptual validation.
- Added Gentle, Balanced, Dense, and Custom dynamics intensity independently from target loudness. Compressor calibration now fits active-frame median/p95 gain reduction in the native Rust chain and enforces an absolute peak-reduction cap.
- Added guided second-passage verification with exact deterministic native rendering, spectral target error, frequency-dependent SNR, loudness variation, compressor/de-esser reduction statistics, true peak, limiter activity, clipping, and noise-floor change. Candidates resolve to accept, reduce, retry, or rollback.
- Added capture metadata and immediate room-tone guidance, config migration/round-trip coverage, native rendering diagnostics, focused algorithm/corpus/UI-path tests, and updated release documentation.

## v1.8.9 - 2026-07-27

- Reworked Auto-EQ measurement reliability around matched room-noise spectra, frequency-dependent SNR, explicit unavailable-reference diagnostics, and safe abstention instead of spectral-percentile pseudo-SNR.
- Preserved broad microphone/voice tilt by default, kept detrending as an explicit experiment, and made gain curvature plus adjacency constraints aware of actual log-frequency spacing.
- Removed per-frame DC before spectral analysis and added regressions for DC invariance, noise-reference behavior, tilt policy, confidence abstention, and constrained optimization.
- Replaced aggregate-spectrum de-esser setup with time-local, noise-supported unvoiced/sibilant evidence aligned to the live Rust de-esser, preventing steady microphone brightness from being misclassified.
- Added exact Rust-chain compressor threshold calibration and verified recommended de-essing/compression against measured native gain reduction.
- Corrected loudness math and terminology with 400 ms momentary plus 3 second short-term K-weighted measurements, and relabelled house EQ curves without false BS.1770 or IEC conformance claims.
- Made the standalone Auto-EQ dialog refuse abstained corrections and offer a new recording instead.

## v1.8.8 - 2026-07-27

- Fixed the portable Windows bundle's SciPy runtime collection. `scipy.signal` transitively imports `scipy.integrate`, `scipy.interpolate`, and `scipy.stats`; none of these modules are excluded from the executable now.
- Added a packaging regression guard so `package_smoke.py` fails if a required SciPy runtime module is excluded again.

## v1.8.7 - 2026-07-27

- Added continuous VAD-posterior gain shaping with sample-rate-aware smoothing, preserving the existing low-latency VAD state machine while avoiding binary gate jumps around uncertain speech.
- Added phase-continuity-assisted fractional tracking for drifting 49-61 Hz hum, with log-power interpolation and bounded alias selection before notch retuning.
- Improved Auto-EQ measurement quality with Silero-posterior-aware voiced-window selection, robust RMS shape-outlier rejection, and coverage/backend diagnostics.
- Applied the same Silero posteriors to Auto Voice Setup's speech metrics and exposed an explicit energy-analysis fallback when native inference is unavailable.
- Made latency calibration route-aware: output-to-input measurements are compensated at their measured route delay, without inferring a symmetric one-way value by dividing by two.
- Added targeted DSP, measurement, migration, and latency regression tests and bumped the synchronized project version to `1.8.7`.
- Reduced the portable bundle by 12,997,224 bytes (4.30%, 302,232,972 to 289,235,748 bytes in the Windows build) by stripping native release symbols, excluding unused SciPy namespaces, and pruning unused Qt SVG payloads without removing required models, render fallback, or DSP dependencies.
- Refreshed the hash-locked development tools; the only remaining audit exception is the three upstream MCP advisories pulled by Semgrep's unused optional MCP server, documented in the release gates.

## v1.8.6 - 2026-07-10

- Replaced interpolated peak estimation with a band-limited 4x true-peak detector validated against an independent reference oversampler, while keeping the limiter path allocation-free.
- Upgraded phase-safe mono to stateful fractional-delay alignment and adaptive input cleanup to track drifting 49-61 Hz hum plus harmonics with smoothly retuned, single-topology filtering.
- Made biquad automation duration sample-rate independent and benchmarked rapid parameter morphing in release mode.
- Added SciPy golden vectors for VAD resampling at 48 kHz and 44.1 kHz, including adversarial high-frequency noise.
- Made Python-only Auto-EQ headroom simulation explicitly advisory; authoritative safety decisions now require the native Rust chain simulator.
- Upgraded Auto Voice Setup with VAD-masked BS.1770 short-term loudness, loudness range, robust speech-band features, labelled fixtures, offline chain validation, and uncertainty-aware apply behavior.
- Evaluated a DPSS multi-taper, multi-resolution spectrum estimator across speakers and microphone positions; retained Welch/Hamming because the fixture improvement did not meet the 0.75 dB materiality threshold.
- Added seeded concurrent control/DSP stress tests for atomic snapshots, dirty-flag rearming, suppressor model switching, resets, and finite bounded output in debug and release modes.
- Separated hardware-only CPAL/WASAPI enumeration smoke tests from headless CI after hosted Windows runners exposed a native output-enumeration access violation without an audio endpoint.
- Completed the PyO3 Python typing surface and resolved all Pyright errors.
- Hardened DeepFilter loading so only bootstrap-registered canonical assets are trusted by default; external DLL/model paths require `AUDIOFORGE_ALLOW_EXTERNAL_DF=1` and then take explicit precedence.
- Hardened CI and release workflows with SHA-pinned actions, least-privilege permissions, hash-locked Python dependencies, Dependabot updates, pip-audit, Semgrep SARIF, and RustSec auditing.
- Upgraded PyO3/numpy bindings to 0.29, removed unused dependency features, and cleared the local Python and Rust vulnerability audits.
- Bumped project metadata, documentation, release notes, presets, and packaging to `1.8.6`.

## v1.8.5 - 2026-07-07

- Added input-channel intelligence with selectable left/right/average/max-RMS/phase-safe mono mixdown modes and negative-correlation warnings for phase-cancellation-prone stereo inputs.
- Added safer output protection with careful limiter headroom, output clip diagnostics, and true-peak-style warning telemetry surfaced in the health UI.
- Improved gate/VAD behavior with smoother gate decisions, chatter detection, and compact health-panel warnings for rapid open/close transitions.
- Added a compact diagnostics health panel for input level, output clipping/true peak, gate chatter, suppressor backend state, callback age, and underruns.
- Improved voice dynamics with compressor sidechain high-pass detection, de-esser detector confidence, and RNNoise model-input soft clipping for hot input.
- Made Auto-EQ more controllable with explicit adaptive/static target modes, smoothing strength controls, and regularized narrow-residual correction diagnostics.
- Bumped the project metadata, docs, release notes, and packaging version to `1.8.5`.

## v1.8.4 - 2026-06-29

- Hardened config and startup recovery against valid-but-wrong persisted JSON, malformed `last_preset` values, and stale recovery cooldown timers.
- Fixed DSP and analysis correctness issues: compressor auto makeup now meters post-compression output, EQ quality boost/cut diagnostics use signed excursions correctly, and spectral-tilt fitting handles nonzero-intercept responses.
- Replaced dominant-channel multichannel input selection with deterministic mixdown to avoid channel switching on stereo and multi-input interfaces.
- Extracted output queue writing into an explicit output writer with tests for drift retiming, discontinuity fades, limiter clamping, no-op writes, and queue-full short writes.
- Added drift-retime spectral regression coverage and idle-DSP wakeup diagnostics with bounded idle backoff.
- Split more processor helpers into focused routing, resampling, diagnostics, and output-writer modules while keeping public APIs stable.
- Preserved documented MicEq compatibility boundaries while removing stale direct-test config import fallback code and obsolete source comments.
- Refined runtime diagnostics so historical output underrun/recovery totals stay visible without making the Drops chip warn forever; active/new underruns and real output short-write loss remain warning signals.
- Reduced the packaged `dist/AudioForge` footprint by pruning the duplicate top-level `mic_eq_core` native extension and making package smoke reject that duplicate.

## v1.8.3 - 2026-06-29

- Finished the public AudioForge identity cleanup by removing the superseded root build helper, renaming the icon asset, and updating stale MicEq-facing package strings.
- Made Windows-only support and DeepFilterNet activation clearer in the README, including the launcher role and release/build workflow split.
- Added desktop rotating-file logging under the AudioForge app data directory and replaced UI runtime `print()` calls with named loggers.
- Split stream recovery and device-selection handling out of the main window, and separated fast meter polling from slower diagnostics/recovery polling.

## v1.8.1 - 2026-06-08

- Completed the strict realtime migration hardening pass with bounded atomic control snapshots, dirty-flag re-arming for unstable snapshots, and regression coverage for the RT update path.
- Removed DeepFilter runtime error string allocation from the realtime failure path while preserving backend failure diagnostics.
- Surfaced fixed RT buffer overflows, callback errors, and active RT error names in the UI health strip.
- Added the Windows release packaging workflow that builds the portable archive, publishes the `.7z` and checksum, and can reuse verified runtime assets from an existing release archive.

## v1.8.0 - 2026-05-12

- Completed the EQ quality pass with confidence-weighted Auto-EQ solving, post-solve validation, target-error metrics, and visible calibration diagnostics.
- Added live EQ interaction quality checks for overlapping bands, shelf/peak stacking, narrow boosts, combined boost, and response ripple.
- Tuned dynamics behavior for a fuller EQ chain with speech-aware auto makeup, smoother adaptive compressor release, additional gate diagnostics, and DSP regression coverage.
- Hardened DeepFilter bootstrap so automatic runtime discovery only trusts application-owned paths and no longer accepts `df.dll` from the process current working directory.
- Tightened portable bundle smoke checks to require exact PyInstaller `_internal` runtime asset locations and reject misplaced decoy DLL/model/native-extension files.
- Rebuilt and republished the Windows portable bundle as `v1.8.0`.

## v1.7.18 - 2026-05-06

- Replaced fixed-grid Auto EQ placement with dynamic spectrum-driven center frequencies while preserving the existing 10-band preset/DSP shape.
- Added editable per-band EQ frequency controls and graph markers so manually tuned and Auto EQ-computed band positions are visible and adjustable in the UI.
- Split Auto EQ, config catalogs, app startup helpers, and large Rust processor/VAD/DeepFilter test/API sections into focused modules while preserving public imports and runtime behavior.
- Fixed packaged startup for split config catalog modules under PyInstaller, rebuilt the portable archive, and published the `v1.7.18` release asset.

## v1.7.17 - 2026-04-30

- Moved DeepFilter backend construction and VAD inference out of realtime processing paths; VAD modes now consume cached worker probabilities and degrade deterministically when unavailable.
- Fixed muted/recording output draining, raw-recording duration validation, startup preset restore IDs, full compressor preset loading, calibration-owned processor cleanup, and corrupt config boolean/geometry handling.
- Added release asset hash verification, exact bundle smoke checks, dependency metadata retention, native-extension staleness checks, and tighter version/package test coverage.

## v1.7.16 - 2026-04-27

- Reworked the DSP safety path with transparent limiter clamping and non-overshooting linear drift retiming.
- Replaced raw EQ coefficient interpolation with dual-filter output crossfades so live EQ changes avoid unstable intermediate IIR states.
- Updated dynamics behavior with RMS gate detection plus hold/hysteresis, linear-domain compressor peak/RMS detection, depth-aware adaptive compressor release, and dynamic-EQ de-essing that avoids phase-shifted sidechain subtraction.

## v1.7.15 - 2026-04-27

- Hardened realtime audio against suppressor lock contention, dirty-control update loss, non-finite DSP parameters, post-retime limiter overshoot, and input callback allocations.
- Tightened preset/config validation for non-finite numbers, strict booleans, atomic writes, external preset imports, Auto-EQ center-frequency persistence, and calibration-owned processor cleanup.
- Strengthened release quality gates with DirectML packaging, full-feature asset checks, package smoke tests, version checks, Pyright/Ruff/Clippy/pytest CI, and a valid optimized Windows icon.

## v1.7.14 - 2026-03-27

- Fixed the VAD startup path to report a neutral probability before the first inference window instead of forcing an initial false negative, and made compressor loudness reset fail coherently instead of silently leaving stale meter history behind.
- Switched the VAD gate experience to noise-floor-driven auto threshold by default, aligned the Rust/Python defaults, and surfaced the live noise floor plus effective threshold directly in the gate UI.
- Kept the VAD probability threshold control active in VAD modes while auto thresholding is on, so speech sensitivity and level thresholding are no longer conflated.

## v1.7.13 - 2026-03-27

- Split the latest work into a release-corrected build: Auto-EQ now applies spectral tilt removal, voiced-frame spectrum selection, SNR-aware boost caps, adjacent-band coupling limits, and bounded center-frequency nudging.
- Hardened the real-time transport path with safer resampler buffering, output recovery accounting, EQ parameter validation, raw-monitor handling, limiter state resets, and corrected RNNoise frame smoothing.
- Kept the RT path non-blocking on control-lock contention, switched limiter lookahead peak tracking to an amortized O(1) window max, and corrected compressor release-meter rounding/storage.
- Stabilized persisted device identities and latency-profile migration, refreshed the README/release flow, and repackaged the Windows archive under the new release version.

## v1.7.12 - 2026-03-20

- Completed DSP redesign updates with canonical compressor knee/detector semantics, sample-rate-aware limiter lookahead latency reporting, split-band de-esser recombination, and percentile-based VAD floor tracking.
- Fixed limiter lookahead peak planning to include the active output decision window and preserved gate gain smoothing during VAD force-close transitions.
- Upgraded Auto-EQ to a two-stage dense-grid optimizer (gain-only then gain+Q) with bounded Q regularization and gain-ripple penalties, and made Auto-EQ calibration follow the user-selected UI input/output devices.
- Added/updated regression coverage across gate/compressor/limiter/VAD/EQ/resampler paths and refreshed release packaging hooks for SciPy hidden-import handling.

## v1.7.11 - 2026-03-19

- Preferred native `48 kHz` input configs when available and made required input/output resampler setup fail fast at startup instead of falling through to wrong-rate processing.
- Reworked the real-time reliability path with proactive input backlog shedding, gentler output catch-up, shorter underrun tails, and deferred gate/suppressor control updates so hot audio blocks stop depending on `try_lock()` fallbacks.
- Standardized calibration and analysis sample-rate handling around the runtime processor rate and surfaced new backlog, clip, and resampler diagnostics in the main window.

## v1.7.10 - 2026-03-15

- Fixed the packaged Windows build startup failure caused by excluding and pruning SciPy's `_highspy` runtime payload.
- Restored the required SciPy optimize module in `AudioForge.spec` and stopped the bundle-pruning step from deleting it.
- Rebuilt the portable EXE and verified that `AudioForge.exe` starts normally from `dist/AudioForge`.

## v1.7.9 - 2026-03-13

- Fixed the main-window dark-theme regression by removing the broad forced-light styling and limiting custom styling to explicit action buttons and health chips.
- Rebalanced the splitter layout for the tabbed control column and EQ pane, with clamped persisted sizes and wider pane floors so labels and EQ controls stop clipping on 1366-wide displays.
- Polished action-row spacing and tab-page margins to make the reworked `Cleanup` and `Dynamics` views read cleanly without changing DSP behavior.

## v1.7.8 - 2026-03-13

- Removed steady-state VAD buffer draining and per-window scratch allocation by switching Silero VAD to reusable cursor-based buffers.
- Hard-gated audio-adjacent debug logging in the VAD, gate, and processor paths so release builds stop printing from those hot sections.
- Pruned the unused bundled `scipy.optimize._highspy` payload to trim the Windows package further without dropping features.

## v1.7.7 - 2026-03-09

- Hardened the headless self-test by using expected playback windows, a normalized correlation score, and a wider retry delay ladder while keeping the `0.25` confidence threshold unchanged.
- Removed steady-state `try_lock()` use from the downstream DSP chain and raw-recording tap, and mirrored suppressor latency into atomics for lock-free reporting.
- Deleted the unused legacy `RecordingWorker` path and tightened the packaged build flow around the checked-in spec and bundle-pruning helper.

## v1.7.6 - 2026-03-09

- Trimmed the packaged Windows app further by dropping the duplicated runtime icon payload and pruning unused Qt translations/PDF binaries after build.
- Prefer bundled `_internal/models` and `_internal/df.dll` more aggressively in the frozen launcher/runtime path to keep packaged asset lookup canonical.
- Kept the full-feature release payload intact: RNNoise, VAD, DeepFilter LL, and DeepFilter Standard remain bundled.

## v1.7.5 - 2026-03-09

- Hardened DeepFilter runtime loading and surfaced backend availability/error state in the UI.
- Added runtime diagnostics for suppressor non-finite resets, stream recovery, underruns, and dropped samples.
- Moved calibration and latency-calibration processor access back to the main Qt thread to avoid `unsendable` PyO3 cross-thread calls.
- Tuned watchdog recovery suppression for recording/calibration workflows.
- Switched the Windows package build to the checked-in `AudioForge.spec` and removed the redundant packaged `python/mic_eq` source tree.

## v1.7.4 - 2026-03-09

- Fixed CPAL stream setup for `f32`, `i16`, and `u16` devices so startup no longer depends on float-native hardware.
- Added output-side resampling so non-48 kHz playback devices receive correctly timed audio.
- Removed steady-state per-frame DeepFilterNet allocations from the realtime processing path.
- Trimmed packaged release size by relying on bundled internal DeepFilter assets instead of duplicating them next to the executable.

## v1.7.3 - 2026-02-28

- Fixed suppressor non-finite output poisoning the DSP/output pipeline after extended silence.
- Added runtime suppressor output sanitization and automatic suppressor reinitialization on detection.
- Reworked stop/start suppressor reset path to rebuild suppressor state for reliable in-app recovery.

## v1.7.2 - 2026-02-25

- Guarded biquad Q from zero to prevent NaNs.

## v1.7.1 - 2026-02-25

Note: releases were behind master; this rollup includes changes since v1.5.3.

- Added callback-based stream supervisor with auto-recovery and backoff.
- Added headless health check and self-test tools.
- Improved DeepFilterNet auto-enable when assets are present.
- Refactored downstream DSP chain to remove duplicated logic.
- Split output mute flag from recording state.
- Switched recording level meter to a sliding RMS window.
- Added/updated tests for preset VAD persistence.
- Misc: health check warmup handling, clearer docs, packaging notes.
