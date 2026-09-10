# AudioForge

**Shape your live microphone sound. Keep the processing on your PC.**

[![Latest release](https://img.shields.io/github/v/release/FueledByRedBull/audio-forge?label=download&color=287dba)](https://github.com/FueledByRedBull/audio-forge/releases/latest)
[![CI on master](https://github.com/FueledByRedBull/audio-forge/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/FueledByRedBull/audio-forge/actions/workflows/ci.yml?query=branch%3Amaster)
[![Windows x64](https://img.shields.io/badge/platform-Windows_x64-287dba)](#status)
[![Source license: MIT](https://img.shields.io/badge/source-MIT-64748b)](LICENSE)
[![Packaged app: GPLv3](https://img.shields.io/badge/packaged_app-GPLv3-64748b)](#license)

AudioForge is a Windows desktop microphone processor for calls, streaming, and
recording. Reduce background noise, shape your voice with editable EQ and
dynamics, and use guided calibration to find a starting point. Audio processing
runs locally through a Rust engine and a PyQt interface.

**[Download](#download)** · [First setup](#using-the-app) · [Features](#what-it-does) · [Build from source](#quick-start-from-source) · [Get help](#help-and-contributing)

Current version: `v1.12.0`

![AudioForge main window showing sanitized input and virtual-route output selection, cleanup controls, and the editable ten-band EQ.](docs/images/audioforge-routing-eq.png)

## Download

Get the [latest published Windows release](https://github.com/FueledByRedBull/audio-forge/releases/latest).
Choose **one** app download:

| Download | Best for | Start here |
| --- | --- | --- |
| `AudioForge-v<version>-win64.msi` | A normal installation for your Windows user | Run the installer, then open AudioForge. |
| `AudioForge-v<version>-win64-ultra.7z` | A portable folder you can keep where you want | Extract the entire archive, then run `AudioForge.exe`. |

Both include the application runtime: **you do not need Python or Rust to run
the packaged app**. Keep the portable folder's files together. If Windows cannot
extract the `.7z` archive, use [7-Zip](https://www.7-zip.org/) or choose the MSI.
Windows 10 (1809+) and Windows 11 x64 are the compatibility targets; see
[validation and known limits](#status) for configurations actually tested.

The release also includes `AudioForge-v<version>-source.7z`, corresponding
source for redistribution/rebuilding, and `SHA256SUMS.txt` for checking downloads.
See [release instructions](RELEASING.md) for the complete asset and evidence layout.

## Using The App

For a call or stream, the signal travels through this route:

```text
Microphone -> AudioForge -> Virtual audio route -> Call / streaming app
```

A virtual audio route connects one app's output to another app's input. Select
its output/playback side in AudioForge and its input/recording side in the
receiving app; their names depend on the installed driver.

1. **Choose a route.** Select your microphone as input. For monitoring, choose
   headphones as output. For calls or streaming, choose a virtual audio route
   you have installed, then select its receiving endpoint as the microphone in
   your destination app. AudioForge does not install a virtual audio driver.
2. **Start and check.** Press **Start Processing**, speak normally, and check the
   meters for signal and clipping. A successful check means both AudioForge
   and your destination app show input activity, and a test recording in the
   destination contains your processed voice.
3. **Shape your sound.** Choose noise suppression and a gate mode, then adjust
   EQ/dynamics manually, run Auto-EQ, or use Auto Voice Setup for guided calibration.
4. **Save what works.** Save a complete sound preset to recall your processing
   settings. Use mute when you need to silence transmission without quitting.
5. **Measure latency only if needed.** Optional route calibration improves the
   reported estimate; it does not reduce physical audio delay.

AudioForge opens with processing stopped; use **Start Processing** to send audio.
**Stop Processing** leaves the app open. Closing the main window or choosing
**File > Exit** stops audio and quits. AudioForge does not start with Windows.

## What It Does

| Your goal | Tools in AudioForge |
| --- | --- |
| Reduce background noise | RNNoise or DeepFilterNet suppression, speech-aware gating, and input cleanup. |
| Shape your voice | Editable ten-band EQ, Auto-EQ, and guided Auto Voice Setup. |
| Control harshness and peaks | De-esser, compressor, and lookahead limiter. |
| Keep control during daily use | Presets, undo/redo, mute, level meters, and runtime diagnostics. |
| Understand your route | Input/output selection, monitoring, and optional latency measurement. |

<details>
<summary>See dynamics controls and Auto Voice Setup</summary>

### Dynamics processing

![AudioForge dynamics view showing compressor and limiter controls, health indicators, and the editable EQ.](docs/images/audioforge-processing.png)

### Auto Voice Setup

![AudioForge Auto Voice Setup dialog showing target and dynamics choices plus sanitized validated recommendation summaries.](docs/images/audioforge-auto-voice-setup.png)

</details>

Screenshots show deterministic, sanitized app state.

<details>
<summary>Detailed processing features and settings behavior</summary>

User-facing tools:

- AI noise suppression with RNNoise and optional DeepFilterNet backends.
- Smart noise gate with threshold-only, VAD-assisted, and VAD-only modes, including smoothed continuous VAD-posterior gain control around uncertain speech.
- Auto thresholding that tracks the live noise floor in VAD modes.
- 10-band parametric EQ with per-band bell, notch, shelf, and pass filters,
  click-safe bypass, selectable 12–48 dB/octave Butterworth pass slopes, and
  constrained mouse/keyboard graph editing synchronized with numeric controls.
- Auto-EQ calibration that combines energy and Silero speech posteriors, rejects shape outliers, uses matched noise-referenced per-band reliability when available, and abstains when a safe correction is unsupported.
- Auto-EQ headroom validation through the native chain simulator; Python-only fallback results are visibly advisory.
- Auto Voice Setup with noise-reference integrity checks, Silero-posterior-aware speech masking, calibrated soft de-esser fusion, independent Gentle/Balanced/Dense/Custom dynamics intensity, bounded multi-parameter native compressor calibration (threshold, ratio, attack, release), and guided second-passage verification.
- Dynamic-EQ de-esser, compressor with speech-aware auto makeup gain driven by calibrated VAD and noise-floor evidence, and lookahead limiter.
- Band-limited 4x true-peak detection and limiting, validated against an independent offline reference.
- Stateful phase-safe mono alignment and adaptive 49-61 Hz hum/harmonic tracking for difficult input sources.
- Per device-pair route-aware latency calibration profiles; measured output-to-input route delay is applied directly instead of assuming symmetric one-way latency.
- Raw monitor and bypass paths for troubleshooting.
- Bounded full-processing undo/redo (`Ctrl+Z` / `Ctrl+Shift+Z`) for manual
  edits, presets, Auto-EQ, and Auto Voice Setup, with realtime state excluded.

Presets use a versioned typed-band schema; migration tests preserve explicit
user values and response parity. The graph and numeric controls share that
schema and the native Rust response renderer. Undo snapshots contain validated
processing settings only—not audio, device handles, DSP delay state, or meter
history. These contracts are enforced by the config, EQ, graph, and history
tests rather than duplicated across separate design documents.

Operational tools:

- Input/output meters and runtime diagnostics.
- Dropped-sample, backlog, callback-stall, and recovery counters.
- Stream restart/backoff handling.
- Device refresh that preserves current selections when possible.
- Portable PyInstaller packaging with bundled runtime assets.

Useful behavior to know:

- Device refresh keeps the current selection when the same device is still available.
- Input/output stream setup prefers 48 kHz configs when available.
- In VAD modes, auto threshold is the default path; the UI shows live noise floor and effective threshold.
- Phase-safe mono retains fractional-delay history across input callbacks instead of re-estimating from isolated blocks.
- Adaptive cleanup tracks off-nominal mains hum and its harmonic with fractional frequency/phase continuity, and selects one high-pass response instead of cascading filters.
- Auto-EQ and Auto Voice Setup use native Silero posteriors when available and report an explicit energy-analysis fallback when they are not.
- Auto Voice Setup rejects unusable room tone, restricts boosts for questionable references, and reports device/time/channel mismatch or recapture guidance.
- Voice Setup candidates remain temporary until a second passage checks repeatability through EQ, de-essing, compression, and the selected limiter settings. Gate, noise suppression, input cleanup, and live loudness adaptation are outside this offline check; confirm the result in your destination app.
- Preset loading preserves saved `VAD Assisted` and `VAD Only` gate modes instead of collapsing them back to `Threshold Only`.
- Diagnostics separate input drops, backlog recovery, output recovery, output short-write loss, and active output underrun streaks. Historical output underrun and recovery totals stay visible without forcing the health chip into a warning state after the stream has recovered.
- `Help > Export Diagnostics...` writes a versioned, size-bounded support
  snapshot. It allowlists configuration and runtime health fields,
  pseudonymizes device identities with report-local keys, and excludes raw
  audio, raw device names, environment variables, secrets, and arbitrary
  paths.

</details>

## Status

AudioForge's compatibility targets are Windows 10 (1809 or later) and Windows
11 x64. The final v1.12.0 portable EXE and MSI passed hosted software and package
validation. Earlier candidate hardware tests passed on Windows 11 with USB and
virtual routes at 48 kHz, including 30-minute runs and model switching. Those
measurements do not qualify the final binary, which has not repeated the full
hardware run. Windows 10, analog input, 44.1 kHz, and physical device lifecycle
cases remain unqualified. See the [release workflow](RELEASING.md#automated-workflow)
for validation and optional hardware evidence.
Linux and macOS builds are not supported.

DeepFilterNet support is intentionally opt-in for source runs. Packaged builds register and enable verified bundled assets during application bootstrap; RNNoise remains the safe default when those assets are absent. External DLL/model paths are ignored unless `AUDIOFORGE_ALLOW_EXTERNAL_DF=1` is explicitly set.
DeepFilter model/DLL initialization and Silero VAD inference are prepared off the realtime DSP loop; the audio path only swaps ready suppressor state and consumes cached VAD probabilities.

Objective DSP decisions and release evidence are indexed in
[`evaluation/README.md`](evaluation/README.md). Tracked reports contain compact
aggregates, gates, hashes, decisions, and limitations; raw per-case details are
optional ignored outputs, not repository content.

## Help and Contributing

- **No sound?** Check that processing is started, mute is off, and the selected
  output reaches the input selected in your destination app. Watch the meters
  to see where the signal stops.
- **Report a bug or request a feature:** [open an issue](https://github.com/FueledByRedBull/audio-forge/issues/new/choose).
  Include your app version, expected behavior, and steps to reproduce. Use
  **Help > Export Diagnostics...** for a bounded support snapshot; inspect any
  attachment before sharing it.
- **Contribute:** follow [CONTRIBUTING.md](CONTRIBUTING.md) for development and checks.
- **Report a security issue:** use [SECURITY.md](SECURITY.md).
- **Review measured behavior:** see the [evaluation index](evaluation/README.md).

## DSP Chain

Normal processing path:

```text
Mic Input -> Input Cleanup (DC block + one selected/adaptive HP) -> Noise Gate -> Noise Suppression
-> De-Esser -> 10-Band EQ -> Compressor -> Limiter -> Output
```

Special paths:

- `Bypass` skips voice effects while retaining input conditioning and configured output protection.
- `Raw Monitor` skips input filtering and voice effects for diagnostics, retains configured output protection, and takes precedence over Bypass.

Latency labels include engine/suppressor timing plus measured route delay when enabled. Calibration measures the selected output-to-input route and adds it to the reported estimate; it does not reduce physical delay. A directional one-way split is left unset unless independently measured. End-to-end latency still depends on the selected devices, driver mode, buffer sizing, and routing path.

## Requirements

These requirements are for **building from source**, not running the download.

- Windows 10 (1809 or later) or Windows 11, x64
- CPython 3.13.15 x64
- Rust 1.94.0, selected by `rust-toolchain.toml`
- `maturin`
- A virtual environment in `.venv` is assumed by the packaging script.

## Quick Start From Source

See [CONTRIBUTING.md](CONTRIBUTING.md) for checks and contribution guidance,
[SECURITY.md](SECURITY.md) for vulnerability reporting, and
[third-party notices](licenses/THIRD_PARTY_NOTICES.md) for binary distribution
terms. AudioForge's original source is MIT; the PyQt6-based application is
distributed under GPLv3 together with its dependency notices.

```powershell
git clone https://github.com/FueledByRedBull/audio-forge.git
cd audio-forge

py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements/dev.txt
.\.venv\Scripts\python.exe python/tools/fetch_release_assets.py
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .

.\.venv\Scripts\python.exe -m maturin develop --release
.\.venv\Scripts\python.exe -m mic_eq
```

You can also use the installed console entrypoint:

```powershell
.\.venv\Scripts\mic-eq.exe
```

## Configuration

RNNoise is the default safe noise-suppression backend. DeepFilterNet is opt-in for source and development runs; set `AUDIOFORGE_ENABLE_DEEPFILTER=1` after registering app-owned assets or when intentionally using opted-in external assets. Packaged builds register canonical bundled DLL/model paths and auto-enable DeepFilterNet when both are present.

See [Development Assets](#development-assets) for the full runtime asset and environment-variable list.

<details>
<summary>Developer reference: assets, packaging, tests, and repository layout</summary>

## Development Assets

Full-feature development and release builds use the tracked `release-assets.json` manifest. Obtain each listed asset from the documented source, place it at the manifest `path`, and verify before packaging:

```powershell
.\.venv\Scripts\python.exe python/tools/verify_release_assets.py
```

For a cleaner fresh-clone setup, you can hydrate those assets from the matching GitHub release:

```powershell
.\.venv\Scripts\python.exe python/tools/fetch_release_assets.py
```

The fallback release is pinned once in `release-assets.json`; it is an asset
source, not the application version. Silero v6.2.1 is downloaded directly from
the immutable source recorded in the same manifest; every file is then verified
by size and SHA-256.

Create `models/` in the repo root for local runtime discovery:

- `models/DeepFilterNet3_ll_onnx.tar.gz`
- `models/DeepFilterNet3_onnx.tar.gz`
- `models/silero_vad.onnx`

Native runtime libraries:

- `df.dll` in the repo root for development runs.
- `target/onnxruntime-cpu/lib/onnxruntime.dll` and `onnxruntime_providers_shared.dll` from the pinned CPU-only ONNX Runtime package.
- Bundled under `dist/AudioForge/_internal` for portable builds.

Environment variables:

- `AUDIOFORGE_ENABLE_DEEPFILTER=1`
- `AUDIOFORGE_ALLOW_EXTERNAL_DF=1`
- `DEEPFILTER_MODEL_PATH`
- `DEEPFILTER_LIB_PATH`
- `VAD_MODEL_PATH`
- `AUDIOFORGE_FIXED_INPUT_BUFFER_FRAMES`
- `AUDIOFORGE_FIXED_OUTPUT_BUFFER_FRAMES`

Packaged builds use bootstrap-registered canonical DeepFilter assets by default. Ambient `DEEPFILTER_LIB_PATH` and `DEEPFILTER_MODEL_PATH` values are ignored. Set `AUDIOFORGE_ALLOW_EXTERNAL_DF=1` only when you intentionally want a valid external path to take precedence; any missing external path falls back to the registered bundled asset.

The two fixed-buffer variables are optional diagnostics for callback-size
consistency. Values must be 16–8192 frames and fit the endpoint's advertised
range; AudioForge preflights the request and otherwise keeps the driver default.

## Build Portable EXE

The packaging entry point rebuilds the Rust extension before freezing:

```powershell
.\.venv\Scripts\python.exe python/tools/fetch_release_assets.py
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```

Packaging script behavior:

- Rebuilds `python/mic_eq/mic_eq_core*.pyd` from the current source and lockfile.
- Validates required full-feature runtime assets against `release-assets.json`.
- Reuses PyInstaller's analysis cache by default; pass `-Clean` for a cold PyInstaller rebuild.
- Bundles the Python runtime with PyInstaller.
- Bundles GPLv3 distribution terms, dependency inventory, and retained license notices.
- Writes `_internal/audioforge-build.json`; package smoke rejects a bundle whose version differs from the source tree.
- Prunes unused Qt payload, duplicate native-extension payload, and app-local
  UCRT/API-set files with `python/tools/prune_bundle.py` while retaining
  dependency metadata and licenses. AudioForge targets Windows 10/11 and
  relies on the operating system UCRT, which Windows always uses on those
  versions even if a local copy is present.
- The release profile strips native symbols, and packaging excludes only unused
  SciPy namespaces plus Qt SVG payloads. Each candidate emits generated
  artifact metadata and a per-file bundle manifest; do not copy candidate
  sizes, file counts, or hashes into pre-release prose.
- Keeps the application self-contained in `dist/AudioForge`.

Portable output:

- `dist/AudioForge/AudioForge.exe`
- Bundled assets and runtime files under `dist/AudioForge/_internal`

## Build the MSI installer

After building the portable payload, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_msi.ps1
```

The installer uses the same portable payload and installs for the current user.
The script acquires and verifies its pinned WiX tooling. CI compares the
extracted and installed file tree with the portable bundle and checks uninstall
preserves user configuration. MSI candidates are published only after the same
release qualification gates as the portable archive.

## Create Release Archive

The portable folder is intended to be archived as a single distributable:

```powershell
& "C:/Program Files/7-Zip/7z.exe" a -t7z -mx=9 -m0=lzma2 -mmt=on -ms=on `
  .\AudioForge-v1.12.0-win64-ultra.7z .\dist\AudioForge\*
```

The v1.10.0 bundle was measured with ZIP/Deflate, tar.gz, tar.xz, tar.zst,
solid LZMA, and solid LZMA2. The command above was the smallest verified
format. Treat the generated `.metadata.json`, `.manifest.json`, and `.sha256`
sidecars in the release evidence archive as authoritative. See
`evaluation/archive-format-benchmark.json` for the historical format
comparison.

## Testing

CI-equivalent checks:

```powershell
.\.venv\Scripts\python.exe -m ruff check python/mic_eq python/tests python/tools
.\.venv\Scripts\python.exe -m pyright
.\.venv\Scripts\python.exe -m pytest python/tests -q
.\.venv\Scripts\python.exe -m pip_audit --require-hashes -r requirements/runtime.txt --disable-pip
.\.venv\Scripts\python.exe -m pip_audit --require-hashes -r requirements/dev.txt --disable-pip
.\.venv\Scripts\python.exe python/tools/run_semgrep.py --sarif semgrep-results.sarif
.\.venv\Scripts\python.exe python/tools/check_versions.py
.\.venv\Scripts\python.exe python/tools/check_workflows.py
.\.venv\Scripts\python.exe python/tools/package_smoke.py --source-only
cargo fmt --check
cargo audit
cargo test -p mic_eq_core
cargo test --release -p mic_eq_core --test stress_tests seeded_control_and_dsp_loops_remain_finite_under_contention
cargo test --release -p mic_eq_core audio::input::tests::benchmark_phase_safe_mono_callback_cost -- --ignored --nocapture
cargo test --release -p mic_eq_core dsp::biquad::tests::benchmark_biquad_morph_cost -- --ignored --nocapture
cargo clippy -p mic_eq_core --all-targets -- -D warnings
```

Packaged-build smoke check after `build_exe.ps1`:

```powershell
.\.venv\Scripts\python.exe python/tools/verify_release_assets.py
.\.venv\Scripts\python.exe python/tools/package_smoke.py
```

Headless runtime checks:

```powershell
.\.venv\Scripts\python.exe python/tools/health_check.py --duration 1800
.\.venv\Scripts\python.exe python/tools/self_test.py
.\.venv\Scripts\python.exe python/tools/evaluate_hardware_validation.py `
  --health-input "<microphone>" --health-output "<virtual output>" `
  --correlation-input "<loopback input>" --correlation-output "<loopback output>"
```

## Repository Layout

Regenerate the sanitized README screenshots with
`python/tools/capture_repository_screenshots.py`; existing checks cover their
dimensions, hashes, alt text, and privacy boundary.

- `python/mic_eq`: PyQt application, analysis code, persistence, and source/development entrypoints.
- `rust-core`: Rust audio engine exposed through PyO3.
- `python/tests`: Python test suite.
- `python/tools`: health, package, and release validation helpers.
- `.github/workflows/ci.yml`: Windows CI for Python and Rust checks.
- `build_exe.ps1`: PyInstaller packaging script.
- `AudioForge.spec`: canonical portable package definition.
- `launcher.py`: PyInstaller/frozen-app launcher used by `AudioForge.spec`; source/development runs use `python -m mic_eq` or the `mic-eq` console entrypoint.

</details>

## Roadmap

Versioned [GitHub milestones](https://github.com/FueledByRedBull/audio-forge/milestones)
and issues carrying the
[`roadmap` label](https://github.com/FueledByRedBull/audio-forge/issues?q=is%3Aissue+label%3Aroadmap)
are the source of truth for planned work and explicit holds. Keep personal
working notes outside the repository and do not treat them as authoritative.

## License

The original AudioForge source is MIT; see [LICENSE](LICENSE). Portable and MSI
distributions that include PyQt6 are distributed under GPLv3 together with the
notices in [licenses/THIRD_PARTY_NOTICES.md](licenses/THIRD_PARTY_NOTICES.md)
and the corresponding source materials described in
[licenses/SOURCE_DISTRIBUTION.md](licenses/SOURCE_DISTRIBUTION.md).

## Acknowledgments

- RNNoise by Jean-Marc Valin
- DeepFilterNet by Hendrik Schroter and contributors
- Silero VAD contributors
