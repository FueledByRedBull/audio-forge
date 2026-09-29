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

Current version: `v1.13.0` — [release notes](release-notes/release-notes-v1.13.0.md).

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

### In the current source checkout (unreleased)

- Use **Compare Recording** after calibration to hear the same passage as
  original, current, and proposed processing. Choose headphones or speakers
  explicitly. Optional level matching affects playback only; actual level
  changes and peak protection remain visible. Switching stays aligned; Stop
  ends playback, and the preview destination is saved separately. The preview includes input
  cleanup, gating, suppression, both EQ stages, and dynamics. Recordings stay
  in memory until the dialogs close.
- **Options > Tray & Background** enables close-to-tray and the **Ctrl+Alt+M** global mute
  shortcut. Both are optional. With close-to-tray enabled, closing the window
  keeps audio running; **File > Exit** or **Quit AudioForge** in the tray stops it.
  Relaunching shows the existing window; Details reveals technical health counters.
- Select **Normal**, **Bypass**, or **Raw** from the processing-mode control.
  Saved calibration status becomes stale when its route or processing settings change.
- Calibration establishes an Auto-EQ stage; a separate tone stage
  preserves it when you change gains, templates, filter types, or slopes.
  Speech alone cannot identify the microphone's frequency response. Automatic
  targets preserve the recorded voice and apply bounded tonal preferences;
  Natural / No Added Tone does not attempt to flatten the voice spectrum.
  Warm / Full Voice adds broad low-mid body with restrained upper presence;
  Adaptive keeps it subtle, while Static uses the stronger catalog curve.
  Older presets retain their combined EQ response as the tone stage.
  Tone edits preserve matching microphone evidence; changed output settings
  make previous output verification stale.
- `Ctrl+S` updates your current saved preset; `Ctrl+Shift+S` saves a copy. Quit
  and preset replacement offer Save, Discard, or Cancel for unsaved sound changes.
- Auto Voice Setup tests gate and suppression choices across available RNNoise
  and DeepFilterNet models with the proposed dynamics, within a 35 ms suppressor-latency bound.
  DeepFilter settings stay within the DeepFilter family because automatic
  switches back to RNNoise failed clean-speech preservation checks.
  It keeps the current choices unless a candidate passes safety and speech
  preservation checks and improves on a separate part of the capture.
  Runtime is a pass/fail performance limit, so CPU timing differences do not
  rank otherwise viable sound settings.
  Final candidate and second-take checks render the combined cleanup, gate,
  suppression, EQ, and dynamics settings; adjustments are checked again before
  keeping the result.
  Analysis reports the current stage and candidate progress; compression candidates
  run in small parallel batches while retaining the same quality and safety checks.

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
- Auto Voice Setup with noise-reference integrity checks, Silero-posterior-aware speech masking, calibrated soft de-esser fusion, independent Gentle/Balanced/Dense/Custom dynamics intensity, and bounded native compressor calibration. EQ is fitted before compressor calibration; the compressor search includes the requested auto makeup and the final native full-chain headroom check. Native DSP validates the complete candidate; unavailable or unsafe headroom blocks Apply. Lower Target loudness and rerun when the requested level cannot leave safe headroom. Natural / No Added Tone remains neutral.
- Dynamic-EQ de-esser, compressor with speech-aware auto makeup gain driven by calibrated VAD and noise-floor evidence, and lookahead limiter.
- Band-limited 16x true-peak detection and limiting with independent offline burst and speech checks. Final peak protection uses 320 samples of lookahead when enabled (6.67 ms at 48 kHz), in addition to the other processing and device delays.
- Stateful phase-safe mono alignment and adaptive 49-61 Hz hum/harmonic tracking for difficult input sources.
- Per device-pair route-aware latency calibration profiles; measured output-to-input route delay is applied directly instead of assuming symmetric one-way latency.
- Raw monitor and bypass paths for troubleshooting.
- Bounded full-processing undo/redo (`Ctrl+Z` / `Ctrl+Shift+Z`) for manual
  edits, presets, Auto-EQ, and Auto Voice Setup, with realtime state excluded.

Operational tools:

- Input/output meters and runtime diagnostics.
- Dropped-sample, backlog, callback-stall, and recovery counters.
- Stream restart/backoff handling.
- Portable PyInstaller packaging with bundled runtime assets.

Useful behavior to know:

- Device refresh keeps the current selection when the same device is still available.
- Input/output stream setup prefers 48 kHz configs when available.
- VAD Assisted uses the tracked noise floor when auto threshold is enabled. VAD Only opens from speech confidence; its displayed noise floor is not the opening threshold.
- Phase-safe mono retains fractional-delay history across input callbacks instead of re-estimating from isolated blocks.
- Adaptive cleanup tracks off-nominal mains hum and its harmonic with fractional frequency/phase continuity, and selects one high-pass response instead of cascading filters.
- Auto-EQ and Auto Voice Setup analyze 48 kHz captures, use native Silero posteriors when available, and report an explicit energy-analysis fallback when they are not.
- Auto Voice Setup rejects unusable room tone, restricts boosts for questionable references, and reports device/time/channel mismatch or recapture guidance.
- Voice Setup candidates remain temporary until a second passage checks repeatability through input cleanup, gate, noise suppression, EQ, de-essing, compression, and the selected limiter settings. Recorded verification cannot reproduce live device dropouts or worker scheduling delays; confirm the result in your destination app.
- Verification reuses a valid second passage when adjusting processing. Only recording problems request another take; unsuccessful bounded adjustments restore your previous settings with the specific reason.
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
11 x64. Release-specific software and package validation results are recorded
in each release's evidence archive. Earlier candidate hardware tests passed on
Windows 11 with USB and virtual routes at 48 kHz, including 30-minute runs and model switching. Those
measurements do not qualify the final binary, which has not repeated the full
hardware run. Windows 10, analog input, 44.1 kHz, and physical device lifecycle
cases remain unqualified. See the [release workflow](RELEASING.md#automated-workflow)
for validation and optional hardware evidence.
Linux and macOS builds are not supported.

Objective DSP decisions and release evidence are indexed in
[`evaluation/README.md`](evaluation/README.md), including historical provenance
limits. New reports contain compact
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
- GitHub CLI (`gh`) configured for release downloads, and 7-Zip (`7z` on PATH
  or installed in `C:/Program Files/7-Zip`), for runtime asset hydration.
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

.\.venv\Scripts\python.exe -m maturin develop --release --locked
.\.venv\Scripts\python.exe -m mic_eq
```

The fallback release is pinned once in `release-assets.json`; it supplies
runtime assets independently of the application version. Hydration verifies
each asset against the manifest.

## Configuration

RNNoise is the default suppression backend. For source runs, enable DeepFilterNet
with `AUDIOFORGE_ENABLE_DEEPFILTER=1` after hydrating the runtime assets. Packaged
builds enable verified bundled DeepFilter assets automatically. External DLL/model
paths require an explicit opt-in; see [runtime configuration](CONTRIBUTING.md#runtime-assets-and-configuration).

## Development and Packaging

- [CONTRIBUTING.md](CONTRIBUTING.md): development checks, runtime configuration,
  and sanitized screenshot generation.
- [RELEASING.md](RELEASING.md): portable/MSI builds, package validation,
  release archives, and publication.
- [evaluation/README.md](evaluation/README.md): objective DSP evidence and retention.

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
