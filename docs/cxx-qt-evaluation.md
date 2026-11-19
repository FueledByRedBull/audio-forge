# cxx-qt shell evaluation

Evaluated on 2026-11-19 at commit `04091405dc7530e6bcf2081fc70df76825200278`
(version 1.23.0).

## Question

Should AudioForge replace its Python application (PySide6 6.11.1, frozen with
PyInstaller 6.21.0) with a Rust binary that drives the existing QML through
cxx-qt? This is the only option considered that removes Python from the shipped
app; the realtime engine is already Rust (`rust-core`, crate `mic_eq_core`).

## Decision

Not now. The Rust binary is feasible, but the work is mostly a rewrite of about
41,000 lines of Python, including 11,800 lines of NumPy analysis whose output
the shipped evaluation evidence depends on. It would not improve the audio path,
which has no Python on it. The conditions that would reopen it are listed at the
end.

## What the app is today

| Area | Files | Lines |
| --- | ---: | ---: |
| `python/mic_eq/ui/` (Python) | 65 | 24,836 |
| `python/mic_eq/ui/qml/` (QML) | 23 | 3,652 |
| `python/mic_eq/analysis/` (including `auto_eq_parts/`, 3,902 lines) | 26 | 11,834 |
| `python/mic_eq/config_parts/` | 8 | 3,213 |
| `python/mic_eq/` top level (`diagnostics_export.py` 832, `runtime_snapshot.py`, `config.py`, ...) | 6 | 1,347 |
| Python total | 105 | 41,230 |
| `rust-core/src` | | 57,674 |

The Rust core holds the audio I/O (`cpal` 0.17.3 on Windows, 0.18.2 with
PipeWire on Linux, `wasapi_render.rs`), the DSP chain, RNNoise (`nnnoiseless`
0.5.2), DeepFilter over FFI, Silero VAD through `ort` 2.0.0-rc.11, and the PyO3
0.29 bindings (`audio/processor/python_api.rs` 3,343 lines,
`analysis/python_api.rs` 562, `analysis/headroom.rs` 554). Cargo.lock has 205
packages.

## 1. Packaging

### Today

- `build_exe.ps1` (192 lines) verifies the runtime assets
  (`verify_release_assets.py`), rebuilds the extension with `maturin develop
  --release --locked`, collects dependency notices (`license_inventory.py`,
  496 lines), runs PyInstaller on `AudioForge.spec`, then prunes the bundle
  (`prune_bundle.py`, 170 lines). `build_linux.sh` (68 lines) is the Linux
  counterpart and uses `maturin build` plus `linux_package.py` (453 lines) to
  strip RPATHs and filter host libraries.
- `AudioForge.spec` (249 lines) carries the QML directory as data, the notice
  inventory, `df.dll`/`libdf.so`, ONNX Runtime and three model files, a
  hidden-import list (QtOpenGL must stay; libshiboken imports it), an exclude
  list (SciPy, ssl, readline, PyQt6), and on Linux a `readelf` walk that keeps
  only directly needed Qt and host libraries.
- `release-assets.json` pins six native assets by size and SHA-256: `df.dll`,
  `onnxruntime.dll`, `onnxruntime.lib`, `onnxruntime_providers_shared.dll` on
  Windows; `libonnxruntime.so.1.23.2` and `libdf.so` on Linux. None of them is
  Python-specific; all survive a shell change unchanged.
- `build_msi.ps1` (170 lines) and `installer/AudioForge.wxs` (71 lines, WiX 7)
  harvest `dist/AudioForge/**` into a per-user MSI. The only app coupling is
  `AudioForge.exe --remove-login-startup` at uninstall.
- Linux: `python/tools/aur_package.py` writes the `audioforge-bin` PKGBUILD
  from the release tarball (licences `MIT`, `LGPL-3.0-only`);
  `installer/linux/` holds the desktop entry and AppStream metainfo. There is
  no `packaging/` directory.

A local 1.23.0 Windows build (not compared with the published archive) is
225 MB. Of that, about 70 MB is Qt itself (Qt6 DLLs, plugins, QML modules,
`opengl32sw.dll`), 45 MB models, 31 MB DeepFilter and ONNX Runtime. The
Python-only part is roughly 60 MB: PySide6 bindings (23 MB of `.pyd`),
NumPy and its OpenBLAS (28 MB), `python313.dll` and the standard library
(about 9 MB), shiboken6 and pywin32.

### With a cxx-qt binary

- Build: `cxx-qt-build` locates Qt through `qmake` (the `QMAKE` variable or
  `PATH`), needs a C++ compiler and a full Qt SDK at build time. The PySide6 wheels
  are not a C++ SDK, so CI would install Qt 6.11 for MSVC and for Linux
  (for example with `aqtinstall`), cached per runner. The `qt_minimal` download
  option is aimed at QML-only use and does not cover every module the app uses.
- Deployment: `windeployqt` (Windows) and a `linuxdeploy`-style step or the
  existing `readelf` walk (Linux) replace PyInstaller's collection. The prune
  rules, the QtOpenGL exception and the frozen-only failure cases would be
  re-derived for the new layout, and `package_smoke.py` (522 lines) and the
  reviewed path baselines (`release-bundle-path-baseline*.json`) rewritten.
- MSI and AUR: small changes. The WiX source harvests a directory and keeps
  working; the PKGBUILD repackages a tarball. AppStream and desktop files are
  unaffected.
- Licences: cxx-qt and cxx-qt-lib 0.10.0 are `MIT OR Apache-2.0`, compatible
  with the MIT project. Qt stays LGPLv3 with dynamic linking, which is the
  obligation the app already meets for the Qt inside PySide6. What changes is
  provenance: `licenses/SOURCE_DISTRIBUTION.md` and `source-manifest.json`
  currently point at the Qt for Python 6.11.1 source and the Qt modules behind
  the PySide6 wheels; they would point at the Qt SDK release instead, and the
  PySide6/shiboken6 notices go away. Static linking (supported by cxx-qt) would
  add LGPL relinking obligations and is not worth it.
- Build cost: an additional C++ toolchain and Qt SDK per CI image, plus
  cxx-qt's generated C++ per QObject. The current Windows release build job
  allows 120 minutes and the Linux one 90; no cxx-qt build time was measured
  here. Removing maturin, PyInstaller and the hashed `requirements/` lock
  reduces the Python tooling, but the repository still needs Python for
  `python/tools` (30 scripts, 8 of which import `mic_eq`).

Packaging is not the blocker. It is 3 to 5 weeks of work and the size saving is
real (about a quarter of the bundle), but nothing in the current packaging
fails today.

## 2. Python-only code

### Already native

The 1.23.0 cycle moved the analysis primitives, latency calibration, the
headroom verdict and the processing-settings ranges into Rust. The PyO3 module
(`rust-core/src/lib.rs`) exports 3 classes (`AudioProcessor`, `GateMode`,
`DeviceInfo`) and 28 functions, among them `analyze_latency`,
`generate_probe_signal`, `resample_poly`, `band_mean`,
`find_peaks_distance_prominence`, `analyze_offline_vad`,
`simulate_auto_eq_chain`, `simulate_eq_v2`, `eq_magnitude_response_v2`,
`processing_ranges`/`processing_defaults` and the virtual-microphone calls.
Every DSP stage, the resampler and the simulators that Voice Setup and Auto-EQ
render through are already Rust.

### Still Python

| Area | Lines | Rust counterpart |
| --- | ---: | --- |
| `analysis/voice_setup.py`, `joint_tuning.py`, `compressor_calibration.py`, `failure_detection.py` | 4,592 | Simulators only; the decisions, refusals and verification rules are Python |
| `analysis/auto_eq_parts/` (`optimizer.py` 1,237, `eq_fit.py` 1,035, `headroom.py` 529, ...) | 3,902 | Headroom verdict and EQ response are native; the fit and its NumPy solvers are not |
| `analysis/spectrum.py`, `noise_reference.py`, `eq_quality.py`, `listening_comparison.py`, de-esser corpus and fusion | 3,081 | None |
| `analysis/latency_calibration.py` | 122 | `analysis/latency.rs` (thin wrapper left) |
| `config_parts/` (presets 881, settings 844, app config 645, migrations and validation) | 3,213 | `config_ranges.rs` holds ranges and defaults only |
| `ui/` Qt-free modules | 5,156 | None |
| `ui/` QtCore-only modules | 6,801 | None |
| `ui/` QtWidgets and QQuickWidget hosts | 10,693 | None; cxx-qt has no QtWidgets bindings |
| `ui/` QtGui/QtQuick painted items and models | 2,186 | None |
| Platform glue (part of the `ui/` rows above): `login_startup.py` (pywin32 ShellLink), `desktop_integration.py` (single instance, hotkey filter), `global_shortcuts_portal.py` and `session_bus.py` (jeepney D-Bus), `application_bus.py` | 1,381 | None; the `windows` crate (0.62.2 in the lock) and `zbus` would replace them |

### Numerical parity

NumPy is imported by 27 modules. The analysis uses `np.fft.rfft` in
`voice_setup.py`, `spectrum.py`, `noise_reference.py`, `signal_processing.py`
and the de-esser modules, and `np.linalg` (`solve`, `svd`, `qr`, `cholesky`) in
the Auto-EQ fit. Two results from the native ports bound what a full port can
promise:

- NumPy's pocketfft and `realfft` round differently. The latency port
  therefore matches its NumPy reference only to rounding (at most 2.3e-15 on
  the test cases, tolerance 1e-12 in `test_latency_calibration.py`), while the
  measured lag, decision and counts match exactly. Where the ports avoided FFTs
  or fed NumPy's own window, sort and filter values in from Python
  (`rust-core/src/analysis.rs`), they are bit-identical.
- A port of `noise_reference.py` was dropped on 2026-11-19 because its spectra
  cannot be reproduced bit for bit in Rust.

Once Python leaves, the reference values that today come from NumPy through the
callers (`np.hanning`, the resampling filter, the peak sort) must be computed in
Rust too, so the bit-identical ports would also degrade to rounding parity. The
Auto-EQ fit already shows how sensitive the decisions are: SciPy moved by one
unit in the last place changed one recommendation status in 2,084 cases (see
the Auto-EQ solver study in `evaluation/README.md`). A Rust Voice Setup and
Auto-EQ would therefore change some decisions on the bench set. Under
`CONTRIBUTING.md` that is a tuning change: it needs before and after bench
reports and a maintainer decision, area by area, and the released evaluation
reports describe the Python behaviour, not the port.

## 3. UI seams

The November refactors split the main window. These modules have no PySide6
import and could move to Rust first, behind the PyO3 boundary, without changing
the shell: `ui/health.py` (912, the `HealthMonitor`),
`ui/processing_configuration.py` (797), `ui/setup_messages.py` (486),
`ui/capture_session.py` (343), `ui/auto_eq_explanation.py` (314),
`ui/config_history.py` (273), `ui/level_check.py` (256),
`ui/device_selection.py` (215), `ui/input_preferences.py` (106),
`ui/processing_mode_state.py` (98) and smaller ones (5,156 lines in 20 files).

These depend on QtCore only (signals, timers, threads) and map onto cxx-qt
QObjects: `ui/voice_setup_flow.py` (2,756, `VoiceSetupFlow`),
`ui/engine_session.py` (642), `ui/eq_state.py` (466), `ui/analysis_worker.py`
(256, `QThread`), the stage states (`gate_state.py`, `compressor_state.py`,
`stage_state.py`, ...) and the six scene view models (6,801 lines in 22 files).

The QML depends on these context objects:

| Scene | Context object | Members read |
| --- | --- | ---: |
| `Main.qml` (1,286 non-blank lines) | `ui`: `QuickBridge` in `quick_shell.py` | `model`, `levels`, `stepBand`, `resetDrops`, `exportDiagnostics` |
| `VoiceSetup.qml` | `vm`: `VoiceSetupView` | 28 |
| `AutoEqCalibration.qml` | `vm`: `CalibrationView` | 25 |
| `FirstRunSetup.qml` | `vm`: `FirstRunSetupView` | 21 |
| `ListeningComparison.qml` | `vm`: `ListeningComparisonView` | 16 |
| `LatencyCalibration.qml` | `vm`: `LatencyCalibrationView` | 12 |
| `RecordingProgress.qml` | `vm`: `RecordingProgressView` | 4 |
| all | `theme` (a dict) | |

`Main.qml` reads one dynamic `ui.model` dictionary that `QuickBridge` builds
from about 20 proxy objects (`SettingControl`, `NoiseControl`, `EQControl`,
`ActionControl`, `GateText`, `ChoiceState`, ...). In cxx-qt each becomes a typed
`#[cxx_qt::bridge]` QObject, and the dictionary becomes typed properties or a
list model. The four registered items (`EQGraphItem`, `EQComparisonItem`,
`QuickMeterItem`, `LevelTrace`) are `QQuickPaintedItem` subclasses that draw
with the QPainter renderers in `eq_graph.py` (1,179) and `level_meter.py` (572).

The larger gap is the host. Every scene still runs inside a `QQuickWidget` in
a QtWidgets window or dialog: `main_window.py` (3,648, `QMainWindow`, 56
`QAction`s, `QSystemTrayIcon`, `QMenu`), five dialog modules (4,000 lines),
`app_bootstrap.py` (703) and 99 `QMessageBox` uses. cxx-qt does not bind
QtWidgets, so all of this must become QML (`ApplicationWindow`, Qt Quick
Dialogs, a tray from `Qt.labs.platform` or a Rust crate) before or during the
switch. The software-rendering relaunch in `quick_shell.py` is written against
`QQuickWidget` and would move to `QQuickWindow`. The listening comparison plays
audio through QtMultimedia's `QAudioSink`, which cxx-qt-lib does not bind.

Moving the hosts to QML is worth doing on its own under PySide6: it is the
same work either way, it removes QtWidgets from the bundle, and it is the step
that makes a later shell change mechanical.

## 4. Tests

`pytest --collect-only` finds 6,675 tests in 170 files (67,050 lines, 2,535
test functions; parametrisation does the rest).

- 2,245 tests are in the 69 files that import PySide6. They test Python view
  models, dialogs and scenes and would be rewritten as Rust tests of the
  QObjects plus a smaller set of QML scene tests (`QtQuickTest` or a Rust
  harness). Accessibility checks on the scenes (`test_quick_accessibility.py`,
  113) could be kept if a scene-level harness is built.
- 1,524 tests in `test_headroom_native_parity.py` and most of the 755 in
  `test_signal_processing.py` compare native results with the NumPy reference.
  Without Python in the app they become fixture comparisons against stored
  NumPy outputs, or are retired.
- About 960 tests cover packaging, release, versioning, evaluation hygiene and
  bench tooling (`test_package_tools.py` alone has 590). They stay in Python
  but need new expectations for the bundle layout.
- The remaining tests exercise Python analysis and configuration logic
  (`test_auto_eq.py`, `test_voice_setup.py`, `test_config_migration.py`, ...)
  and are rewritten with the code they test. The preset migration tests carry
  over as JSON fixtures.

The Rust side has 826 `#[test]` functions, including the realtime source-scan
and no-allocation tests and the seeded stress test in `rust-core/tests/`. They
carry over unchanged.

## 5. Effort and risks

Estimates for one developer familiar with the code, in weeks:

| Work | Weeks | Driven by |
| --- | ---: | --- |
| QtWidgets hosts to QML (main window, tray, dialogs, message boxes) | 5 to 8 | 10,700 lines; tray and renderer-fallback behaviour; needed even under PySide6 |
| Qt-free and QtCore UI logic to Rust QObjects | 6 to 10 | 12,000 lines; `VoiceSetupFlow` and `EngineSession` lifecycles |
| Analysis to Rust (Voice Setup, Auto-EQ fit, spectra, noise reference) | 8 to 14 | 11,800 lines; FFT and LAPACK-style solver parity; bench reports per area |
| Configuration, presets, migrations, diagnostics export | 3 to 4 | 4,500 lines; preset schema compatibility |
| Platform glue (login startup, single instance, hotkeys, portals) | 2 to 3 | Windows COM and Linux D-Bus edge cases |
| Packaging, licences, CI | 3 to 5 | Qt SDK provisioning, windeployqt, path baselines, source manifest |
| Test rewrite | 4 to 6 | 2,245 Qt tests and the parity suites |
| Total | 31 to 50 | |

Roughly 7 to 12 months, with no user-visible feature in that time.

Risks:

- cxx-qt states it is in early development with frequent API changes. It
  shipped 0.8.0 (2025-12-18), 0.9.0 (2026-06-23) and 0.10.0 (2026-08-24): three
  breaking minor releases in nine months. A shell of this size would track each.
- Analysis parity. Rounding differences can flip Voice Setup refusals and
  Auto-EQ recommendations; each area becomes a tuning change with its own
  bench report, and the released evaluation evidence stops describing the
  shipped code.
- Two languages across the QML boundary become two plus C++ (cxx-qt generates
  C++ and needs a C++ toolchain), and the QML scene tests lose pytest's
  fixtures.
- The realtime engine gains nothing. Python is not on the callback, render or
  DSP-loop path today, so latency, CPU and audio output are unchanged by the
  switch.
- Gains are modest and measurable without the rewrite: about 60 MB of bundle,
  interpreter start-up, and the PyO3 binding layer (about 4,500 lines).

## 6. Recommendation

Not now. The cost is a 31 to 50 week rewrite with a tuning-change burden on
every analysis area, for a smaller bundle and one fewer language.

A spike that loads `LatencyCalibration.qml` from a cxx-qt binary is not
recommended either: that scene has 12 view-model members over an analysis that
is already native, so it would only confirm what the cxx-qt documentation
already shows. The hard parts are the QtWidgets hosts, the dynamic `ui.model`
bridge and analysis parity, and a two-week spike reaches none of them.

What would change the answer:

1. No module under `python/mic_eq/ui` imports QtWidgets or QQuickWidgets (the
   scenes run in a QML `ApplicationWindow` under PySide6). Do this first; it
   pays off regardless.
2. The Python left in the app is mostly view-model glue: analysis ported where
   parity holds, and the maintainer has accepted rounding-level parity with
   bench reports for the rest.
3. A product need Python cannot meet: start-up time, bundle size limits,
   antivirus flags on the PyInstaller bootloader, or a platform PyInstaller
   does not serve.
4. cxx-qt keeps its API across two consecutive minor releases, or reaches 1.0.

When conditions 1 and 2 hold, run a spike on `Main.qml` with `QuickBridge`
ported, the tray and one painted item (`EQGraphItem`), built and deployed with
windeployqt in CI. It succeeds when the window, tray and graph run from the
deployed folder on a clean Windows machine and the packaging smoke test passes.
Stop at three weeks, or earlier if the `ui.model` bridge needs more than one
generated QObject per proxy type.
