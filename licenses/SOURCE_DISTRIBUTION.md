# Corresponding source distribution

`licenses/source-manifest.json` records the exact source archives and SHA-256
digests needed to reconstruct the AudioForge Windows build. It is generated from the
locked Python requirements, the resolved Windows Cargo graph, the Qt source
corresponding to the bundled PySide6 Qt libraries, the CPython interpreter, and
the pinned Windows-only source package.

The manifest status is authoritative: `complete` means every recorded source,
recipe file, and hydratable runtime asset identity is present and verified;
`incomplete` lists blockers and cannot pass release verification. A source-built
runtime output such as `df.dll` is derived from the included source and recipe,
so its candidate-specific binary is checked by its build attestation instead of
being copied into the corresponding-source receipt. The release process must
replace any inherited binary whose build identity cannot be reproduced before
publishing a complete source bundle.

Recipe text digests use canonical LF content, so a Windows checkout's CRLF
conversion cannot make the recorded recipe differ from the tagged source.

## Generate and hydrate the manifest

Run these commands from the repository root with the locked Python environment:

```powershell
.\.venv\Scripts\python.exe python/tools/source_distribution.py manifest `
  --output licenses/source-manifest.json
.\.venv\Scripts\python.exe python/tools/source_distribution.py download `
  --manifest licenses/source-manifest.json `
  --output build/source-distribution `
  --include-runtime-assets
.\.venv\Scripts\python.exe python/tools/source_distribution.py verify `
  --manifest licenses/source-manifest.json `
  --source-dir build/source-distribution `
  --include-runtime-assets
```

`download` skips blocked entries, source-built outputs, and runtime assets by
default. Use `--include-runtime-assets` when the release source bundle must
also carry the pinned CPU ONNX Runtime archive and model blobs listed in
`release-assets.json`; source-built outputs remain represented by their source
archives and build attestations. The runtime entry verifies the archive digest
and records each extracted DLL digest separately.

`--require-receipt` is used for release promotion after `bundle`; it checks
that the hydrated archive list, manifest digest, and resolved revision match
`source-receipt.json`.

Set `AUDIOFORGE_SOURCE_DIR` to the hydrated directory and
`AUDIOFORGE_SOURCE_REVISION` to the release tag before running the package
builder. The dependency inventory then reports `complete` only after this
receipt check; without both variables it reports `pending` with an explicit
blocker.

Create the release bundle from the tagged, committed revision selected above
in `AUDIOFORGE_SOURCE_REVISION`:

```powershell
.\.venv\Scripts\python.exe python/tools/source_distribution.py bundle `
  --manifest licenses/source-manifest.json `
  --output build/source-distribution `
  --revision $env:AUDIOFORGE_SOURCE_REVISION `
  --include-runtime-assets
```

The command fails if the manifest has blockers, a downloaded digest is missing
or different, the hydration directory contains stale archives, or the selected
revision is dirty or has a different project version. It writes the exact
dependency archives below `build/source-distribution/archives/` and a
`AudioForge-project-source.tar` archive made with `git archive` for the selected
revision, plus `source-receipt.json` binding the hydrated archive digests to the
resolved revision and manifest digest. Run it after committing the generator,
manifest, license notices, and release recipe so the project archive includes
those files.

## Rebuilding the locked stack

The manifest currently covers:

* CPython 3.13.15 and the exact Windows source-deps selected by
  `PCbuild/get_externals.bat`, plus Python package sources for NumPy 2.5.1, SciPy 1.18.0,
  PySide6-Essentials, PySide6-Addons and shiboken6 6.11.1, PyInstaller 6.21.0,
  and pywin32 311.
  The CPython source entry retains both its main license and `PC/crtlicense.txt`
  for the Microsoft runtime DLLs bundled by the Windows distribution.
* Qt for Python's shared `pyside-setup-everywhere-src-6.11.1.tar.xz` source
  archive for all three binding packages; they publish wheels without PyPI
  source distributions.
* Qt base 6.11.1, which supplies the Core, Gui, Network, Widgets, and Windows
  platform plugin libraries present in the PySide6 runtime bundle, plus Qt's
  declarative (QML and Qt Quick), image-format and multimedia module/plugin
  sources.
* CPU ONNX Runtime 1.23.2 source, its pinned repository submodules, and the
  CPU FetchContent sources from `cmake/deps.txt` and
  `onnxruntime_external_deps.cmake`. GPU, WebGPU, training, and benchmark-only
  inputs are excluded because this release enables only the CPU provider.
* The CPython external dependency archives retained by the official 3.13.15
  source graph, including the OpenSSL 3.0.21 source input, the pinned
  NumPy/SciPy OpenBLAS recipes and upstream sources, and Mesa 11.2.2 for Qt's
  `opengl32sw.dll`. The portable package may exclude unused `libcrypto` and
  `libssl` payloads when the packaging rules prove they are not loaded.
* The pinned DeepFilterNet source, its resolved Cargo graph, the recorded
  `tract-linalg` patch, and the locked Windows C API build recipe under
  `build-support/deepfilter/` and `build_deepfilter.ps1`.
* Every registry crate resolved by `cargo metadata --locked` for the Windows
  `extension-module` build, with checksums taken from `Cargo.lock`.

Use CPython 3.13.15 x64, Rust 1.94.0, the archive digests in the manifest,
and the locked requirements. Build the native extension with:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release --locked
```

Build PySide6 and shiboken6 from the pinned Qt for Python source archive with
the corresponding Qt module sources, following its included instructions and
the [Qt for Python build guide](https://doc.qt.io/qtforpython-6/building_from_source/index.html).
Build pywin32 from the `b311` source tag;
PyPI publishes wheels for this package rather than a source distribution.
Retain the upstream license files and generated dependency inventory alongside
the source archive. AudioForge's original source remains MIT; the bundled
Qt for Python and selected Qt modules use LGPLv3, with each dependency's
terms retained as documented in `licenses/THIRD_PARTY_NOTICES.md`.

The official licensing references are
[Qt for Python licenses](https://doc.qt.io/qtforpython-6/licenses.html),
[Qt's LGPL obligations](https://www.qt.io/development/open-source-lgpl-obligations),
[Qt licensing](https://doc.qt.io/qt-6/licensing.html), and
[Qt's third-party license list](https://doc.qt.io/qt-6/licenses-used-in-qt.html).

## Replacing the Qt libraries

AudioForge loads Qt dynamically and does not prohibit modification or reverse
engineering for debugging modifications to those libraries. Close AudioForge
and make a separate copy of the complete portable directory before testing
modified libraries. In that copy, replace the applicable `Qt6*.dll` files in
`_internal/PySide6/` and their matching plugins under
`_internal/PySide6/plugins/`. Use a compatible Windows x64 Qt build with the
same module configuration and ABI; keep a matching set of libraries and
plugins. A different Python binding ABI requires rebuilding PySide6/shiboken6
and the package, rather than replacing Qt DLLs alone.

Run `AudioForge.exe --smoke-test` in the disposable copy, then test normal
startup and the features affected by the modified libraries. Keep the original
copy for recovery. The release's corresponding-source archive records the
exact Qt and Qt for Python sources and build instructions, and the complete
license texts and component inventory remain under `_internal/licenses/`.
An MSI installs the same portable layout; perform replacement testing in a
separate copy so an installer repair does not overwrite your changes.

## DeepFilter build provenance

`build_deepfilter.ps1` verifies the pinned upstream commit, model archives,
Cargo lock, local `tract-linalg` patch, compiler target, required exports, and
the output digest before the DLL is copied into a release build. The provenance
record also carries the deterministic parity and latency results used to accept
the replacement. Regenerate the manifest after changing this recipe and run
`verify` without `--allow-incomplete` before release promotion.
