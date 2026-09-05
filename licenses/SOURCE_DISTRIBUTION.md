# Corresponding source distribution

`licenses/source-manifest.json` records the exact source archives and SHA-256
digests needed to reconstruct the v2.0 Windows build. It is generated from the
locked Python requirements, the resolved Windows Cargo graph, the Qt source
corresponding to the bundled PyQt6 Qt libraries, the CPython interpreter, and
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
  --output build/source-distribution
.\.venv\Scripts\python.exe python/tools/source_distribution.py verify `
  --manifest licenses/source-manifest.json `
  --source-dir build/source-distribution `
  --allow-incomplete
```

`download` skips blocked entries, source-built outputs, and runtime assets by
default. `verify` requires `--allow-incomplete` while any release blocker,
such as restricted DirectML terms, remains. Use `--include-runtime-assets` when
the release source bundle must also carry the verified model blobs listed in
`release-assets.json`; source-built outputs remain represented by their source
archives and build attestations. For a packaged asset such as DirectML, the
manifest verifies the downloaded package digest and records the extracted DLL
digest separately, but a restricted asset remains blocked from a complete
release manifest.

`--require-receipt` is used for release promotion after `bundle`; it checks
that the hydrated archive list, manifest digest, and resolved revision match
`source-receipt.json`.

Set `AUDIOFORGE_SOURCE_DIR` to the hydrated directory and
`AUDIOFORGE_SOURCE_REVISION` to the release tag before running the package
builder. The dependency inventory then reports `complete` only after this
receipt check; without both variables it reports `pending` with an explicit
blocker.

Create the release bundle from the tagged, committed revision:

```powershell
.\.venv\Scripts\python.exe python/tools/source_distribution.py bundle `
  --manifest licenses/source-manifest.json `
  --output build/source-distribution `
  --revision v2.0.0 `
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

* CPython 3.12.10 and Python package sources for NumPy 2.5.1, SciPy 1.18.0,
  PyQt6 6.11.0, PyQt6-sip 13.11.1, PyInstaller 6.21.0, and pywin32 311.
* PyQt6's declared build sources, sip 6.16.1 and PyQt-builder 1.19.1.
* Qt base 6.11.1, which supplies the Core, Gui, Network, Widgets, and Windows
  platform plugin libraries present in the PyQt6 runtime bundle, plus Qt's
  image-format plugin sources.
* OpenSSL 3.0.16 and the CPython external dependency archives corresponding to
  the bundled `libcrypto`/`libssl`, the pinned NumPy/SciPy OpenBLAS recipes and
  upstream sources, and Mesa 11.2.2 for Qt's `opengl32sw.dll`.
* The pinned DeepFilterNet source, its resolved Cargo graph, the recorded
  `tract-linalg` patch, and the locked Windows C API build recipe under
  `build-support/deepfilter/` and `build_deepfilter.ps1`.
* Every registry crate resolved by `cargo metadata --locked` for the Windows
  `extension-module` build, with checksums taken from `Cargo.lock`.

Use CPython 3.12.10 x64, Rust 1.94.0, the archive digests in the manifest,
and the locked requirements. Build the native extension with:

```powershell
maturin develop --release --locked
```

Build PyQt6 from its sdist with the pinned sip and PyQt-builder sources and
the corresponding Qt module sources. Build pywin32 from the `b311` source tag;
PyPI publishes wheels for this package rather than a source distribution.
Retain the upstream license files and generated dependency inventory alongside
the source archive. AudioForge's original source remains MIT; the combined
PyQt6 application is distributed under GPLv3 as documented in
`licenses/THIRD_PARTY_NOTICES.md`.

The official licensing references are [GNU GPLv3 rationale](https://www.gnu.org/licenses/gpl3-final-rationale.pdf),
[Riverbank's PyQt overview](https://www.riverbankcomputing.com/software/pyqt/),
[Riverbank's PyQt downloads](https://riverbankcomputing.com/software/pyqt/download),
[Qt licensing](https://doc.qt.io/qt-6/licensing.html), and
[Qt's third-party license list](https://doc.qt.io/qt-6/licenses-used-in-qt.html).

## DeepFilter build provenance

`build_deepfilter.ps1` verifies the pinned upstream commit, model archives,
Cargo lock, local `tract-linalg` patch, compiler target, required exports, and
the output digest before the DLL is copied into a release build. The provenance
record also carries the deterministic parity and latency results used to accept
the replacement. Regenerate the manifest after changing this recipe and run
`verify` without `--allow-incomplete` before release promotion.
