# Releasing AudioForge

## Windows release flow

### Automated workflow

The preferred release path is the `Release package` workflow.

Prepare source, version, release notes, and documentation in one release commit.
Keep the upcoming release notes in `release-notes/`; published notes remain on
[GitHub Releases](https://github.com/FueledByRedBull/audio-forge/releases) and in
their tagged source. The [changelog](CHANGELOG.md) keeps the repository history.
Keep artifact sizes, hashes, and file counts in the generated sidecars, so their
publication does not require a second documentation commit. The
`fallback_release_tag` is a standing source of dependency assets, independent of
the application version: leave it unchanged while those assets and hashes are
unchanged. Advance it only when a new dependency-asset source has been verified.
Do not move a released tag to reconcile documentation.

Publication requires hardware qualification and complete corresponding source.
The workflows below produce the exact candidate evidence and sidecars; a passing
hosted build cannot replace either gate.

### Release candidates

Release candidates use one canonical tag spelling: `vMAJOR.MINOR.PATCH-rc.N`.
The source version may use PEP 440 (`1.12.0rc1`); the release tooling maps it to
Cargo's `1.12.0-rc.1`, the canonical tag, and an artifact name containing that
tag. MSI has only three numeric version fields, so the release tooling reserves
the last value in each 100-value patch block for the final release: `1.12.0rc1`
maps to MSI `1.12.1`; the final `1.12.2` maps to MSI `1.12.299`. RC sequences are bounded
to 1 through 98 and patch components to 654. MSI's numeric value is an internal
installer identity; app and release metadata retain the public `1.12.2` version.
RC promotion creates or verifies a GitHub prerelease; final promotion rejects a
prerelease state.

Runtime binaries and models are intentionally not stored in Git. The asset
fetcher builds `df.dll` from the pinned DeepFilter recipe, downloads CPU-only
ONNX Runtime and Silero from their recorded upstream inputs, and obtains both
DeepFilter model archives from the standing asset-source release. Every input
is checked against `release-assets.json`. The model fallback can use raw assets
or extract the exact models from an existing portable archive.

For local release prep from a clean clone, you can mirror that behavior with:

```powershell
.\.venv\Scripts\python.exe python/tools/fetch_release_assets.py
```

Then run the workflow with:

- `release_tag`: the target tag, for example `v1.12.2`.
- `asset_source_tag`: optional published release override for pinned
  DeepFilter model assets. Leave blank to use the repository
  `AUDIOFORGE_ASSET_SOURCE_TAG` override when configured, then the
  `fallback_release_tag` pinned in `release-assets.json`. Silero v6.2.1 comes
  from its immutable direct URL.

On `v*` tag pushes, the workflow hydrates and verifies the corresponding-source
inputs before building, then builds and validates a Windows candidate. It
retains the portable archive, per-user MSI, corresponding-source archive, and
their generated checksums/metadata/manifests as one immutable Actions artifact.
The separately dispatched
`Qualify release candidate on hardware` workflow downloads those exact bytes
onto the labelled AudioForge Windows audio runner, verifies provenance, and
binds its measurements to the archive SHA-256. The v1.12.2 hardware gate
requires Windows 11 x64, USB microphone and virtual routes at observed 48 kHz,
an automated baseline of at least 1,800 seconds, and model switching/restoration.
Windows 10, analog or built-in input, 44.1 kHz, and physical reconnect,
default-device change, and sleep/resume cases remain compatibility targets or
unqualified cases; they are not v1.12.2 release verification claims or
mandatory gates. If those physical lifecycle cases are run, they still require
explicit operator observation and measured events. Coverage is never inferred
from a baseline and does not imply every device or combination was tested.
Publication is a third, explicit promotion step:
it downloads those same bytes and both qualification reports, verifies every
sidecar and report against the archive SHA-256, and uploads without rebuilding.
Promotion prepares a draft, verifies existing or newly uploaded assets by hash,
and publishes only after the complete asset set is present. A durable evidence
archive retains the package report, hardware matrix, and underlying case reports
alongside the portable/MSI artifacts. Set
`AUDIOFORGE_ASSET_SOURCE_TAG` when candidate builds should pull raw assets or
an existing package archive from a standing asset-source release. The workflow
still verifies all downloaded/extracted assets against `release-assets.json`
before packaging.

### Local fallback

Build the Rust extension with all configured features:

```powershell
.\.venv\Scripts\python.exe python/tools/fetch_release_assets.py
.\.venv\Scripts\python.exe -m maturin develop --release
```

Verify the source runtime assets. Stale files already under `dist/` are not valid packaging inputs:

```powershell
.\.venv\Scripts\python.exe python\tools\verify_release_assets.py
```

For a release candidate, hydrate the exact corresponding-source set and keep
the receipt tied to the tag commit before running `build_exe.ps1`:

```powershell
$env:AUDIOFORGE_SOURCE_DIR = "build/source-distribution"
$env:AUDIOFORGE_SOURCE_REVISION = (git rev-parse HEAD).Trim()
.\.venv\Scripts\python.exe python\tools\source_distribution.py bundle `
  --manifest licenses/source-manifest.json `
  --output $env:AUDIOFORGE_SOURCE_DIR `
  --revision $env:AUDIOFORGE_SOURCE_REVISION `
  --include-runtime-assets
```

Build the portable application from the checked-in PyInstaller spec:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```

This first rebuilds the native extension with locked Cargo resolution and
collects dependency notices, then reuses PyInstaller's analysis cache. Add
`-Clean` only when you need a cold PyInstaller rebuild.

Build an MSI from the resulting portable tree with `build_msi.ps1`. The MSI and
portable archive each receive checksum, metadata, and exact payload manifest
sidecars; the source archive receives a checksum and receipt metadata sidecar.
Validate extraction plus per-user installation/uninstallation with
`python/tools/msi_smoke.py`; the installed payload must match the portable files.

Run the release validation checks:

```powershell
.\.venv\Scripts\python.exe -m ruff check python/mic_eq python/tests python/tools
.\.venv\Scripts\python.exe -m pyright
.\.venv\Scripts\python.exe -m pytest python/tests -q
.\.venv\Scripts\python.exe -m pip_audit --require-hashes -r requirements/runtime.txt --disable-pip
.\.venv\Scripts\python.exe -m pip_audit --require-hashes -r requirements/dev.txt --disable-pip
.\.venv\Scripts\python.exe python\tools\run_semgrep.py --sarif semgrep-results.sarif
.\.venv\Scripts\python.exe python\tools\check_versions.py
.\.venv\Scripts\python.exe python\tools\check_workflows.py
.\.venv\Scripts\python.exe python\tools\package_smoke.py --source-only
cargo fmt --check
cargo audit
cargo test -p mic_eq_core
cargo test --release -p mic_eq_core --test stress_tests seeded_control_and_dsp_loops_remain_finite_under_contention
cargo test --release -p mic_eq_core audio::input::tests::benchmark_phase_safe_mono_callback_cost -- --ignored --nocapture
cargo test --release -p mic_eq_core dsp::biquad::tests::benchmark_biquad_morph_cost -- --ignored --nocapture
cargo clippy -p mic_eq_core --all-targets -- -D warnings
.\.venv\Scripts\python.exe python\tools\package_smoke.py
.\.venv\Scripts\python.exe python\tools\self_test.py
```

Create the distributable archive:

```powershell
& "C:\Program Files\7-Zip\7z.exe" a -t7z -mx=9 -m0=lzma2 -mmt=on -ms=on `
  .\AudioForge-v1.12.2-win64-ultra.7z .\dist\AudioForge\*
```

This setting is retained from a final-bundle comparison against ZIP/Deflate,
tar.gz, tar.xz, tar.zst, and solid LZMA. Solid LZMA2 with automatic BCJ2
filtering was smallest; the exact measurements are recorded in
`evaluation/archive-format-benchmark.json`.

Compute the checksum:

```powershell
Get-FileHash .\AudioForge-v1.12.2-win64-ultra.7z -Algorithm SHA256
```

For a real candidate, generate and verify all provenance sidecars instead of
writing release facts manually:

```powershell
.\.venv\Scripts\python.exe python\tools\release_provenance.py create `
  --bundle .\dist\AudioForge `
  --archive .\AudioForge-v1.12.2-win64-ultra.7z `
  --baseline .\evaluation\release-bundle-path-baseline.json `
  --output-dir .

.\.venv\Scripts\python.exe python\tools\release_provenance.py verify `
  --bundle .\dist\AudioForge `
  --archive .\AudioForge-v1.12.2-win64-ultra.7z `
  --checksum .\AudioForge-v1.12.2-win64-ultra.7z.sha256 `
  --manifest .\AudioForge-v1.12.2-win64-ultra.7z.manifest.json `
  --metadata .\AudioForge-v1.12.2-win64-ultra.7z.metadata.json `
  --baseline .\evaluation\release-bundle-path-baseline.json
```

Candidate and promotion:

1. Commit tracked source/doc/version changes.
2. Create annotated tag `v1.12.2`.
3. Confirm the standing runtime-asset source is still available and matches the
   manifest. Existing verified assets do not need uploading again for each version.
4. Push `master` and `v1.12.2`, or run the `Release package` workflow manually
   to create a candidate.
5. Record the candidate workflow run ID and generated archive SHA-256.
6. On a temporary or standing self-hosted runner labelled `self-hosted`,
   `windows`, `x64`, and `audioforge-hardware`, run `Qualify release candidate
   on hardware` with the candidate run ID and digest plus explicitly selected
   health/correlation routes for the release-qualified USB microphone and
   virtual-route cases at observed 48 kHz. Run the automated `baseline` and
   `model_configuration_change` scenarios, including model switching and
   restoration. The workflow refuses a health duration below 1,800 seconds and
   uploads a privacy-safe digest-bound report. Windows 10, analog or built-in
   input, 44.1 kHz, and physical lifecycle cases may be collected as
   compatibility evidence but are not mandatory v1.12.2 gates.
7. Run `Assemble release hardware gate` with all qualification run IDs. It checks
   the producing workflows, revisions, source-report hashes, and required
   release-qualified OS/device/rate/scenario coverage. An incomplete matrix
   blocks publication.
8. Run `Promote release candidate` with the candidate workflow run ID,
   hardware-gate workflow run ID, release tag, and approved archive SHA-256.
   Promotion downloads the candidate plus both qualification reports, verifies
   the evidence against the tag commit and digest, requires completed source
   distribution, and publishes the draft only after verifying all uploaded bytes.

## Packaging notes

- `AudioForge.spec` is the canonical package definition.
- Packaged builds register canonical bundled DeepFilter paths. Ambient paths stay disabled unless `AUDIOFORGE_ALLOW_EXTERNAL_DF=1` deliberately enables an external override.
- Install `requirements/dev.txt` with `--require-hashes`; do not release from an environment resolved directly from open-ended `pyproject.toml` constraints.
- Review every Semgrep warning in the generated SARIF. The CI gate fails reviewed ERROR-severity findings, while warning-level FFI and process-boundary findings require human triage.
- A clean `cargo audit` is mandatory; do not add RustSec ignores merely to make a release pass.
- Routine Python and Rust Dependabot version PRs are disabled; Dependabot
  security updates remain enabled. Dependency refreshes must be scoped
  independently and pass the applicable build, benchmark, hardware, and
  package gates instead of arriving as lockfile batches.
- Keep `release-assets.json` current with the source-built `df.dll`, CPU-only ONNX Runtime DLLs, both DeepFilter model tarballs, and `models/silero_vad.onnx`.
- `build_exe.ps1` fails before PyInstaller if a required asset is missing or hash
  mismatched, or the current-source native rebuild fails. File timestamps are not
  source provenance. Reduced builds without models are not a supported edition.
- `python/tools/package_smoke.py` verifies exact bundled DLL/model/native-extension and license-notice presence, rejects duplicate top-level native-extension payloads, and rejects a stale bundle-version manifest.
- `python/tools/prune_bundle.py` must not remove dependency `.dist-info` directories; license/metadata retention is part of the release gate. It may remove duplicate native-extension payloads only when the canonical `_internal/mic_eq/mic_eq_core*.pyd` copy is present.
- AudioForge targets Windows 10 and Windows 11 for compatibility and relies on
  the system UCRT.
  Microsoft documents the UCRT as an operating-system component on Windows 10
  and later, states that the system copy is always used on Windows 10/11, and
  does not recommend local deployment for performance and security reasons:
  <https://learn.microsoft.com/en-us/cpp/windows/universal-crt-deployment>.
  `prune_bundle.py` removes app-local `ucrtbase.dll` and `api-ms-win-*.dll`;
  package smoke must fail if they return.
- `evaluation/release-bundle-path-baseline.json` controls reviewed bundle path
  additions/removals. Binary hashes are recorded for provenance, but are not
  treated as reproducible-build expectations.
- The release profile strips native symbols without changing optimization level. The package spec excludes only unused SciPy namespaces, while the prune step removes unused Qt SVG payloads; keep both NumPy/SciPy BLAS DLLs, `opengl32sw.dll`, all required models, CPU-only ONNX Runtime, and df.dll because they are runtime dependencies.
- CI audits both Cargo graphs: the application lockfile and the independent
  `build-support/deepfilter/Cargo.lock` graph used for `df.dll`. The DeepFilter
  audit ignores only its two reviewed unmaintained-crate notices
  (`RUSTSEC-2024-0436` and `RUSTSEC-2024-0370`); vulnerability findings remain
  release blockers.
- Obtain CPU-only ONNX Runtime and model files from the exact upstream
  package/blob identities in `release-assets.json`. Build `df.dll` with the
  pinned recipe and retain its per-build attestation; matching source and
  settings do not by themselves promise byte-identical output across machines.
- Original AudioForge source stays MIT; combined PyQt6 distributions use GPLv3.
  Before final publication, complete the corresponding-source arrangements and
  component review in `licenses/THIRD_PARTY_NOTICES.md`. Generated license
  inventories alone do not fulfill corresponding-source requirements.

## Strict realtime regression gates

- The CPAL input callback, CPAL output callback, and post-initialization DSP loop are strict RT regions. They must not use blocking locks, `try_lock`, formatting/logging, vector growth APIs, or Vec-returning suppressor convenience APIs.
- Keep the RT source-scan tests passing whenever code inside a marked `RT_REGION_*` block changes.
- Keep control changes flowing through atomic snapshots or bounded queues; model loading and suppressor construction must remain outside the RT loop.
- Release validation must include fixed-buffer overflow/drop diagnostics checks and a model-discovery smoke pass proving bundled DeepFilter and Silero assets are preferred over CWD/user-directory assets unless an explicit override is set.
- `release-assets.json` paths and bundle paths must stay repository-relative and must not contain `..` traversal.
