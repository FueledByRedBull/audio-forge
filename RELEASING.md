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

GitHub release immutability is enabled for future publications. Assemble and
verify every asset on the draft before publishing; publication locks the assets
and tag and creates GitHub's cryptographically verifiable release attestation.
Existing releases retain their original protection status. See
[GitHub's release verification instructions](https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/secure-your-dependencies/verify-release-integrity).

Publication requires exact-artifact software/package validation and complete
corresponding source. Hardware measurements are optional supporting evidence;
release notes must identify their tested revision and any untested coverage.
A self-hosted runner is not required to publish.

### Release candidates

Release candidates use one canonical tag spelling: `vMAJOR.MINOR.PATCH-rc.N`.
The source version may use PEP 440 (`1.12.0rc1`); the release tooling maps it to
Cargo's `1.12.0-rc.1`, the canonical tag, and an artifact name containing that
tag. MSI has only three numeric version fields, so the release tooling reserves
the last value in each 100-value patch block for the final release: `1.12.0rc1`
maps to MSI `1.12.1`; the final `1.12.0` maps to MSI `1.12.99`. RC sequences are bounded
to 1 through 98 and patch components to 654. MSI's numeric value is an internal
installer identity; app and release metadata retain the public `1.12.0` version.
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

- `release_tag`: leave blank to validate the selected branch before tagging;
  supply an existing tag only when rebuilding it.
- `asset_source_tag`: optional published release override for pinned
  DeepFilter model assets. Leave blank to use the repository
  `AUDIOFORGE_ASSET_SOURCE_TAG` override when configured, then the
  `fallback_release_tag` pinned in `release-assets.json`. Silero v6.2.1 comes
  from its immutable direct URL.

On manual dispatch, the workflow hydrates and verifies the corresponding-source
inputs before building, then builds and validates a Windows candidate. It
retains the portable archive, per-user MSI, corresponding-source archive, and
their generated checksums/metadata/manifests as one immutable Actions artifact
for three days. Promote within that window; expired candidates must be rebuilt
and validated before promotion.
Hardware qualification workflows can collect additional measurements when a
suitable runner is available. They are optional and do not block publication.
Never attribute measurements from an earlier candidate to the final binary.
Publication downloads the same candidate bytes and automated qualification
report, verifies sidecars and report against the archive SHA-256, and uploads
without rebuilding. Promotion prepares a draft, verifies uploaded assets by
hash, and publishes only after the complete asset set is present. A durable
evidence archive retains the package report and all checksum, metadata, manifest,
and native-provenance sidecars. Publish five files: portable app, MSI,
corresponding source, evidence archive, and one `SHA256SUMS.txt` file.
For reruns, preflight compares GitHub's SHA-256 digests; the final verification
still downloads and hashes every published file. Set
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

The `Release package` workflow owns archive creation and provenance commands.
Its solid LZMA2 settings come from `evaluation/archive-format-benchmark.json`.

Candidate and promotion:

1. Commit and push tracked source/doc/version changes.
2. Confirm the standing runtime-asset source is still available and matches the
   manifest. Existing verified assets do not need uploading again for each version.
3. Dispatch `Release package` on that branch with `release_tag` blank. Both
   build and exact-archive validation must pass. Fix failed attempts on the
   branch without creating tags or changing the release version.
4. Record the successful candidate run ID, source commit, and archive SHA-256.
5. Create and push annotated tag `v1.12.1` at that exact source commit. Tag
   pushes do not rebuild the candidate; subsequent gates use the same bytes.
6. Review available hardware evidence and describe untested configurations in
   the release notes. Hardware runs are optional; retain the candidate revision
   and digest with any measurements rather than implying broader coverage.
7. Run `Promote release candidate` with the candidate workflow run ID, release
   tag, and approved archive SHA-256. Promotion verifies the automated evidence,
   source distribution, and uploaded bytes before publishing. Release notes come
   from the selected workflow revision; the binary remains bound to the tag.

## Packaging notes

- `licenses/source-manifest.json` is generated by
  `python/tools/source_distribution.py manifest` from the pinned dependency and
  runtime-asset inputs. Regenerate it when those inputs change; review the
  resulting source closure instead of hand-editing generated entries.
- `build-support/deepfilter/provenance.json` owns the static build recipe.
  `build_deepfilter.ps1` writes the actual toolchain and output identity to
  `target/deepfilter/df.dll.provenance.json`. Retain that per-build attestation
  with its DLL; the recipe alone does not attest to a particular binary.
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

## Local build cleanup

Inspect exact output paths before deleting them. Completed `dist/` packages and
Rust compilation outputs are disposable only after required candidates and
unpublished evidence have been retained elsewhere. Prefer the build tool's
scoped cleanup command for its own outputs.

Do not delete `target/` or `build/` wholesale: they may also contain the active
Python installation, hydrated runtime DLLs, source receipts, and unpublished
measurements. Preserve virtual environments, `models/`, corpora, and any inputs
needed to reproduce retained evidence. Check resolved paths and directory links;
an ignored path is not proof that its contents are disposable.

## Strict realtime regression gates

- The CPAL input callback, CPAL output callback, and post-initialization DSP loop are strict RT regions. They must not use blocking locks, `try_lock`, formatting/logging, vector growth APIs, or Vec-returning suppressor convenience APIs.
- Keep the RT source-scan tests passing whenever code inside a marked `RT_REGION_*` block changes.
- Keep control changes flowing through atomic snapshots or bounded queues; model loading and suppressor construction must remain outside the RT loop.
- Release validation must include fixed-buffer overflow/drop diagnostics checks and a model-discovery smoke pass proving bundled DeepFilter and Silero assets are preferred over CWD/user-directory assets unless an explicit override is set.
- `release-assets.json` paths and bundle paths must stay repository-relative and must not contain `..` traversal.
