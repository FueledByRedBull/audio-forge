# Releasing AudioForge

A release is built once by the `Release package` workflow, then promoted
without rebuilding. The public release has exactly five files: the portable
archive, the per-user MSI, the corresponding-source archive, an evidence
archive, and one `SHA256SUMS.txt`.

## Automated workflow

1. **Prepare one release commit.** Update the version, `CHANGELOG.md`, the
   upcoming `release-notes/release-notes-v<version>.md`, and documentation.
   Run `python/tools/check_versions.py`. Published notes live on
   [GitHub Releases](https://github.com/FueledByRedBull/audio-forge/releases)
   and in their tagged source, so only the upcoming notes stay in
   `release-notes/`. Don't write archive sizes, hashes, or file counts into
   docs; the generated checksum and metadata sidecars own them.
2. **Build a candidate.** Dispatch `Release package` on the branch with
   `release_tag` blank. It hydrates and verifies the runtime assets and
   corresponding source, builds the portable app and MSI, and validates the
   exact archive. Fix failures on the branch without tagging or changing the
   version. Set `asset_source_tag` only to override where model assets come
   from (see [Runtime assets](#runtime-assets)). Supply an existing tag in
   `release_tag` only when rebuilding that tag.
3. **Record the candidate:** workflow run ID, source commit, and archive
   SHA-256. The candidate artifact expires after three days; an expired
   candidate must be rebuilt and revalidated.
4. **Tag the candidate commit** with an annotated `v<version>` tag. A tag push
   doesn't rebuild anything. Never move a released tag to fix documentation;
   reconcile documentation in a later commit.
5. **Review hardware evidence (optional).** Hardware qualification workflows
   can add measurements when a suitable runner exists; they don't block
   publication. Keep each measurement bound to its candidate revision and
   digest, never attribute an earlier candidate's results to the final
   binary, and name untested configurations in the release notes.
6. **Promote.** Run `Promote release candidate` with `release_tag`,
   `candidate_run_id`, and `expected_archive_sha256`. It downloads the same
   candidate bytes, verifies sidecars, evidence, and corresponding source
   against that digest, assembles a draft, and publishes only when all five
   files are present and verified. Release notes come from the workflow
   revision; the binary stays bound to the tag. Reruns compare GitHub's
   SHA-256 digests in preflight, and the final check downloads and hashes
   every published file.

GitHub release immutability is enabled: publication locks the assets and tag
and creates GitHub's
[release attestation](https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/secure-your-dependencies/verify-release-integrity).
Verify every asset on the draft before publishing. A self-hosted runner isn't
required.

## Versioning

Final tags are `vMAJOR.MINOR.PATCH`; release candidates are
`vMAJOR.MINOR.PATCH-rc.N`. The source version may use PEP 440 (`1.12.0rc1`);
`python/tools/release_version.py` maps it to Cargo's `1.12.0-rc.1`, the tag,
and the artifact name. MSI has only three numeric fields, so each 100-value
patch block reserves its last value for the final release: `1.12.0rc1` is MSI
`1.12.1` and final `1.12.0` is MSI `1.12.99`. RC numbers run 1–98 and patch
components up to 654. The MSI number is an installer identity only; the app
and release metadata keep the public version. RC promotion creates or verifies
a GitHub prerelease; final promotion rejects one.

## Runtime assets

Runtime binaries and models are never committed. `release-assets.json` owns
their paths, sizes, hashes, origins, and licenses, and
`python/tools/fetch_release_assets.py` hydrates them:

- `df.dll` is built from the pinned DeepFilter recipe
  (`build-support/deepfilter/`, `build_deepfilter.ps1`). The recipe is not an
  attestation; keep the per-build `target/deepfilter/df.dll.provenance.json`
  with its DLL. Matching source and settings don't promise byte-identical
  output across machines.
- CPU-only ONNX Runtime and Silero v6.2.1 come from their recorded upstream
  identities.
- Both DeepFilter model archives come from the standing asset-source release:
  the workflow's `asset_source_tag` input, else the repository
  `AUDIOFORGE_ASSET_SOURCE_TAG` variable, else `fallback_release_tag` in
  `release-assets.json`. The fallback can supply raw assets or extract the
  exact models from an existing portable archive.

Every downloaded or extracted asset is verified against `release-assets.json`.
Leave `fallback_release_tag` unchanged while the dependency assets are
unchanged; advance it only for a verified new asset source.

## Local build

From a clean clone with the hashed development environment:

```powershell
.\.venv\Scripts\python.exe python/tools/fetch_release_assets.py
.\.venv\Scripts\python.exe -m maturin develop --release --locked
.\.venv\Scripts\python.exe python\tools\verify_release_assets.py
```

Stale files under `dist/` are never valid packaging inputs. For a release
candidate, hydrate the corresponding source for the exact commit first (see
[licenses/SOURCE_DISTRIBUTION.md](licenses/SOURCE_DISTRIBUTION.md)):

```powershell
$env:AUDIOFORGE_SOURCE_DIR = "build/source-distribution"
$env:AUDIOFORGE_SOURCE_REVISION = (git rev-parse HEAD).Trim()
.\.venv\Scripts\python.exe python\tools\source_distribution.py bundle `
  --manifest licenses/source-manifest.json `
  --output $env:AUDIOFORGE_SOURCE_DIR `
  --revision $env:AUDIOFORGE_SOURCE_REVISION `
  --include-runtime-assets
```

Then build the portable tree from `AudioForge.spec` and the MSI from it:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
powershell -ExecutionPolicy Bypass -File .\build_msi.ps1
```

`build_exe.ps1` verifies runtime assets, rebuilds the native extension with
locked Cargo resolution, collects dependency notices, and reuses PyInstaller's
analysis cache (`-Clean` forces a cold PyInstaller build). It fails before
PyInstaller on a missing or mismatched asset or a failed native rebuild.
Builds without models aren't a supported edition. The MSI payload must match
the portable tree; `python/tools/msi_smoke.py` checks extraction and per-user
install and uninstall. Archive creation and provenance commands belong to the
workflow; its solid LZMA2 settings come from
`evaluation/archive-format-benchmark.json`.

## Release checks

Run the [development checks](CONTRIBUTING.md), including the release-mode
stress test, then:

```powershell
.\.venv\Scripts\python.exe -m pip_audit --require-hashes -r requirements/runtime.txt --disable-pip
.\.venv\Scripts\python.exe -m pip_audit --require-hashes -r requirements/dev.txt --disable-pip
.\.venv\Scripts\python.exe python\tools\run_semgrep.py --sarif semgrep-results.sarif
.\.venv\Scripts\python.exe python\tools\check_versions.py
.\.venv\Scripts\python.exe python\tools\check_workflows.py
.\.venv\Scripts\python.exe python\tools\package_smoke.py --source-only
cargo audit
.\.venv\Scripts\python.exe python\tools\package_smoke.py
.\.venv\Scripts\python.exe python\tools\self_test.py
```

- **Semgrep:** CI fails reviewed ERROR findings; triage every WARNING (FFI and
  process-boundary findings) by hand in the SARIF.
- **RustSec:** `cargo audit` must be clean for the application lockfile; never
  add an ignore just to release. The separate
  `build-support/deepfilter/Cargo.lock` graph ignores only two reviewed
  unmaintained-crate notices (`RUSTSEC-2024-0436`, `RUSTSEC-2024-0370`);
  vulnerabilities there still block.
- **Python:** install `requirements/dev.txt` with `--require-hashes`; never
  release from an environment resolved from open `pyproject.toml` ranges.
  Neither the runtime nor the development audit has ignores; Semgrep 1.179.0
  allows a patched PyJWT, so the former PyJWT exception is gone.
- **Dependabot:** routine version PRs are off; security updates stay on.
  Scope each dependency refresh on its own and pass the build, benchmark,
  hardware, and package gates.

## Packaging invariants

- `AudioForge.spec` is the package definition. `package_smoke.py` checks the
  exact bundled DLLs, models, native extension, and license notices, rejects
  duplicate top-level native-extension payloads, and rejects a stale
  bundle-version manifest.
- `evaluation/release-bundle-path-baseline.json` gates reviewed bundle path
  additions and removals. Its binary hashes are provenance, not
  reproducible-build expectations.
- `prune_bundle.py` keeps every dependency `.dist-info` directory, removes a
  duplicate native-extension payload only when the canonical
  `_internal/mic_eq/mic_eq_core*.pyd` exists, and removes app-local
  `ucrtbase.dll` and `api-ms-win-*.dll`: Windows 10 and 11 always use the
  [system UCRT](https://learn.microsoft.com/en-us/cpp/windows/universal-crt-deployment).
  Package smoke fails if they return.
- Keep both NumPy/SciPy BLAS DLLs, `opengl32sw.dll`, all models, the CPU-only
  ONNX Runtime, and `df.dll`. The spec excludes only unused SciPy namespaces
  and the prune step removes unused Qt SVG payloads. The release profile
  strips native symbols without changing optimization.
- Bundled DeepFilter and Silero assets take precedence over the working
  directory and user paths. External DeepFilter paths require
  `AUDIOFORGE_ALLOW_EXTERNAL_DF=1`. Runtime-asset and bundle paths stay
  repository-relative without `..`.
- Release validation covers fixed-buffer overflow and drop diagnostics and
  model-discovery smoke checks.
- `licenses/source-manifest.json` is generated by
  `source_distribution.py manifest`; regenerate it when its inputs change and
  review the source closure instead of hand-editing entries.
- Original AudioForge source stays MIT; the combined PyQt6 distribution is
  GPLv3. Complete the corresponding-source arrangements and component review
  in [licenses/THIRD_PARTY_NOTICES.md](licenses/THIRD_PARTY_NOTICES.md) before
  publishing; a generated license inventory alone doesn't satisfy them.

## Local build cleanup

Check exact output paths before deleting. Completed `dist/` packages and Rust
build outputs are disposable only after required candidates and unpublished
evidence are kept elsewhere; prefer each tool's own clean command. Don't
delete `target/` or `build/` wholesale: they can hold the project Python, the
hydrated runtime DLLs, source receipts, and unpublished measurements. Keep
virtual environments, `models/`, and corpora.
