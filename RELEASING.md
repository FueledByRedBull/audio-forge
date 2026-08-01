# Releasing AudioForge

## Windows release flow

### Automated workflow

The preferred release path is the `Release package` workflow.

Because runtime binaries and models are intentionally not stored in Git, make these assets available on a GitHub Release first:

- `df.dll`
- `DirectML.dll`
- `DeepFilterNet3_ll_onnx.tar.gz`
- `DeepFilterNet3_onnx.tar.gz`
- `silero_vad.onnx`

Raw assets are preferred. If the asset-source release only has an existing `AudioForge-*-win64-ultra.7z`, the workflow can extract those same runtime assets from that archive and still verifies them against `release-assets.json` before packaging.

For local release prep from a clean clone, you can mirror that behavior with:

```powershell
.\.venv\Scripts\python.exe python/tools/fetch_release_assets.py
```

Then run the workflow with:

- `release_tag`: the target tag, for example `v1.11.0`.
- `asset_source_tag`: optional published release override for pinned
  DeepFilter/DirectML assets. Leave blank to use the repository
  `AUDIOFORGE_ASSET_SOURCE_TAG` override when configured, then the
  `fallback_release_tag` pinned in `release-assets.json`. Silero v6.2.1 comes
  from its immutable direct URL.

On `v*` tag pushes, the workflow builds and validates a Windows candidate and
retains the archive plus generated checksum, metadata, and per-file manifest
as one immutable Actions artifact. The separately dispatched
`Qualify release candidate on hardware` workflow downloads those exact bytes
onto the labelled AudioForge Windows audio runner, verifies provenance, and
binds selected-route plus 30-minute physical-hardware evidence to the archive
SHA-256. Publication is a third, explicit promotion step: it downloads those
same bytes and both qualification reports, verifies every sidecar and report
against the archive SHA-256, and uploads without rebuilding. Set
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

Build the portable application from the checked-in PyInstaller spec:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```

This reuses PyInstaller's analysis cache for faster repeat builds. Add `-Clean` only when you need a cold PyInstaller rebuild.

Run the release validation checks:

The current Semgrep release pins `mcp==1.23.3` for its optional MCP server;
AudioForge only invokes `semgrep scan`, so the three upstream MCP advisories
are listed explicitly below until Semgrep publishes a compatible pin. Runtime
dependencies remain unignored.

```powershell
.\.venv\Scripts\python.exe -m ruff check python/mic_eq python/tests python/tools
.\.venv\Scripts\python.exe -m pyright
.\.venv\Scripts\python.exe -m pytest python/tests -q
.\.venv\Scripts\python.exe -m pip_audit --require-hashes -r requirements/runtime.txt
.\.venv\Scripts\python.exe -m pip_audit --require-hashes -r requirements/dev.txt `
  --ignore-vuln PYSEC-2026-3481 `
  --ignore-vuln PYSEC-2026-3482 `
  --ignore-vuln PYSEC-2026-3483
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
  .\AudioForge-v1.11.0-win64-ultra.7z .\dist\AudioForge\*
```

This setting is retained from a final-bundle comparison against ZIP/Deflate,
tar.gz, tar.xz, tar.zst, and solid LZMA. Solid LZMA2 with automatic BCJ2
filtering was smallest; the exact measurements are recorded in
`evaluation/archive-format-benchmark.json`.

Compute the checksum:

```powershell
Get-FileHash .\AudioForge-v1.11.0-win64-ultra.7z -Algorithm SHA256
```

For a real candidate, generate and verify all provenance sidecars instead of
writing release facts manually:

```powershell
.\.venv\Scripts\python.exe python\tools\release_provenance.py create `
  --bundle .\dist\AudioForge `
  --archive .\AudioForge-v1.11.0-win64-ultra.7z `
  --baseline .\evaluation\release-bundle-path-baseline.json `
  --output-dir .

.\.venv\Scripts\python.exe python\tools\release_provenance.py verify `
  --bundle .\dist\AudioForge `
  --archive .\AudioForge-v1.11.0-win64-ultra.7z `
  --checksum .\AudioForge-v1.11.0-win64-ultra.7z.sha256 `
  --manifest .\AudioForge-v1.11.0-win64-ultra.7z.manifest.json `
  --metadata .\AudioForge-v1.11.0-win64-ultra.7z.metadata.json `
  --baseline .\evaluation\release-bundle-path-baseline.json
```

Candidate and promotion:

1. Commit tracked source/doc/version changes.
2. Create annotated tag `v1.11.0`.
3. Upload the raw runtime assets listed above to the GitHub Release or to the configured `asset_source_tag` release. An existing verified `AudioForge-*-win64-ultra.7z` on that release can also be used as the asset source.
4. Push `master` and `v1.11.0`, or run the `Release package` workflow manually
   to create a candidate.
5. Record the candidate workflow run ID and generated archive SHA-256.
6. On a self-hosted runner labelled `self-hosted`, `windows`, `x64`, and
   `audioforge-hardware`, run `Qualify release candidate on hardware` with that
   candidate run ID and digest, the intended physical microphone/output route,
   and the loopback correlation route. The workflow refuses a health duration
   below 1,800 seconds and uploads a digest-bound hardware report.
7. Run `Promote release candidate` with the candidate workflow run ID, hardware
   qualification workflow run ID, release tag, and approved archive SHA-256.
   Promotion downloads the candidate plus both the automated and hardware
   qualification reports, verifies all three against the tag commit and
   digest, then uploads those same bytes without rebuilding.

## Packaging notes

- `AudioForge.spec` is the canonical package definition.
- Packaged builds register canonical bundled DeepFilter paths. Ambient paths stay disabled unless `AUDIOFORGE_ALLOW_EXTERNAL_DF=1` deliberately enables an external override.
- Install `requirements/dev.txt` with `--require-hashes`; do not release from an environment resolved directly from open-ended `pyproject.toml` constraints.
- Review every Semgrep warning in the generated SARIF. The CI gate fails reviewed ERROR-severity findings, while warning-level FFI and process-boundary findings require human triage.
- A clean `cargo audit` is mandatory; do not add RustSec ignores merely to make a release pass.
- Dependabot groups routine Python and Rust patch updates only. Minor, major,
  pre-release-runtime, and 0.x API migrations must be scoped independently and
  pass the applicable build, benchmark, hardware, and package gates; security
  updates remain enabled independently of the routine version-update policy.
- Keep `release-assets.json` current with the required `df.dll`, `target/release/DirectML.dll`, both DeepFilter model tarballs, and `models/silero_vad.onnx`.
- `build_exe.ps1` fails before PyInstaller if a required asset is missing, hash mismatched, or the local `mic_eq_core*.pyd` is older than Rust sources.
- `python/tools/package_smoke.py` verifies exact bundled DLL/model/native-extension and license-notice presence, rejects duplicate top-level native-extension payloads, and rejects a stale bundle-version manifest.
- `python/tools/prune_bundle.py` must not remove dependency `.dist-info` directories; license/metadata retention is part of the release gate. It may remove duplicate native-extension payloads only when the canonical `_internal/mic_eq/mic_eq_core*.pyd` copy is present.
- AudioForge supports Windows 10 and Windows 11 and relies on the system UCRT.
  Microsoft documents the UCRT as an operating-system component on Windows 10
  and later, states that the system copy is always used on Windows 10/11, and
  does not recommend local deployment for performance and security reasons:
  <https://learn.microsoft.com/en-us/cpp/windows/universal-crt-deployment>.
  `prune_bundle.py` removes app-local `ucrtbase.dll` and `api-ms-win-*.dll`;
  package smoke must fail if they return.
- `evaluation/release-bundle-path-baseline.json` controls reviewed bundle path
  additions/removals. Binary hashes are recorded for provenance, but are not
  treated as reproducible-build expectations.
- The release profile strips native symbols without changing optimization level. The package spec excludes only unused SciPy namespaces, while the prune step removes unused Qt SVG payloads; keep both NumPy/SciPy BLAS DLLs, `opengl32sw.dll`, all required models, DirectML, and df.dll because they are runtime dependencies.
- Obtain `DirectML.dll` from the pinned Microsoft DirectML redistributable package, `df.dll` from the pinned DeepFilter runtime build/artifact, and model files from the pinned model artifacts documented in `release-assets.json`.

## Strict realtime regression gates

- The CPAL input callback, CPAL output callback, and post-initialization DSP loop are strict RT regions. They must not use blocking locks, `try_lock`, formatting/logging, vector growth APIs, or Vec-returning suppressor convenience APIs.
- Keep the RT source-scan tests passing whenever code inside a marked `RT_REGION_*` block changes.
- Keep control changes flowing through atomic snapshots or bounded queues; model loading and suppressor construction must remain outside the RT loop.
- Release validation must include fixed-buffer overflow/drop diagnostics checks and a model-discovery smoke pass proving bundled DeepFilter and Silero assets are preferred over CWD/user-directory assets unless an explicit override is set.
- `release-assets.json` paths and bundle paths must stay repository-relative and must not contain `..` traversal.
