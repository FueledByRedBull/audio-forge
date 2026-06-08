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

Then run the workflow with:

- `release_tag`: the target tag, for example `v1.8.1`.
- `asset_source_tag`: the release tag containing the raw assets. Leave blank to use the repository variable `AUDIOFORGE_ASSET_SOURCE_TAG`, or the target release when that variable is unset.
- `upload_to_release`: enabled when running manually and the generated archive should be uploaded to the GitHub Release.

On `v*` tag pushes, the workflow builds the Windows package, uploads the `.7z` plus `.sha256` as workflow artifacts, and uploads them to the matching GitHub Release. Set `AUDIOFORGE_ASSET_SOURCE_TAG` when tag-push builds should pull raw assets or an existing package archive from a standing asset-source release. The workflow still verifies all downloaded/extracted assets against `release-assets.json` before packaging.

### Local fallback

Build the Rust extension with all configured features:

```powershell
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

Run the release validation checks:

```powershell
.\.venv\Scripts\python.exe -m ruff check python/mic_eq python/tests python/tools
.\.venv\Scripts\python.exe -m pyright
.\.venv\Scripts\python.exe -m pytest python/tests -q
.\.venv\Scripts\python.exe python\tools\check_versions.py
.\.venv\Scripts\python.exe python\tools\package_smoke.py --source-only
cargo fmt --check
cargo test -p mic_eq_core
cargo clippy -p mic_eq_core --all-targets -- -D warnings
.\.venv\Scripts\python.exe python\tools\package_smoke.py
.\.venv\Scripts\python.exe python\tools\self_test.py
```

Create the distributable archive:

```powershell
& "C:\Program Files\7-Zip\7z.exe" a -t7z -mx=9 -m0=lzma2 -mmt=on -ms=on `
  .\AudioForge-v1.8.1-win64-ultra.7z .\dist\AudioForge\*
```

Compute the checksum:

```powershell
Get-FileHash .\AudioForge-v1.8.1-win64-ultra.7z -Algorithm SHA256
```

Manual publish:

1. Commit tracked source/doc/version changes.
2. Create annotated tag `v1.8.1`.
3. Upload the raw runtime assets listed above to the GitHub Release or to the configured `asset_source_tag` release. An existing verified `AudioForge-*-win64-ultra.7z` on that release can also be used as the asset source.
4. Push `master` and `v1.8.1`, or run the `Release package` workflow manually with `upload_to_release` enabled.

## Packaging notes

- `AudioForge.spec` is the canonical package definition.
- Packaged builds prefer the bundled `df.dll`; set `AUDIOFORGE_ALLOW_EXTERNAL_DF=1` only for deliberate override/testing.
- Keep `release-assets.json` current with the required `df.dll`, `target/release/DirectML.dll`, both DeepFilter model tarballs, and `models/silero_vad.onnx`.
- `build_exe.ps1` fails before PyInstaller if a required asset is missing, hash mismatched, or the local `mic_eq_core*.pyd` is older than Rust sources.
- `python/tools/package_smoke.py` verifies exact bundled DLL/model/native-extension presence and dependency license or `.dist-info` metadata.
- `python/tools/prune_bundle.py` must not remove dependency `.dist-info` directories; license/metadata retention is part of the release gate.
- Obtain `DirectML.dll` from the pinned Microsoft DirectML redistributable package, `df.dll` from the pinned DeepFilter runtime build/artifact, and model files from the pinned model artifacts documented in `release-assets.json`.

## Strict realtime regression gates

- The CPAL input callback, CPAL output callback, and post-initialization DSP loop are strict RT regions. They must not use blocking locks, `try_lock`, formatting/logging, vector growth APIs, or Vec-returning suppressor convenience APIs.
- Keep the RT source-scan tests passing whenever code inside a marked `RT_REGION_*` block changes.
- Keep control changes flowing through atomic snapshots or bounded queues; model loading and suppressor construction must remain outside the RT loop.
- Release validation must include fixed-buffer overflow/drop diagnostics checks and a model-discovery smoke pass proving bundled DeepFilter and Silero assets are preferred over CWD/user-directory assets unless an explicit override is set.
- `release-assets.json` paths and bundle paths must stay repository-relative and must not contain `..` traversal.
