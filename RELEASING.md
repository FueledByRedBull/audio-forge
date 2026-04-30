# Releasing AudioForge

## Windows release flow

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
  .\AudioForge-v1.7.17-win64-ultra.7z .\dist\AudioForge\*
```

Compute the checksum:

```powershell
Get-FileHash .\AudioForge-v1.7.17-win64-ultra.7z -Algorithm SHA256
```

Publish:

1. Commit tracked source/doc/version changes.
2. Create annotated tag `v1.7.17`.
3. Push `master` and `v1.7.17`.
4. Create the GitHub release and upload the `7z`.

## Packaging notes

- `AudioForge.spec` is the canonical package definition.
- Packaged builds prefer the bundled `df.dll`; set `AUDIOFORGE_ALLOW_EXTERNAL_DF=1` only for deliberate override/testing.
- Keep `release-assets.json` current with the required `df.dll`, `target/release/DirectML.dll`, both DeepFilter model tarballs, and `models/silero_vad.onnx`.
- `build_exe.ps1` fails before PyInstaller if a required asset is missing, hash mismatched, or the local `mic_eq_core*.pyd` is older than Rust sources.
- `python/tools/package_smoke.py` verifies exact bundled DLL/model/native-extension presence and dependency license or `.dist-info` metadata.
- `python/tools/prune_bundle.py` must not remove dependency `.dist-info` directories; license/metadata retention is part of the release gate.
- Obtain `DirectML.dll` from the pinned Microsoft DirectML redistributable package, `df.dll` from the pinned DeepFilter runtime build/artifact, and model files from the pinned model artifacts documented in `release-assets.json`.
