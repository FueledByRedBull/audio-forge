# Releasing AudioForge

## Windows release flow

Build the Rust extension with all configured features:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release
```

Build the portable application from the checked-in PyInstaller spec:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
```

Run the release validation checks:

```powershell
cd rust-core
cargo test -p mic_eq_core --tests
cd ..
.\.venv\Scripts\python.exe -m pytest python/tests -v
.\.venv\Scripts\python.exe python\tools\self_test.py
```

Create the distributable archive:

```powershell
& "C:\Program Files\7-Zip\7z.exe" a -t7z -mx=9 -m0=lzma2 -mmt=on -ms=on `
  .\AudioForge-v1.7.11-win64-ultra.7z .\dist\AudioForge\*
```

Compute the checksum:

```powershell
Get-FileHash .\AudioForge-v1.7.11-win64-ultra.7z -Algorithm SHA256
```

Publish:

1. Commit tracked source/doc/version changes.
2. Create annotated tag `v1.7.11`.
3. Push `master` and `v1.7.11`.
4. Create the GitHub release and upload the `7z`.

## Packaging notes

- `AudioForge.spec` is the canonical package definition.
- Packaged builds prefer the bundled `df.dll`; set `AUDIOFORGE_ALLOW_EXTERNAL_DF=1` only for deliberate override/testing.
- Keep both DeepFilter model tarballs and the VAD model in `models/` for full-feature releases.
