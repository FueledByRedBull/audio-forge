# AudioForge v1.8.1

## Highlights

- Hardened strict realtime control updates so unstable atomic snapshots are deferred and retried instead of applying a torn parameter state.
- Kept DeepFilter runtime failure handling allocation-free on the realtime path while preserving backend diagnostics.
- Surfaced RT buffer overflow, callback error, and active RT error diagnostics in the main UI health strip.
- Added automated Windows release packaging through GitHub Actions, including archive/checksum upload and verified runtime asset reuse from an existing release archive.

## Validation

- `cargo fmt --check`
- `cargo test -p mic_eq_core`
- `cargo clippy -p mic_eq_core --all-targets -- -D warnings`
- `cargo test -p mic_eq_core --release`
- `.\\.venv\\Scripts\\python.exe -m ruff check python/mic_eq python/tests python/tools`
- `.\\.venv\\Scripts\\python.exe -m pyright`
- `.\\.venv\\Scripts\\python.exe -m pytest python/tests -q`
- `.\\.venv\\Scripts\\python.exe -m maturin develop --release`
- `.\\.venv\\Scripts\\python.exe python\\tools\\check_versions.py`
- `.\\.venv\\Scripts\\python.exe python\\tools\\verify_release_assets.py`
- `powershell -ExecutionPolicy Bypass -File .\\build_exe.ps1`
- `.\\.venv\\Scripts\\python.exe python\\tools\\package_smoke.py --source-only`
- `.\\.venv\\Scripts\\python.exe python\\tools\\package_smoke.py`
- `.\\.venv\\Scripts\\python.exe python\\tools\\self_test.py`
- `actionlint .github/workflows/ci.yml .github/workflows/release-package.yml`
- `semgrep scan --metrics=off`

## Artifact

- `AudioForge-v1.8.1-win64-ultra.7z`
- SHA-256: published as `AudioForge-v1.8.1-win64-ultra.7z.sha256`
