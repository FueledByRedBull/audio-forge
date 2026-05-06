# AudioForge v1.7.18

## Highlights

- Auto EQ now solves dynamic center frequencies from the measured spectrum instead of nudging a fixed frequency grid.
- EQ bands remain preset-compatible at 10 slots while storing computed frequency, gain, and Q values.
- The EQ panel now exposes editable per-band frequency spinboxes, and the EQ graph marks active computed band positions.
- Auto EQ, config catalogs, app startup helpers, and large Rust processor tests/API/control sections were split into focused modules without changing public imports.
- The packaged app now imports split config catalog modules correctly under PyInstaller.

## Validation

- `.\.venv\Scripts\python.exe -m ruff check python/mic_eq python/tests python/tools`
- `.\.venv\Scripts\python.exe -m pyright`
- `.\.venv\Scripts\python.exe -m pytest python/tests -q`
- `.\.venv\Scripts\python.exe python\tools\check_versions.py`
- `.\.venv\Scripts\python.exe python\tools\package_smoke.py --source-only`
- `cargo fmt --check`
- `cargo test -p mic_eq_core`
- `cargo clippy -p mic_eq_core --all-targets -- -D warnings`
- `.\.venv\Scripts\python.exe -m maturin develop --release`
- `powershell -ExecutionPolicy Bypass -File .\build_exe.ps1`
- `.\.venv\Scripts\python.exe python\tools\package_smoke.py`
- Packaged executable launch smoke check

## Artifact

- `AudioForge-v1.7.18-win64-ultra.7z`
- Size: `108,915,715` bytes (`103.87 MiB`)
- SHA-256: `17F844C8FBF8E4D79F0A3617969D637ECA89005DD853C902039CEEE19570D987`
