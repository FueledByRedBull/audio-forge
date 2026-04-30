# AudioForge v1.7.17

## Highlights

- DeepFilter backend construction now happens outside realtime DSP processing. The audio loop swaps only ready suppressor instances.
- Silero VAD inference now runs in a non-realtime worker. VAD-assisted mode falls back to level gating when probability is stale or unavailable; VAD-only closes in that state.
- Muted and recording output callbacks drain buffered audio while writing silence, preventing stale playback after unmute.
- Startup preset restore, compressor preset loading, raw-recording validation, latency calibration cleanup, and corrupt config handling were hardened.
- Packaging now verifies release assets by hash, rejects stale native extensions, retains dependency metadata/licenses, and smoke-checks exact model/DLL/native-extension contents.

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
- `.\.venv\Scripts\python.exe python\tools\self_test.py`

## Artifact

- `AudioForge-v1.7.17-win64-ultra.7z`
- Size: `108,788,363` bytes (`103.75 MiB`)
- SHA-256: `D182F4A4177EACD99416BBC183C55D8E627E7063576124653ACDDFFB6DCAA38D`
