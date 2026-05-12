# AudioForge v1.8.0

## Highlights

- Completed the EQ quality pass with confidence-weighted Auto-EQ solving, post-solve validation, target-error metrics, and calibration diagnostics.
- Added live EQ interaction checks for overlapping bands, shelf/peak stacking, narrow boosts, combined boost, and response ripple.
- Tuned the wider dynamics chain with speech-aware auto makeup gain, smoother adaptive compressor release, gate fused-score diagnostics, and additional DSP regression coverage.
- DeepFilter runtime discovery now ignores the process current working directory and only enables local DeepFilter assets from trusted application/runtime roots.
- Package smoke validation now checks exact PyInstaller bundle paths for DLLs, model files, and the native extension instead of accepting recursive basename matches.
- The Windows portable bundle was rebuilt with the hardened release checks.

## Validation

- `.\.venv\Scripts\python.exe -m pytest python/tests/test_app_bootstrap.py python/tests/test_package_tools.py -q`
- `.\.venv\Scripts\python.exe -m pytest python/tests/test_auto_eq.py python/tests/test_eq_ui_sync.py python/tests/test_ui_sample_rate_and_diagnostics.py -q`
- `.\.venv\Scripts\python.exe -m ruff check python/mic_eq/ui/app_bootstrap.py python/tools/package_smoke.py python/tests/test_app_bootstrap.py python/tests/test_package_tools.py`
- `.\.venv\Scripts\python.exe -m ruff check python/mic_eq python/tests python/tools`
- `.\.venv\Scripts\python.exe -m pyright`
- `.\.venv\Scripts\python.exe python\tools\check_versions.py`
- `.\.venv\Scripts\python.exe python\tools\package_smoke.py --source-only`
- `.\.venv\Scripts\python.exe -m maturin develop --release`
- `cargo fmt --check`
- `cargo test -p mic_eq_core`
- `cargo clippy -p mic_eq_core --all-targets -- -D warnings`
- `powershell -ExecutionPolicy Bypass -File .\build_exe.ps1`
- `.\.venv\Scripts\python.exe python\tools\package_smoke.py`
- `.\.venv\Scripts\python.exe python\tools\self_test.py`

## Artifact

- `AudioForge-v1.8.0-win64-ultra.7z`
- Size: `108,824,029` bytes (`103.78 MiB`)
- SHA-256: `92938ED4CFBB747EB47AF0435C210EDFBC635DA896C520086B949CFF3B97A7F2`
