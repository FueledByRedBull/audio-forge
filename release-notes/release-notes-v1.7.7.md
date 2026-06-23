# AudioForge v1.7.7

- Hardened the headless self-test with expected playback windows, a more resilient normalized correlation score, and a fixed retry ladder while keeping the `0.25` confidence threshold.
- Removed steady-state `try_lock()` use from the downstream DSP chain and raw recording tap, and mirrored suppressor latency into atomics for lock-free reporting.
- Deleted the unused legacy `RecordingWorker` path and tightened the packaged build flow around `AudioForge.spec` and `python/tools/prune_bundle.py`.
- Tightened frozen runtime asset lookup so bundled `_internal` assets are preferred over root-level duplicates.

## Validation

- `cargo test -p mic_eq_core --lib -- --nocapture`
- `python -m pytest python/tests -q`
- `python -m maturin develop --release`
- `powershell -ExecutionPolicy Bypass -File .\build_exe.ps1`
- `python python/tools/self_test.py` passed 5 consecutive runs on March 9, 2026

## Artifact

- `AudioForge-v1.7.7-win64-ultra.7z`
- Size: `99,025,678` bytes (`94.44 MiB`)
- SHA-256: `C0307D8BBAEBF3DE9276A2731F51305B61B537F887D8BFE09B7294DBCEA29C92`
