# AudioForge v1.7.8

## Highlights

- Removed steady-state VAD buffer draining and per-window scratch allocation by switching Silero VAD to reusable cursor-based buffers.
- Hard-gated audio-adjacent debug logging in the VAD, gate, and processor paths so release builds stay quiet in those hot sections.
- Pruned the unused bundled `scipy.optimize._highspy` payload to shrink the packaged Windows app without dropping features.

## Validation

- `cargo test -p mic_eq_core --lib -- --nocapture`
- `python -m pytest python/tests -q`
- `.\\.venv\\Scripts\\python.exe -m maturin develop --release`
- `powershell -ExecutionPolicy Bypass -File .\\build_exe.ps1`
- `python python/tools/self_test.py` passed 5 consecutive default runs on March 13, 2026
- packaged app startup smoke test passed

## Artifact

- `AudioForge-v1.7.8-win64-ultra.7z`
- Size: `97,352,410` bytes (`92.84 MiB`)
- SHA-256: `A91B9197170882BFCCB395262639E022A72E9097EF347DF971F419E0240BDAB0`
