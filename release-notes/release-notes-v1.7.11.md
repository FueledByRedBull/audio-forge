# AudioForge v1.7.11

## Highlights

- Preferred native `48 kHz` input configs when the selected device supports them and made required resampler setup fail fast at startup instead of silently running the DSP/output path at the wrong rate.
- Reworked overload and recovery handling with proactive input backlog shedding, gentler output catch-up, shorter underrun tails, and deferred gate/suppressor control updates so the real-time path stops depending on `try_lock()` fallbacks.
- Standardized calibration and analysis sample-rate handling around the runtime processor rate and added new backlog, clip, and resampler diagnostics to the main window.

## Validation

- `cargo test -p mic_eq_core --tests`
- `python -m pytest python/tests -v`
- `python python/tools/self_test.py`
- `powershell -ExecutionPolicy Bypass -File .\\build_exe.ps1`
- direct packaged EXE startup check from `dist/AudioForge`

## Artifact

- `AudioForge-v1.7.11-win64-ultra.7z`
- Size: `99,158,472` bytes (`94.56 MiB`)
- SHA-256: `EB7E7C60F8372006E185EB8A20C8659ED3E56EB5E8BCF879F7320B0CC6792113`
