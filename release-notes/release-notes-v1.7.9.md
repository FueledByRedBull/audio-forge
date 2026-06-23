# AudioForge v1.7.9

## Highlights

- Fixed the recent main-window styling regression by removing the forced light-surface theme and scoping custom styling to the action buttons and health chips.
- Rebalanced the splitter sizing and pane minimums so the tabbed control column and EQ panel stay readable on narrower desktop widths.
- Tightened the action-row spacing and tab-page margins to make the new `Cleanup` and `Dynamics` layout feel cleaner without changing any DSP behavior.

## Validation

- `cargo test -p mic_eq_core --lib -- --nocapture`
- `python -m pytest python/tests -q`
- `.\\.venv\\Scripts\\python.exe -m maturin develop --release`
- `powershell -ExecutionPolicy Bypass -File .\\build_exe.ps1`
- `python python/tools/self_test.py`
- direct Qt visual pass of the patched main window on March 13, 2026

## Artifact

- `AudioForge-v1.7.9-win64-ultra.7z`
- Size: `97,361,501` bytes (`92.85 MiB`)
- SHA-256: `A66F5AD49AA982F280541E5D8749AB9A4614C73CDB33F76F29E8CA2CA27BF14F`
