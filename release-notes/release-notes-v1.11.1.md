# AudioForge v1.11.1

This patch makes the Windows interface usable on ordinary single-monitor
layouts without changing the audio-processing chain.

## Interface fixes

- Restored window geometry is clamped to one available display.
- The main splitter switches orientation when horizontal space is limited.
- Device controls, processing actions, health indicators, EQ bands, and preset
  buttons reflow instead of extending beyond the screen.
- Cleanup and Dynamics controls no longer clip inside hidden horizontal scroll
  regions.
- Auto-EQ, Auto Voice Setup, latency calibration, and first-run dialogs resize
  and scroll vertically at their supported minimum sizes.
- Numeric controls fit their largest legal values. Long status text wraps, and
  long combo-box options elide without forcing their parent wider.
- The application now applies one explicit dark Fusion palette with tested text
  contrast.

## Compatibility and validation

No DSP algorithm, model, processing order, preset value, or noise-suppression
operating point changed. Existing configurations migrate to v1.11.1 without a
behavioral conversion.

Automated UI checks cover widths from 900 through 1920 pixels, workflow-dialog
minimum sizes, keyboard order, accessible names, semantic colors, and hidden
horizontal overflow. Exact package facts are published in the generated
checksum, metadata, and manifest sidecars.
