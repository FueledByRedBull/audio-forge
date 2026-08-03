# AudioForge v1.11.2

This patch finishes the responsive-layout work from v1.11.1. It does not
change the audio-processing chain.

## Interface fixes

- The default 1280x850 window uses the one-row device, action, and status
  layouts again, eliminating the unnecessary full-window vertical scrollbar.
- Narrow windows still reflow safely and remain operable through deliberate
  vertical scrolling at the supported 900x640 minimum.
- The ten EQ bands wrap into rows without horizontal scrolling. Five visible
  bands at ordinary widths are the first row of the complete 5x2 layout, not
  missing or clipped controls.
- Level-meter ticks and numbers have a visible gap, so the top scale mark no
  longer resembles `-0`.
- The runtime-counter chip shows a compact health summary. Its tooltip and
  accessibility description retain the complete diagnostic counter string.

## Compatibility and validation

No DSP algorithm, model, processing order, preset value, or suppression
setting changed. Existing configurations migrate to v1.11.2 without a
behavioral conversion.

Automated coverage checks the default and minimum window sizes, responsive
breakpoints, hidden horizontal overflow, level-meter geometry, detailed
diagnostic retention, keyboard access, and semantic color contrast. Exact
package facts are published in the generated checksum, metadata, and manifest
sidecars.
