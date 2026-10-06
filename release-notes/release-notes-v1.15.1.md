AudioForge 1.15.1 hardens settings changes and preset saving, and gives the
Qt Quick and classic interfaces shared application state.

## Downloads

- **Installer:** [AudioForge-v1.15.1-win64.msi](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.15.1/AudioForge-v1.15.1-win64.msi)
- **Portable:** [AudioForge-v1.15.1-win64-ultra.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.15.1/AudioForge-v1.15.1-win64-ultra.7z)

Choose one. The corresponding-source and evidence archives are for
redistribution and verification; SHA256SUMS checks the downloaded files.
AudioForge remains unsigned. Virtual audio cable setup is described in the
[README](https://github.com/FueledByRedBull/audio-forge#set-up).

## Fixes

- Rejected or still-pending processing changes cannot enter saved presets.
  Undo and rollback restore the last accepted configuration; failed recovery
  keeps output blocked until a complete configuration is reapplied.
- EQ graph and numeric edits share one queue, preventing older graph callbacks
  from overwriting a newer edit.
- Quick and classic controls share exact values, control metadata and commands.
  Route and shell state update through notifications rather than widget polling.
- The Quick EQ graph paints and handles input directly using the same graph
  logic as the classic interface. Classic processing panels are created only
  when that view is used, including graphics-failure fallback.

The visible layout, audio processing algorithms, defaults and persisted preset
format are unchanged. Existing presets are promoted to the current version
without changing their settings. SciPy remains the qualified Auto-EQ solver.

## Validation and limits

The evidence archive identifies the exact source revision and artifact hashes
covered by automated package validation. Source tests cover both views,
rejected writes, preset save/undo, graph interactions and rendering fallback.
Offscreen rendering and automated package checks do not establish hands-on
microphone, Narrator, physical GPU/high-DPI display, reconnect or sleep-recovery
results for every setup.

The v1.15.0 audio timing exception remains: one component-event statistic was
1.2151 ms against a 0.5 ms limit; every complete processing kernel in that study
stayed below 10 ms. No new audio-performance claim is made in this UI hotfix.
See [evaluation/README.md](https://github.com/FueledByRedBull/audio-forge/blob/master/evaluation/README.md)
for the retained evidence and limitations.

Windows 10 (1809 or later) and Windows 11, 64-bit.
