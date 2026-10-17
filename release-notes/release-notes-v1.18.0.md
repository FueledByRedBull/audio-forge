AudioForge 1.18.0 is a small release. Voice Setup no longer replaces your EQ
with a flat one when Auto-EQ holds back its tone change, the diagnostics export
says why a recording was judged weak, and a failed software-rendering restart
now ends on the error screen. It changes no Voice Setup or Auto-EQ decision, and
the audio engine is unchanged.

## Downloads

- **Installer:** [AudioForge-v1.18.0-win64.msi](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.18.0/AudioForge-v1.18.0-win64.msi)
- **Portable:** [AudioForge-v1.18.0-win64-ultra.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.18.0/AudioForge-v1.18.0-win64-ultra.7z)

Choose one. The corresponding-source and evidence archives support
redistribution and verification; SHA256SUMS.txt checks the downloaded files.
AudioForge remains unsigned. Virtual audio cable setup is described in the
[README](https://github.com/FueledByRedBull/audio-forge#set-up).

## New

- When Voice Setup judges a recording weak, the diagnostics export now names
  each cause with its own code: `snr_under_6_db`, `snr_under_8_db`,
  `speech_under_2_s`, `spectrum_fallback`, `noise_reference_not_usable` or
  `capture_quality_low` (`weak_capture` when none of them applies). Before, all of
  them exported as `weak_capture`. The messages you read, the thresholds and the
  analysis are unchanged.

## Fixes

- Applying Voice Setup suggestions after Auto-EQ held back its tone change
  replaced your correction EQ with a flat one, and could switch a disabled EQ on.
  Your EQ now stays as it is. The other settings apply as before, and the tone
  line says the tone is kept.
- After an Auto-EQ error, Voice Setup refuses the result as incomplete, but the
  tone line also said everything else below still applies. It no longer does.
- If the main window cannot draw and AudioForge cannot start its software-rendering
  copy either, it used to quit and leave nothing running. It now stays open on the
  error screen, which says the restart failed, with **Export diagnostics** and
  **Quit**. The tray icon and mute hotkey come back.
- The level check's advice for a voice that is too quiet said a higher level would
  clip the loudest moments. It now says a higher level would push them past the
  limit.

## Changes

- The diagnostics export (still schema version 3) no longer includes
  `main_control_tab_index`, which was always 0.
- Voice Setup and Auto-EQ decide as they did in 1.17.0. Three changes that were
  tried (keeping your EQ when only the tone change is refused, a by-band tonal
  check, and an RNNoise default strength of 0.7) each failed a predefined gate on
  held-out recordings, so none was adopted.

Live audio processing, sound defaults and the preset format are unchanged, and
settings from earlier versions load as before. The audio engine's code changes
only by deleting methods nothing called.

## Validation and limits

The evidence archive records the exact candidate revision, hashes and automated
qualification results, including portable startup and MSI lifecycle checks.

The path where the software-rendering copy cannot start is covered by automated
tests: the app stays running, keeps the single-instance guard and shows the error
screen. It was not tried on a machine where starting a process fails. The
software restart itself was tried once on a packaged 1.17.0 development build
with the Direct3D adapter forced to fail, and the error screen and the
software-rendering path are covered by tests that run offscreen; they show that
the scene works under software rendering, not that every failing driver is
detected. A machine with no usable graphics adapter was not tested.

The keyboard and screen reader support from 1.17.0 was checked with automated
tests that use Qt's accessibility interfaces. A real screen reader and voice
control were not tested.

No audio-hardware run of this build is claimed. The hour-long run on a real USB
microphone in the 1.17.0 notes was a 1.17.0 development build and is not
attributed to this one.

Voice Setup can still refuse a setup that would have been no worse than your
current settings: in a held-out study of 170 recordings, that happened for 58,
mostly through the weak-capture check and the final check.

The v1.15.0 audio timing exception remains: one component-event statistic was
1.2151 ms against a 0.5 ms limit; every complete processing kernel in that study
stayed below 10 ms.

Windows 10 (1809 or later) and Windows 11, 64-bit.
