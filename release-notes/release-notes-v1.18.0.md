AudioForge 1.18.0 is a small release. Voice Setup no longer replaces your EQ
with a flat one when Auto-EQ holds back its tone change, the diagnostics export
gives each weak-capture cause a stable code, and a failed software-rendering restart
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

- When Voice Setup judges a recording weak, the diagnostics export now also
  gives each cause a short, stable code: `snr_under_6_db`, `snr_under_8_db`,
  `speech_under_2_s`, `spectrum_fallback`, `noise_reference_not_usable` or
  `capture_quality_low` (`weak_capture` when none of them applies). The codes it
  already derived from the reason text are still there; a voice under 6 dB over
  the room had none of its own before. The messages you read, the thresholds and
  the analysis are unchanged.

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
- Closing Guided Setup during the level check left audio running. Closing it now
  stops the audio if the check started it, and leaves it running if it was already
  on.
- Test my sound, Auto-EQ and Auto Voice Setup mute your live output while they
  record, and now say so. Before, someone on a call could go silent without
  knowing why.
- A preset with a number too large to read no longer stops AudioForge from
  opening; it is skipped like any other damaged preset. The same kind of value in
  the settings file no longer resets all your other settings.
- If AudioForge crashes or its window fails to open, the reason now goes to its
  log (`app.log`, or `crash.log` for a hard crash). A window that fails to open
  also shows a message with the log's location.
- With a microphone and an output on separate clocks, such as two USB devices,
  the Health page could flag an engine fault for a few seconds at a time, and
  Guided Setup's route check could fail, while AudioForge was only keeping the
  two in step. Neither happens now.

## Changes

- The diagnostics export (still schema version 3) no longer includes
  `main_control_tab_index`, which was always 0.
- Voice Setup and Auto-EQ decide as they did in 1.17.0. Three changes that were
  tried (keeping your EQ when only the tone change is refused, a by-band tonal
  check, and an RNNoise default strength of 0.7) each failed a predefined gate on
  held-out recordings, so none was adopted. Two more, built later, were not
  evaluated and are off.

Live audio processing, sound defaults and the preset format are unchanged, and
settings from earlier versions load as before. The audio engine's code changes
only by deleting methods no production code called.

## Validation and limits

The evidence archive records the exact candidate revision, hashes and automated
qualification results, including portable startup and MSI lifecycle checks.

This release changes the order of the software-rendering restart: close the
window, start the copy, and quit only if it started. Both outcomes are covered by
automated tests that run offscreen; on this build the restart was not tried on a
packaged app or on a machine where starting a process fails. The 1.17.0 order was
tried once on a packaged development build with the Direct3D adapter forced to
fail. The tests show that the scene works under software rendering, not that
every failing driver is detected. A machine with no usable graphics adapter was
not tested.

The keyboard and screen reader support from 1.17.0 was checked with automated
tests that use Qt's accessibility interfaces. A real screen reader and voice
control were not tested.

This build ran for an hour on Windows 11 with a real USB microphone and a silent
output on a separate clock, output muted, factory settings. Memory stayed within
1.5 % of its five-minute reading. There were no dropouts, restarts or output
recoveries, and Health named no fault except gate chatter: the room noise sat
near the default gate threshold. The hands-on checklist in RELEASING.md was also
run on this build. These parts were driven through Windows UI Automation and
passed: first launch, the three pages, Start and Stop (with room noise, not
speech), Mute in the window, closing to the tray and launching again, quitting
with the tray off, closing Guided Setup during the level check, the diagnostics
export, and the pages at the minimum window size. The maintainer ran the rest
(edits and undo, preset loads, a muted microphone, a device change, the tray
menu and tooltip, Ctrl+Q and the unsaved-changes prompt, login startup) and
reported no failure. That is one machine and one route. Audible output, sleep
and resume, and other audio applications were not tested.

Voice Setup can still refuse a setup that would have been no worse than the
settings it would replace. In a held-out study (17 speakers, 170 cases at factory
settings or with a hand-set tone EQ, judged within the study's tolerances), that
happened in 58 cases, mostly through the weak-capture check and the final check.

The v1.15.0 audio timing exception remains: one component-event statistic was
1.2151 ms against a 0.5 ms limit; every complete processing kernel in that study
stayed below 10 ms.

Windows 10 (1809 or later) and Windows 11, 64-bit.
