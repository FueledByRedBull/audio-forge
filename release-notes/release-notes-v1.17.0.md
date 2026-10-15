AudioForge 1.17.0 removes the classic window view. If the main window cannot
draw, AudioForge now restarts itself once with software rendering, and otherwise
shows an error screen with Export diagnostics. The status bar now matches what
the audio is doing, keyboard and screen reader support is improved, starting
audio no longer counts dropouts that did not happen, and undo reaches past a
device change.

## Downloads

- **Installer:** [AudioForge-v1.17.0-win64.msi](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.17.0/AudioForge-v1.17.0-win64.msi)
- **Portable:** [AudioForge-v1.17.0-win64-ultra.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.17.0/AudioForge-v1.17.0-win64-ultra.7z)

Choose one. The corresponding-source and evidence archives support
redistribution and verification; SHA256SUMS.txt checks the downloaded files.
AudioForge remains unsigned. Virtual audio cable setup is described in the
[README](https://github.com/FueledByRedBull/audio-forge#set-up).

## New

- If the main window cannot draw, which happens when Direct3D 11 cannot start
  (no usable adapter, a broken driver), AudioForge now restarts itself once with
  software rendering. It does this when the window has been shown for 2 to 3
  seconds and the controls still have not appeared. Nothing is saved, so the next
  launch tries normal rendering again. It does not restart while audio is
  running, or if you cancel the quit at the unsaved-changes prompt; those cases
  show the error screen. Before, the window stayed blank.
- If the main window cannot start at all, AudioForge shows an error screen
  instead of a blank window: what failed, an **Export diagnostics** button and
  **Quit**. At a login start, where the window stays hidden, audio is not started
  and the tray says the window could not open.
- Fixes to the status bar at the bottom of the main window:
  - The Health indicator names the most serious problem and fits the bar at the
    smallest window size.
  - Health advice and tooltips clear when processing stops.
  - The status dot follows whether audio is really running, and shows muted.
  - Device activity gets its start-up grace after every Start.
  - While the output is muted, the Output row says so and shows the levels from
    before the mute, and the OUT meter reads MUTE.
- Keyboard and screen readers in the main window: Return and
  Enter now press buttons, switches, tabs, disclosures and setting rows, as
  Space does. Up and Down move a fader one step. Switches announce their on and
  off state as it changes, tabs say which page is current, and a picker reads its
  choice ("Processing mode: Normal"). Undo, Redo and the help buttons have
  proper names, and the page scroll bars are named. The EQ graph shows keyboard
  focus. The status line and the transmission and calibration texts are
  announced. The Health indicator in the status bar is a button the keyboard
  reaches. These were checked with Qt's accessibility interfaces, not with a
  screen reader.

## Fixes

- Every Start counted three or four output dropouts although no sound was lost,
  which turned the Drops row yellow and put "Check the items marked below" at
  the top of Health. The output stream starts before the first processed audio
  arrives, and the gap was counted as dropouts. Underruns before the output is
  filled are now start-up, not dropouts, and the start-up count appears only in
  the diagnostics export. If a stream writes but never catches up, its dropouts
  show again one second after the first write.
- Undo can reach past a device change. After a device or input-mode change,
  every Ctrl+Z first undid an edit you never made, so earlier edits were out of
  reach. Ctrl+Z also works right after you type a value into a box and press
  Enter.
- A change to the limiter's ceiling, release or careful-output mode made in
  Bypass or Raw monitor now takes effect at once. Before, it waited until you
  returned to Normal, and a lower ceiling did not lower the output.
- The OUT meter's CLIP tag no longer lights in Bypass or Raw monitor while the
  limiter is on. Those modes skip only the lookahead limiter; the final
  true-peak limiter and the ceiling still apply, so a full-scale output there is
  the limiter holding the ceiling. This corrects the 1.16.0 note, which said the
  limiter does not run in those modes.
- Reported latency in Bypass and Raw monitor no longer includes the lookahead
  limiter, which those modes skip. In Raw monitor the figure also never updated,
  so it kept the value from before the switch, or 0 ms in a session started in
  Raw.
- A gate chattering on room noise no longer reads as "the audio engine reported
  a fault" on the Drops row and the Health indicator; it says the gate is
  chattering, which is what the Gate row already said.
- Guided Setup no longer comes back at every launch after you reopen it from
  Settings and close it before finishing, pause it, or quit while it is open.
  This also affected 1.15.3. A paused re-run now resumes at its first pending
  step.
- The EQ gain fader tells screen readers decibels (+2.5 dB), not tenths (25).
- A click on a read-only value box no longer shows the editable border, and the
  box still takes focus so Ctrl+C copies its value.
- The Drops tooltip no longer says "Right-click to reset", which the main window
  does not support. The EQ help lists "Delete resets the selected band" again.

## Changes

- The classic widget view is gone, and so is the `AUDIOFORGE_QML` setting. The
  1.15.x notes offered `AUDIOFORGE_QML=0` as a workaround; it no longer has any
  effect. If the main window cannot draw, the restart and the error screen above
  take its place.
- In the diagnostics export (still schema version 3), `view` is now `quick` or
  `error`, never `classic`. When it is `error`, `classic_reason` names the
  failure: `bridge_failed`, `qml_not_ready`, `scene_graph_error` or
  `no_renderer`. After a software restart the graphics backend reads
  `software_after_failure`. The export also carries the count of start-up output
  underruns, `output_priming_underrun_total`.
- A malformed WAV header that claims a huge chunk now fails with a plain error
  instead of running out of memory, in the development tools that read WAV
  files.
- Release checks now fail on any new Semgrep warning, require `--locked` on
  every cargo build, test and clippy command, and accept only a manually
  dispatched candidate and an annotated tag for promotion. Tests that reached
  shared behaviour through the classic panels now use the shared state and the
  real main window. None of this changes the app.

Live audio processing, sound defaults and the preset format are unchanged,
except that a limiter change made in Bypass or Raw monitor applies at once and
start-up underruns are no longer counted as dropouts. Existing presets retain
their settings, and settings from earlier versions load as before.

## Validation and limits

The evidence archive records the exact candidate revision, hashes and automated
qualification results, including portable startup and MSI lifecycle checks.

The software restart was tried once on a packaged development build (before
the final fixes in this release) with the Direct3D adapter forced to fail: the app quit, came back with software rendering and drew
the window. Started normally, it was left alone. The error screen and the
software-rendering path are covered by automated tests that run offscreen; they
show that the scene works under software rendering, not that every failing
driver is detected. A machine with no usable graphics adapter was not tested.

The status bar, keyboard, screen reader and undo changes have automated tests
that use offscreen rendering and Qt's accessibility interfaces. A real screen
reader and voice control were not tested. The limiter and latency changes have
Rust tests and source scans.

A packaged development build from before the last three fixes in this release
(the reported latency, the status bar and the gate chatter label) ran for an
hour on one real USB microphone, with factory settings and the output muted on
a silent endpoint. It had no dropouts, restarts or output recoveries, and its
memory stayed between 800 and 817 MB after the first five minutes. That run is
where the gate chatter label was found. It is one machine and one route, and
the release build itself was not run on audio hardware.

The start-up change adds a few atomic loads and one store to the output
callback, and the limiter change adds a flag check and a settings read to each
processing block. Both are covered by the allocation and realtime source-scan
tests; no new timing measurement is claimed.

The v1.15.0 audio timing exception remains: one component-event statistic was
1.2151 ms against a 0.5 ms limit; every complete processing kernel in that study
stayed below 10 ms.

Windows 10 (1809 or later) and Windows 11, 64-bit.
