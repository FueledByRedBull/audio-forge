AudioForge 1.18.2 is an interface release. It asks before quitting while audio
runs, moves Stop away from Mute, follows Windows contrast themes, and fixes
switches that an assistive technology flipped without changing the setting. It
changes no audio processing, no Voice Setup or Auto-EQ decision and no sound
default.

**Settings format.** The settings file (`config.json`) is now schema version 2.
1.12.0 through 1.18.1 keep their defaults, keep a recovery copy, show
"AudioForge found a newer config format. It was preserved; settings changes cannot be saved until this build is upgraded."
and will not save over a settings file this version has written. If you go
back to an older build, expect its default settings until you return to
1.18.2. The preset format is unchanged, but presets saved by 1.18.2 are
stamped 1.18.2, and 1.18.1 and older refuse them as newer.

## Downloads

- **Installer:** [AudioForge-v1.18.2-win64.msi](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.18.2/AudioForge-v1.18.2-win64.msi)
- **Portable:** [AudioForge-v1.18.2-win64-ultra.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.18.2/AudioForge-v1.18.2-win64-ultra.7z)

Choose one. The corresponding-source and evidence archives support
redistribution and verification; SHA256SUMS.txt checks the downloaded files.
AudioForge remains unsigned. Virtual audio cable setup is described in the
[README](https://github.com/FueledByRedBull/audio-forge#set-up).

## Changes

- **Quit question.** Closing the window, Ctrl+Q or **Quit AudioForge** in the
  tray while audio runs and close-to-tray is off asks "Stop Audio and Quit?".
  You can Quit, Keep Running in Tray (offered only while the tray icon shows) or
  Cancel. Keep Running in Tray hides the window for that close and leaves the
  close-to-tray setting alone. Don't ask again counts only once a Quit goes
  ahead. The question never appears on a Windows sign-out or shutdown, or when
  audio is stopped. **Settings > Tray and background > Ask before quitting while
  audio is running** turns it back on.
- **Top bar.** Stop no longer sits beside Mute. The bar reads Mute, the mode
  picker, Start/Stop.
- **Windows contrast themes.** While a contrast theme is on, the window takes
  the system colours; otherwise the dark look is unchanged. If you change the
  theme while AudioForge runs, the status line asks you to restart it. The
  peak-hold line takes the window text colour, and the EQ band numbers and the
  clip label keep dark text on their colours.
- **Layout and text.** Text in the main window's controls follows the app font
  and none is under 12 px.
  The route row stacks below 1200 px, and the band editor keeps two columns only
  from 1400 px so the gain fader keeps its width. Switched-off stages show an
  Off badge instead of dimming, and each view has one primary button.
- **Accessibility.** Banners and status messages are announced, and banners
  sit under the top bar. Tooltips appear sooner; the help (?) tips open from
  the keyboard and close on Esc or when focus leaves, and the page scrolls to
  keep the focused control in view. Picker popups stay inside the window, open
  Advanced sections survive a layout change, and the rail and meters have
  label-first accessible names with roles. Reduced motion no longer slows the
  meters or stops the level trace. Tab order is tested at 1200, 1400 and
  1800 px. Nothing was checked with a real screen reader or voice control.
- **Wording.** Locked device pickers and Refresh say why. A failed device
  listing shows a warning instead of an "Error:" entry in the list. The gate's
  voice-detection rows read Voice threshold and Voice pre-gain. Start, Stop and
  Export diagnostics name their shortcuts, Health advice names where each
  setting lives, and the tray tooltip fits Windows' 127 characters. Pressing
  Start with no output selected says to choose one in the Output list. The
  Auto-EQ chip is shorter, with its numbers in the tooltip, and the band editor
  hides Q and slope when the filter does not use them. Settings rows for the
  startup preset and device presets show the current choice.
- **Dialogs.** The listening comparison keeps **Keep Proposed** and **Reject
  Proposed** in view and says the live output is muted. Auto-EQ and Voice Setup
  show that notice only while a recording mutes the output. The latency dialog shows Total latency and
  Confidence, with the other readings behind Show details.

## Fixes

- A switch flipped with an assistive technology's toggle command (the Windows
  UI Automation toggle that some screen readers and voice control use) now
  changes the setting. In 1.18.1 and 1.18.0 it changed on screen but not in the
  app. A switch also acts only while it is enabled, and returns to the setting's
  state if the change is refused.
- Peak hold and the clip strip run on elapsed time.

## Validation and limits

The evidence archive records the exact candidate revision, hashes and automated
qualification results, including portable startup and MSI lifecycle checks.

The Rust audio core is unchanged since 1.18.1 apart from a license field in its
manifest, so steady-state audio is the same. The interface changes are covered
by automated tests, not by a run on audio hardware.

This build ran for an hour on Windows 11 with a real USB microphone and a silent
output on a separate clock, output muted, factory settings. Memory stayed within
2 % of its five-minute reading. There were no dropouts, restarts or output
recoveries, and Health named no fault except gate chatter: the room noise sat
near the default gate threshold. The hands-on checklist in RELEASING.md was also
run on this build. These parts were driven through Windows UI Automation and
passed: first launch, the three pages, Start and Stop (with room noise, not
speech), Mute in the window, switch edits with Undo and Redo, a preset load and
its Undo, closing to the tray and launching again, the new quit question and its
Cancel, Cancel at the unsaved-changes prompt, quitting with the tray off,
opening and closing Guided Setup, the diagnostics export, and the pages at the
minimum window size. The maintainer ran the rest (the meter following speech,
keyboard undo after an EQ drag and a gate threshold change, a muted microphone,
a device change, the tray menu and tooltip, Ctrl+Q, closing Guided Setup during
the level check, login startup on the portable copy and the MSI) and reported no
failure. That is one machine and one route. Audible output, sleep and resume,
and other audio applications were not tested.

I24 and U24 input is covered by sample-conversion tests. No device that
delivers those formats is claimed to have been tried.

Voice Setup can still refuse a setup that would have been no worse than the
settings it would replace; that is unchanged from 1.18.0.

The v1.15.0 audio timing exception remains: one component-event statistic was
1.2151 ms against a 0.5 ms limit; every complete processing kernel in that study
stayed below 10 ms.

Windows 10 (1809 or later) and Windows 11, 64-bit.
