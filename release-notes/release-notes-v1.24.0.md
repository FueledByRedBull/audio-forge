AudioForge 1.24.0 is a new main window. A toolbar, a sidebar with the signal
chain and an inspector panel replace the tab rail and the stage cards, and the
Mic page is now called Overview. The Equalizer page draws your microphone's
spectrum behind the EQ curves and has a Hold to hear original button. Ctrl+K
opens a command palette, Ctrl+Shift+A opens a compact window to keep beside a
call, and when a device is lost the Overview offers Restart audio and Choose
another microphone. Stopping live audio now asks first. The audio path is
unchanged.

**Upgrading from an older version.** 1.24.0 no longer converts settings and
presets saved before 1.23.0, as the 1.23.0 notes announced. If you are on an
older version, install 1.23.0 first, start it once and quit it so it saves your
settings in the current format, and open and re-save each preset you want to
keep. Then update. Anything 1.23.0 or newer wrote loads as before. What is no
longer converted:

- A preset stamped before 1.23.0, or not stamped at all, does not open. It
  stays on disk and the message says to open it in 1.23.0, save it again, then
  open it here. A preset that stores its EQ as plain band arrays or as one
  combined stage is refused the same way. A startup, last-used or route-bound
  preset refused for its age keeps its reference, so it opens again once you
  re-save it.
- A `config.json` with no `schema_version`, or one below 2, is not converted.
  AudioForge starts on default settings, keeps the old file as
  `config.json.corrupt` and shows a warning that says to run 1.23.0 once.
- A config from before the setup wizard starts first-run setup like a fresh
  install, where it used to be marked completed with every step skipped.
- Calibration records at version 1 and latency profiles saved without a route
  latency are dropped, so those results are gone until you calibrate again.
- Device preset bindings saved as a bare string and input preferences keyed by
  a plain friendly name are dropped.
- The `MicEq` folder used by 1.8.2 and older is no longer copied into the
  `AudioForge` folder.

**Settings format.** The settings file (`config.json`) stays at schema version 2
and the preset format is unchanged. Presets saved by 1.24.0 are stamped 1.24.0,
and 1.23.0 and older refuse them as newer.

## Downloads

Windows, choose one:

- **Installer:** [AudioForge-v1.24.0-win64.msi](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.24.0/AudioForge-v1.24.0-win64.msi)
- **Portable:** [AudioForge-v1.24.0-win64-ultra.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.24.0/AudioForge-v1.24.0-win64-ultra.7z)

Linux, choose one:

- **AppImage:** [AudioForge-v1.24.0-x86_64.AppImage](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.24.0/AudioForge-v1.24.0-x86_64.AppImage)
- **Tarball:** [AudioForge-v1.24.0-linux-x86_64.tar.gz](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.24.0/AudioForge-v1.24.0-linux-x86_64.tar.gz)
- **AUR package (Arch):** `audioforge-bin`. It is not a release file; it
  repackages the tarball and is updated after the release is published.

The AppImage is one file with nothing to unpack: make it executable and run it.
Take the tarball if you want a plain folder, or want `./install.sh` to add
AudioForge to your application menu. On Arch, the AUR package installs the
tarball's contents under `/opt/audioforge`, with an `audioforge` command and a
menu entry, and lets pacman manage them.

Verification files, not needed to run the app:

- [AudioForge-v1.24.0-source.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.24.0/AudioForge-v1.24.0-source.7z)
  and
  [AudioForge-v1.24.0-evidence.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.24.0/AudioForge-v1.24.0-evidence.7z)
  support redistribution and verification.
- [AudioForge-v1.24.0-SHA256SUMS.txt](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.24.0/AudioForge-v1.24.0-SHA256SUMS.txt)
  checks the downloaded files (`Get-FileHash` on Windows, `sha256sum` on Linux).

AudioForge remains unsigned. Windows still needs a virtual audio cable to feed
other apps; setup is described in the
[README](https://github.com/FueledByRedBull/audio-forge#set-up). Linux setup
is in the [Linux section](https://github.com/FueledByRedBull/audio-forge#linux).

## Changes

- **Main window.** The toolbar holds the input and output pickers, Mode, Mute,
  the IN and OUT level bars with a CLIP tag, the latency and Start or Stop. It
  fits at 900 px at every font size from 9 to 14 pt; when the row is short, the
  name, the Mode caption, the meter readings and the latency caption drop out
  in turn. The sidebar holds the preset picker, an Overview row and the signal
  chain, from Input cleanup to Limiter. Each stage row shows its current
  setting (`-40 dB`, `RNNoise · 100%`, `Cut 4.0 dB`, `Idle`) and carries the
  stage's switch. Clicking a row opens the stage's settings in an inspector
  beside the page; Esc, the close button or a second click closes it. From
  1200 px the inspector docks beside the page, and below that it lies over it.
- **Overview.** One status sentence with a mark that differs by shape (ring,
  disc, triangle, diamond, square), a 10-second level ribbon, an EQ preview, a
  route check and the window's shortcuts. A mute the engine refuses is no
  longer shown as muted. With no microphone, Start is disabled and the Overview
  lists what AudioForge checked, with Refresh audio devices.
- **Look.** The theme is darker, with one amber accent. Mute is violet, Start
  is amber and Stop is neutral, with a play and a stop mark so they differ by
  shape too. The level meters run blue-grey, yellow, red, and a lost device has
  its own pink. Text is at least 12 px (11 px for axis ticks and keycaps), and
  every text pair measures at least 4.5:1.
- **Equalizer page.** Show picks what the graph draws (Total, Tone or
  Auto-EQ). The total is a bright line over a faint gradient, the high-pass
  corner is named on the curve, and the selected band's settings sit in a panel
  docked under the graph with one sentence that describes the band. Undo is in
  the page header, and a chip counts the edits since the last save.
- **Input analyser.** While the stream runs, the Equalizer graph shows the
  microphone's spectrum, before any processing, as a blue fill behind the
  curves. The scale is fixed, so a quiet take looks quieter than a loud one. It
  reads nothing on another page, in the compact window, while minimized or
  after Stop. The engine's copy of the input does not allocate or lock, and the
  output is bit for bit the same with it on, off or stalled. Health counts
  blocks the analyser dropped (`ISD`), and the diagnostics export adds
  `input_spectrum_dropped_blocks` and `input_spectrum_worker_failed`.
- **Hold to hear original** plays your voice without the equalizer while you
  hold the button, Space on the button, or B with the button or the graph
  focused. Callers hear it too, since there is one output, and a note over the
  graph says so. The equalizer comes back on release, when focus or the window
  moves away, on Stop and when the device is lost. A hold adds no undo step and is never saved.
- **Command palette.** Ctrl+K searches commands: Start, Stop, Mute, Run Auto
  Voice Setup, Test my sound, each stage switch, each built-in preset and five
  main settings. Left and Right change a setting by its own step, and Enter
  applies it as one undoable edit. Each row says what callers hear.
- **Compact window.** From the tray menu or Ctrl+Shift+A: status, health,
  meters, Mute, Presets, Expand and the latency, on top of other windows unless
  you turn off Keep on top. It has no Stop button, and Esc does nothing, so a
  stray key during a call changes nothing. Closing it returns the main window
  to how it was.
- **Device loss.** When the audio runs but its callbacks have stalled, the
  Overview offers **Restart audio**. When the device is lost it also offers
  **Choose another microphone**, which stops the audio, selects the device and
  starts on it. If that start fails, the page says so and that callers hear
  nothing until the audio is back.
- **Stop asks first.** Stopping live audio from Stop, File > Stop Processing or
  Ctrl+. asks "Callers will hear nothing until audio is started again." It
  asks only while unmuted audio runs; quitting, a lost device and Restart audio
  never ask.
- **Clipping.** On Windows the Overview's clipping notice has **Show me how**,
  which opens Windows Sound settings. **Remind me later** hides the notice until
  the next Start or Stop.
- **Dialogs.** Auto Voice Setup, first-run setup, Auto-EQ Voice Tone, Latency
  Calibration and the listening comparison share one header with an Esc keycap
  and a close cross. Voice Setup reads "Step N of 5", and while a take records
  it shows the seconds recorded, the input level and three checks: too quiet,
  clipped and too short. The checks only describe the take; the same analysis
  as before decides whether setup refuses it.
- **Accessibility.** In the Windows contrast themes the selected sidebar row,
  its focus ring and the command palette's selected rows use the
  highlighted-text colour on the highlight fill: the row's name measures 14.4:1
  in High Contrast Black and 15.1:1 in High Contrast White, where it was 1.0 to
  1.5:1. The meters follow system colours and measure at least 14:1 on their
  track, where they fell to 1.5 to 2.8:1, and the check in a step mark stays
  visible in High Contrast White. In the level history the output line is dashed, so input and
  output no longer differ by colour alone. Hold to hear original can be latched
  for a screen reader or a switch. The setup and calibration dialogs announce
  each phase and status change, and warning banners announce when they appear.
  The main pages scroll with the arrow keys, Page Up, Page Down, Home and End,
  Health rows are Tab stops, and card titles have the Heading role.
- **Security and build.** The development lock ignores CVE-2026-126360 in
  `opentelemetry-api` 1.37.0, which Semgrep pins and reads only with tracing
  on, which `run_semgrep.py` never enables. The DeepFilter build moves `rustix`
  to 1.1.6 for RUSTSEC-2026-0486. The pinned ONNX Runtime 1.23.2 CPU files now
  come from Microsoft's signed NuGet package, byte-identical to the existing
  pins, and its source archive from the project's `third-party-sources`
  release.

## Limits

The new window, the Equalizer page, the compact window, the command palette and
the dialogs are covered by offscreen scene tests: keyboard, focus, accessible
names, contrast and layout across font sizes. They haven't been tried with a
screen reader such as Narrator, on Linux desktops by hand, or with Windows
transparency on every renderer; where transparency is off or unsupported the
inspector draws a solid surface. Restart audio and Choose another microphone
are covered by automated tests; a real device lost mid-call hasn't been tried.

VB-CABLE's shorter output period (added in 1.21.0) hasn't been tried on routes
that resample, on a driver that fails it while running, or on Windows 10. The
silence at start (1.21.0) hasn't been tried on real Linux hardware.

Linux has had little hands-on testing. The only setup tried by hand is Arch
Linux with the COSMIC desktop. Ubuntu 24.04 runs the automated tests, and no
other distribution on the supported list has been run. No Linux release has had
a hardware qualification run.

Not tried yet on Linux:

- The GlobalShortcuts portal on a real KDE Plasma or GNOME session, including
  the rebind after `xdg-desktop-portal --replace`. COSMIC has no portal, so the
  global shortcut there needs the `--toggle-mute` command.
- The D-Bus address escaping and the late RTKit diagnostics on a Linux desktop;
  both are covered by tests only.
- Realtime behaviour at small PipeWire buffers: PipeWire quantum 128, 64 and 32
  passed on COSMIC with PipeWire 1.6.10 in the 1.20.0 retest; smaller quanta and
  other hardware are untried.
- How the window groups with its launcher entry in an X11 dock.
- Python 3.14 source builds; CI and the packaged app use 3.13.17.

Voice Setup still refuses an invalid room-noise recording and a weak
voice-to-noise ratio, for example TV or music close to the level of your voice.
Retaking the recording does not help while the sound keeps playing. It can also
refuse a setup that would have been no worse than the settings it would
replace. Its 35 ms latency budget compares the offline engine, not the live
DeepFilterNet worker.

Windows 10 (1809 or later) and Windows 11, 64-bit. Linux: 64-bit x86_64 with
PipeWire, Ubuntu 24.04 or later or a distribution of similar age.
