AudioForge 1.22.0 makes the latency figure in Health account for the output
device's own buffer, and rebuilds the latency calibration dialog and Guided
Setup in QML. On the one Windows machine it was checked on, Health read about
35 ms before and about 63 ms now, against about 113 ms measured from
AudioForge's input to the Windows output mix. Audio is unchanged. This release
also fixes a double-click on Stop that restarted audio, a stale frame when
RNNoise is turned back on, and several smaller faults.

**If Health's latency reads higher than before,** that is the display catching
up, not a change in the audio. Until you measure your route it reads "at least
N ms"; Measure Route Latency in Settings gives the full delay.

**Settings format.** The settings file (`config.json`) stays at schema version 2
and the preset format is unchanged. Presets saved by 1.22.0 are stamped 1.22.0,
and 1.21.0 and older refuse them as newer.

## Downloads

Windows, choose one:

- **Installer:** [AudioForge-v1.22.0-win64.msi](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.22.0/AudioForge-v1.22.0-win64.msi)
- **Portable:** [AudioForge-v1.22.0-win64-ultra.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.22.0/AudioForge-v1.22.0-win64-ultra.7z)

Linux, choose one:

- **AppImage:** [AudioForge-v1.22.0-x86_64.AppImage](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.22.0/AudioForge-v1.22.0-x86_64.AppImage)
- **Tarball:** [AudioForge-v1.22.0-linux-x86_64.tar.gz](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.22.0/AudioForge-v1.22.0-linux-x86_64.tar.gz)
- **AUR package (Arch):** `audioforge-bin`. It is not a release file; it
  repackages the tarball and is updated after the release is published.

The AppImage is one file with nothing to unpack: make it executable and run it.
Take the tarball if you want a plain folder, or want `./install.sh` to add
AudioForge to your application menu. On Arch, the AUR package installs the
tarball's contents under `/opt/audioforge`, with an `audioforge` command and a
menu entry, and lets pacman manage them.

Verification files, not needed to run the app:

- [AudioForge-v1.22.0-source.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.22.0/AudioForge-v1.22.0-source.7z)
  and
  [AudioForge-v1.22.0-evidence.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.22.0/AudioForge-v1.22.0-evidence.7z)
  support redistribution and verification.
- [AudioForge-v1.22.0-SHA256SUMS.txt](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.22.0/AudioForge-v1.22.0-SHA256SUMS.txt)
  checks the downloaded files (`Get-FileHash` on Windows, `sha256sum` on Linux).

AudioForge remains unsigned. Windows still needs a virtual audio cable to feed
other apps; setup is described in the
[README](https://github.com/FueledByRedBull/audio-forge#set-up). Linux setup
is in the [Linux section](https://github.com/FueledByRedBull/audio-forge#linux).

## Changes

- **Health latency includes the output device buffer.** On Windows the figure
  now adds the output device's buffer and half a callback period, which it left
  out before (about 27 ms on the one output it was measured on). Until you
  measure the route, Health reads "at least N ms" and the tooltip says the
  figure is a lower bound. With a saved route in use it reads "about N ms", and
  saved route measurements are unaffected. The output queue's share is averaged
  over about a second, so the figure no longer jumps by several milliseconds
  between readings. On Linux the buffer can't be read yet, and the tooltip says
  it is not included.
- **Latency calibration and Guided Setup are rebuilt in QML.** Steps, order,
  saved progress and wording are unchanged. The minimum width grows with the
  text; the old fixed widths (440 px for Guided Setup, 460 px for calibration)
  stay the smallest size. Tab goes first to the scrollable page, which scrolls
  with the arrow keys, Page Up, Page Down, Home and End, and then to the
  controls. Enter on a focused button presses it; with no button focused it
  presses Accept (or Run) in calibration and the main button in Guided Setup.
  Screen readers read each reading once, the device pickers' names carry the
  current choice, and Guided Setup's texts can be read through UI Automation.
  In Guided Setup, focus returns to the button that opened Voice Setup or
  latency calibration when it closes.
- **A quieter idle DeepFilterNet worker.** With noise suppression off, a worker
  that woke about 1000 times a second now polls every 3 ms after 500 ms without
  a frame, so the first frame after a pause is picked up up to 3 ms later.
- **Diagnostics.** Exports gain `output_device_latency_ms`,
  `output_device_buffer_frames`, `output_device_period_frames`,
  `output_device_buffer_known`, `output_device_buffer_failure_code` and
  `latency_compensation_ms`. A zero buffer with a failure code means the device
  buffer could not be read and counts as 0.

## Fixes

- A double-click on Stop no longer restarts audio. Start is ignored for 0.5 s
  after a click on Stop.
- Turning RNNoise back on no longer replays audio from before it was bypassed.
  The first frame after re-enabling is silent, then output resumes.
- Health no longer reports "Gate: CHATTERING (0 times)" right after a restart.
- The About window title carries the version, so UI Automation and screen
  readers can read it.
- Linux: realtime priority that RTKit grants after AudioForge's 2 s wait now
  shows in diagnostics instead of reading as normal priority.
- Linux: after `xdg-desktop-portal --replace`, the global mute shortcut binds
  again instead of staying dead until AudioForge restarts.
- Linux: the D-Bus session address built from `XDG_RUNTIME_DIR` is escaped, so
  a path containing `,`, `;` or `%` works.
- Arch packaging: the `PKGBUILD` and `.SRCINFO` for `audioforge-bin` are written
  with LF line endings on every platform, which makepkg requires.

## Limits

Health's latency still leaves out about 50 ms of Windows audio-engine and route
delay that nothing inside AudioForge can read; only a measured route covers it,
which is why the label says "at least". The output buffer reading has been
checked live on one Windows output device, on the standard path only. On
VB-Audio's shorter-period path and on other devices it has not been run live.
The rebuilt dialogs haven't been tried with a screen reader or with real GPU
rendering.

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
