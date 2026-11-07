AudioForge 1.23.0 lowers the delay on Linux: the streams now ask PipeWire for a
512-frame quantum and keep a shorter output queue, and Health counts PipeWire's
own stream delay. On one laptop (CachyOS, PipeWire 1.6.10) the delay from input
to AudioForge Mic went from about 119 ms to about 67 ms with RNNoise, and from
about 104 ms to about 52 ms in bypass; a real microphone adds its own capture
buffer. This release also removes a click when RNNoise is turned back on and a
staircase in Auto makeup gain, tightens Windows output probing and recovery,
and fixes Start and Stop, preset loading and several smaller faults. Windows
audio is otherwise unchanged.

**Upgrading from an older version.** 1.23.0 is the last release that converts
old settings and preset formats; 1.24.0 removes those conversions and opens
only presets saved by 1.23.0 or newer. If you are on an older version, install
1.23.0 first, start it once and quit it so it saves your settings in the
current format, and open and re-save each preset you want to keep. The
conversions cover the `MicEq` config folder used by 1.8.2 and older,
`config.json` files older than schema 2, presets saved before 1.23.0 (down to
unversioned ones), presets that store the EQ as plain band arrays or as a
single combined stage, calibration records older than version 2, and latency
profiles saved before route-aware calibration.

**Settings format.** The settings file (`config.json`) stays at schema version 2
and the preset format is unchanged. Presets saved by 1.23.0 are stamped 1.23.0,
and 1.22.0 and older refuse them as newer.

## Downloads

Windows, choose one:

- **Installer:** [AudioForge-v1.23.0-win64.msi](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.23.0/AudioForge-v1.23.0-win64.msi)
- **Portable:** [AudioForge-v1.23.0-win64-ultra.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.23.0/AudioForge-v1.23.0-win64-ultra.7z)

Linux, choose one:

- **AppImage:** [AudioForge-v1.23.0-x86_64.AppImage](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.23.0/AudioForge-v1.23.0-x86_64.AppImage)
- **Tarball:** [AudioForge-v1.23.0-linux-x86_64.tar.gz](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.23.0/AudioForge-v1.23.0-linux-x86_64.tar.gz)
- **AUR package (Arch):** `audioforge-bin`. It is not a release file; it
  repackages the tarball and is updated after the release is published.

The AppImage is one file with nothing to unpack: make it executable and run it.
Take the tarball if you want a plain folder, or want `./install.sh` to add
AudioForge to your application menu. On Arch, the AUR package installs the
tarball's contents under `/opt/audioforge`, with an `audioforge` command and a
menu entry, and lets pacman manage them.

Verification files, not needed to run the app:

- [AudioForge-v1.23.0-source.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.23.0/AudioForge-v1.23.0-source.7z)
  and
  [AudioForge-v1.23.0-evidence.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.23.0/AudioForge-v1.23.0-evidence.7z)
  support redistribution and verification.
- [AudioForge-v1.23.0-SHA256SUMS.txt](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.23.0/AudioForge-v1.23.0-SHA256SUMS.txt)
  checks the downloaded files (`Get-FileHash` on Windows, `sha256sum` on Linux).

AudioForge remains unsigned. Windows still needs a virtual audio cable to feed
other apps; setup is described in the
[README](https://github.com/FueledByRedBull/audio-forge#set-up). Linux setup
is in the [Linux section](https://github.com/FueledByRedBull/audio-forge#linux).

## Changes

- **Linux: 512-sample quantum and a shorter output queue.** With neither
  `AUDIOFORGE_FIXED_INPUT_BUFFER_FRAMES` nor `AUDIOFORGE_FIXED_OUTPUT_BUFFER_FRAMES`
  set, the streams ask PipeWire for 512 frames instead of leaving it at 1024.
  The variables still override this, and an out-of-range value such as 0 leaves
  the choice to PipeWire. PipeWire runs input and output in one cycle, so the
  output queue now centers on 15 ms instead of 35 ms. Windows is unchanged.
  Health's latency adds PipeWire's stream delay, two cycles of the callback
  size, which it left out before; in the same runs it read 7 to 41 ms low and
  now reads 2 to 8 ms above the measurement. A saved Linux latency calibration
  refreshes its engine signature on the first Start and keeps the measured
  route latency.
- **Linux: DeepFilterNet worker at realtime priority**, requested the way the
  DSP thread's is. RTKit is asked from a helper thread for at most 100 ms, so a
  slow answer never looks like a stalled worker, and `dsp_scheduling_degraded`
  also reads true while a worker waits for RTKit or was refused.
- **Linux: device lists and quitting.** Other apps' PipeWire streams are
  left out of the device lists, and the query gives up after 2 s. In the frozen
  build, SIGTERM and SIGHUP end the session without the quit question; Ctrl+C
  does not. The build no longer hands `LD_LIBRARY_PATH`, `QT_PLUGIN_PATH` and
  `QML2_IMPORT_PATH` to the programs it opens, which made a file manager load
  the bundle's Qt and fail.
- **Windows output probing and recovery.** The output-buffer probe has a 2 s
  limit, so a probe stuck in the driver no longer blocks Start, and it opens
  the stream's real sample format. An output is refused for the session only if
  its short-period stream fails before it has filled 1 s of audio; a stream that
  ran longer is retried on the next Start. A noise suppressor error no longer
  overwrites an input or output stream error, which used to drop a stream
  restart.
- **Per-run readings reset.** Stop and Start zero latency, DSP time, levels,
  buffer lengths, noise floor, meters and VAD probabilities instead of showing
  the last run's numbers. Start no longer writes a 0 ms engine latency into a
  route's saved latency profile.
- **Start and Stop through one owner.** Start from the File menu or Ctrl+Return
  right after a Stop is no longer ignored; the 0.5 s lockout applies only to
  the Start/Stop button. Calibration, Auto-EQ, Voice Setup and Test my sound
  now report a failed stop ("Stopping audio failed: ...") and retry it on the
  next release. A dialog opened during a Start, Stop or recovery says "Audio is
  starting, stopping or recovering. Wait a moment and try again." instead of
  starting the engine a second time.
- **Presets and settings.** A last-used preset that the engine refused once, or
  that hit an unexpected error while loading, is kept for the next launch;
  only a missing or invalid preset clears it. Cancel on "Save your sound
  changes?" while loading a route-bound preset is a cancel. An out-of-range
  float in `config.json` is clamped into its range instead of being reset to
  the default.
- **Voice Setup and Auto-EQ checks.** The headroom check refuses a band or
  chain setting that is not a finite number, and Voice Setup refuses captures
  that contain NaN or infinity. The noise reference ignores DC offset. The
  spectrum's VAD window is exact at 44.1 kHz.
- **Environment switches.** `AUDIOFORGE_ENABLE_DEEPFILTER` and
  `AUDIOFORGE_ALLOW_EXTERNAL_DF` accept 1, true, yes and on, in any case.

## Fixes

- Turning RNNoise back on no longer clicks. The first frame after re-enabling
  used to be exact silence; it now follows the current input and fades from dry
  to wet.
- Auto makeup gain ramps per sample instead of stepping once per block (about
  4 dB at each block edge before), and holds in digital silence instead of
  rising to its headroom cap.
- An input gap or a processing-path switch no longer resets the gate's chatter
  count; only Start does.
- Closing the "Stop Audio and Quit?" box without choosing a button cancels the
  quit. With no tray button it used to fail on the missing tray icon.
- A failed device refresh reaches the warning instead of escaping, and the
  listening comparison's "No playback device" notice clears when a device
  appears.
- A window shown again after its shutdown ran resumes the virtual-microphone
  watch.
- An ImportError raised while the app runs now shows its traceback; a missing
  native core still prints the "Build the Rust core" steps.
- Undo and redo that fail while committing a pending edit show the "could not
  be restored" warning.
- Health no longer reads a boolean in a numeric field as a number.
- Linux: creating AudioForge Mic no longer waits more than 2 s on a failed
  create.

## Limits

The Linux delay figures come from 60-second runs on one machine, with a
loopback pair standing in for the microphone. Underruns, output recoveries and
input drops were 0 in every run. Other hardware, PipeWire versions and longer
sessions are untried. DeepFilterNet at realtime priority hasn't been run under
load, and the frozen Linux build's signal handling and import path haven't been
exercised outside tests.

The Windows output probe, the stricter short-period refusal rule, and the new
owner for Start, Stop and dialogs are covered by automated tests only; they
haven't been run on real devices or in the packaged app. Loading presets on
real devices is likewise untried.

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
