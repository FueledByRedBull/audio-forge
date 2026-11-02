AudioForge 1.21.0 cuts the delay between your microphone and VB-CABLE. With a
USB microphone and DeepFilterNet, the microphone to CABLE Output delay drops from
about 198 ms to about 134 ms, and with DeepFilterNet LL it is 104 to 115 ms.
VB-CABLE's own delay moves by about 10 ms between starts, so treat these as
approximate. Starts also stop logging dropouts in their first half second.

**If you measured your route latency,** measure it again. A saved figure reads
high until you do, by up to about 48 ms on VB-CABLE. Only the displayed estimate
is affected, never the audio.

**Settings format.** The settings file (`config.json`) stays at schema version 2
and the preset format is unchanged. Presets saved by 1.21.0 are stamped 1.21.0,
and 1.20.0 and older refuse them as newer.

## Downloads

Windows, choose one:

- **Installer:** [AudioForge-v1.21.0-win64.msi](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.21.0/AudioForge-v1.21.0-win64.msi)
- **Portable:** [AudioForge-v1.21.0-win64-ultra.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.21.0/AudioForge-v1.21.0-win64-ultra.7z)

Linux, choose one:

- **AppImage:** [AudioForge-v1.21.0-x86_64.AppImage](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.21.0/AudioForge-v1.21.0-x86_64.AppImage)
- **Tarball:** [AudioForge-v1.21.0-linux-x86_64.tar.gz](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.21.0/AudioForge-v1.21.0-linux-x86_64.tar.gz)
- **AUR package (Arch):** `audioforge-bin`. It is not a release file; it
  repackages the tarball and is updated after the release is published.

The AppImage is one file with nothing to unpack: make it executable and run it.
Take the tarball if you want a plain folder, or want `./install.sh` to add
AudioForge to your application menu. On Arch, the AUR package installs the
tarball's contents under `/opt/audioforge`, with an `audioforge` command and a
menu entry, and lets pacman manage them.

Verification files, not needed to run the app:

- [AudioForge-v1.21.0-source.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.21.0/AudioForge-v1.21.0-source.7z)
  and
  [AudioForge-v1.21.0-evidence.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.21.0/AudioForge-v1.21.0-evidence.7z)
  support redistribution and verification.
- [AudioForge-v1.21.0-SHA256SUMS.txt](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.21.0/AudioForge-v1.21.0-SHA256SUMS.txt)
  checks the downloaded files (`Get-FileHash` on Windows, `sha256sum` on Linux).

AudioForge remains unsigned. Windows still needs a virtual audio cable to feed
other apps; setup is described in the
[README](https://github.com/FueledByRedBull/audio-forge#set-up). Linux setup
is in the [Linux section](https://github.com/FueledByRedBull/audio-forge#linux).

## Changes

- **Shorter output period on VB-CABLE.** On Windows, output to VB-CABLE and other
  VB-Audio devices opens on a 128-frame audio engine period (about 2.7 ms at
  48 kHz) when the device offers one and its mix is 32-bit float at the stream's
  rate and channel count. The period applies to every app using the device. Other
  outputs keep the standard path, and VB-CABLE is the only device measured. If the
  shorter stream fails while running, that device uses the standard path until the
  app restarts. Setting `AUDIOFORGE_FIXED_OUTPUT_BUFFER_FRAMES` to anything, 0
  included, keeps the standard path.
- **Less audio queued.** Windows keeps 15 ms queued on 10 ms callbacks and 8 ms
  on callbacks up to 6 ms when no resampler runs. Degraded scheduling keeps the
  older target.
- **Diagnostics.** Exports gain `output_low_period_frames`,
  `output_low_period_failure_code`, `output_largest_callback_frames` and
  `output_sustained_callback_frames`.

## Fixes

- A start plays silence until the output queue holds one callback's worth of audio
  plus 15 ms, on Windows and Linux. That removes the dropouts most starts logged
  in their first half second.

## Limits

The shorter period hasn't been tried on routes that resample, on a driver that
fails it while running, or on Windows 10. The start silence hasn't been tried on
real Linux hardware.

Linux has had little hands-on testing. The only setup tried by hand is Arch
Linux with the COSMIC desktop. Ubuntu 24.04 runs the automated tests, and no
other distribution on the supported list has been run. No Linux release has had
a hardware qualification run.

Not tried yet on Linux:

- The GlobalShortcuts portal on a real KDE Plasma or GNOME session. COSMIC has
  no portal, so the global shortcut there needs the `--toggle-mute` command.
- Realtime behaviour at very small PipeWire buffers.
- How the window groups with its launcher entry in an X11 dock.
- Python 3.14 source builds; CI and the packaged app use 3.13.17.

If RTKit answers after AudioForge's 2 s wait, the processing thread can end up
realtime while diagnostics report normal priority.

Each live DeepFilterNet worker wakes about 1000 times a second, including while
noise suppression is off.

Voice Setup still refuses an invalid room-noise recording and a weak
voice-to-noise ratio, for example TV or music close to the level of your voice.
Retaking the recording does not help while the sound keeps playing. It can also
refuse a setup that would have been no worse than the settings it would
replace. Its 35 ms latency budget compares the offline engine, not the live
DeepFilterNet worker.

Windows 10 (1809 or later) and Windows 11, 64-bit. Linux: 64-bit x86_64 with
PipeWire, Ubuntu 24.04 or later or a distribution of similar age.
