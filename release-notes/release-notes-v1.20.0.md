AudioForge 1.20.0 runs natively on 64-bit Linux with PipeWire, as an AppImage,
a tarball, or an AUR package for Arch. On Linux it creates its own **AudioForge
Mic** microphone, so no virtual cable is needed. Windows audio is unchanged: it
stays on cpal 0.17.3 with the same audio code. A noise model chosen in the short
window while Start was still building its first engine now takes effect, and the
build and the packaged app move to Python 3.13.17.

**Settings format.** The settings file (`config.json`) stays at schema version 2
and the preset format is unchanged. Presets saved by 1.20.0 are stamped 1.20.0,
and 1.19.0 and older refuse them as newer.

## Downloads

Windows, choose one:

- **Installer:** [AudioForge-v1.20.0-win64.msi](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.20.0/AudioForge-v1.20.0-win64.msi)
- **Portable:** [AudioForge-v1.20.0-win64-ultra.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.20.0/AudioForge-v1.20.0-win64-ultra.7z)

Linux, choose one:

- **AppImage:** [AudioForge-v1.20.0-x86_64.AppImage](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.20.0/AudioForge-v1.20.0-x86_64.AppImage)
- **Tarball:** [AudioForge-v1.20.0-linux-x86_64.tar.gz](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.20.0/AudioForge-v1.20.0-linux-x86_64.tar.gz)
- **AUR package (Arch):** `audioforge-bin`. It is not a release file; it
  repackages the tarball and is updated after the release is published.

The AppImage is one file with nothing to unpack: make it executable and run it.
Take the tarball if you want a plain folder, or want `./install.sh` to add
AudioForge to your application menu. On Arch, the AUR package installs the
tarball's contents under `/opt/audioforge`, with an `audioforge` command and a
menu entry, and lets pacman manage them.

Verification files, not needed to run the app:

- [AudioForge-v1.20.0-source.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.20.0/AudioForge-v1.20.0-source.7z)
  and
  [AudioForge-v1.20.0-evidence.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.20.0/AudioForge-v1.20.0-evidence.7z)
  support redistribution and verification.
- [AudioForge-v1.20.0-SHA256SUMS.txt](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.20.0/AudioForge-v1.20.0-SHA256SUMS.txt)
  checks the downloaded files (`Get-FileHash` on Windows, `sha256sum` on Linux).

AudioForge remains unsigned. Windows still needs a virtual audio cable to feed
other apps; setup is described in the
[README](https://github.com/FueledByRedBull/audio-forge#set-up). Linux setup
is in the [Linux section](https://github.com/FueledByRedBull/audio-forge#linux).

## Changes

- **Linux.** AudioForge runs on 64-bit Linux with PipeWire: Ubuntu 24.04 or
  later, Fedora 40 or later, Debian 13, Arch, and distributions based on them.
  A running PipeWire is required. The bundle uses the system's PipeWire, ALSA,
  PulseAudio, D-Bus, graphics, font and C++ libraries, which supported desktops
  already have. RNNoise, DeepFilterNet and the voice detector ship with it.
- **AudioForge Mic.** While AudioForge runs it adds a mono, 48 kHz microphone
  that Discord, OBS, browsers and other apps record from. Setup picks
  **AudioForge Mic Input** as the output. The microphone exists only while
  AudioForge runs and is built again if PipeWire restarts. AudioForge doesn't
  make it the system default, but with no other microphone connected the
  desktop may.
- **No feedback loop.** While AudioForge Mic is the system default, Start
  with **Follow system default** as the input says no microphone is available,
  and a running session stops once instead of feeding its own output back in.
  Press Start after connecting a microphone. Diagnostics export a count as
  `own_microphone_refusal_count`.
- **Devices.** The pickers show PipeWire's device names and offer "Follow system
  default". Sink monitors and AudioForge's own microphone are left out.
- **Mute on Linux.** `audioforge --toggle-mute` (it also works on Windows, as
  `AudioForge.exe --toggle-mute`) mutes or unmutes the running
  copy and never starts a second one. The application menu entry has the same
  **Toggle Output Mute** action. The global shortcut (Ctrl+Alt+M, under
  **Settings > Tray and background**) asks the desktop for its key through the
  XDG GlobalShortcuts portal, which KDE Plasma, GNOME 48 and later, and
  Hyprland provide. With no portal, or if the desktop refuses the key, the
  status line says so and names the `--toggle-mute` command to bind in your
  keyboard settings.
- **Login startup, tray and priority.** Login startup writes an XDG autostart
  entry under the same rules as the Windows shortcut; in an AppImage it names
  the image file. The tray icon needs a panel that shows StatusNotifierItem
  icons, and it appears when the panel starts after AudioForge. The processing
  thread and PipeWire's callback thread ask RTKit for realtime priority; if
  RTKit is missing or refuses, processing continues at normal priority and
  diagnostics say so.
- **Files.** Settings and presets live in `~/.config/AudioForge`, or in
  `$XDG_CONFIG_HOME/AudioForge` when that is an absolute path. Level advice
  names your sound settings instead of Windows. Source builds accept Python
  3.14 as well as 3.13.
- **Python 3.13.17.** The build and the packaged app move from 3.13.15. The
  corresponding-source manifest's OpenSSL pin follows CPython's build files,
  from 3.0.21 to 3.5.9; neither bundle ships OpenSSL.
- **Release files.** Release package builds the Linux tarball and AppImage from
  the same commit as the Windows files. Promotion checks them on Linux against
  the candidate run and the tag commit, and publishes all seven files together.

## Fixes

- A noise model chosen in the short window while Start was still building its
  first engine could show as selected while the previous model kept running. It now applies, including for
  a change made by a preset load, undo or redo, or an automatic restart. If the
  model fails to load, the previous one keeps running and the status line says
  to choose the model again and to check the model files or use Help > Export
  Diagnostics. The count of dropped selections is in diagnostics and shows as
  NSD in the Drops tooltip.

## Limits

Linux has had little hands-on testing. The only setup tried by hand is Arch
Linux with the COSMIC desktop. Ubuntu 24.04 runs the automated tests, and no
other distribution on the supported list has been run. No Linux release has had
a hardware qualification run.

Not tried yet:

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
