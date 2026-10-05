AudioForge 1.15.0 rebuilds the main window, moves the app from PyQt6 to
PySide6, and makes turning processing stages on and off click-free.

## Which file do I need?

Download **one** of these. Not sure? Choose the installer.

| File | Choose it if |
| --- | --- |
| [AudioForge-v1.15.0-win64.msi](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.15.0/AudioForge-v1.15.0-win64.msi) | You want a normal install for your Windows user. **Recommended.** |
| [AudioForge-v1.15.0-win64-ultra.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.15.0/AudioForge-v1.15.0-win64-ultra.7z) | You want a portable folder. Extract everything, then run `AudioForge.exe`. "Ultra" is the compression level. |

You don't need the other files to run AudioForge: the SHA256SUMS file checks
your download, the evidence archive holds validation records, and the source
archive is the corresponding source for redistribution.

To use AudioForge in Discord, OBS, Zoom, or Teams you also need a virtual audio
cable such as [VB-CABLE](https://vb-audio.com/Cable/); see the
[setup steps](https://github.com/FueledByRedBull/audio-forge#set-up).
AudioForge isn't code-signed, so Windows SmartScreen may warn before it runs.
Compare `Get-FileHash <file>` with `AudioForge-v1.15.0-SHA256SUMS.txt`, then
choose **More info > Run anyway**.

## Changes

- The main window is rebuilt around cards: a rail for Mic, Health and
  Settings, one card per processing stage with a switch and an Advanced
  section, and a larger equalizer graph with a single band editor. The menu
  bar is gone; its actions are on the Presets button and the Settings page,
  and every keyboard shortcut works as before. No parameter, range, default or
  preset format changed.
- Auto Voice Setup and Auto-EQ Voice Tone show one step at a time and describe
  their results in plain language. Test my sound records five seconds and
  plays back the raw and processed versions.
- Turning the limiter, compressor or EQ off and on no longer clicks or shifts
  the audio timeline.
- Phrase onsets pass through noise suppression more cleanly on the RNNoise and
  DeepFilter LL routes, and the automatic de-esser responds to persistent
  harsh resonances.
- The app can start with Windows from a login shortcut you configure under
  Settings > Tray and background. It is off by default.
- The interface now uses PySide6 and Qt under the LGPL, with component notices,
  corresponding source and library replacement instructions. Packages of
  earlier versions that contain PyQt6 keep their GPLv3 terms.
- Startup defers loading SciPy until Auto-EQ needs its solver. Resampling,
  spectrum analysis and peak detection use equivalent NumPy operations;
  the window layout and controls are unchanged by this cleanup.

See the [1.15.0 changelog](https://github.com/FueledByRedBull/audio-forge/blob/master/CHANGELOG.md#v1150)
for the complete change list.

## Validation and compatibility

The evidence archive records the candidate's source revision, artifact hashes
and automated package checks. Use that record to identify the exact build
covered by validation.

The audio follow-up behind the onset and de-esser changes was adopted under an
explicit timing exception: one component's 99th-percentile processing time was
1.2151 ms against a 0.5 ms limit, while every complete processing kernel in
that study stayed below 10 ms. See [evaluation/README.md](https://github.com/FueledByRedBull/audio-forge/blob/master/evaluation/README.md)
for the reports and their limits.

The new window has automated coverage, including a check that every keyboard
focus stop has a screen-reader name and layout renders at the window sizes of
125% to 200% display scaling. It has not yet been through a hands-on pass with
a real microphone, Narrator, or a physical high-DPI display.

AudioForge is built for Windows 10 (1809 or later) and Windows 11, 64-bit.
Automated checks and recorded-audio evaluations do not establish physical
reconnect, sleep recovery, accessibility, or listening results for every
microphone, device route, and display setup. Those configurations remain outside
this release's qualification claims.
