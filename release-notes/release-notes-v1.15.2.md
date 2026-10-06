AudioForge 1.15.2 fixes startup window flashes and Windows login-shortcut
inspection, opens a larger default window, and removes unused Pythonwin/MFC bindings.

## Downloads

- **Installer:** [AudioForge-v1.15.2-win64.msi](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.15.2/AudioForge-v1.15.2-win64.msi)
- **Portable:** [AudioForge-v1.15.2-win64-ultra.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.15.2/AudioForge-v1.15.2-win64-ultra.7z)

Choose one. The corresponding-source and evidence archives support
redistribution and verification; SHA256SUMS.txt checks the downloaded files.
AudioForge remains unsigned. Virtual audio cable setup is described in the
[README](https://github.com/FueledByRedBull/audio-forge#set-up).

## Fixes

- Open the main window at 1440 x 960 by default, fitted to the available screen.
  Saved window sizes remain unchanged.
- Attach health labels and menu controls before showing them so startup does
  not briefly create small standalone windows.
- Fit the classic interface's preset and setup actions into two rows on compact
  windows, avoiding horizontal overflow while retaining every control.
- Include the timezone helper required by Windows to read shortcut timestamps.
  Its absence could make an existing login shortcut appear unreadable, preventing
  configuration, automatic processing at login, or removal during uninstall.
- Exclude unused Pythonwin/MFC UI bindings. Windows COM integration remains.
- Check native shortcut save, reload and inspection during packaged startup
  qualification using a temporary shortcut, without changing login registration.

Controls, audio processing, sound defaults and preset format are unchanged. Existing
presets retain their settings. QtOpenGL and the Mesa fallback remain included.

## Validation and limits

The evidence archive records the exact candidate revision, hashes and automated
qualification results, including portable startup and MSI lifecycle checks.
An isolated COM roundtrip does not establish actual Windows login-session
behavior on every setup. No new physical-audio or graphics-hardware claim is made.

The v1.15.0 audio timing exception remains: one component-event statistic was
1.2151 ms against a 0.5 ms limit; every complete processing kernel in that study
stayed below 10 ms. See the retained
[evaluation evidence](https://github.com/FueledByRedBull/audio-forge/blob/master/evaluation/README.md).

Windows 10 (1809 or later) and Windows 11, 64-bit.
