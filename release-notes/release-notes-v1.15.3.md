AudioForge 1.15.3 fixes window restore, the Stop button, EQ band dragging and
the main window's level meters, lets Auto Voice Setup work with microphones
that deliver silence, and fits its loudness target to the final headroom check.

## Downloads

- **Installer:** [AudioForge-v1.15.3-win64.msi](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.15.3/AudioForge-v1.15.3-win64.msi)
- **Portable:** [AudioForge-v1.15.3-win64-ultra.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.15.3/AudioForge-v1.15.3-win64-ultra.7z)

Choose one. The corresponding-source and evidence archives support
redistribution and verification; SHA256SUMS.txt checks the downloaded files.
AudioForge remains unsigned. Virtual audio cable setup is described in the
[README](https://github.com/FueledByRedBull/audio-forge#set-up).

## Fixes

- The window reopens where and how it was closed, including maximized, with
  its title bar on screen. The saved position was restored as a different
  rectangle from the one that was saved. Restoring AudioForge from the tray, or
  starting it a second time, no longer shrinks a maximized window.
- The Stop button could stay disabled while audio was running. Start and Stop
  now follow the same processing state.
- The IN and OUT level meters in the main window stopped repainting after their
  first frame in 1.15.2. They repaint on every reading again.
- Auto Voice Setup no longer stops at the room-noise step on a microphone or
  headset that delivers digital silence when nobody speaks
  ([issue 82](https://github.com/FueledByRedBull/audio-forge/issues/82)). It
  continues with a note and takes the noise estimate from the pauses in the
  voice recording. A silent voice recording gets a clear message to check that
  the microphone is not muted and the right input is selected.
- An unsteady room-noise recording, for example from a gate that opens now and
  then, continues with a note that setup will be more cautious instead of
  asking for a retake that cannot help.
- Clipped, unreadable or unusable setup recordings are refused as soon as they
  finish, with a plain instruction to record again, instead of failing after
  the voice step.
- In the main window, a refused noise-suppression backend switch shows the
  "Model Switch Failed" box and a successful switch confirms in the status
  line, as the classic view does.
- Dragging an EQ band handle in the main window no longer turns into scrolling
  the page. Pages scroll with the wheel, the scroll bar, the keyboard and
  touch, not by dragging with the mouse, so a click that moves a little on a
  button or switch is no longer lost.
- Clicking an EQ band handle selects it without changing the band, and a drag
  no longer starts with a jump. Clicking a control that is partly scrolled out
  of view no longer moves the page under the pointer.
- Escape cancels an edit in a value box, and the read-only gain readout no
  longer blocks the mouse wheel after it has been clicked.

## Changes

- When peaks at the requested loudness leave less headroom than the final check
  requires, Voice Setup lowers its loudness target by whole decibels until the
  check passes, and the review says so. Where the limiter is on, makeup is
  automatic and the voice is not too hot for makeup, the final check and
  verification let the limiter take up to 3 dB off the loudest peaks; the final
  true-peak stage keeps its limit. Elsewhere the earlier rule holds.
- For a microphone that is too hot for any makeup, the review now advises
  lowering the input gain and says by about how much is still missing at the
  lowest loudness. Earlier versions said only that headroom validation did not
  pass.
- Voice Setup verification's tonal check no longer counts what the noise
  suppressor removes as a tonal error. It had rejected setups whose EQ was as
  intended, most often with RNNoise at full strength and with DeepFilterNet. It
  now measures the same recordings with the suppressor switched off, and a
  wrong EQ is still caught.
- Voice Setup verification now says when noise suppression is dulling
  consonants, by about how much, and suggests lowering its strength or, unless
  it is already in use, using DeepFilterNet LL. This is advice and does not
  change the verification result.
- Voice Setup decides whether a capture can be relied on from explicit
  conditions rather than caps on its quality score. This change alone does not
  alter which proposals are recommended; the capture quality shown for a weak
  capture is no longer capped.
- Diagnostics export includes the input RMS and peak levels, so a microphone
  that is silent at rest shows up in a support report.
- The portable bundle no longer ships the unused SciPy `integrate`,
  `interpolate`, `ndimage` and `stats` subpackages, or unused Qt plugins (the
  Direct2D platform plugin, seven image-format plugins, the TLS backends, and
  the touch and network-information plugins). The packaged startup test now
  also runs a real Auto-EQ fit.
- With the Qt Quick view active, the classic widget view is built only when it
  is needed. This is internal; no measured speed-up is claimed.

Live audio processing, sound defaults, controls and preset format are
unchanged. Existing presets retain their settings. The saved window position
gains a maximized flag, and settings from earlier versions load as before.

## Validation and limits

The evidence archive records the exact candidate revision, hashes and automated
qualification results, including portable startup and MSI lifecycle checks.

The loudness fit and the limiter allowance were qualified on held-out simulated
users, judged against predefined gates; the reports are
`voice-setup-loudness-fit-2026-10-report.json` and
`voice-setup-limiter-allowance-narrow-2026-10-report.json` in the
[evaluation evidence](https://github.com/FueledByRedBull/audio-forge/blob/master/evaluation/README.md).
The users are speech in synthetic rooms with recorded noise, not microphone
captures. Among passing proposals the largest limiter gain reduction was 5.62 dB
against the app's 6 dB warning. A broader allowance applied to every proposal
was tested and rejected.

The consonant-dulling advice was checked on held-out simulated users; a lower
default RNNoise strength was tested on the same users and not adopted. The
tonal-check change was judged against predefined gates on the 36 users who got
Auto-EQ settings out of 48 held-out simulated users from a second set of
speakers (108 user-setting pairs;
`voice-setup-verification-tonal-check-2026-10-report.json`); its first design
failed on the first set. The tonal check is still nearly blind to errors below
1 kHz.

The window, meter, Stop button, EQ graph and Voice Setup fixes have automated
source tests that use offscreen rendering and simulated recordings. They do not
establish results with every headset, display arrangement or Windows setup, and
no new physical-audio or graphics-hardware claim is made.

The v1.15.0 audio timing exception remains: one component-event statistic was
1.2151 ms against a 0.5 ms limit; every complete processing kernel in that study
stayed below 10 ms.

Windows 10 (1809 or later) and Windows 11, 64-bit.
