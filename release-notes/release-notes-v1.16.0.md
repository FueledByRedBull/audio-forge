AudioForge 1.16.0 checks your input level before a setup take, shows clipping
on the meters, tells you in plain words what to do about Health numbers and
setup results, adds a readout and reset to the EQ graph, and fixes the final
check in Auto Voice Setup, which had not been running inside the app.

## Downloads

- **Installer:** [AudioForge-v1.16.0-win64.msi](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.16.0/AudioForge-v1.16.0-win64.msi)
- **Portable:** [AudioForge-v1.16.0-win64-ultra.7z](https://github.com/FueledByRedBull/audio-forge/releases/download/v1.16.0/AudioForge-v1.16.0-win64-ultra.7z)

Choose one. The corresponding-source and evidence archives support
redistribution and verification; SHA256SUMS.txt checks the downloaded files.
AudioForge remains unsigned. Virtual audio cable setup is described in the
[README](https://github.com/FueledByRedBull/audio-forge#set-up).

## New

- Voice Setup and Auto-EQ start with a three-second level check. It tells you
  if the microphone is too hot, too quiet or silent. For a hot or quiet voice
  it gives the level it heard and the change to make. A good level continues
  with one click. A hot or quiet level can be checked again or continued
  anyway; silence can only be checked again.
- The IN and OUT peak lines catch the highest sample between updates (OUT:
  before the limiter), so a short transient still moves them. A CLIP tag shows once six samples at full scale arrive within three seconds,
  which is the flat top Voice Setup counts as clipping. One touch of full scale
  only warns that peaks are too high. The OUT tag stays dark while the limiter
  holds the output at its ceiling, and lights when the limiter is off or does
  not run (Bypass, Raw monitor).
- Health gives the number and the action. For example: "Input peaks at -0.3 dB,
  close to the top of the microphone's range, so loud sounds can be cut off and
  sound distorted. Lower the microphone level by about 6 dB in Windows or on the
  interface." The device, dropout, buffer, backend and recovery lines carry
  their numbers too.
- Setup results are written in plain words. Each reason a recording was not
  used says what was withheld and what to do. The confidence percentage is gone
  from the headline; the capture-quality score is in a tooltip.
- The Review page draws the proposed EQ over the current one.
- The EQ graph reads out the frequency and gain under the pointer. Double-click
  a band handle, or press Delete on the selected band, to reset it as one undo
  step. In Bypass and Raw monitor the graph says "EQ not running". In Bypass it
  draws the fixed input filtering the signal still gets (adaptive cleanup is not
  drawn); in Raw monitor the line is flat.
- The diagnostics export (schema version 3) adds the last setup result and which
  view and graphics backend the app is using. It contains no message text.

## Fixes

- The final check and compressor calibration in Auto Voice Setup never ran
  inside the app. The dialog passed settings the analysis rejected, and the
  error was swallowed, so every release since v1.13.0 showed "Advisory
  recommendations only". They run now, and the review names the error if one
  happens again.
- A hot microphone is no longer a dead end in Auto-EQ. When no full-size
  correction meets the headroom rule and the current EQ is already over it, the
  dialog offers the largest correction that is no worse than what you have. If
  none qualifies, it says the microphone level is too high and by about how much.
  When the correction's own boost is most of the excess, that amount counts only
  your recording's part, and the advice says the boost is the cause.
- One stray full-scale sample no longer rejects a take. Clipping now needs a
  flat top of about 125 microseconds or more, or more than 20 separate hits.
- Auto-EQ refuses clipped, silent and non-finite takes as soon as they finish.
- In Bypass, Voice Setup now checks the normal chain and restores Bypass
  afterwards. Before, verification "accepted" a render of nothing.
- The Review page stays filled after the verification take, says when Apply
  will switch the EQ on, and shows makeup gain for manual-makeup compressors.
  "Compare Recording" before Apply uses the EQ that Apply writes.
- Controls come back after a failed or rejected capture. Retake and Compare come
  back after a failed or rolled-back verification take.
- Consonant-dulling advice appears only when noise suppression is on, and it
  suggests DeepFilterNet LL only to RNNoise users.
- The Auto-EQ tone modes are named "Subtle" and "Full strength". Stored values
  are unchanged. The help text no longer says the EQ is calibrated to your voice
  and microphone.
- During a take, the level message names the loudest recent peak. It used to
  name the latest fraction of a second, which could fall between words.
- After the second Voice Setup recording passes, the main button opens the
  final comparison. It used to be greyed out.
- The Drops line in Health says why it needs attention.
- At the smallest window size nothing runs under the level meters, and long
  preset names are shortened.

## Changes

- The EQ graph draws each enabled band's curve, the correction layer as a dashed
  line, the fixed input filter (the DC blocker and an 80 Hz high-pass) as a
  dotted line, and the thick curve for what the signal really gets: the input
  filter plus the EQ. Adaptive cleanup, which can raise the high-pass to 100 or
  120 Hz and notch hum, is not drawn. Shelf and filter handles sit on their own
  curves.
- Spectrum analysis is faster, with identical results.
- SciPy is no longer part of the app. The Auto-EQ fit uses NumPy, and the
  portable bundle is smaller. On 2,084 test cases, 96.2 % of fitted responses
  are within 0.05 dB of SciPy's, about the same share as SciPy against itself
  (96.7 %). The largest difference was 2.30 dB, against 2.44 dB for SciPy against
  itself.
- Auto-EQ can propose slightly different settings than 1.15.3. Of 2,071 fits,
  34 differ by more than 0.25 dB, against 40 for SciPy compared with itself, and
  19 by more than 1 dB, as many as for SciPy against itself.
- Release checks run as parallel jobs, and unused evaluator tools and code were
  removed. The release workflow no longer repeats the seeded contention stress
  test in debug mode; it still runs in release mode. None of this changes the
  app.

Live audio processing, sound defaults and the preset format are unchanged.
Existing presets retain their settings, and settings from earlier versions load
as before.

## Validation and limits

The evidence archive records the exact candidate revision, hashes and automated
qualification results, including portable startup and MSI lifecycle checks.

The NumPy fit was compared with SciPy on 2,084 cases, with SciPy against
itself as the yardstick, because SciPy is not stable against itself. The port
changed no failure or recommendation status; the yardstick changed one. The
check on allowed status changes had nothing to measure, as there were none
(`auto-eq-solver-port-2026-10-report.json` in the
[evaluation evidence](https://github.com/FueledByRedBull/audio-forge/blob/master/evaluation/README.md)).
The product resampler was re-checked on the current source and all 11
predefined checks pass (`resampler-quality-2026-10-report.json`).
The clipping rule was tried on 12 VoiceBank takes hard-clipped digitally by 0 to
6 dB. It refuses every take clipped by 1 dB or more and accepts 17 of the 24
takes clipped by 0.25 or 0.5 dB. The CLIP tag lights for 16 of the 29 cases it accepts, because
the tag counts how often full scale was reached and the rule judges damage. No
microphone captures of real clipping were tested.

The level check, clipping rules, Health advice and EQ graph changes have
automated tests that use offscreen rendering and simulated recordings. They do
not establish results with every microphone, audio interface, display or Windows
setup, and no new physical-audio or graphics-hardware claim is made. The new
peak and clip counters add atomic writes in the audio threads and are covered by
the allocation and realtime source-scan tests; no new timing measurement is
claimed.

The v1.15.0 audio timing exception remains: one component-event statistic was
1.2151 ms against a 0.5 ms limit; every complete processing kernel in that study
stayed below 10 ms.

Windows 10 (1809 or later) and Windows 11, 64-bit.
