# AudioForge 1.12.1

This update improves first setup, preset handling, calibration recovery, and daily microphone controls.

## Changes

- Show saved/modified preset identity and distinguish complete sound presets from EQ templates.
- Recover from invalid Last Used presets with defaults and a visible explanation.
- Keep user mute independent of calibration/recovery, with explicit save-failure feedback.
- Remember microphone-specific preferences and keep device selectors aligned with running streams.
- Guide first-run routing, speech/clipping checks, and voice calibration; keep latency optional.
- Preserve calibration capture identity, rollback and Undo; reject stale or incomplete results.
- Handle failed route, latency, onboarding and window-state writes without claiming settings were saved.
- Reset compact health indicators correctly after processing stops.
- Deduplicate corresponding-source inputs, reuse existing validation helpers, and consolidate public downloads into five files.
- Improve README downloads, routing instructions, screenshots, badges and help navigation.

## Downloads

Choose the per-user MSI installer or extract the full portable archive and run
`AudioForge.exe`. Python and Rust are included in the packaged runtime as needed;
no developer toolchain installation is required. The release also supplies
corresponding source, a consolidated evidence archive, and `SHA256SUMS.txt`.
Original AudioForge source is MIT; combined PyQt6 distributions use GPLv3 with
retained dependency notices and corresponding source.

## Validation and compatibility

Publication requires successful automated software, portable/MSI, corresponding-source,
and exact archived candidate validation. The same validated bytes are promoted.
See the evidence archive for revision-specific results and digests.

Windows 10 (1809+) and Windows 11 x64 remain compatibility targets. Earlier
candidate hardware measurements covered Windows 11 USB and virtual routes at
48 kHz; those results do not qualify this release's binary. This version has
not completed a new physical hardware soak. Windows 10, analog/built-in inputs,
actual 44.1 kHz capture, physical reconnect, default-device changes, and
sleep/resume remain unqualified. Linux and macOS are not supported.
