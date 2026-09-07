# Product follow-up after v1.12.0

Planning checklist, not a release gate or a claim that these features are implemented.
Recheck each item after [PR #62](https://github.com/FueledByRedBull/audio-forge/pull/62)
lands. Keep this draft separate from release work; do not merge solely to store a plan.
When implementation resumes, link the corresponding issue/PR beside its checkbox
and use that issue for detailed decisions and acceptance criteria.

Goal: choose a microphone and destination, calibrate the intended settings, hear
and choose the result, edit it, save it, and restore it predictably during daily use.

Sources: [Analyze Missing Features](https://chatgpt.com/share/6a9f4593-2228-83eb-9056-e78fd2ac1371),
[Analyse Missing Features](https://chatgpt.com/share/6a9f459c-8910-83eb-a88a-261e8af6b568),
and the owner's STUFF.md consolidation, reviewed September 8, 2026.
The reviews mostly describe `de2612f`; PR #62 subsequently fixed several findings.
The checklist below is self-contained and does not require access to those chats.

## Already addressed in PR #62

These are implemented on that PR, not necessarily on the default branch. Preserve
their regression coverage; reopen only if current evidence shows a remaining defect.

- [x] Match Voice Setup analysis, applied limiter settings, and control readback.
- [x] Run verification in a cancellable worker and reject incomplete candidates.
- [x] Restore temporary calibration reliability only within a valid capture context.
- [x] Clear stale Auto-EQ results on retake/failure, retain their target identity,
  and enable EQ within the same undoable application.
- [x] Update saved preset identity/Last Used and refresh startup/route menus on open.
- [x] Invalidate EQ diagnostics after relevant manual changes.
- [x] Provide device selectors inside onboarding and correct Minimal Processing,
  full-preset saving, latency, and bypass descriptions.

## 1. Preset identity and scope

- [ ] Show the current preset name and saved/modified state. Verify save, save-as,
  load, manual edit, undo/redo, and reopen; preserve explicit startup/route overrides.
- [ ] Distinguish replacement **EQ templates**, **complete sound presets**, and
  **calibration targets** in labels and actions. EQ-only application must leave
  gate, suppressor, dynamics, and output protection unchanged.
- [ ] Reuse the duplicated Voice/Bass Cut/Presence recipes through the existing
  catalog. Define a complete-preset baseline once and test resolved settings;
  avoid copying every default or creating a second preset subsystem.
- [ ] Describe Warm & Clear as a strong replacement contour and distinguish bass
  trimming from rumble high-pass filtering. Do not retune recipes without evidence.

## 2. Consistent calibration and settings application

- [ ] Make EQ-only and Full Voice Setup share capture ownership, cancellation,
  candidate validity, review, application, and rollback rules. Preserve useful
  shortcuts and scope-specific analysis options, including optional noise capture.
- [ ] Carry proposed settings, allowed scope, capture/options identity, and verified
  stages with each candidate. Apply/Save must not reconstruct these from changed widgets.
- [ ] Reuse validation and application paths for bulk preset/calibration/restore
  operations, read back accepted values, and create one coherent undo operation.
  Keep manual controls responsive; extract shared code only where duplication warrants it.
- [ ] Check both flows for failed analysis, stale signals, cancellation, partial
  application failure, changed inputs/options, and preservation of unrelated stages.

## 3. Controlled listening comparison — decision before implementation

- [ ] Revisit held decisions [#24](https://github.com/FueledByRedBull/audio-forge/issues/24)
  and [#25](https://github.com/FueledByRedBull/audio-forge/issues/25). Decide whether
  a small in-app audition or the broader experimental harness is warranted.
- [ ] If approved, compare current/proposed settings using the same recorded passage,
  aligned delay and optional speech-based loudness matching. Show the actual level
  difference separately; retain an untouched reference and an explicit keep/reject choice.
- [ ] Bound playback, protect output, release recordings on close, and verify matching
  accuracy and click-free switching. Disclose which stages are rendered: downstream
  comparison is not full live-chain verification or proof of listening preference.
  Add per-stage/milder alternatives only if basic comparison proves useful.

## 4. Independent tone, intensity, and loudness

- [ ] Remove the hard coupling between tonal target and requested LUFS. Use-case
  presets may initialize tone, compression intensity, and loudness independently.
- [ ] Keep Gentle/Balanced/Dense intensity understandable after calibration; show
  actual gain reduction and mark manual changes as customized. Keep expert controls.
- [ ] Distinguish one-time calibration, continuous adaptation, and manual control.
  Verify changing tone alone preserves the loudness goal through analyze/apply/save/load.

## 5. Microphone configuration and evidence

- [ ] Associate channel interpretation and cleanup preferences with stable device
  identity. Define behavior for new, missing, and changed endpoints and route overrides.
- [ ] Keep transferable sound presets separate from device setup and temporary
  room-noise evidence. Define invalidation before adding durable calibration metadata;
  do not persist raw audio, device handles, runtime buffers, or stale confidence.
- [ ] Test switching microphones, reconnecting, restarting, and migration of existing
  global preferences without silently transferring one microphone's assumptions.

## 6. Everyday controls and onboarding

- [ ] Add user mute separate from temporary calibration/recovery mute. Calibration
  completion, cancellation, and errors must never clear the user's mute choice.
- [ ] Show microphone, destination, active preset, transmission/processing state,
  and a compact health summary; retain detailed diagnostics for investigation.
- [ ] Consider opt-in tray operation and mute shortcut; distinguish close, quit,
  start-with-Windows, and start-processing behavior. Keep audio activity unmistakable.
- [ ] Guide route selection, speech-level/clipping checks, destination reception,
  and calibration. Keep latency measurement optional under advanced diagnostics.
- [ ] Replace competing bypass/raw checkboxes with one clear processing-mode choice
  if it improves use. Preserve conditioning, transport, sanitization, and protection;
  make raw diagnostic operation conspicuous.

## 7. Calibration and DSP integration experiments

- [ ] Evaluate suppression choices together with gating and downstream processing
  instead of treating downstream verification as a complete automatic tuning solution.
- [ ] Investigate adjusting character without discarding microphone correction.
  Revisit the existing correction/tone experiment before proposing another EQ stage.
- [ ] Define clean-speech preservation, noise, gate-tail, clipping/headroom, latency,
  and CPU gates before changes; compare representative held-out material and listening
  results where approved. Retain the incumbent when benefit is not established.

No audit here establishes a need to replace the EQ, compressor, limiter, or DSP
architecture. A new suppressor, virtual microphone driver, broad framework, or
cosmetic redesign is not a prerequisite for this checklist.

## Separate compatibility follow-up

- [ ] Expand exact-artifact evidence to Windows 10, analog/built-in inputs, actual
  44.1 kHz capture, and physical reconnect/default-device/sleep-resume as resources
  become available. Keep untested configurations explicit; do not reinstate them
  as v1.12.0 publication gates without a separate support-policy decision.
