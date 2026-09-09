# Concrete follow-up work after v1.12.0

Implementation checklist for [PR #63](https://github.com/FueledByRedBull/audio-forge/pull/63).
Unchecked items still require implementation or verification.
Optional ideas and experiments were recorded in the closed, unmerged planning
[PR #64](https://github.com/FueledByRedBull/audio-forge/pull/64). Listening comparison,
tray operation, and background controls remain outside this implementation pass.
Preserve the fixes in merged [PR #62](https://github.com/FueledByRedBull/audio-forge/pull/62).
The owner authorized merging after final CI and package validation on September 10.
This document owns concrete maintenance and verification follow-ups; the local
`PROJECT_LEDGER.md` inventories all audit items, completed work, and optional holds.

Goal: choose a microphone and destination, calibrate the intended settings, hear
and choose the result, edit it, save it, and restore it predictably during daily use.

Sources: [Analyze Missing Features](https://chatgpt.com/share/6a9f4593-2228-83eb-9056-e78fd2ac1371),
[Analyse Missing Features](https://chatgpt.com/share/6a9f459c-8910-83eb-a88a-261e8af6b568),
and the owner's STUFF.md consolidation, reviewed September 8, 2026.
The reviews mostly describe `de2612f`; PR #62 subsequently fixed several findings.
The checklist below is self-contained and does not require access to those chats.

## Already addressed in PR #62

These are implemented on the default branch. Preserve
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

- [x] Show the current preset name and saved/modified state. Verify save, save-as,
  load, manual edit, undo/redo, and reopen; preserve explicit startup/route overrides.
- [x] Recover from an invalid Last Used preset without blocking startup. Clear
  the unusable reference and retain a visible explanation while using defaults.
- [x] Distinguish replacement **EQ templates**, **complete sound presets**, and
  **calibration targets** in labels and actions. EQ-only application must leave
  gate, suppressor, dynamics, and output protection unchanged.
- [x] Reuse Voice/Bass Cut/Presence recipes through the existing catalog.
- [x] Define a complete-preset baseline once and test resolved settings; avoid
  copying every default or creating a second preset subsystem.
- [x] Describe Warm & Clear as a strong replacement contour and distinguish bass
  trimming from rumble high-pass filtering. Do not retune recipes without evidence.

## 2. Consistent calibration and settings application

EQ-only and Full Voice Setup retain separate dialog orchestration and apply
settings through the existing panels. The items below cover shared helpers and
tested application/rollback behavior; a unified calibration session and an
independent application service remain unimplemented.

- [x] Make EQ-only and Full Voice Setup share capture ownership, cancellation,
  candidate validity, review, application, and rollback rules. Preserve useful
  shortcuts and scope-specific analysis options, including optional noise capture.
- [x] Carry proposed settings, allowed scope, capture/options identity, and verified
  stages with each candidate. Apply/Save must not reconstruct these from changed widgets.
- [x] Reuse validation and application paths for bulk preset/calibration/restore
  operations, read back accepted values, and create one coherent undo operation.
  Keep manual controls responsive; extract shared code only where duplication warrants it.
- [x] Check both flows for failed analysis, stale signals, cancellation, partial
  application failure, changed inputs/options, and preservation of unrelated stages.

## 3. Independent tone, intensity, and loudness

- [x] Remove the hard coupling between tonal target and requested LUFS. Use-case
  presets may initialize tone, compression intensity, and loudness independently.
- [x] Keep Gentle/Balanced/Dense intensity understandable after calibration; show
  actual gain reduction and mark manual changes as customized. Keep expert controls.
- [x] Distinguish one-time calibration, continuous adaptation, and manual control.
  Verify changing tone alone preserves the loudness goal through analyze/apply/save/load.

## 4. Microphone configuration and evidence

- [x] Associate channel interpretation and cleanup preferences with stable device
  identity. Define behavior for new, missing, and changed endpoints and route overrides.
- [x] Keep transferable sound presets separate from device setup and temporary
  room-noise evidence. Define invalidation before adding durable calibration metadata;
  do not persist raw audio, device handles, runtime buffers, or stale confidence.
- [x] Test switching microphones, reconnecting, restarting, and migration of existing
  global preferences without silently transferring one microphone's assumptions.
- [x] Defer device refresh while processing, so displayed endpoints and applied
  route preferences continue to match the running stream.

## 5. Everyday controls and onboarding

- [x] Add user mute separate from temporary calibration/recovery mute. Calibration
  completion, cancellation, and errors must never clear the user's mute choice.
- [x] Apply live mute before saving the preference. Failed or protected config
  writes must leave the requested live state applied and report that it was not saved;
  a failed native mute must never be reported as successfully muted.
- [x] Report failed saves for route preferences, latency calibration, and setup
  progress. Keep setup usable and let calibration saves be retried; never report
  an in-memory change as persisted.
- [x] Show microphone, destination, active preset, transmission/processing state,
  and a compact health summary; retain detailed diagnostics for investigation.

- [x] Clarify close, quit, start-with-Windows, and start-processing behavior.
  Keep audio activity unmistakable.
- [x] Guide route selection, speech-level/clipping checks, destination reception,
  and calibration. Keep latency measurement optional under advanced diagnostics.
  Require explicit destination confirmation after the live-stream check, and
  expose the existing user mute control inside setup. Resume saved latency steps
  without including latency in the default numbered journey. Resume skipped voice
  setup directly, and show live levels and clipping feedback in the route step.

## Separate compatibility follow-up

- [ ] Expand exact-artifact evidence to Windows 10, analog/built-in inputs, actual
  44.1 kHz capture, and physical reconnect/default-device/sleep-resume as resources
  become available. Keep untested configurations explicit; do not reinstate them
  as v1.12.0 publication gates without a separate support-policy decision.

## Reconciled audit follow-ups (September 10)

These are follow-ups, not newly invented merge gates. "Needs verification" is
not a confirmed defect. Historical audits describe their original revisions;
their old unchecked boxes do not override current evidence or support policy.

- [ ] C01 - After merging, check the final packaged app on the owner's setup:
  first launch, route selection, mute, save/reopen, clean quit, reconnect, and
  sleep/resume. Record the exact archive digest and observed result; physical
  actions require the owner and must not be inferred from simulated tests.
- [ ] C02 - Complete the separate compatibility follow-up above when equipment
  is available: Windows 10, built-in/analog inputs, actual 44.1 kHz capture,
  physical reconnect, default-device changes, and sleep/resume (AF-09 / #35).
  Existing 48 kHz runs do not qualify 44.1 kHz; changing Windows Default Format
  alone does not force the processor to open at that rate.
- [ ] C03 - Reconcile remaining acceptance evidence for AF-16: identify existing
  executable failure/comment/unreachable-step tests before adding any missing
  cases. Decide whether a standard workflow linter adds coverage worth its cost.
  Keep parsed policy checks; do not build a general workflow framework.
- [ ] C04 - Perform the previously proposed all-ref history audit for exposed
  credentials and large/generated objects using redacted results. No completed
  scan is claimed. Rotate a discovered credential before considering history
  repair; do not rewrite history for cosmetic cleanup.
- [ ] C05 - Verify current branch/tag/publishing protections, administrative
  bypass, force-push/deletion policy, and dependency-alert reconciliation.
  Record observed settings; documentation or an old API response is not proof
  of current server policy (AF-20). Do not alter protections as part of checking.
- [ ] C06 - Locate remaining measured acceptance evidence before making broader
  claims: supported clean-build/contributor paths, warm/cold audit-cache timings,
  and the old de-esser CPU / physical latency-repeatability targets (AF-07/19/20).
  Native heartbeat and allocation-scope regressions already exist; preserve them
  rather than reopening their fixed defects or rerunning benchmarks without need.
- [ ] C07 - Verify the old generic offline non-48-kHz EQ-prediction limitation
  and exact extracted-runtime asset equality before treating either as closed.
  The former was not reproduced in the fixed-48-kHz live UI; these were audit
  limitations/hardening suggestions, not demonstrated shipped exploits.
- [ ] C08 - After the merge, refresh the clean main checkout and retire obsolete
  worktrees only after preserving branch-only notes and unpublished evidence.
  Reclaim disposable caches/build output selectively. Keep `.venv313`, its base
  interpreter under `target/python313-nuget`, runtime assets, models, corpora,
  and the protected rewrite directory. A Git ignore rule is not deletion proof.
- [ ] C09 - Review the combined-distribution inventory and corresponding-source
  arrangements when shipping changed dependencies (AF-02/08). The GPL distribution
  policy, inventories, pinned origins, and source packaging exist; no independent
  legal review or bit-for-bit rebuild of every upstream dependency is claimed.

Historical plans are superseded where they call for deleting the now-required
`df.dll`, treating all `target/` content as disposable, using `build.ps1`,
restoring `ROADMAP.md`, or forcing obsolete version numbers. The de-esser and
latency wizard are implemented; physical repeatability/performance targets
remain subject to C01/C02/C06. Optional proposals belong in PR #64's document.

## Maintenance and size follow-up

Preserve release evidence, regression tests, both supported models, and Git history.

- [x] Bound candidate retention to three days and reuse compiled caches across
  source-only edits. Keep separate job/profile caches; measure build time before
  replacing them with another compiler cache tool.
- [x] Add size comparisons to the existing release-provenance path. It already
  records every bundled file's size and total bytes; reuse that manifest, compare
  with the preceding release, and report the largest changes in the job summary.
  Avoid another public attachment or mandatory release gate.

- [x] Define local cleanup around disposable outputs and retained inputs first.
  Keep active Python installations/virtual environments, models, corpora, and
  unpublished evidence out of broad deletion targets. Prefer existing build-tool
  cleanup commands; reuse existing tools where sufficient. The one-time cleanup is complete.

- [x] Consolidate evaluation plumbing only where repeated code is demonstrated.
  Reuse the existing WAV helper and provenance utilities; keep named experiments
  and historical decisions. Check consumers and reproduction requirements before
  archiving rejected experiments such as DPDFNet.

## Release-contract consolidation

The implementation review correctly distinguishes live configuration from frozen
measurement evidence. Preserve historical hashes and corruption-rejection tests;
do not delete verification merely because two boundaries inspect the same bytes.

- [x] Leave action-pin validation in workflow policy, removing its duplicate from
  package smoke. Compare source-manifest ORT identities with `release-assets.json`
  instead of maintaining another literal digest in the test.
- [x] Reuse one existing asset loader/validator across live consumers, including
  the ORT probe. Eliminate hardcoded expected digests; preserve path, duplicate,
  size/hash, and archive-origin validation. Keep historical probe inputs/results
  tied to the actual tested runtime; do not relabel old evidence after tool edits.
- [x] Reduce package-smoke/workflow source-string checks one invariant at a time,
  after proving equivalent behavioral or artifact coverage. Keep action pins,
  permissions, active blocking gates, source verification, and commit bindings.

- [x] Clarify generated source-manifest ownership and DeepFilter recipe versus
  build-attestation ownership; derive repeated live facts from their owner.
  Preserve exact source closure and actual toolchain/output identities.
- [x] Simplify version checks around declared metadata. UI and preset versions
  already derive from `__version__`; retain cross-language metadata consistency
  without treating prose examples as additional version authorities.

No new build package, universal evaluation schema, or percentage-deletion target
is required. Extract shared code only when it removes demonstrated duplication.

## September 8 ponytail findings: implemented in this PR

| ID | Disposition |
| --- | --- |
| PT-01 | Deduplicate corresponding-source inputs by content identity. |
| PT-02 | Publish five assets, consolidating sidecars into evidence and checksums. |
| PT-03 | Reuse matching remote digests on promotion preflight; retain final byte checks. |
| PT-04 | Use ordinary imports for test tools where isolation is unnecessary. |
| PT-05 | Reuse Qt slider/spinbox binding without removing stage-specific behavior. |
| PT-06 | Reuse the WAV reader while retaining caller-specific validation. |
| PT-07 | Reuse settings serialization for preset application. |
| PT-08 | Use native combo lookup for simple string identities. |
| PT-09 | Reuse catalog recipes for EQ buttons. |
| PT-10 | Remove the unused recorded-audio getter and its getter-only test. |

All ten are implemented, not ten pending deletion requests. The original audit's
line/byte savings were estimates or historical measurements, not new package
claims. Required validation, source records, and retained evaluations remain.

## Validate PR #63 before merging

Record commit-specific results on PR #63 and its CI/Release package runs,
so recording successful builds does not require another source commit.

- Build a fresh portable package and MSI from the final PR revision.
- Exercise the changed source-packaging, evidence-generation, and candidate
  validation paths; verify installation and applicable upgrade behavior.
- Validate promotion behavior without publishing or creating a tag.
- Review the final diff and CI results, then merge under the September 10 authorization.
