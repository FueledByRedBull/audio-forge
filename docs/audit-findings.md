# AudioForge correctness audit

## Scope

Source: `885a5e9`, AudioForge 1.11.4. Five agents were explicitly launched with
`gpt-5.6-luna`, reasoning effort `max`: DSP, runtime, configuration, UI/analysis,
and packaging/release. The main reviewer ran checks, challenged initial claims,
and independently reproduced the principal failures. Checkpoints and focused
second passes were used. This is not a guarantee of finding every defect.

Application source and release configuration were not changed. The existing
extension was rebuilt for current-source tests; no new third-party dependency
was installed. This report is the only intended source-tree addition. No real
microphone capture, reconnect, remote workflow, or publication was performed.

## Findings

### F01 - P1: Cancel can be followed by hidden microphone capture

Locations: `python/mic_eq/ui/calibration_dialog.py:561`,
`python/mic_eq/ui/calibration_dialog.py:566`,
`python/mic_eq/ui/calibration_dialog.py:1007`,
`python/mic_eq/ui/voice_setup_dialog.py:448`,
`python/mic_eq/ui/voice_setup_dialog.py:529`.

Both dialogs schedule a 100 ms single-shot capture callback. Reject cleans up
the current tap but neither cancels that callback nor invalidates its state.
Real headless Qt dialogs with a mocked running processor reproduce: start a
phase, reject immediately, process events for 160 ms; capture then starts and
recovery suppression becomes true again. Calibration also reproduces after the
creating function returns and garbage collection runs. This is unexpected
capture after Cancel, not evidence of transmission or disk storage. Repair:
own/cancel the delayed timer and invalidate the phase before cleanup.

### F02 - P1: A differently named preset silently replaces an existing preset

Locations: `python/mic_eq/config_parts/presets.py:564`,
`python/mic_eq/ui/main_window.py:3505`.

Filename sanitization is many-to-one; save uses unconditional `os.replace()`
without UI collision confirmation. Executed reproduction: saving `A/B`, then
`A:B`, produces only `A_B.json`, containing the second preset. Repair: separate
explicit overwrite from creation and confirm or reject occupied destinations.

### F03 - P1: Failed automatic restart can strand processing stopped

Locations: `rust-core/src/audio/processor/recovery.rs:54`,
`rust-core/src/audio/processor/recovery.rs:98`,
`python/mic_eq/ui/main_window.py:2620`,
`python/mic_eq/ui/main_window.py:2140`.

After both selected/default restart attempts fail, Rust is stopped with recovery
pending. Python returns from diagnostics whenever stopped, never servicing the
retry. Start remains disabled; Stop also returns early. Executed mocked UI
reproduction: three ticks cause zero recovery calls and Stop does not restore
Start. Native transitions were source-traced. Repair: service pending recovery
while stopped and reconcile actual engine/UI states.

### F04 - P2: A small malformed preset can prevent startup

Locations: `python/mic_eq/config_parts/presets.py:141`,
`python/mic_eq/config_parts/presets.py:630`,
`python/mic_eq/config_parts/presets.py:634`,
`python/mic_eq/ui/main_window.py:1239`.

Menu construction calls `list_presets()`, whose boundary omits parser/deep-copy
recursion failures. Executed reproduction: an 11,001-byte JSON preset with
1,100 nested objects propagates `RecursionError`. Condition: malformed data in
the enumerated preset directory, not a demonstrated remote attack. Repair:
bound nesting or normalize parsing failures to `PresetValidationError`.

### F05 - P2: Partial legacy migration is silently treated as complete

Location: `python/mic_eq/config_parts/shared.py:43`.

`copytree()` can create a partial destination before raising. The error is
ignored; destination existence prevents future retries. Executed failure
injection copies one file then raises `OSError`; the next lookup never retries
and remaining presets are absent. Old files remain manually recoverable in the
legacy directory. Repair: stage migration or retain a resumable failure state
without overwriting newer user files.

### F06 - P2: A valid device name corrupts its generated route key

Locations: `python/mic_eq/config_parts/shared.py:197`,
`python/mic_eq/config_parts/shared.py:222`,
`python/mic_eq/config_parts/app_config.py:258`,
`python/mic_eq/config_parts/app_config.py:272`.

Legacy `||` splitting runs before JSON parsing. A generated structured route
containing device name `USB Mic || Desk` is misparsed. Executed `AppConfig`
round trip loses the original binding key; latency profiles share the parser.
Condition: name-backed identities when endpoint IDs are unavailable; stable
endpoint keys normally omit names. Repair: parse structured keys first.

### F07 - P2: Refresh disables two controls' signals

Locations: `python/mic_eq/ui/main_window.py:1696`,
`python/mic_eq/ui/main_window.py:1820`.

Refresh blocks input-channel/input-cleanup selectors but only unblocks the
device selectors. Startup separately unblocks all four, hiding this until a
later Refresh. Executed with real Qt combos: both remain signal-blocked and
changing them emits zero events. Repair: restore each prior signal-block state
on all exits, or avoid blocking controls that refresh never modifies.

### F08 - P2: Opening a route discards stable endpoint identity

Locations: `python/mic_eq/ui/device_selection.py:97`,
`rust-core/src/audio/input.rs:328`, `rust-core/src/audio/output.rs:137`,
`rust-core/src/audio/processor/recovery.rs:49`.

Selections contain endpoint IDs, but startup forwards only friendly name and
old ordinal. Native code enumerates again and opens the nth matching name;
recovery retains the same stale selection scheme. Simulated reproduction:
select A at ordinal zero, reorder duplicate names to B,A; the actual Python
helper still forwards zero and a simulated native enumerator opens B. No real
reconnect was tested. Repair: preserve/verify endpoint identity through opening
and recovery, failing closed when the selected endpoint cannot be resolved.

### F09 - P2: Background Auto-EQ still blocks Python UI callbacks

Locations: `python/mic_eq/ui/analysis_worker.py:81`,
`python/mic_eq/analysis/auto_eq_parts/headroom.py:102`,
`rust-core/src/audio/processor/python_api.rs:472`.

The worker's native simulation retains the GIL during DSP computation, starving
Python GUI handlers despite QThread. Executed against the rebuilt release
extension: a 10-second, 48 kHz input took 299 ms to simulate and caused a 301 ms
gap in a separate 5 ms Python heartbeat. The UI reviewer also reproduced Qt
timer starvation. Repair: detach expensive native work after acquiring inputs
safely; another Python thread alone does not solve this.

### F10 - P2: Offline simulations accept NaNs and panic on invalid rates

Locations: `rust-core/src/audio/processor/python_api.rs:27`,
`rust-core/src/audio/processor/python_api.rs:198`,
`rust-core/src/audio/processor/python_api.rs:480`,
`rust-core/src/dsp/compressor.rs:393`.

Offline validation is inconsistent with stricter typed EQ APIs. Executed on
the rebuilt extension: NaN makeup gain returns NaN audio from
`simulate_auto_makeup_control`; a positive 40 Hz rate passed to
`simulate_auto_eq_chain` raises `PanicException` because a clamp's 20 Hz minimum
exceeds its 18 Hz maximum, even with compression disabled. Scope: exported
offline/evaluation APIs, not proof the real-time UI sends invalid settings.
Huge allocation cases were not executed. Repair: validate finite settings/audio
and supported rate bounds before constructing DSP or allocating buffers.

### F11 - P2: Promotion does not establish qualification run origin

Locations: `.github/workflows/release-promote.yml:56`,
`.github/workflows/release-promote.yml:72`,
`.github/workflows/release-promote.yml:111`,
`python/tools/release_provenance.py:368`.

Dispatch descriptions require successful specific qualification workflows, but
run IDs are not checked for expected workflow identity or successful conclusion.
Verification accepts report status/hash/commit without establishing its producer.
Executed local validator reproduction: wrong repository/ref/run metadata and a
four-field passing report yield no errors when archive hash and commit match.
The approved archive SHA is still an effective content check: this is missing
qualification provenance, not demonstrated replacement of approved bytes or
bypass of GitHub authorization. Repair: check producer workflow, run conclusion,
repository/ref/commit, and report role before promotion.

### F12 - P2: An existing draft can remain unpublished after promotion

Location: `.github/workflows/release-promote.yml:146`.

The workflow creates a release only if `gh release view` fails. Otherwise it
uploads assets without checking draft state or publishing it, and never asserts
the final public state. Evidence: source-traced branch, not a live GitHub test.
Condition: a pre-existing draft for that tag. Repair: explicitly handle draft
versus public releases and verify the promised state after successful uploads.

### F13 - P2: Discarded Voice Setup analysis can overwrite the reset state

Locations: `python/mic_eq/ui/voice_setup_dialog.py:104`,
`python/mic_eq/ui/voice_setup_dialog.py:720`,
`python/mic_eq/ui/voice_setup_dialog.py:1028`,
`python/mic_eq/ui/voice_setup_dialog.py:1090`.

Retake is available after capture while analysis runs. Teardown waits 1.5 seconds
but ignores timeout, clears the reference, and neither cancels the Voice Setup
worker nor disconnects/invalidates its results. Executed with the real Qt worker
and a 2-second mocked analysis: accept Retake after 50 ms; the dialog is idle at
1.75 seconds, then the discarded result changes it back to completed with
Apply Voice Setup enabled. No crash was observed or claimed. The calibrator's
separate worker does check a stop flag before success, so this finding is
specific to Voice Setup. Repair: retain ownership until termination and reject
results from canceled/obsolete capture generations.

### F14 - P2: EQ re-enable restores stale filter state without a transition

Locations: `rust-core/src/dsp/eq.rs:370`,
`rust-core/src/dsp/eq.rs:475`,
`rust-core/src/audio/processor/dsp_loop.rs:617`,
`rust-core/src/audio/processor/dsp_loop.rs:659`.

Disabled EQ is skipped; filter state freezes. Re-enable only switches a boolean.
There is no EQ-specific wet/dry ramp in the surrounding production path; the
other output recovery fades are armed by discontinuities, not this control.

Main-reviewer native reproduction linked a small stdin-compiled Rust harness
against the current project rlib, with no production-source change. Warm an
80 Hz, amplitude-0.01 sine through an 80 Hz +12 dB low shelf (Q=0.707), bypass
for 128 samples, then re-enable at 48 kHz. Native results:

```text
EQ re-enable step:       -0.015851116
Stale-state error:       -0.014204288
Post-limiter max step:    0.015851116
```

The current native true-peak limiter at -1.5 dB leaves the abrupt step intact
because the signal is below its ceiling. This demonstrates a waveform
discontinuity/click risk, not a listening-test claim. Repair: add a click-safe
enable transition and a native regression test; validate audible changes using
the project's objective evidence requirements.

## Validation

| Check | Result |
| --- | --- |
| Ruff, application/tests/tools | Pass |
| Pyright | 0 errors, 0 warnings |
| cargo fmt | Pass |
| Current-source Rust unit tests | 310 passed, 5 ignored |
| Rust integration contention | 1 passed, 1 hardware ignore |
| Rust doc test | 1 passed |
| Release seeded contention | Pass |
| Native EQ transition harness | Reproduces F14, including final limiter |
| All-target Clippy, warnings denied | Pass |
| Maturin release rebuild | Pass using existing ONNX cache |
| Full Python suite | 543 passed, 1 temp-layout assumption failure |
| Version/workflow/source-package checks | Pass |
| Release asset verification | Fails: target/release/DirectML.dll missing |

The Python failure is `test_corpus_manifests_never_embed_machine_absolute_paths`
at `python/tests/test_vad_evaluation_tools.py:115`. Workspace-local `--basetemp`
made its alleged external file repository-relative, so the helper correctly
returned a relative path rather than only a basename. All assertions passed
with a synthetic external path and filesystem write mocked. The full suite is
not reported as green.

Initial Rust execution failed with `STATUS_DLL_NOT_FOUND`; a command-local PATH
entry for the existing Python base resolved python312.dll loading. Initial
offline release linking failed for ONNX; pointing at the existing library cache
resolved it. Neither remedy changed project configuration.

## Limits and rejected claims

- No portable-bundle build, real-device lifecycle test, loopback measurement,
  hardware qualification, or new held-out audible-quality evaluation. Missing
  release-path DirectML is a local prerequisite, not a source regression.
- No networked advisory scan, Semgrep download, remote workflow verification,
  or published-artifact reconciliation.
- No concrete callback-allocation, atomic-snapshot, or FFI memory-safety defect
  established. Passing tests do not prove absence.
- Pending probe cancellation and recording-complete versus explicit-stop
  semantics were not called UI bugs without a failing normal-user flow.
- Mutating an already-built diagnostics snapshot is not evidence of an actual
  UI export leak. That initial claim was dropped.
- An initial non-48 kHz live-DSP mismatch claim was withdrawn: live input is
  resampled to the fixed 48 kHz processing/recording domain, matching the graph.
  Generic offline analysis still assumes 48 kHz in its predictor without
  enforcing that contract; the reviewer measured a 0.536 dB prediction mismatch
  for a generated bell when supplied 96 kHz. This is not a current live UI bug.
- Noise-reference reliability is capture-derived, expiring evidence rather than
  an ordinary persisted user control. Its omission from presets was not called
  a schema defect without a product requirement to persist evidence validity.
- Malicious custom ONNX models, knowingly mismatched opt-in model overrides,
  mutable upstream tools, and deliberate human qualification attestations were
  not presented as confirmed shipped-runtime defects.
- Structural bundle checks and self-generated provenance do not independently
  compare bundled assets to the pinned release manifest. Normal builds already
  verify source assets; final extracted-asset equality is a hardening item, not
  a proven current artifact substitution.
- Traversal probes did not escape with the installed extractor. This does not
  certify every archive format or extractor.
