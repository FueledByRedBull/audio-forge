# Optional ideas and experiments after v1.12.0

Separate from the concrete work in [PR #63](https://github.com/FueledByRedBull/audio-forge/pull/63).
Planning only: no feature or experiment below is implemented by this PR.
PR #64 is closed and unmerged. This document remains the optional backlog;
updating it does not reopen the PR or authorize these experiments. The local
`PROJECT_LEDGER.md` contains the complete audit/plan inventory and dispositions.

For each item, define the problem, expected benefit, acceptance criteria, and a
bounded comparison against current behavior. Record adopt, reject, or defer with
a reason. Preserve useful evidence and discard unsuccessful prototype code.
None of these items is a merge requirement for PR #63 or a v1.12.0 release gate.

## A. Controlled listening comparison

- [ ] O01 - Revisit held decisions [#24](https://github.com/FueledByRedBull/audio-forge/issues/24)
  and [#25](https://github.com/FueledByRedBull/audio-forge/issues/25). Decide whether
  a small in-app audition or the broader experimental harness is warranted.
- [ ] O02 - If approved, compare current/proposed settings using the same recorded passage,
  aligned delay and optional speech-based loudness matching. Show the actual level
  difference separately; retain an untouched reference and an explicit keep/reject choice.
- [ ] O03 - Bound playback, protect output, release recordings on close, and verify matching
  accuracy and click-free switching. Disclose which stages are rendered: downstream
  comparison is not full live-chain verification or proof of listening preference.
  Add per-stage/milder alternatives only if basic comparison proves useful.

## B. Optional everyday features

- [ ] O04 - Consider opt-in tray operation and a mute shortcut.
- [ ] O05 - Replace competing bypass/raw checkboxes with one clear processing-mode choice
  if it improves use. Preserve conditioning, transport, sanitization, and protection;
  make raw diagnostic operation conspicuous.
- [ ] O06 - Consider persistent calibration storage only after establishing usefulness
  and invalidation rules; never retain raw audio or stale evidence implicitly.

## C. DSP and calibration experiments

- [ ] O07 - Evaluate suppression choices together with gating and downstream processing
  instead of treating downstream verification as a complete automatic tuning solution.
- [ ] O08 - Investigate adjusting character without discarding microphone correction.
  Revisit the existing correction/tone experiment before proposing another EQ stage.
- [ ] O09 - Define clean-speech preservation, noise, gate-tail, clipping/headroom, latency,
  and CPU gates before changes; compare representative held-out material and listening
  results where approved. Retain the incumbent when benefit is not established.

## D. Dependency and package experiments

- [ ] O10 - Measure a SciPy-free prototype before adopting it: compressed package size,
  startup time, held-out Auto-EQ results, resampling/filtering/latency correctness,
  numerical stability, and maintenance cost. The v1.12.0 manifest attributes
  about 90 MB uncompressed to SciPy; this is not the compressed saving. Preserve
  SciPy unless the complete replacement demonstrates worthwhile benefit.
- [ ] O11 - Audit corresponding-source closure against the pinned Windows CPU build.
  Classify required build/license inputs and optional upstream test/web tooling,
  measure archive sizes, and prove the reduced set still rebuilds the shipped
  dependencies before removing entries. Content-addressed deduplication is
  implemented in PR #63; do not prune by filename alone.
- [ ] O12 - Measure installation time and dependency weight before splitting the dev
  lock into test, security, and packaging environments. Retain hashed resolution
  and audit coverage; avoid multiple lockfiles without a measured payoff.

## E. Conditional structural changes

- [ ] O13 - When changing large UI/DSP modules, extract a concrete responsibility and
  retain its regression coverage. Split tests by behavior when navigation suffers;
  file length alone does not justify controllers or a wholesale rewrite.
- [ ] O14 - Consolidate artifact metadata/payload manifests if it reduces producer and
  consumer code together. Preserve archived-release readability, checksums,
  producer identity, and rejection of missing/corrupt/mismatched candidate data.
  Keep the existing five public release downloads.

No new framework, universal schema, percentage-deletion target, or permanent
feature flag is required merely to conduct an experiment. Preserve existing
security properties, historical evidence, and regression coverage.

## F. Reconciled historical recommendations and holds

These items are retained decisions or conditional proposals, not unfinished
requirements for PR #63. A new justified task and acceptance gate must precede
implementation. Concrete verification work is in PR #63's product-followup document.

- [ ] O15 - AF-18: consider a unified calibration coordinator, independent
  settings/application service, audio-session controller, and pure Rust offline
  kernel boundary only when a real change exposes duplicated ownership. Existing
  shared helpers do not constitute the complete proposed controller architecture.
- [ ] O16 - #20/#21: independent speech-event annotations remain not planned
  until a product decision requires human labels. Listening infrastructure
  (#24/#25) is already covered by section A, not a second active project.
  The old roadmap's consented, redistributable multi-speaker/microphone/room/
  language corpus, blind de-esser ratings and held-out refit are part of this
  same hold, not approved data collection.
- [ ] O17 - Teliko listening comparison remains unperformed, not failed. Reopen
  only with one fixed raw voice take, matched incumbent/candidate renders, safe
  loudness alignment, and a blind comparison. Do not invent preference evidence.
- [ ] O18 - #43: DPDFNet remains rejected and absent. A revisit needs the exact
  artifacts and an independent runtime; historical dropouts cannot be attributed
  solely to the model when the old custom runner/output is unavailable.
- [ ] O19 - #44: any future update feature must use authenticated notifications;
  do not replace runtime models in place. #45/#46: band splitting or a new neural
  backend needs predefined clean-speech, noise, latency, CPU, and stability gates.
- [ ] O20 - #47: cross-platform work remains held; assess Linux before macOS.
  #48: per-application automatic presets require a new demonstrated use case.
- [ ] O21 - Process isolation is conditional on reproducing a literally
  non-returning native backend with a pinned artifact. Returned errors, worker
  death, and starvation already have recovery paths; do not add an isolation
  framework for an unobserved hang.
- [ ] O22 - Broader physical DC compensation needs a demonstrated problem on
  another interface; the recorded Razer route did not justify it. Keep current
  phase-safe routing and its transition protection.
- [ ] O23 - Preserve rejected/retained DSP decisions: keep `nnnoiseless`, the
  current processing order, combined ten-band Auto-EQ, and 0.5 ms limiter lookahead.
  Revisit Xiph, wider/sparse EQ pools, correction/tone splitting, or other models
  only with a new predefined gate and complete comparable evidence.
- [ ] O24 - Historical January polish ideas: reassess whether a dedicated VAD
  showcase preset provides anything beyond current gate modes and presets.
  Do not recreate old version-only migrations, obsolete public names, or removed
  planning infrastructure merely to satisfy an old checklist.
- [ ] O25 - Historical de-esser/latency extensions: split-band de-essing and
  automatic Voice Setup already exist. Full automatic hardware-loopback detection
  without user routing remains optional; do not claim a measured route delay
  identifies physical one-way latency or removes that delay from the audio path.
- [ ] O26 - Historical Qt/Semgrep dependency refresh proposals are conditional
  on current advisories, reachable functionality, measured benefit and the
  existing hashed dependency/package checks. Old suggested versions are not
  upgrade targets. Preserve required notices, source inputs and regression gates.

Historical sources: [roadmap](https://github.com/FueledByRedBull/audio-forge/blob/a99d7ff3fe99fed40ab545efd934aa5148cc1d3f/ROADMAP.md),
[listening protocol](https://github.com/FueledByRedBull/audio-forge/blob/011888346f9071f8fd4d2b3e0292428d99a40513/evaluation/LISTENING_PROTOCOL.md),
and [readiness review](https://github.com/FueledByRedBull/audio-forge/blob/0767172b268f8e85e662885fb0b156ea3215d63f/docs/v2-readiness.md).
The old protocol's listener counts, trial/score rules and acceptance thresholds
remain historical proposals; they are not reinstated as release requirements.

"Unchecked" in this planning document means held/not adopted, not approval to
build. Existing negative decisions remain valid until their reopening condition
changes. No dependency replacement, new service, or feature flag is implied.
