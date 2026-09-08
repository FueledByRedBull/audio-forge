# Optional ideas and experiments after v1.12.0

Separate from the concrete work in [PR #63](https://github.com/FueledByRedBull/audio-forge/pull/63).
Planning only: no feature or experiment below is implemented by this PR.
Keep this PR draft and unmerged until implementation and review are authorized.

For each item, define the problem, expected benefit, acceptance criteria, and a
bounded comparison against current behavior. Record adopt, reject, or defer with
a reason. Preserve useful evidence and discard unsuccessful prototype code.
None of these items is a merge requirement for PR #63 or a v1.12.0 release gate.

## A. Controlled listening comparison

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

## B. Optional everyday features

- [ ] Consider opt-in tray operation and a mute shortcut.
- [ ] Replace competing bypass/raw checkboxes with one clear processing-mode choice
  if it improves use. Preserve conditioning, transport, sanitization, and protection;
  make raw diagnostic operation conspicuous.
- [ ] Consider persistent calibration storage only after establishing usefulness
  and invalidation rules; never retain raw audio or stale evidence implicitly.

## C. DSP and calibration experiments

- [ ] Evaluate suppression choices together with gating and downstream processing
  instead of treating downstream verification as a complete automatic tuning solution.
- [ ] Investigate adjusting character without discarding microphone correction.
  Revisit the existing correction/tone experiment before proposing another EQ stage.
- [ ] Define clean-speech preservation, noise, gate-tail, clipping/headroom, latency,
  and CPU gates before changes; compare representative held-out material and listening
  results where approved. Retain the incumbent when benefit is not established.

## D. Dependency and package experiments

- [ ] Measure a SciPy-free prototype before adopting it: compressed package size,
  startup time, held-out Auto-EQ results, resampling/filtering/latency correctness,
  numerical stability, and maintenance cost. The v1.12.0 manifest attributes
  about 90 MB uncompressed to SciPy; this is not the compressed saving. Preserve
  SciPy unless the complete replacement demonstrates worthwhile benefit.
- [ ] Audit corresponding-source closure against the pinned Windows CPU build.
  Classify required build/license inputs and optional upstream test/web tooling,
  measure archive sizes, and prove the reduced set still rebuilds the shipped
  dependencies before removing entries. Content-addressed deduplication is
  implemented in PR #63; do not prune by filename alone.
- [ ] Measure installation time and dependency weight before splitting the dev
  lock into test, security, and packaging environments. Retain hashed resolution
  and audit coverage; avoid multiple lockfiles without a measured payoff.

## E. Conditional structural changes

- [ ] When changing large UI/DSP modules, extract a concrete responsibility and
  retain its regression coverage. Split tests by behavior when navigation suffers;
  file length alone does not justify controllers or a wholesale rewrite.
- [ ] Consolidate artifact metadata/payload manifests if it reduces producer and
  consumer code together. Preserve archived-release readability, checksums,
  producer identity, and rejection of missing/corrupt/mismatched candidate data.
  Keep the existing five public release downloads.

No new framework, universal schema, percentage-deletion target, or permanent
feature flag is required merely to conduct an experiment. Preserve existing
security properties, historical evidence, and regression coverage.
