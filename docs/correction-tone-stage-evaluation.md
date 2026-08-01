# Correction and tone EQ stage experiment

This document pre-registers the roadmap issue 28 candidate. The shipped
incumbent has one ten-band stage. The candidate has two ordered ten-band stages:

```text
de-esser -> Auto-EQ correction -> user/preset tone -> compressor -> limiter
```

Auto-EQ may replace only correction. Manual controls and tone presets may
replace only tone. Global EQ bypass skips both stages; each band retains its
typed bypass. Correction and tone responses are rendered independently by the
native Rust implementation and add in dB when shown as one curve.

## Candidate schema and migration

The evaluation-only shape contains `correction` and `tone`, each holding the
same validated ten-band v2 schema. Migrating an incumbent preset places its
existing bands in tone and creates a flat correction stage, preserving response
exactly. Applying Auto-EQ later replaces correction while leaving the canonical
tone payload byte-for-byte unchanged.

This shape is not accepted by production preset loading unless the candidate is
retained. The existing per-band `stage` token is not permission to activate a
second runtime stage.

## Frozen gates

Every gate is mandatory:

1. Combined-to-two-stage migration and correction-only rendering must match the
   incumbent native response within 1e-9 dB.
2. Auto-EQ replacement must preserve the tone payload exactly, and malformed or
   incorrectly sized stages must fail validation.
3. Native renders and the downstream chain must remain finite; true peak may
   not exceed the effective limiter ceiling by more than 0.05 dB, and P95
   limiter gain reduction must remain at or below 3 dB on the normalized clean
   corpus arm.
4. Both stages must report zero added algorithmic latency. Candidate P95 EQ
   realtime factor must be at most 0.01 and at most 2.25 times the incumbent
   one-stage path.
5. At least eight native-48-kHz corpus cases and all four frozen tone profiles
   (flat, presence, warm, bass cut) must complete.
6. Bounded undo must round-trip both stages as one immutable configuration
   transaction if the candidate is integrated.
7. Because response parity demonstrates safety rather than an objective product
   benefit, the two-stage path must also reduce P95 EQ runtime by at least 5%
   relative to the one-stage incumbent before its added architecture is justified.

The decision is conjunctive. Objective parity alone cannot justify doubling the
stage architecture; without a measurable benefit, the simpler incumbent wins.

## Decision behavior

`python/tools/evaluate_correction_tone_stages.py` writes
`evaluation/correction-tone-stage-report.json`. If any gate fails, no second
native stage, controls, preset shape, or undo payload is retained. Production
presets must then reject non-`combined` stage values rather than silently
ignoring them.

## Recorded result

Eleven of twelve native-48-kHz corpus cases completed; one short pair was
honestly excluded after incumbent Auto-EQ rejected it as unclear. The candidate
preserved migration response and canonical tone payload exactly, covered all
four tone fixtures, remained finite, added zero latency, produced no limiter
gain reduction or true-peak overshoot, and measured a 1.258 P95 EQ-runtime
ratio with a 0.00154 P95 realtime factor.

The candidate measured a 1.258 P95 runtime ratio rather than the required 5%
improvement. It is rejected: AudioForge retains one combined ten-band stage,
does not add second-stage DSP or controls, and schema v2 rejects non-`combined`
stage tokens so no preset can request ignored behavior.
