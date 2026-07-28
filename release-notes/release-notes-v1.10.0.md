# AudioForge v1.10.0

## Voice activity detection: corrected adapter and Silero v6.2.1

This release fixes a defect in every previously shipped build: the Silero VAD
adapter sent 512-sample frames without the documented 64-sample rolling context
and looked up the recurrent state output under a wrong optional name, so the
model ran with truncated input and a recurrent state frozen at zero. The
corrected adapter sends the full 576-sample tensor, requires the `stateN`
output, and validates the model contract at startup so an incompatible model
fails loudly instead of silently degrading.

On top of the corrected adapter, the bundled model is upgraded from Silero VAD
v5.1.2 to v6.2.1. The repair and the upgrade were evaluated as independent
decisions on a speaker-disjoint labeled corpus; both passed their retention
gates. Held-out speech-event recall improved from roughly 62% to 96%, noise-only
false openings decreased, and P95 onset latency did not regress. VAD
probabilities are now Platt-calibrated on a dedicated calibration split.

Gating will behave differently after this update because detection is
substantially more reliable. The default gate VAD threshold moves from 0.4 to
0.48. Presets older than v1.10.0 that still contain exactly 0.4 are migrated;
other saved thresholds are preserved. The old format did not record whether an
exact 0.4 came from the default or was deliberately re-entered by the user, so
that ambiguous exact value cannot be distinguished during migration.

## Speech-aware auto makeup gain

Compressor auto makeup no longer decides speech activity from absolute RMS
alone. It fuses the calibrated VAD posterior, VAD availability and freshness,
level relative to the measured noise floor, and noise-reference reliability,
and falls back deterministically to level-based behavior when evidence is stale
or absent. Measurement also exposed and fixed a smoothing-coefficient bug in
block-rate control updates. On controlled fixtures, false makeup activation on
noise-only material dropped from 8.4 dB to 0 dB with unchanged speech
convergence and reduced makeup movement.

## Broader Auto Voice Setup compressor search

The closed-loop compressor calibration now searches threshold, ratio, attack,
and release together under a deterministic 68-simulation budget, scored by a
normalized objective covering loudness error, median and P95 gain reduction,
peak headroom, 2-8 Hz pumping, and silence gain. On the held-out corpus the
median objective improved 10.44% over threshold-only search with 83.33% of captures
improving and no hard-safety regressions.

## Auto-EQ confidence redesign

Auto-EQ no longer treats ordinary phonetic variation as measurement noise.
Measurement uncertainty, coverage, and phonetic representativeness are separate
signals; confidence binds once inside the solver bounds instead of through
duplicated post-solve scaling and caps; refinement can move bands in either
direction within safety bounds; and unsupported bands abstain individually with
a constraint-preserving re-solve instead of discarding the entire
recommendation. Recommendations are less timid where evidence is strong and
still abstain where it is not.

## DPDFNet evaluated and rejected

DPDFNet-2 48 kHz high-resolution was benchmarked against both bundled
DeepFilterNet3 backends on held-out noisy speech. It won every noisy-speech
component gate and met CPU realtime targets, but failed the clean-speech
preservation gates (dropout and spectral distance), so it was rejected and all
production integration, bundled assets, and dependencies were removed. The
noise-suppression backends are unchanged from v1.9.0. The DeepFilterNet
standard-model latency description was corrected from ~40 ms to ~30 ms.

## Compatibility

Existing presets migrate to v1.10.0 automatically. The only value migration is
an exact legacy gate VAD threshold of 0.4 to 0.48; all other thresholds and
settings are preserved.
The public Python API, preset schema, ten-band EQ contract, and bundled model
set are otherwise unchanged.

## Release and package integrity

The Windows release job now obtains and verifies pinned runtime assets before
building or running model-bearing Rust tests. RustSec runs on its pinned Node 24
action revision. Portable bundles include explicit AudioForge, DeepFilterNet,
Silero VAD, and DirectML notices plus a source-version manifest that package
smoke validates, preventing an older `dist` directory from passing as v1.10.0.
