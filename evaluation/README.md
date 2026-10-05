# Evaluation evidence

This directory records why AudioForge's sound-processing defaults are what
they are. Decisions use predefined pass/fail rules applied to held-out
data. A candidate normally replaces the incumbent only when it passes, and an
inconclusive result keeps the incumbent. The audio follow-up below has an
explicit user-approved timing exception; its failed gate remains a failure.
The JSON reports hold configuration, source and asset hashes, aggregate
metrics, the rule, the decision, and its limitations. Per-case detail stays
out of Git (`--details-output` under ignored `models/evaluation-details/`,
or a CI artifact).

## Decisions

| Area | Decision | Evidence |
| --- | --- | --- |
| Noise gate, VAD modes | Keep the existing gate and VAD rules. The historical speech-presence replacement failed in both modes; the current follow-up compensates supported suppressors' wet output separately. | [Historical gate study](#gate-study-october-2026); `gate-speech-presence-2026-10-report.json`; [audio follow-up](audio-followups-2026-10-report.json) |
| Voice Setup gate/suppression tuner | Corrected check ordering: select the frontend with compression bypassed, then calibrate compression and verify the full chain. Continuous speech-then-noise processing checks post-speech noise against the incumbent. The original October study graded an uncalibrated compressor and never changed settings; its rejected relative checks, fitted selector and reference-free onset measures remain rejected. | Current [implementation](../python/mic_eq/analysis/joint_tuning.py) and [regressions](../python/tests/test_joint_tuning.py); historical study: `voice-setup-tuner-2026-10-report.json` |
| Gate with a neural suppressor | Add aligned wet-output compensation for RNNoise and DeepFilter LL while preserving the gated neural input and dry mix. The historical study kept the factory gate: it cost 4–7 dB at phrase onsets but kept long pauses 6–10 dB quieter, and gate off/VAD Assisted failed its rule. The new follow-up has separate onset and quiet-pause evidence. | Current [audio follow-up](audio-followups-2026-10-report.json); historical `gate-policy-2026-10-report.json` |
| Processing order | Gate before noise suppression; de-esser before EQ. | `processing-order-2026-10-report.json` (gate); `processing-order-2026-09-report.json` (de-esser/EQ) |
| Limiter | 0.5 ms lookahead, zero output true-peak overshoot. | `limiter-lookahead-2026-10-report.json`. Its removed predecessors shifted already-aligned audio, so their gain-envelope and transient-shape metrics were invalid; correctly aligned, the three lookaheads score within 0.01 dB. |
| DeepFilter | 30 dB attenuation, beta 0.0. | `deepfilter-hardening-report.json`, `deepfilter-fullband-report.json` |
| VAD | Silero v6.2.1 with independent calibration and multi-speaker validation. | `vad-model-selection-report.json`, `vad-v6.2.1-report.json` |
| Voice preservation | All 78 held-out gate cases and all 60 current EQ cases pass; Natural stays neutral and adaptive tone stays bounded. Doesn't establish perceptual quality or reproduce the live post-yell issue. | `voice-preservation-report.json` (the change against its predecessor, 48 EQ cases); `voice-preservation-2026-09-report.json` (no change since v1.13.0) |
| Warm target | All 60 dialog and 60 held-out EQ cases pass; Warm adds bounded low-mid body. | `warm-auto-eq-report.json` |
| Auto-EQ confidence | Historical calibration of the former absolute-spectrum objective, not of today's bounded tonal adjustment. | `auto-eq-confidence-calibration.json` |
| Compressor | VAD/reliability-driven automatic makeup; all six real-speech gates and the −30 dB aliasing gate (worst −48.2 dB) pass. Calibration searches threshold only; the expanded search failed held-out qualification and stays disabled. | `auto-makeup-real-speech-2026-09-report.json`, `dynamics-aliasing-2026-09-report.json`, `compressor-expanded-search-2026-09-report.json`, `compressor-control-report.json`, `compressor-search-report.json` |
| De-esser detector | Historical transient setup detector: all 96 generated corpus clips classified correctly, with no false positives. This scores detection, not the audio change or the new persistent-shape model. | `deesser-corpus-v1-report.json` (coefficient fit); `deesser-corpus-v1-2026-09-report.json` (re-run); current [audio follow-up](audio-followups-2026-10-report.json) |
| De-esser on real speech | Add persistent notch-complement power concentration with a shared spectral-background reference, plus a signed-curvature setup model. The earlier study's no-change decision remains historical: its setup model enabled 65% of harsh, 62% of bright and 42% of ordinary simulated voices, while Auto reduced the sibilance band by only ~0.3–0.7 dB. The current change has separate benefit and preservation evidence. | Current [audio follow-up](audio-followups-2026-10-report.json); historical `deesser-real-speech-2026-10-report.json` |
| Resampling | 128-tap Blackman product resampler. | `resampler-quality-report.json` |
| Manual EQ types | Bell, notch, shelf and pass filters with selectable slopes. | `eq-filter-types-report.json` |
| Auto-EQ candidate pool | Nested wider pools and sparse type selection rejected. | `eq-candidate-pool-report.json`, `sparse-auto-eq-filter-report.json` |
| Auto-EQ fitter | Keep the dynamic band layout. A fixed-grid, gains-only fitter tracked the intended curve more closely on average (0.06 vs 0.13 dB RMS) but failed its rule on 56 of 2,084 cases: Warm with Broad smoothing fit worse, and 14 recommendations flipped between apply and reduced (October 2026). | `auto-eq-fixed-grid-2026-10-report.json` |
| Auto-EQ Adaptive layer | Keep. Its voice-balance inputs saturate for 97–100% of real voices, so Adaptive is effectively a gentler fixed version of each preset and where it measures cannot matter; fixing the offsets explicitly failed its equivalence rule on noisy captures and was no closer to the clean-voice curve (October 2026). | `auto-eq-tone-layer-2026-10-report.json` |
| Microphone correction | Not automated. Estimating a mic's coloration from 10 s of speech with a population prior made it worse (+0.18 dB RMS), because speaker, room and noise differences (2–9 dB per band) are as large as the colorations (October 2026). | `mic-coloration-prior-2026-10-report.json` |
| EQ stages | Independent correction and tone stages (requested product change); 12 cases pass safety, cost, schema, and zero-added-latency gates. | `correction-tone-product-report.json` |
| Joint gate/model tuning | September qualification: all 66 cases pass; DeepFilter incumbents stay within their family because switches to RNNoise failed clean-speech checks. | `product-joint-tuning-2026-09-report.json`, `product-joint-tuning-deepfilter-2026-09-report.json`, `product-joint-tuning-deepfilter-ll-2026-09-report.json` |
| RNNoise backend | Keep `nnnoiseless`; upstream Xiph was slower and regressed clean preservation. | `rnnoise-backend-comparison.json` |
| DPDFNet | Rejected; it failed clean-speech preservation. Historical failures aren't independently reproducible from this checkout. | `dpdfnet-vs-deepfilternet3-report.json`, `dpdfnet-official-evalset-report.json` |
| ONNX Runtime | Official CPU-only 1.23.2 matched the preserved baseline across 498 captures. | `onnxruntime-cpu-probe.json` |
| Release integrity | Reviewed package paths gate the candidate; archive sidecars own artifact facts. | `release-bundle-path-baseline.json`, `archive-format-benchmark.json` |

The historical reports retain their measured pipeline and source identities.
The tuner ordering repair leaves compression calibration and full-chain
verification in place; it does not qualify the selectors rejected by the older
study. Compressor ownership changes preserve render/state parity, and calibration
still searches threshold only. Shared native K-weighting and NumPy correlation
passed numerical and decision checks; no speed improvement is claimed. Package
size reporting found a larger portable tree, and startup/idle-memory comparison
did not qualify. These implementation checks do not establish a new held-out
audio-quality result.

## Audio follow-up (October 2026)

The current source keeps the gate before noise suppression and sends the same
gated samples into the neural models. RNNoise and DeepFilter LL use gate controls
aligned with their delay to compensate only the wet output; the dry mix and
declared latency are preserved. Standard DeepFilter receives no compensation.
Its onset and quiet-pause checks pass, but they are reused from the original
24-case EARS run through a source/native bridge; the gate was not rendered
again. That run's overall result failed on an older de-esser recommendation
check, not on its gate checks.

Auto de-essing measures notch-complement power concentration against a shared
spectral background, and Voice Setup adds a signed spectral-curvature model for
persistent evidence. The recorded fresh benefit and preservation gates for
ordinary and bright speech pass. Manual, disabled and zero-cap paths remain
unchanged.
The Windows audio worker now registers with MMCSS Pro Audio after initialization
and before streams start, propagates registration failure through startup
cleanup, and checks same-thread reversion on normal shutdown.

Adoption uses an explicit user-approved timing exception. One mandatory
component-event p99 was **1.2151 ms**, exceeding the unchanged **0.5 ms** limit;
the component qualification remains failed, and 9 of 18 routes failed the
original frozen timing bounds. All **294,000** complete kernels
in that study were below **10 ms**, with a candidate maximum of **5.8616 ms**.
A separate bounded prefix diagnostic did not reproduce the spike and does not
establish its cause or overturn the failure. Earlier timing failures and the
unresolved total added-route cost remain part of the evidence. The
[follow-up report](audio-followups-2026-10-report.json) is the current adoption
record; historical reports retain their original sources, decisions and limits.
Final main-source/native binding and package validation remain separate
requirements. These corpus and synthetic-route results establish neither
universal/perceptual quality nor a live-hardware or package pass.

## Gate study (October 2026)

**Corpus.** 24 EARS speakers (12 female, 12 male; 48 kHz anechoic, CC BY-NC
4.0), 30 s of freeform speech each. EARS trims silences, so phrases are cut
at their own inter-word gaps and separated by log-uniform 0.25–2.5 s pauses,
which also gives exact speech/pause labels. Noise is channel 1 of eight 48 kHz
DEMAND environments (CC BY 4.0 on Zenodo): kitchen, living room, washing machine,
office, meeting room, hallway, cafeteria, traffic. Each speaker is mixed with
two environments at 5, 15 and 25 dB SNR (144 mixtures), split 12/12 speakers
by gender into training and held-out halves. `evaluate_gate_corpus.py fetch`
builds it (manifest SHA-256 `67dac07c…6105`). It replaces the earlier order
evaluation's 12 clips, which were all 0 dB SNR, three babble-type noises and
16 kHz source audio.

**Metrics,** against the clean reference: ESTOI, SI-SDR, onset level (first
100 ms of each phrase relative to its body), pause and post-speech tail level
relative to the noisy input, and floor modulation (tail minus deep-pause
level, which measures pumping). Intervals are 95% bootstraps clustered by
speaker. The VAD control is causal: each 10 ms gate block sees the last 32 ms
Silero window that ended by its first sample, as the live worker can deliver.

**Processing order** (`processing-order-2026-10-report.json`). With the
default Threshold Only gate, moving the gate after suppression clips phrase
onsets by 7.9 dB (RNNoise), 3.3 dB (DeepFilter) and 4.0 dB (DeepFilter LL)
and lowers ESTOI on every backend. In VAD Assisted it gains ESTOI (+0.005 to
+0.008) but makes RNNoise tails 0.85 dB louder and costs DeepFilter LL
0.62 dB SI-SDR. The order is global, so the gate stays before suppression.
The previous order report decided on a "pumping" metric that measured 2–8 Hz
in the gate gain, which is the syllable rate; on this corpus it flips sign.
On top of a neural suppressor, every gate trades onsets for pause
attenuation: the default gate adds 0.8 dB of pause attenuation over RNNoise
alone and costs about 1 dB at onsets.

**Speech-presence candidate** (`gate-speech-presence-2026-10-report.json`,
rejected). The candidate replaced the VAD modes' heuristics with the OM-LSA
gain form G = Gmin^(1 − p) (Cohen & Berdugo, 2001), p being one logistic over
the VAD posterior and, in VAD Assisted, the level above the noise floor.
Weights were fitted on the training speakers. Each mode's bias was the
midpoint of the biases matching the current gate's pause attenuation there
through RNNoise and DeepFilter LL. The rule in `evaluate_gate_corpus.py`'s
docstring was fixed first: on held-out speakers, with the VAD available and
unavailable, on both backends, ESTOI non-inferior (lower bound ≥ −0.002),
pauses and tails under +1 dB louder, SI-SDR lower bound above −0.1 dB, plus a
material win on RNNoise.

| Held-out, VAD available | ESTOI | SI-SDR dB | Onsets dB | Pumping dB | Failed |
| --- | --- | --- | --- | --- | --- |
| VAD Only, RNNoise | +0.035 | +1.07 | +14.8 | −15.2 | pauses up to +1.35 dB louder |
| VAD Only, DeepFilter LL | +0.047 | +3.01 | +16.6 | −19.5 | — |
| VAD Assisted, RNNoise | +0.005 | +0.02 | +0.3 | +0.7 | — |
| VAD Assisted, DeepFilter LL | +0.003 | −0.18 [−0.45, +0.03] | −0.7 | +2.0 | SI-SDR |

Both modes also failed with the VAD unavailable (DeepFilter LL SI-SDR −0.12,
interval including zero; VAD Only on RNNoise, pauses up to +1.03 dB louder).
VAD Only's speech gains are large, so a retry is worth it: the gaps are its
pause level on RNNoise and its fallback. A retry must use fresh EARS
speakers, since these 12 held-out speakers have now informed the design. An
earlier pass of this study used VAD results interpolated across windows that
had not finished yet, which overstated onsets; its numbers are superseded.

Reproduce with `evaluate_gate_corpus.py qualify --incumbent <build>
--candidate <build>`. The incumbent is commit `4f77bb0` (extension SHA-256
`587cfc9e…558c`); the candidate is that commit plus the patch, fit and
calibration preserved under ignored
`models/evaluation-details/gate-speech-presence-2026-10/` (extension SHA-256
`644bd363…d692`).

## Corpora

Evaluation corpora live under ignored `models/` and are never release assets.
Each fetcher pins sources and hashes: `evaluate_gate_corpus.py fetch`
(EARS/DEMAND), `evaluate_voice_setup.py fetch` (VoiceBank/DEMAND, CC BY 4.0;
EARS is non-commercial, so nothing shipped is fitted on it),
`fetch_dpdfnet_evaluation_assets.py` (DPDFNet EvalSet, also the speech corpus
for the Auto-EQ, makeup, DeepFilter and RNNoise evaluations),
`fetch_cross_take_corpus.py`, `fetch_deepfilter_fullband_corpus.py`,
`fetch_vad_child_validation_corpus.py`, and `build_vad_evaluation_corpus.py`.

## Provenance and hygiene

`python/tools/check_evaluation_hygiene.py` rejects absolute paths, stale
source hashes, malformed audible-change contracts, privacy leaks, and
oversized reports. A report with `source_revision` was committed unchanged at
that revision and its source hashes are checked there. A released report is
checked at the first release tag that shipped it unchanged, so later code
changes don't make released evidence stale. New or edited reports are checked against
the working tree. Merge branches that carry reports with a merge commit, since
squash or rebase drops the pinned commits. Superseded reports and the
evaluators of closed decisions (DPDFNet EvalSet, RNNoise backends, Auto-EQ
candidate pool, sparse filters, Auto-EQ confidence) were removed after v1.14.0;
their reports stay, and the tools remain in that release's source. Several
evaluators default `--report` to a tracked historical file: pass an explicit
path and add new measurements beside it. The older
`compressor-control-report.json`, `compressor-search-report.json`, and
`rnnoise-backend-comparison.json` predate complete source provenance; they
support their decisions but don't reproduce the current checkout. The
September `dynamics-aliasing-2026-09-report.json`,
`compressor-expanded-search-2026-09-report.json` and
`deesser-corpus-v1-2026-09-report.json` come from evaluators that record
neither source nor native build identity. All September measurements used one
extension built from `b8afe65`, SHA-256
`a4cb583960c8ca643686ec6e993eb056cc2db9e4399febbbc786041548dab931` (recorded in
the voice-preservation report); for those three reports that identity is
stated here, not verifiable from the reports. The de-esser corpus report
scores the setup-time detector, not the audio change, and no tracked
evaluator compares a same-input limiter render across builds.

## Interpretation

Objective metrics establish behavior only for the recorded corpus, hardware,
and configuration. Unobserved devices, routes, operating systems and voices
are limitations, not inferences. Human listening isn't a release gate.
