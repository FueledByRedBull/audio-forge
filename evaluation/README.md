# Evaluation evidence

This directory stores compact decision records, not raw benchmark dumps.
New tracked reports retain experiment configuration, source/asset hashes,
aggregate metrics, predefined gates, the resulting decision, and limitations.
Per-case detail is written only when an evaluator's optional
`--details-output` argument is supplied and should stay under ignored
`models/evaluation-details/` or in a CI artifact.
Routine test results and screenshot-generation reports belong under ignored
`build/`, not in this directory.

## Current decisions

| Area | Evidence | Outcome |
| --- | --- | --- |
| DeepFilter | `deepfilter-hardening-report.json`, `deepfilter-fullband-report.json` | Retain 30 dB attenuation and beta 0.0. |
| VAD | `vad-model-selection-report.json`, `vad-v6.2.1-report.json` | Retain Silero v6.2.1 with independent calibration and multi-speaker validation. |
| Voice preservation | `voice-preservation-report.json`; `voice-preservation-2026-09-report.json` | All 78 held-out gate cases and 48 EQ cases pass predefined regression limits. Natural remains neutral; adaptive tone changes stay bounded. Against v1.13.0, the simplified gate state leaves measured gate energy retention unchanged in all 78 cases (0.0 dB delta), and all 60 current EQ cases pass. This does not establish perceptual quality or reproduce the live post-yell issue. |
| Warm target and capture validation | `warm-auto-eq-report.json` | All 60 dialog cases and 60 held-out EQ cases pass. Clear-speech validation is independent of fitting smoothing; Warm adds bounded low-mid body. |
| Auto-EQ confidence | `auto-eq-confidence-calibration.json` | Historical calibration for the former absolute-spectrum fitting objective; not a perceptual validation of the current bounded tonal adjustment. |
| Compressor control | `auto-makeup-real-speech-report.json`, `compressor-control-report.json`, `compressor-search-report.json`; `auto-makeup-real-speech-2026-09-report.json`, `dynamics-aliasing-2026-09-report.json`, `compressor-expanded-search-2026-09-report.json` | Retain VAD/reliability-driven makeup; all six real-speech gates and the -30 dB aliasing gate (worst -48.2 dB) pass after the presence-band and envelope changes. Production calibration searches the threshold only. The expanded search failed held-out qualification (9 of 12 captures measurable, 0% median improvement, 25% improved, safety failed) and stays disabled. |
| Processing order | `processing-order-report.json`; `processing-order-2026-09-report.json` | Retain gate before suppression and de-esser before EQ; unchanged after the de-esser body-reference change. |
| Limiter | `limiter-lookahead-report.json`; `limiter-lookahead-2026-09-report.json` | Adopt 0.5 ms lookahead after corrected paired, delay-flushed scoring; the ramped lookahead gain still selects 0.5 ms with zero output true-peak overshoot. |
| Resampling | `resampler-quality-report.json` | Retain the 128-tap Blackman product path. |
| Manual typed EQ | `eq-filter-types-report.json` | Retain manual bell/notch/shelf/pass types and selectable slopes. |
| Auto-EQ candidate pool | `eq-candidate-pool-report.json`, `sparse-auto-eq-filter-report.json` | Reject the tested nested wider pools and sparse type-selecting candidate. |
| EQ stage split | `correction-tone-product-report.json`; historical `correction-tone-stage-report.json` | Retain the explicitly requested independent stages: 12 cases pass safety, EQ-kernel cost, schema and zero-added-latency gates. The historical proposal was closed before this product request. |
| Joint gate/model tuning | `product-joint-tuning.json`, `product-joint-tuning-deepfilter-ll.json`, `product-joint-tuning-deepfilter.json`; `product-joint-tuning-2026-09-report.json`, `product-joint-tuning-deepfilter-ll-2026-09-report.json`, `product-joint-tuning-deepfilter-2026-09-report.json` | All 66 cases pass at `b31dc8a`; 19 candidates applied and 47 incumbents retained. DeepFilter incumbents stay within their family after unrestricted switches to RNNoise failed clean-speech checks. On the PR #68 build all 66 cases pass again, but only 4 candidates are applied (RNNoise 2, DeepFilter LL 1, DeepFilter 1) instead of 19. A same-environment v1.13.0 RNNoise run (recorded in PR #68, not tracked) reproduces the previous 6, so the difference comes from this PR: 8 of 22 RNNoise selections change (6 now retain the incumbent, 2 newly apply a candidate), each within 0.044 of clean-error change, consistent with scoring through the updated compressor. |
| RNNoise | `rnnoise-backend-comparison.json` | Retain `nnnoiseless`; upstream Xiph was materially slower and regressed clean preservation. |
| DPDFNet | `dpdfnet-vs-deepfilternet3-report.json`, `dpdfnet-official-evalset-report.json` | Rejected and absent; historical clean failures are not independently reproducible from this checkout. |
| ONNX Runtime backend | `onnxruntime-cpu-probe.json` | Official CPU-only 1.23.2 matched the preserved 3.12 baseline across 498 captures and 20,908 frames. |
| Release integrity | `release-bundle-path-baseline.json` | Reviewed package paths gate the candidate; exact archive sidecars and digest-bound qualification own artifact facts. |

The `python/tools/evaluate_*.py` commands regenerate current measurements;
historical decisions retain their stated reproduction limits.
The older `compressor-control-report.json`, `compressor-search-report.json`,
`deesser-corpus-v1-report.json`, `dynamics-aliasing-report.json`, and
`rnnoise-backend-comparison.json` preserve historical results with incomplete
source provenance. Their binary or corpus hashes, where present, do not identify
the complete generator and application source used. They support the recorded
decisions, not an exact reproduction or validation of the current checkout.
Preserve those measurements; record exact source and asset identities when
generating replacement evidence, rather than assigning guessed provenance.
Reports with `source_revision` preserve the original committed measurement;
hygiene verifies both the unchanged report and its source hashes at that commit.
The three current joint-tuning reports were measured from clean source
`b31dc8aa2516cd0c56c1593af9677211b42a5457`, recorded in
`measurement_source_revision`; their source hashes are verified at that commit.
Native/DLL hashes identify the measured binaries and were compared locally;
Git source-history validation alone cannot verify those binary objects.
The previous reports and per-case evidence remain in Git at `76ed1e9`.
Their declared joint-tuning implementation hash could not be reconciled with
measurement revision `d64957c`; preserve them as historical measurements,
not exact-source qualification or a reproducible baseline for the new reports.
New joint-tuning runs write compact, model-specific reports; optional
`--details-output models/evaluation-details/joint-tuning.json` keeps full case details.
Corpora and model assets are hash-pinned under ignored `models/`
directories and are never bundled merely because they exist locally.
`python/tools/check_evaluation_hygiene.py` rejects absolute paths, stale source
hashes, malformed audible-change contracts, privacy leaks, and oversized
tracked reports.

## PR #68 re-qualification

The September 2026 `*-2026-09-report.json` files re-run the existing evaluators
on the PR #68 build (`b8afe65`) after its compressor, limiter, de-esser, and gate
changes. Each file sits beside the historical report it re-measures; neither
replaces the other. `deesser-corpus-v1-2026-09-report.json` re-scores the
generated de-esser corpus (all 96 clips classified correctly; no false positives).

No tracked evaluator measures de-esser audio change or a same-input limiter
comparison across builds. The de-esser body-reference change therefore rests on
its native unit tests, including negative controls, and the processing-order
evaluation; the corpus report scores the setup-time detector, not the DSP. The limiter change also has the current lookahead
evaluation above. A one-off same-input comparison against
v1.13.0 with the compressor disabled passed all ten existing limiter gates over
15 cases; it is recorded in PR #68, not as a tracked report.

The auto-makeup, limiter, processing-order, and voice-preservation reports
record their source hashes. The dynamics-aliasing, expanded-search, and de-esser
corpus reports come from evaluators that record neither source nor native build
identity. All September measurements used one native extension built from
`b8afe65` sources, SHA-256
`a4cb583960c8ca643686ec6e993eb056cc2db9e4399febbbc786041548dab931` (recorded in
the voice-preservation report); for those three reports that identity is stated
here, not verifiable from the reports themselves.

## External benchmark tooling

PESQ, STOI, SoundFile, and RemoteZip are evaluation-only dependencies; their
versions and dataset/model revisions are recorded in the reports that use
them. They are not product dependencies.

The RNNoise comparison wrapper is
`python/tools/rnnoise_upstream_benchmark.c`. Its exact pinned Zig/Xiph build
command is kept in that source file beside the wrapper it produces. Pass the
result to `evaluate_rnnoise_backends.py --upstream-binary`.

Keep the DPDFNet asset fetcher: its pinned EvalSet is also the speech corpus for
current Auto-EQ, makeup, DeepFilter, RNNoise, and processing-order evaluations.
`evaluate_dpdfnet_evalset.py` reproduces the official-output comparison only;
retaining it does not restore the historical clean-failure experiment or add
DPDFNet to the product.

## Interpretation

Objective metrics establish behavior only for the recorded corpus, hardware,
and configuration. Unobserved devices, operating-system versions, routes, or
voices are listed as limitations rather than inferred. Hardware qualification
is optional for publication; retain the candidate revision and digest with any
measurements and state the configurations actually tested. See
[RELEASING](../RELEASING.md).

`python/tools/evaluate_voice_regressions.py` compares gate-only native rendering
before and after a rebuild on the existing held-out VAD corpus. Its optional
`--eq-corpus models/cross_take_eval` checks neutral preservation and bounded
tonal change across the six held-out speakers. Write baseline and case details
under ignored `build/`; these checks do not establish microphone-response
recovery, perceptual quality, or live worker timing. Joint tuning's later segment
tests the selected gate/suppressor with fixed downstream settings, not an
independently trained and evaluated whole calibration pipeline.
