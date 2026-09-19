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
| Voice preservation | `voice-preservation-report.json` | All 78 held-out gate cases and 48 EQ cases pass predefined regression limits. Natural remains neutral; adaptive tone changes stay bounded. This does not establish perceptual quality or reproduce the live post-yell issue. |
| Warm target and capture validation | `warm-auto-eq-report.json` | All 60 dialog cases and 60 held-out EQ cases pass. Clear-speech validation is independent of fitting smoothing; Warm adds bounded low-mid body. |
| Auto-EQ confidence | `auto-eq-confidence-calibration.json` | Historical calibration for the former absolute-spectrum fitting objective; not a perceptual validation of the current bounded tonal adjustment. |
| Compressor control | `auto-makeup-real-speech-report.json`, `compressor-control-report.json`, `compressor-search-report.json` | Retain VAD/reliability-driven makeup and bounded search. |
| Processing order | `processing-order-report.json` | Retain gate before suppression and de-esser before EQ. |
| Limiter | `limiter-lookahead-report.json` | Adopt 0.5 ms lookahead after corrected paired, delay-flushed scoring. |
| Resampling | `resampler-quality-report.json` | Retain the 128-tap Blackman product path. |
| Manual typed EQ | `eq-filter-types-report.json` | Retain manual bell/notch/shelf/pass types and selectable slopes. |
| Auto-EQ candidate pool | `eq-candidate-pool-report.json`, `sparse-auto-eq-filter-report.json` | Reject the tested nested wider pools and sparse type-selecting candidate. |
| EQ stage split | `correction-tone-product-report.json`; historical `correction-tone-stage-report.json` | Retain the explicitly requested independent stages: 12 cases pass safety, EQ-kernel cost, schema and zero-added-latency gates. The historical proposal was closed before this product request. |
| Joint gate/model tuning | `product-joint-tuning.json`, `product-joint-tuning-deepfilter-ll.json`, `product-joint-tuning-deepfilter.json` | All 66 cases pass; 17 candidates applied and 49 incumbents retained. DeepFilter incumbents stay within their family after unrestricted switches to RNNoise failed clean-speech checks. |
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
The three historical joint-tuning reports retain their original per-case evidence.
New joint-tuning runs write compact, model-specific reports; optional
`--details-output models/evaluation-details/joint-tuning.json` keeps full case details.
Corpora and model assets are hash-pinned under ignored `models/`
directories and are never bundled merely because they exist locally.
`python/tools/check_evaluation_hygiene.py` rejects absolute paths, stale source
hashes, malformed audible-change contracts, privacy leaks, and oversized
tracked reports.

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
