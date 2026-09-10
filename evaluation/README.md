# Evaluation evidence

This directory stores compact decision records, not raw benchmark dumps.
Tracked JSON keeps only experiment configuration, source/asset hashes,
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
| Auto-EQ confidence | `auto-eq-confidence-calibration.json` | Retain calibrated capture and per-band confidence gates. |
| Compressor control | `auto-makeup-real-speech-report.json`, `compressor-control-report.json`, `compressor-search-report.json` | Retain VAD/reliability-driven makeup and bounded search. |
| Processing order | `processing-order-report.json` | Retain gate before suppression and de-esser before EQ. |
| Limiter | `limiter-lookahead-report.json` | Adopt 0.5 ms lookahead after corrected paired, delay-flushed scoring. |
| Resampling | `resampler-quality-report.json` | Retain the 128-tap Blackman product path. |
| Manual typed EQ | `eq-filter-types-report.json` | Retain manual bell/notch/shelf/pass types and selectable slopes. |
| Auto-EQ candidate pool | `eq-candidate-pool-report.json`, `sparse-auto-eq-filter-report.json` | Reject the tested nested wider pools and sparse type-selecting candidate. |
| EQ stage split | `correction-tone-stage-report.json` | Safe/cost-eligible, but closed because no product benefit was demonstrated. |
| RNNoise | `rnnoise-backend-comparison.json` | Retain `nnnoiseless`; upstream Xiph was materially slower and regressed clean preservation. |
| DPDFNet | `dpdfnet-vs-deepfilternet3-report.json`, `dpdfnet-official-evalset-report.json` | Rejected and absent; historical clean failures are not independently reproducible from this checkout. |
| ONNX Runtime backend | `onnxruntime-cpu-probe.json` | Official CPU-only 1.23.2 matched the preserved 3.12 baseline across 498 captures and 20,908 frames. |
| Release integrity | `release-bundle-path-baseline.json` | Reviewed package paths gate the candidate; exact archive sidecars and digest-bound qualification own artifact facts. |

The `python/tools/evaluate_*.py` commands regenerate current measurements;
historical decisions retain their stated reproduction limits.
Reports with `source_revision` preserve the original committed measurement;
hygiene verifies both the unchanged report and its source hashes at that commit.
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
