# Evaluation evidence

This directory stores compact decision records, not raw benchmark dumps.
Tracked JSON keeps only experiment configuration, source/asset hashes,
aggregate metrics, predefined gates, the resulting decision, and limitations.
Per-case detail is written only when an evaluator's optional
`--details-output` argument is supplied and should stay under ignored
`models/evaluation-details/` or in a CI artifact.

## Current decisions

| Area | Evidence | Outcome |
| --- | --- | --- |
| DeepFilter | `deepfilter-hardening-report.json`, `deepfilter-fullband-report.json` | Retain 30 dB attenuation and beta 0.0. |
| Auto-EQ confidence | `auto-eq-confidence-calibration.json` | Retain calibrated capture and per-band confidence gates. |
| Compressor control | `auto-makeup-real-speech-report.json`, `compressor-control-report.json`, `compressor-search-report.json` | Retain VAD/reliability-driven makeup and bounded search. |
| Processing order | `processing-order-report.json` | Retain gate before suppression and de-esser before EQ. |
| Limiter | `limiter-lookahead-report.json` | Retain 2 ms lookahead. |
| Resampling | `resampler-quality-report.json` | Retain the 128-tap Blackman product path. |
| Manual typed EQ | `eq-filter-types-report.json` | Retain manual bell/notch/shelf/pass types and selectable slopes. |
| Auto-EQ candidate pool | `eq-candidate-pool-report.json`, `sparse-auto-eq-filter-report.json` | Reject wider/sparse type-selecting candidates. |
| EQ stage split | `correction-tone-stage-report.json` | Reject separate correction/tone stages. |
| RNNoise | `rnnoise-backend-comparison.json` | Retain `nnnoiseless`; upstream Xiph was materially slower and regressed clean preservation. |
| DPDFNet | `dpdfnet-vs-deepfilternet3-report.json`, `dpdfnet-official-evalset-report.json` | Rejected and absent from production. |
| Release integrity | `release-bundle-path-baseline.json`, `release-trends.json` | Exact archive sidecars and digest-bound qualification own artifact facts. |

The corresponding `python/tools/evaluate_*.py` command regenerates each
record. Corpora and model assets are hash-pinned under ignored `models/`
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

## Interpretation

Objective metrics establish behavior only for the recorded corpus, hardware,
and configuration. Unobserved devices, operating-system versions, routes, or
voices are listed as limitations rather than inferred. Release promotion
requires one clean, SHA-bound automated 30-minute baseline from the exact
candidate; broader hardware/lifecycle coverage remains optional evidence.
