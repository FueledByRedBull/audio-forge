# Evaluation evidence

The JSON files in this directory are concise, reviewable decision records.
Large downloaded corpora, model files, and optional per-condition detail remain
under ignored local `models/` paths and are not release payloads.

Reports retain the experiment definition, immutable source revision and hashes,
aggregate metrics, uncertainty, component gates, decision, and limitations.
Evaluation tools use repository-relative paths in reports. Per-condition
DPDFNet rows are written only when `--details-output` is supplied.

## v1.11.0 qualification records

- `deepfilter-hardening-report.json`: attenuation/post-filter selection,
  partial-strength alignment, clean/noisy quality, and frame-tail timing. This
  broad multilingual gate retains the production 30 dB/beta 0.0 operating
  point.
- `auto-makeup-real-speech-report.json`: native streaming auto-makeup control
  on paired multilingual speech.
- `auto-eq-confidence-calibration.json`: held-out confidence-threshold
  calibration and explicit insufficient-evidence cases.
- `dynamics-aliasing-report.json`: 48 kHz dynamics versus a 192 kHz reference.
- `processing-order-report.json`: generated/proxy gate/suppressor and
  de-esser/EQ comparisons. Both candidates fail predefined objective gates, so
  product order remains unchanged.
- `limiter-lookahead-report.json`: 0.5, 1.0, and 2.0 ms comparison through the
  exact protected compressor/main-limiter/true-peak chain, using controlled
  fixtures and 12 native-48-kHz real speakers. The 0.5 ms arm fails event
  preservation and the 1 ms arm does not meet the predefined 1.5 ms material
  latency-reduction gate, so 2 ms remains selected.
- `rnnoise-backend-comparison.json`: shipped `nnnoiseless` versus pinned
  current upstream Xiph RNNoise.
- `hardware-validation.json`: selected-route correlation and sustained
  release-machine callback health, tied to the exact native-extension and
  measurement-tool hashes.
- `hardware-validation-v<version>-published.json`: exact downloaded release
  archive qualification, including archive/tree identity, startup and model
  discovery, selected-route correlation, machine/OS details, and 30-minute
  selected-route callback health. Promotion consumes the equivalent hash-bound
  report emitted by the self-hosted hardware workflow; each report states its
  actual device class and does not infer physical topology from device names.
- `hardware-validation-v1.11.0-local.json`: the current uncommitted exact-local-
  archive case. Package, startup, model discovery, correlation, and the strict
  30-minute health arm pass. Output underruns remain at the post-warmup baseline
  of two, and all drop/recovery/restart/callback/overflow/backend/RT/non-finite
  failure counters remain zero. This is qualification evidence, not a
  promotable release attestation, because its source revision is uncommitted.
- `hardware-matrix-v1.11.0-local.json`: aggregates that case and records the
  matrix as incomplete. Windows 10, built-in/USB device classes, 44.1 kHz,
  and buffer/default/reconnect/route/sleep lifecycle cases remain unobserved.
- `release-trends.json`: release-to-release bundle, archive, runtime, latency,
  and quality measurements; unavailable historical values are recorded as
  `not_measured` with an explicit reason.
- `resampler-quality-report.json`: exact 44.1/48 kHz streaming-path passband,
  offline-reference, alias/image, delay, round-trip, drift, and realtime gates.
- `deepfilter-fullband-report.json`: native-48-kHz VoiceBank-DEMAND
  clean/noisy pairs, controlled 8-20 kHz noise, and a qualified local
  physical-microphone noise capture across LL/Standard, 12/20/30/80 dB
  attenuation, and 50/100% product strength. Silence-only attenuation remains
  diagnostic; clean/noisy speech preservation, lower tails, runtime, and
  headroom select the release setting. The objective arm retains 30 dB.
- `cross-take-auto-eq-report.json`: speaker-disjoint repeated-reading
  comparison of the shipped single-take Auto-EQ against an evaluation-only
  frequency-dependent cross-take confidence candidate. The candidate failed
  both median and lower-tail held-out target-fit gates, so its product
  integration was removed and the incumbent remains unchanged.
- `diagnostics-export-report.json`: adversarial privacy, schema, determinism,
  finite-value, report-local pseudonym, and size-bound gates for the
  allowlisted support snapshot exported from the Help menu.
- `eq-filter-types-report.json`: typed-schema and manual-filter retention gates
  covering incumbent parity, analytic notch/Butterworth math, randomized
  boundaries, native-48-kHz clean audio, crest-factor/headroom observability,
  stressed full-chain limiter safety, realtime cost, and zero added latency.
- `sparse-auto-eq-filter-report.json`: pre-registered held-out repeated-take
  comparison of an evaluation-only sparse/type-selecting Auto-EQ candidate
  against the unchanged incumbent, including native clean/headroom, CPU,
  latency, and constraint gates. The candidate is rejected.
- `eq-candidate-pool-report.json`: 12/14/16-candidate placement experiment.
  Every variant regressed median fit and/or lower-tail/risk/runtime gates, so
  the corrected dynamic ten-band incumbent remains selected.
- `correction-tone-stage-report.json`: migration, ownership, native rendering,
  headroom, CPU, latency, and material-benefit gates for the rejected two-stage
  EQ candidate; the product retains one combined stage.
- `ui-screenshot-report.json`: fixed-scale, sanitized repository screenshot
  dimensions, hashes, source provenance, alt-text coverage, and capture privacy
  contract. Regenerate it with `python/tools/capture_repository_screenshots.py`.

Every audible-change record includes a machine-readable configuration/assets,
latency, CPU/P99, clean-preservation, and decision-gate contract.
`python/tools/check_evaluation_hygiene.py` enforces that contract and rejects
machine-local paths.

## Benchmark-only Python packages

These packages are not AudioForge runtime dependencies. The retained reports
were produced with:

```text
pesq==0.0.4
pystoi==0.4.1
remotezip==0.12.3
soundfile==0.13.1
```

Install them only in an evaluation environment. `pesq` and `pystoi` reproduce
the pinned official DPDFNet subset metrics; `remotezip` and `soundfile` fetch
and decode the deterministic Samromur child-speech subset. The package versions,
dataset/model revisions, manifest hashes, and scope limits are recorded in the
reports and fetch tools.

The RNNoise comparison additionally compiles the pinned official Xiph C source
and model with a hash-verified Zig 0.16.0 archive. The compiler, upstream source,
generated benchmark executable, and downloaded model remain ignored
benchmark-only assets; none are production dependencies or release payloads.

Rebuild the evaluation-only upstream executable from repository root after
checking out Xiph RNNoise commit
`70f1d256acd4b34a572f999a05c87bf00b67730d` under
`models/benchmarks/upstream-rnnoise`. The Zig archive used for the recorded
report had SHA-256
`68659eb5f1e4eb1437a722f1dd889c5a322c9954607f5edcf337bc3684a75a7e`.
Set `$zig` to the verified Zig 0.16.0 executable, then run:

```powershell
$zig = "models/benchmarks/zig-windows-x86_64-0.16.0/zig.exe"
$rnnoise = "models/benchmarks/upstream-rnnoise"

& $zig cc -O3 -DRNNOISE_BUILD `
  "-I$rnnoise/include" "-I$rnnoise/src" `
  python/tools/rnnoise_upstream_benchmark.c `
  "$rnnoise/src/denoise.c" `
  "$rnnoise/src/rnn.c" `
  "$rnnoise/src/pitch.c" `
  "$rnnoise/src/kiss_fft.c" `
  "$rnnoise/src/celt_lpc.c" `
  "$rnnoise/src/nnet.c" `
  "$rnnoise/src/nnet_default.c" `
  "$rnnoise/src/parse_lpcnet_weights.c" `
  "$rnnoise/src/rnnoise_data.c" `
  "$rnnoise/src/rnnoise_tables.c" `
  -o models/benchmarks/rnnoise-upstream-benchmark.exe
```

This deliberately builds the scalar path used by the recorded comparison; it
does not define `RNN_ENABLE_X86_RTCD` or compile the SSE4.1/AVX2 translation
units. Pass the resulting executable to
`python/tools/evaluate_rnnoise_backends.py --upstream-binary`.

## Interpretation limits

- Unit and synthetic fixtures establish control-law correctness within their
  recorded conditions; reports do not generalize beyond those conditions.
- The official DPDFNet subset reproduces metric direction on 36 stratified
  conditions, not the paper's full tables or DNSMOS/NISQA/PRISM claims.
- The separate 48 kHz product-path report decides AudioForge retention,
  including clean-speech preservation and measured CPU behavior.
- The local supplied DPDFNet2/4/8 output directories were pruned after the
  rejection. Clean, Noisy, DeepFilterNet3, the pinned manifest, fetch tool, and
  concise reports preserve reusable evidence and reproducibility.
## Native-48-kHz DeepFilter corpus

`python/tools/fetch_deepfilter_fullband_corpus.py` range-fetches a deterministic
24-pair subset from the official 48 kHz VoiceBank-DEMAND test archives and
writes a hash-pinned ignored manifest under
`models/deepfilter_fullband_eval/`. The selected p232/p257 files are
speaker-disjoint from the training set and retain the dataset's CC BY 4.0
terms. `python/tools/capture_microphone_noise.py` adds a 10-30 second,
VAD-qualified quiet physical-microphone capture; it is explicitly treated as
microphone, preamp, electrical, and room noise rather than laboratory
self-noise. Neither corpus nor personal capture is shipped in the product.

`python/tools/evaluate_eq_filter_types.py` reuses the clean 48 kHz subset for
manual-EQ retention. It deliberately does not resample the 16 kHz DPDFNet
corpus: native-rate EQ and future 16/48 kHz band-splitting claims are separate
experiments.

## Repeated-take Auto-EQ corpus

`python/tools/fetch_cross_take_corpus.py` downloads the official RAVDESS
audio-only speech archive from Zenodo, verifies the published archive MD5 and
a pinned local SHA-256, normalizes the two stereo source exceptions to mono,
and writes a portable hash-pinned manifest under
`models/cross_take_eval/`. The frozen 12/6/6 actor-disjoint split contains
144 repeated pairs from 24 adult actors. Evaluation aggregates the three
selected delivery conditions per actor and statement into product-length
independent first and second takes.

RAVDESS is licensed CC BY-NC-SA 4.0. The downloaded corpus is ignored,
benchmark-only, not shipped, and is not suitable for commercial redistribution
under AudioForge's release terms. `python/tools/evaluate_cross_take_auto_eq.py`
reproduces the rejected candidate without exposing a second-take control path
in production Auto-EQ or Voice Setup.
