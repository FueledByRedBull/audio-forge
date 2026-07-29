# Evaluation evidence

The JSON files in this directory are concise, reviewable decision records.
Large downloaded corpora, model files, and optional per-condition detail remain
under ignored local `models/` paths and are not release payloads.

Reports retain the experiment definition, immutable source revision and hashes,
aggregate metrics, uncertainty, component gates, decision, and limitations.
Evaluation tools use repository-relative paths in reports. Per-condition
DPDFNet rows are written only when `--details-output` is supplied.

## v1.10.1 hardening records

- `deepfilter-hardening-report.json`: attenuation/post-filter selection,
  partial-strength alignment, clean/noisy quality, and frame-tail timing.
- `auto-makeup-real-speech-report.json`: native streaming auto-makeup control
  on paired multilingual speech.
- `auto-eq-confidence-calibration.json`: held-out confidence-threshold
  calibration and explicit insufficient-evidence cases.
- `dynamics-aliasing-report.json`: 48 kHz dynamics versus a 192 kHz reference.
- `processing-order-report.json`: gate/suppressor and de-esser/EQ candidates.
- `limiter-lookahead-report.json`: 0.5, 1.0, and 2.0 ms comparison.
- `rnnoise-backend-comparison.json`: shipped `nnnoiseless` versus pinned
  current upstream Xiph RNNoise.
- `hardware-validation.json`: selected-route correlation and sustained
  release-machine callback health, tied to the exact native-extension and
  measurement-tool hashes.
- `release-trends.json`: release-to-release bundle, archive, runtime, latency,
  and quality measurements; unavailable historical values are recorded as
  `not_measured` with an explicit reason.

Every audible-change record includes a machine-readable configuration/assets,
latency, CPU/P99, clean-preservation, decision-gate, and listening-status
contract. `python/tools/check_evaluation_hygiene.py` enforces that contract and
rejects machine-local paths.

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

- Unit and synthetic fixtures establish control-law correctness; they are not
  perceptual listening evidence.
- The official DPDFNet subset reproduces metric direction on 36 stratified
  conditions, not the paper's full tables or DNSMOS/NISQA/PRISM claims.
- The separate 48 kHz product-path report decides AudioForge retention,
  including clean-speech preservation and measured CPU behavior.
- The local supplied DPDFNet2/4/8 output directories were pruned after the
  rejection. Clean, Noisy, DeepFilterNet3, the pinned manifest, fetch tool, and
  concise reports preserve reusable evidence and reproducibility.
- `LISTENING_PROTOCOL.md` defines the human review procedure for claims that
  objective metrics cannot settle.
