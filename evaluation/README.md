# Evaluation evidence

The JSON files in this directory are concise, reviewable decision records.
Large downloaded corpora, model files, and optional per-condition detail remain
under ignored local `models/` paths and are not release payloads.

Reports retain the experiment definition, immutable source revision and hashes,
aggregate metrics, uncertainty, component gates, decision, and limitations.
Evaluation tools use repository-relative paths in reports. Per-condition
DPDFNet rows are written only when `--details-output` is supplied.

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

## Interpretation limits

- Unit and synthetic fixtures establish control-law correctness; they are not
  perceptual listening evidence.
- The official DPDFNet subset reproduces metric direction on 36 stratified
  conditions, not the paper's full tables or DNSMOS/NISQA/PRISM claims.
- The separate 48 kHz product-path report decides AudioForge retention,
  including clean-speech preservation and measured CPU behavior.
- `LISTENING_PROTOCOL.md` defines the human review procedure for claims that
  objective metrics cannot settle.
