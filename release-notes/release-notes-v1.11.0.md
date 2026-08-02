# AudioForge v1.11.0

This release expands AudioForge's editing and device-management surface while
keeping candidate DSP changes behind predefined objective gates.

## Typed EQ and direct editing

Every EQ band now has a versioned runtime type. Manual controls support bell,
notch, low/high shelf, and low/high-pass filters, with selectable
12, 24, 36, and 48 dB/octave Butterworth pass slopes. The graph uses the exact
native Rust response and supports constrained mouse and keyboard editing, so the
display and realtime topology share one source of truth. Existing ten-band
presets migrate without losing user-authored values.

Sparse type-selecting Auto-EQ and separate correction/tone-stage candidates did
not pass their retention gates and were removed. Auto-EQ therefore remains one
combined ten-band correction, while the new filter types remain available for
manual use.

## Safer configuration and setup

- Bounded undo/redo covers complete processing snapshots rather than EQ alone.
- Device-pair presets use stable Windows endpoint identities, duplicate-name
  ordinals, and reconnect-safe matching instead of relying on display names.
- A resumable first-run shell connects the existing route selection, latency
  calibration, and Auto Voice Setup workflows without duplicating their DSP.
- Auto-EQ abstention now explains capture-, reliability-, and band-level reasons
  without weakening the underlying safety decision.
- Diagnostics export is versioned, privacy-safe by default, and omits raw device
  names, paths, audio, and saved preset names/files. It includes only an
  allowlisted snapshot of the active DSP settings needed for support.
- Semantic theme tokens, keyboard/focus behavior, accessible names, contrast
  checks, and deterministic sanitized screenshots now have automated coverage.

## Evidence and release integrity

The release pipeline builds once, generates checksum/metadata/per-file manifest
sidecars, validates those exact archive bytes, binds hardware qualification to
the archive digest, and promotes without rebuilding. Packaging enforces the
Windows system-UCRT policy. Native 48 kHz DeepFilter and 44.1/48 kHz product
resampler reports record hashes, settings, latency, clean-preservation, and
realtime gates.

DeepFilter's production operating point remains 30 dB attenuation with no
post-filter blend. The broad multilingual and native-full-band comparisons now
agree on that setting when silence-only attenuation is diagnostic and the
clean/noisy speech, lower-tail, runtime, and headroom checks decide retention.

Processing-order candidates fail their objective gates. Shorter limiter arms
either fail event-preservation checks or do not deliver the predefined material
latency reduction, so the shipping order and 2 ms limiter remain unchanged.

The Windows output queue now keeps 10 ms more callback headroom. On the exact
local package, the strict 30-minute virtual-route run added no post-warmup
underruns or other loss/recovery/error events; the bounded tradeoff is about
10 ms more engine buffering.

## Compatibility

Configuration and presets migrate to schema/version v1.11.0 while preserving
explicit user choices. DPDFNet remains rejected and absent. Existing noise
models, the default ten-band Auto-EQ layout, processing order, and 2 ms limiter
lookahead remain unchanged.

Persisted endpoint identities now fail closed if their exact Windows endpoint
has disappeared rather than attaching to same-name replacement hardware.
Malformed saved device names are ignored, and a preset requesting a missing
neural backend falls back visibly and deterministically to RNNoise.

## Validation

The source tree is gated by Rust and Python unit/integration tests, release-mode
realtime contention stress, Ruff, Pyright, Clippy, RustSec, locked Python
dependency audits, Semgrep, workflow validation, evaluation source-hash hygiene,
asset verification, a clean Windows bundle build, package smoke, provenance
verification, and hidden executable startup. Exact artifact facts belong to the
generated `.sha256`, `.metadata.json`, and `.manifest.json` sidecars.

Publication requires a clean workflow candidate plus a SHA-bound automated
30-minute selected-route baseline from those exact bytes. Broader physical
device, Windows-version, 44.1 kHz, and lifecycle coverage is optional evidence
and remains explicitly unobserved rather than implied.
