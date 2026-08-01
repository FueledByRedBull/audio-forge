# EQ preset and runtime contract

AudioForge preset schema version 2 stores exactly ten immutable EQ bands. The
serialized band list is the source of truth; the legacy `band_freqs`,
`band_gains`, and `band_qs` arrays are migration inputs and compatibility views,
not additional stored state.

## Serialized shape

```json
{
  "schema_version": 2,
  "enabled": true,
  "bands": [
    {
      "filter_type": "high_pass",
      "frequency_hz": 70.0,
      "gain_db": 0.0,
      "q": 0.7071067811865476,
      "bandwidth_mode": "q",
      "bandwidth_octaves": null,
      "slope_db_per_octave": 24,
      "stage": "combined",
      "enabled": true
    }
  ]
}
```

Every field is required. Unknown fields, malformed values, future schema
versions, non-finite numbers, and band counts other than ten are rejected.

| Field | Contract |
|---|---|
| `filter_type` | `bell`, `notch`, `low_shelf`, `high_shelf`, `high_pass`, or `low_pass` |
| `frequency_hz` | 20–20,000 Hz and below the runtime Nyquist limit |
| `gain_db` | −12 to +12 dB |
| `q` | 0.1–10; canonical coefficient value |
| `bandwidth_mode` | `q` or `octaves` |
| `bandwidth_octaves` | `null` for Q mode; required for octave mode |
| `slope_db_per_octave` | 12, 24, 36, or 48 |
| `stage` | `combined`; reserved field, other values are rejected |
| `enabled` | Per-band click-safe bypass |

Q is always the canonical value sent to the native DSP. Octave mode is allowed
only for bell and notch filters, and its stored octave width must resolve to the
same Q under the RBJ digital-bandwidth equation at AudioForge's fixed 48 kHz
engine rate. This prevents two conflicting coefficient sources. The current UI
preserves valid octave metadata but exposes Q editing; a manual frequency, type,
or Q edit explicitly returns that band to Q mode.

## Native behavior

Rust is the sole coefficient and magnitude-response implementation. Python
passes a six-field tuple:

```text
(filter_type, frequency_hz, gain_db, q, slope_db_per_octave, enabled)
```

- Bell, low shelf, and high shelf use frequency, gain, and Q.
- Notch uses frequency and Q; gain is retained in the preset but ignored.
- High-pass and low-pass use frequency and slope; gain and user Q are retained
  but ignored.
- Pass slopes are cascades of one to four second-order Butterworth sections.
  Each supported slope is exactly −3.0103 dB at cutoff. Section Q values are
  derived from the Butterworth poles, not hand-tuned.
- Band bypass, type edits, coefficient edits, and slope changes use a
  sample-rate-scaled 1.5 ms parallel-state crossfade.
- Sections are preallocated. Inactive slope sections stop processing after
  their bypass transition, so the default ten-band path does not pay the cost
  of forty biquads.
- These filters add no lookahead or algorithmic latency.

The legacy three-field native API remains supported. Calling its batch setter
deliberately restores the historical low-shelf / eight-bell / high-shelf
layout; this is how existing Auto-EQ output remains behaviorally compatible.

## Migration and provenance

Legacy presets migrate to:

- band 0: low shelf;
- bands 1–8: bell;
- band 9: high shelf;
- 12 dB/octave, combined stage, enabled.

Legacy frequency, gain, and Q provenance is transferred to the corresponding
per-band fields. New type, slope, stage, bandwidth, and enabled fields are
marked `migration_default`. Explicit values are never overwritten by later
default migrations.

## Retention gates

The expanded implementation is retained only if all of these hold:

1. Default and legacy responses remain numerically equivalent to the incumbent.
2. Bell/shelf center behavior, notch rejection, and all pass-filter cutoffs
   satisfy analytic checks.
3. Every type remains finite at supported frequency, gain, Q, and slope
   boundaries.
4. Live type, slope, and bypass edits remain click-bounded.
5. The realtime path remains allocation-free and adds no blocking control path.
6. Default CPU cost does not materially regress; additional cost scales only
   with selected pass-filter order.
7. Schema migration, malformed-input rejection, exact preset round-trip, UI
   synchronization, and native/curve parity pass.
8. Native-48-kHz corpus evaluation preserves the default path exactly, reports
   pre-limiter headroom accurately, and keeps the stressed full-chain output at
   the effective true-peak ceiling without non-finite output.

The pre/post biquad benchmark was unchanged within run noise (6.55 to
6.52 ns/sample steady). The ten-band default measured 33.23 ns/sample; replacing
one default section with a 48 dB/octave four-section pass filter measured
41.02 ns/sample (1.23× for that deliberately steeper configuration).

The corpus arm also records an important non-failure: a cut-only minimum-phase
configuration reduced RMS but raised waveform true peak by up to 2.68 dB as its
phase response changed crest factor. Magnitude gain therefore cannot be used as
a time-domain peak bound. The product contract is explicit pre-limiter
observation plus downstream limiter safety; under the stress arm, full-chain
true-peak overshoot was 0.0007 dB and all output remained finite.

## Deliberate boundaries

- Auto-EQ still emits the historical shelf/bell layout. Sparse automatic type
  selection is a separate measured candidate.
- Separate correction/tone ownership was evaluated and rejected because the
  candidate provided no material objective benefit. Schema v2 therefore
  rejects `correction` and `tone` instead of silently ignoring them.
- Band dragging uses this schema's existing validated ranges and does not add
  another coefficient implementation.
- The controlled headroom/corpus report remains the final retention evidence;
  unit tests do not substitute for that report.
