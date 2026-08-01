# Sparse Auto-EQ filter-type experiment

This document pre-registers the evaluation for roadmap issue 42. The experiment
is deliberately offline-only. Manual typed EQ is a retained product feature;
Auto-EQ continues to emit its incumbent low-shelf, bell, and high-shelf layout
unless every gate below passes.

## Candidate

For each capture, the incumbent Auto-EQ result is the starting point. The
candidate may:

- disable an incumbent section;
- keep the incumbent type;
- change a low-frequency bell to a low shelf;
- change a high-frequency bell to a high shelf; or
- change a sufficiently deep, narrow, high-confidence cut to a notch.

The eligibility rules are fixed before evaluation:

- low-shelf alternatives require a center at or below 500 Hz;
- high-shelf alternatives require a center at or above 3 kHz;
- notch alternatives require gain at or below -6 dB, Q at least 3, and band
  confidence at least 0.65;
- disabled sections retain valid native parameters but contribute no response;
- all candidates contain exactly ten slots and pass the native typed-EQ
  validator.

Selection is deterministic greedy coordinate descent. It starts from the
incumbent, tests one eligible disable or type substitution at a time, accepts
the strict best improvement, and repeats until no operation improves the
training objective by more than 1e-6 dB. Lexicographic operation order breaks
ties.

The objective is expressed entirely in dB:

```text
J = weighted_target_RMSE_dB
    + 0.04 dB * active_section_count
    + 0.08 dB * active_notch_count
```

The fixed section penalty rewards a simpler correction only when the extra
training error is correspondingly small. The notch surcharge reflects the
native notch's complete center-frequency null and prevents a single narrow bin
from winning cheaply. No held-out value is used while selecting a capture's
candidate.

## Corpus and split

The primary experiment uses the pinned native-48-kHz RAVDESS repeated-take
manifest under `models/cross_take_eval`. The three selected delivery conditions
are concatenated per actor and statement with 250 ms separators. Take 01 fits
the incumbent and sparse candidate; take 02 is held out.

Only manifest-defined validation and test actors are scored. Constants and
gates are not changed after inspecting validation or test output. The corpus
contains acted English speech and is not representative of every microphone,
language, room, or speaking style. The source license prevents shipping its
audio with AudioForge.

## Retention gates

Every gate is mandatory:

1. At least 20 comparable held-out cases must complete.
2. Median held-out target-error improvement must be at least 0 dB and the 10th
   percentile must be no worse than -0.35 dB.
3. Candidate cross-take response disagreement must not exceed the incumbent by
   more than 0.10 dB at the median or 0.25 dB at the 90th percentile.
4. Median active-section count must fall by at least one.
5. Native clean renders must be finite, candidate true peak may not exceed the
   incumbent by more than 0.50 dB, and candidate P95 limiter gain reduction may
   not exceed the incumbent by more than 0.50 dB.
6. Candidate P95 runtime may not exceed 1.10 times the incumbent, absolute P95
   realtime factor must be at most 0.01, and both paths must report zero added
   algorithmic latency.
7. Every selected band must satisfy the eligibility and native-validation
   rules above.
The decision is conjunctive. A missing measurement is a failed gate. If any
gate fails, the candidate remains evaluation-only and the incumbent product
path is retained.

## Evidence

`python/tools/evaluate_sparse_auto_eq_filters.py` writes the portable decision
record to `evaluation/sparse-auto-eq-filter-report.json`. The report records
source and corpus hashes, exact constants, per-split aggregate metrics, native
runtime and latency, clean/headroom measurements, failed checks, and the
retention decision.

## Recorded result

The native-48-kHz run completed all 24 held-out cases. The candidate improved
median held-out target error by 0.0499 dB, had a -0.0266 dB 10th-percentile
change, removed three active sections at the median, improved median cross-take
response disagreement by 0.0610 dB, and reduced P95 realtime factor to 0.861
times the incumbent. It selected bells and justified low/high shelves; no notch
survived the objective.

The candidate nevertheless failed the frozen clean-preservation gate: its
worst pre-limiter clean true-peak regression was +2.6522 dB against the allowed
+0.50 dB. The downstream limiter stayed within its ceiling and required less
P95 reduction, but that does not retroactively waive the pre-limiter gate. The
objective decision is therefore **reject**: no selector was added to product
Auto-EQ, and the incumbent ten-band layout remains authoritative.
