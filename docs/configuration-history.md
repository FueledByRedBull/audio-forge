# Processing configuration history

AudioForge keeps a bounded 50-entry undo/redo history for processing
configuration. `Ctrl+Z` undoes and `Ctrl+Shift+Z` redoes; the Edit menu and the
main-window Undo button expose the same commands.

## Snapshot boundary

Each entry is a canonical JSON serialization of the validated preset schema:
gate, typed EQ, suppressor selection/strength, de-esser, compressor, limiter,
and master bypass. The JSON payload and its label/source are immutable.

The history deliberately excludes:

- audio samples and analysis recordings;
- filter delay lines, model recurrent state, meters, and limiter history;
- device stream handles, callback state, and recovery counters; and
- application geometry and unrelated device-selection UI state.

This boundary makes restoration deterministic without pretending realtime
state can be rewound safely.

## Transactions

Continuous widget signals are debounced into one entry after the gesture
settles. Preset application, Auto-EQ, and Auto Voice Setup are recorded as
named transactions. Modal calibration suppresses intermediate preview states.
Restoring history suppresses new entries.

Undo and redo are transactional: the target snapshot is parsed and validated,
then applied, and only then does the cursor move. A failed validation or apply
leaves the cursor unchanged. Recording after undo discards the old redo branch.
Identical payloads are deduplicated and the oldest entries are evicted above
the fixed bound.

## Migration provenance

Snapshots retain the preset schema's `value_provenance` map. Loading, undoing,
or redoing a migrated preset preserves `migration_default` values. When a
manual edit is committed, the previous and current canonical payloads are
compared and only changed value paths become `explicit`; untouched migration
provenance is not broadened accidentally.

## Retention checks

The implementation is retained only while focused tests establish:

- bounded history, deduplication, multi-step undo/redo, and branch semantics;
- transactional cursor behavior on malformed data and restore failure;
- provenance preservation and changed-path marking;
- exclusion of audio/runtime state from snapshots; and
- actual main-window wiring for manual edits, presets, Auto-EQ, undo, and redo.

The feature is session-local. History is not written to disk, synchronized
between processes, or advertised as recovery after a crash.
