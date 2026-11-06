"""The runtime snapshot reads the engine's diagnostics dict once, with the readers' defaults."""

import math

from mic_eq.runtime_snapshot import RuntimeSnapshot


def test_missing_keys_get_the_defaults_the_readers_use():
    snapshot = RuntimeSnapshot.parse({})

    assert snapshot == RuntimeSnapshot()
    assert (snapshot.present, snapshot.valid) == (frozenset(), frozenset())
    assert snapshot.noise_model == "rnnoise"
    assert snapshot.noise_backend_available is True
    assert snapshot.input_cleanup_mode == "off"
    assert snapshot.input_phase_rescue_strategy == "none"
    assert snapshot.output_underrun_total == 0
    assert snapshot.latency_compensation_ms == 0.0
    assert snapshot.input_stereo_correlation is None
    assert snapshot.reported("output_underrun_total") is None


def test_values_of_the_right_type_are_kept_as_the_engine_sent_them():
    snapshot = RuntimeSnapshot.parse({
        "output_underrun_total": 3,
        "clip_peak_db": -1.25,
        "output_short_term_lufs": float("nan"),
        "gate_auto_relax_active": True,
        "rt_error_name": "input stream error",
        "noise_backend_error": None,
        "not_a_field": "ignored",
    })

    assert type(snapshot.output_underrun_total) is int and snapshot.output_underrun_total == 3
    assert snapshot.clip_peak_db == -1.25
    # Whether a NaN means anything is the reader's decision, not the snapshot's.
    assert snapshot.output_short_term_lufs is not None and math.isnan(snapshot.output_short_term_lufs)
    assert snapshot.gate_auto_relax_active is True
    assert snapshot.rt_error_name == "input stream error"
    assert snapshot.present == {
        "output_underrun_total", "clip_peak_db", "output_short_term_lufs", "gate_auto_relax_active",
        "rt_error_name", "noise_backend_error",
    }
    # A None the engine sent is present but not a value.
    assert snapshot.valid == snapshot.present - {"noise_backend_error"}
    assert snapshot.reported("output_underrun_total") == 3


def test_a_value_of_the_wrong_type_gets_the_default_and_is_marked_invalid():
    snapshot = RuntimeSnapshot.parse({
        "output_underrun_total": "3",
        "input_crest_factor_db": True,  # a bool is not a number
        "gate_auto_relax_active": 1,
        "input_cleanup_mode": 2,
        "noise_backend_available": None,
    })

    assert snapshot.output_underrun_total == 0
    assert snapshot.input_crest_factor_db is None
    assert snapshot.gate_auto_relax_active is False
    assert snapshot.input_cleanup_mode == "off"
    assert snapshot.noise_backend_available is True
    assert snapshot.valid == frozenset()
    assert len(snapshot.present) == 5
    assert all(snapshot.reported(name) is None for name in snapshot.present)


def test_the_engine_dict_parses_into_every_field_it_has(qapp):
    from mic_eq import AudioProcessor

    processor = AudioProcessor()
    try:
        diagnostics = processor.get_runtime_diagnostics()
    finally:
        processor.stop()
    snapshot = RuntimeSnapshot.parse(diagnostics)

    # Every value the engine sends for a field has the field's type; None marks an absent reading.
    assert snapshot.present - snapshot.valid == {name for name in snapshot.present if diagnostics[name] is None}
    assert snapshot.noise_model == diagnostics["noise_model"]
    assert snapshot.output_sample_rate == diagnostics["output_sample_rate"]
