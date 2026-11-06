"""The engine's runtime diagnostics, read once into typed fields.

``AudioProcessor.get_runtime_diagnostics()`` returns a dict of about ninety
keys. Health, the setup dialogs and the support export all read it, so it is
parsed here once. Each field is named after its key. A key that is missing,
or whose value has the wrong type, gets the field's default: the value the
readers used for a missing key. ``present`` and ``valid`` keep what the
defaults hide, which the export and the stream-event window need: a key the
engine did not send is left out of the export, and one it sent with a bad
value is exported as ``null``.

Numbers keep the type the engine gave (an ``int`` counter stays an ``int``)
and are not checked for being finite; the readers decide what a NaN means.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, get_args, get_type_hints


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """One reading of the engine's runtime diagnostics; build it with :meth:`parse`."""

    # Noise suppression
    noise_model: str = "rnnoise"
    noise_attenuation_limit_db: float | None = None
    noise_post_filter_beta: float | None = None
    noise_backend_available: bool = True
    noise_backend_failed: bool = False
    noise_backend_error: str | None = None
    noise_selection_dropped_count: float = 0
    suppressor_non_finite_count: float = 0
    downstream_non_finite_count: float | None = None
    # Stream health and recovery
    input_dropped_samples: float = 0
    input_backlog_recovery_count: float = 0
    input_backlog_dropped_samples: float = 0
    lock_contention_count: float = 0
    output_underrun_total: float = 0
    output_priming_underrun_total: float | None = None
    output_underrun_streak: float = 0
    jitter_dropped_samples: float | None = None
    output_retime_adjustment_count: float | None = None
    output_resampler_error_count: float | None = None
    output_recovery_event_count: float = 0
    output_recovery_count: float = 0
    output_short_write_dropped_samples: float = 0
    stream_restart_count: float = 0
    input_callback_error_count: float = 0
    output_callback_error_count: float = 0
    input_stream_error_count: float = 0
    output_stream_error_count: float = 0
    rt_buffer_overflow_count: float = 0
    rt_error_name: str | None = None
    rejected_control_value_count: float | None = None
    dsp_scheduling_degraded: bool = False
    recovery_suppressed: bool = False
    last_restart_reason: str | None = None
    last_stream_error: str | None = None
    auto_makeup_error: str | None = None
    vad_worker_error: str | None = None
    # Input conditioning
    input_channel_mode: str | None = None
    input_cleanup_mode: str = "off"
    input_cleanup_hum_detected: bool = False
    input_cleanup_rumble_detected: bool = False
    input_cleanup_high_pass_hz: float | None = None
    input_crest_factor_db: float | None = None
    input_stereo_correlation: float | None = None
    input_phase_warning_count: float = 0
    input_phase_rescue_strategy: str = "none"
    input_phase_estimated_delay_samples: float | None = None
    input_phase_polarity_flipped: bool = False
    clip_event_count: float | None = None
    clip_peak_db: float | None = None
    # Output stage
    output_crest_factor_db: float | None = None
    output_short_term_lufs: float | None = None
    output_clip_event_count: float | None = None
    output_clip_peak_db: float | None = None
    output_true_peak_event_count: float = 0
    output_true_peak_db: float | None = None
    output_true_peak_gain_reduction_db: float | None = None
    output_true_peak_gain_reduction_history_db: float = 0.0
    output_true_peak_headroom_db: float | None = None
    limiter_gain_reduction_db: float | None = None
    limiter_peak_gain_reduction_db: float | None = None
    limiter_gain_reduction_history_db: float = 0.0
    limiter_careful_output_enabled: bool = False
    limiter_effective_ceiling_db: float | None = None
    raw_monitor_enabled: bool = False
    # Gate and de-esser
    gate_chatter_event_count: float = 0
    gate_auto_relax_active: bool = False
    gate_fused_score: float | None = None
    deesser_detector_confidence: float | None = None
    # Format, buffers and latency
    input_resampler_active: bool = False
    output_resampler_active: bool = False
    input_sample_rate: float | None = None
    output_sample_rate: float | None = None
    input_fixed_buffer_frames: float = 0
    output_fixed_buffer_frames: float = 0
    output_low_period_frames: float | None = None
    output_low_period_failure_code: float | None = None
    output_largest_callback_frames: float | None = None
    output_sustained_callback_frames: float | None = None
    output_resampler_delay_samples: float | None = None
    engine_latency_ms: float | None = None
    total_latency_ms: float | None = None
    route_latency_ms: float | None = None
    output_device_latency_ms: float | None = None
    output_device_buffer_frames: float | None = None
    output_device_period_frames: float | None = None
    output_device_buffer_known: bool = False
    output_device_buffer_failure_code: float | None = None
    latency_compensation_ms: float = 0.0
    # Added by the window for the support export, not sent by the engine
    input_rms_db: float | None = None
    input_peak_db: float | None = None
    own_microphone_refusal_count: float | None = None
    default_source_watch_failed: bool = False
    # Which known keys the source had, and which of those had a value of the right type
    present: frozenset[str] = frozenset()
    valid: frozenset[str] = frozenset()

    @classmethod
    def parse(cls, diagnostics: Mapping[str, object]) -> RuntimeSnapshot:
        """Read ``diagnostics`` once; keys this class does not name are ignored."""
        values: dict[str, Any] = {}
        present = frozenset(name for name in _FIELD_KINDS if name in diagnostics)
        for name in present:
            value = diagnostics[name]
            if _ACCEPTS[_FIELD_KINDS[name]](value):
                values[name] = value
        return cls(**values, present=present, valid=frozenset(values))

    def reported(self, name: str) -> object:
        """The value the engine sent for ``name``, or ``None`` when it sent none of the right type."""
        return getattr(self, name) if name in self.valid else None


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_ACCEPTS = {
    "number": _is_number,
    "bool": lambda value: isinstance(value, bool),
    "text": lambda value: isinstance(value, str),
}


def _kind(hint: object) -> str:
    if hint is bool:
        return "bool"
    return "text" if str in (get_args(hint) or (hint,)) else "number"


_HINTS = get_type_hints(RuntimeSnapshot)
_FIELD_KINDS = {
    field.name: _kind(_HINTS[field.name])
    for field in fields(RuntimeSnapshot)
    if field.name not in {"present", "valid"}
}
