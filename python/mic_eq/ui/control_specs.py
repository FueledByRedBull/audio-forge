"""Shared presentation and numeric ranges for processing controls."""

from __future__ import annotations

from dataclasses import dataclass

from ..config_parts.validation import VALIDATION_RANGES


GATE_MODES = ("Threshold Only", "VAD Assisted", "VAD Only")

CONTROL_RANGES = {
    section: {
        name: bounds
        for name, bounds in ranges.items()
        if isinstance(bounds[0], (int, float))
    }
    for section, ranges in VALIDATION_RANGES.items()
    if section in {"limiter", "deesser", "compressor", "gate"}
}
CONTROL_RANGES["compressor"] = CONTROL_RANGES["compressor"] | {
    "base_release_ms": (20.0, 200.0),
}


@dataclass(frozen=True, slots=True)
class ControlSpec:
    accessible_name: str
    tooltip: str
    step: float = 0.0
    suffix: str = ""
    decimals: int = 2
    choices: tuple[str, ...] = ()
    label: str = ""
    widget_label: str | None = None


CONTROL_SPECS = {
    "limiter": {
        "enabled": ControlSpec(
            "Enable limiter",
            "Prevents signal from exceeding ceiling level.\n"
            "Acts as a safety net to prevent clipping.",
        ),
        "careful_output_enabled": ControlSpec(
            "Enable careful output mode",
            "Adds conservative output headroom by limiting the effective ceiling to -1.5 dB.",
            label="Careful output mode",
        ),
        "ceiling_db": ControlSpec(
            "Limiter ceiling", "Maximum output level (brick-wall ceiling)", 0.1, " dB",
            label="Ceiling",
        ),
        "release_ms": ControlSpec(
            "Limiter release time", "How fast the limiter recovers", 5.0, " ms",
            label="Release",
        ),
    },
    "deesser": {
        "enabled": ControlSpec(
            "Enable de-esser",
            "Reduces harsh sibilance (s, sh, t) using dynamic attenuation.",
        ),
        "auto_enabled": ControlSpec(
            "Enable automatic de-esser",
            "Learns average sibilance and applies dynamic reduction automatically.",
            label="Auto",
        ),
        "auto_amount": ControlSpec("Automatic de-esser amount", "", 0.05, label="Amount"),
        "low_cut_hz": ControlSpec(
            "De-esser low cutoff", "", 100.0, " Hz", label="Low cut", widget_label="Low Cut",
        ),
        "high_cut_hz": ControlSpec(
            "De-esser high cutoff", "", 100.0, " Hz", label="High cut", widget_label="High Cut",
        ),
        "threshold_db": ControlSpec("De-esser threshold", "", 1.0, " dB", label="Threshold"),
        "ratio": ControlSpec("De-esser ratio", "", 0.5, ":1", label="Ratio"),
        "attack_ms": ControlSpec("De-esser attack time", "", 0.1, " ms", label="Attack"),
        "release_ms": ControlSpec("De-esser release time", "", 5.0, " ms", label="Release"),
        "max_reduction_db": ControlSpec(
            "De-esser maximum reduction", "", 0.5, " dB",
            label="Max reduction", widget_label="Max Red",
        ),
    },
    "compressor": {
        "enabled": ControlSpec(
            "Enable compressor",
            "Reduces dynamic range by attenuating loud signals.\n"
            "Helps maintain consistent volume levels.",
        ),
        "threshold_db": ControlSpec(
            "Compressor threshold", "Level above which compression begins", 1.0, " dB",
            label="Threshold",
        ),
        "ratio": ControlSpec(
            "Compressor ratio", "Compression ratio (higher = more compression)", 0.5, ":1",
            label="Ratio",
        ),
        "attack_ms": ControlSpec(
            "Compressor attack time", "How fast the compressor responds to loud signals", 1.0, " ms",
            label="Attack",
        ),
        "release_ms": ControlSpec(
            "Compressor release time", "How fast the compressor recovers after loud signals", 10.0, " ms",
            label="Release",
        ),
        "makeup_gain_db": ControlSpec(
            "Compressor makeup gain", "Gain added after compression to restore volume", 0.5, " dB",
            label="Makeup gain", widget_label="Makeup Gain",
        ),
        "adaptive_release": ControlSpec(
            "Enable adaptive release",
            "Release time adapts based on signal dynamics.\n"
            "Scales from 50ms to 400ms based on sustained overage.\n"
            "Longer release for consistent loud signals, shorter for transients.",
            label="Adaptive release",
        ),
        "base_release_ms": ControlSpec(
            "Adaptive compressor base release",
            "Base release time when adaptive mode is enabled", 5.0, " ms",
            label="Base release", widget_label="Base Release",
        ),
        "sidechain_highpass_enabled": ControlSpec(
            "Enable compressor sidechain high-pass",
            "Ignores low-frequency plosives and rumble in the compressor detector without filtering the audio.",
            label="Sidechain high-pass",
        ),
        "auto_makeup_enabled": ControlSpec(
            "Enable automatic makeup gain",
            "Automatically adjust makeup gain from post-compression EBU R128 loudness measurement.\n"
            "Maintains post-compressor output level relative to target LUFS.\n"
            "Uses the selected Target LUFS value.",
            label="Auto makeup gain",
        ),
        "target_lufs": ControlSpec(
            "Automatic makeup target loudness", "Target loudness level (-24 to -12 LUFS)", 1.0, " LUFS",
            label="Target LUFS",
        ),
    },
    "gate": {
        "enabled": ControlSpec(
            "Enable noise gate",
            "Reduces gain when signal falls below threshold.\n"
            "Helps eliminate background noise during silence.",
        ),
        "threshold_db": ControlSpec(
            "Gate manual threshold", "Signal level below which gate closes", 1.0, " dB",
            label="Threshold",
        ),
        "attack_ms": ControlSpec(
            "Gate attack time", "Time for gate to open when signal exceeds threshold", 1.0, " ms",
            label="Attack",
        ),
        "release_ms": ControlSpec(
            "Gate release time", "Time for gate to close when signal drops below threshold", 10.0, " ms",
            label="Release",
        ),
        "gate_mode": ControlSpec(
            "Gate operating mode",
            "Threshold Only: Traditional gate using level threshold\n"
            "VAD Assisted: Gate opens when level exceeded OR speech detected\n"
            "VAD Only: Gate opens solely based on speech probability",
            choices=GATE_MODES, label="Gate mode", widget_label="Gate Mode",
        ),
        "vad_threshold": ControlSpec(
            "Voice activity threshold", "Speech probability threshold (0.3-0.7)", 0.01,
            label="VAD threshold", widget_label="VAD Threshold",
        ),
        "vad_hold_time_ms": ControlSpec(
            "Voice activity hold time", "Gate hold time after speech ends (prevents chatter)", 10.0, " ms",
            label="Hold time", widget_label="Hold Time",
        ),
        "vad_pre_gain": ControlSpec(
            "Voice activity pre-gain", "Pre-gain to boost weak signals for better VAD detection",
            0.5, decimals=1, label="VAD pre-gain", widget_label="VAD Pre-Gain",
        ),
        "auto_threshold_enabled": ControlSpec(
            "Enable automatic gate threshold",
            "Automatically set the gate level threshold to noise floor + margin.",
            label="Auto threshold", widget_label="Auto Threshold",
        ),
        "gate_margin_db": ControlSpec(
            "Automatic gate margin", "Margin above noise floor for gate threshold (0-20 dB)", 1.0, " dB",
            label="Margin",
        ),
    },
}


def control_spec(section: str, key: str) -> ControlSpec:
    return CONTROL_SPECS[section][key]


def control_label(section: str, key: str) -> str:
    """Return the Quick row label for a setting; use in state_field/state_row."""
    return control_spec(section, key).label


def widget_control_label(section: str, key: str) -> str:
    """Return the classic field label without its existing trailing colon."""
    spec = control_spec(section, key)
    return spec.widget_label or spec.label


def apply_control_presentation(control, section: str, key: str) -> None:
    spec = control_spec(section, key)
    control.setAccessibleName(spec.accessible_name)
    control.setToolTip(spec.tooltip)


def configure_numeric_control(
    spinbox,
    section: str,
    key: str,
    *,
    slider=None,
    slider_scale: float = 1.0,
) -> None:
    spec = control_spec(section, key)
    minimum, maximum = CONTROL_RANGES[section][key]
    spinbox.setRange(minimum, maximum)
    spinbox.setSingleStep(spec.step)
    spinbox.setSuffix(spec.suffix)
    spinbox.setDecimals(spec.decimals)
    spinbox.setToolTip(spec.tooltip)
    spinbox.setAccessibleName(spec.accessible_name)
    if slider is not None:
        slider.setRange(
            int(round(minimum * slider_scale)),
            int(round(maximum * slider_scale)),
        )
