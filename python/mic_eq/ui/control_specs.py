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
            "De-esser low cutoff", "", 100.0, " Hz", label="Low cut",
        ),
        "high_cut_hz": ControlSpec(
            "De-esser high cutoff", "", 100.0, " Hz", label="High cut",
        ),
        "threshold_db": ControlSpec("De-esser threshold", "", 1.0, " dB", label="Threshold"),
        "ratio": ControlSpec("De-esser ratio", "", 0.5, ":1", label="Ratio"),
        "attack_ms": ControlSpec("De-esser attack time", "", 0.1, " ms", label="Attack"),
        "release_ms": ControlSpec("De-esser release time", "", 5.0, " ms", label="Release"),
        "max_reduction_db": ControlSpec(
            "Max reduction (de-esser)", "", 0.5, " dB",
            label="Max reduction",
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
            label="Makeup gain",
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
            label="Base release",
        ),
        "sidechain_highpass_enabled": ControlSpec(
            "Enable compressor sidechain high-pass",
            "Ignores low-frequency plosives and rumble in the compressor detector without filtering the audio.",
            label="Sidechain high-pass",
        ),
        "auto_makeup_enabled": ControlSpec(
            "Auto makeup gain (compressor)",
            "Automatically adjust makeup gain from post-compression EBU R128 loudness measurement.\n"
            "Maintains post-compressor output level relative to target LUFS.\n"
            "Uses the selected Target LUFS value.",
            label="Auto makeup gain",
        ),
        "target_lufs": ControlSpec(
            "Target LUFS (automatic makeup)", "Target loudness level (-24 to -12 LUFS)", 1.0, " LUFS",
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
            "Gate mode",
            "Threshold Only: Traditional gate using level threshold\n"
            "VAD Assisted: Gate opens when level exceeded OR speech detected\n"
            "VAD Only: Gate opens solely based on speech probability",
            choices=GATE_MODES, label="Gate mode",
        ),
        "vad_threshold": ControlSpec(
            "Voice threshold (speech detection)", "Speech probability threshold (0.3-0.7)", 0.01,
            label="Voice threshold",
        ),
        "vad_hold_time_ms": ControlSpec(
            "Voice activity hold time", "Gate hold time after speech ends (prevents chatter)", 10.0, " ms",
            label="Hold time",
        ),
        "vad_pre_gain": ControlSpec(
            "Voice pre-gain (speech detection)", "Pre-gain to boost weak signals for better speech detection",
            0.5, decimals=1, label="Voice pre-gain",
        ),
        "auto_threshold_enabled": ControlSpec(
            "Auto threshold (gate)",
            "Automatically set the gate level threshold to noise floor + margin.",
            label="Auto threshold",
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
