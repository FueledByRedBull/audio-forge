"""What both views show before any live data arrives, as tables.

The classic widgets and the Qt Quick scene render the states built from these.
Neither view module owns them, so neither has to import the other.
"""

from __future__ import annotations

from typing import NamedTuple

INPUT_CHANNEL_MODE_OPTIONS = (
    ("Average", "average"),
    ("Left", "left"),
    ("Right", "right"),
    ("Max RMS", "max_rms"),
    ("Phase-safe mono", "phase_safe_mono"),
)
INPUT_CLEANUP_MODE_OPTIONS = (
    ("Off", "off"),
    ("Gentle", "gentle"),
    ("Strong", "strong"),
)
PROCESSING_MODE_OPTIONS = (
    ("Normal", "normal"),
    ("Bypass", "bypass"),
    ("Raw Monitor", "raw"),
)
DROPPED_DIAGNOSTICS_TOOLTIP = (
    "Audio that was dropped, with the engine's other warning counters.\n"
    "Right-click to reset the dropped-sample count."
)


class _TextSpec(NamedTuple):
    text: str
    name: str
    tip: str = ""
    shown: bool = True


class _ChoiceSpec(NamedTuple):
    name: str
    tip: str = ""
    options: tuple[tuple[str, str], ...] = ()


class _ActionSpec(NamedTuple):
    text: str
    tip: str = ""
    name: str = ""
    enabled: bool = True
    checkable: bool = False


# Keyed by the label attribute each text stands for in the classic view.
TEXT_SPECS = {
    "device_warning_banner": _TextSpec(
        "Warning: No audio devices detected. Check your audio drivers and connections.",
        "Audio device warning",
        shown=False,
    ),
    "config_warning_banner": _TextSpec("", "Configuration warning", shown=False),
    "preset_status_label": _TextSpec(
        "Preset: Default (saved)", "Active preset and saved state"
    ),
    "transmission_status_label": _TextSpec(
        "Transmission: Stopped", "Audio transmission state"
    ),
    "calibration_status_label": _TextSpec(
        "No saved calibration", "Saved calibration validity"
    ),
    "health_summary_label": _TextSpec("Health: --", "Compact audio health summary"),
    "route_status_label": _TextSpec("Route: --", "Current audio route"),
    "health_advice_label": _TextSpec(
        "Start processing to see audio health.", "Audio health advice"
    ),
    "input_health_label": _TextSpec(
        "Input: --",
        "Input health",
        "How loud your microphone peaks over the last few seconds. Warns before it reaches full scale and clips.",
    ),
    "output_health_label": _TextSpec(
        "Output: --",
        "Output health",
        "Whether the sound leaving AudioForge is at risk of being cut off, and how hard the limiter is working.",
    ),
    "gate_health_label": _TextSpec(
        "Gate: --",
        "Gate health",
        "Warns when the noise gate opens and closes rapidly, which makes your voice stutter.",
    ),
    "backend_diag_label": _TextSpec(
        "Backend: --",
        "Noise suppression backend health",
        "Which noise-suppression model is running and whether it is working.",
    ),
    "callback_health_label": _TextSpec(
        "Device activity: --",
        "Audio device activity",
        "How long ago the input and output devices last sent or asked for sound. Warns when a device looks stuck.",
    ),
    "underrun_health_label": _TextSpec(
        "Dropouts: --",
        "Output dropout health",
        "Times the output ran out of audio and the sound gapped. Warns on recent or repeated dropouts.",
    ),
    "latency_label": _TextSpec(
        "Latency: --",
        "Processing latency",
        "The delay between your voice and the output, and how long processing takes for each small chunk of audio.",
    ),
    "buffer_label": _TextSpec(
        "Buffer: --",
        "Audio buffer health",
        "Audio waiting to be processed.\n"
        "OK is normal, FILLING UP means a build-up, BACKED UP means a large backlog.",
    ),
    "dropped_label": _TextSpec(
        "Drops: --", "Dropped audio samples", DROPPED_DIAGNOSTICS_TOOLTIP
    ),
    "recovery_diag_label": _TextSpec(
        "Recovery: --",
        "Stream recovery health",
        "Times the audio stream restarted, and output recoveries.\n"
        "Small timing corrections are normal and do not warn.",
    ),
    # Only the widget view shows it, after the Qt Quick view could not take over.
    "interface_notice": _TextSpec("", "Interface notice", shown=False),
}
# The chips the health summary is computed from, in the order it reports them.
HEALTH_DECISION_KEYS = (
    "input_health_label",
    "output_health_label",
    "gate_health_label",
    "backend_diag_label",
    "callback_health_label",
    "underrun_health_label",
)
HEALTH_LAYOUT_KEYS = (
    "latency_label",
    "buffer_label",
    "dropped_label",
    "recovery_diag_label",
)
# Keyed by the prefix of the combo attribute each choice stands for.
CHOICE_SPECS = {
    "input": _ChoiceSpec("Input audio device"),
    "output": _ChoiceSpec("Output audio device"),
    "processing_mode": _ChoiceSpec(
        "Processing mode",
        "Normal applies the voice chain. Bypass keeps input conditioning and output protection. "
        "Raw Monitor skips input filtering and the voice chain for diagnostics; it is session-only "
        "and is not stored in presets.",
        PROCESSING_MODE_OPTIONS,
    ),
    "input_channel_mode": _ChoiceSpec(
        "Input channel mode",
        "How multichannel input is converted to mono. Use Left/Right or Phase-safe mono if stereo channels cancel.",
        INPUT_CHANNEL_MODE_OPTIONS,
    ),
    "input_cleanup_mode": _ChoiceSpec(
        "Input cleanup mode",
        "Optional adaptive input cleanup after the fixed safe pre-filter. Off preserves the existing DC/80 Hz path.",
        INPUT_CLEANUP_MODE_OPTIONS,
    ),
}
# Keyed by the button attribute each action stands for.
ACTION_SPECS = {
    "refresh_btn": _ActionSpec(
        "Refresh", "Refresh audio devices", "Refresh audio devices"
    ),
    "start_btn": _ActionSpec("Start Processing"),
    "stop_btn": _ActionSpec("Stop Processing", enabled=False),
    "_undo_auto_eq_button": _ActionSpec(
        "Undo",
        "Undo the most recent processing-configuration edit (Ctrl+Z)",
        enabled=False,
    ),
    "redo_button": _ActionSpec(
        "Redo", "Redo the edit that was just undone (Ctrl+Shift+Z)", enabled=False
    ),
    "test_sound_button": _ActionSpec(
        "Test my sound",
        "Record five seconds, then play back the raw recording and the "
        "result of the current settings. Output is muted while you listen.",
    ),
    "auto_eq_button": _ActionSpec(
        "Auto-EQ",
        "Record a short passage and set the EQ toward a tone preset\n"
        "Choose a tone preset, read a short passage, then review the result",
    ),
    "auto_voice_setup_button": _ActionSpec(
        "Auto Voice Setup",
        "Record room noise and speech, then calibrate EQ, gate/VAD,\n"
        "de-esser, and compressor in one pass",
    ),
    "user_mute_checkbox": _ActionSpec(
        "Mute",
        "Keep transmission muted until you turn this off. "
        "Calibration and stream recovery use a separate temporary mute.",
        "Mute output",
        checkable=True,
    ),
}
