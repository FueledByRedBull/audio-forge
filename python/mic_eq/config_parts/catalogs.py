"""Built-in presets and target curve catalog builders."""

from typing import Any

from .settings import CompressorSettings
from .shared import CURRENT_VERSION


COMPLETE_PRESET_SCOPE_LABEL = "Complete sound preset"
CALIBRATION_TARGET_SCOPE_LABEL = "Calibration target"


def build_builtin_presets(
    preset_cls: type,
    gate_settings_cls: type,
    eq_settings_cls: type,
    rnnoise_settings_cls: type,
) -> dict[str, Any]:
    """Use the public dataclass defaults as the complete-preset baseline."""
    return {
        'voice': preset_cls(
            name="Voice Clarity",
            description=f"{COMPLETE_PRESET_SCOPE_LABEL}: reduced bass and boosted presence, with gate, suppressor, dynamics, and output protection",
            version=CURRENT_VERSION,
            gate=gate_settings_cls(),
            eq=eq_settings_cls(
                enabled=True,
                band_gains=[-3.0, -2.0, 0.0, 1.0, 2.0, 3.0, 2.0, 0.0, -1.0, -2.0],
                band_qs=[0.7, 1.0, 1.2, 1.4, 1.6, 2.0, 1.8, 1.2, 0.9, 0.7]
            ),
            rnnoise=rnnoise_settings_cls(),
        ),
        'bass_cut': preset_cls(
            name="Bass Cut",
            description=f"{COMPLETE_PRESET_SCOPE_LABEL}: trims low bass in the EQ; separate from rumble high-pass filtering",
            version=CURRENT_VERSION,
            gate=gate_settings_cls(),
            eq=eq_settings_cls(
                enabled=True,
                band_gains=[-12.0, -6.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                band_qs=[0.5, 0.7, 0.9, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41]
            ),
            rnnoise=rnnoise_settings_cls(),
        ),
        'presence': preset_cls(
            name="Presence Boost",
            description=f"{COMPLETE_PRESET_SCOPE_LABEL}: boosted presence with gate, suppressor, dynamics, and output protection",
            version=CURRENT_VERSION,
            gate=gate_settings_cls(),
            eq=eq_settings_cls(
                enabled=True,
                band_gains=[0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 3.0, 1.0, 0.0, 0.0],
                band_qs=[1.41, 1.41, 1.41, 1.41, 2.0, 2.5, 2.0, 1.5, 1.41, 1.41]
            ),
            rnnoise=rnnoise_settings_cls(),
        ),
        'flat': preset_cls(
            name="Flat",
            description=f"{COMPLETE_PRESET_SCOPE_LABEL}: neutral EQ with gate, suppressor, dynamics, and output protection",
            version=CURRENT_VERSION,
            gate=gate_settings_cls(),
            eq=eq_settings_cls(
                enabled=True,
                band_gains=[0.0] * 10,
                band_qs=[1.41] * 10
            ),
            rnnoise=rnnoise_settings_cls(),
        ),
        'minimal': preset_cls(
            name="Minimal Processing",
            description=f"{COMPLETE_PRESET_SCOPE_LABEL}: gate and suppressor with output protection; EQ, de-essing, and compression disabled",
            version=CURRENT_VERSION,
            gate=gate_settings_cls(
                threshold_db=-45.0,
                attack_ms=5.0,
                release_ms=150.0,
            ),
            eq=eq_settings_cls(
                enabled=False,
                band_gains=[0.0] * 10,
                band_qs=[1.41] * 10
            ),
            rnnoise=rnnoise_settings_cls(),
            compressor=CompressorSettings(enabled=False),
        ),
        'aggressive_denoise': preset_cls(
            name="Aggressive Denoise",
            description=f"{COMPLETE_PRESET_SCOPE_LABEL}: tighter gate and reduced bass/treble with suppressor, dynamics, and output protection",
            version=CURRENT_VERSION,
            gate=gate_settings_cls(
                threshold_db=-35.0,
                attack_ms=5.0,
                release_ms=50.0,
            ),
            eq=eq_settings_cls(
                enabled=True,
                band_gains=[-6.0, -3.0, 0.0, 0.0, 1.0, 2.0, 1.0, -1.0, -3.0, -6.0],
                band_qs=[0.6, 0.8, 1.2, 1.4, 1.8, 2.0, 1.6, 1.2, 0.8, 0.6]
            ),
            rnnoise=rnnoise_settings_cls(),
        ),
    }


def build_target_curves(target_curve_cls: type) -> dict[str, Any]:
    """Build Auto-EQ target curve catalog using the public TargetCurve dataclass."""
    return {
        'broadcast': target_curve_cls(
            name="Broadcast-style Voice",
            description=f"{CALIBRATION_TARGET_SCOPE_LABEL}: a house curve for clear, balanced broadcast-style speech",
            band_targets=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0]
        ),
        'podcast': target_curve_cls(
            name="Podcast / Voice-Over",
            description=f"{CALIBRATION_TARGET_SCOPE_LABEL} with enhanced presence for intimate vocal recording",
            band_targets=[0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        ),
        'streaming': target_curve_cls(
            name="Streaming / Gaming",
            description=f"{CALIBRATION_TARGET_SCOPE_LABEL} that cuts through game audio with aggressive presence",
            band_targets=[-1.0, 0.0, 1.0, 2.0, 4.0, 5.0, 4.0, 2.0, 0.0, -2.0]
        ),
        'flat': target_curve_cls(
            name="Neutral Reference",
            description=f"{CALIBRATION_TARGET_SCOPE_LABEL} for broad microphone-response correction",
            band_targets=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
    }
