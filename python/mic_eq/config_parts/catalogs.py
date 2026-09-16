"""Built-in presets and target curve catalog builders."""

from .presets import Preset
from .settings import CompressorSettings, EQSettings, GateSettings, RNNoiseSettings, TargetCurve
from .shared import CURRENT_VERSION


COMPLETE_PRESET_SCOPE_LABEL = "Complete sound preset"
CALIBRATION_TARGET_SCOPE_LABEL = "Calibration target"


def build_builtin_presets() -> dict[str, Preset]:
    """Use the public dataclass defaults as the complete-preset baseline."""
    return {
        'voice': Preset(
            name="Voice Clarity",
            description=f"{COMPLETE_PRESET_SCOPE_LABEL}: reduced bass and boosted presence, with gate, suppressor, dynamics, and output protection",
            version=CURRENT_VERSION,
            gate=GateSettings(),
            eq=EQSettings(
                enabled=True,
                band_gains=[-3.0, -2.0, 0.0, 1.0, 2.0, 3.0, 2.0, 0.0, -1.0, -2.0],
                band_qs=[0.7, 1.0, 1.2, 1.4, 1.6, 2.0, 1.8, 1.2, 0.9, 0.7]
            ),
            rnnoise=RNNoiseSettings(),
        ),
        'bass_cut': Preset(
            name="Bass Cut",
            description=f"{COMPLETE_PRESET_SCOPE_LABEL}: trims low bass in the EQ; separate from rumble high-pass filtering",
            version=CURRENT_VERSION,
            gate=GateSettings(),
            eq=EQSettings(
                enabled=True,
                band_gains=[-12.0, -6.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                band_qs=[0.5, 0.7, 0.9, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41]
            ),
            rnnoise=RNNoiseSettings(),
        ),
        'presence': Preset(
            name="Presence Boost",
            description=f"{COMPLETE_PRESET_SCOPE_LABEL}: boosted presence with gate, suppressor, dynamics, and output protection",
            version=CURRENT_VERSION,
            gate=GateSettings(),
            eq=EQSettings(
                enabled=True,
                band_gains=[0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 3.0, 1.0, 0.0, 0.0],
                band_qs=[1.41, 1.41, 1.41, 1.41, 2.0, 2.5, 2.0, 1.5, 1.41, 1.41]
            ),
            rnnoise=RNNoiseSettings(),
        ),
        'flat': Preset(
            name="Flat",
            description=f"{COMPLETE_PRESET_SCOPE_LABEL}: neutral EQ with gate, suppressor, dynamics, and output protection",
            version=CURRENT_VERSION,
            gate=GateSettings(),
            eq=EQSettings(
                enabled=True,
                band_gains=[0.0] * 10,
                band_qs=[1.41] * 10
            ),
            rnnoise=RNNoiseSettings(),
        ),
        'minimal': Preset(
            name="Minimal Processing",
            description=f"{COMPLETE_PRESET_SCOPE_LABEL}: gate and suppressor with output protection; EQ, de-essing, and compression disabled",
            version=CURRENT_VERSION,
            gate=GateSettings(
                threshold_db=-45.0,
                attack_ms=5.0,
                release_ms=150.0,
            ),
            eq=EQSettings(
                enabled=False,
                band_gains=[0.0] * 10,
                band_qs=[1.41] * 10
            ),
            rnnoise=RNNoiseSettings(),
            compressor=CompressorSettings(enabled=False),
        ),
        'aggressive_denoise': Preset(
            name="Aggressive Denoise",
            description=f"{COMPLETE_PRESET_SCOPE_LABEL}: tighter gate and reduced bass/treble with suppressor, dynamics, and output protection",
            version=CURRENT_VERSION,
            gate=GateSettings(
                threshold_db=-35.0,
                attack_ms=5.0,
                release_ms=50.0,
            ),
            eq=EQSettings(
                enabled=True,
                band_gains=[-6.0, -3.0, 0.0, 0.0, 1.0, 2.0, 1.0, -1.0, -3.0, -6.0],
                band_qs=[0.6, 0.8, 1.2, 1.4, 1.8, 2.0, 1.6, 1.2, 0.8, 0.6]
            ),
            rnnoise=RNNoiseSettings(),
        ),
    }


def build_target_curves() -> dict[str, TargetCurve]:
    """Build Auto-EQ target curve catalog using the public TargetCurve dataclass."""
    return {
        'broadcast': TargetCurve(
            name="Broadcast-style Voice",
            description=f"{CALIBRATION_TARGET_SCOPE_LABEL}: a house curve for clear, balanced broadcast-style speech",
            band_targets=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0]
        ),
        'podcast': TargetCurve(
            name="Podcast / Voice-Over",
            description=f"{CALIBRATION_TARGET_SCOPE_LABEL} with enhanced presence for intimate vocal recording",
            band_targets=[0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        ),
        'streaming': TargetCurve(
            name="Streaming / Gaming",
            description=f"{CALIBRATION_TARGET_SCOPE_LABEL} that cuts through game audio with aggressive presence",
            band_targets=[-1.0, 0.0, 1.0, 2.0, 4.0, 5.0, 4.0, 2.0, 0.0, -2.0]
        ),
        'flat': TargetCurve(
            name="Neutral Reference",
            description=f"{CALIBRATION_TARGET_SCOPE_LABEL} for broad microphone-response correction",
            band_targets=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
    }
