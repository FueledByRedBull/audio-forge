"""Built-in presets and target curve catalog builders."""

from typing import Any


def build_builtin_presets(
    preset_cls: type,
    gate_settings_cls: type,
    eq_settings_cls: type,
    rnnoise_settings_cls: type,
) -> dict[str, Any]:
    """Build bundled presets using the public config dataclasses."""
    return {
        'voice': preset_cls(
            name="Voice Clarity",
            description="Optimized for voice communication - cuts low end rumble and boosts presence",
            version="1.7.18",
            gate=gate_settings_cls(enabled=True, threshold_db=-40.0, attack_ms=10.0, release_ms=100.0,
                                   gate_mode=0, vad_threshold=0.4, vad_hold_time_ms=200.0, vad_pre_gain=1.0,
                                   auto_threshold_enabled=True, gate_margin_db=10.0),
            eq=eq_settings_cls(
                enabled=True,
                band_gains=[-3.0, -2.0, 0.0, 1.0, 2.0, 3.0, 2.0, 0.0, -1.0, -2.0],
                band_qs=[0.7, 1.0, 1.2, 1.4, 1.6, 2.0, 1.8, 1.2, 0.9, 0.7]
            ),
            rnnoise=rnnoise_settings_cls(enabled=True, strength=1.0, model='rnnoise'),
        ),
        'bass_cut': preset_cls(
            name="Bass Cut",
            description="High-pass effect to remove low frequency rumble and proximity effect",
            version="1.7.18",
            gate=gate_settings_cls(enabled=True, threshold_db=-40.0, attack_ms=10.0, release_ms=100.0,
                                   gate_mode=0, vad_threshold=0.4, vad_hold_time_ms=200.0, vad_pre_gain=1.0,
                                   auto_threshold_enabled=True, gate_margin_db=10.0),
            eq=eq_settings_cls(
                enabled=True,
                band_gains=[-12.0, -6.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                band_qs=[0.5, 0.7, 0.9, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41]
            ),
            rnnoise=rnnoise_settings_cls(enabled=True, strength=1.0, model='rnnoise'),
        ),
        'presence': preset_cls(
            name="Presence Boost",
            description="Enhances voice presence and intelligibility",
            version="1.7.18",
            gate=gate_settings_cls(enabled=True, threshold_db=-40.0, attack_ms=10.0, release_ms=100.0,
                                   gate_mode=0, vad_threshold=0.4, vad_hold_time_ms=200.0, vad_pre_gain=1.0,
                                   auto_threshold_enabled=True, gate_margin_db=10.0),
            eq=eq_settings_cls(
                enabled=True,
                band_gains=[0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 3.0, 1.0, 0.0, 0.0],
                band_qs=[1.41, 1.41, 1.41, 1.41, 2.0, 2.5, 2.0, 1.5, 1.41, 1.41]
            ),
            rnnoise=rnnoise_settings_cls(enabled=True, strength=1.0, model='rnnoise'),
        ),
        'flat': preset_cls(
            name="Flat",
            description="No EQ processing - flat frequency response",
            version="1.7.18",
            gate=gate_settings_cls(enabled=True, threshold_db=-40.0, attack_ms=10.0, release_ms=100.0,
                                   gate_mode=0, vad_threshold=0.4, vad_hold_time_ms=200.0, vad_pre_gain=1.0,
                                   auto_threshold_enabled=True, gate_margin_db=10.0),
            eq=eq_settings_cls(
                enabled=True,
                band_gains=[0.0] * 10,
                band_qs=[1.41] * 10
            ),
            rnnoise=rnnoise_settings_cls(enabled=True, strength=1.0, model='rnnoise'),
        ),
        'minimal': preset_cls(
            name="Minimal Processing",
            description="Gate and RNNoise only - no EQ",
            version="1.7.18",
            gate=gate_settings_cls(enabled=True, threshold_db=-45.0, attack_ms=5.0, release_ms=150.0,
                                   gate_mode=0, vad_threshold=0.4, vad_hold_time_ms=200.0, vad_pre_gain=1.0,
                                   auto_threshold_enabled=True, gate_margin_db=10.0),
            eq=eq_settings_cls(
                enabled=False,
                band_gains=[0.0] * 10,
                band_qs=[1.41] * 10
            ),
            rnnoise=rnnoise_settings_cls(enabled=True, strength=1.0, model='rnnoise'),
        ),
        'aggressive_denoise': preset_cls(
            name="Aggressive Denoise",
            description="Maximum noise reduction with tight gate",
            version="1.7.18",
            gate=gate_settings_cls(enabled=True, threshold_db=-35.0, attack_ms=5.0, release_ms=50.0,
                                   gate_mode=0, vad_threshold=0.4, vad_hold_time_ms=200.0, vad_pre_gain=1.0,
                                   auto_threshold_enabled=True, gate_margin_db=10.0),
            eq=eq_settings_cls(
                enabled=True,
                band_gains=[-6.0, -3.0, 0.0, 0.0, 1.0, 2.0, 1.0, -1.0, -3.0, -6.0],
                band_qs=[0.6, 0.8, 1.2, 1.4, 1.8, 2.0, 1.6, 1.2, 0.8, 0.6]
            ),
            rnnoise=rnnoise_settings_cls(enabled=True, strength=1.0, model='rnnoise'),
        ),
    }


def build_target_curves(target_curve_cls: type) -> dict[str, Any]:
    """Build Auto-EQ target curve catalog using the public TargetCurve dataclass."""
    return {
        'broadcast': target_curve_cls(
            name="Broadcast Standard",
            description="ITU-R BS.1770 compliant - professional broadcast voice",
            band_targets=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0]
        ),
        'podcast': target_curve_cls(
            name="Podcast / Voice-Over",
            description="Enhanced presence for intimate vocal recording",
            band_targets=[0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        ),
        'streaming': target_curve_cls(
            name="Streaming / Gaming",
            description="Cuts through game audio mix with aggressive presence",
            band_targets=[-1.0, 0.0, 1.0, 2.0, 4.0, 5.0, 4.0, 2.0, 0.0, -2.0]
        ),
        'flat': target_curve_cls(
            name="Flat Response",
            description="No frequency correction - measure mic as-is",
            band_targets=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
    }
