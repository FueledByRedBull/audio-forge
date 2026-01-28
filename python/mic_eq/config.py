"""
Configuration and preset management for MicEq

Handles saving and loading of presets (JSON format).
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


class PresetValidationError(Exception):
    """Raised when preset validation fails with actionable message."""
    pass


# Default preset directory
def get_presets_dir() -> Path:
    """Get the presets directory, creating it if necessary."""
    if os.name == 'nt':  # Windows
        base = Path(os.environ.get('APPDATA', Path.home()))
    else:
        base = Path.home() / '.config'

    presets_dir = base / 'MicEq' / 'presets'
    presets_dir.mkdir(parents=True, exist_ok=True)
    return presets_dir


def get_config_file() -> Path:
    """Get the main config file path."""
    if os.name == 'nt':  # Windows
        base = Path(os.environ.get('APPDATA', Path.home()))
    else:
        base = Path.home() / '.config'

    config_dir = base / 'MicEq'
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / 'config.json'


@dataclass
class GateSettings:
    """Noise gate settings."""
    enabled: bool = True
    threshold_db: float = -40.0
    attack_ms: float = 10.0
    release_ms: float = 100.0
    # VAD settings (v1.2.0+)
    gate_mode: int = 0  # 0=ThresholdOnly, 1=VadAssisted, 2=VadOnly
    vad_threshold: float = 0.5  # Speech probability threshold (0.3-0.8)
    vad_hold_time_ms: float = 200.0  # Hold time in milliseconds (0-500)


@dataclass
class EQSettings:
    """10-band parametric EQ settings."""
    enabled: bool = True
    # Gains for each band in dB (-12 to +12)
    # Bands: 80Hz (LS), 160, 320, 640, 1.2k, 2.5k, 5k, 8k, 12k, 16kHz (HS)
    band_gains: list[float] = field(default_factory=lambda: [0.0] * 10)
    # Q factor for each band (0.1 to 10.0)
    # Lower Q = wider bandwidth (~3 octaves at 0.1)
    # Higher Q = narrower bandwidth (~0.1 octave at 10.0)
    # Default 1.41 = ~1 octave bandwidth (standard parametric EQ)
    band_qs: list[float] = field(default_factory=lambda: [1.41] * 10)


@dataclass
class RNNoiseSettings:
    """Noise suppression settings (RNNoise or DeepFilterNet)."""
    enabled: bool = True
    strength: float = 1.0  # 0.0 = dry, 1.0 = fully processed
    model: str = "rnnoise"  # "rnnoise" or "deepfilter"


@dataclass
class CompressorSettings:
    """Compressor settings."""
    enabled: bool = True
    threshold_db: float = -20.0
    ratio: float = 4.0
    attack_ms: float = 10.0
    release_ms: float = 200.0
    makeup_gain_db: float = 0.0
    adaptive_release: bool = False  # v1.2.0+
    base_release_ms: float = 50.0  # v1.2.0+
    auto_makeup_enabled: bool = False  # v1.3.0+
    target_lufs: float = -18.0  # v1.3.0+


@dataclass
class LimiterSettings:
    """Hard limiter settings."""
    enabled: bool = True
    ceiling_db: float = -0.5
    release_ms: float = 50.0


@dataclass
class TargetCurve:
    """Target frequency response curve for Auto-EQ calibration."""
    name: str  # Display name (e.g., "Broadcast Standard")
    description: str  # What this curve is for (1-2 sentences)
    # 10-band frequency targets in dB (-12 to +12)
    # Bands: 80, 160, 320, 640, 1.2k, 2.5k, 5k, 8k, 12k, 16kHz
    band_targets: list[float]  # Must be exactly 10 values


# Validation ranges for preset parameters
VALIDATION_RANGES = {
    'gate': {
        'threshold_db': (-80.0, -10.0),
        'attack_ms': (0.1, 100.0),
        'release_ms': (10.0, 1000.0),
        'gate_mode': (0, 2),  # Integer enum
        'vad_threshold': (0.3, 0.8),  # Speech probability
        'vad_hold_time_ms': (0.0, 500.0),  # Milliseconds
    },
    'eq': {
        'band_gain': (-12.0, 12.0),
        'band_q': (0.1, 10.0),
    },
    'rnnoise': {
        'strength': (0.0, 1.0),  # 0% to 100%
        'model': ['rnnoise', 'deepfilter'],  # Valid model choices
    },
    'compressor': {
        'threshold_db': (-60.0, 0.0),
        'ratio': (1.0, 20.0),
        'attack_ms': (0.1, 100.0),
        'release_ms': (10.0, 1000.0),
        'makeup_gain_db': (0.0, 24.0),
        'adaptive_release': (bool, None),  # Boolean flag
        'auto_makeup_enabled': (bool, None),  # Boolean flag
        'target_lufs': (-24.0, -12.0),  # LUFS range
    },
    'limiter': {
        'ceiling_db': (-12.0, 0.0),
        'release_ms': (10.0, 500.0),
    },
}


def _validate_range(value: float, min_val: float, max_val: float, param_name: str, section: str) -> float:
    """Validate and clamp a value to a range, raising error if way out of bounds."""
    # Allow small tolerance (10%) beyond range before rejecting
    tolerance = (max_val - min_val) * 0.1
    if value < min_val - tolerance or value > max_val + tolerance:
        raise PresetValidationError(
            f"Invalid {param_name} in {section}: {value} "
            f"(must be between {min_val} and {max_val})"
        )
    # Clamp to exact range
    return max(min_val, min(max_val, value))


@dataclass
class Preset:
    """Complete preset with all settings."""
    name: str = "Default"
    description: str = ""
    version: str = "1.5.0"  # Version field for migration
    gate: GateSettings = field(default_factory=GateSettings)
    eq: EQSettings = field(default_factory=EQSettings)
    rnnoise: RNNoiseSettings = field(default_factory=RNNoiseSettings)
    compressor: CompressorSettings = field(default_factory=CompressorSettings)
    limiter: LimiterSettings = field(default_factory=LimiterSettings)
    bypass: bool = False

    def to_dict(self) -> dict:
        """Convert preset to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'gate': asdict(self.gate),
            'eq': asdict(self.eq),
            'rnnoise': asdict(self.rnnoise),
            'compressor': asdict(self.compressor),
            'limiter': asdict(self.limiter),
            'bypass': self.bypass,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Preset':
        """Create preset from dictionary with validation."""
        try:
            # Version-aware migration
            version = data.get('version', '1.0.0')

            # Migrate v1.0 presets → v1.1
            if version < '1.1.0':
                # Add missing strength field to RNNoise settings
                if 'rnnoise' in data:
                    data['rnnoise'].setdefault('strength', 1.0)  # Default to full processing
                else:
                    data['rnnoise'] = {'enabled': True, 'strength': 1.0}

                # Update version
                data['version'] = '1.1.0'
                version = '1.1.0'

            # Migrate v1.1 presets → v1.2
            if version < '1.2.0':
                # Add missing model field to RNNoise settings
                if 'rnnoise' in data:
                    data['rnnoise'].setdefault('model', 'rnnoise')  # Default to RNNoise
                else:
                    data['rnnoise'] = {'enabled': True, 'strength': 1.0, 'model': 'rnnoise'}

                # Update version
                data['version'] = '1.2.0'

            # Migrate v1.2 presets → v1.3
            if version < '1.3.0':
                # Add missing auto makeup gain fields to compressor settings
                if 'compressor' in data:
                    data['compressor'].setdefault('auto_makeup_enabled', False)
                    data['compressor'].setdefault('target_lufs', -18.0)
                else:
                    data['compressor'] = {
                        'enabled': True,
                        'threshold_db': -20.0,
                        'ratio': 4.0,
                        'attack_ms': 10.0,
                        'release_ms': 200.0,
                        'makeup_gain_db': 0.0,
                        'adaptive_release': False,
                        'base_release_ms': 50.0,
                        'auto_makeup_enabled': False,
                        'target_lufs': -18.0,
                    }

                # Update version
                data['version'] = '1.3.0'

            # Migrate v1.3 presets → v1.4 (no format changes, version bump only)
            if version < '1.4.0':
                data['version'] = '1.4.0'
                version = '1.4.0'

            # Migrate v1.4 presets → v1.5 (no format changes, version bump only)
            if version < '1.5.0':
                data['version'] = '1.5.0'
                version = '1.5.0'

            # Extract and validate gate settings
            gate_data = data.get('gate', {})
            gate_ranges = VALIDATION_RANGES['gate']
            validated_gate = GateSettings(
                enabled=gate_data.get('enabled', True),
                threshold_db=_validate_range(
                    gate_data.get('threshold_db', -40.0),
                    *gate_ranges['threshold_db'],
                    'threshold_db', 'gate'
                ),
                attack_ms=_validate_range(
                    gate_data.get('attack_ms', 10.0),
                    *gate_ranges['attack_ms'],
                    'attack_ms', 'gate'
                ),
                release_ms=_validate_range(
                    gate_data.get('release_ms', 100.0),
                    *gate_ranges['release_ms'],
                    'release_ms', 'gate'
                ),
                gate_mode=int(_validate_range(  # Convert to int
                    gate_data.get('gate_mode', 0),  # Default to ThresholdOnly
                    *gate_ranges['gate_mode'],
                    'gate_mode', 'gate'
                )),
                vad_threshold=_validate_range(
                    gate_data.get('vad_threshold', 0.5),  # Default 0.5
                    *gate_ranges['vad_threshold'],
                    'vad_threshold', 'gate'
                ),
                vad_hold_time_ms=_validate_range(
                    gate_data.get('vad_hold_time_ms', 200.0),  # Default 200ms
                    *gate_ranges['vad_hold_time_ms'],
                    'vad_hold_time_ms', 'gate'
                ),
            )

            # Extract and validate EQ settings
            eq_data = data.get('eq', {})
            band_gains = eq_data.get('band_gains', [0.0] * 10)
            band_qs = eq_data.get('band_qs', [1.41] * 10)

            # Validate each band gain and Q
            eq_ranges = VALIDATION_RANGES['eq']
            validated_gains = [
                _validate_range(gain, *eq_ranges['band_gain'], f'band_gains[{i}]', 'eq')
                for i, gain in enumerate(band_gains)
            ]
            validated_qs = [
                _validate_range(q, *eq_ranges['band_q'], f'band_qs[{i}]', 'eq')
                for i, q in enumerate(band_qs)
            ]

            validated_eq = EQSettings(
                enabled=eq_data.get('enabled', True),
                band_gains=validated_gains,
                band_qs=validated_qs,
            )

            # Extract and validate compressor settings
            comp_data = data.get('compressor', {})
            comp_ranges = VALIDATION_RANGES['compressor']
            validated_comp = CompressorSettings(
                enabled=comp_data.get('enabled', True),
                threshold_db=_validate_range(
                    comp_data.get('threshold_db', -20.0),
                    *comp_ranges['threshold_db'],
                    'threshold_db', 'compressor'
                ),
                ratio=_validate_range(
                    comp_data.get('ratio', 4.0),
                    *comp_ranges['ratio'],
                    'ratio', 'compressor'
                ),
                attack_ms=_validate_range(
                    comp_data.get('attack_ms', 10.0),
                    *comp_ranges['attack_ms'],
                    'attack_ms', 'compressor'
                ),
                release_ms=_validate_range(
                    comp_data.get('release_ms', 200.0),
                    *comp_ranges['release_ms'],
                    'release_ms', 'compressor'
                ),
                makeup_gain_db=_validate_range(
                    comp_data.get('makeup_gain_db', 0.0),
                    *comp_ranges['makeup_gain_db'],
                    'makeup_gain_db', 'compressor'
                ),
                adaptive_release=comp_data.get('adaptive_release', False),
                base_release_ms=_validate_range(
                    comp_data.get('base_release_ms', 50.0),
                    20.0,
                    200.0,
                    'base_release_ms', 'compressor'
                ),
                auto_makeup_enabled=comp_data.get('auto_makeup_enabled', False),
                target_lufs=_validate_range(
                    comp_data.get('target_lufs', -18.0),
                    *comp_ranges['target_lufs'],
                    'target_lufs', 'compressor'
                ),
            )

            # Extract and validate limiter settings
            lim_data = data.get('limiter', {})
            lim_ranges = VALIDATION_RANGES['limiter']
            validated_lim = LimiterSettings(
                enabled=lim_data.get('enabled', True),
                ceiling_db=_validate_range(
                    lim_data.get('ceiling_db', -0.5),
                    *lim_ranges['ceiling_db'],
                    'ceiling_db', 'limiter'
                ),
                release_ms=_validate_range(
                    lim_data.get('release_ms', 50.0),
                    *lim_ranges['release_ms'],
                    'release_ms', 'limiter'
                ),
            )

            # Extract and validate RNNoise settings
            rnnoise_data = data.get('rnnoise', {})
            rnnoise_ranges = VALIDATION_RANGES.get('rnnoise', {})

            # Validate model choice
            model = rnnoise_data.get('model', 'rnnoise')
            valid_models = rnnoise_ranges.get('model', ['rnnoise', 'deepfilter'])
            if model not in valid_models:
                model = 'rnnoise'  # Fallback to default

            validated_rnnoise = RNNoiseSettings(
                enabled=rnnoise_data.get('enabled', True),
                strength=_validate_range(
                    rnnoise_data.get('strength', 1.0),
                    *rnnoise_ranges.get('strength', (0.0, 1.0)),
                    'strength', 'rnnoise'
                ),
                model=model,
            )

            return cls(
                name=data.get('name', 'Unnamed'),
                description=data.get('description', ''),
                version=data.get('version', '1.2.0'),
                gate=validated_gate,
                eq=validated_eq,
                rnnoise=validated_rnnoise,
                compressor=validated_comp,
                limiter=validated_lim,
                bypass=data.get('bypass', False),
            )
        except (KeyError, TypeError, ValueError) as e:
            # Convert generic errors to actionable validation errors
            raise PresetValidationError(
                f"Preset data is invalid or corrupted: {e}"
            )


def save_preset(preset: Preset, filepath: Optional[Path] = None) -> Path:
    """
    Save a preset to a JSON file.

    Args:
        preset: The preset to save
        filepath: Optional custom path. If None, saves to presets directory.

    Returns:
        The path where the preset was saved.
    """
    if filepath is None:
        # Generate filename from preset name
        safe_name = "".join(c if c.isalnum() or c in ' -_' else '_' for c in preset.name)
        safe_name = safe_name.strip().replace(' ', '_')
        if not safe_name:
            safe_name = "preset"
        filepath = get_presets_dir() / f"{safe_name}.json"

    filepath = Path(filepath)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(preset.to_dict(), f, indent=2)

    return filepath


def load_preset(filepath: Path) -> Preset:
    """
    Load a preset from a JSON file.

    Args:
        filepath: Path to the preset file.

    Returns:
        The loaded preset.

    Raises:
        PresetValidationError: If validation fails or path is unsafe.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    filepath = Path(filepath)

    # Path traversal protection - check before resolving
    path_str = str(filepath)
    if '..' in path_str:
        raise PresetValidationError(
            f"Invalid preset path: '{filepath.name}' - path traversal not allowed"
        )

    # Resolve to absolute path
    filepath = filepath.resolve()
    path_str = str(filepath)

    # Block system directories
    if path_str.startswith('/etc') or path_str.startswith('C:\\Windows'):
        raise PresetValidationError(
            f"Invalid preset path: '{filepath.name}' - system paths not allowed"
        )

    if not filepath.exists():
        raise PresetValidationError(f"Preset file not found: '{filepath.name}'")

    if not filepath.suffix.lower() == '.json':
        raise PresetValidationError(
            f"Invalid preset file: '{filepath.name}' - must be a .json file"
        )

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return Preset.from_dict(data)


def list_presets() -> list[tuple[str, Path]]:
    """
    List all available presets.

    Returns:
        List of (preset_name, filepath) tuples.
    """
    presets_dir = get_presets_dir()
    presets = []

    for filepath in presets_dir.glob('*.json'):
        try:
            preset = load_preset(filepath)
            presets.append((preset.name, filepath))
        except (json.JSONDecodeError, KeyError):
            # Skip invalid files
            continue

    return sorted(presets, key=lambda x: x[0].lower())


@dataclass
class AppConfig:
    """Application configuration (persisted settings)."""
    last_input_device: str = ""
    last_output_device: str = ""
    last_preset: str = ""
    window_geometry: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'last_input_device': self.last_input_device,
            'last_output_device': self.last_output_device,
            'last_preset': self.last_preset,
            'window_geometry': self.window_geometry,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AppConfig':
        """Create config from dictionary."""
        return cls(
            last_input_device=data.get('last_input_device', ''),
            last_output_device=data.get('last_output_device', ''),
            last_preset=data.get('last_preset', ''),
            window_geometry=data.get('window_geometry'),
        )


def save_config(config: AppConfig) -> None:
    """Save application configuration."""
    filepath = get_config_file()
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=2)


def load_config() -> AppConfig:
    """Load application configuration, returning defaults if not found."""
    filepath = get_config_file()

    if not filepath.exists():
        return AppConfig()

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return AppConfig.from_dict(data)
    except (json.JSONDecodeError, KeyError):
        return AppConfig()


# Built-in presets
BUILTIN_PRESETS = {
    'voice': Preset(
        name="Voice Clarity",
        description="Optimized for voice communication - cuts low end rumble and boosts presence",
        version="1.2.0",
        gate=GateSettings(enabled=True, threshold_db=-40.0, attack_ms=10.0, release_ms=100.0,
                         gate_mode=0, vad_threshold=0.5, vad_hold_time_ms=200.0),
        eq=EQSettings(
            enabled=True,
            band_gains=[-3.0, -2.0, 0.0, 1.0, 2.0, 3.0, 2.0, 0.0, -1.0, -2.0],
            band_qs=[0.7, 1.0, 1.2, 1.4, 1.6, 2.0, 1.8, 1.2, 0.9, 0.7]  # Wide cuts, focused boosts
        ),
        rnnoise=RNNoiseSettings(enabled=True, strength=1.0, model='rnnoise'),
    ),
    'bass_cut': Preset(
        name="Bass Cut",
        description="High-pass effect to remove low frequency rumble and proximity effect",
        version="1.2.0",
        gate=GateSettings(enabled=True, threshold_db=-40.0, attack_ms=10.0, release_ms=100.0,
                         gate_mode=0, vad_threshold=0.5, vad_hold_time_ms=200.0),
        eq=EQSettings(
            enabled=True,
            band_gains=[-12.0, -6.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            band_qs=[0.5, 0.7, 0.9, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41, 1.41]  # Wide rolloff on low end
        ),
        rnnoise=RNNoiseSettings(enabled=True, strength=1.0, model='rnnoise'),
    ),
    'presence': Preset(
        name="Presence Boost",
        description="Enhances voice presence and intelligibility",
        version="1.2.0",
        gate=GateSettings(enabled=True, threshold_db=-40.0, attack_ms=10.0, release_ms=100.0,
                         gate_mode=0, vad_threshold=0.5, vad_hold_time_ms=200.0),
        eq=EQSettings(
            enabled=True,
            band_gains=[0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 3.0, 1.0, 0.0, 0.0],
            band_qs=[1.41, 1.41, 1.41, 1.41, 2.0, 2.5, 2.0, 1.5, 1.41, 1.41]  # Narrow focus on presence frequencies
        ),
        rnnoise=RNNoiseSettings(enabled=True, strength=1.0, model='rnnoise'),
    ),
    'flat': Preset(
        name="Flat",
        description="No EQ processing - flat frequency response",
        version="1.2.0",
        gate=GateSettings(enabled=True, threshold_db=-40.0, attack_ms=10.0, release_ms=100.0,
                         gate_mode=0, vad_threshold=0.5, vad_hold_time_ms=200.0),
        eq=EQSettings(
            enabled=True,
            band_gains=[0.0] * 10,
            band_qs=[1.41] * 10  # Default Q
        ),
        rnnoise=RNNoiseSettings(enabled=True, strength=1.0, model='rnnoise'),
    ),
    'minimal': Preset(
        name="Minimal Processing",
        description="Gate and RNNoise only - no EQ",
        version="1.2.0",
        gate=GateSettings(enabled=True, threshold_db=-45.0, attack_ms=5.0, release_ms=150.0,
                         gate_mode=0, vad_threshold=0.5, vad_hold_time_ms=200.0),
        eq=EQSettings(
            enabled=False,
            band_gains=[0.0] * 10,
            band_qs=[1.41] * 10  # Default Q
        ),
        rnnoise=RNNoiseSettings(enabled=True, strength=1.0, model='rnnoise'),
    ),
    'aggressive_denoise': Preset(
        name="Aggressive Denoise",
        description="Maximum noise reduction with tight gate",
        version="1.2.0",
        gate=GateSettings(enabled=True, threshold_db=-35.0, attack_ms=5.0, release_ms=50.0,
                         gate_mode=0, vad_threshold=0.5, vad_hold_time_ms=200.0),
        eq=EQSettings(
            enabled=True,
            band_gains=[-6.0, -3.0, 0.0, 0.0, 1.0, 2.0, 1.0, -1.0, -3.0, -6.0],
            band_qs=[0.6, 0.8, 1.2, 1.4, 1.8, 2.0, 1.6, 1.2, 0.8, 0.6]  # Varied Q for targeted corrections
        ),
        rnnoise=RNNoiseSettings(enabled=True, strength=1.0, model='rnnoise'),
    ),
}


# Test functions for VAD preset persistence
def _test_vad_preset_persistence():
    """Test VAD settings persist across preset save/load."""
    from pathlib import Path
    import tempfile

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_vad_preset.json"

        # Create preset with VAD settings
        original = Preset(
            name="VAD Test",
            gate=GateSettings(
                enabled=True,
                threshold_db=-35.0,
                attack_ms=5.0,
                release_ms=50.0,
                gate_mode=1,  # VAD Assisted
                vad_threshold=0.6,
                vad_hold_time_ms=150.0
            )
        )

        # Save preset
        save_preset(original, test_file)

        # Load preset
        loaded = load_preset(test_file)

        # Verify all fields match
        assert loaded.gate.gate_mode == 1, f"gate_mode mismatch: {loaded.gate.gate_mode}"
        assert loaded.gate.vad_threshold == 0.6, f"vad_threshold mismatch: {loaded.gate.vad_threshold}"
        assert loaded.gate.vad_hold_time_ms == 150.0, f"vad_hold_time_ms mismatch: {loaded.gate.vad_hold_time_ms}"

        print("PASS: VAD preset persistence test passed")


def _test_backward_compatibility():
    """Test old presets load with default VAD values."""
    old_preset_data = {
        'name': 'Old Preset',
        'version': '1.1.0',
        'gate': {
            'enabled': True,
            'threshold_db': -40.0,
            'attack_ms': 10.0,
            'release_ms': 100.0,
            # No gate_mode, vad_threshold, vad_hold_time_ms
        }
    }

    loaded = Preset.from_dict(old_preset_data)

    # Verify defaults applied
    assert loaded.gate.gate_mode == 0, "Default gate_mode should be 0"
    assert loaded.gate.vad_threshold == 0.5, "Default vad_threshold should be 0.5"
    assert loaded.gate.vad_hold_time_ms == 200.0, "Default vad_hold_time_ms should be 200.0"

    print("PASS: Backward compatibility test passed")


if __name__ == "__main__":
    _test_vad_preset_persistence()
    _test_backward_compatibility()
