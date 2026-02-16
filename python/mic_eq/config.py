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


def _version_tuple(version: str) -> tuple[int, int, int]:
    """Convert semantic version string to tuple for safe ordering."""
    parts = str(version).split(".")
    normalized = []
    for i in range(3):
        try:
            normalized.append(int(parts[i]))
        except (IndexError, ValueError):
            normalized.append(0)
    return tuple(normalized)  # type: ignore[return-value]


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


def get_preset_imports_dir() -> Path:
    """Get the preset imports directory, creating it if necessary."""
    imports_dir = get_presets_dir().parent / 'imports'
    imports_dir.mkdir(parents=True, exist_ok=True)
    return imports_dir


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
    vad_threshold: float = 0.4  # Speech probability threshold (0.3-0.7) - lowered from 0.5 for better soft speech detection
    vad_hold_time_ms: float = 200.0  # Hold time in milliseconds (0-500)
    vad_pre_gain: float = 1.0  # Pre-gain to boost weak signals for VAD (1.0-10.0)
    # Auto-threshold settings (v1.6.0+)
    auto_threshold_enabled: bool = False  # Auto-adjust threshold based on noise floor
    gate_margin_db: float = 10.0  # Margin above noise floor for auto-threshold (0-20 dB) - increased from 6.0 for better noise rejection


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


# EQ band frequencies (Hz) - matches 10-band EQ in DSP chain
# Bands: 80Hz (LS), 160, 320, 640, 1.28k, 2.5k, 5k, 8k, 12k, 16kHz (HS)
EQ_FREQUENCIES = [80.0, 160.0, 320.0, 640.0, 1280.0, 2500.0, 5000.0, 8000.0, 12000.0, 16000.0]

# Default Q factor for auto-EQ bands (1/3 octave bandwidth)
# Q = 4.33 gives ~1/3 octave bandwidth for parametric EQ
AUTO_EQ_DEFAULT_Q = 4.33

# Auto-EQ analysis validation thresholds
ANALYSIS_MIN_PEAK_COUNT = 3      # Minimum peaks to detect voice
ANALYSIS_MIN_DYNAMIC_RANGE = 20  # Minimum dB range (peak - floor)
ANALYSIS_MIN_SNR = 12            # Minimum signal-to-noise ratio (dB) - lowered for real-world recordings
ANALYSIS_MAX_SPECTRAL_FLATNESS = 0.8  # Maximum flatness (1.0 = white noise)


@dataclass
class RNNoiseSettings:
    """Noise suppression settings (RNNoise or DeepFilterNet)."""
    enabled: bool = True
    strength: float = 1.0  # 0.0 = dry, 1.0 = fully processed
    model: str = "rnnoise"  # "rnnoise", "deepfilter-ll", or "deepfilter"


@dataclass
class DeEsserSettings:
    """De-esser settings."""
    enabled: bool = False
    auto_enabled: bool = True
    auto_amount: float = 0.5
    low_cut_hz: float = 4000.0
    high_cut_hz: float = 9000.0
    threshold_db: float = -28.0
    ratio: float = 4.0
    attack_ms: float = 2.0
    release_ms: float = 80.0
    max_reduction_db: float = 6.0


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
        'vad_threshold': (0.3, 0.7),  # Speech probability (lowered min for better soft speech)
        'vad_hold_time_ms': (0.0, 500.0),  # Milliseconds
        'vad_pre_gain': (1.0, 10.0),  # Pre-gain multiplier
        'auto_threshold_enabled': (bool, None),  # Boolean flag
        'gate_margin_db': (0.0, 20.0),  # Margin in dB
    },
    'eq': {
        'band_gain': (-12.0, 12.0),
        'band_q': (0.1, 10.0),
    },
    'rnnoise': {
        'strength': (0.0, 1.0),  # 0% to 100%
        'model': ['rnnoise', 'deepfilter-ll', 'deepfilter'],  # Valid model choices
    },
    'deesser': {
        'auto_amount': (0.0, 1.0),
        'low_cut_hz': (2000.0, 12000.0),
        'high_cut_hz': (2200.0, 16000.0),
        'threshold_db': (-60.0, -6.0),
        'ratio': (1.0, 20.0),
        'attack_ms': (0.1, 50.0),
        'release_ms': (5.0, 500.0),
        'max_reduction_db': (0.0, 24.0),
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
    version: str = "1.7.0"  # Version field for migration (includes de-esser settings)
    gate: GateSettings = field(default_factory=GateSettings)
    eq: EQSettings = field(default_factory=EQSettings)
    rnnoise: RNNoiseSettings = field(default_factory=RNNoiseSettings)
    deesser: DeEsserSettings = field(default_factory=DeEsserSettings)
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
            'deesser': asdict(self.deesser),
            'compressor': asdict(self.compressor),
            'limiter': asdict(self.limiter),
            'bypass': self.bypass,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Preset':
        """Create preset from dictionary with validation."""
        try:
            # Version-aware migration
            version_tuple = _version_tuple(data.get('version', '1.0.0'))

            # Migrate v1.0 presets -> v1.1
            if version_tuple < _version_tuple('1.1.0'):
                if 'rnnoise' in data:
                    data['rnnoise'].setdefault('strength', 1.0)
                else:
                    data['rnnoise'] = {'enabled': True, 'strength': 1.0}
                data['version'] = '1.1.0'
                version_tuple = _version_tuple('1.1.0')

            # Migrate v1.1 presets -> v1.2
            if version_tuple < _version_tuple('1.2.0'):
                if 'rnnoise' in data:
                    data['rnnoise'].setdefault('model', 'rnnoise')
                else:
                    data['rnnoise'] = {'enabled': True, 'strength': 1.0, 'model': 'rnnoise'}
                data['version'] = '1.2.0'
                version_tuple = _version_tuple('1.2.0')

            # Migrate v1.2 presets -> v1.3
            if version_tuple < _version_tuple('1.3.0'):
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
                data['version'] = '1.3.0'
                version_tuple = _version_tuple('1.3.0')

            # Migrate v1.3 presets -> v1.4 (no format changes)
            if version_tuple < _version_tuple('1.4.0'):
                data['version'] = '1.4.0'
                version_tuple = _version_tuple('1.4.0')

            # Migrate v1.4 presets -> v1.5 (no format changes)
            if version_tuple < _version_tuple('1.5.0'):
                data['version'] = '1.5.0'
                version_tuple = _version_tuple('1.5.0')

            # Migrate v1.5 presets -> v1.6 (auto-threshold defaults)
            if version_tuple < _version_tuple('1.6.0'):
                if 'gate' in data:
                    data['gate'].setdefault('auto_threshold_enabled', False)
                    data['gate'].setdefault('gate_margin_db', 10.0)
                    if data['gate'].get('vad_threshold', 0.5) == 0.5:
                        data['gate']['vad_threshold'] = 0.4
                else:
                    data['gate'] = {
                        'auto_threshold_enabled': False,
                        'gate_margin_db': 10.0,
                        'vad_threshold': 0.4,
                    }
                data['version'] = '1.6.0'
                version_tuple = _version_tuple('1.6.0')

            # Migrate v1.6 presets -> v1.7 (de-esser defaults)
            if version_tuple < _version_tuple('1.7.0'):
                if 'deesser' in data:
                    data['deesser'].setdefault('enabled', False)
                    data['deesser'].setdefault('auto_enabled', True)
                    data['deesser'].setdefault('auto_amount', 0.5)
                    data['deesser'].setdefault('low_cut_hz', 4000.0)
                    data['deesser'].setdefault('high_cut_hz', 9000.0)
                    data['deesser'].setdefault('threshold_db', -28.0)
                    data['deesser'].setdefault('ratio', 4.0)
                    data['deesser'].setdefault('attack_ms', 2.0)
                    data['deesser'].setdefault('release_ms', 80.0)
                    data['deesser'].setdefault('max_reduction_db', 6.0)
                else:
                    data['deesser'] = asdict(DeEsserSettings())
                data['version'] = '1.7.0'
                version_tuple = _version_tuple('1.7.0')

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
                gate_mode=int(_validate_range(
                    gate_data.get('gate_mode', 0),
                    *gate_ranges['gate_mode'],
                    'gate_mode', 'gate'
                )),
                vad_threshold=_validate_range(
                    gate_data.get('vad_threshold', 0.4),
                    *gate_ranges['vad_threshold'],
                    'vad_threshold', 'gate'
                ),
                vad_hold_time_ms=_validate_range(
                    gate_data.get('vad_hold_time_ms', 200.0),
                    *gate_ranges['vad_hold_time_ms'],
                    'vad_hold_time_ms', 'gate'
                ),
                vad_pre_gain=_validate_range(
                    gate_data.get('vad_pre_gain', 1.0),
                    *gate_ranges['vad_pre_gain'],
                    'vad_pre_gain', 'gate'
                ),
                auto_threshold_enabled=gate_data.get('auto_threshold_enabled', False),
                gate_margin_db=_validate_range(
                    gate_data.get('gate_margin_db', 10.0),
                    *gate_ranges['gate_margin_db'],
                    'gate_margin_db', 'gate'
                ),
            )

            # Extract and validate EQ settings
            eq_data = data.get('eq', {})
            band_gains = eq_data.get('band_gains', [0.0] * 10)
            band_qs = eq_data.get('band_qs', [1.41] * 10)

            if not isinstance(band_gains, (list, tuple)):
                raise PresetValidationError("Invalid band_gains in eq: expected list of 10 values")
            if not isinstance(band_qs, (list, tuple)):
                raise PresetValidationError("Invalid band_qs in eq: expected list of 10 values")
            if len(band_gains) != 10:
                raise PresetValidationError(
                    f"Invalid band_gains in eq: expected 10 values, got {len(band_gains)}"
                )
            if len(band_qs) != 10:
                raise PresetValidationError(
                    f"Invalid band_qs in eq: expected 10 values, got {len(band_qs)}"
                )

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
            model = rnnoise_data.get('model', 'rnnoise')
            valid_models = rnnoise_ranges.get('model', ['rnnoise', 'deepfilter-ll', 'deepfilter'])
            if model not in valid_models:
                model = 'rnnoise'

            validated_rnnoise = RNNoiseSettings(
                enabled=rnnoise_data.get('enabled', True),
                strength=_validate_range(
                    rnnoise_data.get('strength', 1.0),
                    *rnnoise_ranges.get('strength', (0.0, 1.0)),
                    'strength', 'rnnoise'
                ),
                model=model,
            )

            # Extract and validate de-esser settings
            deesser_data = data.get('deesser', {})
            deesser_ranges = VALIDATION_RANGES['deesser']
            low_cut_hz = _validate_range(
                deesser_data.get('low_cut_hz', 4000.0),
                *deesser_ranges['low_cut_hz'],
                'low_cut_hz', 'deesser'
            )
            high_cut_hz = _validate_range(
                deesser_data.get('high_cut_hz', 9000.0),
                *deesser_ranges['high_cut_hz'],
                'high_cut_hz', 'deesser'
            )
            if high_cut_hz <= low_cut_hz + 200.0:
                high_cut_hz = min(16000.0, low_cut_hz + 200.0)
                low_cut_hz = min(low_cut_hz, high_cut_hz - 200.0)

            validated_deesser = DeEsserSettings(
                enabled=deesser_data.get('enabled', False),
                auto_enabled=bool(deesser_data.get('auto_enabled', True)),
                auto_amount=_validate_range(
                    float(deesser_data.get('auto_amount', 0.5)),
                    *deesser_ranges['auto_amount'],
                    'auto_amount', 'deesser'
                ),
                low_cut_hz=low_cut_hz,
                high_cut_hz=high_cut_hz,
                threshold_db=_validate_range(
                    deesser_data.get('threshold_db', -28.0),
                    *deesser_ranges['threshold_db'],
                    'threshold_db', 'deesser'
                ),
                ratio=_validate_range(
                    deesser_data.get('ratio', 4.0),
                    *deesser_ranges['ratio'],
                    'ratio', 'deesser'
                ),
                attack_ms=_validate_range(
                    deesser_data.get('attack_ms', 2.0),
                    *deesser_ranges['attack_ms'],
                    'attack_ms', 'deesser'
                ),
                release_ms=_validate_range(
                    deesser_data.get('release_ms', 80.0),
                    *deesser_ranges['release_ms'],
                    'release_ms', 'deesser'
                ),
                max_reduction_db=_validate_range(
                    deesser_data.get('max_reduction_db', 6.0),
                    *deesser_ranges['max_reduction_db'],
                    'max_reduction_db', 'deesser'
                ),
            )

            return cls(
                name=data.get('name', 'Unnamed'),
                description=data.get('description', ''),
                version=data.get('version', '1.7.0'),
                gate=validated_gate,
                eq=validated_eq,
                rnnoise=validated_rnnoise,
                deesser=validated_deesser,
                compressor=validated_comp,
                limiter=validated_lim,
                bypass=data.get('bypass', False),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as e:
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
    requested_path = Path(filepath)

    if requested_path.suffix.lower() != '.json':
        raise PresetValidationError(
            f"Invalid preset file: '{requested_path.name}' - must be a .json file"
        )

    if not requested_path.exists():
        raise PresetValidationError(f"Preset file not found: '{requested_path.name}'")

    try:
        resolved_path = requested_path.resolve(strict=True)
    except OSError as e:
        raise PresetValidationError(
            f"Invalid preset path: '{requested_path.name}' - {e}"
        )

    if not resolved_path.is_file():
        raise PresetValidationError(
            f"Invalid preset path: '{requested_path.name}' - not a file"
        )

    allowed_roots = [
        get_presets_dir().resolve(),
        get_preset_imports_dir().resolve(),
    ]
    within_allowed_root = any(
        root == resolved_path or root in resolved_path.parents
        for root in allowed_roots
    )
    if not within_allowed_root:
        allowed_display = ", ".join(str(root) for root in allowed_roots)
        raise PresetValidationError(
            f"Invalid preset path: '{requested_path.name}' - "
            f"path must be inside allowed preset roots: {allowed_display}"
        )

    with open(resolved_path, 'r', encoding='utf-8') as f:
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
        except (json.JSONDecodeError, KeyError, PresetValidationError, TypeError, ValueError):
            # Skip invalid files
            continue

    return sorted(presets, key=lambda x: x[0].lower())


def generate_auto_eq_preset_name(target_curve: str) -> str:
    """
    Generate auto-EQ preset name based on target curve.

    Args:
        target_curve: One of 'broadcast', 'podcast', 'streaming', 'flat'

    Returns:
        Preset name like "Auto-EQ Broadcast"
    """
    curve_display_names = {
        'broadcast': 'Broadcast',
        'podcast': 'Podcast',
        'streaming': 'Streaming',
        'flat': 'Flat',
    }
    curve_name = curve_display_names.get(target_curve.lower(), target_curve.title())
    return f"Auto-EQ {curve_name}"


@dataclass
class LatencyCalibrationProfile:
    """Measured latency calibration result for one input/output pair."""
    measured_round_trip_ms: float
    estimated_one_way_ms: float
    applied_compensation_ms: float
    confidence: float
    sample_rate: int = 48000
    timestamp_utc: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'LatencyCalibrationProfile':
        return cls(
            measured_round_trip_ms=float(data.get('measured_round_trip_ms', 0.0)),
            estimated_one_way_ms=float(data.get('estimated_one_way_ms', 0.0)),
            applied_compensation_ms=float(data.get('applied_compensation_ms', 0.0)),
            confidence=float(data.get('confidence', 0.0)),
            sample_rate=int(data.get('sample_rate', 48000)),
            timestamp_utc=str(data.get('timestamp_utc', '')),
        )


@dataclass
class AppConfig:
    """Application configuration (persisted settings)."""
    last_input_device: str = ""
    last_output_device: str = ""
    last_preset: str = ""
    startup_preset: str = ""  # Preset to load on startup (empty = last used)
    window_geometry: Optional[dict] = None
    use_measured_latency: bool = True
    latency_calibration_profiles: dict[str, LatencyCalibrationProfile] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'last_input_device': self.last_input_device,
            'last_output_device': self.last_output_device,
            'last_preset': self.last_preset,
            'startup_preset': self.startup_preset,
            'window_geometry': self.window_geometry,
            'use_measured_latency': self.use_measured_latency,
            'latency_calibration_profiles': {
                key: profile.to_dict()
                for key, profile in self.latency_calibration_profiles.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AppConfig':
        """Create config from dictionary."""
        raw_profiles = data.get('latency_calibration_profiles', {}) or {}
        parsed_profiles: dict[str, LatencyCalibrationProfile] = {}
        if isinstance(raw_profiles, dict):
            for key, value in raw_profiles.items():
                if isinstance(value, dict):
                    try:
                        parsed_profiles[str(key)] = LatencyCalibrationProfile.from_dict(value)
                    except (TypeError, ValueError):
                        continue

        return cls(
            last_input_device=data.get('last_input_device', ''),
            last_output_device=data.get('last_output_device', ''),
            last_preset=data.get('last_preset', ''),
            startup_preset=data.get('startup_preset', ''),
            window_geometry=data.get('window_geometry'),
            use_measured_latency=bool(data.get('use_measured_latency', True)),
            latency_calibration_profiles=parsed_profiles,
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
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return AppConfig()


# Built-in presets
BUILTIN_PRESETS = {
    'voice': Preset(
        name="Voice Clarity",
        description="Optimized for voice communication - cuts low end rumble and boosts presence",
        version="1.7.0",
        gate=GateSettings(enabled=True, threshold_db=-40.0, attack_ms=10.0, release_ms=100.0,
                         gate_mode=0, vad_threshold=0.4, vad_hold_time_ms=200.0, vad_pre_gain=1.0,
                         auto_threshold_enabled=False, gate_margin_db=10.0),
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
        version="1.7.0",
        gate=GateSettings(enabled=True, threshold_db=-40.0, attack_ms=10.0, release_ms=100.0,
                         gate_mode=0, vad_threshold=0.4, vad_hold_time_ms=200.0, vad_pre_gain=1.0,
                         auto_threshold_enabled=False, gate_margin_db=10.0),
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
        version="1.7.0",
        gate=GateSettings(enabled=True, threshold_db=-40.0, attack_ms=10.0, release_ms=100.0,
                         gate_mode=0, vad_threshold=0.4, vad_hold_time_ms=200.0, vad_pre_gain=1.0,
                         auto_threshold_enabled=False, gate_margin_db=10.0),
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
        version="1.7.0",
        gate=GateSettings(enabled=True, threshold_db=-40.0, attack_ms=10.0, release_ms=100.0,
                         gate_mode=0, vad_threshold=0.4, vad_hold_time_ms=200.0, vad_pre_gain=1.0,
                         auto_threshold_enabled=False, gate_margin_db=10.0),
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
        version="1.7.0",
        gate=GateSettings(enabled=True, threshold_db=-45.0, attack_ms=5.0, release_ms=150.0,
                         gate_mode=0, vad_threshold=0.4, vad_hold_time_ms=200.0, vad_pre_gain=1.0,
                         auto_threshold_enabled=False, gate_margin_db=10.0),
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
        version="1.7.0",
        gate=GateSettings(enabled=True, threshold_db=-35.0, attack_ms=5.0, release_ms=50.0,
                         gate_mode=0, vad_threshold=0.4, vad_hold_time_ms=200.0, vad_pre_gain=1.0,
                         auto_threshold_enabled=False, gate_margin_db=10.0),
        eq=EQSettings(
            enabled=True,
            band_gains=[-6.0, -3.0, 0.0, 0.0, 1.0, 2.0, 1.0, -1.0, -3.0, -6.0],
            band_qs=[0.6, 0.8, 1.2, 1.4, 1.8, 2.0, 1.6, 1.2, 0.8, 0.6]  # Varied Q for targeted corrections
        ),
        rnnoise=RNNoiseSettings(enabled=True, strength=1.0, model='rnnoise'),
    ),
}


# Target curves for Auto-EQ calibration
TARGET_CURVES = {
    'broadcast': TargetCurve(
        name="Broadcast Standard",
        description="ITU-R BS.1770 compliant - professional broadcast voice",
        band_targets=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0]
    ),
    'podcast': TargetCurve(
        name="Podcast / Voice-Over",
        description="Enhanced presence for intimate vocal recording",
        band_targets=[0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    ),
    'streaming': TargetCurve(
        name="Streaming / Gaming",
        description="Cuts through game audio mix with aggressive presence",
        band_targets=[-1.0, 0.0, 1.0, 2.0, 4.0, 5.0, 4.0, 2.0, 0.0, -2.0]
    ),
    'flat': TargetCurve(
        name="Flat Response",
        description="No frequency correction - measure mic as-is",
        band_targets=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
            version="1.7.0",
            gate=GateSettings(
                enabled=True,
                threshold_db=-35.0,
                attack_ms=5.0,
                release_ms=50.0,
                gate_mode=1,  # VAD Assisted
                vad_threshold=0.4,  # Updated to 0.4
                vad_hold_time_ms=150.0,
                vad_pre_gain=2.5,  # Test with non-default gain
                auto_threshold_enabled=True,  # Test auto-threshold
                gate_margin_db=8.0  # Test custom margin
            )
        )

        # Save preset
        save_preset(original, test_file)

        # Load preset
        loaded = load_preset(test_file)

        # Verify all fields match
        assert loaded.gate.gate_mode == 1, f"gate_mode mismatch: {loaded.gate.gate_mode}"
        assert loaded.gate.vad_threshold == 0.4, f"vad_threshold mismatch: {loaded.gate.vad_threshold}"
        assert loaded.gate.vad_hold_time_ms == 150.0, f"vad_hold_time_ms mismatch: {loaded.gate.vad_hold_time_ms}"
        assert loaded.gate.vad_pre_gain == 2.5, f"vad_pre_gain mismatch: {loaded.gate.vad_pre_gain}"
        assert loaded.gate.auto_threshold_enabled == True, f"auto_threshold_enabled mismatch: {loaded.gate.auto_threshold_enabled}"
        assert loaded.gate.gate_margin_db == 8.0, f"gate_margin_db mismatch: {loaded.gate.gate_margin_db}"

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
            # No gate_mode, vad_threshold, vad_hold_time_ms, vad_pre_gain, auto_threshold_enabled, gate_margin_db
        }
    }

    loaded = Preset.from_dict(old_preset_data)

    # Verify defaults applied (after migration to v1.6.0, vad_threshold becomes 0.4)
    assert loaded.gate.gate_mode == 0, "Default gate_mode should be 0"
    assert loaded.gate.vad_threshold == 0.4, "Default vad_threshold should be 0.4 (migrated from 0.5)"
    assert loaded.gate.vad_hold_time_ms == 200.0, "Default vad_hold_time_ms should be 200.0"
    assert loaded.gate.vad_pre_gain == 1.0, "Default vad_pre_gain should be 1.0"
    assert loaded.gate.auto_threshold_enabled == False, "Default auto_threshold_enabled should be False"
    assert loaded.gate.gate_margin_db == 10.0, "Default gate_margin_db should be 10.0"

    print("PASS: Backward compatibility test passed")


if __name__ == "__main__":
    _test_vad_preset_persistence()
    _test_backward_compatibility()
