"""The native headroom helpers against the Python code they replaced, bit for bit.

headroom_reference.py is the verbatim Python copy. Floats are compared by bit pattern and
containers by type and key order, so a -0.0, a float where an int belongs or a reordered
key counts as a difference. A failure has to match in exception type and message too.
"""

import copy
import itertools
import math
import struct
import types
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import headroom_reference as reference
from mic_eq import mic_eq_core as native
from mic_eq.analysis import voice_setup
from mic_eq.config import Preset

NAN = float("nan")
INF = float("inf")


class _Opaque:
    def __repr__(self) -> str:
        return "<opaque>"


class _FloatLike:
    def __repr__(self) -> str:
        return "<float-like>"

    def __float__(self) -> float:
        return 2.5

    def __int__(self) -> int:
        return 7

    def __bool__(self) -> bool:
        return True


def _same(a: Any, b: Any, path: str = "value") -> None:
    assert type(a) is type(b), f"{path}: {type(a).__name__} != {type(b).__name__} ({a!r} vs {b!r})"
    if isinstance(a, float):
        assert struct.pack("<d", a) == struct.pack("<d", b), f"{path}: {a!r} != {b!r}"
    elif isinstance(a, dict):
        assert list(a) == list(b), f"{path}: keys {list(a)} != {list(b)}"
        for key in a:
            _same(a[key], b[key], f"{path}[{key!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for index, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{index}]")
    elif isinstance(a, np.generic):
        assert a.tobytes() == b.tobytes(), f"{path}: {a!r} != {b!r}"
    elif isinstance(a, (_Opaque, _FloatLike)):
        pass  # a deep copy of an object without equality; the type was compared above
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def _outcome(call: Callable[[], Any]) -> tuple[str, Any]:
    try:
        return "ok", call()
    except Exception as error:  # noqa: BLE001 - the error is the observation
        return "error", (type(error), str(error))


def _agree(old: Callable[[], Any], new: Callable[[], Any]) -> None:
    expected, actual = _outcome(old), _outcome(new)
    assert expected[0] == actual[0], f"{expected} != {actual}"
    if expected[0] == "ok":
        _same(expected[1], actual[1])
    else:
        assert expected[1] == actual[1]


# A settings value of each kind a section may hold.
VALUES = [
    None, True, False, 0, 1, -3, 2**70, 10**400, 0.0, -0.0, 1.5, -2.25, 1e308, NAN, INF, -INF,
    "1.5", "x", "", "nan", "7", b"3", [], [1], (2,), {}, _Opaque(), _FloatLike(),
    np.float32(0.1), np.float64(-0.0), np.int64(4), np.bool_(True), np.bool_(False),
    np.float32("nan"), np.array([1.0]),
]


def _id(value: Any) -> str:
    return f"{type(value).__name__}:{value!r}"[:40]


@pytest.mark.parametrize("value", VALUES, ids=_id)
@pytest.mark.parametrize(
    "path",
    [
        ("return_output_alias",), ("deesser", "enabled"), ("deesser", "auto_amount"),
        ("deesser", "high_cut_hz"), ("compressor", "enabled"), ("compressor", "threshold_db"),
        ("compressor", "noise_reference_reliability"), ("limiter", "enabled"),
        ("limiter", "ceiling_db"), ("gate", "enabled"), ("gate", "gate_mode"),
        ("gate", "threshold_db"), ("gate", "vad_pre_gain"), ("gate_settings", "gate_mode"),
        ("suppressor", "enabled"), ("suppressor", "strength"), ("suppressor", "model"),
        ("rnnoise", "model"), ("suppression", "strength"), ("suppressor_settings", "enabled"),
    ],
)
def test_one_nested_value_is_read_like_the_python_code(path, value):
    chain: dict[str, Any] = {path[0]: {path[1]: value}} if len(path) == 2 else {"return_output_audio": value}
    _agree(
        lambda: reference._flatten_chain_settings(copy.deepcopy(chain)),
        lambda: native.flatten_chain_settings(copy.deepcopy(chain)),
    )


@pytest.mark.parametrize("value", VALUES, ids=_id)
@pytest.mark.parametrize(
    "key",
    [
        "full_chain", "input_pre_filtered", "input_cleanup_mode", "processing_mode",
        "vad_probabilities", "vad_available", "auto_makeup_activity",
        "return_auto_makeup_activity", "limiter_enabled", "compressor_ratio", "noise_model",
        "gate_mode", "return_output_audio", "eq_enabled", "eq_bands", "unknown_key",
    ],
)
def test_one_native_boundary_key_overrides_or_is_dropped_like_the_python_code(key, value):
    chain = {key: value, "limiter": {"enabled": False}}
    _agree(
        lambda: reference._flatten_chain_settings(copy.deepcopy(chain)),
        lambda: native.flatten_chain_settings(copy.deepcopy(chain)),
    )


def _section_cases() -> list[Any]:
    section: dict[str, Any] = {"enabled": False, "ratio": 2}
    return [
        None, {}, 0, "", [], False, section, [1], "text", 5, _Opaque(),
        types.MappingProxyType(section), np.array([1]),
    ]


@pytest.mark.parametrize("name", [
    "deesser", "compressor", "limiter", "gate", "gate_settings", "suppressor", "suppression",
    "rnnoise", "suppressor_settings",
])
@pytest.mark.parametrize("section", _section_cases(), ids=_id)
def test_a_section_of_any_type_is_handled_like_the_python_code(name, section):
    chain = {name: section}
    _agree(
        lambda: reference._flatten_chain_settings(chain),
        lambda: native.flatten_chain_settings(chain),
    )


@pytest.mark.parametrize("chain", [
    None, {}, [], 0, "text", [("gate", {})], 5, types.MappingProxyType({"limiter": {"enabled": False}}),
    {"gate": {}, "gate_settings": {"gate_mode": 2}},
    {"gate": 0, "gate_settings": {"gate_mode": 2}},
    {"suppressor": {}, "suppression": {"strength": 0.5}, "rnnoise": {"strength": 0.25}},
    {"suppressor": 0, "suppression": 0, "rnnoise": {"model": "deepfilter"}},
    {"deesser": {"auto_amount": "x"}, "limiter": {"ceiling_db": NAN}},
    {"limiter": {"ceiling_db": NAN}, "deesser": {"auto_amount": "x"}},
    {"compressor": {"enabled": "false"}, "gate": {"gate_mode": "x"}},
    {"vad_probabilities": [0.5, 0.25], "full_chain": True, "input_pre_filtered": False},
    {"gate_mode": 2, "gate": {"gate_mode": 1}, "limiter_enabled": False, "limiter": {"enabled": True}},
])
def test_whole_chains_are_handled_like_the_python_code(chain):
    _agree(
        lambda: reference._flatten_chain_settings(copy.deepcopy(chain)),
        lambda: native.flatten_chain_settings(copy.deepcopy(chain)),
    )


def test_the_shipped_preset_flattens_like_the_python_code():
    chain = {
        key: value
        for key, value in Preset().to_dict().items()
        if key in ("deesser", "compressor", "limiter", "gate", "suppressor")
    }
    chain.update(full_chain=True, input_pre_filtered=False, processing_mode="normal")
    _agree(
        lambda: reference._flatten_chain_settings(copy.deepcopy(chain)),
        lambda: native.flatten_chain_settings(copy.deepcopy(chain)),
    )


def test_an_override_is_a_copy_not_the_callers_object():
    probabilities = [0.5, 0.25]
    flat = native.flatten_chain_settings({"vad_probabilities": probabilities})
    assert flat["vad_probabilities"] == probabilities
    assert flat["vad_probabilities"] is not probabilities


def test_the_flat_dict_has_the_documented_defaults_in_order():
    _same(
        native.flatten_chain_settings(None),
        reference._flatten_chain_settings(None),
    )
    assert next(iter(native.flatten_chain_settings(None))) == "return_output_audio"


# The EQ half of simulate_candidate_chain's legacy branch.
def _legacy_eq_cases() -> list[dict[str, Any]]:
    freqs = [80.0, 160.0, 320.0, 640.0, 1280.0, 2500.0, 5000.0, 8000.0, 12000.0, 16000.0]
    base = {"band_freqs": freqs, "band_gains": [1.0] * 10, "band_qs": [1.41] * 10}
    cases: list[dict[str, Any]] = [
        base, {}, {**base, "enabled": False}, {**base, "eq_enabled": False},
        {**base, "enabled": None, "eq_enabled": False}, {**base, "enabled": "no"},
        {**base, "enabled": np.bool_(False)}, {**base, "enabled": 1},
        {"band_freqs": freqs[:9], "band_gains": [0.0] * 10, "band_qs": [1.0] * 10},
        {"band_freqs": None, "band_gains": None, "band_qs": None},
        {"band_freqs": tuple(freqs), "band_gains": np.zeros(10), "band_qs": [2] * 10},
        {"band_freqs": freqs, "band_gains": [0] * 10, "band_qs": ["1.5"] * 10},
    ]
    for field, bad in itertools.product(("band_freqs", "band_gains", "band_qs"), VALUES):
        cases.append({**base, field: [bad, *base[field][1:]]})
    return cases


@pytest.mark.parametrize("eq_settings", _legacy_eq_cases(), ids=lambda case: _id(case)[:60])
def test_legacy_eq_settings_are_read_like_the_python_code(eq_settings):
    _agree(
        lambda: reference._legacy_eq_inputs(copy.deepcopy(eq_settings)),
        lambda: native.legacy_eq_inputs(copy.deepcopy(eq_settings)),
    )


def _readings() -> list[Any]:
    names = reference._NATIVE_SAFETY_METRICS
    good = [0.0, 1.0, 2.5, -0.5, 1e-300, -1e300, 1e308, -0.0, 0.5, 3, np.float32(1.25), np.int64(2), np.float16(0.5)]
    bad = [None, True, "1.0", NAN, INF, -INF, [], np.bool_(True), 10**400, 2**70, np.float32("nan"), _Opaque()]
    cases: list[Any] = [{}, None, [], {"unrelated": 1.0}]
    for pre, lim, tp in itertools.product(good, [0.0, 0.5, 1.0, 1.0000000000000002, 2.0], [0.0, 0.25, 0.5, 0.6]):
        cases.append(dict(zip(names, (pre, lim, tp))))
    base = dict(zip(names, (2.0, 0.0, 0.0)))
    for name, value in itertools.product(names, bad):
        cases.append({**base, name: value})
    for name in names:
        missing = dict(base)
        del missing[name]
        cases.append(missing)
    cases.append({**{name: INF for name in names}})
    cases.append({names[0]: NAN, names[1]: 10**400, names[2]: 0.0})
    cases.append({names[0]: 2.0, names[1]: INF, names[2]: 10**400})
    return cases


@pytest.mark.parametrize("batch", range(4))
def test_the_headroom_verdict_is_the_python_codes(batch):
    cases = _readings()[batch::4]
    for reading in cases:
        _agree(
            lambda: reference._headroom_excess_db(reading),
            lambda: native.headroom_excess_db(reading),
        )
        _agree(
            lambda: reference._is_headroom_safe(reading),
            lambda: native.is_headroom_safe(reading),
        )
        _agree(
            lambda: reference._safety_metric_problems(reading),
            lambda: native.safety_metric_problems(reading),
        )


@pytest.mark.parametrize("batch", range(4))
def test_voice_setup_reads_the_released_excess_like_the_python_code(batch):
    names = reference._NATIVE_SAFETY_METRICS
    cases = _readings()[batch::4]
    cases += [{**dict(zip(names, (2.0, 0.0, 0.0))), name: value} for name in names for value in VALUES]
    for reading in cases:
        _agree(
            lambda: reference._released_excess_db(reading),
            lambda: voice_setup._released_excess_db(reading),
        )


def test_the_excess_is_the_signed_zero_the_python_max_returns():
    names = reference._NATIVE_SAFETY_METRICS
    # 1.0 - 1.0 is +0.0 and the others are -1.0 and -0.5; a tie of -0.0 and 0.0 keeps the first.
    for reading in (
        {names[0]: 1.0, names[1]: 0.0, names[2]: 0.0},
        {names[0]: 2.0, names[1]: 1.0, names[2]: 0.5},
        {names[0]: 2.0, names[1]: 1.0, names[2]: -0.0},
    ):
        _same(native.headroom_excess_db(reading), reference._headroom_excess_db(reading))


def test_the_thresholds_and_metric_names_are_the_python_ones():
    assert native.HEADROOM_TARGET_DB == reference.HEADROOM_TARGET_DB
    assert native.LIMITER_GAIN_REDUCTION_WARN_DB == reference.LIMITER_GAIN_REDUCTION_WARN_DB
    assert native.TRUE_PEAK_GAIN_REDUCTION_WARN_DB == reference.TRUE_PEAK_GAIN_REDUCTION_WARN_DB
    assert native.NATIVE_SAFETY_METRICS == reference._NATIVE_SAFETY_METRICS
    assert isinstance(native.NATIVE_SAFETY_METRICS, tuple)
    assert math.isfinite(native.HEADROOM_TARGET_DB)
