"""The plain wording for the reasons the setup analysis gives."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mic_eq.ui.setup_messages import is_mapped, plain_error, plain_reason, plain_reasons

ANALYSIS = Path(__file__).resolve().parents[1] / "mic_eq" / "analysis"
MODULES = (
    "voice_setup.py",
    "noise_reference.py",
    "failure_detection.py",
    "joint_tuning.py",
    "compressor_calibration.py",
    "auto_eq_parts/optimizer.py",
    "auto_eq_parts/headroom.py",
)
# Words the analysis uses among itself that a person setting up a microphone should not meet.
JARGON = (
    "p95", "VAD", "dBTP", "candidate", "holdout", "held-out", "incumbent", "stationary",
    "transient", "full-chain", "DSP", "SNR", "predefined", "validation", "pre-gain",
)


def _sample(node: ast.AST) -> str:
    """The text a literal reason produces, with every formatted value as 1.5."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            str(part.value) if isinstance(part, ast.Constant) else "1.5" for part in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _sample(node.left) + _sample(node.right)
    return "1.5"


def _literals(node: ast.AST) -> list[str]:
    """Every reason text an expression can produce."""
    if isinstance(node, ast.IfExp):
        return _literals(node.body) + _literals(node.orelse)
    if isinstance(node, ast.Dict):
        return [text for value in node.values for text in _literals(value)]
    if isinstance(node, ast.GeneratorExp):
        return _literals(node.elt)
    if isinstance(node, ast.Subscript):
        return _literals(node.value)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [text for element in node.elts for text in _literals(element)]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join":
        return [text for arg in node.args for text in _literals(arg)]
    if isinstance(node, ast.Constant):
        return [node.value] if isinstance(node.value, str) else []
    if isinstance(node, (ast.JoinedStr, ast.BinOp)):
        return [_sample(node)]
    return []


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _reason_literals() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}

    def add(module: str, texts: list[str]) -> None:
        for text in texts:
            # Codes such as "accepted" are not sentences, and a bare placeholder says nothing.
            if " " in text.strip() and text.strip("1.5 :") != "":
                found.setdefault(text, set()).add(module)

    for module in MODULES:
        tree = ast.parse((ANALYSIS / module).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _name(node.func)
                receiver = _name(node.func.value) if isinstance(node.func, ast.Attribute) else ""
                if name in ("append", "extend") and "reasons" in receiver:
                    # Findings kept for a log or a developer, not for the person.
                    if receiver not in ("hard_fail_reasons", "soft_failures"):
                        add(module, [t for arg in node.args for t in _literals(arg)])
                elif name == "_verification_outcome" and len(node.args) > 1:
                    add(module, _literals(node.args[1]))
                elif name == "_add_abstention" and len(node.args) > 1:
                    add(module, _literals(node.args[1]))
                elif name == "_unavailable_result":
                    for keyword in node.keywords:
                        if keyword.arg == "reason":
                            add(module, _literals(keyword.value))
            elif isinstance(node, ast.Assign) and module in (
                "failure_detection.py",
                "auto_eq_parts/headroom.py",
            ):
                if any(_name(target) == "reason" for target in node.targets):
                    add(module, _literals(node.value))
            elif isinstance(node, ast.FunctionDef) and node.name == "_compressor_p95_target_status":
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Return) and isinstance(inner.value, ast.Tuple):
                        add(module, _literals(inner.value.elts[1]))
    # The silent-reference reason is a module constant appended by name.
    from mic_eq.analysis.noise_reference import SILENT_REFERENCE_REASON

    found.setdefault(SILENT_REFERENCE_REASON, set()).add("noise_reference.py")
    return found


def test_the_walk_finds_the_reasons_the_analysis_gives():
    literals = _reason_literals()

    # A handful from each source, so a broken walk cannot pass by finding nothing.
    for expected in (
        "too little VAD-active speech",
        "room-noise capture is not stationary",
        "compressor p95 gain reduction target is unavailable",
        "no candidate passed every predefined gate",
        "relative noise floor increased by 1.5 dB",
        "no validated correction improved the target safely",
    ):
        assert expected in literals, expected
    assert len(literals) > 60


def test_every_reason_the_analysis_gives_has_a_plain_sentence():
    unmapped = sorted(text for text in _reason_literals() if not is_mapped(text))

    assert unmapped == [], (
        "These analysis reasons have no entry in setup_messages (or auto_eq_explanation); "
        f"add one so the person never reads them raw: {unmapped}"
    )


def test_the_plain_sentences_are_plain_and_finished():
    for text in _reason_literals():
        plain = plain_reason(text)
        assert plain.endswith((".", ")")), (text, plain)
        assert not any(word.lower() in plain.lower() for word in JARGON), (text, plain)


def test_known_bad_wording_is_replaced_and_numbers_are_kept():
    assert plain_reason("compressor p95 gain reduction target is unavailable") == (
        "The compressor could not be tuned on your recording, so its settings are a rough guess."
    )
    assert plain_reason(
        "compressor p95 gain reduction target missed: measured 2.70 dB, target 3.50 dB (±0.75 dB)"
    ).startswith("The compressor evens out your voice by 2.70 dB; the chosen style aims for 3.50 dB.")
    assert "typed candidate" not in plain_reason(
        "full-chain validation failed: typed candidate EQ settings are unavailable (ValueError: x)"
    )
    assert plain_reason("room-noise capture is not stationary").startswith(
        "The room noise was not steady"
    )
    assert "6 dB" in plain_reason(
        "the microphone level leaves too little headroom; lower the input gain by about 6 dB"
    )


def test_an_unknown_reason_is_shown_as_it_was_given():
    assert not is_mapped("something the analysis learned later")
    assert plain_reason("something the analysis learned later") == "something the analysis learned later"
    assert plain_reasons(["a new reason", "a new reason"]) == ["a new reason"]


def test_reasons_that_say_the_same_thing_are_said_once():
    reasons = [
        "room-noise capture is not stationary",
        "room-noise capture contains strong transients",
        "too little VAD-active speech",
        "room-noise capture is not stationary",
    ]

    plain = plain_reasons(reasons)

    assert len(plain) == 3 and len(set(plain)) == 3


@pytest.mark.parametrize(
    ("raw", "expected_start"),
    [
        ("Audio capture exceeded its 15-second time limit.", "The microphone stopped sending sound."),
        ("No audio capture progress was received.", "The microphone stopped sending sound."),
        ("Recording failed - no audio data", "The microphone stopped sending sound."),
        ("Room-noise capture was too short for setup.", "The room-noise recording was too short."),
        ("Voice capture was too short for setup.", "The voice recording was too short."),
        ("Audio too short for FFT: need 4096 samples, got 100", "The recording was too short"),
    ],
)
def test_known_errors_are_explained(raw, expected_start):
    assert plain_error(raw).startswith(expected_start)


def test_an_unknown_error_keeps_its_text():
    plain = plain_error("disk exploded")

    assert "disk exploded" in plain and plain != "disk exploded"


# A reason the analysis builds from a runtime value, so no literal shows it to the walk.
_COMPOSED_REASONS = (
    "full-chain validation failed: typed candidate EQ settings are unavailable (ValueError: x)",
)


def test_every_rule_can_be_reached_by_some_reason():
    from mic_eq.ui.setup_messages import _COMPILED_REASONS

    reached: set[int] = set()
    for text in (*_reason_literals(), *_COMPOSED_REASONS):
        first = next(
            (i for i, (pattern, _) in enumerate(_COMPILED_REASONS) if pattern.fullmatch(text.strip())),
            None,
        )
        if first is not None:
            reached.add(first)

    shadowed = [pattern.pattern for i, (pattern, _) in enumerate(_COMPILED_REASONS) if i not in reached]
    assert shadowed == [], f"An earlier rule answers every reason these rules were written for: {shadowed}"


def test_a_changed_sample_rate_gets_its_own_restart_advice():
    assert plain_reason("sample rate changed between noise and voice captures") == (
        "The sample rate changed between the room-noise and voice recordings. "
        "Restart processing and record both again."
    )


def test_a_reason_never_says_what_became_of_the_previous_settings():
    # Only the dialog knows whether they were put back.
    for text in _reason_literals():
        assert "put back" not in plain_reason(text), text


@pytest.mark.parametrize(
    "reason",
    [
        "insufficient repeatable voiced windows",
        "capture quality score is too low",
        "noise-referenced SNR is too low",
        "room-noise reference is invalid",
    ],
)
def test_a_tone_step_abstention_says_what_to_do(reason):
    sentence = plain_reason(reason)

    assert sentence.endswith(".") and ("Record again" in sentence or "Move closer" in sentence)
