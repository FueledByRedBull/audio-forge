"""An unexpected error inside an analysis worker leaves a traceback in the log."""

from __future__ import annotations


import numpy as np
import pytest

from mic_eq.ui import analysis_worker, listening_comparison_dialog, voice_setup_dialog


def _audio() -> np.ndarray:
    return np.zeros(480, dtype=np.float32)


def _voice_worker():
    return voice_setup_dialog.VoiceSetupWorker(
        _audio(), _audio(), 48_000, "broadcast",
        vad_available=False, dynamics_intensity="medium",
        custom_target_p95_db=-20.0, custom_peak_cap_db=-3.0,
        limiter_settings=None, noise_metadata=None, voice_metadata=None,
    )


def _verification_worker():
    return voice_setup_dialog.VoiceSetupVerificationWorker(
        _audio(), _audio(), _audio(), 48_000, {}, "broadcast"
    )


def _comparison_worker():
    return listening_comparison_dialog.ListeningComparisonWorker(_audio(), 48_000, {}, {})


@pytest.mark.parametrize(
    ("module", "function", "make_worker", "message"),
    [
        (analysis_worker, "analyze_auto_eq", lambda: analysis_worker.AnalysisWorker(_audio(), 48_000), "Auto-EQ"),
        (voice_setup_dialog, "analyze_voice_setup", _voice_worker, "Voice Setup analysis"),
        (voice_setup_dialog, "validate_voice_setup_verification", _verification_worker, "verification"),
        (listening_comparison_dialog, "render_comparison", _comparison_worker, "Listening comparison"),
    ],
)
def test_a_worker_exception_is_logged_with_its_traceback(
    monkeypatch, caplog, module, function, make_worker, message
):
    def explode(*_args, **_kwargs):
        raise KeyError("band_gains")

    monkeypatch.setattr(module, function, explode)
    worker = make_worker()
    failures: list[str] = []
    worker.failed.connect(failures.append)

    worker.run()  # on this thread: the signals are direct and the log record is ours

    assert len(failures) == 1
    record = next(r for r in caplog.records if message in r.getMessage())
    assert record.exc_info is not None and record.exc_info[0] is KeyError


@pytest.mark.parametrize(
    ("module", "function", "make_worker"),
    [
        (analysis_worker, "analyze_auto_eq", lambda: analysis_worker.AnalysisWorker(_audio(), 48_000)),
        (voice_setup_dialog, "analyze_voice_setup", _voice_worker),
        (voice_setup_dialog, "validate_voice_setup_verification", _verification_worker),
    ],
)
def test_an_exception_without_a_message_still_gives_the_user_a_sentence(
    monkeypatch, module, function, make_worker
):
    def explode(*_args, **_kwargs):
        raise ValueError()

    monkeypatch.setattr(module, function, explode)
    worker = make_worker()
    failures: list[str] = []
    worker.failed.connect(failures.append)

    worker.run()

    assert failures == [analysis_worker.UNEXPECTED_FAILURE]
