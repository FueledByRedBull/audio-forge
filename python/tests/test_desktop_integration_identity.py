from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _single_instance_name(**environment: str) -> str:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from mic_eq.ui.desktop_integration import SINGLE_INSTANCE_SERVER; "
            "print(SINGLE_INSTANCE_SERVER)",
        ],
        # An installed copy of the package must not shadow the one under test.
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]), **environment},
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return result.stdout.strip()


@pytest.mark.skipif(sys.platform != "win32", reason="the account SID is a Windows concept")
def test_single_instance_name_follows_the_account_not_the_user_name_variables():
    plain = _single_instance_name()
    spoofed = _single_instance_name(
        USERNAME="someone-else", USER="someone-else", LOGNAME="someone-else", LNAME="someone-else"
    )

    assert plain == spoofed
    assert plain.startswith("AudioForge.SingleInstance.v1.S-1-")
