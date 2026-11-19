"""Names for the platform's own settings in user-facing text."""

from __future__ import annotations

import sys

IS_WINDOWS = sys.platform == "win32"

# Where the user changes the microphone level: "in Windows or on the microphone".
SOUND_SETTINGS = "Windows" if IS_WINDOWS else "your sound settings"
