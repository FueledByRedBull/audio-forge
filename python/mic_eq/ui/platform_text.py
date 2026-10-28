"""Names for the platform's own settings in user-facing text."""

from __future__ import annotations

import sys

# Where the user changes the microphone level: "in Windows or on the microphone".
SOUND_SETTINGS = "Windows" if sys.platform == "win32" else "your sound settings"
