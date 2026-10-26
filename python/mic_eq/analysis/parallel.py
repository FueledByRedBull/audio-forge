"""How many native renders the setup analysis runs side by side."""

from __future__ import annotations

import os

# The native simulators release the GIL and keep their DSP state to themselves, so
# independent candidates can render together. Half the logical processors, from 4 to 8,
# leaves the audio engine and the window their share: a 16-thread machine gets 8.
ANALYSIS_WORKERS = max(4, min(8, (os.cpu_count() or 4) // 2))
