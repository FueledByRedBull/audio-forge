"""How many native renders the setup analysis runs side by side."""

from __future__ import annotations

import os


def analysis_workers(cpu_count: int | None) -> int:
    """Half the logical processors, from 4 to 8, and never more than all but one of them.

    The native simulators release the GIL and keep their DSP state to themselves, so
    independent candidates can render together. Leaving a processor free keeps the audio
    engine and the window responsive: a 16-thread machine gets 8, a 4-thread one 3.
    """
    cpus = cpu_count or 4
    return min(max(4, min(8, cpus // 2)), max(1, cpus - 1))


ANALYSIS_WORKERS = analysis_workers(os.cpu_count())
