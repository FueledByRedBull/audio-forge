"""How many analysis candidates run side by side on a machine with so many logical processors."""

import os

import pytest

from mic_eq.analysis import parallel
from mic_eq.analysis.parallel import analysis_workers


@pytest.mark.parametrize(
    ("cpus", "workers"),
    [(None, 3), (1, 1), (2, 1), (3, 2), (4, 3), (5, 4), (8, 4), (12, 6), (16, 8), (32, 8)],
)
def test_workers_are_half_the_processors_between_4_and_8_but_leave_one_free(cpus, workers):
    assert analysis_workers(cpus) == workers


def test_the_workers_never_use_every_processor_and_never_fall_below_one():
    for cpus in range(1, 40):
        assert 1 <= analysis_workers(cpus) <= max(1, cpus - 1), cpus


def test_the_module_uses_the_count_of_this_machine():
    assert parallel.ANALYSIS_WORKERS == analysis_workers(os.cpu_count())
