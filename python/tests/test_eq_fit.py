"""Unit tests for the Auto-EQ solvers in eq_fit.py.

SciPy appears only here, as the reference the solvers are checked against.
"""

from __future__ import annotations

import ast
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mic_eq.analysis.auto_eq_parts import eq_fit
from mic_eq.analysis.auto_eq_parts.eq_fit import (
    bounded_least_squares,
    project_gains,
    refine_gains,
    _project_onto_polytope,
)


class Cancelled(RuntimeError):
    pass


def rosenbrock(x):
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


T = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
Y = np.array([2.1, 1.45, 1.05, 0.8, 0.55, 0.48, 0.3])


def exp_fit(p):
    return p[0] * np.exp(-p[1] * T) + p[2] - Y


def curve(x):
    return np.array([x[0] + 2 * x[1] - 2, x[0] ** 2 + x[1] ** 2 - 1, x[0] * x[1] - 0.2, 3 * x[2] - x[0] * x[1] - 0.5])


LOOSE = dict(ftol=1e-4, xtol=1e-4, gtol=1e-6)
PROBLEMS = {
    "rosenbrock": (rosenbrock, [-1.2, 1.0], [-2, -2], [2, 2], dict(LOOSE, max_nfev=120)),
    "rosenbrock_active_bound": (rosenbrock, [-1.2, 1.0], [-2, -2], [0.5, 2], dict(LOOSE, max_nfev=120)),
    "exp_fit": (exp_fit, [1.0, 1.0, 0.0], [0, 0, -1], [5, 5, 2], dict(LOOSE, max_nfev=180)),
    "exp_fit_capped": (exp_fit, [3.0, 0.1, 1.0], [0, 0, -1], [5, 5, 2], dict(ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=4)),
    "curve": (curve, [0.1, 0.1, 0.1], [0, 0, 0], [1, 1, 1], dict(LOOSE, max_nfev=120)),
}


# ------------------------------------------------------------------ module hygiene


def test_module_imports_no_scipy():
    tree = ast.parse(Path(eq_fit.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "scipy" not in imported


# ------------------------------------------------------------------ least squares vs SciPy


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_least_squares_matches_scipy(name):
    """Iteration counts depend on the LAPACK build behind the SVD, so only the outcome is compared."""
    scipy_optimize = pytest.importorskip("scipy.optimize")
    fun, x0, lb, ub, kw = PROBLEMS[name]
    expected = scipy_optimize.least_squares(fun, x0, bounds=(lb, ub), method="trf", **kw)
    actual = bounded_least_squares(fun, x0, (lb, ub), **kw)
    assert actual.success == expected.success
    np.testing.assert_allclose(actual.x, expected.x, atol=1e-6)
    assert actual.cost == pytest.approx(expected.cost, rel=1e-5, abs=1e-12)


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_least_squares_is_bit_identical_to_scipy_with_scipys_svd(name, monkeypatch):
    """The port itself is exact: only the BLAS/LAPACK build behind svd differs."""
    scipy_optimize = pytest.importorskip("scipy.optimize")
    import scipy.linalg

    monkeypatch.setattr(np.linalg, "svd", lambda a, full_matrices=True: scipy.linalg.svd(a, full_matrices=full_matrices))
    fun, x0, lb, ub, kw = PROBLEMS[name]
    expected = scipy_optimize.least_squares(fun, x0, bounds=(lb, ub), method="trf", **kw)
    actual = bounded_least_squares(fun, x0, (lb, ub), **kw)
    np.testing.assert_array_equal(actual.x, expected.x)
    assert (actual.status, actual.nfev, actual.njev, actual.cost) == (
        expected.status, expected.nfev, expected.njev, expected.cost)


def test_least_squares_x_scale_array_is_accepted():
    result = bounded_least_squares(rosenbrock, [-1.2, 1.0], ([-2, -2], [2, 2]), x_scale=[1.0, 100.0],
                                   ftol=1e-10, xtol=1e-10, gtol=1e-10)
    assert result.success
    assert result.x[0] == pytest.approx(1.0, abs=1e-5)


def test_least_squares_cap_is_failure_status_zero():
    result = bounded_least_squares(rosenbrock, [-1.2, 1.0], ([-2, -2], [2, 2]), ftol=1e-12, xtol=1e-12,
                                   gtol=1e-12, max_nfev=3)
    assert (result.status, result.success, result.nfev) == (0, False, 3)


def test_least_squares_backs_off_after_a_non_finite_trial_residual():
    scipy_optimize = pytest.importorskip("scipy.optimize")

    def make():
        calls = []

        def fun(x):
            calls.append(1)
            # call 0 is x0, call 1 the finite-difference probe, call 2 the first trial step
            return np.array([np.nan if len(calls) == 3 else x[0] - 1.5])

        return fun

    actual = bounded_least_squares(make(), [0.0], ([-1.0], [2.0]))
    expected = scipy_optimize.least_squares(make(), [0.0], bounds=([-1.0], [2.0]), method="trf")
    assert actual.x[0] == pytest.approx(1.5, abs=1e-6)
    assert (actual.status, actual.nfev) == (expected.status, expected.nfev)


def non_finite_jacobian(bad, at_call):
    """Residuals of ``x - 1`` that are ``bad`` in one component on call ``at_call``.

    Call 1 is the start, call 2 the first finite-difference probe, call 3 the first trial step and
    call 4 the probe at the accepted point.
    """
    calls = []

    def fun(x):
        calls.append(1)
        return np.array([x[0] - 1.0, bad if len(calls) == at_call else 0.0])

    return fun


@pytest.mark.parametrize("at_call", [2, 4])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_least_squares_rejects_non_finite_jacobian(bad, at_call):
    with pytest.raises(ValueError, match="infs or NaNs"):
        bounded_least_squares(non_finite_jacobian(bad, at_call), [0.0], ([-2.0], [2.0]))


@pytest.mark.parametrize("at_call", [2, 4])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_non_finite_jacobian_fails_as_in_scipy(bad, at_call):
    scipy_optimize = pytest.importorskip("scipy.optimize")
    with pytest.raises(ValueError, match="infs or NaNs"):
        scipy_optimize.least_squares(non_finite_jacobian(bad, at_call), [0.0], bounds=([-2.0], [2.0]), method="trf")


def test_least_squares_propagates_exceptions_unchanged():
    calls = []

    def fun(x):
        calls.append(1)
        if len(calls) == 9:
            raise Cancelled("stop")
        return rosenbrock(x)

    with pytest.raises(Cancelled, match="stop"):
        bounded_least_squares(fun, [-1.2, 1.0], ([-2, -2], [2, 2]))
    assert len(calls) == 9


def test_least_squares_passes_args():
    seen = []

    def fun(x, scale, offset):
        seen.append((scale, offset))
        return np.array([scale * x[0] - offset])

    result = bounded_least_squares(fun, [0.0], ([-5.0], [5.0]), args=(2.0, 3.0))
    assert result.x[0] == pytest.approx(1.5, abs=1e-6)
    assert set(seen) == {(2.0, 3.0)}


@pytest.mark.parametrize(
    "x0,lb,ub",
    [([0.0, 0.5], [0.0, 0.0], [0.0, 1.0]), ([2.0, 0.5], [0.0, 0.0], [1.0, 1.0]), ([0.5], [0.0, 0.0], [1.0, 1.0]),
     ([0.5, 0.5], [-np.inf, -np.inf], [np.inf, np.inf])],
)
def test_least_squares_rejects_bad_input(x0, lb, ub):
    with pytest.raises(ValueError):
        bounded_least_squares(rosenbrock, x0, (lb, ub))


def test_least_squares_rejects_a_non_finite_start_before_any_jacobian_evaluation():
    calls = []

    def residual(x):
        calls.append(x.copy())
        return np.array([np.nan if len(calls) == 1 else x[0] - 0.25])

    with pytest.raises(ValueError, match="not finite"):
        bounded_least_squares(residual, [0.5], ([0.0], [1.0]))
    assert len(calls) == 1  # x0 only: a finite-difference Jacobian would probe more points


def test_least_squares_rejects_bad_options():
    with pytest.raises(ValueError):
        bounded_least_squares(rosenbrock, [0.0, 0.0], ([-1, -1], [1, 1]), max_nfev=0)
    with pytest.raises(ValueError):
        bounded_least_squares(rosenbrock, [0.0, 0.0], ([-1, -1], [1, 1]), x_scale=[1.0, -1.0])


# ------------------------------------------------------------------ polytope projection


def brute_force_projection(a, normals, bounds):
    """Exhaustive KKT enumeration; the unique optimum has non-negative multipliers."""
    n, k = a.size, bounds.size
    for size in range(0, n + 1):
        for subset in itertools.combinations(range(k), size):
            idx = list(subset)
            if idx:
                gram = normals[idx] @ normals[idx].T
                if abs(np.linalg.det(gram)) < 1e-12:
                    continue
                lam = np.linalg.solve(gram, normals[idx] @ a - bounds[idx])
            else:
                lam = np.zeros(0)
            if np.any(lam < -1e-9):
                continue
            x = a - (normals[idx].T @ lam if idx else 0.0)
            if np.all(normals @ x <= bounds + 1e-9):
                return x
    return None


def test_goldfarb_idnani_matches_exhaustive_kkt_enumeration():
    rng = np.random.default_rng(12345)
    for case in range(300):
        n, k = 2 + case % 3, 3 + case % 5
        normals = rng.uniform(-1, 1, (k, n))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        bounds = rng.uniform(0.0, 1.0, k)  # origin feasible
        a = rng.uniform(-3, 3, n)
        expected = brute_force_projection(a, normals, bounds)
        actual = _project_onto_polytope(a, normals, bounds)
        assert expected is not None and actual is not None
        np.testing.assert_allclose(actual, expected, atol=1e-8, err_msg=f"case {case}")


def test_projection_detects_empty_polytope():
    normals = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert _project_onto_polytope(np.zeros(2), normals, np.array([-1.0, -1.0])) is None


def chain(rng, n=10, fixed=()):
    limits = rng.uniform(0.5, 6.0, n - 1)
    lower = rng.uniform(-12, -0.5, n)
    upper = rng.uniform(0.5, 12, n)
    for i in fixed:
        lower[i] = upper[i] = 0.0
    return lower, upper, limits


def test_gain_projection_is_feasible_optimal_idempotent():
    rng = np.random.default_rng(99)
    for case in range(50):
        lower, upper, limits = chain(rng, fixed=(3, 4) if case % 5 == 0 else ())
        g0 = rng.uniform(-15, 15, 10)
        solved = project_gains(g0, lower, upper, limits)
        assert solved.success
        x = solved.x
        assert np.all(x >= lower - 1e-11) and np.all(x <= upper + 1e-11)
        assert np.all(np.abs(np.diff(x)) <= limits + 1e-11)
        for _ in range(4):
            y = project_gains(rng.uniform(-15, 15, 10), lower, upper, limits).x
            assert (g0 - x) @ (y - x) <= 1e-8
        np.testing.assert_allclose(project_gains(x, lower, upper, limits).x, x, atol=1e-12)


def test_projection_leaves_feasible_point_untouched():
    g0 = np.array([1.0, 2.5, 4.0, 2.0])
    np.testing.assert_array_equal(project_gains(g0, [-12] * 4, [12] * 4, [3.0] * 3).x, g0)


def test_projection_callback_and_validation():
    seen = []
    project_gains([0, 0], [-1, -1], [1, 1], [1.0], seen.append)
    assert len(seen) == 1
    seen.clear()
    project_gains([0, 0], [0, 0], [0, 0], [1.0], seen.append)  # nothing free: SciPy skipped the callback
    assert seen == []
    with pytest.raises(Cancelled):
        project_gains([0, 0], [-1, -1], [1, 1], [1.0], lambda x: (_ for _ in ()).throw(Cancelled()))
    invalid: tuple[dict[str, Any], ...] = (
        dict(g0=[np.nan, 0], lower=[-1, -1], upper=[1, 1], adjacent_limit=[1.0]),
        dict(g0=[0], lower=[-1, -1], upper=[1, 1], adjacent_limit=[1.0]),
        dict(g0=[0, 0], lower=[1, 1], upper=[-1, -1], adjacent_limit=[1.0]),
    )
    for bad in invalid:
        with pytest.raises(ValueError):
            project_gains(**bad)


def test_gain_projection_with_tight_and_degenerate_bounds():
    g0 = np.array([5.0, -5.0, 3.0, -4.0])
    wide = [12.0] * 4
    tight = project_gains(g0, [-1e-9] * 4, [1e-9] * 4, [1.0] * 3)
    assert tight.success
    np.testing.assert_allclose(tight.x, np.clip(g0, -1e-9, 1e-9), atol=1e-12)
    # A zero adjacency limit forces equal gains: the mean, clipped into the shared bounds.
    flat = project_gains(g0, [-12.0] * 4, wide, [0.0] * 3)
    assert flat.success
    np.testing.assert_allclose(flat.x, g0.mean(), atol=1e-12)
    capped = project_gains(g0, [-12.0] * 4, [12.0, 12.0, -1.0, 12.0], [0.0] * 3)
    np.testing.assert_allclose(capped.x, -1.0, atol=1e-12)
    # A free band between fixed neighbours may only move by the adjacency limit.
    pinned = project_gains([0.0, 10.0, 0.0], [0.0, -12.0, 0.0], [0.0, 12.0, 0.0], [3.0, 3.0])
    assert pinned.success
    np.testing.assert_allclose(pinned.x, [0.0, 3.0, 0.0], atol=1e-12)
    # Zero-width bounds everywhere: nothing is free, the fixed values are returned.
    frozen = project_gains(g0, [1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [0.0, 1.0, 0.0])
    assert frozen.success
    np.testing.assert_array_equal(frozen.x, [1.0, 1.0, 2.0, 2.0])
    # Fixed neighbours further apart than the limit leave no feasible point.
    assert not project_gains([1.0, 10.0, 2.0], [1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [0.0, 0.0]).success


# ------------------------------------------------------------------ refinement


def cubic_residual(target, cubic):
    target = np.asarray(target, dtype=float)
    return lambda g: (g - target) + cubic * g**3


def test_refinement_of_linear_residual_equals_projection():
    rng = np.random.default_rng(5)
    for _ in range(40):
        limits = rng.uniform(0.5, 6.0, 9)
        lower, upper = np.full(10, -12.0), np.full(10, 12.0)
        target = rng.uniform(-14, 14, 10)
        expected = project_gains(target, lower, upper, limits).x
        solved = refine_gains(cubic_residual(target, 0.0), np.zeros(10), lower, upper, limits)
        assert solved.success
        np.testing.assert_allclose(solved.x, expected, atol=1e-6)


def test_refinement_of_nonlinear_residual_matches_projected_gradient_optimum():
    n, limits = 6, np.full(5, 2.0)
    lower, upper = np.full(n, -8.0), np.full(n, 8.0)
    target = np.array([6.0, -5.0, 7.0, 0.5, -7.0, 3.0])
    cubic = 0.02
    solved = refine_gains(cubic_residual(target, cubic), np.zeros(n), lower, upper, limits)
    assert solved.success
    x = np.zeros(n)
    for _ in range(20000):  # reference: projected gradient descent with the exact projection
        r = (x - target) + cubic * x**3
        step = project_gains(x - 0.02 * 2.0 * r * (1.0 + 3.0 * cubic * x**2), lower, upper, limits).x
        converged = np.max(np.abs(step - x)) < 1e-13
        x = step
        if converged:
            break

    def cost(g):
        return float(np.sum(((g - target) + cubic * g**3) ** 2))

    assert cost(solved.x) <= cost(x) + 1e-9
    np.testing.assert_allclose(solved.x, x, atol=1e-4)


def test_refinement_cap_is_failure_and_no_free_variables_is_success():
    solved = refine_gains(cubic_residual([3, -3, 2, 1], 0.05), np.zeros(4), [-12] * 4, [12] * 4, [20.0] * 3,
                          max_iter=1)
    assert not solved.success and solved.iterations == 1
    frozen = refine_gains(cubic_residual([5] * 4, 0.0), np.zeros(4), [0] * 4, [0] * 4, [3.0] * 3)
    assert frozen.success and np.all(frozen.x == 0.0)
    conflict = refine_gains(cubic_residual([5, 5], 0.0), [0.0, 5.0], [0.0, 5.0], [0.0, 5.0], [1.0])
    assert not conflict.success


def test_refinement_with_fixed_bands():
    lower = np.array([0.0, -12.0, -12.0, 0.0])
    upper = np.array([0.0, 12.0, 12.0, 0.0])
    solved = refine_gains(cubic_residual([5] * 4, 0.0), np.zeros(4), lower, upper, [3.0] * 3)
    assert solved.success and solved.x[0] == 0.0 and solved.x[3] == 0.0
    np.testing.assert_allclose(solved.x[1:3], 3.0, atol=1e-6)


def test_refinement_propagates_exceptions_from_residual_and_callback():
    calls = []

    def residual(g):
        calls.append(1)
        if len(calls) == 7:
            raise Cancelled("residual")
        return g - 4.0

    with pytest.raises(Cancelled, match="residual"):
        refine_gains(residual, np.zeros(4), [-12] * 4, [12] * 4, [3.0] * 3)
    with pytest.raises(Cancelled, match="callback"):
        refine_gains(lambda g: g - 4.0, np.zeros(4), [-12] * 4, [12] * 4, [3.0] * 3,
                     lambda x: (_ for _ in ()).throw(Cancelled("callback")))


def test_refinement_with_tight_and_degenerate_bounds():
    tight = refine_gains(cubic_residual([4.0] * 4, 0.0), np.zeros(4), [-1e-9] * 4, [1e-9] * 4, [1.0] * 3)
    assert tight.success and np.all(np.abs(tight.x) <= 1e-9 + 1e-12)
    flat = refine_gains(cubic_residual([6.0, -2.0, 4.0, 0.0], 0.0), np.zeros(4), [-12.0] * 4, [12.0] * 4, [0.0] * 3)
    assert flat.success
    np.testing.assert_allclose(flat.x, 2.0, atol=1e-6)


def test_refinement_rejects_residual_length_change_and_bad_start():
    calls = []

    def residual(g):
        calls.append(1)
        return g - 4.0 if len(calls) < 4 else (g - 4.0)[:2]

    with pytest.raises(ValueError):
        refine_gains(residual, np.zeros(4), [-12] * 4, [12] * 4, [3.0] * 3)
    with pytest.raises(ValueError):
        refine_gains(lambda g: np.array([np.nan]), np.zeros(2), [-1, -1], [1, 1], [1.0])


# ------------------------------------------------------------------ cancellation through the fit


def synthetic_fit_inputs():
    freqs = np.geomspace(20.0, 20000.0, 64)
    return freqs, -70.0 + 10.0 / (1.0 + (freqs / 200.0) ** 2), np.zeros_like(freqs)


def test_cancellation_surfaces_from_every_stage_of_the_fit():
    """A cancel check that fires in the first solve, the second solve or the gain refinement."""
    from mic_eq.analysis.auto_eq import calculate_eq_bands
    from mic_eq.analysis.cancellation import AnalysisCancelled

    inputs = synthetic_fit_inputs()
    seen = []

    def count():
        seen.append(1)
        return False

    calculate_eq_bands(*inputs, cancel_check=count)
    total = len(seen)
    assert total > 20

    # For these inputs about the first 57 % of checks are the first solve, the
    # next 35 % the second, then a few refinement steps and the validation.
    for trigger in (3, round(total * 0.75), total - 5):
        checks = []

        def cancel():
            checks.append(1)
            return len(checks) >= trigger

        with pytest.raises(AnalysisCancelled):
            calculate_eq_bands(*inputs, cancel_check=cancel)
        assert len(checks) == trigger


# ------------------------------------------------------------------ side by side with SLSQP

ADJACENCY_TOLERANCE = 1e-9  # what optimizer.py accepts above the unmargined limit
SLSQP_MARGIN = 1e-7


def _fit_problem(rng, bands, saturate):
    """Gain-refinement inputs shaped like the app's: a measured curve, a target that asks for a
    large per-band correction (`saturate` dB scale), loose or tight gain bounds."""
    from mic_eq.analysis.auto_eq_parts.constants import GAIN_MAX_DB, GAIN_MIN_DB

    centers = np.sort(np.geomspace(60, 14000, bands) * np.exp(rng.normal(0, 0.08, bands)))
    dense = np.geomspace(20, 20000, 256)
    measured = rng.normal(0, 1, dense.size).cumsum() * rng.choice([0.1, 0.5, 2.0]) - 60
    target = (
        measured
        + np.interp(np.log(dense), np.log(centers), rng.normal(0, saturate, bands))
        + rng.normal(0, 1, dense.size)
    )
    qs = rng.uniform(0.5, 4.0, bands)
    weights = rng.uniform(0.2, 1.5, dense.size)
    lower = np.full(bands, GAIN_MIN_DB)
    upper = np.full(bands, GAIN_MAX_DB)
    if rng.random() < 0.15:  # tight, partly fixed bounds
        upper = rng.uniform(0, GAIN_MAX_DB, bands)
        lower = -rng.uniform(0, -GAIN_MIN_DB, bands)
        fixed = rng.random(bands) < 0.2
        lower[fixed] = upper[fixed] = 0.0
    start = rng.uniform(lower - 2, upper + 2) if rng.random() < 0.5 else rng.normal(0, 4, bands)
    return start, dense, measured, target, qs, centers, weights, lower, upper


def _feasible(gains, lower, upper, limits):
    return bool(
        np.all(np.isfinite(gains))
        and np.all(gains >= lower - 1e-9)
        and np.all(gains <= upper + 1e-9)
        and np.all(np.abs(np.diff(gains)) <= limits + ADJACENCY_TOLERANCE)
    )


def _slsqp_refinement(problem):
    """The previous SciPy implementation of optimizer._constrained_gain_refinement, verbatim in
    objective, constraint, bounds and options; returns (result, objective, start, limits)."""
    from scipy.optimize import minimize

    from mic_eq.analysis.auto_eq_parts.constants import LAMBDA_G, LAMBDA_TILT
    from mic_eq.analysis.auto_eq_parts.optimizer import (
        _adjacent_gain_limits,
        _log_frequency_gain_curvature,
    )
    from mic_eq.analysis.auto_eq_parts.response import _predict_eq_response

    gains, dense, measured, target, qs, centers, weights, lower, upper = problem
    limits = _adjacent_gain_limits(centers)
    log_centers = np.log10(np.clip(centers, 1e-6, None))
    centered = log_centers - float(np.mean(log_centers))
    denominator = float(np.sum(centered**2))

    def objective(candidate):
        response = _predict_eq_response(dense, candidate, qs, centers, sample_rate=48000.0)
        error = target - (measured + response)
        curvature = _log_frequency_gain_curvature(candidate, centers)
        tilt = float(np.dot(centered, candidate) / denominator)
        return float(
            np.sum(weights * error * error)
            + LAMBDA_G * np.sum(curvature * curvature)
            + LAMBDA_TILT * tilt * tilt
        )

    bounds = list(zip(lower.tolist(), upper.tolist(), strict=True))
    constraints = ({"type": "ineq", "fun": lambda g: limits - SLSQP_MARGIN - np.abs(np.diff(g))},)
    projection = project_gains(np.clip(gains, lower, upper), lower, upper, limits - SLSQP_MARGIN)
    start = projection.x if projection.success else np.zeros_like(gains)
    result = minimize(objective, start, method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-7, "maxiter": 120, "disp": False})
    return result, objective, limits


def test_gain_refinement_matches_or_beats_slsqp_and_survives_its_failures():
    """The NumPy solve against the SLSQP solve it replaced, on one seeded set of problems.

    SLSQP is only run from the exact projection the app starts from.  Roughly half of this set
    makes it stop with 'positive directional derivative' or report success on a point that
    breaks the adjacency limit; the app discards those solves, the NumPy solve must not.
    """
    pytest.importorskip("scipy.optimize")
    import warnings

    from mic_eq.analysis.auto_eq_parts.optimizer import _constrained_gain_refinement

    rng = np.random.default_rng(20261014)
    slsqp_failures = both_solved = 0
    worst_excess = 0.0
    for case in range(24):
        problem = _fit_problem(rng, bands=8, saturate=12.0 if case % 6 == 0 else 30.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference, objective, limits = _slsqp_refinement(problem)
        lower, upper = problem[7], problem[8]
        reference_ok = bool(reference.success) and _feasible(reference.x, lower, upper, limits)
        gains, success = _constrained_gain_refinement(*problem, None, 48000.0)

        assert _feasible(gains, lower, upper, limits), case
        if success:
            assert np.isfinite(objective(gains)), case
        if reference_ok:
            both_solved += 1
            assert success, case
            excess = (objective(gains) - objective(reference.x)) / max(abs(objective(reference.x)), 1.0)
            worst_excess = max(worst_excess, excess)
            assert excess <= 1e-5, (case, excess)
        else:
            slsqp_failures += 1
            assert success, case  # the NumPy solve converged where SLSQP did not

    assert slsqp_failures >= 5, "the problem set no longer provokes SLSQP failures"
    assert both_solved >= 3, "the problem set no longer exercises the SLSQP-succeeds path"


def test_gain_projection_is_feasible_and_no_further_than_slsqp():
    """Alternating +/-12 dB starts force many adjacency constraints active at once.

    SLSQP's projection converges on every start here (its iteration-cap failures only showed up
    on the app's own fits), so this checks feasibility and distance, not failure handling.
    """
    scipy_optimize = pytest.importorskip("scipy.optimize")
    import warnings

    rng = np.random.default_rng(20261015)
    for case in range(40):
        bands = 10
        centers = np.sort(np.geomspace(60, 14000, bands))
        limits = np.minimum(6.0, 12.0 * np.maximum(np.diff(np.log2(centers)), 1e-6)) - SLSQP_MARGIN
        lower = np.full(bands, -12.0)
        upper = np.full(bands, 12.0)
        start = rng.choice([-12.0, 12.0], bands)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference = scipy_optimize.minimize(
                lambda g: float(np.sum(np.square(g - start))), start, method="SLSQP",
                bounds=list(zip(lower, upper, strict=True)),
                constraints=({"type": "ineq", "fun": lambda g: limits - np.abs(np.diff(g))},),
                options={"ftol": 1e-9, "maxiter": 120, "disp": False})
        projection = project_gains(start, lower, upper, limits)
        assert projection.success and _feasible(projection.x, lower, upper, limits), case
        if reference.success:
            assert float(np.sum(np.square(projection.x - start))) <= float(reference.fun) + 1e-6, case
