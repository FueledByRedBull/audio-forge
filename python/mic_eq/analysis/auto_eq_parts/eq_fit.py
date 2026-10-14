"""NumPy-only solvers for the Auto-EQ fitter (no SciPy).

Replaces the four ``scipy.optimize`` calls in ``optimizer.py``:

* :func:`bounded_least_squares` - a trimmed port of ``scipy.optimize.least_squares`` for the
  exact option set the app uses: ``method="trf"``, ``loss="linear"``, ``x_scale=1.0``,
  ``jac="2-point"`` with bounds, ``diff_step=None``, ``tr_solver="exact"`` (dense SVD), finite
  bounds, no callback.  The code follows ``scipy/optimize/_lsq/trf.py`` (``trf_bounds``),
  ``_lsq/common.py``, ``_lsq/least_squares.py`` and the 2-point branch of ``_numdiff.py``
  statement by statement, so the iteration path, ``nfev`` counting, termination tests and
  status codes are SciPy's.
* :func:`project_gains` - exact Euclidean projection of the band gains onto the box plus
  adjacent-difference polytope (was SLSQP, ``ftol=1e-9``).
* :func:`refine_gains` - constrained Gauss-Newton refinement of a sum-of-squares objective
  inside the same polytope (was SLSQP, ``ftol=1e-7``).

Exceptions raised by a callable (for example ``AnalysisCancelled``) propagate unchanged.

The least-squares part is derived from SciPy, which is distributed under this licence:

    Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The projection and refinement solvers are original code (a Goldfarb-Idnani dual active-set QP
for the identity Hessian and a Gauss-Newton/SQP loop around it).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import copysign
from typing import Any

import numpy as np
from numpy.linalg import norm

EPS = np.finfo(float).eps

TERMINATION_MESSAGES = {
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`ftol` termination condition is satisfied.",
    3: "`xtol` termination condition is satisfied.",
    4: "Both `ftol` and `xtol` termination conditions are satisfied.",
}


# ---------------------------------------------------------------------------------------
# scipy/optimize/_lsq/common.py (the parts trf_bounds with the exact solver uses)
# ---------------------------------------------------------------------------------------


def intersect_trust_region(x, s, Delta):
    a = np.dot(s, s)
    if a == 0:
        raise ValueError("`s` is zero.")

    b = np.dot(x, s)

    c = np.dot(x, x) - Delta**2
    if c > 0:
        raise ValueError("`x` is not within the trust region.")

    d = np.sqrt(b * b - a * c)  # Root from one fourth of the discriminant.

    # Computations below avoid loss of significance, see "Numerical Recipes".
    q = -(b + copysign(d, b))
    t1 = q / a
    t2 = c / q

    if t1 < t2:
        return t1, t2
    else:
        return t2, t1


def solve_lsq_trust_region(n, m, uf, s, V, Delta, initial_alpha=None, rtol=0.01, max_iter=10):
    def phi_and_derivative(alpha, suf, s, Delta):
        denom = s**2 + alpha
        p_norm = norm(suf / denom)
        phi = p_norm - Delta
        phi_prime = -np.sum(suf**2 / denom**3) / p_norm
        return phi, phi_prime

    suf = s * uf

    # Check if J has full rank and try Gauss-Newton step.
    if m >= n:
        threshold = EPS * m * s[0]
        full_rank = s[-1] > threshold
    else:
        full_rank = False

    if full_rank:
        p = -V.dot(uf / s)
        if norm(p) <= Delta:
            return p, 0.0, 0

    alpha_upper = norm(suf) / Delta

    if full_rank:
        phi, phi_prime = phi_and_derivative(0.0, suf, s, Delta)
        alpha_lower = -phi / phi_prime
    else:
        alpha_lower = 0.0

    if initial_alpha is None or not full_rank and initial_alpha == 0:
        alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper) ** 0.5)
    else:
        alpha = initial_alpha

    it = 0
    for it in range(max_iter):
        if alpha < alpha_lower or alpha > alpha_upper:
            alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper) ** 0.5)

        phi, phi_prime = phi_and_derivative(alpha, suf, s, Delta)

        if phi < 0:
            alpha_upper = alpha

        ratio = phi / phi_prime
        alpha_lower = max(alpha_lower, alpha - ratio)
        alpha -= (phi + Delta) * ratio / Delta

        if np.abs(phi) < rtol * Delta:
            break

    p = -V.dot(suf / (s**2 + alpha))

    # Make the norm of p equal to Delta, p is changed only slightly during
    # this. It is done to prevent p lie outside the trust region (which can
    # cause problems later).
    p *= Delta / norm(p)

    return p, alpha, it + 1


def update_tr_radius(Delta, actual_reduction, predicted_reduction, step_norm, bound_hit):
    if predicted_reduction > 0:
        ratio = actual_reduction / predicted_reduction
    elif predicted_reduction == actual_reduction == 0:
        ratio = 1
    else:
        ratio = 0

    if ratio < 0.25:
        Delta = 0.25 * step_norm
    elif ratio > 0.75 and bound_hit:
        Delta *= 2.0

    return Delta, ratio


def build_quadratic_1d(J, g, s, diag=None, s0=None) -> tuple[Any, ...]:
    v = J.dot(s)
    a = np.dot(v, v)
    if diag is not None:
        a += np.dot(s * diag, s)
    a *= 0.5

    b = np.dot(g, s)

    if s0 is not None:
        u = J.dot(s0)
        b += np.dot(u, v)
        c = 0.5 * np.dot(u, u) + np.dot(g, s0)
        if diag is not None:
            b += np.dot(s0 * diag, s)
            c += 0.5 * np.dot(s0 * diag, s0)
        return a, b, c
    else:
        return a, b


def minimize_quadratic_1d(a, b, lb, ub, c=0):
    t = [lb, ub]
    if a != 0:
        extremum = -0.5 * b / a
        if lb < extremum < ub:
            t.append(extremum)
    t = np.asarray(t)
    y = t * (a * t + b) + c
    min_index = np.argmin(y)
    return t[min_index], y[min_index]


def evaluate_quadratic(J, g, s, diag=None):
    if s.ndim == 1:
        Js = J.dot(s)
        q = np.dot(Js, Js)
        if diag is not None:
            q += np.dot(s * diag, s)
    else:
        Js = J.dot(s.T)
        q = np.sum(Js**2, axis=0)
        if diag is not None:
            q += np.sum(diag * s**2, axis=1)

    linear = np.dot(s, g)

    return 0.5 * q + linear


def in_bounds(x, lb, ub):
    return np.all((x >= lb) & (x <= ub))


def step_size_to_bound(x, s, lb, ub):
    non_zero = np.nonzero(s)
    s_non_zero = s[non_zero]
    steps = np.empty_like(x)
    steps.fill(np.inf)
    with np.errstate(over="ignore"):
        steps[non_zero] = np.maximum((lb - x)[non_zero] / s_non_zero, (ub - x)[non_zero] / s_non_zero)
    min_step = np.min(steps)
    return min_step, np.equal(steps, min_step) * np.sign(s).astype(int)


def find_active_constraints(x, lb, ub, rtol=1e-10):
    active = np.zeros_like(x, dtype=int)

    if rtol == 0:
        active[x <= lb] = -1
        active[x >= ub] = 1
        return active

    lower_dist = x - lb
    upper_dist = ub - x

    lower_threshold = rtol * np.maximum(1, np.abs(lb))
    upper_threshold = rtol * np.maximum(1, np.abs(ub))

    lower_active = np.isfinite(lb) & (lower_dist <= np.minimum(upper_dist, lower_threshold))
    active[lower_active] = -1

    upper_active = np.isfinite(ub) & (upper_dist <= np.minimum(lower_dist, upper_threshold))
    active[upper_active] = 1

    return active


def make_strictly_feasible(x, lb, ub, rstep=1e-10):
    x_new = x.copy()

    active = find_active_constraints(x, lb, ub, rstep)
    lower_mask = np.equal(active, -1)
    upper_mask = np.equal(active, 1)

    if rstep == 0:
        x_new[lower_mask] = np.nextafter(lb[lower_mask], ub[lower_mask])
        x_new[upper_mask] = np.nextafter(ub[upper_mask], lb[upper_mask])
    else:
        x_new[lower_mask] = lb[lower_mask] + rstep * np.maximum(1, np.abs(lb[lower_mask]))
        x_new[upper_mask] = ub[upper_mask] - rstep * np.maximum(1, np.abs(ub[upper_mask]))

    tight_bounds = (x_new < lb) | (x_new > ub)
    x_new[tight_bounds] = 0.5 * (lb[tight_bounds] + ub[tight_bounds])

    return x_new


def CL_scaling_vector(x, g, lb, ub):
    v = np.ones_like(x)
    dv = np.zeros_like(x)

    mask = (g < 0) & np.isfinite(ub)
    v[mask] = ub[mask] - x[mask]
    dv[mask] = -1

    mask = (g > 0) & np.isfinite(lb)
    v[mask] = x[mask] - lb[mask]
    dv[mask] = 1

    return v, dv


def check_termination(dF, F, dx_norm, x_norm, ratio, ftol, xtol):
    ftol_satisfied = dF < ftol * F and ratio > 0.25
    xtol_satisfied = dx_norm < xtol * (xtol + x_norm)

    if ftol_satisfied and xtol_satisfied:
        return 4
    elif ftol_satisfied:
        return 2
    elif xtol_satisfied:
        return 3
    else:
        return None


# ---------------------------------------------------------------------------------------
# scipy/optimize/_numdiff.py: approx_derivative(method="2-point", bounds=...)
# ---------------------------------------------------------------------------------------


def _adjust_scheme_to_bounds_one_sided(x0, h, lb, ub):
    """``_adjust_scheme_to_bounds(x0, h, 1, "1-sided", lb, ub)`` without the unused outputs."""
    if np.all((lb == -np.inf) & (ub == np.inf)):
        return h

    h_total = h * 1
    h_adjusted = h.copy()

    lower_dist = x0 - lb
    upper_dist = ub - x0

    x = x0 + h_total
    violated = (x < lb) | (x > ub)
    fitting = np.abs(h_total) <= np.maximum(lower_dist, upper_dist)
    h_adjusted[violated & fitting] *= -1

    forward = (upper_dist >= lower_dist) & ~fitting
    h_adjusted[forward] = upper_dist[forward] / 1
    backward = (upper_dist < lower_dist) & ~fitting
    h_adjusted[backward] = -lower_dist[backward] / 1

    return h_adjusted


def _two_point_jacobian(fun, x0, f0, lb, ub):
    sign_x0 = (x0 >= 0).astype(x0.dtype) * 2 - 1
    rstep = EPS**0.5
    h = (rstep * sign_x0 * np.maximum(1.0, np.abs(x0))).astype(x0.dtype)
    h = _adjust_scheme_to_bounds_one_sided(x0, h, lb, ub)

    n = x0.size
    J_transposed = np.empty((n, f0.size), dtype=np.result_type(x0, f0, np.float64))

    def x_generator(x0, h):
        for i in range(n):
            x1 = np.copy(x0)
            x1[i] = x0[i] + h[i]
            yield x1

    f_evals = map(fun, x_generator(x0, h))
    dx = [(x0[i] + h[i]) - x0[i] for i in range(n)]
    df = [np.atleast_1d(f_eval) - f0 for f_eval in f_evals]
    df_dx = [delf / delx for delf, delx in zip(df, dx)]

    for i, v in enumerate(df_dx):
        J_transposed[i] = v

    if f0.size == 1:
        J_transposed = np.ravel(J_transposed)

    # SciPy returns the transposed view (Fortran order) and copies it with ``astype``; the
    # memory layout decides BLAS summation order, so it is reproduced.
    return np.atleast_2d(J_transposed.T).astype(float)


# ---------------------------------------------------------------------------------------
# scipy/optimize/_lsq/trf.py
# ---------------------------------------------------------------------------------------


def select_step(x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta):
    """Select the best step according to Trust Region Reflective algorithm."""
    if in_bounds(x + p, lb, ub):
        p_value = evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)
        return p, p_h, -p_value

    p_stride, hits = step_size_to_bound(x, p, lb, ub)

    # Compute the reflected direction.
    r_h = np.copy(p_h)
    r_h[hits.astype(bool)] *= -1
    r = d * r_h

    # Restrict trust-region step, such that it hits the bound.
    p *= p_stride
    p_h *= p_stride
    x_on_bound = x + p

    # Reflected direction will cross first either feasible region or trust
    # region boundary.
    _, to_tr = intersect_trust_region(p_h, r_h, Delta)
    to_bound, _ = step_size_to_bound(x_on_bound, r, lb, ub)

    # Find lower and upper bounds on a step size along the reflected
    # direction, considering the strict feasibility requirement.
    r_stride = min(to_bound, to_tr)
    if r_stride > 0:
        r_stride_l = (1 - theta) * p_stride / r_stride
        if r_stride == to_bound:
            r_stride_u = theta * to_bound
        else:
            r_stride_u = to_tr
    else:
        r_stride_l = 0
        r_stride_u = -1

    # Check if reflection step is available.
    if r_stride_l <= r_stride_u:
        a, b, c = build_quadratic_1d(J_h, g_h, r_h, s0=p_h, diag=diag_h)
        r_stride, r_value = minimize_quadratic_1d(a, b, r_stride_l, r_stride_u, c=c)
        r_h *= r_stride
        r_h += p_h
        r = r_h * d
    else:
        r_value = np.inf

    # Now correct p_h to make it strictly interior.
    p *= theta
    p_h *= theta
    p_value = evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)

    ag_h = -g_h
    ag = d * ag_h

    to_tr = Delta / norm(ag_h)
    to_bound, _ = step_size_to_bound(x, ag, lb, ub)
    if to_bound < to_tr:
        ag_stride = theta * to_bound
    else:
        ag_stride = to_tr

    a, b = build_quadratic_1d(J_h, g_h, ag_h, diag=diag_h)
    ag_stride, ag_value = minimize_quadratic_1d(a, b, 0, ag_stride)
    ag_h *= ag_stride
    ag *= ag_stride

    if p_value < r_value and p_value < ag_value:
        return p, p_h, -p_value
    elif r_value < p_value and r_value < ag_value:
        return r, r_h, -r_value
    else:
        return ag, ag_h, -ag_value


@dataclass
class LeastSquaresResult:
    """The fields of SciPy's ``OptimizeResult`` that callers of ``least_squares`` use."""

    x: np.ndarray
    cost: float
    fun: np.ndarray
    jac: np.ndarray
    grad: np.ndarray
    optimality: float
    active_mask: np.ndarray
    nfev: int
    njev: int
    status: int
    message: str
    success: bool


def bounded_least_squares(
    fun: Callable[..., Any],
    x0: Any,
    bounds: tuple[Any, Any],
    *,
    args: tuple = (),
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    max_nfev: int | None = None,
    x_scale: Any = None,
) -> LeastSquaresResult:
    """``least_squares(fun, x0, args=args, bounds=bounds, method="trf", ...)`` without SciPy.

    ``fun(x, *args)`` returns the residual vector.  Fixed here (the options optimizer.py uses):
    linear loss, 2-point finite-difference Jacobian honouring the bounds, exact (SVD)
    trust-region solver, ``x_scale`` 1 (an array of positive scales is accepted), at least one
    finite bound.  ``x_scale="jac"``, robust losses, callbacks and sparse Jacobians are not
    ported.  Exceptions from ``fun`` propagate unchanged.
    """
    if max_nfev is not None and max_nfev <= 0:
        raise ValueError("`max_nfev` must be None or positive integer.")

    x0 = np.atleast_1d(np.asarray(x0)).astype(float)
    if x0.ndim > 1:
        raise ValueError("`x0` must have at most 1 dimension.")

    if len(bounds) != 2:
        raise ValueError("`bounds` must contain 2 elements.")
    lb, ub = (np.asarray(b, dtype=float) for b in bounds)
    if lb.ndim == 0:
        lb = np.resize(lb, x0.size)
    if ub.ndim == 0:
        ub = np.resize(ub, x0.size)
    if lb.shape != x0.shape or ub.shape != x0.shape:
        raise ValueError("Inconsistent shapes between bounds and `x0`.")
    if np.any(lb >= ub):
        raise ValueError("Each lower bound must be strictly less than each upper bound.")
    if not in_bounds(x0, lb, ub):
        raise ValueError("Initial guess is outside of provided bounds")
    if np.all(lb == -np.inf) and np.all(ub == np.inf):
        raise ValueError("At least one finite bound is required (unbounded trf is not ported).")

    if x_scale is None:
        x_scale = 1.0
    try:
        x_scale = np.asarray(x_scale, dtype=float)
        valid = bool(np.all(np.isfinite(x_scale)) and np.all(x_scale > 0))
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError("`x_scale` must be array_like with positive numbers.")
    if x_scale.ndim == 0:
        x_scale = np.resize(x_scale, x0.shape)
    if x_scale.shape != x0.shape:
        raise ValueError("Inconsistent shapes between `x_scale` and `x0`.")

    ftol, xtol, gtol = (0 if tol is None else tol for tol in (ftol, xtol, gtol))
    if ftol < EPS and xtol < EPS and gtol < EPS:
        raise ValueError(
            f"At least one of the tolerances must be higher than machine epsilon ({EPS:.2e})."
        )

    x0 = make_strictly_feasible(x0, lb, ub)

    def residuals(x):
        return np.atleast_1d(fun(np.copy(x), *args))

    f0 = residuals(x0)
    if f0.ndim != 1:
        raise ValueError(f"`fun` must return at most 1-d array_like. f0.shape: {f0.shape}")
    if not np.all(np.isfinite(f0)):
        raise ValueError("Residuals are not finite in the initial point.")
    J0 = _two_point_jacobian(residuals, x0, f0, lb, ub)

    def jac(x, f):
        return _two_point_jacobian(residuals, x, f, lb, ub)

    return _trf_bounds(residuals, jac, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev, x_scale)


def _trf_bounds(fun, jac, x0, f0, J0, lb, ub, ftol, xtol, gtol, max_nfev, x_scale):
    x = x0.copy()

    f = f0
    f_true = f.copy()
    nfev = 1

    J = J0
    njev = 1
    m, n = J.shape

    cost = 0.5 * np.dot(f, f)

    g = J.T.dot(f)

    scale, scale_inv = x_scale, 1 / x_scale

    v, dv = CL_scaling_vector(x, g, lb, ub)
    v[dv != 0] *= scale_inv[dv != 0]
    Delta = norm(x0 * scale_inv / v**0.5)
    if Delta == 0:
        Delta = 1.0

    g_norm = norm(g * v, ord=np.inf)

    f_augmented = np.zeros(m + n)
    J_augmented = np.empty((m + n, n))

    if max_nfev is None:
        max_nfev = x0.size * 100

    alpha = 0.0  # "Levenberg-Marquardt" parameter
    x_new, f_new, cost_new = x, f, cost  # always replaced before use (keeps type checkers calm)

    termination_status = None
    actual_reduction = None

    while True:
        v, dv = CL_scaling_vector(x, g, lb, ub)

        g_norm = norm(g * v, ord=np.inf)
        if g_norm < gtol:
            termination_status = 1

        if termination_status is not None or nfev == max_nfev:
            break

        # v is recomputed in the variables after applying `x_scale`.
        v[dv != 0] *= scale_inv[dv != 0]

        # Here, we apply two types of scaling.
        d = v**0.5 * scale

        # C = diag(g * scale) Jv
        diag_h = g * dv * scale

        # "hat" gradient.
        g_h = d * g

        f_augmented[:m] = f
        J_augmented[:m] = J * d
        J_h = J_augmented[:m]  # Memory view.
        J_augmented[m:] = np.diag(diag_h**0.5)
        # numpy.linalg.svd returns (U, s, Vh) exactly like scipy.linalg.svd, which also rejects a
        # non-finite matrix first (the same ValueError).
        U, s, V = np.linalg.svd(np.asarray_chkfinite(J_augmented), full_matrices=False)
        V = V.T
        uf = U.T.dot(f_augmented)

        # theta controls step back step ratio from the bounds.
        theta = max(0.995, 1 - g_norm)

        actual_reduction = -1
        while actual_reduction <= 0 and nfev < max_nfev:
            p_h, alpha, n_iter = solve_lsq_trust_region(n, m, uf, s, V, Delta, initial_alpha=alpha)

            p = d * p_h  # Trust-region solution in the original space.
            step, step_h, predicted_reduction = select_step(
                x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta
            )

            x_new = make_strictly_feasible(x + step, lb, ub, rstep=0)
            f_new = fun(x_new)
            nfev += 1

            step_h_norm = norm(step_h)

            if not np.all(np.isfinite(f_new)):
                Delta = 0.25 * step_h_norm
                continue

            # Usual trust-region step quality estimation.
            cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new
            Delta_new, ratio = update_tr_radius(
                Delta, actual_reduction, predicted_reduction, step_h_norm, step_h_norm > 0.95 * Delta
            )

            step_norm = norm(step)
            termination_status = check_termination(
                actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol
            )
            if termination_status is not None:
                break

            alpha *= Delta / Delta_new
            Delta = Delta_new

        if actual_reduction > 0:
            x = x_new

            f = f_new
            f_true = f.copy()

            cost = cost_new

            J = jac(x, f)
            njev += 1

            g = J.T.dot(f)

    if termination_status is None:
        termination_status = 0

    active_mask = find_active_constraints(x, lb, ub, rtol=xtol)
    return LeastSquaresResult(
        x=x,
        cost=cost,
        fun=f_true,
        jac=J,
        grad=g,
        optimality=g_norm,
        active_mask=active_mask,
        nfev=nfev,
        njev=njev,
        status=termination_status,
        message=TERMINATION_MESSAGES[termination_status],
        success=termination_status > 0,
    )


# ---------------------------------------------------------------------------------------
# Polytope projection: Goldfarb-Idnani dual active-set method with an identity Hessian.
# ---------------------------------------------------------------------------------------

_FEASIBILITY_TOLERANCE = 1e-12


def _project_onto_polytope(a, normals, bounds):
    """Euclidean projection of ``a`` onto ``{x : normals @ x <= bounds}`` (rows of unit length).

    Returns ``None`` when the set is empty or the iteration limit is hit (degenerate cycling).
    Deterministic; exact to rounding error.
    """
    n = a.size
    count = bounds.size
    x = a.astype(float).copy()
    active: list[int] = []
    u: list[float] = []
    max_outer = 50 * (count + n) + 100

    for _ in range(max_outer):
        slack = bounds - normals @ x
        if active:
            slack[active] = np.inf
        p = int(np.argmin(slack)) if count else -1
        if p < 0 or slack[p] >= -_FEASIBILITY_TOLERANCE:
            return x
        # Goldfarb-Idnani uses constraints c.x + d >= 0, i.e. c = -normal, d = bound.
        np_vec = -normals[p]
        u_p = 0.0

        for _ in range(max_outer):
            q = len(active)
            if q:
                columns = -normals[active].T  # n x q
                q_mat, r_full = np.linalg.qr(columns, mode="complete")
                r_mat = r_full[:q, :q]
            else:
                q_mat = np.eye(n)
                r_mat = np.zeros((0, 0))
            d = q_mat.T @ np_vec
            z = q_mat[:, q:] @ d[q:]
            if q:
                if np.min(np.abs(np.diag(r_mat))) < 1e-14:
                    return None
                r = np.linalg.solve(r_mat, d[:q])
            else:
                r = np.zeros(0)

            s_p = bounds[p] - normals[p] @ x
            t1 = np.inf
            drop_index = None
            for k in range(q):
                if r[k] > 0:
                    t = u[k] / r[k]
                    if t < t1:
                        t1 = t
                        drop_index = k
            zz = float(z @ z)
            t2 = -s_p / zz if zz > 1e-18 else np.inf
            t = min(t1, t2)
            if not np.isfinite(t):
                return None

            if np.isfinite(t2):
                x = x + t * z
            for k in range(q):
                u[k] = max(u[k] - t * r[k], 0.0)
            u_p += t

            if t2 <= t1:
                active.append(p)
                u.append(u_p)
                break
            if drop_index is not None:
                del active[drop_index]
                del u[drop_index]
        else:
            return None
        if p not in active:
            return None
    return None


@dataclass
class GainSolve:
    """Outcome of :func:`project_gains` / :func:`refine_gains`."""

    x: np.ndarray
    #: True when the solver converged to the constrained optimum within its limits.
    success: bool
    iterations: int = 0
    #: Sum of squared residuals at ``x`` (refinement only; NaN for the projection).
    cost: float = float("nan")
    message: str = field(default="")


class _GainLayout:
    """Box bounds plus ``|g[i+1] - g[i]| <= adjacent_limit[i]`` over the free variables.

    Variables with ``lower == upper`` are fixed (SciPy's ``minimize`` also removed them before
    calling SLSQP); the rest are the free variables.
    """

    def __init__(self, lower, upper, adjacent_limit):
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        limit = np.asarray(adjacent_limit, dtype=float)
        n = lower.size
        if lower.ndim != 1 or n == 0 or upper.shape != lower.shape or limit.shape != (n - 1,):
            raise ValueError("gain bounds and adjacent limits have inconsistent lengths")
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or not np.all(np.isfinite(limit)):
            raise ValueError("gain bounds must not contain NaN and limits must be finite")
        if np.any(lower > upper):
            raise ValueError("gain lower bound exceeds upper bound")
        self.lower, self.upper, self.limit, self.n = lower, upper, limit, n
        self.free = np.flatnonzero(upper > lower)
        self.fixed_value = lower.copy()
        self._rows: tuple[np.ndarray, np.ndarray] | None = None
        self._rows_built = False

    def expand(self, free_values):
        full = self.fixed_value.copy()
        full[self.free] = free_values
        return full

    def rows(self):
        """``(normals, bounds)`` over the free variables, or ``None`` when two fixed neighbours
        already violate their adjacency limit (empty polytope)."""
        if self._rows_built:
            return self._rows
        nf = self.free.size
        position = np.full(self.n, -1)
        position[self.free] = np.arange(nf)
        normals: list[np.ndarray] = []
        bounds: list[float] = []

        def add(vector, bound):
            length = norm(vector)
            normals.append(vector / length)
            bounds.append(bound / length)

        for k, i in enumerate(self.free):
            e = np.zeros(nf)
            e[k] = 1.0
            if np.isfinite(self.upper[i]):
                add(e.copy(), self.upper[i])
            if np.isfinite(self.lower[i]):
                add(-e, -self.lower[i])
        for i in range(self.n - 1):
            limit = self.limit[i]
            lo, hi = position[i], position[i + 1]
            for sign in (1.0, -1.0):  # sign * (g[i+1] - g[i]) <= limit
                e = np.zeros(nf)
                if lo >= 0 and hi >= 0:
                    e[hi], e[lo] = sign, -sign
                    add(e, limit)
                elif lo >= 0:  # g[i+1] fixed
                    e[lo] = -sign
                    add(e, limit - sign * self.fixed_value[i + 1])
                elif hi >= 0:  # g[i] fixed
                    e[hi] = sign
                    add(e, limit + sign * self.fixed_value[i])
                elif sign * (self.fixed_value[i + 1] - self.fixed_value[i]) > limit:
                    self._rows, self._rows_built = None, True
                    return None
        self._rows = (np.array(normals).reshape(len(normals), nf), np.array(bounds))
        self._rows_built = True
        return self._rows


def _check_gains(g0, layout):
    g0 = np.asarray(g0, dtype=float)
    if g0.shape != (layout.n,):
        raise ValueError("gain vector length does not match the bounds")
    if not np.all(np.isfinite(g0)):
        raise ValueError("gains must be finite")
    return g0


def project_gains(
    g0, lower, upper, adjacent_limit, callback: Callable[[np.ndarray], Any] | None = None
) -> GainSolve:
    """Exact projection of ``g0`` onto the gain polytope: ``argmin ||g - g0||^2``.

    ``adjacent_limit`` already includes the safety margin (``adjacent_gain_limits - 1e-7``).
    ``callback(x)`` is called once with the result (SciPy called it once per SLSQP iteration;
    the projection is a single step here) unless every variable is fixed, where SciPy skipped it
    too.  ``success`` is False only when the polytope is empty.
    """
    layout = _GainLayout(lower, upper, adjacent_limit)
    g0 = _check_gains(g0, layout)
    start = g0[layout.free]
    rows = layout.rows()
    solved = None if rows is None else _project_onto_polytope(start, *rows)
    if solved is None:
        x, success = layout.expand(start), False
    else:
        x, success = layout.expand(solved), True
    if layout.free.size and callback is not None:
        callback(x.copy())
    return GainSolve(x=x, success=success, iterations=1)


def _refine_jacobian(residual, x, layout, r0, step):
    """Residual Jacobian over the free variables; second-order differences inside the bounds."""
    columns = []
    probe = x.copy()
    for i in layout.free:
        lower_room = x[i] - layout.lower[i]
        upper_room = layout.upper[i] - x[i]
        h = step * max(1.0, abs(x[i]))

        def evaluate(value, i=i):
            probe[i] = value
            try:
                out = np.asarray(residual(probe.copy()), dtype=float)
            finally:
                probe[i] = x[i]
            if out.shape != r0.shape:
                raise ValueError("residual length changed between calls")
            return out

        if lower_room >= h and upper_room >= h:
            columns.append((evaluate(x[i] + h) - evaluate(x[i] - h)) / (2.0 * h))
        else:
            direction = 1.0 if upper_room >= lower_room else -1.0
            h = min(h, 0.5 * max(lower_room, upper_room))
            if not h > 0.0:
                columns.append(np.zeros_like(r0))
                continue
            first = evaluate(x[i] + direction * h)
            second = evaluate(x[i] + direction * 2.0 * h)
            columns.append(direction * (-3.0 * r0 + 4.0 * first - second) / (2.0 * h))
    return np.array(columns).T if columns else np.zeros((r0.size, 0))


def refine_gains(
    residual: Callable[[np.ndarray], Any],
    g0,
    lower,
    upper,
    adjacent_limit,
    callback: Callable[[np.ndarray], Any] | None = None,
    *,
    max_iter: int = 120,
    x_tol: float = 1e-8,
    fd_step: float = 1e-5,
) -> GainSolve:
    """Minimise ``||residual(g)||^2`` over the gain polytope, starting from ``g0``.

    ``residual(g)`` returns the stacked residual vector (weighted error, curvature and tilt
    terms); the objective is ``residual(g) @ residual(g)``.  Gauss-Newton with a Levenberg ridge,
    an exact QP (the projection routine in a Cholesky-transformed space) for the constrained
    step, and a halving line search on the cost.  ``success`` is False when the iteration cap is
    reached, no descent step exists or the polytope is empty; "no free variables" is success when
    the fixed values satisfy the adjacency limits.  ``callback(x)`` runs after each accepted
    step.  Exceptions from ``residual`` or ``callback`` propagate unchanged.
    """
    layout = _GainLayout(lower, upper, adjacent_limit)
    g0 = _check_gains(g0, layout)
    free = layout.free
    nf = free.size

    def failure(x, iterations, cost=float("nan")):
        return GainSolve(x=x, success=False, iterations=iterations, cost=cost)

    rows = layout.rows()
    if rows is None:
        return failure(g0.copy(), 0)
    normals, row_bounds = rows
    start = g0[free]
    free_x = _project_onto_polytope(start, normals, row_bounds)
    if free_x is None:
        return failure(g0.copy(), 0)

    x = layout.expand(free_x)
    r = np.asarray(residual(x.copy()), dtype=float)
    cost = float(r @ r)
    if not np.isfinite(cost):
        raise ValueError("objective is not finite at the starting point")
    if nf == 0:
        return GainSolve(x=x, success=True, iterations=0, cost=cost)

    ridge_scale = 1e-12
    for iteration in range(1, max_iter + 1):
        jac = _refine_jacobian(residual, x, layout, r, fd_step)
        gradient = jac.T @ r
        hessian = jac.T @ jac
        trace_scale = max(float(np.trace(hessian)) / nf, 1.0)
        slacks = row_bounds - normals @ free_x

        ridge = ridge_scale * trace_scale
        accepted = None
        for _attempt in range(30):
            try:
                chol = np.linalg.cholesky(hessian + ridge * np.eye(nf))
            except np.linalg.LinAlgError:
                ridge = max(ridge * 10.0, 1e-10 * trace_scale)
                continue
            # d = L^-T y; minimise 0.5 |y - a|^2 with a = -L^-1 gradient.
            a = -np.linalg.solve(chol, gradient)
            transformed = np.linalg.solve(chol, normals.T).T
            lengths = norm(transformed, axis=1)
            keep = lengths > 0
            y_normals = transformed[keep] / lengths[keep, None]
            y_bounds = np.maximum(slacks[keep], 0.0) / lengths[keep]
            y = _project_onto_polytope(a, y_normals, y_bounds)
            if y is None:
                return failure(x, iteration, cost)
            d = np.linalg.solve(chol.T, y)
            step_inf = float(np.max(np.abs(d)))
            if step_inf < x_tol:
                return GainSolve(x=x, success=True, iterations=iteration, cost=cost)

            # Halving line search on the cost; the segment stays feasible (convexity).
            t = 1.0
            while t >= 1e-10:
                trial = np.clip(free_x + t * d, layout.lower[free], layout.upper[free])
                trial_r = np.asarray(residual(layout.expand(trial)), dtype=float)
                trial_cost = float(trial_r @ trial_r)
                if np.isfinite(trial_cost) and trial_cost < cost:
                    accepted = (trial, trial_r, trial_cost, t * step_inf)
                    break
                t *= 0.5
            if accepted is not None:
                break
            ridge = max(ridge * 10.0, 1e-8 * trace_scale)
        if accepted is None:
            return failure(x, iteration, cost)
        free_x, r, cost, moved = accepted
        x = layout.expand(free_x)
        if callback is not None:
            callback(x.copy())
        if moved < x_tol:
            return GainSolve(x=x, success=True, iterations=iteration, cost=cost)
    return failure(x, max_iter, cost)
