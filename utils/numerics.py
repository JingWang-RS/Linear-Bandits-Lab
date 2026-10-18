"""Solve the optimization problems used by bandit algorithms and check that the solutions are valid."""

import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint, OptimizeResult, brentq, minimize


def checked_newton(fun, x0, *, jac, hess, operation="convex optimization", tolerance=1e-7):
    """Solve a smooth strictly convex objective, checking its gradient directly.

    Ridge-regularized likelihoods have positive definite Hessians. A damped
    Newton solve avoids declaring convergence from a tiny objective change
    while the score equation is still inaccurate. It also avoids trust-region
    rejection when the predicted improvement is below floating-point precision.
    """
    point = np.asarray(x0, dtype=float).copy()
    for iteration in range(101):
        value = float(fun(point))
        gradient = np.asarray(jac(point), dtype=float)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise RuntimeError(f"{operation}: nonfinite objective or gradient")
        residual = float(np.linalg.norm(gradient, ord=np.inf))
        if residual <= tolerance:
            return OptimizeResult(
                x=point, fun=value, jac=gradient, success=True, status=0, nit=iteration
            )
        if iteration == 100:
            break
        curvature = np.asarray(hess(point), dtype=float)
        if not np.isfinite(curvature).all():
            raise RuntimeError(f"{operation}: nonfinite Hessian")
        try:
            np.linalg.cholesky(curvature)
            step = np.linalg.solve(curvature, gradient)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"{operation}: Hessian is not positive definite") from exc
        decrease = float(gradient @ step)
        if not np.isfinite(decrease) or decrease <= 0:
            raise RuntimeError(f"{operation}: invalid Newton direction")
        scale = 1.0
        # Permit objective rounding at the optimum; the gradient check above
        # remains the convergence criterion and is never relaxed.
        rounding = 16 * np.finfo(float).eps * max(1.0, abs(value))
        for _ in range(60):
            candidate = point - scale * step
            candidate_value = float(fun(candidate))
            if np.isfinite(candidate_value) and candidate_value <= (
                value - 1e-4 * scale * decrease + rounding
            ):
                point = candidate
                break
            scale *= 0.5
        else:
            raise RuntimeError(f"{operation}: Newton line search failed")
    raise RuntimeError(f"{operation}: gradient residual {residual:.3g} exceeds {tolerance:g}")


def _violation(constraints, x):
    maximum = 0.0
    for constraint in constraints:
        if isinstance(constraint, dict):
            values = np.atleast_1d(constraint["fun"](x))
            residual = np.abs(values) if constraint["type"] == "eq" else -values
        elif isinstance(constraint, (LinearConstraint, NonlinearConstraint)):
            values = (
                constraint.A @ x if isinstance(constraint, LinearConstraint) else constraint.fun(x)
            )
            values = np.atleast_1d(values)
            residual = np.concatenate(
                [
                    np.atleast_1d(constraint.lb - values),
                    np.atleast_1d(values - constraint.ub),
                ]
            )
        else:
            raise TypeError("unsupported optimization constraint type")
        if not np.isfinite(np.asarray(values)).all():
            return np.inf
        maximum = max(maximum, float(np.max(residual)))
    return maximum


def checked_minimize(
    fun,
    x0,
    *,
    jac=None,
    hess=None,
    constraints=(),
    bounds=None,
    method="SLSQP",
    options=None,
    operation="optimization",
    constraint_violation=None,
):
    """Reject failed or infeasible solver results; never substitute an estimator."""
    constraints = (
        (constraints,)
        if isinstance(constraints, (dict, LinearConstraint, NonlinearConstraint))
        else tuple(constraints)
    )
    settings = {"ftol": 1e-10, "maxiter": 500} if method == "SLSQP" else {}
    settings.update(options or {})
    result = minimize(
        fun,
        np.asarray(x0, dtype=float),
        jac=jac,
        hess=hess,
        constraints=constraints,
        bounds=bounds,
        method=method,
        options=settings,
    )
    if not result.success:
        raise RuntimeError(
            f"{operation}: solver {method} failed (status={result.status}, "
            f"iterations={getattr(result, 'nit', None)}): {result.message}"
        )
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise RuntimeError(f"{operation}: solver returned a nonfinite result")
    violation = _violation(constraints, result.x)
    if constraint_violation is not None:
        custom_residual = float(constraint_violation(result.x))
        if not np.isfinite(custom_residual):
            raise RuntimeError(f"{operation}: nonfinite constraint residual")
        violation = max(violation, custom_residual)
    if bounds is not None:
        if hasattr(bounds, "lb"):
            violation = max(
                violation, float(np.max(bounds.lb - result.x)), float(np.max(result.x - bounds.ub))
            )
        else:
            for value, (lower, upper) in zip(result.x, bounds, strict=True):
                if lower is not None:
                    violation = max(violation, float(lower - value))
                if upper is not None:
                    violation = max(violation, float(value - upper))
    if not np.isfinite(violation) or violation > 1e-7:
        raise RuntimeError(f"{operation}: solver constraint residual {violation:.3g} exceeds 1e-7")
    return result


def _metric(metric, size):
    metric = np.asarray(metric, dtype=float)
    if metric.shape != (size, size) or not np.isfinite(metric).all():
        raise ValueError("projection metric must be a finite square matrix")
    if not np.allclose(metric, metric.T, rtol=0.0, atol=1e-12):
        raise ValueError("projection metric must be symmetric")
    np.linalg.cholesky(metric)
    return metric


def project_ball(point, metric, radius):
    """Exact convex ball projection using the monotone scalar KKT multiplier equation."""
    point = np.asarray(point, dtype=float)
    if point.ndim != 1 or not np.isfinite(point).all():
        raise ValueError("projection point must be a finite vector")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("projection radius must be finite and positive")
    metric = _metric(metric, point.size)
    if np.linalg.norm(point) <= radius:
        return point.copy()
    eigenvalues, basis = np.linalg.eigh(metric)
    if np.any(eigenvalues <= 0):
        raise ArithmeticError("projection metric spectrum is not positive")
    coordinates = basis.T @ point
    upper = float(np.linalg.norm(metric @ point) / radius)
    if not np.isfinite(upper) or upper <= 0:
        raise ArithmeticError("projection multiplier scale is not representable")

    def coordinates_at(value):
        return eigenvalues / (eigenvalues + upper * value) * coordinates

    def equation(value):
        return float(np.linalg.norm(coordinates_at(value)) / radius - 1)

    root, status = brentq(equation, 0.0, 1.0, xtol=1e-15, rtol=1e-14, full_output=True)
    if not status.converged:
        raise RuntimeError("metric ball projection: scalar KKT solver did not converge")
    projected = basis @ coordinates_at(root)
    residual = abs(np.linalg.norm(projected) / radius - 1)
    if not np.isfinite(projected).all() or residual > 1e-8:
        raise RuntimeError(f"metric ball projection: normalized constraint residual {residual}")
    return projected
