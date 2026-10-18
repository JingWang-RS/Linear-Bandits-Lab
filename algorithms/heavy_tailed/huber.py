"""Huber loss and regression routines used to reduce the influence of extreme rewards."""

import numpy as np
from scipy.optimize import minimize


def huber_loss(residual, threshold):
    """Huber's piecewise loss, including the zero-feature threshold +infinity."""
    residual, threshold = np.broadcast_arrays(
        np.asarray(residual, dtype=float), np.asarray(threshold, dtype=float)
    )
    values = np.empty_like(residual)
    quadratic = np.abs(residual) <= threshold
    values[quadratic] = residual[quadratic] ** 2 / 2
    values[~quadratic] = (
        threshold[~quadratic] * np.abs(residual[~quadratic])
        - threshold[~quadratic] ** 2 / 2
    )
    return values


def huber_derivative(residual, threshold):
    """The derivative dictated by the Huber loss, not numerical clipping."""
    residual, threshold = np.broadcast_arrays(
        np.asarray(residual, dtype=float), np.asarray(threshold, dtype=float)
    )
    return np.sign(residual) * np.minimum(np.abs(residual), threshold)


def huber_objective(theta, features, rewards, sigmas, taus, regularization):
    residuals = (rewards - features @ theta) / sigmas
    return float(
        regularization * (theta @ theta) / 2 + np.sum(huber_loss(residuals, taus))
    )


def huber_gradient(theta, features, rewards, sigmas, taus, regularization):
    residuals = (rewards - features @ theta) / sigmas
    return regularization * theta - features.T @ (
        huber_derivative(residuals, taus) / sigmas
    )


def fit_huber_ball(
    initial, features, rewards, sigmas, taus, regularization, radius, operation
):
    """Minimize the exact convex Huber objective over a Euclidean ball.

    SLSQP can report a line-search or iteration failure at the optimum on the
    ball boundary. A failed result is accepted only with an independent
    convex certificate: g@theta + radius*||g|| bounds f(theta)-min(f) globally.
    Its tolerance is 1e-8 in objective units plus floating-point roundoff,
    not a relative tolerance scaled by possibly enormous squared rewards.
    """
    objective = lambda theta: huber_objective(
        theta, features, rewards, sigmas, taus, regularization
    )
    gradient = lambda theta: huber_gradient(
        theta, features, rewards, sigmas, taus, regularization
    )
    constraints = {
        "type": "ineq",
        "fun": lambda theta: radius**2 - theta @ theta,
        "jac": lambda theta: -2 * theta,
    }
    failures, candidates = [], []
    for start in (initial, np.zeros_like(initial)):
        result = minimize(
            objective,
            start,
            jac=gradient,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1e-10, "maxiter": 500},
        )
        if np.isfinite(result.x).all() and np.isfinite(result.fun):
            norm = float(np.linalg.norm(result.x))
            if norm <= radius + 1e-7:
                theta = result.x * min(1.0, radius / norm) if norm else result.x.copy()
                grad = gradient(theta)
                scale = radius * float(np.linalg.norm(grad))
                gap = float(grad @ theta + scale)
                tolerance = 1e-8 + 64 * np.finfo(float).eps * (1 + scale)
                if np.isfinite(grad).all() and np.isfinite(gap):
                    if result.success or -tolerance <= gap <= tolerance:
                        result.x = theta
                        result.fun = objective(theta)
                        result.convex_gap = max(0.0, gap)
                        return result
        candidates.append(result)
        failures.append(f"status={result.status}: {result.message}")
    # A failed line search may stop outside the ball. After the normal retries,
    # project its candidate and require the same global convex certificate.
    # This leaves all previously successful fits unchanged.
    for result in candidates:
        norm = float(np.linalg.norm(result.x))
        if not np.isfinite(norm):
            continue
        theta = result.x * min(1.0, radius / norm) if norm else result.x.copy()
        grad = gradient(theta)
        scale = radius * float(np.linalg.norm(grad))
        gap = float(grad @ theta + scale)
        tolerance = 1e-8 + 64 * np.finfo(float).eps * (1 + scale)
        if np.isfinite(grad).all() and np.isfinite(gap) and -tolerance <= gap <= tolerance:
            value = objective(theta)
            if np.isfinite(value):
                result.x, result.fun, result.convex_gap = theta, value, max(0.0, gap)
                return result
    raise RuntimeError(
        f"{operation}: uncertified Huber solution ({'; '.join(failures)})"
    )
