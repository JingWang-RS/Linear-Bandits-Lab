"""SCB-WeightUCB from TIT'26: Revisiting Weighted Strategy for Non-stationary Parametric Bandits and MDPs.

Also includes LogUCB1 from Improved Optimistic Algorithms for Logistic Bandits
(ICML 2020) and its restarted version."""

import numpy as np
from scipy.optimize import NonlinearConstraint
from scipy.special import expit

from utils.numerics import checked_minimize

from .logistic import LogisticEstimate


class SCBStaticUCB(LogisticEstimate):
    """LogUCB1 from Improved Optimistic Algorithms for Logistic Bandits (ICML 2020)."""

    @property
    def beta(self):
        root = np.sqrt(self.likelihood_regularization)
        return (
            root * (self.parameter_bound + 0.5)
            + 2 / root * np.log(1 / self.delta)
            + self.dimension
            / root
            * np.log(
                4
                * (
                    1
                    + self.k_mu
                    * self.confidence_count
                    / (self.likelihood_regularization * self.dimension)
                )
            )
        )

    @property
    def radius_multiplier(self):
        return np.sqrt(4 + 8 * self.parameter_bound) * self.k_mu / np.sqrt(self.c_mu)

    def projection_objective(self, theta):
        diff = self.gt(theta) - self.gt(self.theta_hat)
        return float(diff @ np.linalg.solve(self.hessian(theta), diff))

    def projection_gradient(self, theta):
        diff = self.gt(theta) - self.gt(self.theta_hat)
        dual = np.linalg.solve(self.hessian(theta), diff)
        xs, _ = self._history()
        p = expit(xs @ theta)
        third = self._weights * p * (1 - p) * (1 - 2 * p)
        return 2 * diff - xs.T @ (third * (xs @ dual) ** 2)

    def _projection(self):
        try:
            return super()._projection()
        except RuntimeError:
            pass
        # SCB's metric depends on theta, so the GLB boundary-only argument does
        # not apply. Solve the original full ball problem with a constrained
        # trust region when SLSQP rejects both of its starting points.
        radius = self.parameter_bound
        scale = max(1.0, self.projection_objective(np.zeros(self.dimension)))
        initial = self.theta_tilde / radius
        initial /= max(1.0, np.linalg.norm(initial))
        constraint = NonlinearConstraint(
            lambda u: u @ u,
            -np.inf,
            1.0,
            jac=lambda u: 2 * u,
            hess=lambda u, multipliers: 2 * multipliers[0] * np.eye(self.dimension),
        )
        for gradient_tolerance, barrier, barrier_tolerance in (
            (1e-13, 1e-9, 1e-10),
            (1e-14, 1e-11, 1e-12),
        ):
            result = checked_minimize(
                lambda u: self.projection_objective(radius * u) / scale,
                initial,
                jac=lambda u: radius * self.projection_gradient(radius * u) / scale,
                constraints=constraint,
                method="trust-constr",
                options={
                    "gtol": gradient_tolerance,
                    "xtol": 1e-12,
                    "barrier_tol": 1e-12,
                    "initial_barrier_parameter": barrier,
                    "initial_barrier_tolerance": barrier_tolerance,
                    "maxiter": 1000,
                },
                operation="SCB reward-space projection trust-region fallback",
            )
            point = result.x
            gradient = radius * self.projection_gradient(radius * point) / scale
            norm_squared = float(point @ point)
            multiplier = (
                max(0.0, -float(gradient @ point) / (2 * norm_squared))
                if norm_squared > 0
                else 0.0
            )
            stationarity = np.linalg.norm(gradient + 2 * multiplier * point, ord=np.inf)
            complementarity = abs(multiplier * (norm_squared - 1))
            if (
                norm_squared <= 1 + 1e-9
                and stationarity <= 1e-7
                and complementarity <= 1e-7
                and np.isfinite(gradient).all()
            ):
                self.projection_count += 1
                self.projection_fallback_count += 1
                return radius * point
            # A solver can meet its own gradient stopping rule before barrier
            # bias is small enough. Refine the same problem; never relax KKT.
            initial = point / max(1.0, np.linalg.norm(point))
        raise RuntimeError(
            "SCB projection failed independent KKT checks "
            f"(norm squared={norm_squared:.12g}, stationarity={stationarity:.3g}, "
            f"complementarity={complementarity:.3g})"
        )


class SCBRestartUCB(SCBStaticUCB):
    """Restart LogUCB1 after H observations; keep the original confidence-bound clock."""

    DEFAULTS = {**SCBStaticUCB.DEFAULTS, "restart_interval": 100}

    def __init__(self, *, restart_interval=100, **kwargs):
        self.restart_interval = self._interval(restart_interval)
        super().__init__(**kwargs)


class SCBWeightUCB(SCBStaticUCB):
    """SCB-WeightUCB: discount older rewards and use the logistic model's curvature."""

    DEFAULTS = {**SCBStaticUCB.DEFAULTS, "gamma": 0.97}

    def __init__(self, *, gamma=0.97, **kwargs):
        self.gamma = self._discount(gamma)
        super().__init__(**kwargs)
