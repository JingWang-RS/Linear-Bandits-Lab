"""GLB-WeightUCB from TIT'26: Revisiting Weighted Strategy for Non-stationary Parametric Bandits and MDPs.

Also includes the GLM-UCB, restarted GLM-UCB, and BVD-GLM-UCB baselines.
Their original paper titles are listed in algorithms/__init__.py."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from utils.numerics import checked_minimize

from .logistic import LogisticEstimate


class GLBStaticUCB(LogisticEstimate):
    """GLM-UCB from Parametric Bandits: The Generalized Linear Case (NeurIPS 2010)."""

    @property
    def beta(self):
        return np.sqrt(
            self.regularization
        ) * self.c_mu * self.parameter_bound + self.noise_scale * np.sqrt(
            2 * np.log(1 / self.delta)
            + self.dimension
            * np.log1p(
                self.confidence_count
                * self.action_bound**2
                / (self.regularization * self.dimension)
            )
        )

    @property
    def radius_multiplier(self):
        return 2 * self.k_mu / self.c_mu

    def projection_objective(self, theta):
        diff = self.gt(theta) - self.gt(self.theta_hat)
        return float(diff @ self.inv_V @ diff)

    def projection_gradient(self, theta):
        diff = self.gt(theta) - self.gt(self.theta_hat)
        # H and inv_V need not commute. The old analytic Jacobian reversed them.
        return 2 * self.hessian(theta) @ self.inv_V @ diff

    def projection_hessian(self, theta):
        hessian = self.hessian(theta)
        dual = self.inv_V @ (self.gt(theta) - self.gt(self.theta_hat))
        xs, _ = self._history()
        p = expit(xs @ theta)
        third = self._weights * p * (1 - p) * (1 - 2 * p)
        return 2 * hessian @ self.inv_V @ hessian + 2 * xs.T @ (
            (third * (xs @ dual))[:, None] * xs
        )

    def _projection(self):
        try:
            return super()._projection()
        except RuntimeError:
            if self.dimension != 2:
                raise
        # For this GLB objective, its unconstrained gradient can vanish only at
        # theta_hat: the likelihood Hessian and inv_V are positive definite.
        # An out-of-ball theta_hat therefore puts every constrained minimum on
        # the circle. This exact coordinate change avoids SLSQP's boundary line
        # search failures, without changing the reward-space objective.
        radius = self.parameter_bound
        scale = max(1.0, self.projection_objective(np.zeros(2)))

        def direction(angle):
            return np.array([np.cos(angle), np.sin(angle)])

        def derivative(angle):
            unit = direction(angle)
            tangent = np.array([-unit[1], unit[0]])
            return float(
                radius * self.projection_gradient(radius * unit) @ tangent / scale
            )

        grid = np.linspace(0, 2 * np.pi, 129)
        derivatives = np.array([derivative(angle) for angle in grid])
        if not np.isfinite(derivatives).all():
            raise RuntimeError("GLB circle projection has nonfinite derivatives")
        candidates = []
        for index in range(len(grid) - 1):
            left, right = derivatives[index : index + 2]
            if left <= 0 <= right:
                angle = brentq(derivative, grid[index], grid[index + 1], xtol=1e-14)
                unit = direction(angle)
                point = radius * unit
                tangent = np.array([-unit[1], unit[0]])
                gradient = self.projection_gradient(point)
                tangential_residual = abs(radius * gradient @ tangent) / scale
                radial_multiplier = -radius * gradient @ unit / scale
                second_derivative = (
                    radius**2 * tangent @ self.projection_hessian(point) @ tangent
                    - radius * gradient @ unit
                ) / scale
                # Independently certify constrained stationarity and local
                # minimality. A failed optimizer result is never accepted.
                if (
                    tangential_residual <= 1e-7
                    and radial_multiplier >= -1e-9
                    and second_derivative >= -1e-9
                    and np.linalg.norm(point) <= radius * (1 + 1e-12)
                ):
                    candidates.append((self.projection_objective(point), point))
        if not candidates:
            raise RuntimeError("GLB circle projection found no feasible KKT minimum")
        value, point = min(candidates, key=lambda item: item[0])
        if not np.isfinite(value):
            raise RuntimeError("GLB circle projection has nonfinite objective")
        self.projection_count += 1
        self.projection_fallback_count = (
            getattr(self, "projection_fallback_count", 0) + 1
        )
        return point


class GLBRestartUCB(GLBStaticUCB):
    """Restart GLM-UCB after H observations; keep the original confidence-bound clock."""

    DEFAULTS = {**GLBStaticUCB.DEFAULTS, "restart_interval": 100}

    def __init__(self, *, restart_interval=100, **kwargs):
        self.restart_interval = self._interval(restart_interval)
        super().__init__(**kwargs)


class GLBWeightUCB(GLBStaticUCB):
    """GLB-WeightUCB: fit logistic rewards while giving older observations less weight."""

    DEFAULTS = {**GLBStaticUCB.DEFAULTS, "gamma": 0.97}

    def __init__(self, *, gamma=0.97, **kwargs):
        self.gamma = self._discount(gamma)
        super().__init__(**kwargs)


class BVDGLMUCB(GLBWeightUCB):
    """BVD-GLM-UCB from Regret Bounds for Generalized Linear Bandits under Parameter Drift (2021).

    ``aistats23`` preserves the archived experiment's update from the newly
    updated V. ``squared-discount`` instead updates the prior auxiliary matrix,
    the intended squared-weight recurrence. These are different variants.
    """

    DEFAULTS = {**GLBWeightUCB.DEFAULTS, "covariance_mode": "aistats23"}

    def __init__(self, *, covariance_mode="aistats23", **kwargs):
        if covariance_mode not in {"aistats23", "squared-discount"}:
            raise ValueError("covariance_mode must be aistats23 or squared-discount")
        self.covariance_mode = covariance_mode
        super().__init__(**kwargs)
        self.resolved_parameters["covariance_mode"] = covariance_mode

    def _update_auxiliary_covariance(self, outer):
        previous = self.V if self.covariance_mode == "aistats23" else self.tilde_V
        self.tilde_V = (
            self.gamma**2 * previous
            + outer
            + (1 - self.gamma**2) * self.regularization * np.eye(self.dimension)
        )

    def _projection(self):
        from scipy.linalg import fractional_matrix_power
        from scipy.optimize import NonlinearConstraint

        d = self.dimension
        np.linalg.cholesky(self.tilde_V)
        square_root = fractional_matrix_power(self.tilde_V, 0.5)
        target = self.gt(self.theta_hat)
        metric = np.linalg.matrix_power(self.inv_V, 2)

        def objective(point):
            diff = (
                self.gt(point[:d])
                + self._projection_beta * (square_root @ point[d:])
                - target
            )
            return float(diff @ (metric @ diff))

        # This joint objective can have many minimizers. Scaling theta by S
        # changes which minimizer SLSQP selects and thus changes later actions.
        # Retain the source coordinates, zero start, norm constraints, numerical
        # Jacobian and stopping settings; check success and feasibility afterward.
        constraints = (
            NonlinearConstraint(
                lambda point: np.linalg.norm(point[:d]), 0, self.parameter_bound
            ),
            NonlinearConstraint(lambda point: np.linalg.norm(point[d:]), 0, 1),
        )

        def solve(options):
            return checked_minimize(
                objective,
                np.zeros(2 * d),
                constraints=constraints,
                options=options,
                operation="BVD-GLM-UCB joint reward-space projection",
            )

        try:
            result = solve({"ftol": 1e-6, "maxiter": 100})
        except RuntimeError:
            # Retry a failed or infeasible source solve in the same coordinates.
            result = solve({"ftol": 1e-10, "maxiter": 500})
        self.projection_count += 1
        return result.x[:d]
