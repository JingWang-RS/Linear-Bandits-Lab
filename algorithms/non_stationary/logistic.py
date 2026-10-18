"""Fit logistic reward models for the non-stationary GLB and SCB algorithms."""

import numpy as np
from scipy.special import expit

from algorithms.generalized_linear.common import LogisticPolicy
from algorithms.base import AlgorithmRequirements
from utils.numerics import checked_minimize, checked_newton


class LogisticEstimate(LogisticPolicy):
    """Estimate logistic reward parameters, refitting after the configured number of observations.

    Covariance and projection use every observation, even between likelihood
    fits. Projection objectives are supplied by GLB and SCB respectively; they
    are local constrained numerical solves, without a global optimality claim.
    """

    REQUIREMENTS = AlgorithmRequirements(
        links=("sigmoid",), feedback_kinds=("bernoulli",), categorical_parameters=True
    )
    DEFAULTS = {
        "delta": 0.05,
        "regularization": 2.0,
        "noise_scale": 0.5,
        "k_mu": 0.25,
        "c_mu": None,
        "lazy_update_frequency": 5,
        "lazy_update_threshold": 200,
    }

    def __init__(
        self,
        *,
        dimension,
        horizon,
        parameter_bound,
        action_bound,
        rng,
        outcomes=1,
        reward_vector=None,
        delta=0.05,
        regularization=2.0,
        noise_scale=0.5,
        k_mu=0.25,
        c_mu=None,
        lazy_update_frequency=5,
        lazy_update_threshold=200,
    ):
        super().__init__(
            dimension=dimension,
            horizon=horizon,
            parameter_bound=parameter_bound,
            action_bound=action_bound,
            rng=rng,
            outcomes=outcomes,
            reward_vector=reward_vector,
            delta=delta,
        )
        self._binary()
        self.regularization = self._regularization(regularization)
        actual_curvature = float(
            expit(parameter_bound * action_bound)
            * expit(-parameter_bound * action_bound)
        )
        self.c_mu = actual_curvature if c_mu is None else float(c_mu)
        if not np.isfinite(self.c_mu) or not 0 < self.c_mu <= actual_curvature * (
            1 + 1e-12
        ):
            raise ValueError(
                "c_mu must be a positive lower bound on logistic curvature"
            )
        if not np.isfinite(k_mu) or k_mu < 0.25:
            raise ValueError("k_mu must bound the logistic derivative (at least 0.25)")
        if not np.isfinite(noise_scale) or noise_scale < 0.5:
            raise ValueError("noise_scale must bound Bernoulli noise (at least 0.5)")
        for name, value in (
            ("lazy_update_frequency", lazy_update_frequency),
            ("lazy_update_threshold", lazy_update_threshold),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        self.noise_scale, self.k_mu = float(noise_scale), float(k_mu)
        self.lazy_update_frequency = int(lazy_update_frequency)
        self.lazy_update_threshold = int(lazy_update_threshold)
        self.likelihood_regularization = self.regularization * self.c_mu
        self.gamma = getattr(self, "gamma", 1.0)
        self.restart_interval = getattr(self, "restart_interval", None)
        self.resolved_parameters = {
            "c_mu": self.c_mu,
            "likelihood_regularization": self.likelihood_regularization,
            "projection": "checked local constrained source objective",
        }
        self.reset()

    @staticmethod
    def _discount(gamma):
        if not np.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError("gamma must lie in (0, 1]")
        return float(gamma)

    @staticmethod
    def _interval(value):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError("restart_interval must be a positive integer")
        return int(value)

    def _clear_estimator(self):
        self.count = 0
        self._xs = np.empty((32, self.dimension))
        self._ys = np.empty(32)
        self._weights = np.empty(0)
        self.theta_hat = np.zeros(self.dimension)
        self.theta_tilde = np.zeros(self.dimension)
        self.V = self.regularization * np.eye(self.dimension)
        self.inv_V = np.eye(self.dimension) / self.regularization
        self.H = self.likelihood_regularization * np.eye(self.dimension)
        self.tilde_V = self.V.copy()
        self.effective_weight = 0.0

    def reset(self):
        self.t = 0
        self.projection_count = 0
        self.projection_fallback_count = 0
        self._clear_estimator()

    def _history(self):
        return self._xs[: self.count], self._ys[: self.count]

    def loss(self, theta):
        xs, ys = self._history()
        logits = xs @ theta
        return float(
            0.5 * self.likelihood_regularization * (theta @ theta)
            # A BLAS dot over thousands of positive likelihood terms can vary
            # by more than a near-optimum Newton decrease. Extended-precision
            # accumulation preserves the same objective without stalling its
            # line search or relaxing the independently checked gradient.
            + np.sum(
                self._weights * (np.logaddexp(0, logits) - ys * logits),
                dtype=np.longdouble,
            )
        )

    def gt(self, theta):
        xs, _ = self._history()
        return self.likelihood_regularization * theta + xs.T @ (
            self._weights * expit(xs @ theta)
        )

    def loss_gradient(self, theta):
        xs, ys = self._history()
        return self.likelihood_regularization * theta + xs.T @ (
            self._weights * (expit(xs @ theta) - ys)
        )

    def hessian(self, theta):
        xs, _ = self._history()
        p = expit(xs @ theta)
        return self.likelihood_regularization * np.eye(self.dimension) + xs.T @ (
            (self._weights * p * (1 - p))[:, None] * xs
        )

    def _projection(self):
        radius = self.parameter_bound
        # Scaling the coordinates and objective leaves the source minimizer
        # unchanged and avoids poor SLSQP conditioning at large ridge values.
        scale = max(1.0, self.projection_objective(np.zeros(self.dimension)))
        constraint = {
            "type": "ineq",
            "fun": lambda u: 1 - u @ u,
            "jac": lambda u: -2 * u,
        }

        def solve(initial):
            return checked_minimize(
                lambda u: self.projection_objective(radius * u) / scale,
                initial,
                jac=lambda u: radius * self.projection_gradient(radius * u) / scale,
                constraints=constraint,
                operation=f"{type(self).__name__} reward-space projection",
            )

        initial = self.theta_tilde / radius
        initial /= max(1.0, np.linalg.norm(initial))
        try:
            result = solve(initial)
        except RuntimeError:
            # SLSQP can reject its line search at a previous boundary solution.
            # Restart the same objective at the source's feasible zero point.
            result = solve(np.zeros(self.dimension))
        self.projection_count += 1
        return radius * result.x

    def update(self, x, feedback):
        feature, outcome = self._feature(x), self._outcome(feedback)
        # BVD uses the confidence radius from the decision preceding this update.
        self._projection_beta = self.beta
        if self.count == len(self._ys):
            self._xs = np.concatenate((self._xs, np.empty_like(self._xs)))
            self._ys = np.concatenate((self._ys, np.empty_like(self._ys)))
        self._xs[self.count], self._ys[self.count] = feature, outcome
        self.count += 1
        self._weights = self.gamma ** np.arange(self.count - 1, -1, -1)
        outer = np.outer(feature, feature)
        self.V = (
            self.gamma * self.V
            + outer
            + (1 - self.gamma) * self.regularization * np.eye(self.dimension)
        )
        self.inv_V = np.linalg.solve(self.V, np.eye(self.dimension))
        self._update_auxiliary_covariance(outer)
        self.effective_weight = self.gamma**2 * self.effective_weight + 1
        if (
            self.count - 1
        ) % self.lazy_update_frequency == 0 or self.count < self.lazy_update_threshold:
            result = checked_newton(
                self.loss,
                self.theta_hat,
                jac=self.loss_gradient,
                hess=self.hessian,
                operation=f"{type(self).__name__} weighted maximum likelihood",
            )
            self.theta_hat = result.x
            self.H = self.hessian(self.theta_hat)
        self.theta_tilde = (
            self.theta_hat.copy()
            if np.linalg.norm(self.theta_hat) <= self.parameter_bound
            else self._projection()
        )
        self.t += 1
        if self.restart_interval is not None and self.count == self.restart_interval:
            self._clear_estimator()

    def _update_auxiliary_covariance(self, outer):
        pass

    @property
    def confidence_count(self):
        # The source restarted estimator uses its global clock in the radius.
        return (
            self.t + 1 if self.restart_interval is not None else self.effective_weight
        )

    def scores(self, features):
        xs = self._candidates(features)
        return expit(
            xs @ self.theta_tilde
        ) + self.radius_multiplier * self.beta * self._norms(xs, self.inv_V)
