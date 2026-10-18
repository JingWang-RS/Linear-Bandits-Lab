"""OL2M from ICML'16: Online Stochastic Linear Optimization under One-bit Feedback."""

from math import ceil, log2

import numpy as np
from scipy.special import expit

from algorithms.base import AlgorithmRequirements
from utils.numerics import project_ball

from .common import LogisticPolicy


class OL2M(LogisticPolicy):
    REQUIREMENTS = AlgorithmRequirements(
        links=("sigmoid",),
        feedback_kinds=("bernoulli",),
        stationary_parameters=True,
        categorical_parameters=True,
    )
    DEFAULTS = {"delta": 0.05, "regularization": None}

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
        regularization=None,
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
        self._unit_actions()
        self.regularization = self._regularization(
            dimension if regularization is None else regularization
        )
        self.beta = float(0.5 * expit(-self.parameter_bound))
        if self.beta <= 0:
            raise ValueError("parameter_bound exceeds OL2M curvature numerical range")
        self.resolved_parameters = {
            "delta": self.delta,
            "regularization": self.regularization,
            "beta": self.beta,
            "first_observation_peeling_round": 2,
        }
        self.reset()

    def reset(self):
        self.t, self.projection_count = 0, 0
        self.theta_hat = np.zeros(self.dimension)
        self.V = self.regularization * np.eye(self.dimension)
        self.inv_V = np.eye(self.dimension) / self.regularization

    def confidence_radius(self):
        if self.t == 0:
            return float(np.sqrt(self.regularization) * self.parameter_bound)
        # Eq. (13) gives log(0) at n=1. Use its n=2 bound conservatively there.
        n = max(2, self.t)
        tau = np.log(2 * ceil(2 * log2(n)) * n * n / self.delta)
        sign, logdet = np.linalg.slogdet(self.V)
        if sign != 1:
            raise ArithmeticError("OL2M design must be positive definite")
        gamma = 8 * self.parameter_bound + self.regularization * self.parameter_bound**2
        gamma += (8 / self.beta + 16 * self.parameter_bound / 3) * tau
        gamma += 2 / self.beta * (logdet - self.dimension * np.log(self.regularization))
        return float(np.sqrt(gamma))

    def scores(self, features):
        xs = self._candidates(features)
        return xs @ self.theta_hat + self.confidence_radius() * self._norms(xs, self.inv_V)

    def update(self, x, feedback):
        x = self._feature(x)
        outcome = self._outcome(feedback)
        gradient = (expit(x @ self.theta_hat) - outcome) * x
        coeff = self.beta / 2
        inverse_x = self.inv_V @ x
        self.V += coeff * np.outer(x, x)
        self.inv_V -= coeff * np.outer(inverse_x, inverse_x) / (1 + coeff * (x @ inverse_x))
        proposal = self.theta_hat - self.inv_V @ gradient
        if np.linalg.norm(proposal) > self.parameter_bound:
            self.theta_hat = project_ball(proposal, self.V, self.parameter_bound)
            self.projection_count += 1
        else:
            self.theta_hat = proposal
        self.t += 1
