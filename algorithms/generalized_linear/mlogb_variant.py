"""An experimental MLogB variant based on NeurIPS'23:
Online (Multinomial) Logistic Bandit: Improved Regret and Constant Computation Cost."""

import numpy as np
from scipy.linalg import eigvalsh

from algorithms.base import AlgorithmRequirements
from utils.numerics import project_ball

from .common import LogisticPolicy


class MLogBUCBVariant(LogisticPolicy):
    """Update the logistic reward estimate once per observation, using experimental tuning.

    The empirical alpha/lambda/online step and one-term bonus differ from the
    NeurIPS 2023 OFUL-MLogB theorem. There is no theorem guarantee for this variant.
    """

    REQUIREMENTS = AlgorithmRequirements(
        links=("sigmoid", "softmax"),
        feedback_kinds=("bernoulli", "categorical"),
        stationary_parameters=True,
        categorical_parameters=True,
    )
    DEFAULTS = {"delta": 0.05, "regularization": None, "eta": None}

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
        eta=None,
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
        self._unit_actions()
        self.alpha = self.parameter_bound + 0.5 * np.log(self.outcomes + 1)
        self.eta = float(self.alpha / 2 if eta is None else eta)
        if not np.isfinite(self.eta) or self.eta <= 0:
            raise ValueError("eta must be finite and positive")
        self.regularization = self._regularization(
            np.sqrt(self.outcomes) * self.alpha * self.parameter_bound
            if regularization is None
            else regularization
        )
        self.R = float(np.linalg.norm(self.reward_vector[1:]))
        self.resolved_parameters = {
            "delta": self.delta,
            "regularization": self.regularization,
            "eta": self.eta,
            "alpha": self.alpha,
            "reward_norm": self.R,
        }
        self.reset()

    def reset(self):
        self.t, self.projection_count = 0, 0
        self.theta_hat = np.zeros(self.size)
        self.H = self.regularization * np.eye(self.size)
        self.inv_H = np.eye(self.size) / self.regularization

    def confidence_radius(self):
        t = self.t + 1
        lam = self.regularization
        s_mu = np.log(self.outcomes + 1) + 2 * np.log((self.outcomes + 1) * t)
        gamma_a = s_mu * (
            lam + np.sqrt(lam) / 4 + 4 / np.sqrt(lam) * np.log(np.sqrt(1 + 2 * t) / self.delta)
        )
        gamma_b = (
            7
            * self.alpha
            / np.sqrt(6)
            * self.outcomes
            * self.dimension
            * np.log1p((t + 1) * self.action_bound / 2)
        )
        return float(
            np.sqrt(
                4 * lam * self.outcomes * self.parameter_bound**2 + self.alpha * (gamma_a + gamma_b)
            )
        )

    def local_radius(self, x):
        # H^-1 A is generally nonsymmetric. Symmetric generalized eigenvalues
        # (A,H) have the same spectrum, with no artificial symmetrization.
        a = np.kron(np.eye(self.outcomes), np.outer(x, x))
        maximum = float(eigvalsh(a, self.H, subset_by_index=[self.size - 1, self.size - 1])[0])
        if maximum < 0 or not np.isfinite(maximum):
            raise ArithmeticError(
                "MLogB generalized design eigenvalue must be finite and nonnegative"
            )
        return float(np.sqrt(maximum))

    def scores(self, features):
        xs = self._candidates(features)
        beta = self.confidence_radius()
        return np.array(
            [
                self.reward_vector @ self.probabilities(self.theta_hat, x)
                + self.R * beta * self.local_radius(x)
                for x in xs
            ]
        )

    def update(self, x, feedback):
        x = self._feature(x)
        outcome = self._outcome(feedback)
        p = self.probabilities(self.theta_hat, x)[1:]
        target = np.zeros(self.outcomes)
        if outcome:
            target[outcome - 1] = 1
        gradient = np.kron(p - target, x)
        curvature = np.kron(np.diag(p) - np.outer(p, p), np.outer(x, x))
        metric = self.H / (2 * self.eta) + curvature / 2
        proposal = self.theta_hat - np.linalg.solve(metric, gradient)
        if np.linalg.norm(proposal) > self.parameter_bound:
            self.theta_hat = project_ball(proposal, metric, self.parameter_bound)
            self.projection_count += 1
        else:
            self.theta_hat = proposal
        new_curvature = np.kron(self._curvature(self.theta_hat, x), np.outer(x, x))
        self.H += new_curvature
        self.inv_H = np.linalg.solve(self.H, np.eye(self.size))
        self.t += 1
