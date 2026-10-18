"""Hvt-UCB from ICML'25: Heavy-Tailed Linear Bandits: Huber Regression with One-Pass Update."""

import numpy as np

from algorithms.base import AlgorithmRequirements
from bandit_environments.reward_models.feedback import Feedback

from .common import _HeavyBase
from .huber import huber_derivative


class HvtUCB(_HeavyBase):
    """Estimate linear rewards with one Huber-regression update per observation.

    ``implementation='theory'`` uses alpha=4 and coefficient107. ``'icml25'``
    restores the source experiment's clock, scale rule and tuned confidence.
    """

    REQUIREMENTS = AlgorithmRequirements(
        links=("identity",),
        feedback_kinds=("additive",),
        stationary_parameters=True,
        known_moment="selected-or-global",
    )
    DEFAULTS = _HeavyBase.DEFAULTS | {
        "sigma_min": 0.1,
        "alpha": 4.0,
        "noise_scale_multiplier": 1.0,
    }

    def __init__(
        self, *, sigma_min=0.1, alpha=4.0, noise_scale_multiplier=1.0, **kwargs
    ):
        super().__init__(**kwargs)
        if (
            not np.isfinite(alpha)
            or alpha <= 0
            or (self.implementation == "theory" and alpha != 4)
        ):
            raise ValueError(
                "alpha must be positive; the theoretical definition requires alpha=4"
            )
        if not np.isfinite(noise_scale_multiplier) or noise_scale_multiplier <= 0:
            raise ValueError("noise_scale_multiplier must be finite and positive")
        if self.implementation == "theory" and noise_scale_multiplier != 1:
            raise ValueError("noise tuning requires implementation='icml25'")
        self.noise_scale_multiplier = float(noise_scale_multiplier)
        if not np.isfinite(sigma_min) or sigma_min <= 0:
            raise ValueError("sigma_min must be finite and positive")
        self.alpha, self.sigma_min = float(alpha), float(sigma_min)
        self.kappa = self.dimension * np.log1p(
            self.action_bound**2
            * self.horizon
            / (self.sigma_min**2 * self.regularization * self.alpha * self.dimension)
        )
        self.tau0 = (
            np.sqrt(2 * self.kappa)
            * np.log(3 * self.horizon) ** self.q
            / self.log_confidence ** (1 / (1 + self.moment_exponent))
        )
        self.reset()

    def reset(self):
        super().reset()
        self.sigma = self.sigma_min
        self.tau = self.tau0

    @property
    def beta(self):
        # At epsilon=1, t^0 is 1 including t=0, directly following Eq.(beta).
        if self.implementation == "icml25":
            # The server experiment used t=1 initially, coefficient 1 rather
            # than the theorem's 107, and a tuned confidence divisor.
            return float(
                (
                    np.sqrt(self.regularization * (2 + 4 * self.parameter_bound**2))
                    + self.log_confidence * self.tau0 * (self.count + 1) ** self.q
                )
                / self.confidence_scale
            )
        return float(
            np.sqrt(self.regularization * (2 + 4 * self.parameter_bound**2))
            + 107 * self.log_confidence * self.tau0 * self.count**self.q
        )

    def update(self, x, feedback: Feedback):
        x = self._feature(x)
        reward, root = self._reward(feedback), self._noise_root(feedback)
        t = self.count + 1
        width = self._width(x)
        if self.implementation == "icml25":
            # Match the experiment's pre-incremented update clock and its
            # empirical scale rule. The supplied noise metadata stays exact;
            # the multiplier is explicitly an algorithm tuning parameter.
            update_t = t + 1
            sigma = max(
                root * self.noise_scale_multiplier,
                self.sigma_min,
                2 * width * self.beta / (self.tau0 * update_t**self.q),
            )
        else:
            update_t = t
            sigma = max(
                root,
                self.sigma_min,
                np.sqrt(2 * self.beta / (self.tau0 * np.sqrt(self.alpha) * t**self.q))
                * width,
            )
        w = width / (sigma * np.sqrt(self.alpha))
        # The zero-feature limit supplies no parameter information.
        tau = np.inf if w == 0 else self.tau0 * np.sqrt(1 + w**2) / w * update_t**self.q
        covariance = self._covariance + np.outer(x, x) / (sigma**2 * self.alpha)
        residual = (reward - x @ self._theta) / sigma
        gradient = -x / sigma * huber_derivative(residual, tau)
        theta = self._projected_step(x, gradient, covariance)
        self._covariance, self._theta = covariance, theta
        self.count, self.sigma, self.tau = t, sigma, tau
