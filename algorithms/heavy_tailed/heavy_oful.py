"""Heavy-OFUL from NeurIPS'23:
Tackling Heavy-Tailed Rewards in Reinforcement Learning with Function Approximation:
Minimax Optimal and Instance-Dependent Regret Bounds."""

import numpy as np

from algorithms.base import AlgorithmRequirements
from bandit_environments.reward_models.feedback import Feedback

from .common import _HeavyBase
from .huber import fit_huber_ball


class HeavyOFUL(_HeavyBase):
    """Fit Huber regression to all observations and choose an optimistic action."""

    REQUIREMENTS = AlgorithmRequirements(
        links=("identity",),
        feedback_kinds=("additive",),
        stationary_parameters=True,
        known_moment="selected-or-global",
    )
    DEFAULTS = _HeavyBase.DEFAULTS | {"sigma_min": 0.1, "noise_scale_multiplier": 1.0}

    def __init__(self, *, sigma_min=0.1, noise_scale_multiplier=1.0, **kwargs):
        super().__init__(**kwargs)
        if not np.isfinite(sigma_min) or sigma_min <= 0:
            raise ValueError("sigma_min must be finite and positive")
        if not np.isfinite(noise_scale_multiplier) or noise_scale_multiplier <= 0:
            raise ValueError("noise_scale_multiplier must be finite and positive")
        if self.implementation == "theory" and noise_scale_multiplier != 1:
            raise ValueError("noise tuning requires implementation='icml25'")
        self.noise_scale_multiplier = float(noise_scale_multiplier)
        self.sigma_min = float(sigma_min)
        self.kappa = self.dimension * np.log1p(
            self.action_bound**2
            * self.horizon
            / (self.sigma_min**2 * self.regularization * self.dimension)
        )
        self.tau0 = (
            np.sqrt(2 * self.kappa)
            * np.log(3 * self.horizon) ** self.q
            / self.log_confidence ** (1 / (1 + self.moment_exponent))
        )
        self.c0 = 1 / np.sqrt(23 * self.log_confidence)
        self.c1 = np.log(3 * self.horizon) ** (2 * self.q) / (
            48 * self.log_confidence ** (2 / (1 + self.moment_exponent))
        )
        self.reset()

    def reset(self):
        super().reset()
        self.sigma, self.tau = self.sigma_min, self.tau0
        # Full-history estimation needs O(T*d) storage, but should not copy
        # the entire history on every update before the optimizer even runs.
        self._feature_history = np.empty((self.horizon, self.dimension))
        self._reward_history = np.empty(self.horizon)
        self._sigma_history = np.empty(self.horizon)
        self._tau_history = np.empty(self.horizon)

    @property
    def features(self):
        return self._feature_history[: self.count].copy()

    @property
    def rewards(self):
        return self._reward_history[: self.count].copy()

    @property
    def sigmas(self):
        return self._sigma_history[: self.count].copy()

    @property
    def taus(self):
        return self._tau_history[: self.count].copy()

    @property
    def beta(self):
        return float(
            (
                3 * np.sqrt(self.regularization) * self.parameter_bound
                + 24 * self.tau0 * self.log_confidence * self.count**self.q
            )
            / self.confidence_scale
        )

    def update(self, x, feedback: Feedback):
        x = self._feature(x)
        reward, root = self._reward(feedback), self._noise_root(feedback)
        t, width = self.count + 1, self._width(x)
        sigma = max(
            root * self.noise_scale_multiplier,
            self.sigma_min,
            width / self.c0,
            np.sqrt(self.action_bound * self.parameter_bound * width)
            / (2 * self.c1 * self.kappa) ** 0.25,
        )
        w = width / sigma
        tau = np.inf if w == 0 else self.tau0 * np.sqrt(1 + w**2) / w * t**self.q
        self._feature_history[self.count] = x
        self._reward_history[self.count] = reward
        self._sigma_history[self.count] = sigma
        self._tau_history[self.count] = tau
        features = self._feature_history[:t]
        rewards = self._reward_history[:t]
        sigmas = self._sigma_history[:t]
        taus = self._tau_history[:t]
        result = fit_huber_ball(
            self._theta,
            features,
            rewards,
            sigmas,
            taus,
            self.regularization,
            self.parameter_bound,
            operation=f"HeavyOFUL estimation at observation {t}",
        )
        covariance = self._covariance + np.outer(x, x) / sigma**2
        self._covariance, self._theta = covariance, result.x.copy()
        self.count, self.sigma, self.tau = t, sigma, tau
