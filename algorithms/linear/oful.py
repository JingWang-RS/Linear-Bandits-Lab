"""OFUL from NeurIPS'11: Improved Algorithms for Linear Stochastic Bandits."""

import numpy as np

from algorithms.base import Policy
from algorithms.base import AlgorithmRequirements
from bandit_environments.reward_models.feedback import Feedback


class OFUL(Policy):
    """OFUL with the determinant upper-bound radius of Abbasi-Yadkori et al. (2011).

    ``noise_scale`` is a known sub-Gaussian parameter, not a variance. The
    estimator is ridge least squares. Dense solves are deliberate: the short
    synthetic experiments do not need an incrementally maintained inverse.
    """

    REQUIREMENTS = AlgorithmRequirements(
        links=("identity",),
        feedback_kinds=("additive",),
        subgaussian_radius=True,
        drift_baseline_note="stationary OFUL used as a drifting-parameter baseline",
    )
    DEFAULTS = {"noise_scale": 0.1, "delta": 0.05, "regularization": 1.0}

    def __init__(
        self,
        *,
        dimension: int,
        horizon: int,
        parameter_bound: float,
        action_bound: float,
        rng: np.random.Generator,
        noise_scale: float = 0.1,
        delta: float = 0.05,
        regularization: float = 1.0,
    ):
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, (int, np.integer))
            or dimension < 1
        ):
            raise ValueError("dimension must be a positive integer")
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not np.isfinite(parameter_bound) or parameter_bound < 0:
            raise ValueError("parameter_bound must be finite and nonnegative")
        if not np.isfinite(action_bound) or action_bound <= 0:
            raise ValueError("action_bound must be finite and positive")
        if not np.isfinite(noise_scale) or noise_scale < 0:
            raise ValueError("noise_scale must be a finite nonnegative sub-Gaussian parameter")
        if not np.isfinite(delta) or not 0 < delta < 1:
            raise ValueError("delta must lie strictly between zero and one")
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        self.dimension = int(dimension)
        self.horizon = int(horizon)
        self.parameter_bound = float(parameter_bound)
        self.action_bound = float(action_bound)
        self.noise_scale = float(noise_scale)
        self.delta = float(delta)
        self.regularization = float(regularization)
        self.rng = rng
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self._covariance = self.regularization * np.eye(self.dimension)
        self._response = np.zeros(self.dimension)

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance.copy()

    @property
    def response(self) -> np.ndarray:
        return self._response.copy()

    @property
    def inverse_covariance(self) -> np.ndarray:
        return np.linalg.solve(self._covariance, np.eye(self.dimension))

    @property
    def theta_hat(self) -> np.ndarray:
        return np.linalg.solve(self._covariance, self._response)

    @property
    def beta(self) -> float:
        effective_count = self.count
        return self._radius(effective_count)

    def _radius(self, effective_count: float) -> float:
        return float(
            np.sqrt(self.regularization) * self.parameter_bound
            + self.noise_scale
            * np.sqrt(
                2 * np.log(1 / self.delta)
                + self.dimension
                * np.log1p(
                    effective_count * self.action_bound**2 / (self.regularization * self.dimension)
                )
            )
        )

    def _feature(self, x: np.ndarray) -> np.ndarray:
        feature = np.asarray(x, dtype=float)
        if feature.shape != (self.dimension,):
            raise ValueError(f"feature shape must be ({self.dimension},), got {feature.shape}")
        if not np.all(np.isfinite(feature)):
            raise ValueError("features must be finite")
        if np.linalg.norm(feature) > self.action_bound * (1 + 1e-12):
            raise ValueError("feature norm exceeds action_bound")
        return feature

    def _candidates(self, features: np.ndarray, arm_ids: np.ndarray | None = None) -> np.ndarray:
        candidates = np.asarray(features, dtype=float)
        if candidates.ndim != 2 or candidates.shape[1] != self.dimension or candidates.shape[0] < 1:
            raise ValueError(f"candidate shape must be (n, {self.dimension}) with n >= 1")
        if arm_ids is not None and np.asarray(arm_ids).shape != (len(candidates),):
            raise ValueError("arm_ids shape must match candidate count")
        if not np.all(np.isfinite(candidates)):
            raise ValueError("candidate features must be finite")
        if np.any(np.linalg.norm(candidates, axis=1) > self.action_bound * (1 + 1e-12)):
            raise ValueError("candidate norm exceeds action_bound")
        return candidates

    def _width_squared(self, candidates: np.ndarray) -> np.ndarray:
        solutions = np.linalg.solve(self._covariance, candidates.T).T
        return np.einsum("ij,ij->i", candidates, solutions)

    def scores(self, features: np.ndarray) -> np.ndarray:
        candidates = self._candidates(features)
        width_squared = self._width_squared(candidates)
        if np.any(width_squared < 0) or not np.all(np.isfinite(width_squared)):
            raise ArithmeticError("confidence norm is invalid; covariance solve failed")
        scores = candidates @ self.theta_hat + self.beta * np.sqrt(width_squared)
        if not np.all(np.isfinite(scores)):
            raise ArithmeticError("nonfinite confidence scores")
        return scores

    def _observation(self, x: np.ndarray, feedback: Feedback) -> tuple[np.ndarray, float]:
        feature = self._feature(x)
        reward = float(feedback.reward)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        return feature, reward

    def update(self, x: np.ndarray, feedback: Feedback) -> None:
        feature, reward = self._observation(x, feedback)
        self._covariance += np.outer(feature, feature)
        self._response += reward * feature
        self.count += 1
