"""Shared parameter checks and action scores for heavy-tailed bandit algorithms."""

import numpy as np

from algorithms.base import Policy
from utils.numerics import project_ball


class _HeavyBase(Policy):
    DEFAULTS = {
        "delta": 0.05,
        "regularization": 1.0,
        "moment_exponent": 0.5,
        "moment_bound": None,
        "implementation": "theory",
        "confidence_scale": 1.0,
    }

    def __init__(
        self,
        *,
        dimension,
        horizon,
        parameter_bound,
        action_bound,
        rng,
        delta=0.05,
        regularization=1.0,
        moment_exponent=0.5,
        moment_bound=None,
        implementation="theory",
        confidence_scale=1.0,
    ):
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, (int, np.integer))
            or dimension < 1
        ):
            raise ValueError("dimension must be a positive integer")
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, (int, np.integer))
            or horizon < 1
        ):
            raise ValueError("horizon must be a positive integer")
        for label, value in (
            ("parameter_bound", parameter_bound),
            ("action_bound", action_bound),
            ("regularization", regularization),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be finite and positive")
        if not np.isfinite(delta) or not 0 < delta < 1:
            raise ValueError("delta must lie strictly between zero and one")
        if not np.isfinite(moment_exponent) or not 0 < moment_exponent <= 1:
            raise ValueError("moment_exponent must lie in (0, 1]")
        if moment_bound is not None and (
            not np.isfinite(moment_bound) or moment_bound < 0
        ):
            raise ValueError(
                "moment_bound must be a finite nonnegative noise moment root"
            )
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        if implementation not in {"theory", "icml25"}:
            raise ValueError("implementation must be 'theory' or 'icml25'")
        if not np.isfinite(confidence_scale) or confidence_scale <= 0:
            raise ValueError("confidence_scale must be finite and positive")
        if implementation == "theory" and confidence_scale != 1:
            raise ValueError("confidence tuning requires implementation='icml25'")
        self.implementation = implementation
        self.confidence_scale = float(confidence_scale)
        self.dimension, self.horizon = int(dimension), int(horizon)
        self.parameter_bound, self.action_bound = (
            float(parameter_bound),
            float(action_bound),
        )
        self.regularization, self.delta = float(regularization), float(delta)
        self.moment_exponent = float(moment_exponent)
        self.moment_bound = None if moment_bound is None else float(moment_bound)
        self.rng = rng
        self.q = (1 - self.moment_exponent) / (2 * (1 + self.moment_exponent))
        self.log_confidence = np.log(2 * self.horizon**2 / self.delta)

    def reset(self):
        self.count = 0
        self._covariance = self.regularization * np.eye(self.dimension)
        self._theta = np.zeros(self.dimension)

    @property
    def covariance(self):
        return self._covariance.copy()

    @property
    def inverse_covariance(self):
        return np.linalg.solve(self._covariance, np.eye(self.dimension))

    @property
    def theta_hat(self):
        return self._theta.copy()

    def _feature(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dimension,) or not np.isfinite(x).all():
            raise ValueError(f"feature must be finite with shape ({self.dimension},)")
        if np.linalg.norm(x) > self.action_bound * (1 + 1e-12):
            raise ValueError("feature norm exceeds action_bound")
        return x

    def _candidates(self, features, arm_ids=None):
        features = np.asarray(features, dtype=float)
        ids = None if arm_ids is None else np.asarray(arm_ids)
        if (
            features.ndim != 2
            or features.shape[1] != self.dimension
            or len(features) == 0
        ):
            raise ValueError(
                f"candidate shape must be (n, {self.dimension}) with n >= 1"
            )
        if ids is not None and ids.shape != (len(features),):
            raise ValueError("arm_ids shape must match candidate count")
        if not np.isfinite(features).all():
            raise ValueError("candidate features must be finite")
        if np.any(np.linalg.norm(features, axis=1) > self.action_bound * (1 + 1e-12)):
            raise ValueError("candidate norm exceeds action_bound")
        return features, ids

    def _reward(self, feedback):
        if self.count >= self.horizon:
            raise ValueError("observations exceed configured horizon")
        reward = float(feedback.reward)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        return reward

    def _noise_root(self, feedback):
        root = feedback.noise_moment_root
        if root is None:
            if self.moment_bound is None:
                raise ValueError(
                    "known noise_moment_root or global moment_bound is required"
                )
            return self.moment_bound
        root = float(root)
        if not np.isfinite(root) or root < 0:
            raise ValueError("noise_moment_root must be finite and nonnegative")
        if self.moment_bound is not None and root > self.moment_bound * (1 + 1e-12):
            raise ValueError("known noise moment root exceeds configured moment_bound")
        return root

    def _width(self, x):
        squared = float(x @ np.linalg.solve(self._covariance, x))
        if not np.isfinite(squared) or squared < 0:
            raise ArithmeticError("confidence norm is invalid; covariance solve failed")
        return np.sqrt(squared)

    def scores(self, features):
        features, _ = self._candidates(features)
        squared = np.einsum(
            "ij,ij->i", features, np.linalg.solve(self._covariance, features.T).T
        )
        if np.any(squared < 0) or not np.isfinite(squared).all():
            raise ArithmeticError("confidence norm is invalid; covariance solve failed")
        scores = features @ self._theta + self.beta * np.sqrt(squared)
        if not np.isfinite(scores).all():
            raise ArithmeticError("nonfinite confidence scores")
        return scores

    def _projected_step(self, x, gradient, covariance):
        point = self._theta - np.linalg.solve(covariance, gradient)
        return project_ball(point, covariance, self.parameter_bound)
