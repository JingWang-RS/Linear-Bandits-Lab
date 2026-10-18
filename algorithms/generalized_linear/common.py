"""Shared logistic reward estimates and probability calculations."""

import numpy as np
from scipy.special import logsumexp, softmax

from algorithms.base import Policy


class LogisticPolicy(Policy):
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
    ):
        for name, value in [("dimension", dimension), ("horizon", horizon), ("outcomes", outcomes)]:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in [("parameter_bound", parameter_bound), ("action_bound", action_bound)]:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive for logistic algorithms")
        if not np.isfinite(delta) or not 0 < delta < 1:
            raise ValueError("delta must be strictly between zero and one")
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        if reward_vector is None:
            reward_vector = np.arange(outcomes + 1, dtype=float)
        rewards = np.asarray(reward_vector, dtype=float)
        if rewards.shape != (outcomes + 1,) or not np.isfinite(rewards).all():
            raise ValueError("reward_vector must have outcomes + 1 finite entries")
        if rewards[0] != 0 or np.any(rewards < 0) or not np.any(rewards[1:] > 0):
            raise ValueError(
                "reward_vector requires baseline reward 0, nonnegative entries, and a positive nonbaseline reward"
            )
        self.dimension, self.horizon, self.outcomes = int(dimension), int(horizon), int(outcomes)
        self.parameter_bound, self.action_bound, self.delta = (
            float(parameter_bound),
            float(action_bound),
            float(delta),
        )
        self.rng, self.reward_vector = rng, rewards.copy()
        self.size = self.outcomes * self.dimension

    def _binary(self):
        if self.outcomes != 1 or not np.array_equal(self.reward_vector, [0.0, 1.0]):
            raise ValueError("binary algorithm requires outcomes=1 and reward_vector=[0,1]")

    def _unit_actions(self):
        if self.action_bound > 1:
            raise ValueError("this logistic algorithm requires action_bound <= 1")

    def _regularization(self, value):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("regularization must be finite and positive")
        return float(value)

    def _feature(self, x):
        value = np.asarray(x, dtype=float)
        if value.shape != (self.dimension,) or not np.isfinite(value).all():
            raise ValueError(f"feature must be a finite vector of dimension {self.dimension}")
        if np.linalg.norm(value) > self.action_bound * (1 + 1e-12):
            raise ValueError("feature norm exceeds action_bound")
        return value

    def _candidates(self, features, arm_ids=None):
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.dimension or len(values) < 1:
            raise ValueError(f"candidate features must have shape (n,{self.dimension}), n>=1")
        if arm_ids is not None and np.asarray(arm_ids).shape != (len(values),):
            raise ValueError("arm_ids shape must match candidate count")
        if not np.isfinite(values).all() or np.any(
            np.linalg.norm(values, axis=1) > self.action_bound * (1 + 1e-12)
        ):
            raise ValueError("candidate features must be finite and satisfy action_bound")
        return values

    def _outcome(self, feedback):
        value = feedback.outcome
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or not 0 <= value <= self.outcomes
        ):
            raise ValueError(f"outcome must be an integer category in 0..{self.outcomes}")
        if not np.isfinite(feedback.reward) or feedback.reward != self.reward_vector[value]:
            raise ValueError("reward does not agree with outcome and known reward_vector")
        return int(value)

    def probabilities(self, theta, x):
        logits = np.asarray(theta).reshape(self.outcomes, self.dimension) @ np.asarray(x)
        return softmax(np.r_[0.0, logits])

    def _curvature(self, theta, x):
        p = self.probabilities(theta, x)[1:]
        return np.diag(p) - np.outer(p, p)

    @staticmethod
    def _norms(features, inverse):
        squares = np.einsum("nd,df,nf->n", features, inverse, features)
        if np.any(squares < 0) or not np.isfinite(squares).all():
            raise ArithmeticError("design quadratic form must be finite and nonnegative")
        return np.sqrt(squares)


class CategoricalHistory(LogisticPolicy):
    """Store observations and fit a regularized multinomial logistic reward model."""

    def _history(self):
        # Updates append observations; solver evaluations reuse the same history.
        # Keep list identities as well as length so reset() invalidates the cache.
        cached = getattr(self, "_history_cache", None)
        if (
            cached is None
            or cached[0] is not self.actions
            or cached[1] is not self.outcomes_history
            or cached[2] != len(self.actions)
            or cached[3] != len(self.outcomes_history)
        ):
            cached = (
                self.actions,
                self.outcomes_history,
                len(self.actions),
                len(self.outcomes_history),
                np.asarray(self.actions, dtype=float).reshape(-1, self.dimension),
                np.asarray(self.outcomes_history, dtype=int),
            )
            self._history_cache = cached
        return cached[4], cached[5]

    def _history_probabilities(self, theta, features):
        logits = features @ np.asarray(theta).reshape(self.outcomes, self.dimension).T
        return softmax(np.column_stack((np.zeros(len(features)), logits)), axis=1)

    def loss(self, theta):
        features, outcomes = self._history()
        logits = features @ np.asarray(theta).reshape(self.outcomes, self.dimension).T
        logits = np.column_stack((np.zeros(len(features)), logits))
        return float(
            0.5 * self.likelihood_regularization * (theta @ theta)
            + np.sum(logsumexp(logits, axis=1) - logits[np.arange(len(features)), outcomes])
        )

    def gt(self, theta):
        features, _ = self._history()
        probabilities = self._history_probabilities(theta, features)[:, 1:]
        return self.likelihood_regularization * np.asarray(theta) + (
            probabilities.T @ features
        ).reshape(-1)

    def loss_gradient(self, theta):
        features, outcomes = self._history()
        residuals = self._history_probabilities(theta, features)
        residuals[np.arange(len(features)), outcomes] -= 1
        return self.likelihood_regularization * np.asarray(theta) + (
            residuals[:, 1:].T @ features
        ).reshape(-1)

    def hessian(self, theta):
        features, _ = self._history()
        probabilities = self._history_probabilities(theta, features)[:, 1:]
        curvature = probabilities[:, :, None] * (
            np.eye(self.outcomes)[None, :, :] - probabilities[:, None, :]
        )
        data_hessian = np.einsum(
            "nkl,ni,nj->kilj", curvature, features, features, optimize=True
        ).reshape(self.size, self.size)
        return self.likelihood_regularization * np.eye(self.size) + data_hessian
