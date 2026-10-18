"""SW-UCB from AISTATS'19: Learning to Optimize under Non-Stationarity."""

from collections import deque

import numpy as np

from algorithms.linear import OFUL
from algorithms.base import AlgorithmRequirements
from bandit_environments.reward_models.feedback import Feedback

from .common import _positive_integer


class SlidingWindowUCB(OFUL):
    """SW-UCB with the fixed-window radius in Cheung et al. (AISTATS 2019).

    Rebuild the bounded-window ridge statistics to avoid cancellation from
    repeated covariance downdates. This changes neither the estimator nor its
    mathematical window semantics.
    """

    REQUIREMENTS = AlgorithmRequirements(
        links=("identity",), feedback_kinds=("additive",), subgaussian_radius=True
    )
    DEFAULTS = {**OFUL.DEFAULTS, "window": 30}

    def __init__(self, *, window: int = 30, **kwargs):
        self.window = _positive_integer(window, "window")
        super().__init__(**kwargs)

    def reset(self) -> None:
        super().reset()
        self._history: deque[tuple[np.ndarray, float]] = deque(maxlen=self.window)

    @property
    def history_size(self) -> int:
        return len(self._history)

    @property
    def beta(self) -> float:
        return float(
            np.sqrt(self.regularization) * self.parameter_bound
            + self.noise_scale
            * np.sqrt(
                self.dimension
                * np.log(
                    (1 + self.window * self.action_bound**2 / self.regularization) / self.delta
                )
            )
        )

    def update(self, x: np.ndarray, feedback: Feedback) -> None:
        feature, reward = self._observation(x, feedback)
        self._history.append((feature.copy(), reward))
        history_features = np.stack([entry[0] for entry in self._history])
        history_rewards = np.array([entry[1] for entry in self._history])
        self._covariance = (
            self.regularization * np.eye(self.dimension) + history_features.T @ history_features
        )
        self._response = history_features.T @ history_rewards
        self.count += 1
