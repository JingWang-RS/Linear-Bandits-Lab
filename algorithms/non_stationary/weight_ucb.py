"""WeightUCB from TIT'26: Revisiting Weighted Strategy for Non-stationary Parametric Bandits and MDPs."""

import numpy as np

from algorithms.linear import OFUL
from algorithms.base import AlgorithmRequirements
from bandit_environments.reward_models.feedback import Feedback


class WeightUCB(OFUL):
    """Estimate linear rewards while giving older observations less weight.

    Implements Algorithm 1 of the TIT paper. Both the covariance and the
    confidence radius use weights ``gamma**(t-s)``. Action selection uses
    this single covariance matrix; squared weights belong to D-LinUCB.
    Setting ``gamma=1`` recovers OFUL; the drifting case uses ``gamma<1``.
    """

    REQUIREMENTS = AlgorithmRequirements(
        links=("identity",), feedback_kinds=("additive",), subgaussian_radius=True
    )
    DEFAULTS = {**OFUL.DEFAULTS, "gamma": 0.97}

    def __init__(self, *, gamma: float = 0.97, **kwargs):
        if not np.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError("gamma must lie in (0, 1]")
        self.gamma = float(gamma)
        super().__init__(**kwargs)

    def reset(self) -> None:
        super().reset()
        self._effective_weight = 0.0

    @property
    def beta(self) -> float:
        return self._radius(self._effective_weight)

    def update(self, x: np.ndarray, feedback: Feedback) -> None:
        feature, reward = self._observation(x, feedback)
        self._covariance = (
            self.gamma * self._covariance
            + np.outer(feature, feature)
            + (1 - self.gamma) * self.regularization * np.eye(self.dimension)
        )
        self._response = self.gamma * self._response + reward * feature
        self._effective_weight = self.gamma * self._effective_weight + 1
        self.count += 1
