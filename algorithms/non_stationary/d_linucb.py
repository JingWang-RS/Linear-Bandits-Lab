"""D-LinUCB from NeurIPS'19: Weighted Linear Bandits for Non-Stationary Environments."""

import numpy as np

from algorithms.base import AlgorithmRequirements
from bandit_environments.reward_models.feedback import Feedback

from .weight_ucb import WeightUCB


class DLinUCB(WeightUCB):
    """D-LinUCB of Russac et al. (NeurIPS 2019), Algorithm 1 and Eq. (5).

    The extra covariance uses squared discount weights. The confidence norm
    is ``sqrt(x.T V^-1 V_squared V^-1 x)`` and its radius likewise uses the
    sum of squared weights. No covariance is replaced with a pseudoinverse.
    """

    REQUIREMENTS = AlgorithmRequirements(
        links=("identity",), feedback_kinds=("additive",), subgaussian_radius=True
    )
    DEFAULTS = {**WeightUCB.DEFAULTS}

    def reset(self) -> None:
        super().reset()
        self._squared_covariance = self.regularization * np.eye(self.dimension)
        self._effective_squared_weight = 0.0

    @property
    def squared_covariance(self) -> np.ndarray:
        return self._squared_covariance.copy()

    @property
    def beta(self) -> float:
        return self._radius(self._effective_squared_weight)

    def _width_squared(self, candidates: np.ndarray) -> np.ndarray:
        solutions = np.linalg.solve(self._covariance, candidates.T).T
        return np.einsum("ij,jk,ik->i", solutions, self._squared_covariance, solutions)

    def update(self, x: np.ndarray, feedback: Feedback) -> None:
        feature = self._feature(x)
        super().update(feature, feedback)
        gamma_squared = self.gamma**2
        self._squared_covariance = (
            gamma_squared * self._squared_covariance
            + np.outer(feature, feature)
            + (1 - gamma_squared) * self.regularization * np.eye(self.dimension)
        )
        self._effective_squared_weight = gamma_squared * self._effective_squared_weight + 1
