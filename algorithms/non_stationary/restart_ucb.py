"""RestartUCB from AISTATS'20: A Simple Approach for Non-stationary Linear Bandits."""

import numpy as np

from algorithms.linear import OFUL
from algorithms.base import AlgorithmRequirements
from bandit_environments.reward_models.feedback import Feedback

from .common import _positive_integer


class RestartUCB(OFUL):
    """Restart OFUL after exactly ``restart_interval`` observations.

    The last (possibly shorter) epoch requires no special horizon handling.
    ``count`` denotes observations retained in the current epoch; the total
    observations and completed restarts are available separately.
    """

    REQUIREMENTS = AlgorithmRequirements(
        links=("identity",), feedback_kinds=("additive",), subgaussian_radius=True
    )
    DEFAULTS = {**OFUL.DEFAULTS, "restart_interval": 30}

    def __init__(self, *, restart_interval: int = 30, **kwargs):
        self.restart_interval = _positive_integer(restart_interval, "restart_interval")
        super().__init__(**kwargs)

    def reset(self) -> None:
        super().reset()
        self.total_updates = 0
        self.restarts = 0

    def update(self, x: np.ndarray, feedback: Feedback) -> None:
        super().update(x, feedback)
        self.total_updates += 1
        if self.count == self.restart_interval:
            super().reset()
            self.restarts += 1
