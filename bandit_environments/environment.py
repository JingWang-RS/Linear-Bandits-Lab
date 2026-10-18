"""Define a bandit environment: available actions, observed rewards, and expected rewards for evaluation."""

import numpy as np

from .actions import ActionProcess
from .parameters import ParameterProcess
from .reward_models import Feedback, RewardModel


class SyntheticEnvironment:
    """Provide actions and rewards for one trial, and compute expected rewards for regret."""

    def __init__(self, config: dict, horizon: int, seed: int):
        self.config = config
        self.horizon = horizon
        self.seed = seed
        self._actions = ActionProcess(config["dimension"], config["action_process"], horizon, seed)
        self._parameters = ParameterProcess(config["parameter_process"], horizon)
        self._reward_model = RewardModel(
            config["mean_model"], config["feedback"], config["noise"], seed
        )

    def context(self, t: int):
        return self._actions.context(t)

    def expected_rewards(self, t: int, features: np.ndarray) -> np.ndarray:
        """Expected rewards used to measure regret; these are not given to the algorithm."""
        return self._reward_model.expected_rewards(features, self._parameters.at(t))

    def feedback(self, t: int, features: np.ndarray, index: int) -> Feedback:
        return self._reward_model.sample(t, features, self._parameters.at(t), index)
