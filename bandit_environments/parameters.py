"""Define the true reward parameters: constant for stationary bandits or rotating over time."""

import numpy as np


class ParameterProcess:
    def __init__(self, config: dict, horizon: int):
        self.config = config
        self.horizon = horizon
        self._initial = np.asarray(config["initial"], dtype=float)

    def at(self, t: int):
        theta = self._initial.copy()
        if self.config["kind"] == "rotation":
            angle = 2 * np.pi * self.config["turns"] * t / self.horizon
            first, second = theta[..., 0].copy(), theta[..., 1].copy()
            theta[..., 0] = np.cos(angle) * first - np.sin(angle) * second
            theta[..., 1] = np.sin(angle) * first + np.cos(angle) * second
        elif self.config["kind"] != "stationary":
            raise ValueError(f"unsupported parameter process {self.config['kind']!r}")
        return theta
