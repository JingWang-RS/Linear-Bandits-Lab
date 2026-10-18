"""Functions that turn an action score into an expected reward, such as identity and sigmoid."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class LinkBounds:
    kappa: float
    lipschitz: float
    mean_bound: float


@dataclass(frozen=True)
class ScalarLink:
    name: str

    def __post_init__(self):
        if self.name not in {"identity", "sigmoid"}:
            raise ValueError(f"unsupported scalar link {self.name!r}")

    def mean(self, value):
        values = np.asarray(value, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("link input must be finite")
        return values if self.name == "identity" else expit(values)

    def derivative(self, value):
        values = np.asarray(value, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("link input must be finite")
        if self.name == "identity":
            return np.ones_like(values)
        return expit(values) * expit(-values)

    def bounds(self, predictor_bound):
        if (
            isinstance(predictor_bound, bool)
            or not np.isfinite(predictor_bound)
            or predictor_bound < 0
        ):
            raise ValueError("predictor_bound must be finite and nonnegative")
        if self.name == "identity":
            return LinkBounds(1.0, 1.0, float(predictor_bound))
        kappa = float(expit(predictor_bound) * expit(-predictor_bound))
        if kappa <= 0:
            raise ValueError("sigmoid derivative lower bound is not numerically positive")
        return LinkBounds(kappa, 0.25, float(expit(predictor_bound)))


def get_scalar_link(name):
    """Choose the identity or sigmoid function for the expected reward."""
    return name if isinstance(name, ScalarLink) else ScalarLink(name)
