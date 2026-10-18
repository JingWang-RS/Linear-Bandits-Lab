"""CRTM from NeurIPS'23: Efficient Algorithms for Generalized Linear Bandits with Heavy-tailed Rewards."""

import numpy as np

from algorithms.base import AlgorithmRequirements
from bandit_environments.reward_models.feedback import Feedback
from bandit_environments.reward_models.links import get_scalar_link

from .common import _HeavyBase


class CRTM(_HeavyBase):
    """Use truncated rewards to limit the effect of heavy-tailed observations.

    The truncation assumption concerns raw reward moments. Minkowski's
    inequality gives reward_moment_root_bound = U + moment_bound, where U
    bounds the scalar link on the public predictor interval [-L*S, L*S].
    """

    REQUIREMENTS = AlgorithmRequirements(
        links=("identity", "sigmoid"),
        feedback_kinds=("additive",),
        stationary_parameters=True,
        known_moment="global",
        scalar_link_parameter=True,
        fixed_repeated_actions=False,
    )
    DEFAULTS = _HeavyBase.DEFAULTS | {"truncation_scale": 1.0}

    def __init__(self, *, link="identity", truncation_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        if getattr(link, "name", link) not in ("identity", "sigmoid"):
            raise ValueError("CRTM supports only identity and sigmoid scalar links")
        self.link = get_scalar_link(link)
        self.link_name = self.link.name
        self.link_bounds = self.link.bounds(self.parameter_bound * self.action_bound)
        self.kappa = self.link_bounds.kappa
        self.mean_bound = self.link_bounds.mean_bound
        if self.action_bound > 1 or self.regularization < 1:
            raise ValueError(
                "CRTM scalar-link definition requires action_bound<=1 and regularization>=1"
            )
        if self.moment_bound is None:
            raise ValueError("CRTM requires a known global noise moment_bound")
        if not np.isfinite(truncation_scale) or truncation_scale <= 0:
            raise ValueError("truncation_scale must be finite and positive")
        if self.implementation == "theory" and truncation_scale != 1:
            raise ValueError("truncation tuning requires implementation='icml25'")
        if self.implementation == "icml25" and self.link_name != "identity":
            raise ValueError("the ICML2025 CRTM experiment uses the identity mean link")
        self.truncation_scale = float(truncation_scale)
        try:
            # Arithmetic failures become contextual errors, never ignored warnings.
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                p = 1 + self.moment_exponent
                log = np.log(4 * self.horizon / self.delta)
                # Keep the established identity arithmetic; the other path restores
                # every kappa factor in Algorithm1 / Theorem1 of the GLB paper.
                if self.link_name == "identity":
                    potential = self.dimension * np.log1p(
                        self.horizon / (2 * self.regularization * self.dimension)
                    )
                else:
                    potential = (
                        self.dimension
                        / self.kappa
                        * np.log1p(
                            self.kappa
                            * self.horizon
                            / (2 * self.regularization * self.dimension)
                        )
                    )
                self.reward_moment_root_bound = self.mean_bound + self.moment_bound
                self.truncation_threshold = (
                    2
                    * self.reward_moment_root_bound
                    / log ** (1 / p)
                    * np.sqrt(potential)
                    * self.horizon**self.q
                )
                self.gamma = (
                    224
                    * self.reward_moment_root_bound**2
                    * log ** (2 * self.moment_exponent / p)
                    * self.horizon ** (2 * self.q)
                    * 4
                    * potential
                    + 2 * self.regularization * self.parameter_bound**2
                    + 48 * self.mean_bound**2 * potential
                )
                if self.implementation == "icml25":
                    # Preserve the server experiment's empirical truncation
                    # and radius, rather than claiming these satisfy Theorem1.
                    self.truncation_threshold = (
                        2
                        * (self.moment_bound / log) ** (1 / p)
                        * np.sqrt(potential)
                        * self.horizon**self.q
                        / self.truncation_scale
                    )
                    self.gamma = (
                        (
                            self.dimension
                            * log ** (2 * self.moment_exponent / p)
                            * np.log(
                                self.horizon / self.dimension * self.regularization
                            )
                            + 1
                        )
                        * self.horizon ** (1 - self.moment_exponent)
                        / p
                    )
                reward_raw_moment_bound = self.reward_moment_root_bound**p
        except (OverflowError, FloatingPointError, ZeroDivisionError) as error:
            raise ValueError(
                f"CRTM confidence formula exceeds finite floating-point range: {error}"
            ) from error
        if not np.isfinite(
            [self.truncation_threshold, self.gamma, reward_raw_moment_bound]
        ).all():
            raise ValueError(
                "CRTM confidence formula exceeds finite floating-point range"
            )
        if self.gamma < 0:
            raise ValueError("CRTM confidence radius must be nonnegative")
        self.resolved_parameters = {
            "link_name": self.link_name,
            "kappa": self.kappa,
            "link_lipschitz": self.link_bounds.lipschitz,
            "mean_bound": self.mean_bound,
            "moment_order": p,
            "noise_moment_root_bound": self.moment_bound,
            "reward_moment_root_bound": self.reward_moment_root_bound,
            "reward_raw_moment_bound": reward_raw_moment_bound,
        }
        self.reset()

    @property
    def beta(self):
        if self.implementation == "icml25":
            return float(self.gamma / self.confidence_scale)
        return float(np.sqrt(self.gamma))

    def update(self, x, feedback: Feedback):
        x, reward = self._feature(x), self._reward(feedback)
        self._noise_root(feedback)
        # The source experiment used the covariance norm, not its inverse,
        # for empirical truncation. Keep that convention explicitly scoped.
        width = (
            np.sqrt(x @ self._covariance @ x)
            if self.implementation == "icml25"
            else self._width(x)
        )
        processed = reward if width * abs(reward) <= self.truncation_threshold else 0.0
        covariance = self._covariance + self.kappa / 2 * np.outer(x, x)
        prediction = x @ self._theta
        if self.link_name != "identity":
            prediction = float(self.link.mean(prediction))
        gradient = (prediction - processed) * x
        theta = self._projected_step(x, gradient, covariance)
        self._covariance, self._theta = covariance, theta
        self.count += 1
