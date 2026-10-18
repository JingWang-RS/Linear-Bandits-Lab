"""CRMM from NeurIPS'23: Efficient Algorithms for Generalized Linear Bandits with Heavy-tailed Rewards."""

import numpy as np

from algorithms.base import AlgorithmRequirements
from bandit_environments.reward_models.feedback import Feedback
from bandit_environments.reward_models.links import get_scalar_link

from .common import _HeavyBase


class CRMM(_HeavyBase):
    """Repeat an action and combine its rewards using the mean-of-medians estimator.

    Normally the pool stays fixed and repetitions refer to the same action.
    ``experiment_mode='icml25-appendix'`` enables the paper's empirical stress
    tests: if that action disappears, repeat the nearest available action.
    Such batches can mix actions and are outside CRMM's theoretical guarantee.
    """

    REQUIREMENTS = AlgorithmRequirements(
        links=("identity", "sigmoid"),
        feedback_kinds=("additive",),
        stationary_parameters=True,
        known_moment="global",
        scalar_link_parameter=True,
        fixed_repeated_actions=True,
    )

    DEFAULTS = _HeavyBase.DEFAULTS | {"experiment_mode": "standard"}

    def __init__(self, *, link="identity", experiment_mode="standard", **kwargs):
        if experiment_mode not in {"standard", "icml25-appendix"}:
            raise ValueError("experiment_mode must be 'standard' or 'icml25-appendix'")
        self.experiment_mode = experiment_mode
        super().__init__(**kwargs)
        if experiment_mode == "icml25-appendix" and self.implementation != "icml25":
            raise ValueError("appendix experiments require implementation='icml25'")
        if getattr(link, "name", link) not in ("identity", "sigmoid"):
            raise ValueError("CRMM supports only identity and sigmoid scalar links")
        self.link = get_scalar_link(link)
        self.link_name = self.link.name
        self.link_bounds = self.link.bounds(self.parameter_bound * self.action_bound)
        self.kappa = self.link_bounds.kappa
        self.mean_bound = self.link_bounds.mean_bound
        if self.action_bound > 1 or self.regularization < 1:
            raise ValueError(
                "CRMM scalar-link definition requires action_bound<=1 and regularization>=1"
            )
        if self.moment_bound is None:
            raise ValueError("CRMM requires a known global noise moment_bound")
        if self.implementation == "icml25" and self.link_name != "identity":
            raise ValueError("the ICML2025 CRMM experiment uses the identity mean link")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                initial_radius = np.sqrt(self.regularization) * self.parameter_bound
                p, log = 1 + self.moment_exponent, np.log(4 * self.horizon / self.delta)
                self.batch_size = int(np.ceil(16 * log))
                v = self.moment_bound**p
                self.C = (4 * v) ** (1 / p)
                # Continuous limit at deterministic noise avoids the undefined 0**(-epsilon).
                self.rho = (
                    0.0
                    if v == 0
                    else 2 * self.C * log
                    + 2 * self.C ** (-self.moment_exponent) * self.batch_size * v
                )
        except (OverflowError, FloatingPointError, ZeroDivisionError) as error:
            raise ValueError(
                f"CRMM confidence formula exceeds finite floating-point range: {error}"
            ) from error
        if not np.isfinite([initial_radius, v, self.C, self.rho]).all():
            raise ValueError(
                "CRMM confidence formula exceeds finite floating-point range"
            )
        self.resolved_parameters = {
            "link_name": self.link_name,
            "kappa": self.kappa,
            "link_lipschitz": self.link_bounds.lipschitz,
            "mean_bound": self.mean_bound,
            "moment_order": p,
            "noise_moment_root_bound": self.moment_bound,
            "noise_raw_moment_bound": v,
            "batch_size": self.batch_size,
            "C": self.C,
            "rho": self.rho,
        }
        # Confidence is monotone in completed batches; check the configured
        # endpoint before accepting a run with unrepresentable formula values.
        final_batches = self.horizon // self.batch_size
        if final_batches:
            self._confidence_gamma(final_batches)
        self.reset()
        if self.implementation == "icml25":
            # Validate the smallest empirical radius at construction time.
            _ = self.beta

    def reset(self):
        super().reset()
        self.completed_batches = 0
        self.pending_rewards = []
        self.pending_arm_id = None
        self._pending_feature = None
        self._selected_feature = None
        self._action_catalog = None

    @property
    def beta(self):
        if self.implementation == "icml25":
            # Historical empirical radius, but with exact r-observation
            # batches: the old implementation discarded every (r+1)th reward.
            t = self.completed_batches + 1
            radius = (
                (self.dimension * np.log(t / self.dimension * self.regularization) + 1)
                * t ** (2 * self.q)
                / self.confidence_scale
            )
            if not np.isfinite(radius) or radius < 0:
                raise ValueError(
                    "ICML2025 CRMM empirical confidence radius must be nonnegative"
                )
            return float(radius)
        t = self.completed_batches
        if t == 0:
            return float(np.sqrt(self.regularization) * self.parameter_bound)
        return float(np.sqrt(self._confidence_gamma(t)))

    def _confidence_gamma(self, t):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if self.link_name == "identity":
                    potential = self.dimension * np.log1p(
                        t / (2 * self.regularization * self.dimension)
                    )
                    noise_term = 2 * self.rho**2 * t ** (2 * self.q)
                else:
                    potential = (
                        self.dimension
                        / self.kappa
                        * np.log1p(
                            self.kappa * t / (2 * self.regularization * self.dimension)
                        )
                    )
                    noise_term = 2 * self.rho**2 / self.kappa * t ** (2 * self.q)
                gamma = (
                    (4 * self.mean_bound**2 + self.C * self.rho * t ** (2 * self.q))
                    * 4
                    * potential
                    + self.regularization * self.parameter_bound**2
                    + noise_term
                )
        except (OverflowError, FloatingPointError, ZeroDivisionError) as error:
            raise ValueError(
                f"CRMM confidence formula at batch {t} exceeds finite floating-point range: {error}"
            ) from error
        if not np.isfinite(gamma):
            raise ValueError(
                f"CRMM confidence formula at batch {t} exceeds finite floating-point range"
            )
        return gamma

    def select(self, features, arm_ids):
        features, ids = self._candidates(features, arm_ids)
        catalog = {key: x.copy() for key, x in zip(ids.tolist(), features, strict=True)}
        if len(catalog) != len(features):
            raise ValueError("CRMM requires unique stable action identities")
        if self._action_catalog is None:
            self._action_catalog = catalog
        elif self.experiment_mode == "icml25-appendix":
            for key, feature in catalog.items():
                if key in self._action_catalog and not np.array_equal(feature, self._action_catalog[key]):
                    raise ValueError("an action identity cannot change its features")
            self._action_catalog.update(catalog)
        elif catalog.keys() != self._action_catalog.keys() or any(
            not np.array_equal(catalog[key], self._action_catalog[key])
            for key in catalog
        ):
            raise ValueError(
                "CRMM requires a fixed action pool with stable features and identities"
            )
        if self.pending_arm_id is None:
            chosen = super().select(features, ids)
            chosen_id = ids[chosen]
            self.pending_arm_id = (
                chosen_id.item() if isinstance(chosen_id, np.generic) else chosen_id
            )
            self._pending_feature = features[chosen].copy()
        matches = np.flatnonzero(ids == self.pending_arm_id)
        if self.experiment_mode == "icml25-appendix":
            if len(matches):
                chosen = int(matches[0])
            else:
                distances = np.linalg.norm(features - self._pending_feature, axis=1)
                nearest = np.flatnonzero(distances == distances.min())
                # Stable identities make a distance tie independent of array order.
                chosen = int(nearest[np.argmin(ids[nearest])])
            self._selected_feature = features[chosen].copy()
            return chosen
        if len(matches) != 1:
            raise ValueError(
                "CRMM pending action identity must remain available exactly once"
            )
        return int(matches[0])

    def update(self, x, feedback: Feedback):
        x, reward = self._feature(x), self._reward(feedback)
        self._noise_root(feedback)
        if self._pending_feature is None:
            raise ValueError("CRMM select must precede update")
        expected = (self._selected_feature if self.experiment_mode == "icml25-appendix"
                    else self._pending_feature)
        if not np.array_equal(x, expected):
            raise ValueError(
                "CRMM update must use the selected action identity's unchanged features"
            )
        observations = self.pending_rewards + [reward]
        if len(observations) == self.batch_size:
            # In the appendix variant, use the last actually played feature,
            # as in the original batch update; never treat its mixed rewards
            # as valid repetitions of one arm in the theoretical implementation.
            covariance = self._covariance + self.kappa / 2 * np.outer(x, x)
            prediction = x @ self._theta
            if self.link_name != "identity":
                prediction = float(self.link.mean(prediction))
            gradient = (prediction - float(np.median(observations))) * x
            theta = self._projected_step(x, gradient, covariance)
            self._covariance, self._theta = covariance, theta
            self.completed_batches += 1
            self.pending_rewards = []
            self.pending_arm_id, self._pending_feature = None, None
            self._selected_feature = None
        else:
            self.pending_rewards = observations
        self.count += 1
