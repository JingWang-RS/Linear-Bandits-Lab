"""Generate linear, binary, or categorical rewards for the selected action."""

import numpy as np
from scipy.special import softmax

from .feedback import Feedback
from .links import get_scalar_link
from .noise import noise_mean, noise_moment_root, sample_centered_noise


class RewardModel:
    """Compute expected rewards and sample the observed reward for a selected action.

    The evaluator supplies the current parameter. Policies receive only Feedback,
    never the parameter or the conditional means of unchosen actions.
    """

    def __init__(self, mean_model: dict, feedback: dict, noise: dict, seed: int):
        self.mean_model = mean_model
        self.feedback_config = feedback
        self.noise = noise
        self._scalar_link = (
            get_scalar_link(mean_model["link"]) if mean_model["link"] != "softmax" else None
        )
        self._noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
        self._scale_rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
        self._feedback_round = 0
        self._unit_moment_root = noise_moment_root(
            noise["distribution"],
            noise["moment_exponent"],
            df=noise["degrees_of_freedom"],
            shape=noise["shape"],
            centered=noise.get("centered", True),
        )
        self._mean_offset = 0.0
        if not noise.get("centered", True):
            # The scale is drawn after action selection. Average it when
            # computing conditional reward means; the offset is common to all arms.
            span = noise["max_log10_scale"] * np.log(10)
            average_scale = np.expm1(span) / span if span else 1.0
            self._mean_offset = noise["scale"] * average_scale * noise_mean(
                noise["distribution"], shape=noise["shape"]
            )

    def probabilities(self, features: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        if self.mean_model["link"] == "sigmoid":
            positive = self._scalar_link.mean(features @ parameter)
            return np.column_stack((1 - positive, positive))
        logits = features @ parameter.T
        return softmax(np.column_stack((np.zeros(len(features)), logits)), axis=1)

    def expected_rewards(self, features: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        if self.feedback_config["kind"] == "additive":
            values = self._scalar_link.mean(features @ parameter)
            return values + self._mean_offset if self._mean_offset else values
        return self.probabilities(features, parameter) @ np.asarray(
            self.feedback_config["reward_vector"]
        )

    def sample(self, t: int, features: np.ndarray, parameter: np.ndarray, index: int) -> Feedback:
        if not 0 <= index < len(features):
            raise ValueError("selected candidate index out of range")
        if t != self._feedback_round:
            raise ValueError("exactly one selected feedback is sampled per consecutive round")
        self._feedback_round += 1
        kind = self.feedback_config["kind"]
        if kind in {"bernoulli", "categorical"}:
            cdf = np.cumsum(self.probabilities(features, parameter)[index])
            cdf[-1] = 1.0
            outcome = int(np.searchsorted(cdf, self._noise_rng.uniform(), side="right"))
            return Feedback(
                reward=float(self.feedback_config["reward_vector"][outcome]), outcome=outcome
            )
        if kind != "additive":
            raise ValueError(f"unsupported feedback kind {kind!r}")
        mean = float(self._scalar_link.mean(features @ parameter)[index])
        noise = self.noise
        log_scale = (
            self._scale_rng.uniform(0, noise["max_log10_scale"])
            if noise["scale_mode"] == "log-uniform"
            else 0.0
        )
        scale = noise["scale"] * 10.0**log_scale
        sample = sample_centered_noise(
            self._noise_rng,
            noise["distribution"],
            df=noise["degrees_of_freedom"],
            shape=noise["shape"],
            centered=noise.get("centered", True),
        )
        root = float(scale * self._unit_moment_root) if noise["known_moment"] else None
        return Feedback(reward=mean + scale * sample, noise_moment_root=root)
