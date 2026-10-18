"""The common algorithm interface and checks for compatible bandit environments."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import numpy as np

from bandit_environments.reward_models.feedback import Feedback
from bandit_environments.reward_models.links import get_scalar_link


class Policy(ABC):
    """Implement scores, update, and reset to add a bandit algorithm.

    scores returns one value per available action. The default select method
    chooses a highest-scoring action and breaks ties with the algorithm's RNG.
    update receives only the chosen action and its observed reward. reset starts
    a fresh trial. Each implementation sets dimension and rng in its constructor.
    """

    @abstractmethod
    def update(self, x: np.ndarray, feedback: Feedback) -> None:
        """Update the estimate and related state using only selected feedback."""

    @abstractmethod
    def scores(self, features: np.ndarray) -> np.ndarray:
        """Return an algorithm-specific decision value for each candidate."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all estimator, history and decision state."""

    def select(self, features: np.ndarray, arm_ids: np.ndarray) -> int:
        """Choose among maximum scores; return the current candidate index."""
        candidates = np.asarray(features, dtype=float)
        if candidates.ndim != 2 or candidates.shape[1] != self.dimension or len(candidates) == 0:
            raise ValueError(f"candidate shape must be (n, {self.dimension}) with n >= 1")
        if np.asarray(arm_ids).shape != (len(candidates),):
            raise ValueError("arm_ids shape must match candidate count")
        if not np.isfinite(candidates).all():
            raise ValueError("candidate features must be finite")
        values = np.asarray(self.scores(candidates), dtype=float)
        if values.shape != (len(candidates),):
            raise ValueError("policy scores must have one scalar per candidate")
        if not np.isfinite(values).all():
            raise ArithmeticError("nonfinite decision scores")
        maximizers = np.flatnonzero(values == values.max())
        return int(self.rng.choice(maximizers))


@dataclass(frozen=True)
class AlgorithmRequirements:
    """Declare which reward models and assumptions an algorithm supports."""

    links: tuple[str, ...]
    feedback_kinds: tuple[str, ...]
    stationary_parameters: bool = False
    subgaussian_radius: bool = False
    known_moment: str = "none"
    categorical_parameters: bool = False
    scalar_link_parameter: bool = False
    fixed_repeated_actions: bool = False
    drift_baseline_note: str | None = None

    def as_dict(self):
        return asdict(self)

    def constructor_arguments(self, environment, horizon):
        """Pass dimensions and known bounds to the algorithm; keep true parameters hidden."""
        parameter = environment["parameter_process"]
        actions = environment["action_process"]
        common = {
            "dimension": environment["dimension"],
            "horizon": horizon,
            "parameter_bound": parameter["bound"],
            "action_bound": actions["bound"],
        }
        if self.categorical_parameters:
            common.update(
                outcomes=environment["mean_model"].get("outcomes", 1),
                reward_vector=environment["feedback"]["reward_vector"],
            )
        if self.scalar_link_parameter:
            common["link"] = get_scalar_link(environment["mean_model"]["link"])
        return common

    def environment_defaults(self, environment):
        noise = environment["noise"]
        defaults = {}
        if self.subgaussian_radius:
            defaults["noise_scale"] = noise["scale"] * 10.0 ** noise["max_log10_scale"]
        if self.known_moment != "none":
            defaults["moment_exponent"] = noise["moment_exponent"]
        return defaults

    def validate(self, environment, parameters, name):
        link = environment["mean_model"]["link"]
        feedback = environment["feedback"]["kind"]
        parameter = environment["parameter_process"]
        noise = environment["noise"]
        if link not in self.links:
            raise ValueError(f"{name}: mean link {link!r} is incompatible; requires {self.links}")
        if feedback not in self.feedback_kinds:
            raise ValueError(
                f"{name}: feedback {feedback!r} is incompatible; requires {self.feedback_kinds}"
            )
        if self.stationary_parameters and parameter["kind"] != "stationary":
            raise ValueError(f"{name} currently requires stationary parameters")
        notes = []
        if not noise.get("centered", True):
            if parameters.get("implementation") != "icml25" and not self.subgaussian_radius:
                raise ValueError(f"{name}: uncentered noise requires an explicit empirical implementation")
            notes.append(
                "ICML appendix empirical setting: positive, nonzero-mean noise; "
                "outside the centered-noise guarantees. Evaluation includes its common mean offset."
            )
        if parameter["kind"] != "stationary" and self.drift_baseline_note:
            notes.append(self.drift_baseline_note)
        if self.subgaussian_radius:
            if noise["distribution"] in {"gaussian", "none"}:
                required_scale = noise["scale"] * 10.0 ** noise["max_log10_scale"]
                scale = parameters["noise_scale"]
                if not isinstance(scale, (int, float)) or scale < required_scale:
                    raise ValueError(
                        f"{name}: noise_scale must bound the configured Gaussian scale"
                    )
            else:
                notes.append(
                    "sub-Gaussian radius is misspecified for heavy-tailed noise; empirical baseline only"
                )
        if self.known_moment != "none":
            if parameters["moment_exponent"] != noise["moment_exponent"]:
                raise ValueError(
                    f"{name}: moment_exponent must match environment noise metadata order"
                )
            bound = parameters["moment_bound"]
            if self.known_moment == "global" and bound is None:
                if not noise["known_moment"]:
                    raise ValueError(
                        f"{name} requires a known global moment_bound or known_moment=true"
                    )
                parameters["moment_bound"] = bound = noise["moment_root_bound"]
            if bound is not None:
                if (
                    not isinstance(bound, (int, float))
                    or not np.isfinite(bound)
                    or bound < noise["moment_root_bound"] * (1 - 1e-12)
                ):
                    raise ValueError(
                        f"{name}: moment_bound must bound the configured NOISE moment root"
                    )
            elif not noise["known_moment"]:
                raise ValueError(
                    f"{name} requires known selected noise metadata or a global moment_bound"
                )
        if self.fixed_repeated_actions:
            appendix = parameters.get("experiment_mode") == "icml25-appendix"
            if appendix:
                if parameters.get("implementation") != "icml25":
                    raise ValueError("CRMM appendix experiments require implementation='icml25'")
                notes.append(
                    "ICML appendix empirical CRMM: asymmetric/time-varying noise is allowed; "
                    "an unavailable repeated arm is replaced by its nearest available arm. "
                    "Mixed-arm batches have no CRMM theorem guarantee."
                )
            if not appendix and environment["action_process"]["mode"] != "fixed":
                raise ValueError(
                    "CRMM requires a fixed stable action pool; varying subsets/resampling rejected"
                )
            if not appendix and noise["distribution"] not in {"gaussian", "student-t", "none"}:
                raise ValueError("CRMM requires symmetric centered noise")
            if not appendix and noise["scale_mode"] != "constant":
                raise ValueError("CRMM currently requires stationary noise scale")
        return notes
