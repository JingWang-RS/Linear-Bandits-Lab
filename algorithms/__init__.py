"""Available algorithms and their source papers. Use create_algorithm to build one from a config."""

import re

import numpy as np

from algorithms.base import Policy, AlgorithmRequirements
from algorithms.generalized_linear import GLMUCB, MNLUCB, OL2M, MLogBUCBVariant
from algorithms.heavy_tailed import CRMM, CRTM, HeavyOFUL, HvtUCB
from algorithms.linear import OFUL
from algorithms.non_stationary import (
    BVDGLMUCB, DLinUCB, GLBRestartUCB, GLBStaticUCB, GLBWeightUCB,
    MasterGLB, MasterOFUL, MasterSCB, RestartUCB,
    SCBRestartUCB, SCBStaticUCB, SCBWeightUCB, SlidingWindowUCB, WeightUCB,
)

ALGORITHMS = {
    "oful": OFUL,
    "restart-ucb": RestartUCB,
    "sliding-window-ucb": SlidingWindowUCB,
    "d-linucb": DLinUCB,
    "weight-ucb": WeightUCB,
    "master-oful": MasterOFUL,
    "master-glb": MasterGLB,
    "master-scb": MasterSCB,
    "glb-static-ucb": GLBStaticUCB,
    "glb-restart-ucb": GLBRestartUCB,
    "glb-weight-ucb": GLBWeightUCB,
    "bvd-glm-ucb": BVDGLMUCB,
    "scb-static-ucb": SCBStaticUCB,
    "scb-restart-ucb": SCBRestartUCB,
    "scb-weight-ucb": SCBWeightUCB,
    "hvt-ucb": HvtUCB,
    "heavy-oful": HeavyOFUL,
    "crtm": CRTM,
    "crmm": CRMM,
    "glm-ucb-regularized": GLMUCB,
    "ol2m": OL2M,
    "mnl-ucb-local": MNLUCB,
    "mlogb-ucb-variant": MLogBUCBVariant,
}
REFERENCES = {
    "oful": "Improved Algorithms for Linear Stochastic Bandits (NeurIPS 2011)",
    "restart-ucb": "A Simple Approach for Non-stationary Linear Bandits (AISTATS 2020)",
    "sliding-window-ucb": "Learning to Optimize under Non-Stationarity (AISTATS 2019)",
    "d-linucb": "Weighted Linear Bandits for Non-Stationary Environments (NeurIPS 2019)",
    "weight-ucb": "Revisiting Weighted Strategy for Non-stationary Parametric Bandits and MDPs (TIT 2026)",
    "master-oful": "Non-stationary Reinforcement Learning without Prior Knowledge: An Optimal Black-box Approach (COLT 2021)",
    "master-glb": "Non-stationary Reinforcement Learning without Prior Knowledge: An Optimal Black-box Approach (COLT 2021)",
    "master-scb": "Non-stationary Reinforcement Learning without Prior Knowledge: An Optimal Black-box Approach (COLT 2021)",
    "glb-static-ucb": "Parametric Bandits: The Generalized Linear Case (NeurIPS 2010)",
    "glb-restart-ucb": "A Simple Approach for Non-stationary Linear Bandits (AISTATS 2020)",
    "glb-weight-ucb": "Revisiting Weighted Strategy for Non-stationary Parametric Bandits and MDPs (TIT 2026)",
    "bvd-glm-ucb": "Regret Bounds for Generalized Linear Bandits under Parameter Drift (2021)",
    "scb-static-ucb": "Improved Optimistic Algorithms for Logistic Bandits (ICML 2020)",
    "scb-restart-ucb": "A Simple Approach for Non-stationary Linear Bandits (AISTATS 2020)",
    "scb-weight-ucb": "Revisiting Weighted Strategy for Non-stationary Parametric Bandits and MDPs (TIT 2026)",
    "hvt-ucb": "Heavy-Tailed Linear Bandits: Huber Regression with One-Pass Update (ICML 2025)",
    "heavy-oful": "Tackling Heavy-Tailed Rewards in Reinforcement Learning with Function Approximation: Minimax Optimal and Instance-Dependent Regret Bounds (NeurIPS 2023)",
    "crtm": "Efficient Algorithms for Generalized Linear Bandits with Heavy-tailed Rewards (NeurIPS 2023)",
    "crmm": "Efficient Algorithms for Generalized Linear Bandits with Heavy-tailed Rewards (NeurIPS 2023)",
    "glm-ucb-regularized": "Parametric Bandits: The Generalized Linear Case (NeurIPS 2010)",
    "ol2m": "Online Stochastic Linear Optimization under One-bit Feedback (ICML 2016)",
    "mnl-ucb-local": "UCB-based Algorithms for Multinomial Logistic Regression Bandits (NeurIPS 2021)",
    "mlogb-ucb-variant": "Online (Multinomial) Logistic Bandit: Improved Regret and Constant Computation Cost (NeurIPS 2023)"
}


def _requirements(cls, name):
    if not isinstance(cls, type) or not issubclass(cls, Policy):
        raise ValueError(f"{name}: registered algorithm must inherit Policy")
    requirements = getattr(cls, "REQUIREMENTS", None)
    if not isinstance(requirements, AlgorithmRequirements):
        raise ValueError(f"{name}: registered algorithm must declare AlgorithmRequirements")
    return requirements


def resolve_algorithms(entries: list, environment: dict, horizon: int) -> list[dict]:
    if not isinstance(entries, list) or not entries:
        raise ValueError("algorithms must be a nonempty list")
    resolved, names = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {
            "name",
            "algorithm",
            "parameters",
            "derived_parameters",
            "definition",
            "reference",
            "assumption_notes",
        }:
            raise ValueError(
                "each algorithm accepts name, algorithm and parameters; derived fields are read-only"
            )
        identifier = entry.get("algorithm")
        if not isinstance(identifier, str) or identifier not in ALGORITHMS:
            raise ValueError(
                f"{identifier!r}: unknown algorithm; see ALGORITHMS in algorithms/__init__.py"
            )
        name = entry.get("name", identifier)
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError("algorithm name must be a filesystem-safe identifier")
        if name in names:
            raise ValueError(f"duplicate algorithm instance name {name!r}")
        names.add(name)
        cls = ALGORITHMS[identifier]
        requirements = _requirements(cls, name)
        params = dict(cls.DEFAULTS)
        params.update(requirements.environment_defaults(environment))
        supplied = entry.get("parameters", {})
        if not isinstance(supplied, dict) or set(supplied) - set(params):
            raise ValueError(f"{name}: unknown algorithm parameter")
        if any(isinstance(value, bool) for value in supplied.values()):
            raise ValueError(f"{name}: numeric algorithm parameters cannot be booleans")
        params.update(supplied)
        assumptions = requirements.validate(environment, params, name)
        reference = REFERENCES[identifier]
        if params.get("implementation") == "icml25":
            assumptions.append(
                "empirically tuned experiment variant; noise metadata retains its true moment, "
                "while confidence and scale multipliers are tuning parameters, not theorem bounds"
            )
        try:
            policy = cls(
                **requirements.constructor_arguments(environment, horizon),
                **params,
                rng=np.random.default_rng(0),
            )
        except (TypeError, ValueError, ArithmeticError) as exc:
            raise ValueError(f"{name}: invalid algorithm configuration: {exc}") from exc
        derived = dict(getattr(policy, "resolved_parameters", {}))
        for key, value in params.items():
            if value is None and key in derived:
                params[key] = derived[key]
        resolved.append(
            {
                "name": name,
                "algorithm": identifier,
                "parameters": params,
                "derived_parameters": derived,
                "reference": reference,
                "assumption_notes": assumptions,
            }
        )
    return resolved


def create_algorithm(entry: dict, environment: dict, horizon: int, rng: np.random.Generator):
    cls = ALGORITHMS[entry["algorithm"]]
    requirements = _requirements(cls, entry["name"])
    return cls(
        **requirements.constructor_arguments(environment, horizon),
        rng=rng,
        **entry["parameters"],
    )
