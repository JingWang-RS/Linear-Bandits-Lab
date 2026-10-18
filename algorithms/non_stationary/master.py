"""MASTER from COLT'21: Non-stationary Reinforcement Learning without Prior Knowledge: An Optimal Black-box Approach."""

from dataclasses import dataclass, replace
import heapq
from math import log, log2, sqrt

import numpy as np

from algorithms.base import Policy
from algorithms.linear import OFUL
from bandit_environments.reward_models.feedback import Feedback

from .glb import GLBStaticUCB
from .scb import SCBStaticUCB


@dataclass
class _ScheduledPolicy:
    """A base algorithm together with its starting and ending rounds."""

    start: int
    end: int
    policy: Policy

    @property
    def length(self) -> int:
        return self.end - self.start + 1


class _MASTER(Policy):
    """Run base algorithms over different time intervals and restart when changes are detected.

    The schedule, shortest-active-instance rule, and the two change tests follow
    the original ``NS_Bandits/algorithms/MASTER.py``. In particular, the change
    tests use the base policy's *unclipped* optimistic score. A base learner is
    updated only when it controls the selected action.

    All schedule draws and random tie breakers share this policy's local RNG.
    Sampling proceeds in the original order: increasing start time, then
    decreasing interval length. The last inserted instance wins equal-length
    schedule ties. Arm ties retain the original random lexicographic rule.
    """

    BASE_CLASS = OFUL

    def __init__(
        self,
        *,
        dimension: int,
        horizon: int,
        parameter_bound: float,
        action_bound: float,
        rng: np.random.Generator,
        n_init: int = 0,
        **base_parameters,
    ):
        if (
            isinstance(n_init, bool)
            or not isinstance(n_init, (int, np.integer))
            or n_init < 0
        ):
            raise ValueError("n_init must be a nonnegative integer")
        self.n_init = int(n_init)
        self._base_arguments = {
            "dimension": dimension,
            "horizon": horizon,
            "parameter_bound": parameter_bound,
            "action_bound": action_bound,
            "rng": rng,
            **base_parameters,
        }
        # The base constructor validates its own mathematical assumptions.
        prototype = self.BASE_CLASS(**self._base_arguments)
        self.dimension = int(dimension)
        self.horizon = int(horizon)
        self.parameter_bound = float(parameter_bound)
        self.action_bound = float(action_bound)
        self.rng = rng
        self.delta = prototype.delta
        self.resolved_parameters = dict(getattr(prototype, "resolved_parameters", {}))
        self.c = log(self.horizon / self.delta)
        self.n_hat = log2(self.horizon) + 1
        self.reset()

    def rho(self, length: int) -> float:
        return 4 * sqrt(self.dimension * self.c) * sqrt(self.dimension * self.c / length)

    def rho_hat(self, length: int) -> float:
        return 6 * self.n_hat * self.c * self.rho(length)

    def reset(self) -> None:
        self.t = 1
        self.n = self.n_init
        self.tn = self.t
        self.total_updates = 0
        self.restart_count = 0
        self.detected_changes = 0
        self.theta_hat = np.zeros(self.dimension)
        self._pending = None
        self._restart_epoch()

    @property
    def schedule(self) -> tuple[tuple[int, int], ...]:
        """Inclusive intervals, in original insertion order, for inspection."""
        return tuple((entry.start, entry.end) for entry in self._schedule)

    @property
    def base_policies(self) -> tuple[Policy, ...]:
        return tuple(entry.policy for entry in self._schedule)

    @property
    def optimistic_scores(self) -> np.ndarray:
        return self._optimism[: self._epoch_count].copy()

    @property
    def rewards(self) -> np.ndarray:
        return self._rewards[: self._epoch_count].copy()

    def _restart_epoch(self) -> None:
        self.tn = self.t
        self._epoch_count = 0
        length = 2**self.n
        self._optimism = np.empty(length)
        self._rewards = np.empty(length)
        self._minimum_optimism = np.inf
        self._schedule = []
        self._ending = {}
        self._active = []
        self._next_start = 0
        self.alg_index = 0
        numerator = self.rho(length)
        for start in range(length):
            # Exactly the pairs visited by the original divisibility test,
            # without scanning all n + 1 scales for every start time.
            maximum_scale = self.n if start == 0 else min(self.n, (start & -start).bit_length() - 1)
            for scale in range(maximum_scale, -1, -1):
                run_length = 2**scale
                probability = numerator / self.rho(run_length)
                if self.rng.binomial(n=1, p=probability):
                    entry = _ScheduledPolicy(
                        start=start,
                        end=start + run_length - 1,
                        policy=self.BASE_CLASS(**self._base_arguments),
                    )
                    self._ending.setdefault(entry.end, []).append(len(self._schedule))
                    self._schedule.append(entry)

    def _active_policy(self) -> Policy:
        offset = self.t - self.tn
        while self._next_start < len(self._schedule):
            index = self._next_start
            entry = self._schedule[index]
            if entry.start > offset:
                break
            heapq.heappush(self._active, (entry.length, -index, entry.end, index))
            self._next_start += 1
        while self._active and self._active[0][2] < offset:
            heapq.heappop(self._active)
        if not self._active:
            raise RuntimeError("MASTER schedule has no active base learner")
        self.alg_index = self._active[0][3]
        return self._schedule[self.alg_index].policy

    def _candidates(self, features: np.ndarray) -> np.ndarray:
        candidates = np.asarray(features, dtype=float)
        if candidates.ndim != 2 or candidates.shape[1] != self.dimension or not len(candidates):
            raise ValueError(f"candidate shape must be (n, {self.dimension}) with n >= 1")
        if not np.isfinite(candidates).all():
            raise ValueError("candidate features must be finite")
        if np.any(np.linalg.norm(candidates, axis=1) > self.action_bound * (1 + 1e-12)):
            raise ValueError("candidate norm exceeds action_bound")
        return candidates

    def scores(self, features: np.ndarray) -> np.ndarray:
        candidates = self._candidates(features)
        values = np.asarray(self._active_policy().scores(candidates), dtype=float)
        if values.shape != (len(candidates),) or not np.isfinite(values).all():
            raise ArithmeticError("MASTER base learner returned invalid confidence scores")
        return values

    def select(self, features: np.ndarray, arm_ids: np.ndarray) -> int:
        if self._pending is not None:
            raise RuntimeError("MASTER requires selected feedback before the next selection")
        candidates = self._candidates(features)
        if np.asarray(arm_ids).shape != (len(candidates),):
            raise ValueError("arm_ids shape must match candidate count")
        values = self.scores(candidates)
        # Consume one uniform per candidate, as in the source implementation;
        # changing this to choice(maximizers) also changes future schedules.
        chosen = int(np.lexsort((self.rng.random(len(values)), values))[-1])
        base = self._schedule[self.alg_index].policy
        self.theta_hat = np.asarray(base.theta_hat, dtype=float).copy()
        self._pending = (base, float(values.max()))
        return chosen

    def test1(self) -> bool:
        """Detect a completed interval whose observed mean exceeds optimism."""
        offset = self.t - self.tn
        for index in self._ending.get(offset, ()):
            entry = self._schedule[index]
            observed_mean = self._rewards[entry.start : entry.end + 1].sum() / entry.length
            if observed_mean >= self._minimum_optimism + 9 * self.rho_hat(entry.length):
                return False
        return True

    def test2(self) -> bool:
        """Detect an epoch-wide optimism gap using the source threshold."""
        count = self._epoch_count
        gap = (self._optimism[:count] - self._rewards[:count]).sum() / count
        return bool(gap < 3 * self.rho_hat(count))

    def update(self, x: np.ndarray, feedback: Feedback) -> None:
        if self._pending is None:
            raise RuntimeError("MASTER.update requires a preceding select call")
        reward = float(feedback.reward)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        base, optimism = self._pending
        base.update(x, feedback)
        self._pending = None
        self._rewards[self._epoch_count] = reward
        self._optimism[self._epoch_count] = optimism
        self._minimum_optimism = min(self._minimum_optimism, optimism)
        self._epoch_count += 1
        passed_test1, passed_test2 = self.test1(), self.test2()
        self.t += 1
        self.total_updates += 1
        if not passed_test1 or not passed_test2:
            self.n = self.n_init
            self.detected_changes += 1
        elif self.t >= self.tn + 2**self.n:
            self.n += 1
        else:
            return
        self.restart_count += 1
        self._restart_epoch()


class MasterOFUL(_MASTER):
    """MASTER using OFUL for linear rewards."""

    BASE_CLASS = OFUL
    REQUIREMENTS = replace(OFUL.REQUIREMENTS, drift_baseline_note=None)
    DEFAULTS = {**OFUL.DEFAULTS, "n_init": 0}


class MasterGLB(_MASTER):
    """MASTER using GLM-UCB for logistic rewards."""

    BASE_CLASS = GLBStaticUCB
    REQUIREMENTS = replace(GLBStaticUCB.REQUIREMENTS, drift_baseline_note=None)
    DEFAULTS = {**GLBStaticUCB.DEFAULTS, "n_init": 0}


class MasterSCB(_MASTER):
    """MASTER using LogUCB1 for logistic rewards."""

    BASE_CLASS = SCBStaticUCB
    REQUIREMENTS = replace(SCBStaticUCB.REQUIREMENTS, drift_baseline_note=None)
    DEFAULTS = {**SCBStaticUCB.DEFAULTS, "n_init": 0}
