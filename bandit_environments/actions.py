"""Define the actions available to the learner: fixed, resampled, or drawn from a larger pool."""

import numpy as np


class ActionProcess:
    def __init__(self, dimension: int, config: dict, horizon: int, seed: int):
        self.dimension = dimension
        self.config = config
        self.horizon = horizon
        pool_seed = seed if config["resample_per_trial"] else config["seed"]
        self._rng = np.random.default_rng(np.random.SeedSequence([pool_seed, 1]))
        # This draw also occurs for resample mode. Removing it shifts every
        # subsequent candidate and breaks the established exogenous protocol.
        self._pool = self._sample_actions(config["pool_size"])
        self._last_context = None
        self._last_round = -1

    def _sample_actions(self, count):
        sampler = self.config.get("sampler", "gaussian-sphere")
        features = (
            self._rng.normal(size=(count, self.dimension))
            if sampler == "gaussian-sphere"
            else self._rng.uniform(-1, 1, size=(count, self.dimension))
        )
        norms = np.linalg.norm(features, axis=1)
        if sampler == "uniform-cube-clipped":
            # NS_Bandits / Log_Bandits: keep cube samples inside the unit
            # ball unchanged, project only samples outside. Not uniform in a ball.
            return features / np.maximum(norms, 1)[:, None] * self.config["bound"]
        if sampler not in {"gaussian-sphere", "uniform-cube-normalized"}:
            raise ValueError(f"unsupported action sampler {sampler!r}")
        if np.any(norms == 0):
            raise ArithmeticError("zero vector sampled for the sphere action pool")
        return features / norms[:, None] * self.config["bound"]

    def context(self, t: int):
        if not 0 <= t < self.horizon:
            raise ValueError("round outside horizon")
        if t == self._last_round:
            return self._last_context
        if t != self._last_round + 1:
            raise ValueError("contexts must be requested in increasing consecutive rounds")
        n = self.config["count"]
        mode = self.config["mode"]
        if mode == "fixed":
            features, ids = self._pool.copy(), np.arange(n)
        elif mode == "resample":
            features, ids = self._sample_actions(n), np.arange(t * n, (t + 1) * n)
        elif mode == "subset":
            ids = self._rng.choice(self.config["pool_size"], size=n, replace=False)
            features = self._pool[ids].copy()
        else:
            raise ValueError(f"unsupported action mode {mode!r}")
        features.flags.writeable = False
        ids.flags.writeable = False
        self._last_round, self._last_context = t, (features, ids)
        return features, ids
