"""Give each algorithm a repeatable random seed, independent of execution order."""

import hashlib

import numpy as np


def algorithm_rng(seed: int, name: str) -> np.random.Generator:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    words = np.frombuffer(digest[:16], dtype="<u4").tolist()
    return np.random.default_rng(np.random.SeedSequence([seed, 100, *words]))
