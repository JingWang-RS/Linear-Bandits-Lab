"""Define the reward information an algorithm receives after choosing an action."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Feedback:
    """The selected action's reward and any known information about its noise."""

    reward: float
    outcome: int | None = None
    noise_moment_root: float | None = None
