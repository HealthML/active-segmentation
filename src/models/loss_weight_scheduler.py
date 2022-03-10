"""Scheduler to reduce pseudo-label loss weights as training progresses."""

import math
from typing import Literal, Optional


class LossWeightScheduler:
    """
    Scheduler to reduce pseudo-label loss weights as training progresses.

    Args:
        algorithm (string): Scheduling algorithm to be used: `"fixed"` | `"linear"` | `"cosine"`:
            - `"fixed"`: the pseudo-label loss weight is not changed.
            - `"linear"`: the pseudo-label loss weight is reduced by the same value in each step.
            - `"cosine"`: the decay of the pseudo-label loss weight follows a cosine curve.
        start_weight (float): Initial pseudo-label loss weight.
        end_weight (float, optional): Final pseudo-label loss weight. Must be specified when :attr:`algorithm` is set to
            `"linear"` or `"cosine"`. Defaults to `None`.
        start_step (int, optional): Initial step index. Defaults to 0.
        end_step (int, optional): Final step index. Must be specified when :attr:`algorithm` is set to `"linear"` or
            `"cosine"`. Defaults to `None`.
    """

    def __init__(
        self,
        algorithm: Literal["fixed", "linear", "cosine"],
        start_weight: float,
        end_weight: Optional[float] = None,
        start_step: int = 0,
        end_step: Optional[int] = None,
    ):
        if algorithm not in ["fixed", "linear", "cosine"]:
            raise ValueError(f"Invalid loss weight scheduling algorithm: {algorithm}")

        self.algorithm = algorithm

        self.start_weight = start_weight

        if self.algorithm in ["linear", "cosine"] and end_weight is None:
            raise ValueError(
                f"The parameter `end_weight` must be specified when `algorithm` is {self.algorithm}. Make sure that in "
                f"the config file the `iterations` parameter is set to a fixed value or the "
                f"`weight_pseudo_labels_decay_steps` option is specified in the loss config."
            )
        self.end_weight = end_weight

        if self.algorithm in ["linear", "cosine"] and end_step is None:
            raise ValueError(
                f"The parameter `end_step` must be specified when `algorithm` is {self.algorithm}. Make sure that in "
                f"the config file the `iterations` parameter is set to a fixed value or the "
                f"`weight_pseudo_labels_decay_steps` option is specified in the loss config."
            )
        self.end_step = end_step

        self.current_step = start_step
        self.current_weigt = self.start_weight

    def step(self) -> None:
        """
        Increases scheduler step.
        """

        self.current_step += 1

    def current_weight(self) -> float:
        """
        Returns:
            float: Current pseudo-label loss weight.
        """

        if self.algorithm == "fixed":
            return self.current_weigt
        if self.algorithm == "linear":
            return self.end_weight + (self.start_weight - self.end_weight) * (
                1 - self.current_step / self.end_step
            )

        # cosine scheduling
        return (
            self.end_weight
            + (self.start_weight - self.end_weight)
            * (1 + math.cos(math.pi * self.current_step / self.end_step))
            / 2
        )
