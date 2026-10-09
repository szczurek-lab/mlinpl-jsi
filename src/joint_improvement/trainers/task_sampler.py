"""Utility classes for task sampling in multitask training."""

from __future__ import annotations

import random


class UniformTaskSampler:
    """
    Task sampler for multitask training that samples tasks based on weights.

    This sampler randomly selects task names according to their weights.
    The training loop is responsible for getting batches from the appropriate loaders.

    Parameters
    ----------
    tasks : dict[str, float]
        Dictionary mapping task names to their sampling weights.
        Tasks with higher weights are sampled more frequently.
        The weights are automatically normalized to sum to 1.0.

    Attributes
    ----------
    task_names : list[str]
        List of task names (in the order they appear in the tasks dict).
    task_probs : list[float]
        Normalized probabilities for each task (used for sampling).

    Examples
    --------
    >>> sampler = UniformTaskSampler(
    ...     tasks={"task_1": 1.0, "task_2": 0.5},
    ... )
    >>> task_name = sampler.sample()
    >>> batch = next(iter(train_loaders[task_name]))
    """

    def __init__(
        self,
        tasks: dict[str, float],
    ) -> None:
        """Initialize the task sampler."""
        if not tasks:
            raise ValueError("tasks dictionary cannot be empty")

        self.task_names = list(tasks.keys())

        # Normalize weights for sampling probability
        task_weights_list = list(tasks.values())
        total_weight = sum(task_weights_list)
        if total_weight == 0:
            raise ValueError("Sum of task weights cannot be zero")
        self.task_probs = [w / total_weight for w in task_weights_list]

    def sample(self) -> str:
        """Sample a task name based on weights.

        Returns
        -------
        str
            Task name sampled according to normalized weights.
        """
        return random.choices(self.task_names, weights=self.task_probs, k=1)[0]

    def get_task_probs(self, task_names: list[str]) -> dict[str, float]:
        """Get task probabilities for given task names.

        Parameters
        ----------
        task_names : list[str]
            List of task names to get probabilities for.

        Returns
        -------
        dict[str, float]
            Dictionary mapping task names to their probabilities.
        """
        return {name: prob for name, prob in zip(self.task_names, self.task_probs, strict=False) if name in task_names}
