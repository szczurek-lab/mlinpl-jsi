"""Linear scheduler for adapting step size (e.g. SBS rounds in generators)."""

from __future__ import annotations


class LinearScheduler:
    """Linear scheduler for adapting step size from ``min_value`` to ``max_value`` over ``max_rounds`` rounds."""

    __slots__ = ("_min_value", "_max_value", "_max_rounds")

    def __init__(self, min_value: float, max_value: float, max_rounds: int) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self._min_value = float(min_value)
        self._max_value = float(max_value)
        self._max_rounds = int(max_rounds)

    def get_value(self, round_idx: int) -> float:
        t = min(round_idx, self._max_rounds - 1) / max(self._max_rounds - 1, 1)
        return self._min_value + t * (self._max_value - self._min_value)

    def __repr__(self) -> str:
        return (
            f"LinearScheduler(min_value={self._min_value}, max_value={self._max_value}, "
            f"max_rounds={self._max_rounds})"
        )
