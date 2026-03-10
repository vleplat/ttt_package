from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass
class TTTDecomposition:
    """Tubal Tensor Train decomposition with a unified 4D-core convention.

    Each core has shape ``(r_{n-1}, I_n, r_n, T)`` including the two boundary
    cores, for which ``r_0 = r_N = 1``.
    """

    cores: List[np.ndarray]

    def __post_init__(self) -> None:
        if not self.cores:
            raise ValueError("A TTT decomposition must contain at least one core.")
        tube_lengths = {core.shape[-1] for core in self.cores}
        if len(tube_lengths) != 1:
            raise ValueError("All cores must share the same tube length.")
        prev_rank = 1
        for idx, core in enumerate(self.cores):
            arr = np.asarray(core)
            if arr.ndim != 4:
                raise ValueError(
                    f"Core {idx} must be 4D with shape (r_prev, I_n, r_next, T); got {arr.shape}."
                )
            if arr.shape[0] != prev_rank:
                raise ValueError(
                    f"Core {idx} has incompatible incoming rank {arr.shape[0]} (expected {prev_rank})."
                )
            prev_rank = arr.shape[2]
            self.cores[idx] = arr
        if self.cores[0].shape[0] != 1:
            raise ValueError("The first core must have left rank 1.")
        if self.cores[-1].shape[2] != 1:
            raise ValueError("The last core must have right rank 1.")

    @property
    def order(self) -> int:
        return len(self.cores)

    @property
    def tube_length(self) -> int:
        return int(self.cores[0].shape[-1])

    @property
    def mode_sizes(self) -> tuple[int, ...]:
        return tuple(int(core.shape[1]) for core in self.cores)

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(int(core.shape[2]) for core in self.cores[:-1])

    @property
    def full_ranks(self) -> tuple[int, ...]:
        return (1,) + self.ranks + (1,)

    @property
    def dtype(self):
        return np.result_type(*[core.dtype for core in self.cores])

    @property
    def shape(self) -> tuple[int, ...]:
        return self.mode_sizes + (self.tube_length,)

    def copy(self) -> "TTTDecomposition":
        return TTTDecomposition([core.copy() for core in self.cores])

