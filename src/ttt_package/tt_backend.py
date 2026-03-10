from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


@dataclass
class TTDecomposition:
    factors: list[np.ndarray]

    def __post_init__(self):
        if not self.factors:
            raise ValueError("TT decomposition must contain at least one core")
        prev = 1
        for i, core in enumerate(self.factors):
            if core.ndim != 3:
                raise ValueError(f"TT core {i} must be 3D, got {core.shape}")
            if core.shape[0] != prev:
                raise ValueError(f"Incompatible left TT rank at core {i}: {core.shape[0]} vs {prev}")
            prev = core.shape[2]
        if self.factors[0].shape[0] != 1 or self.factors[-1].shape[2] != 1:
            raise ValueError("Boundary TT ranks must equal 1")

    @property
    def rank(self) -> tuple[int, ...]:
        return (1,) + tuple(core.shape[2] for core in self.factors[:-1]) + (1,)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(core.shape[1] for core in self.factors)


def parse_options(opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    defaults = {
        "init": "nvec",
        "maxiters": 200,
        "tol": 1e-6,
        "compression": True,
        "compression_accuracy": 1e-6,
        "noise_level": 1e-6,
        "exacterrorbound": True,
        "printitn": 0,
        "core_step": 2,
        "normX": None,
    }
    if opts:
        defaults.update(opts)
    if defaults["core_step"] not in [1, 2]:
        raise ValueError("core_step must be 1 or 2")
    return defaults


def tt_getrank(factors: Sequence[np.ndarray]) -> np.ndarray:
    if not factors:
        raise ValueError("factors must be nonempty")
    return np.asarray([factors[0].shape[0]] + [core.shape[2] for core in factors], dtype=int)


def lowrank_matrix_approx(T: np.ndarray, error_bound: float, exacterrorbound: bool = True):
    u, s, vh = np.linalg.svd(T, full_matrices=False)
    cs = np.cumsum(s**2)
    idx = np.where((cs[-1] - cs) <= error_bound)[0]
    r1 = int(idx[0]) if len(idx) else len(s) - 1
    u = u[:, : r1 + 1]
    s = s[: r1 + 1]
    v = vh[: r1 + 1, :].conj().T
    approx_error = float(cs[-1] - cs[r1])
    s0 = s.copy()
    if exacterrorbound and error_bound > approx_error and len(s) > 0:
        s = s.copy()
        s[0] = s[0] + np.sqrt(error_bound - approx_error)
        approx_error = float(error_bound)
    return u, s, v, approx_error, s0


def tt_svd(input_tensor: np.ndarray, rank: Sequence[int]) -> TTDecomposition:
    tensor_size = input_tensor.shape
    n_dim = len(tensor_size)
    if len(rank) != n_dim + 1:
        raise ValueError("rank must have length tensor_order + 1 and include both boundary ones.")
    rank = list(rank)
    unfolding = np.asarray(input_tensor)
    factors = [None] * n_dim
    for k in range(n_dim - 1):
        n_row = int(rank[k] * tensor_size[k])
        unfolding = unfolding.reshape(n_row, -1)
        n_row, n_col = unfolding.shape
        current_rank = min(n_row, n_col, rank[k + 1])
        U, S, Vh = np.linalg.svd(unfolding, full_matrices=False)
        U = U[:, :current_rank]
        S = S[:current_rank]
        Vh = Vh[:current_rank, :]
        rank[k + 1] = current_rank
        factors[k] = U.reshape(rank[k], tensor_size[k], rank[k + 1])
        unfolding = S[:, None] * Vh
    prev_rank, last_dim = unfolding.shape
    factors[-1] = unfolding.reshape(prev_rank, last_dim, 1)
    return TTDecomposition(factors)


def tt_to_numpy(x: TTDecomposition) -> np.ndarray:
    current = x.factors[0].reshape(x.factors[0].shape[1], x.factors[0].shape[2])
    for core in x.factors[1:]:
        current_shape = current.shape
        left = current.reshape(-1, current_shape[-1])
        right = core.reshape(core.shape[0], -1)
        # Some intermediate TT cores (e.g., in Fourier-domain refinement) can be
        # poorly scaled; avoid spamming runtime warnings while still computing.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            prod = left @ right
        current = prod.reshape(*current_shape[:-1], core.shape[1], core.shape[2])
    return current.reshape(*x.shape)

