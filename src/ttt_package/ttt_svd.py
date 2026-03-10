from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .core import TTTDecomposition
from .tproduct import t_product, tensor_conj_transpose
from .tsvd import truncated_t_svd


def ttt_svd(x: np.ndarray, ranks: Sequence[int]) -> TTTDecomposition:
    """Fixed-rank TTT-SVD.

    Parameters
    ----------
    x : ndarray, shape (I1, ..., IN, T)
    ranks : internal tubal ranks (r1, ..., r_{N-1})

    Returns
    -------
    TTTDecomposition with all cores stored as 4D arrays
    of shape (r_{n-1}, I_n, r_n, T).
    """
    if x.ndim < 3:
        raise ValueError("x must be at least third-order including the tube mode.")
    dims = x.shape
    n = x.ndim - 1
    t = dims[-1]
    if len(ranks) != n - 1:
        raise ValueError(f"Expected {n-1} internal ranks, got {len(ranks)}.")
    full_ranks = (1, *[int(r) for r in ranks], 1)
    current = np.array(x, copy=True)
    cores = []
    for k in range(n - 1):
        left_rank = full_ranks[k]
        right_rank = full_ranks[k + 1]
        current = current.reshape(left_rank * dims[k], -1, t)
        U, S, V = truncated_t_svd(current, right_rank)
        core = U.reshape(left_rank, dims[k], right_rank, t)
        cores.append(core)
        current = t_product(S, tensor_conj_transpose(V))
    last_core = current.reshape(full_ranks[-2], dims[n - 1], 1, t)
    cores.append(last_core)
    return TTTDecomposition(cores)


def reconstruct_ttt(decomp: TTTDecomposition) -> np.ndarray:
    """Reconstruct the full tensor from a TTT decomposition."""
    cores = decomp.cores
    current = cores[0].reshape(cores[0].shape[1], cores[0].shape[2], cores[0].shape[3])
    for core in cores[1:]:
        current_shape = current.shape
        left = current.reshape(-1, current_shape[-2], current_shape[-1])
        right = core.reshape(core.shape[0], core.shape[1] * core.shape[2], core.shape[3])
        prod = t_product(left, right)
        current = prod.reshape(*current_shape[:-2], core.shape[1], core.shape[2], core.shape[3])
    return current.reshape(*decomp.mode_sizes, decomp.tube_length)

