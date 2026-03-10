from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .core import TTTDecomposition
from .tt_backend import TTDecomposition, tt_svd, tt_to_numpy
from .utils import maybe_real, relative_error


@dataclass
class TATCUInfo:
    """Diagnostics returned by :func:`tatcu` when ``return_info=True``."""

    unique_frequency_indices: list[int]
    slice_errors: dict[int, float]
    slice_sweeps: dict[int, int]
    requested_full_tt_ranks: tuple[int, ...]
    effective_full_tt_ranks: tuple[int, ...]
    used_conjugate_symmetry: bool


def _normalize_full_tt_ranks(order: int, tt_ranks: Sequence[int]) -> tuple[int, ...]:
    tt_ranks = tuple(int(r) for r in tt_ranks)
    if len(tt_ranks) == order - 1:
        full = (1, *tt_ranks, 1)
    elif len(tt_ranks) == order + 1:
        full = tt_ranks
    else:
        raise ValueError(
            f"tt_ranks must be either the internal profile of length {order-1} "
            f"or the full profile of length {order+1}."
        )
    if full[0] != 1 or full[-1] != 1:
        raise ValueError("Boundary TT ranks must equal 1.")
    if any(r < 1 for r in full):
        raise ValueError("All TT ranks must be positive integers.")
    return full


def _copy_tt(tt: TTDecomposition) -> TTDecomposition:
    return TTDecomposition([core.copy() for core in tt.factors])


def _tt_left_orthogonalize(tt: TTDecomposition) -> TTDecomposition:
    factors = [core.copy() for core in tt.factors]
    n = len(factors)
    for k in range(n - 1):
        g = factors[k]
        rp, i, rn = g.shape
        q, r = np.linalg.qr(g.reshape(rp * i, rn), mode="reduced")
        new_rank = q.shape[1]
        factors[k] = q.reshape(rp, i, new_rank)
        factors[k + 1] = np.einsum("ab,bcd->acd", r, factors[k + 1])
    return TTDecomposition(factors)


def _tt_right_orthogonalize(tt: TTDecomposition) -> TTDecomposition:
    factors = [core.copy() for core in tt.factors]
    n = len(factors)
    for k in range(n - 1, 0, -1):
        g = factors[k]
        rp, i, rn = g.shape
        q, r = np.linalg.qr(g.reshape(rp, i * rn).T, mode="reduced")
        q = q.T
        rt = r.T
        new_rank = q.shape[0]
        factors[k] = q.reshape(new_rank, i, rn)
        factors[k - 1] = np.einsum("aib,bc->aic", factors[k - 1], rt)
    return TTDecomposition(factors)


def _left_basis(factors: Sequence[np.ndarray]) -> np.ndarray:
    basis = np.ones((1, 1), dtype=np.result_type(*[f.dtype for f in factors]) if factors else np.float64)
    for g in factors:
        basis = np.einsum("pa,air->pir", basis, g).reshape(-1, g.shape[2])
    return basis


def _right_basis(factors: Sequence[np.ndarray]) -> np.ndarray:
    basis = np.ones((1, 1), dtype=np.result_type(*[f.dtype for f in factors]) if factors else np.float64)
    for g in reversed(factors):
        basis = np.einsum("air,rp->aip", g, basis).reshape(g.shape[0], -1)
    return basis


def _update_pair_lr(x: np.ndarray, factors: list[np.ndarray], pair: int, target_rank: int) -> None:
    g1 = factors[pair]
    g2 = factors[pair + 1]
    left = _left_basis(factors[:pair])
    right = _right_basis(factors[pair + 2 :])
    p_left = left.shape[0]
    p_right = right.shape[1]
    x4 = x.reshape(p_left, g1.shape[1], g2.shape[1], p_right)
    theta = np.einsum("pa,pijq,bq->aijb", left.conj(), x4, right.conj())
    mat = theta.reshape(g1.shape[0] * g1.shape[1], g2.shape[1] * g2.shape[2])
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    r = min(target_rank, len(s))
    factors[pair] = u[:, :r].reshape(g1.shape[0], g1.shape[1], r)
    factors[pair + 1] = (np.diag(s[:r]) @ vh[:r, :]).reshape(r, g2.shape[1], g2.shape[2])


def _update_pair_rl(x: np.ndarray, factors: list[np.ndarray], pair: int, target_rank: int) -> None:
    g1 = factors[pair]
    g2 = factors[pair + 1]
    left = _left_basis(factors[:pair])
    right = _right_basis(factors[pair + 2 :])
    p_left = left.shape[0]
    p_right = right.shape[1]
    x4 = x.reshape(p_left, g1.shape[1], g2.shape[1], p_right)
    theta = np.einsum("pa,pijq,bq->aijb", left.conj(), x4, right.conj())
    mat = theta.reshape(g1.shape[0] * g1.shape[1], g2.shape[1] * g2.shape[2])
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    r = min(target_rank, len(s))
    factors[pair] = (u[:, :r] * s[:r]).reshape(g1.shape[0], g1.shape[1], r)
    factors[pair + 1] = vh[:r, :].reshape(r, g2.shape[1], g2.shape[2])


def _tt_atcu_refine_fixed_rank(
    x: np.ndarray,
    tt_init: TTDecomposition,
    max_sweeps: int = 8,
    tol: float = 1e-8,
) -> tuple[TTDecomposition, int, float]:
    """Fixed-rank two-site TT refinement for a dense tensor slice.

    This is a compact, NumPy-only backend suitable for TATCU on moderate-size
    Fourier slices. It performs alternating left-to-right and right-to-left
    two-core updates with exact local SVD solves under a prescribed TT rank
    profile.
    """
    tt = _copy_tt(tt_init)
    full_ranks = tt.rank
    prev_err = relative_error(x, tt_to_numpy(tt))
    if max_sweeps <= 0:
        return tt, 0, prev_err

    for sweep in range(1, max_sweeps + 1):
        tt = _tt_right_orthogonalize(tt)
        factors = [core.copy() for core in tt.factors]
        for pair in range(len(factors) - 1):
            _update_pair_lr(x, factors, pair, full_ranks[pair + 1])
        tt = TTDecomposition(factors)

        tt = _tt_left_orthogonalize(tt)
        factors = [core.copy() for core in tt.factors]
        for pair in range(len(factors) - 2, -1, -1):
            _update_pair_rl(x, factors, pair, full_ranks[pair + 1])
        tt = TTDecomposition(factors)

        err = relative_error(x, tt_to_numpy(tt))
        if abs(prev_err - err) <= tol * max(1.0, prev_err):
            return tt, sweep, err
        prev_err = err

    return tt, max_sweeps, prev_err


def _pad_tt_core(core: np.ndarray, left_rank: int, right_rank: int) -> np.ndarray:
    out = np.zeros((left_rank, core.shape[1], right_rank), dtype=core.dtype)
    out[: core.shape[0], :, : core.shape[2]] = core
    return out


def _unique_fourier_indices(tube_length: int) -> list[int]:
    return list(range(tube_length // 2 + 1))


def _mirror_index(k: int, tube_length: int) -> int:
    return (-k) % tube_length


def tatcu(
    x: np.ndarray,
    tt_ranks: Sequence[int],
    *,
    max_sweeps: int = 8,
    tol: float = 1e-8,
    use_conjugate_symmetry: bool | None = None,
    refine_slice_fn: Optional[Callable[[np.ndarray, TTDecomposition], TTDecomposition]] = None,
    return_info: bool = False,
    verbose: int = 0,
) -> TTTDecomposition | tuple[TTTDecomposition, TATCUInfo]:
    """Tubal alternating two-cores update (TATCU).

    Parameters
    ----------
    x : ndarray, shape (I1, ..., IN, T)
        Input tensor whose last axis is the distinguished tube mode.
    tt_ranks : sequence of int
        Either the internal TT rank profile ``(r1, ..., r_{N-1})`` or the full
        profile ``(1, r1, ..., r_{N-1}, 1)`` used on each Fourier slice.
    max_sweeps : int, optional
        Number of left-right/right-left refinement sweeps per processed slice.
        Set to ``0`` to skip ATCU refinement and keep the TT-SVD initializer.
    tol : float, optional
        Relative stopping tolerance for the slice-wise refinement backend.
    use_conjugate_symmetry : bool, optional
        If ``True`` and ``x`` is real-valued, only the non-redundant Fourier
        slices are processed and the remaining ones are reconstructed by complex
        conjugation. The default is ``True`` for real input and ``False`` for
        complex input.
    refine_slice_fn : callable, optional
        Optional custom slice refiner. It receives ``(slice_fft, tt_init)`` and
        must return a ``TTDecomposition`` with the same TT mode sizes and
        boundary ranks. When omitted, a built-in fixed-rank two-site TT backend
        is used.
    return_info : bool, optional
        If ``True``, return a ``TATCUInfo`` diagnostics object alongside the
        decomposition.
    verbose : int, optional
        Print per-slice progress when positive.

    Returns
    -------
    TTTDecomposition
        All tubal cores are returned with the unified 4D convention
        ``(r_{n-1}, I_n, r_n, T)``.
    """
    x = np.asarray(x)
    if x.ndim < 3:
        raise ValueError("x must have at least one tensor mode and one tube mode.")
    order = x.ndim - 1
    tube_length = x.shape[-1]
    full_ranks = _normalize_full_tt_ranks(order, tt_ranks)

    if use_conjugate_symmetry is None:
        use_conjugate_symmetry = np.isrealobj(x)
    if use_conjugate_symmetry and not np.isrealobj(x):
        use_conjugate_symmetry = False

    x_fft = np.fft.fft(x, axis=-1)
    process_indices = _unique_fourier_indices(tube_length) if use_conjugate_symmetry else list(range(tube_length))

    spectral_cores: list[list[Optional[np.ndarray]]] = [[None] * tube_length for _ in range(order)]
    slice_errors: dict[int, float] = {}
    slice_sweeps: dict[int, int] = {}
    effective_full_ranks = list(full_ranks)

    for k in process_indices:
        slice_k = x_fft[..., k]
        tt0 = tt_svd(slice_k, full_ranks)
        if refine_slice_fn is None:
            ttk, n_sweeps, err = _tt_atcu_refine_fixed_rank(slice_k, tt0, max_sweeps=max_sweeps, tol=tol)
        else:
            ttk = refine_slice_fn(slice_k, tt0)
            if not isinstance(ttk, TTDecomposition):
                raise TypeError("refine_slice_fn must return a TTDecomposition.")
            n_sweeps = max_sweeps
            err = relative_error(slice_k, tt_to_numpy(ttk))
        if ttk.shape != tt0.shape:
            raise ValueError("refine_slice_fn changed the TT slice mode sizes.")
        if ttk.rank[0] != 1 or ttk.rank[-1] != 1:
            raise ValueError("refine_slice_fn must preserve boundary TT ranks equal to 1.")
        for j, core in enumerate(ttk.factors):
            spectral_cores[j][k] = core
            effective_full_ranks[j] = max(effective_full_ranks[j], core.shape[0])
            effective_full_ranks[j + 1] = max(effective_full_ranks[j + 1], core.shape[2])
        slice_errors[k] = err
        slice_sweeps[k] = n_sweeps
        if verbose:
            print(f"[TATCU] slice {k+1}/{tube_length}: sweeps={n_sweeps}, relerr={err:.3e}")

        if use_conjugate_symmetry:
            km = _mirror_index(k, tube_length)
            if km != k:
                for j, core in enumerate(ttk.factors):
                    spectral_cores[j][km] = np.conjugate(core)
                slice_errors[km] = err
                slice_sweeps[km] = n_sweeps

    for k in range(tube_length):
        if spectral_cores[0][k] is None:
            raise RuntimeError(f"Missing spectral TT cores for Fourier slice {k}.")

    tubal_cores = []
    for j in range(order):
        left_rank = effective_full_ranks[j]
        right_rank = effective_full_ranks[j + 1]
        stacked = np.stack(
            [_pad_tt_core(spectral_cores[j][k], left_rank, right_rank) for k in range(tube_length)],
            axis=-1,
        )
        tubal_core = np.fft.ifft(stacked, axis=-1)
        tubal_cores.append(maybe_real(tubal_core, np.isrealobj(x)))

    decomp = TTTDecomposition(tubal_cores)
    if not return_info:
        return decomp
    info = TATCUInfo(
        unique_frequency_indices=process_indices,
        slice_errors=slice_errors,
        slice_sweeps=slice_sweeps,
        requested_full_tt_ranks=tuple(full_ranks),
        effective_full_tt_ranks=tuple(effective_full_ranks),
        used_conjugate_symmetry=bool(use_conjugate_symmetry),
    )
    return decomp, info


tatcu_prototype = tatcu

