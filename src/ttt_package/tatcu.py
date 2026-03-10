from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .core import TTTDecomposition
from .tt_backend import TTDecomposition, tt_svd, tt_to_numpy
from .utils import maybe_real, relative_error


@dataclass
class TATCUInfo:
    """Diagnostics returned by TATCU routines when ``return_info=True``."""

    mode: str
    unique_frequency_indices: list[int]
    slice_errors: dict[int, float]
    slice_abs_errors_sq: dict[int, float]
    slice_target_abs_tol_sq: dict[int, float] | None
    slice_sweeps: dict[int, int]
    requested_full_tt_ranks: tuple[int, ...] | None
    effective_full_tt_ranks: tuple[int, ...]
    used_conjugate_symmetry: bool
    global_target_rel_error: float | None = None
    global_actual_rel_error: float | None = None
    reached_target: bool | None = None
    slice_reached_target: dict[int, bool] | None = None


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


def _frob_sq(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.vdot(x, x).real)


def _truncate_rank_by_energy(singular_values: np.ndarray, abs_tol_sq: float, rank_cap: int | None = None) -> int:
    """Smallest rank r such that tail energy <= abs_tol_sq (optionally capped)."""
    s = np.asarray(singular_values)
    if s.size == 0:
        return 1
    if abs_tol_sq <= 0:
        r = int(rank_cap) if rank_cap is not None else int(s.size)
        return max(1, min(r, int(s.size)))
    total = float(np.sum(s**2))
    if abs_tol_sq >= total:
        # Tail can be as large as the full energy; keep the minimal feasible rank.
        r = 1
    else:
        cs = np.cumsum(s**2)
        # Find smallest r in [1..len(s)] with total - cs[r-1] <= abs_tol_sq
        tail = total - cs
        idx = np.where(tail <= abs_tol_sq)[0]
        r = int(idx[0] + 1) if len(idx) else int(s.size)
    if rank_cap is not None:
        r = min(r, int(rank_cap))
    return max(1, min(r, int(s.size)))


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


def _update_pair_lr_adaptive(
    x: np.ndarray,
    factors: list[np.ndarray],
    pair: int,
    *,
    rank_cap: int,
    slice_abs_tol_sq: float,
    x_norm_sq: float,
) -> None:
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

    # MATLAB-faithful local budget.
    local_budget = float(slice_abs_tol_sq - x_norm_sq + _frob_sq(mat))
    if local_budget <= 0:
        r = min(int(rank_cap), int(s.size))
    else:
        r = _truncate_rank_by_energy(s, local_budget, rank_cap=int(rank_cap))
    factors[pair] = u[:, :r].reshape(g1.shape[0], g1.shape[1], r)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        factors[pair + 1] = (np.diag(s[:r]) @ vh[:r, :]).reshape(r, g2.shape[1], g2.shape[2])


def _update_pair_rl_adaptive(
    x: np.ndarray,
    factors: list[np.ndarray],
    pair: int,
    *,
    rank_cap: int,
    slice_abs_tol_sq: float,
    x_norm_sq: float,
) -> None:
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

    local_budget = float(slice_abs_tol_sq - x_norm_sq + _frob_sq(mat))
    if local_budget <= 0:
        r = min(int(rank_cap), int(s.size))
    else:
        r = _truncate_rank_by_energy(s, local_budget, rank_cap=int(rank_cap))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
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


def _tt_atcu_refine_slice_adaptive(
    x: np.ndarray,
    tt_init: TTDecomposition,
    *,
    full_rank_cap: tuple[int, ...],
    slice_abs_tol_sq: float,
    max_sweeps: int = 8,
    tol: float = 1e-8,
) -> tuple[TTDecomposition, int, float, float]:
    """Slice-adaptive two-site TT refinement (MATLAB-faithful budget logic).

    Returns (tt, n_sweeps, rel_err, abs_err_sq).
    """
    tt = _copy_tt(tt_init)
    x_norm_sq = _frob_sq(x)
    prev_err = relative_error(x, tt_to_numpy(tt))
    if max_sweeps <= 0:
        abs_err_sq = _frob_sq(x - tt_to_numpy(tt))
        return tt, 0, prev_err, abs_err_sq

    for sweep in range(1, max_sweeps + 1):
        tt = _tt_right_orthogonalize(tt)
        factors = [core.copy() for core in tt.factors]
        for pair in range(len(factors) - 1):
            _update_pair_lr_adaptive(
                x,
                factors,
                pair,
                rank_cap=full_rank_cap[pair + 1],
                slice_abs_tol_sq=float(slice_abs_tol_sq),
                x_norm_sq=x_norm_sq,
            )
        tt = TTDecomposition(factors)

        tt = _tt_left_orthogonalize(tt)
        factors = [core.copy() for core in tt.factors]
        for pair in range(len(factors) - 2, -1, -1):
            _update_pair_rl_adaptive(
                x,
                factors,
                pair,
                rank_cap=full_rank_cap[pair + 1],
                slice_abs_tol_sq=float(slice_abs_tol_sq),
                x_norm_sq=x_norm_sq,
            )
        tt = TTDecomposition(factors)

        approx = tt_to_numpy(tt)
        err = relative_error(x, approx)
        if abs(prev_err - err) <= tol * max(1.0, prev_err):
            abs_err_sq = _frob_sq(x - approx)
            return tt, sweep, err, abs_err_sq
        prev_err = err

    approx = tt_to_numpy(tt)
    abs_err_sq = _frob_sq(x - approx)
    return tt, max_sweeps, prev_err, abs_err_sq


def _pad_tt_core(core: np.ndarray, left_rank: int, right_rank: int) -> np.ndarray:
    out = np.zeros((left_rank, core.shape[1], right_rank), dtype=core.dtype)
    out[: core.shape[0], :, : core.shape[2]] = core
    return out


def _unique_fourier_indices(tube_length: int) -> list[int]:
    return list(range(tube_length // 2 + 1))


def _mirror_index(k: int, tube_length: int) -> int:
    return (-k) % tube_length


def tatcu_fixed_rank(
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
    """TATCU with a prescribed TT rank profile on each Fourier slice (fixed-rank).

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
    slice_abs_errors_sq: dict[int, float] = {}
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
            approx = tt_to_numpy(ttk)
            err = relative_error(slice_k, approx)
        if ttk.shape != tt0.shape:
            raise ValueError("refine_slice_fn changed the TT slice mode sizes.")
        if ttk.rank[0] != 1 or ttk.rank[-1] != 1:
            raise ValueError("refine_slice_fn must preserve boundary TT ranks equal to 1.")
        approx = tt_to_numpy(ttk)
        for j, core in enumerate(ttk.factors):
            spectral_cores[j][k] = core
            effective_full_ranks[j] = max(effective_full_ranks[j], core.shape[0])
            effective_full_ranks[j + 1] = max(effective_full_ranks[j + 1], core.shape[2])
        slice_errors[k] = err
        slice_abs_errors_sq[k] = _frob_sq(slice_k - approx)
        slice_sweeps[k] = n_sweeps
        if verbose:
            print(f"[TATCU] slice {k+1}/{tube_length}: sweeps={n_sweeps}, relerr={err:.3e}")

        if use_conjugate_symmetry:
            km = _mirror_index(k, tube_length)
            if km != k:
                for j, core in enumerate(ttk.factors):
                    spectral_cores[j][km] = np.conjugate(core)
                slice_errors[km] = err
                slice_abs_errors_sq[km] = slice_abs_errors_sq[k]
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
        mode="fixed_rank",
        unique_frequency_indices=process_indices,
        slice_errors=slice_errors,
        slice_abs_errors_sq=slice_abs_errors_sq,
        slice_target_abs_tol_sq=None,
        slice_sweeps=slice_sweeps,
        requested_full_tt_ranks=tuple(full_ranks),
        effective_full_tt_ranks=tuple(effective_full_ranks),
        used_conjugate_symmetry=bool(use_conjugate_symmetry),
    )
    return decomp, info


def tatcu_slice_adaptive(
    x: np.ndarray,
    tt_ranks: Sequence[int],
    *,
    slice_abs_tol_sq: float | None = None,
    slice_rel_tol: float | None = None,
    max_tt_ranks: Sequence[int] | None = None,
    max_sweeps: int = 8,
    tol: float = 1e-8,
    use_conjugate_symmetry: bool | None = None,
    return_info: bool = False,
    verbose: int = 0,
) -> TTTDecomposition | tuple[TTTDecomposition, TATCUInfo]:
    """TATCU with slice-wise adaptive TT ranks (Fourier domain) + rank synchronization.

    Notes
    -----
    - `tt_ranks` is treated as the *initial* TT rank profile.
    - If `max_tt_ranks` is provided, the method will try to *increase* slice-wise
      rank caps (up to `max_tt_ranks`) until the slice target budget is met.
      If `max_tt_ranks` is omitted, no rank growth is performed.
    """
    if (slice_abs_tol_sq is None) == (slice_rel_tol is None):
        raise ValueError("Provide exactly one of slice_abs_tol_sq or slice_rel_tol.")

    x = np.asarray(x)
    if x.ndim < 3:
        raise ValueError("x must have at least one tensor mode and one tube mode.")
    order = x.ndim - 1
    tube_length = x.shape[-1]
    init_ranks = _normalize_full_tt_ranks(order, tt_ranks)
    max_ranks = _normalize_full_tt_ranks(order, max_tt_ranks) if max_tt_ranks is not None else init_ranks

    if use_conjugate_symmetry is None:
        use_conjugate_symmetry = np.isrealobj(x)
    if use_conjugate_symmetry and not np.isrealobj(x):
        use_conjugate_symmetry = False

    x_fft = np.fft.fft(x, axis=-1)
    process_indices = _unique_fourier_indices(tube_length) if use_conjugate_symmetry else list(range(tube_length))

    spectral_cores: list[list[Optional[np.ndarray]]] = [[None] * tube_length for _ in range(order)]
    slice_errors: dict[int, float] = {}
    slice_abs_errors_sq: dict[int, float] = {}
    slice_sweeps: dict[int, int] = {}
    slice_target_abs_tol_sq: dict[int, float] = {}
    slice_reached: dict[int, bool] = {}
    effective_full_ranks = list(init_ranks)

    for k in process_indices:
        slice_k = x_fft[..., k]
        if slice_rel_tol is not None:
            target_sq = float(slice_rel_tol) ** 2 * _frob_sq(slice_k)
        else:
            target_sq = float(slice_abs_tol_sq)
        slice_target_abs_tol_sq[k] = target_sq

        cap = list(init_ranks)
        while True:
            tt0 = tt_svd(slice_k, cap)
            ttk, n_sweeps, rel_err, abs_err_sq = _tt_atcu_refine_slice_adaptive(
                slice_k,
                tt0,
                full_rank_cap=tuple(cap),
                slice_abs_tol_sq=target_sq,
                max_sweeps=max_sweeps,
                tol=tol,
            )
            if abs_err_sq <= target_sq * (1.0 + 1e-12):
                break
            # If we cannot grow further, stop.
            if all(cap[i] >= max_ranks[i] for i in range(len(cap))):
                break
            for i in range(1, len(cap) - 1):
                cap[i] = min(cap[i] + 1, max_ranks[i])

        approx = tt_to_numpy(ttk)
        for j, core in enumerate(ttk.factors):
            spectral_cores[j][k] = core
            effective_full_ranks[j] = max(effective_full_ranks[j], core.shape[0])
            effective_full_ranks[j + 1] = max(effective_full_ranks[j + 1], core.shape[2])
        slice_errors[k] = rel_err
        slice_abs_errors_sq[k] = abs_err_sq
        slice_sweeps[k] = n_sweeps
        slice_reached[k] = bool(abs_err_sq <= target_sq * (1.0 + 1e-12))
        if verbose:
            print(
                f"[TATCU/adaptive] slice {k+1}/{tube_length}: sweeps={n_sweeps}, "
                f"relerr={rel_err:.3e}, abs_err_sq={abs_err_sq:.3e}, target_sq={target_sq:.3e}"
            )

        if use_conjugate_symmetry:
            km = _mirror_index(k, tube_length)
            if km != k:
                for j, core in enumerate(ttk.factors):
                    spectral_cores[j][km] = np.conjugate(core)
                slice_errors[km] = rel_err
                slice_abs_errors_sq[km] = abs_err_sq
                slice_sweeps[km] = n_sweeps
                slice_target_abs_tol_sq[km] = target_sq
                slice_reached[km] = slice_reached[k]

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
        mode="slice_adaptive",
        unique_frequency_indices=process_indices,
        slice_errors=slice_errors,
        slice_abs_errors_sq=slice_abs_errors_sq,
        slice_target_abs_tol_sq=slice_target_abs_tol_sq,
        slice_sweeps=slice_sweeps,
        requested_full_tt_ranks=tuple(init_ranks),
        effective_full_tt_ranks=tuple(effective_full_ranks),
        used_conjugate_symmetry=bool(use_conjugate_symmetry),
        slice_reached_target=slice_reached,
    )
    return decomp, info


def tatcu_global_tol(
    x: np.ndarray,
    eps_rel: float,
    *,
    init_tt_ranks: Sequence[int],
    max_tt_ranks: Sequence[int] | None = None,
    max_sweeps: int = 8,
    tol: float = 1e-8,
    use_conjugate_symmetry: bool | None = None,
    verify: bool = True,
    return_info: bool = False,
    verbose: int = 0,
) -> TTTDecomposition | tuple[TTTDecomposition, TATCUInfo]:
    """TATCU with a true global relative-error target (Parseval-budgeted slices)."""
    if eps_rel < 0:
        raise ValueError("eps_rel must be nonnegative.")
    x = np.asarray(x)
    if x.ndim < 3:
        raise ValueError("x must have at least one tensor mode and one tube mode.")
    order = x.ndim - 1
    tube_length = x.shape[-1]

    init_cap = _normalize_full_tt_ranks(order, init_tt_ranks)
    max_cap = _normalize_full_tt_ranks(order, max_tt_ranks) if max_tt_ranks is not None else None

    if use_conjugate_symmetry is None:
        use_conjugate_symmetry = np.isrealobj(x)
    if use_conjugate_symmetry and not np.isrealobj(x):
        use_conjugate_symmetry = False

    x_fft = np.fft.fft(x, axis=-1)
    process_indices = _unique_fourier_indices(tube_length) if use_conjugate_symmetry else list(range(tube_length))

    spectral_cores: list[list[Optional[np.ndarray]]] = [[None] * tube_length for _ in range(order)]
    slice_errors: dict[int, float] = {}
    slice_abs_errors_sq: dict[int, float] = {}
    slice_sweeps: dict[int, int] = {}
    slice_target_abs_tol_sq: dict[int, float] = {}
    effective_full_ranks = list(init_cap)

    for k in process_indices:
        slice_k = x_fft[..., k]
        energy_k = _frob_sq(slice_k)
        target_sq = (float(eps_rel) ** 2) * energy_k
        slice_target_abs_tol_sq[k] = target_sq

        cap = list(init_cap)
        while True:
            tt0 = tt_svd(slice_k, cap)
            ttk, n_sweeps, rel_err, abs_err_sq = _tt_atcu_refine_slice_adaptive(
                slice_k,
                tt0,
                full_rank_cap=tuple(cap),
                slice_abs_tol_sq=target_sq,
                max_sweeps=max_sweeps,
                tol=tol,
            )
            if abs_err_sq <= target_sq * (1.0 + 1e-12):
                break
            if max_cap is None:
                # No explicit maximum: do one conservative growth step, but stop if nothing changes.
                new_cap = cap.copy()
                for i in range(1, len(new_cap) - 1):
                    new_cap[i] = new_cap[i] + 1
                if new_cap == cap:
                    break
                cap = new_cap
            else:
                if all(cap[i] >= max_cap[i] for i in range(len(cap))):
                    break
                for i in range(1, len(cap) - 1):
                    cap[i] = min(cap[i] + 1, max_cap[i])

        approx = tt_to_numpy(ttk)
        for j, core in enumerate(ttk.factors):
            spectral_cores[j][k] = core
            effective_full_ranks[j] = max(effective_full_ranks[j], core.shape[0])
            effective_full_ranks[j + 1] = max(effective_full_ranks[j + 1], core.shape[2])
        slice_errors[k] = rel_err
        slice_abs_errors_sq[k] = abs_err_sq
        slice_sweeps[k] = n_sweeps
        if verbose:
            print(
                f"[TATCU/global] slice {k+1}/{tube_length}: relerr={rel_err:.3e}, "
                f"abs_err_sq={abs_err_sq:.3e}, target_sq={target_sq:.3e}, cap={tuple(cap)}"
            )

        if use_conjugate_symmetry:
            km = _mirror_index(k, tube_length)
            if km != k:
                for j, core in enumerate(ttk.factors):
                    spectral_cores[j][km] = np.conjugate(core)
                slice_errors[km] = rel_err
                slice_abs_errors_sq[km] = abs_err_sq
                slice_sweeps[km] = n_sweeps
                slice_target_abs_tol_sq[km] = target_sq

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

    global_actual = None
    reached = None
    if verify:
        from .ttt_svd import reconstruct_ttt

        x_hat = reconstruct_ttt(decomp)
        global_actual = relative_error(x, x_hat)
        reached = bool(global_actual <= float(eps_rel) * (1.0 + 1e-8))

    if not return_info:
        return decomp
    info = TATCUInfo(
        mode="global_tol",
        unique_frequency_indices=process_indices,
        slice_errors=slice_errors,
        slice_abs_errors_sq=slice_abs_errors_sq,
        slice_target_abs_tol_sq=slice_target_abs_tol_sq,
        slice_sweeps=slice_sweeps,
        requested_full_tt_ranks=None,
        effective_full_tt_ranks=tuple(effective_full_ranks),
        used_conjugate_symmetry=bool(use_conjugate_symmetry),
        global_target_rel_error=float(eps_rel),
        global_actual_rel_error=global_actual,
        reached_target=reached,
    )
    return decomp, info


# Backward-compatible name for the original implementation.
tatcu = tatcu_fixed_rank
tatcu_prototype = tatcu_fixed_rank

