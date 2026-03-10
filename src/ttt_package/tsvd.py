from __future__ import annotations

import numpy as np

from .utils import maybe_real


def t_svd(a: np.ndarray):
    """Full t-SVD of a third-order tensor.

    Supports real and complex inputs and rectangular frontal slices.
    Returns ``U, S, V`` such that ``a = U * S * V^*``.
    """
    if a.ndim != 3:
        raise ValueError(f"t_svd expects a 3D tensor, got {a.shape}.")
    m, n, t = a.shape
    r = min(m, n)
    real_output = np.isrealobj(a)
    a_fft = np.fft.fft(a, axis=2)
    U_fft = np.empty((m, r, t), dtype=np.complex128)
    S_fft = np.zeros((r, r, t), dtype=np.complex128)
    V_fft = np.empty((n, r, t), dtype=np.complex128)
    for i in range(t):
        U_i, s_i, Vh_i = np.linalg.svd(a_fft[:, :, i], full_matrices=False)
        U_fft[:, :, i] = U_i
        S_fft[:, :, i] = np.diag(s_i)
        V_fft[:, :, i] = Vh_i.conj().T
    U = maybe_real(np.fft.ifft(U_fft, axis=2), real_output)
    S = maybe_real(np.fft.ifft(S_fft, axis=2), real_output)
    V = maybe_real(np.fft.ifft(V_fft, axis=2), real_output)
    return U, S, V


def truncated_t_svd(a: np.ndarray, rank: int):
    if a.ndim != 3:
        raise ValueError(f"truncated_t_svd expects a 3D tensor, got {a.shape}.")
    m, n, t = a.shape
    r = min(m, n)
    if rank < 1 or rank > r:
        raise ValueError(f"rank must be in [1, {r}], got {rank}.")
    real_output = np.isrealobj(a)
    a_fft = np.fft.fft(a, axis=2)
    U_fft = np.empty((m, rank, t), dtype=np.complex128)
    S_fft = np.zeros((rank, rank, t), dtype=np.complex128)
    V_fft = np.empty((n, rank, t), dtype=np.complex128)
    for i in range(t):
        U_i, s_i, Vh_i = np.linalg.svd(a_fft[:, :, i], full_matrices=False)
        U_fft[:, :, i] = U_i[:, :rank]
        S_fft[:, :, i] = np.diag(s_i[:rank])
        V_fft[:, :, i] = Vh_i.conj().T[:, :rank]
    U = maybe_real(np.fft.ifft(U_fft, axis=2), real_output)
    S = maybe_real(np.fft.ifft(S_fft, axis=2), real_output)
    V = maybe_real(np.fft.ifft(V_fft, axis=2), real_output)
    return U, S, V


def t_svt(y: np.ndarray, tau: float) -> np.ndarray:
    if tau < 0:
        raise ValueError("tau must be nonnegative.")
    if y.ndim != 3:
        raise ValueError("t_svt expects a 3D tensor.")
    real_output = np.isrealobj(y)
    Y_fft = np.fft.fft(y, axis=2)
    X_fft = np.zeros_like(Y_fft, dtype=np.complex128)
    for i in range(y.shape[2]):
        U, s, Vh = np.linalg.svd(Y_fft[:, :, i], full_matrices=False)
        s_shrunk = np.maximum(s - tau, 0.0)
        X_fft[:, :, i] = (U * s_shrunk) @ Vh
    return maybe_real(np.fft.ifft(X_fft, axis=2), real_output)

