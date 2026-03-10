from __future__ import annotations

import numpy as np

from .utils import maybe_real


def tensor_conj_transpose(a: np.ndarray) -> np.ndarray:
    """Conjugate transpose in the t-product algebra.

    For a tensor ``a`` of shape ``(m, n, T)``, return the tensor of shape
    ``(n, m, T)`` obtained by conjugate-transposing each frontal slice.
    This matches the Fourier-domain characterization used in the paper.
    """
    if a.ndim != 3:
        raise ValueError(f"tensor_conj_transpose expects a 3D tensor, got {a.shape}.")
    a_fft = np.fft.fft(a, axis=2)
    out_fft = np.transpose(np.conjugate(a_fft), (1, 0, 2))
    return maybe_real(np.fft.ifft(out_fft, axis=2), np.isrealobj(a))


def t_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """t-product of two third-order tensors.

    Parameters
    ----------
    a : ndarray, shape (m, p, T)
    b : ndarray, shape (p, n, T)

    Returns
    -------
    ndarray, shape (m, n, T)
    """
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("t_product expects two 3D tensors.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible t-product dimensions: {a.shape} and {b.shape}.")
    if a.shape[2] != b.shape[2]:
        raise ValueError("Both tensors must share the same tube length.")
    prefer_real = np.isrealobj(a) and np.isrealobj(b)
    a_fft = np.fft.fft(a, axis=2)
    b_fft = np.fft.fft(b, axis=2)
    c_fft = np.einsum("mpt,pnt->mnt", a_fft, b_fft)
    return maybe_real(np.fft.ifft(c_fft, axis=2), prefer_real)

