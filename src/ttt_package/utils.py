from __future__ import annotations

import numpy as np


def maybe_real(x: np.ndarray, prefer_real: bool) -> np.ndarray:
    """Return ``x.real`` when ``prefer_real`` is True and the imaginary part is negligible."""
    if prefer_real:
        return np.real_if_close(x, tol=1000).real
    return x


def frobenius_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x).ravel()))


def relative_error(x: np.ndarray, y: np.ndarray) -> float:
    denom = frobenius_norm(x)
    if denom == 0.0:
        return 0.0 if frobenius_norm(y) == 0.0 else float("inf")
    return frobenius_norm(x - y) / denom

