import numpy as np

from ttt_package import reconstruct_ttt
from ttt_package.tatcu import tatcu_fixed_rank, tatcu_global_tol, tatcu_slice_adaptive


def test_tatcu_fixed_rank_smoke_real():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 4, 5, 6))
    decomp = tatcu_fixed_rank(x, (1, 2, 2, 1), max_sweeps=1, tol=1e-9, use_conjugate_symmetry=True)
    xhat = reconstruct_ttt(decomp)
    assert xhat.shape == x.shape
    assert np.isfinite(np.linalg.norm(xhat))


def test_tatcu_fixed_rank_conjugate_symmetry_consistency():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 3, 3, 8))
    decomp_a = tatcu_fixed_rank(x, (1, 2, 2, 1), max_sweeps=0, use_conjugate_symmetry=True)
    decomp_b = tatcu_fixed_rank(x, (1, 2, 2, 1), max_sweeps=0, use_conjugate_symmetry=False)
    xa = reconstruct_ttt(decomp_a)
    xb = reconstruct_ttt(decomp_b)
    rel = np.linalg.norm(xa - xb) / max(np.linalg.norm(xa), 1e-12)
    assert rel < 1e-6


def test_tatcu_fixed_rank_smoke_complex():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 3, 2, 5)) + 1j * rng.standard_normal((2, 3, 2, 5))
    decomp = tatcu_fixed_rank(x, (1, 2, 2, 1), max_sweeps=0, use_conjugate_symmetry=None)
    xhat = reconstruct_ttt(decomp)
    assert np.iscomplexobj(xhat)
    assert xhat.shape == x.shape


def test_tatcu_global_tol_meets_target_on_small_tensor():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 2, 2, 4))
    eps = 1e-6
    decomp, info = tatcu_global_tol(
        x,
        eps,
        init_tt_ranks=(1, 1, 1, 1),
        max_tt_ranks=(1, 4, 4, 1),
        max_sweeps=1,
        tol=1e-10,
        use_conjugate_symmetry=True,
        verify=True,
        return_info=True,
        verbose=0,
    )
    xhat = reconstruct_ttt(decomp)
    rel = np.linalg.norm(x - xhat) / np.linalg.norm(x)
    assert info.global_actual_rel_error is not None
    assert rel <= eps * (1.0 + 1e-4)


def test_tatcu_slice_adaptive_can_meet_slice_budgets_when_full_ranks_allowed():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 2, 2, 4))
    # For a 2x2x2 tensor, TT ranks (1,2,2,1) are full and can represent any slice exactly.
    decomp, info = tatcu_slice_adaptive(
        x,
        tt_ranks=(1, 1, 1, 1),
        max_tt_ranks=(1, 2, 2, 1),
        slice_rel_tol=1e-12,
        max_sweeps=0,  # rely on TT-SVD init + rank growth
        use_conjugate_symmetry=True,
        return_info=True,
        verbose=0,
    )
    xhat = reconstruct_ttt(decomp)
    rel = np.linalg.norm(x - xhat) / np.linalg.norm(x)
    assert rel <= 1e-10
    assert info.slice_reached_target is not None
    assert all(info.slice_reached_target[k] for k in info.slice_reached_target)

