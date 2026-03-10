from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

# Force a non-interactive backend to avoid macOS GUI/backend crashes.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ttt_package import (
    reconstruct_ttt,
    tatcu_fixed_rank,
    tatcu_global_tol,
    tatcu_slice_adaptive,
    ttt_svd,
)
from ttt_package.utils import relative_error


def make_chromatic_gesture_video(height: int = 32, width: int = 32, frames: int = 16) -> np.ndarray:
    """Create a smooth RGB video with gesture-like moving color blobs.

    Returns
    -------
    video : ndarray, shape (height, width, 3, frames)
        Values are in [0, 1]. The last mode is the tube mode (time).
    """
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, height),
        np.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    video = np.zeros((height, width, 3, frames), dtype=np.float64)
    sigma = 0.18

    for t in range(frames):
        theta = 2.0 * np.pi * t / frames

        c1x = 0.45 * np.cos(theta)
        c1y = 0.35 * np.sin(2.0 * theta)
        c2x = -0.35 * np.cos(1.5 * theta + 0.4)
        c2y = 0.45 * np.sin(theta + 0.7)

        g1 = np.exp(-((xx - c1x) ** 2 + (yy - c1y) ** 2) / (2.0 * sigma**2))
        g2 = np.exp(-((xx - c2x) ** 2 + (yy - c2y) ** 2) / (2.0 * (0.8 * sigma) ** 2))

        phi = theta / 2.0
        xr = xx * np.cos(phi) + yy * np.sin(phi)
        yr = -xx * np.sin(phi) + yy * np.cos(phi)
        bar = np.exp(-(yr**2) / (2.0 * 0.05**2)) * np.exp(-(xr**2) / (2.0 * 0.45**2))

        frame = np.zeros((height, width, 3), dtype=np.float64)
        frame[..., 0] = 0.95 * g1 + 0.25 * bar
        frame[..., 1] = 0.75 * g2 + 0.20 * np.roll(g1, shift=t // 4, axis=1)
        frame[..., 2] = 0.45 * g1 + 0.45 * g2 + 0.30 * np.roll(bar, shift=t // 3, axis=0)

        frame *= 0.8 + 0.2 * np.cos(theta - np.array([0.0, 0.8, 1.6]))
        frame -= frame.min()
        frame /= max(frame.max(), 1e-12)
        video[..., t] = frame

    return np.clip(video, 0.0, 1.0)


def factorize_video(video: np.ndarray) -> np.ndarray:
    """Reshape a (32,32,3,T) video into a higher-order tensor with tube mode T."""
    if video.shape[0:3] != (32, 32, 3):
        raise ValueError("This demo expects a video of shape (32, 32, 3, T).")
    t = video.shape[-1]
    return video.reshape(4, 8, 4, 8, 3, t)


def unfactorize_video(x: np.ndarray) -> np.ndarray:
    """Inverse of factorize_video."""
    t = x.shape[-1]
    return x.reshape(32, 32, 3, t)


def psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = np.mean((np.asarray(x) - np.asarray(y)) ** 2)
    if mse <= 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def _save_figure(fig: plt.Figure, out_base: Path, *, dpi: int = 180) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_comparison_figure(video, rec_ttt, rec_tatcu, out_base: Path) -> None:
    frame_ids = [0, video.shape[-1] // 3, 2 * video.shape[-1] // 3, video.shape[-1] - 1]
    fig, axes = plt.subplots(3, len(frame_ids), figsize=(3.0 * len(frame_ids), 8.5))
    row_titles = ["Original", "TTT-SVD", "TATCU"]
    videos = [video, rec_ttt, rec_tatcu]
    for r in range(3):
        for c, k in enumerate(frame_ids):
            axes[r, c].imshow(np.clip(videos[r][..., k], 0.0, 1.0))
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"frame {k}")
        axes[r, 0].set_ylabel(row_titles[r], fontsize=12)
    fig.suptitle("Chromatic gesture demo: original vs TTT-SVD vs TATCU", fontsize=14)
    fig.tight_layout()
    _save_figure(fig, out_base)


def save_error_curve(video, rec_ttt, rec_tatcu, out_base: Path, *, target: float | None = None) -> None:
    err_ttt = [relative_error(video[..., k], rec_ttt[..., k]) for k in range(video.shape[-1])]
    err_tatcu = [relative_error(video[..., k], rec_tatcu[..., k]) for k in range(video.shape[-1])]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(err_ttt, marker="o", label="TTT-SVD")
    ax.plot(err_tatcu, marker="s", label="TATCU")
    if target is not None:
        ax.axhline(float(target), color="k", linestyle="--", linewidth=1.0, alpha=0.65, label="target")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Relative error")
    ax.set_title("Per-frame reconstruction error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, out_base)


def save_side_by_side_gif(video, rec_ttt, rec_tatcu, out_gif: Path) -> None:
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for k in range(video.shape[-1]):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, img, title in zip(
            axes,
            [video[..., k], rec_ttt[..., k], rec_tatcu[..., k]],
            ["Original", "TTT-SVD", "TATCU"],
        ):
            ax.imshow(np.clip(img, 0.0, 1.0))
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"Frame {k}")
        fig.tight_layout()
        tmp = out_gif.with_name(f"{out_gif.stem}__tmp_frame_{k}.png")
        fig.savefig(tmp, dpi=120, bbox_inches="tight")
        plt.close(fig)
        frames.append(imageio.imread(tmp))
        tmp.unlink(missing_ok=True)
    imageio.mimsave(out_gif, frames, duration=0.35, loop=0)


def _parse_int_list(values: list[str]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chromatic gesture demo: TTT-SVD vs TATCU variants.")
    parser.add_argument(
        "--tatcu_mode",
        choices=["fixed_rank", "slice_adaptive", "global_tol"],
        default="fixed_rank",
        help="Which TATCU variant to run.",
    )
    parser.add_argument(
        "--eps_rel",
        type=float,
        default=0.11,
        help="Global relative error target for tatcu_global_tol (used when tatcu_mode=global_tol).",
    )
    parser.add_argument(
        "--slice_rel_tol",
        type=float,
        default=0.11,
        help="Slice relative target for tatcu_slice_adaptive (used when tatcu_mode=slice_adaptive).",
    )
    parser.add_argument(
        "--max_sweeps",
        type=int,
        default=4,
        help="Number of ATCU sweeps per processed Fourier slice.",
    )
    parser.add_argument(
        "--tt_ranks",
        nargs="+",
        default=["1", "4", "6", "6", "3", "1"],
        help="Initial / fixed TT rank profile (full profile with boundary 1s).",
    )
    parser.add_argument(
        "--max_tt_ranks",
        nargs="+",
        default=["1", "8", "10", "10", "6", "1"],
        help="Max TT rank caps (full profile with boundary 1s, used when tatcu_mode=global_tol).",
    )
    parser.add_argument(
        "--frame_target",
        type=float,
        default=None,
        help="Optional per-frame target line to overlay on the error curve (diagnostic only).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = root / "output_figures"

    video = make_chromatic_gesture_video(height=32, width=32, frames=16)
    x = factorize_video(video)

    tubal_ranks = (4, 6, 6, 3)
    full_tt_ranks = _parse_int_list(args.tt_ranks)
    max_tt_ranks = _parse_int_list(args.max_tt_ranks)

    print("Running TTT-SVD...")
    decomp_ttt = ttt_svd(x, tubal_ranks)
    rec_ttt = unfactorize_video(reconstruct_ttt(decomp_ttt))

    print(f"Running TATCU ({args.tatcu_mode})...")
    if args.tatcu_mode == "fixed_rank":
        decomp_tatcu, info = tatcu_fixed_rank(
            x,
            full_tt_ranks,
            max_sweeps=args.max_sweeps,
            tol=1e-7,
            use_conjugate_symmetry=True,
            return_info=True,
            verbose=1,
        )
        target_for_plot = args.frame_target
        tag = f"fixed_rank_r{'-'.join(map(str, full_tt_ranks))}"
    elif args.tatcu_mode == "slice_adaptive":
        decomp_tatcu, info = tatcu_slice_adaptive(
            x,
            full_tt_ranks,
            slice_rel_tol=float(args.slice_rel_tol),
            max_tt_ranks=max_tt_ranks,
            max_sweeps=args.max_sweeps,
            tol=1e-7,
            use_conjugate_symmetry=True,
            return_info=True,
            verbose=1,
        )
        target_for_plot = args.frame_target if args.frame_target is not None else float(args.slice_rel_tol)
        tag = (
            f"slice_adaptive_tol{args.slice_rel_tol:.3g}"
            f"_init{'-'.join(map(str, full_tt_ranks))}"
            f"_max{'-'.join(map(str, max_tt_ranks))}"
        )
    else:
        decomp_tatcu, info = tatcu_global_tol(
            x,
            float(args.eps_rel),
            init_tt_ranks=full_tt_ranks,
            max_tt_ranks=max_tt_ranks,
            max_sweeps=args.max_sweeps,
            tol=1e-7,
            use_conjugate_symmetry=True,
            verify=True,
            return_info=True,
            verbose=1,
        )
        # Global tolerance does not imply a per-frame guarantee, but we can still overlay it as a diagnostic.
        target_for_plot = args.frame_target if args.frame_target is not None else float(args.eps_rel)
        tag = f"global_tol_eps{args.eps_rel:.3g}_init{'-'.join(map(str, full_tt_ranks))}_max{'-'.join(map(str, max_tt_ranks))}"

    rec_tatcu = unfactorize_video(reconstruct_ttt(decomp_tatcu))

    rel_ttt = relative_error(video, rec_ttt)
    rel_tatcu = relative_error(video, rec_tatcu)
    psnr_ttt = psnr(video, rec_ttt)
    psnr_tatcu = psnr(video, rec_tatcu)

    per_frame = np.array([relative_error(video[..., k], rec_tatcu[..., k]) for k in range(video.shape[-1])], dtype=float)
    print("\nPer-frame TATCU errors (diagnostic):")
    print(f"min={per_frame.min():.6f}  max={per_frame.max():.6f}  mean={per_frame.mean():.6f}  std={per_frame.std():.6f}")
    if target_for_plot is not None:
        n_bad = int(np.sum(per_frame > float(target_for_plot)))
        print(f"frames above target={float(target_for_plot):.6f}: {n_bad}/{per_frame.size}")

    print("\n=== Demo summary ===")
    print(f"TTT-SVD relative error : {rel_ttt:.6f}")
    print(f"TATCU   relative error : {rel_tatcu:.6f}")
    print(f"TTT-SVD PSNR           : {psnr_ttt:.3f} dB")
    print(f"TATCU   PSNR           : {psnr_tatcu:.3f} dB")
    print(f"Requested TT ranks     : {info.requested_full_tt_ranks}")
    print(f"Effective TT ranks     : {info.effective_full_tt_ranks}")
    print(f"Unique FFT slices used : {info.unique_frequency_indices}")
    if getattr(info, "global_target_rel_error", None) is not None:
        print(f"Global target rel error: {info.global_target_rel_error}")
        print(f"Global actual rel error: {info.global_actual_rel_error}")
        print(f"Reached target?         {info.reached_target}")

    save_comparison_figure(video, rec_ttt, rec_tatcu, outdir / f"gesture_comparison__{tag}")
    save_error_curve(video, rec_ttt, rec_tatcu, outdir / f"gesture_per_frame_errors__{tag}", target=target_for_plot)
    save_side_by_side_gif(video, rec_ttt, rec_tatcu, outdir / f"gesture_comparison__{tag}.gif")

    print(f"\nSaved outputs to: {outdir}")
    print(f" - {outdir / f'gesture_comparison__{tag}.png'}")
    print(f" - {outdir / f'gesture_comparison__{tag}.pdf'}")
    print(f" - {outdir / f'gesture_per_frame_errors__{tag}.png'}")
    print(f" - {outdir / f'gesture_per_frame_errors__{tag}.pdf'}")
    print(f" - {outdir / f'gesture_comparison__{tag}.gif'}")


if __name__ == "__main__":
    main()

