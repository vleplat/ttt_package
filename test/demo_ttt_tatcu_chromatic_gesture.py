from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib

# Force a non-interactive backend to avoid macOS GUI/backend crashes.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ttt_package import reconstruct_ttt, tatcu, ttt_svd
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


def save_error_curve(video, rec_ttt, rec_tatcu, out_base: Path) -> None:
    err_ttt = [relative_error(video[..., k], rec_ttt[..., k]) for k in range(video.shape[-1])]
    err_tatcu = [relative_error(video[..., k], rec_tatcu[..., k]) for k in range(video.shape[-1])]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(err_ttt, marker="o", label="TTT-SVD")
    ax.plot(err_tatcu, marker="s", label="TATCU")
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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    outdir = root / "output_figures"

    video = make_chromatic_gesture_video(height=32, width=32, frames=16)
    x = factorize_video(video)

    tubal_ranks = (4, 6, 6, 3)
    full_tt_ranks = (1, 4, 6, 6, 3, 1)

    print("Running TTT-SVD...")
    decomp_ttt = ttt_svd(x, tubal_ranks)
    rec_ttt = unfactorize_video(reconstruct_ttt(decomp_ttt))

    print("Running TATCU...")
    decomp_tatcu, info = tatcu(
        x,
        full_tt_ranks,
        max_sweeps=4,
        tol=1e-7,
        use_conjugate_symmetry=True,
        return_info=True,
        verbose=1,
    )
    rec_tatcu = unfactorize_video(reconstruct_ttt(decomp_tatcu))

    rel_ttt = relative_error(video, rec_ttt)
    rel_tatcu = relative_error(video, rec_tatcu)
    psnr_ttt = psnr(video, rec_ttt)
    psnr_tatcu = psnr(video, rec_tatcu)

    print("\n=== Demo summary ===")
    print(f"TTT-SVD relative error : {rel_ttt:.6f}")
    print(f"TATCU   relative error : {rel_tatcu:.6f}")
    print(f"TTT-SVD PSNR           : {psnr_ttt:.3f} dB")
    print(f"TATCU   PSNR           : {psnr_tatcu:.3f} dB")
    print(f"Requested TT ranks     : {info.requested_full_tt_ranks}")
    print(f"Effective TT ranks     : {info.effective_full_tt_ranks}")
    print(f"Unique FFT slices used : {info.unique_frequency_indices}")

    save_comparison_figure(video, rec_ttt, rec_tatcu, outdir / "gesture_comparison")
    save_error_curve(video, rec_ttt, rec_tatcu, outdir / "gesture_per_frame_errors")
    save_side_by_side_gif(video, rec_ttt, rec_tatcu, outdir / "gesture_comparison.gif")

    print(f"\nSaved outputs to: {outdir}")
    print(f" - {outdir / 'gesture_comparison.png'}")
    print(f" - {outdir / 'gesture_comparison.pdf'}")
    print(f" - {outdir / 'gesture_per_frame_errors.png'}")
    print(f" - {outdir / 'gesture_per_frame_errors.pdf'}")
    print(f" - {outdir / 'gesture_comparison.gif'}")


if __name__ == "__main__":
    main()

