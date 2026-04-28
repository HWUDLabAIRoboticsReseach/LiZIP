"""
make_error_heatmap_gif.py - Rotating error heatmap GIF for a NuScenes scan.

Encodes and decodes a .bin file with the C++ backend, computes per-point
nearest-neighbour reconstruction error, then renders a 360° rotating
point cloud coloured by error and saves it as a GIF.

Usage:
    python src/utils/make_error_heatmap_gif.py <input.bin> [--out error_heatmap.gif]
"""

import argparse
import io
import os
import subprocess
import sys
import tempfile

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_bin(path):
    raw = np.fromfile(path, dtype=np.float32)
    for stride in (5, 3, 4):
        if raw.size % stride == 0:
            return raw.reshape(-1, stride)[:, :3]
    raise ValueError(f"Cannot infer point stride for {path}")


def encode_decode(input_bin):
    tmp_lizip = tempfile.mktemp(suffix=".lizip")
    tmp_rec   = tempfile.mktemp(suffix=".bin")
    try:
        main_py = os.path.join(PROJECT_ROOT, "main.py")
        python  = sys.executable

        subprocess.run(
            [python, main_py, "encode", input_bin, tmp_lizip, "--mode", "cpp"],
            check=True, capture_output=True
        )
        subprocess.run(
            [python, main_py, "decode", tmp_lizip, tmp_rec, "--mode", "cpp"],
            check=True, capture_output=True
        )
        return load_bin(tmp_rec)
    finally:
        for f in (tmp_lizip, tmp_rec):
            if os.path.exists(f):
                os.remove(f)


def render_frame(orig, errors, elev, azim, vmin, vmax, cmap):
    fig = plt.figure(figsize=(7, 5.5), dpi=110)
    fig.patch.set_facecolor("#0d0d0d")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d0d0d")

    sc = ax.scatter(
        orig[:, 0], orig[:, 1], orig[:, 2],
        c=errors, cmap=cmap, vmin=vmin, vmax=vmax,
        s=0.35, linewidths=0, alpha=0.85
    )

    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)


    margin = 0.5
    ax.set_xlim(orig[:, 0].min() - margin, orig[:, 0].max() + margin)
    ax.set_ylim(orig[:, 1].min() - margin, orig[:, 1].max() + margin)
    ax.set_zlim(orig[:, 2].min() - margin, orig[:, 2].max() + margin)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.025, shrink=0.6)
    cbar.set_label("NN error (mm)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title("Reconstruction Error Heatmap", color="white", fontsize=11, pad=6)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=110)
    plt.close(fig)
    buf.seek(0)
    return imageio.v2.imread(buf)


def make_gif(input_bin, out_path, n_frames=60, elev=22, fps=20):
    print(f"Loading  {input_bin}")
    orig = load_bin(input_bin)
    print(f"  {len(orig):,} points loaded")

    print("Encoding + decoding via C++ backend …")
    rec = encode_decode(input_bin)
    print(f"  {len(rec):,} points reconstructed")

    print("Computing nearest-neighbour errors …")
    tree = cKDTree(orig)
    dists, _ = tree.query(rec, k=1)
    dists_mm = dists * 1000.0

    vmin = np.percentile(dists_mm, 1)
    vmax = np.percentile(dists_mm, 99)
    cmap = "plasma"

    print(f"  error range (p1–p99): {vmin:.4f}–{vmax:.4f} mm")

    azimuths = np.linspace(0, 360, n_frames, endpoint=False)
    frames = []
    for i, azim in enumerate(azimuths):
        print(f"\r  rendering frame {i+1}/{n_frames} …", end="", flush=True)
        frames.append(render_frame(orig, dists_mm, elev, azim, vmin, vmax, cmap))
    print()

    print(f"Saving GIF -> {out_path}")
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Done. {size_kb:.0f} KB  ({n_frames} frames @ {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description="Rotating error heatmap GIF")
    parser.add_argument("input", help="Original .bin point cloud")
    parser.add_argument("--out", default="error_heatmap.gif", help="Output GIF path")
    parser.add_argument("--frames", type=int, default=60, help="Number of rotation frames")
    parser.add_argument("--fps", type=int, default=20, help="GIF frame rate")
    parser.add_argument("--elev", type=float, default=22, help="Camera elevation angle")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[error] File not found: {args.input}")
        sys.exit(1)

    make_gif(args.input, args.out, n_frames=args.frames, elev=args.elev, fps=args.fps)


if __name__ == "__main__":
    main()
