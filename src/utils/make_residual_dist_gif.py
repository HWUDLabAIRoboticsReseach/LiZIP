"""
make_residual_dist_gif.py - Residual distribution tightening animation.

Shows side-by-side histograms of:
  Left  - raw quantised int32 coordinate spread (centred)
  Right - MLP residuals (tight spike near zero)

Both histograms fill in block-by-block to show the computation live,
then hold on the final state so the viewer can read the labels.

Usage:
    python src/utils/make_residual_dist_gif.py <input.bin> [--out residual_dist.gif]
"""

import argparse
import importlib.util
import io
import os
import sys

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESID_SCALE = 100_000.0
BLOCK_SIZE  = 128
CONTEXT_K   = 3


def _load_module(name, rel_path):
    path = os.path.join(PROJECT_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_bin(path):
    raw = np.fromfile(path, dtype=np.float32)
    for stride in (5, 3, 4):
        if raw.size % stride == 0:
            return raw.reshape(-1, stride)[:, :3]
    raise ValueError(f"Cannot infer point stride for {path}")


def compute_raw_and_residuals(input_path, model_path):
    voxel_sort = _load_module("voxel_sort", "src/python/voxel_sort.py")
    data_loader = _load_module("data_loader", "src/utils/data_loader.py")

    raw   = data_loader.load_point_cloud(input_path)[:, :3]
    pts   = voxel_sort.voxel_quantize_and_sort(raw).astype(np.float32)
    pts_mm = np.round(pts * RESID_SCALE).astype(np.int32)          # int32 quantised

    # load model
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "python"))
    from model import PointPredictorMLP
    model = PointPredictorMLP(context_size=CONTEXT_K, hidden_dim=256)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    n  = len(pts_mm)
    nb = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    pts_t      = torch.from_numpy(pts_mm).float() / RESID_SCALE
    block_starts = np.arange(nb) * BLOCK_SIZE

    raw_per_block  = []   # centred raw coordinates for each block
    res_per_block  = []   # residuals for each block

    with torch.no_grad():
        # context = first CONTEXT_K points of each block
        heads = []
        for s in block_starts:
            h = pts_t[s : s + CONTEXT_K]
            if len(h) < CONTEXT_K:
                pad = torch.zeros(CONTEXT_K - len(h), 3)
                h   = torch.cat([h, pad])
            heads.append(h)
        ctx = torch.stack(heads).reshape(nb, -1).clone()   # (nb, k*3)

        for step in range(BLOCK_SIZE - CONTEXT_K):
            target_idx = block_starts + CONTEXT_K + step
            valid      = target_idx < n
            if not valid.any():
                break

            preds = model(ctx)                             # (nb, 3)
            targs = torch.zeros(nb, 3)
            targs[valid] = pts_t[target_idx[valid]]

            preds_mm = torch.round(preds * RESID_SCALE).int()
            targs_mm = torch.round(targs * RESID_SCALE).int()
            resid_mm = (targs_mm - preds_mm)[valid].numpy()   # int32 residuals

            raw_vals = pts_mm[target_idx[valid]]               # int32 raw coords

            res_per_block.append(resid_mm.flatten())
            raw_per_block.append(raw_vals.flatten())

            recon = (preds_mm + (targs_mm - preds_mm)).float() / RESID_SCALE
            ctx[:, :-3] = ctx[:, 3:]
            ctx[:, -3:] = recon

    raw_all = np.concatenate(raw_per_block)
    res_all = np.concatenate(res_per_block)

    return raw_per_block, res_per_block, raw_all, res_all


BG      = "#0d0d0d"
ACCENT1 = "#e06c75"
ACCENT2 = "#56b6c2"


def make_frame(raw_snap, res_snap, raw_all, res_all, n_bins=120):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=110)
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(wspace=0.35)

    raw_centre = raw_all.mean()
    raw_snap_c = raw_snap - raw_centre

    # shared x-limits: use full-data extents
    raw_all_c  = raw_all - raw_centre
    x_raw_lim  = np.percentile(np.abs(raw_all_c), 99.5)
    x_res_lim  = np.percentile(np.abs(res_all),   99.5)

    panels = [
        (axes[0], raw_snap_c, raw_all_c, x_raw_lim, ACCENT1,
         "Raw quantised coordinates\n(centred, all axes)", "Coordinate value (int32 units)"),
        (axes[1], res_snap,   res_all,   x_res_lim,  ACCENT2,
         "MLP residuals\n(target − prediction)", "Residual (int32 units)"),
    ]

    for ax, data, ref_all, xlim, color, title, xlabel in panels:
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        bins = np.linspace(-xlim, xlim, n_bins + 1)

        if len(data):
            ax.hist(data, bins=bins, color=color, alpha=0.85, linewidth=0)

        counts_full, _ = np.histogram(ref_all, bins=bins)
        ax.stairs(counts_full, bins, color=color, alpha=0.18, linewidth=1.2)

        ax.set_xlim(-xlim, xlim)
        ax.set_yscale("log")
        ax.set_ylim(1, counts_full.max() * 3)

        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.set_xlabel(xlabel, color="#aaa", fontsize=8)
        ax.set_ylabel("Count (log)", color="#aaa", fontsize=8)
        ax.tick_params(colors="#aaa", labelsize=7)

        # std annotation
        if len(data):
            sd = data.std()
            ax.axvline( sd, color=color, lw=0.8, ls="--", alpha=0.6)
            ax.axvline(-sd, color=color, lw=0.8, ls="--", alpha=0.6)
            ax.text(0.97, 0.94, f"std = {sd:,.0f}", transform=ax.transAxes,
                    ha="right", va="top", color=color, fontsize=8)

    # compression ratio annotation on final state
    if len(res_snap) == len(res_all):
        ratio = raw_all_c.std() / max(res_all.std(), 1)
        fig.text(0.5, 0.01,
                 f"Residual std is {ratio:.0f}x smaller than raw coordinate spread  "
                 f"| {len(res_all):,} predicted points",
                 ha="center", color="#aaa", fontsize=8.5)

    fig.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=BG, dpi=110)
    plt.close(fig)
    buf.seek(0)
    return imageio.v2.imread(buf)


def make_gif(input_path, model_path, out_path, n_anim_frames=60, hold_frames=20, fps=20):
    print(f"Loading and running MLP encoder: {input_path}")
    raw_blocks, res_blocks, raw_all, res_all = compute_raw_and_residuals(input_path, model_path)
    print(f"  {len(raw_all):,} predicted points  |  residual std = {res_all.std():.2f} int32 units")

    total_blocks = len(raw_blocks)

    checkpoints = np.linspace(0, total_blocks, n_anim_frames + 1, dtype=int)

    frames = []
    for i, end in enumerate(checkpoints[1:]):
        print(f"\r  rendering frame {i+1}/{n_anim_frames} ...", end="", flush=True)
        raw_snap = np.concatenate(raw_blocks[:end]) if end else np.array([])
        res_snap = np.concatenate(res_blocks[:end]) if end else np.array([])
        frames.append(make_frame(raw_snap, res_snap, raw_all, res_all))

    # hold on final frame
    for _ in range(hold_frames):
        frames.append(frames[-1])

    print()
    print(f"Saving GIF -> {out_path}")
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"Done. {os.path.getsize(out_path)/1024:.0f} KB  "
          f"({len(frames)} frames @ {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description="Residual distribution tightening GIF")
    parser.add_argument("input", help="Original .bin point cloud")
    parser.add_argument("--model", default="models/grid_search/mlp_c3_h256.pth")
    parser.add_argument("--out",    default="residual_dist.gif")
    parser.add_argument("--frames", type=int, default=60, help="Animation frames")
    parser.add_argument("--hold",   type=int, default=25, help="Hold frames at end")
    parser.add_argument("--fps",    type=int, default=20)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[error] File not found: {args.input}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"[error] Model not found: {args.model}")
        sys.exit(1)

    make_gif(args.input, args.model, args.out,
             n_anim_frames=args.frames, hold_frames=args.hold, fps=args.fps)


if __name__ == "__main__":
    main()
