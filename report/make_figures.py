"""Figures for the technical report (static PNG, light print theme).

fig_ablation.png   fold-0 ablations, PQ delta vs the crop-512 reference
fig_tau.png        pooled OOF PQ vs instance-scorer keep threshold tau
fig_overlay.png    OOF prediction vs annotator contours on a validation image
"""
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S2 = os.path.join(ROOT, "runs", "filament-segmentation-2026", "stage2_modeling")
sys.path.insert(0, S2)
from pipeline import PROC, gt_maps_for, imread  # noqa: E402
from scipy import ndimage  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def fig_ablation():
    ref = 0.3979
    rows = [("Native 2048, crop 640", 0.3257), ("50 epochs (vs 30)", 0.4026), ("EfficientNet-B3 encoder", 0.4188),
            ("UNet++ decoder, crop 768", 0.4303), ("Res 1536, crop 1152", 0.4285), ("Full image (crop 1024)", 0.4266),
            ("Crop 768", 0.4291), ("EffNet-B3 + crop 768", 0.4335)]
    rows.sort(key=lambda r: r[1] - ref)
    labels = [r[0] for r in rows]
    delta = np.array([r[1] - ref for r in rows])
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    y = np.arange(len(rows))
    ax.barh(y, delta, height=0.62, color=BLUE, edgecolor=SURFACE, linewidth=2)
    ax.axvline(0, color=INK2, linewidth=0.8)
    for yi, d in zip(y, delta):
        if d >= 0:
            ax.text(d + 0.002, yi, f"{d:+.3f}", va="center", ha="left", fontsize=7.5, color=INK2)
        else:  # long negative bar: label inside, in light ink
            ax.text(d + 0.003, yi, f"{d:+.3f}", va="center", ha="left", fontsize=7.5, color=SURFACE)
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.set_xlabel("Fold-0 PQ change vs. U-Net R34, crop 512 (PQ 0.398)")
    ax.set_xlim(delta.min() - 0.005, delta.max() + 0.02)
    ax.grid(axis="y", visible=False)
    ax.set_title("Context matters more than resolution", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ablation.png"), dpi=300, bbox_inches="tight", pad_inches=0.05)


def fig_tau():
    cv = json.load(open(os.path.join(S2, "runs", "2026-10-12-ens-r34-effb3-768-tta8", "scorer_v2", "scorer_cv_t0.3.json")))
    tau, pq = np.array(cv["tau_curve"]).T
    best = int(np.argmax(pq))
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    ax.plot(tau, pq, color=BLUE, linewidth=2)
    ax.axvline(pq[best] / 2, color=INK2, linewidth=0.8, linestyle="--")
    ax.text(pq[best] / 2 + 0.005, pq.min() + 0.004, "PQ/2 (theory)", fontsize=7.5, color=INK2)
    ax.plot(tau[best], pq[best], "o", markersize=7, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"τ={tau[best]:.2f}, PQ={pq[best]:.3f}", (tau[best], pq[best]), xytext=(10, 4),
                textcoords="offset points", fontsize=7.5, color=INK)
    ax.axhline(cv["baseline_min_area500"], color=GRID, linewidth=1.2)
    ax.text(tau[-1], cv["baseline_min_area500"] + 0.002, "no scorer (min area 500)", fontsize=7.5, color=INK2, ha="right")
    ax.set_xlabel("Keep threshold τ on predicted mean matched IoU")
    ax.set_ylabel("Pooled OOF PQ")
    ax.set_title("Instance scorer operating point", loc="left")
    fig.tight_layout()
    ax.set_ylim(pq.min() - 0.003, pq.max() + 0.008)
    fig.savefig(os.path.join(OUT, "fig_tau.png"), dpi=300, bbox_inches="tight", pad_inches=0.05)


def contours(mask):
    return mask & ~ndimage.binary_erosion(mask, iterations=1)


def fig_overlay(stem=None):
    run = os.path.join(S2, "runs", "2026-10-12-ens-r34-effb3-768-tta8")
    idx = pd.read_csv(os.path.join(PROC, "index.csv"))
    multi = idx.groupby("stem").agg(n_sets=("annot_set_id", "size"), n_fil=("n_filaments", "mean"))
    stem = stem or multi[multi.n_sets == 3].sort_values("n_fil").index[-3]  # a busy 3-annotator image
    img = imread(os.path.join(PROC, "images", stem + ".png"), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
    prob = np.load(os.path.join(run, "oof", stem + ".npy")).astype(np.float32) / 255
    gts = gt_maps_for(stem, 1024, idx)
    gt_any = np.mean([g > 0 for g in gts], axis=0)
    pred = prob > 0.3
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.5))
    for ax, title, layer, color in ((axes[0], f"Annotator agreement ({len(gts)} sets)", gt_any, ORANGE),
                                    (axes[1], "Ensemble prediction (thr 0.3)", pred, BLUE)):
        rgb = np.dstack([img] * 3).astype(np.float32) / 255
        if layer.dtype != bool:
            edge = contours(layer >= 0.5)
            faint = contours((layer > 0) & (layer < 0.5))
            rgb[faint] = np.array(matplotlib.colors.to_rgb(ORANGE)) * 0.5 + 0.5
        else:
            edge = contours(layer)
        rgb[ndimage.binary_dilation(edge)] = matplotlib.colors.to_rgb(color)
        ax.imshow(rgb[150:870, 150:870])
        ax.set_title(title, loc="left")
        ax.axis("off")
    fig.suptitle(f"{stem} (out-of-fold)", x=0.01, ha="left", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_overlay.png"), dpi=300, bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    fig_ablation()
    fig_tau()
    fig_overlay()
    print("figures written to", OUT)
