from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import orjson
import pandas as pd
import quaternion
import scipy.ndimage as ndimage
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation as ScipyRotation
from tbp.monty.frameworks.actions.actions import QuaternionWXYZ
from tbp.monty.frameworks.environments.ycb import DISTINCT_OBJECTS
from tbp.monty.frameworks.models.saliency import (
    MinimumBarrierSalience,
    RobustBackgroundSalience,
    SpectralResidualSalience,
    UniformSalience,
)
from tbp.monty.frameworks.models.saliency.bio import compute_saliency, salience_rgbd

from model_free_saloon import project

SNAPSHOT_DIR = project.paths.data / "snapshots"


def load_detailed_stats(path: os.PathLike, episode: int) -> dict:
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i == episode:
                return list(orjson.loads(line).values())[0]
    raise ValueError(f"Episode {episode} not found in {path}")


def save_snapshots():
    exp_dir = project.paths.results / "snapshots"
    json_path = exp_dir / "detailed_run_stats.json"
    for episode in range(10):
        stats = load_detailed_stats(json_path, episode)
        sm_stats = stats["SM_1"]
        rgba = np.array(sm_stats["raw_observations"][0]["rgba"])
        depth = np.array(sm_stats["raw_observations"][0]["depth"])

        on_obj = depth < 1.0
        print(f"min_depth: {np.min(depth)}")
        perc = on_obj.sum() / on_obj.size
        print(f"perc: {100 * perc:.2f}%")

        object_name = DISTINCT_OBJECTS[episode]
        obj_dir = SNAPSHOT_DIR / object_name
        obj_dir.mkdir(parents=True, exist_ok=True)

        np.save(obj_dir / "rgba.npy", rgba)
        np.save(obj_dir / "depth.npy", depth)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(rgba)
        on_obj_depth = depth[on_obj]
        vmin = np.min(on_obj_depth) * 0.9
        vmax = np.max(on_obj_depth) * 1.1
        axes[1].imshow(depth, vmin=vmin, vmax=vmax, cmap="gray")
        for ax in axes:
            ax.axis("off")
        fig.savefig(obj_dir / "plot.png")
        plt.close()


def load_rgba(object_name: str) -> np.ndarray:
    return np.load(SNAPSHOT_DIR / object_name / "rgba.npy").astype(np.uint8)


def load_depth(object_name: str) -> np.ndarray:
    return np.load(SNAPSHOT_DIR / object_name / "depth.npy")


def load_obs(object_name: str) -> dict:
    return {
        "rgba": load_rgba(object_name),
        "depth": load_depth(object_name),
    }


def save_saliency_maps():
    methods = {
        "spectral_residual": SpectralResidualSalience(),
        "robust_background": RobustBackgroundSalience(),
        "minimum_barrier": MinimumBarrierSalience(),
    }

    exp_dir = project.paths.results / "snapshots"
    json_path = exp_dir / "detailed_run_stats.json"
    for episode in range(10):
        object_name = DISTINCT_OBJECTS[episode]
        obj_dir = SNAPSHOT_DIR / object_name
        obs = load_obs(object_name)

        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        method_names = list(methods.keys())
        for i, method_name in enumerate(method_names):
            method = methods[method_name]
            ax = axes[i]

            sal = method.compute_saliency_map(obs)
            ax.imshow(sal, vmin=0, vmax=1, cmap="gray")
            ax.set_title(method_name)
            ax.axis("off")

        fig.savefig(obj_dir / "saliency_maps.png")


def imshow(
    image,
    vmin=0,
    vmax=1,
    cmap="gray",
    fig=None,
):
    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.axis("off")
    fig.show()
    return fig, ax


# ---------- main visualizers ----------

def _robust_display(img, lo=1.0, hi=99.0):
    """
    Percentile-based contrast stretch for visualization.
    Returns a view rescaled to [0,1] using [p_lo, p_hi] clip.
    """
    img = img.astype(np.float32)
    lo_v, hi_v = np.percentile(img, [lo, hi])
    if hi_v <= lo_v + 1e-6:
        return np.zeros_like(img)
    v = np.clip((img - lo_v) / (hi_v - lo_v), 0.0, 1.0)
    return v


def _nice_layout(n, max_cols=4):
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    return rows, cols


def visualize_maps_grid(
    maps: Dict[str, np.ndarray],
    order: Optional[List[str]] = None,
    title: str = "Feature maps",
    robust_clip=(1.0, 99.0),
    cmap="magma",
):
    """
    Show all maps (2D arrays) in a grid with the same color scaling policy.
    'maps' is the dict returned by your pipeline (keys like 'L','RG','BY','OR','D','SR','S').
    """
    # choose order
    keys = order if order is not None else list(maps.keys())
    keys = [k for k in keys if maps[k].ndim == 2]  # only 2D maps

    # compute global percentiles per-map (independent scaling)
    lo, hi = robust_clip
    rows, cols = _nice_layout(len(keys))

    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, k in zip(axes, keys):
        vis = _robust_display(maps[k], lo=lo, hi=hi)
        im = ax.imshow(vis, cmap=cmap, interpolation="nearest")
        ax.set_title(k, fontsize=12)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # hide any spare axes
    for ax in axes[len(keys) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    plt.show()


def plot_histograms(
    maps: Dict[str, np.ndarray],
    order: Optional[List[str]] = None,
    bins: int = 64,
    title: str = "Per-map value distributions",
):
    """
    Quick look at each map's distribution (helps diagnose overly sparse/flat maps).
    """
    keys = order if order is not None else list(maps.keys())
    keys = [k for k in keys if maps[k].ndim == 2]

    rows, cols = _nice_layout(len(keys))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, k in zip(axes, keys):
        m = maps[k].astype(np.float32).ravel()
        ax.hist(m, bins=bins, range=(0.0, 1.0))
        ax.set_title(k, fontsize=12)
        ax.set_xlim(0, 1)

    for ax in axes[len(keys) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    plt.show()


def summarize_maps_stats(
    maps: Dict[str, np.ndarray],
    order: Optional[List[str]] = None,
    percentiles=(0, 25, 50, 75, 95, 99),
):
    """
    Print summary stats for each map (min/max + selected percentiles).
    """
    keys = order if order is not None else list(maps.keys())
    keys = [k for k in keys if maps[k].ndim == 2]

    for k in keys:
        m = maps[k].astype(np.float32)
        q = np.percentile(m, percentiles)
        print(
            f"{k:>3}  min={m.min():.3f}  "
            + "  ".join([f"p{p:02d}={v:.3f}" for p, v in zip(percentiles, q)])
            + f"  max={m.max():.3f}"
        )


# ---------- (optional) quick composite panel ----------


def visualize_all(
    rgb: Optional[np.ndarray],
    maps: Dict[str, np.ndarray],
    map_order: Optional[List[str]] = None,
):
    """
    Convenience wrapper to show: grid, overlays (if rgb provided), and histograms,
    then print numeric summaries.
    """
    default_order = ["L", "RG", "BY", "OR", "SR", "D", "S"]
    order = (
        map_order if map_order is not None else [k for k in default_order if k in maps]
    )

    visualize_maps_grid(maps, order=order, title="Feature maps (robust scaled)")
    plot_histograms(maps, order=order)
    summarize_maps_stats(maps, order=order)


object_name = "bowl"
obs = load_obs(object_name)
rgba = obs["rgba"]
depth = obs["depth"]
rgb = rgba[:, :, :3]
rgb = rgb / 255.0

# gray = rgb_to_gray(rgb)
# imshow(gray)

# gx, gy, mag = sobel_filter(gray)
# imshow(gx)
# imshow(gy)
# imshow(mag)


# maps = salience_rgbd(rgb, depth, weight_scheme=None)

imshow(rgb)
S = compute_saliency(obs)
# imshow(S)
maps = salience_rgbd(rgb, depth)


visualize_all(rgb, maps)  # full panel
# Or call pieces:
# visualize_maps_grid(maps, order=["L", "RG", "BY", "OR", "SR", "D", "S"])
# visualize_overlays(rgb, maps, order=["L","RG","BY","OR","D","S"])
# plot_histograms(maps, order=["L","RG","BY","OR","SR","D","S"])
# summarize_maps_stats(maps, order=["L","RG","BY","OR","SR","D","S"])
