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

from model_free_saloon import project

RED = np.array([1.0, 0.0, 0.0])
GREE = np.array([0.0, 1.0, 0.0])
BLUE = np.array([0.0, 0.0, 1.0])
YELLOW = np.array([1.0, 1.0, 0.0])


RotationLike = Union[quaternion.quaternion, ArrayLike, ScipyRotation]
QuaternionLike = Union[QuaternionWXYZ, quaternion.quaternion]

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


def sobel_filter(img: np.ndarray):
    """Apply Sobel filter to a grayscale image (2D array).
    Returns gradients in x, y, and the magnitude."""

    # Define Sobel kernels
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # Convolve image with kernels
    gx = ndimage.convolve(img.astype(np.float32), Kx)
    gy = ndimage.convolve(img.astype(np.float32), Ky)

    # Edge magnitude
    mag = np.hypot(gx, gy)  # sqrt(gx**2 + gy**2)
    mag = mag / (mag.max() + 1e-8)  # normalize [0,1]

    return gx, gy, mag


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image [H,W,3] to grayscale [H,W].
    Input can be uint8 (0–255) or float (0–1)."""

    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0  # normalize

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.float32)


def color_opponency_map(rgb, target):
    """
    Emphasize pixels matching a target color.

    Parameters
    ----------
    rgb : np.ndarray
        [H,W,3] float32 image in [0,1].
    target : tuple or list of 3
        Target color (R,G,B), e.g. (1,0,0) for red.

    Returns
    -------
    map : np.ndarray
        [H,W] float32 map in [0,1].
    """

    # target = np.array(target, dtype=np.float32)
    # target /= np.linalg.norm(target) + 1e-8  # normalize

    # dot product per pixel
    dot = np.tensordot(rgb, target, axes=([2], [0]))

    # rescale from [-1,1] → [0,1]
    map_ = (dot + 1) / 2.0
    return map_.astype(np.float32)


# ---------- helpers ----------


def norm01(x, eps=1e-8):
    x = x.astype(np.float32)
    mn, mx = np.min(x), np.max(x)
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def halfwave(x):
    # simple rectification like neuronal ON channels
    return np.maximum(x, 0.0).astype(np.float32)


def dog_center_surround(img, sigma_c, sigma_s, k=1.0):
    # Difference-of-Gaussians (center-surround)
    c = ndi.gaussian_filter(img, sigma=sigma_c, mode="reflect")
    s = ndi.gaussian_filter(img, sigma=sigma_s, mode="reflect")
    return k * (c - s).astype(np.float32)


def multiscale_cs(img, scales: Tuple[Tuple[float, float], ...]):
    # sum of rectified center-surround across scales
    acc = np.zeros(img.shape[:2], dtype=np.float32)
    for sc, ss in scales:
        m = halfwave(dog_center_surround(img, sc, ss))
        acc += norm01(m)
    return norm01(acc)


# ---------- color & luminance feature maps ----------


def rgb_to_gray(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def opponency_RG(rgb):
    # classic L-M (red-green) opponent; rectified into ON channels
    r, g = rgb[..., 0], rgb[..., 1]
    rg_on = halfwave(r - g)
    gr_on = halfwave(g - r)
    return rg_on, gr_on  # (red-on, green-on)


def opponency_BY(rgb):
    # S - (L+M) ~ blue - yellow (yellow ≈ (R+G)/2)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.5 * (r + g)
    b_on = halfwave(b - y)
    y_on = halfwave(y - b)
    return b_on, y_on  # (blue-on, yellow-on)


def color_salience(rgb):
    # center-surround on opponent channels + across-scale normalization
    scales = ((1.0, 3.0), (2.0, 6.0), (4.0, 12.0))
    rg_on, gr_on = opponency_RG(rgb)
    b_on, y_on = opponency_BY(rgb)

    RG = multiscale_cs(rg_on, scales) + multiscale_cs(gr_on, scales)
    BY = multiscale_cs(b_on, scales) + multiscale_cs(y_on, scales)
    return norm01(RG), norm01(BY)


def luminance_salience(rgb):
    lum = rgb_to_gray(rgb)
    scales = ((1.0, 3.0), (2.0, 6.0), (4.0, 12.0))
    return multiscale_cs(lum, scales)


# ---------- orientation / texture feature maps ----------


def sobel_xy(gray):
    # 3x3 Sobel via ndimage
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    gx = ndi.convolve(gray, Kx, mode="reflect")
    gy = ndi.convolve(gray, Ky, mode="reflect")
    return gx, gy


def orientation_energy(gray, num_bins: int = 4):
    """
    Simple orientation channels using gradient steering into bins.
    Returns a list of maps (length num_bins) plus an overall magnitude.
    """
    gx, gy = sobel_xy(gray)
    mag = np.hypot(gx, gy).astype(np.float32)
    ang = np.arctan2(gy, gx) + np.pi  # [0, 2pi)

    bins = []
    for k in range(num_bins):
        theta = k * np.pi / num_bins  # 0..pi
        # soft binning with cosine tuning (like simple cell)
        resp = np.cos(ang - theta)
        resp = halfwave(resp) * mag
        bins.append(norm01(resp))
    return bins, norm01(mag)


def oriented_salience(rgb, num_bins=4):
    gray = rgb_to_gray(rgb)
    bins, mag = orientation_energy(gray, num_bins=num_bins)
    # center-surround on each orientation channel
    scales = ((1.0, 3.0), (2.0, 6.0))
    acc = np.zeros_like(gray, dtype=np.float32)
    for ch in bins:
        acc += multiscale_cs(ch, scales)
    # mix with overall edge magnitude for sharper contours
    return norm01(0.7 * acc + 0.3 * mag)


# ---------- depth feature maps ----------


def depth_nearness(depth, robust=True):
    """
    Convert depth (meters; 0 or NaN = invalid) to nearness in [0,1],
    emphasizing nearer objects.
    """
    d = depth.astype(np.float32).copy()
    d[~np.isfinite(d) | (d <= 0)] = np.nan
    if robust:
        med = np.nanmedian(d)
        mad = np.nanmedian(np.abs(d - med)) + 1e-6
        z = (med - d) / (1.4826 * mad + 1e-6)  # nearer -> larger positive
    else:
        z = -d
    z = np.nan_to_num(z, nan=0.0)
    return norm01(z)


def depth_confidence(depth):
    """
    Cheap confidence: valid pixels with low local variance.
    """
    d = depth.astype(np.float32)
    valid = (np.isfinite(d) & (d > 0)).astype(np.float32)
    sm = ndi.gaussian_filter(np.nan_to_num(d, nan=0.0), 1.0, mode="reflect")
    var = ndi.gaussian_filter(
        (np.nan_to_num(d, nan=0.0) - sm) ** 2, 2.0, mode="reflect"
    )
    var = norm01(var)
    conf = valid * (1.0 - var)
    return conf.astype(np.float32)


def depth_salience(depth):
    """
    Near-field pop-out + center-surround + depth edges.
    """
    near = depth_nearness(depth)  # nearer → higher
    # center-surround on nearness
    scales = ((1.0, 3.0), (2.0, 6.0), (4.0, 12.0))
    cs = multiscale_cs(near, scales)
    # add depth edges to hug object boundaries
    gx, gy = sobel_xy(near)
    edges = norm01(np.hypot(gx, gy))
    ds = norm01(0.8 * cs + 0.2 * edges)
    return ds


# ---------- fusion & post ----------


def fuse_maps(maps: Dict[str, np.ndarray], weights: Dict[str, float]):
    """
    Weighted geometric mean (robust to differing ranges, rewards consensus).
    maps: dict of {name: [H,W] map in [0,1]}
    weights: dict of {name: exponent weight}
    """
    # avoid log(0)
    eps = 1e-6
    log_acc = None
    for k, m in maps.items():
        w = float(weights.get(k, 1.0))
        m = np.clip(m, eps, 1.0).astype(np.float32)
        term = w * np.log(m)
        log_acc = term if log_acc is None else (log_acc + term)
    fused = np.exp(log_acc)
    return norm01(fused)


def spectral_residual_saliency(gray):
    # gray: [H,W] float32 in [0,1]
    f = np.fft.fft2(gray)
    amp = np.abs(f)
    log_amp = np.log(amp + 1e-8)
    # local average in log-spectrum (box or gaussian)
    log_avg = ndi.uniform_filter(log_amp, size=3, mode="reflect")
    resid = log_amp - log_avg
    # reconstruct with original phase
    f_resid = np.exp(resid + 1j * np.angle(f))
    sal = np.abs(np.fft.ifft2(f_resid)) ** 2
    sal = ndi.gaussian_filter(sal, 3.0, mode="reflect")
    return norm01(sal.astype(np.float32))


def fuse_maps_mean(maps, weights):
    num = 0.0
    den = 0.0
    for k, m in maps.items():
        w = float(weights.get(k, 1.0))
        num += w * m
        den += w
    S = num / (den + 1e-8)
    # mild contrast boost (acts like a soft sigmoid)
    return norm01(S**1.2)


from typing import Dict, List, Optional

import numpy as np


def _norm01(x, eps=1e-8):
    mn, mx = x.min(), x.max()
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def _maps_to_matrix(maps: Dict[str, np.ndarray], keys: Optional[List[str]] = None):
    """Stack maps -> X shape [N, K] with per-map z-scoring."""
    if keys is None:
        keys = list(maps.keys())
    H, W = next(iter(maps.values())).shape
    X = []
    for k in keys:
        m = maps[k].astype(np.float32).reshape(-1)
        # z-score per map to compare structure not scale
        mu = m.mean()
        sd = m.std() + 1e-8
        X.append((m - mu) / sd)
    X = np.stack(X, axis=1)  # [N,K]
    return X, (H, W), keys


def corr_shrink_weights(
    maps: Dict[str, np.ndarray],
    base_weights: Dict[str, float],
    keys: Optional[List[str]] = None,
    power: float = 1.0,
):
    """
    Compute redundancy-aware weights: w_i' = w_i / (1 + sum_j |corr(i,j)|^{power}), j≠i
    Returns dict of shrunken, normalized weights.
    """
    X, _, keys = _maps_to_matrix(maps, keys)
    K = X.shape[1]
    # correlation matrix (abs)
    C = np.corrcoef(X, rowvar=False)  # [K,K]
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 0.0)
    R = np.sum(np.abs(C) ** power, axis=1)  # redundancy score per map
    # shrink base weights by redundancy
    w = np.array([float(base_weights.get(k, 1.0)) for k in keys], dtype=np.float32)
    w_shrunk = w / (1.0 + R)
    w_shrunk = np.clip(w_shrunk, 0.0, None)
    # normalize to keep total weight comparable
    if w_shrunk.sum() > 0:
        w_shrunk /= w_shrunk.sum()
    return {k: float(ws) for k, ws in zip(keys, w_shrunk)}, C


def fuse_mean_redundancy_aware(
    maps: Dict[str, np.ndarray],
    base_weights: Dict[str, float],
    keys: Optional[List[str]] = None,
    power: float = 1.0,
):
    """
    Arithmetic mean fusion with correlation-aware weight shrinking.
    """
    if keys is None:
        keys = list(maps.keys())
    w_shrunk, C = corr_shrink_weights(maps, base_weights, keys, power=power)
    # weighted mean
    num = 0.0
    den = 0.0
    for k in keys:
        wk = float(w_shrunk.get(k, 0.0))
        num = num + wk * maps[k].astype(np.float32)
        den = den + wk
    S = num / (den + 1e-8)
    return _norm01(S), w_shrunk, C


# ---------- end-to-end pipeline ----------


def salience_rgbd(rgb, depth=None, weight_scheme="indoor_default"):
    # --- features as before ---
    L = luminance_salience(rgb)
    RG, BY = color_salience(rgb)
    OR = oriented_salience(rgb)

    # new: spectral residual on luminance
    SR = spectral_residual_saliency(rgb_to_gray(rgb))

    maps = {"L": L, "RG": RG, "BY": BY, "OR": OR, "SR": SR}

    if depth is not None:
        D = depth_salience(depth)
        C = depth_confidence(depth)
        maps["D"] = norm01(D * (0.3 + 0.7 * C))

        valid = np.isfinite(depth) & (depth > 0)
        if np.any(valid):
            med = np.median(depth[valid])
            dmin = np.min(depth[valid])
            if (med - dmin) > 0.5:  # meters
                maps["D"] = norm01(maps["D"] ** 1.2)

    # weights
    if weight_scheme == "indoor_default":
        weights = {"L": 0.6, "RG": 0.8, "BY": 0.8, "OR": 0.9, "SR": 1.2}
        if depth is not None:
            weights["D"] = 0.8
    else:
        weights = {k: 1.0 for k in maps.keys()}

    # use mean fusion instead of geometric
    # S = fuse_maps_mean(maps, weights)
    S, _, _ = fuse_mean_redundancy_aware(maps, weights)
    # S = ndi.gaussian_filter(S, 0.1, mode="reflect")  # a touch of smoothing
    maps["S"] = norm01(S)

    return maps


def compute_saliency(obs: dict):
    rgb = obs["rgba"][:, :, :3] / 255.0
    depth = obs["depth"]
    maps = salience_rgbd(rgb, depth)
    return maps["S"]


# ---------- robust display helpers ----------


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


# ---------- main visualizers ----------


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


def visualize_overlays(
    rgb: np.ndarray,
    overlays: Dict[str, np.ndarray],
    order: Optional[List[str]] = None,
    robust_clip=(1.0, 99.0),
    alpha=0.45,
    cmap="jet",
    title="Overlays on RGB",
):
    """
    Overlay each map as a heatmap on top of the RGB image (useful for seeing alignment).
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "rgb must be HxWx3"
    keys = order if order is not None else list(overlays.keys())
    keys = [k for k in keys if overlays[k].ndim == 2]

    rows, cols = _nice_layout(len(keys))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, k in zip(axes, keys):
        ax.imshow(np.clip(rgb, 0.0, 1.0), interpolation="nearest")
        vis = _robust_display(overlays[k], *robust_clip)
        ax.imshow(vis, cmap=cmap, alpha=alpha, interpolation="nearest")
        ax.set_title(f"{k} overlay", fontsize=12)
        ax.set_axis_off()

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
    if rgb is not None:
        # Only overlay the most informative maps (and 'S')
        over_keys = [k for k in order if k in ("L", "RG", "BY", "OR", "D", "S")]
        overlays = {k: maps[k] for k in over_keys}
        visualize_overlays(rgb, overlays, order=over_keys, title="Overlays on RGB")
    plot_histograms(maps, order=order)
    summarize_maps_stats(maps, order=order)


object_name = "mug"
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
imshow(S)

# After running your pipeline:
# maps = salience_rgbd(rgb, depth)  # dict with keys like 'L','RG','BY','OR','SR','D','S'

# visualize_all(rgb, maps)  # full panel
# Or call pieces:
# visualize_maps_grid(maps, order=["L", "RG", "BY", "OR", "SR", "D", "S"])
# visualize_overlays(rgb, maps, order=["L","RG","BY","OR","D","S"])
# plot_histograms(maps, order=["L","RG","BY","OR","SR","D","S"])
# summarize_maps_stats(maps, order=["L","RG","BY","OR","SR","D","S"])
