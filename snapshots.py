from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
import orjson
import pandas as pd
import quaternion
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as ScipyRotation
from tbp.monty.frameworks.actions.actions import QuaternionWXYZ
from tbp.monty.frameworks.environments.ycb import DISTINCT_OBJECTS
from vedo import Mesh, Plotter, Points, Sphere

from model_free_saloon import project

RotationLike = Union[quaternion.quaternion, ArrayLike, ScipyRotation]
QuaternionLike = Union[QuaternionWXYZ, quaternion.quaternion]

SNAPSHOT_DIR = project.paths.data / "snapshots"


def load_detailed_stats(path: os.PathLike, episode: int) -> dict:
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i == episode:
                return list(orjson.loads(line).values())[0]
    raise ValueError(f"Episode {episode} not found in {path}")


def save_snapshot(episode: int):
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
    return np.load(SNAPSHOT_DIR / object_name / "rgba.npy")


def load_depth(object_name: str) -> np.ndarray:
    return np.load(SNAPSHOT_DIR / object_name / "depth.npy")
