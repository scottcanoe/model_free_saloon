from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
import orjson
import pandas as pd
import quaternion
import trimesh
from matplotlib import pyplot as plt
from monty_utils.habitat_dataset import HabitatDataset
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as ScipyRotation
from tbp.monty.frameworks.actions.actions import QuaternionWXYZ
from vedo import Mesh, Plotter, Points, Sphere

from model_free_saloon import project

RotationLike = Union[quaternion.quaternion, ArrayLike, ScipyRotation]
QuaternionLike = Union[QuaternionWXYZ, quaternion.quaternion]


def center_val(arr: np.ndarray) -> np.generic:
    row_mid, col_mid = arr.shape[0] // 2, arr.shape[1] // 2
    return arr[row_mid, col_mid]


def extract(list_of_dicts: list[dict], key: str) -> np.ndarray:
    return np.array([dct[key] for dct in list_of_dicts])


def extract_centers(data: list[np.ndarray]) -> np.ndarray:
    return np.array([center_val(arr) for arr in data])


def extract_observations(sm_stats: dict) -> dict[str, list[np.ndarray]]:
    raw_observations = sm_stats["raw_observations"]
    rgba = [np.array(dct["rgba"]) for dct in raw_observations]
    depth = [np.array(dct["depth"]) for dct in raw_observations]
    grid_shape = depth[0].shape
    semantic_3d = [np.array(dct["semantic_3d"]) for dct in raw_observations]
    xyz, surface = [], []
    for i, sem in enumerate(semantic_3d):
        xyz.append(sem[:, 0:3].reshape(grid_shape + (3,)))
        surface.append(sem[:, 3].reshape(grid_shape).astype(int) > 0)
    return {
        "rgba": rgba,
        "depth": depth,
        "xyz": xyz,
        "surface": surface,
    }


def load_detailed_stats(path: os.PathLike, episode: int) -> dict:
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i == episode:
                return list(orjson.loads(line).values())[0]
    raise ValueError(f"Episode {episode} not found in {path}")


class DataSource:
    def __init__(self, exp_dir: os.PathLike, episode: int = 0):
        self._exp_dir = Path(exp_dir).expanduser()
        self._stats_path = self._exp_dir / "detailed_run_stats.json"

        self._episode = episode

        self.stats = {}
        self.target = {}
        self.view_finder = {}
        self.gsg = {}
        self.n_steps = 0

        self.set_episode(episode)

    @property
    def episode(self) -> int | None:
        return self._episode

    def set_episode(self, episode: int) -> None:
        self.stats = load_detailed_stats(self._stats_path, episode)
        self._episode = episode

        # - target object info
        target_info = self.stats["target"]
        self.target = {
            "object": target_info["primary_target_object"],
            "position": np.array(target_info["primary_target_position"]),
        }
        angles = target_info["primary_target_rotation_euler"]
        self.target["rotation"] = as_scipy_rotation(angles, degrees=True)

        # patch sensor
        sm_stats = self.stats["SM_0"]
        self.patch = {}
        self.patch["observations"] = extract_observations(sm_stats)

        # - view-finder
        sm_stats = self.stats["SM_1"]
        self.view_finder = {}
        self.view_finder["observations"] = extract_observations(sm_stats)

        # - gsg
        self.gsg = sm_stats["gsg_telemetry"]

        # - motor system

        self.n_steps = len(self.view_finder["observations"]["rgba"])


exp_dir = project.paths.results / "snapshots"
json_path = exp_dir / "detailed_run_stats.json"
for episode in range(10):
    stats = load_detailed_stats(json_path, episode)
    sm_stats = stats["SM_1"]
    extract_observations(sm_stats)
    rgba = np.array(sm_stats["raw_observations"][0]["rgba"])
    depth = np.array(sm_stats["raw_observations"][0]["depth"])
    on_obj = depth < 1.0
    print(f"min_depth: {np.min(depth)}")
    perc = on_obj.sum() / on_obj.size
    print(f"perc: {100 * perc:.2f}%")

    plt.imshow(rgba)
    plt.show()
