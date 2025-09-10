from __future__ import annotations

import json
import os
import shutil
from multiprocessing import Value
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from matplotlib import colors
from matplotlib.animation import FuncAnimation, PillowWriter
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as SciPyRotation
from tbp.monty.frameworks.models.motor_system import clean_motor_system_state

from data_utils import RESULTS_DIR, DetailedJSONStatsInterface, extract_raw
from model_free_saloon import project


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    # points: Nx3 array
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    voxel_map = {}
    keep_indices = []

    for idx, voxel in enumerate(map(tuple, voxel_indices)):
        if voxel not in voxel_map:
            voxel_map[voxel] = idx
            keep_indices.append(idx)
    return np.array(keep_indices, dtype=np.int64)


def extract(list_of_dicts: list[dict], key: str) -> np.ndarray:
    return np.array([dct[key] for dct in list_of_dicts])


def grid_center(arr: np.ndarray) -> np.generic | np.ndarray:
    n_rows, n_cols = arr.shape[0], arr.shape[1]
    return arr[n_rows // 2, n_cols // 2]


def grid_raw_observation(raw_observation: dict) -> dict[str, np.ndarray]:
    """Convert the raw observation into a grid of data.

    Args:
        raw_observation: The raw observation.

    Returns:
        The grid of data.
    """
    rgba = raw_observation["rgba"]
    grid_shape = rgba.shape[:2]
    semantic_3d = raw_observation["semantic_3d"]
    points = semantic_3d[:, 0:3].reshape(grid_shape + (3,))
    on_object = semantic_3d[:, 3].reshape(grid_shape).astype(int) > 0
    return {
        "rgba": rgba,
        "depth": raw_observation["depth"],
        "points": points,
        "on_object": on_object,
    }


exp_dir = project.paths.results / "ycb_dev"
all_stats = DetailedJSONStatsInterface(exp_dir / "detailed_run_stats.json")
stats = all_stats[0]

sm1_depth = extract_raw(stats, "SM_1", "depth")
sm1_centers = np.array([grid_center(arr) for arr in sm1_depth])

sm0_depth = extract_raw(stats, "SM_0", "depth")
sm0_centers = np.array([grid_center(arr) for arr in sm0_depth])


ms_telemetry = stats["motor_system"]["telemetry"]

states = extract(ms_telemetry, "state")
goals = extract(ms_telemetry, "driving_goal_state")
processed_observations = extract(ms_telemetry, "processed_observations")
actions = extract(ms_telemetry, "action")
policies = extract(ms_telemetry, "policy_id")
# for g in goals:
#     print(g)

sm_dict = stats["SM_1"]
gsg_telemetry = sm_dict["gsg_telemetry"]
tel = gsg_telemetry[10]
df = tel["decay_field"]
kernels = df["kernels"]
goals = tel["output_goal_state"]
for k in kernels:
    print(k)
