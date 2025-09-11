from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import ArrayLike
from tbp.monty.frameworks.environments.ycb import DISTINCT_OBJECTS
from vedo import Mesh, Plotter, Points, Sphere

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


class ObjectModel:
    """Mutable wrapper for object models.

    Args:
        pos (ArrayLike): The points of the object model as a sequence of points
          (i.e., has shape (n_points, 3)).
        features (Optional[Mapping]): The features of the object model. For
          convenience, the features become attributes of the ObjectModel instance.
    """

    def __init__(
        self,
        pos: ArrayLike,
        features: dict | None = None,
    ):
        self.pos = np.asarray(pos, dtype=float)
        if features:
            for key, value in features.items():
                setattr(self, key, np.asarray(value))

    @property
    def x(self) -> np.ndarray:
        return self.pos[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.pos[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.pos[:, 2]


def load_object_model(
    model_path: os.PathLike,
    object_name: str,
    lm_id: int = 0,
) -> ObjectModel:
    """Load an object model from a pretraining experiment.

    Args:
        model_name (str): The name of the model to load (e.g., `dist_agent_1lm`).
        object_name (str): The name of the object to load (e.g., `mug`).
        checkpoint (Optional[int]): The checkpoint to load. Defaults to None. Most
          pretraining experiments aren't checkpointed, so this is usually None.
        lm_id (int): The ID of the LM to load. Defaults to 0.

    Returns:
        ObjectModel: The loaded object model.

    Example:
        >>> model = load_object_model("dist_agent_1lm", "mug")
        >>> model -= [0, 1.5, 0]
        >>> rotation = R.from_euler("xyz", [0, 90, 0], degrees=True)
        >>> rotated = model.rotated(rotation)
        >>> print(model.rgba.shape)
        (1354, 4)
    """

    data = torch.load(model_path)
    data = data["lm_dict"][lm_id]["graph_memory"][object_name]["patch"]
    points = np.array(data.pos, dtype=float)
    features = ["rgba"]
    feature_dict = {}
    for feature in features:
        if feature not in data.feature_mapping:
            print(f"WARNING: Feature {feature} not found in data.feature_mapping")
            continue
        idx = data.feature_mapping[feature]
        feature_data = np.array(data.x[:, idx[0] : idx[1]])
        if feature == "rgba":
            feature_data = feature_data / 255.0
        feature_dict[feature] = feature_data

    return ObjectModel(points, features=feature_dict)


bio_model_path = project.paths.results / "pretrain_bio/pretrained/model.pt"
standard_model_path = project.paths.results / "pretrain_standard/pretrained/model.pt"




object_name = "potted_meat_can"
# model_path = bio_model_path
model_path = standard_model_path
# model_path = f"/Users/scott/tbp/projects/model_free_saloon/results/pretrain_standard/pretrain_standard-parallel_train_episode_{object_name}/pretrained/model.pt"

lm_id = 0

data = torch.load(model_path)
data = data["lm_dict"][lm_id]["graph_memory"][object_name]["patch"]
points = np.array(data.pos, dtype=float)
idx = data.feature_mapping["hsv"]
hsv = np.array(data.x[:, idx[0] : idx[1]])
rgb = matplotlib.colors.hsv_to_rgb(hsv)  # Nx3, values in [0,1]
alpha = np.ones((rgb.shape[0], 1), dtype=rgb.dtype)
rgba = np.concatenate([rgb, alpha], axis=1)  # Nx4

print(f"Number of points: {points.shape[0]}")
plotter = Plotter()
pts = Points(points, r=15)
plotter.show(pts, axes=4, interactive=True)
