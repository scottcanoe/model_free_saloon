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

# Configuration
model_path = standard_model_path
lm_id = 0

# Determine model type for filename
model_type = "bio" if model_path == bio_model_path else "standard"
object_names = ['mug', 'bowl', 'potted_meat_can', 'spoon', 'strawberry', 
                'mustard_bottle', 'dice', 'golf_ball', 'c_lego_duplo', 'banana']

# Load model data
data = torch.load(model_path)
graph_memory = data["lm_dict"][lm_id]["graph_memory"]

# Create 2x5 subplot figure
fig = plt.figure(figsize=(15, 7))
point_counts = []

for i, object_name in enumerate(object_names):
    ax = fig.add_subplot(2, 5, i + 1, projection='3d')
    
    # Set better viewing angle for 3D objects
    # ax.view_init(elev=20, azim=45)
    
    try:
        # Load object data
        obj_data = graph_memory[object_name]["patch"]
        points = np.array(obj_data.pos, dtype=float)
        
        # Get HSV colors and convert to RGB
        if "hsv" in obj_data.feature_mapping:
            idx = obj_data.feature_mapping["hsv"]
            hsv = np.array(obj_data.x[:, idx[0] : idx[1]])
            rgb = matplotlib.colors.hsv_to_rgb(hsv)
            colors = rgb
        else:
            # Fallback to default color if HSV not available
            colors = 'blue'
        
        # Plot 3D scatter
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=colors, s=1, alpha=0.6)
        
        # Set title and labels
        ax.set_title(f'{object_name}\n({points.shape[0]} points)', fontsize=10)
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        
        # Make ticks smaller
        ax.tick_params(labelsize=6)
        
        point_counts.append(points.shape[0])
        print(f"{object_name}: {points.shape[0]} points")
        
    except KeyError:
        # Handle missing objects
        ax.text(0.5, 0.5, 0.5, f'{object_name}\n(not found)', 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=10)
        ax.set_title(f'{object_name}\n(not found)', fontsize=10)

plt.tight_layout(pad=3.0, h_pad=4.0)

# Calculate and display average
avg_points = np.mean(point_counts) if point_counts else 0
print(f"\nAverage number of points: {avg_points:.1f}")

# Save plot with average in filename to results directory
results_dir = project.paths.results
results_dir.mkdir(exist_ok=True)
plot_filename = results_dir / f'object_models_{model_type}_avg_{avg_points:.0f}_points.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved as: {plot_filename}")

plt.show()
