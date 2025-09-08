from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
import orjson
import pandas as pd
import quaternion
import trimesh
from monty_utils.habitat_dataset import HabitatDataset
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as ScipyRotation
from tbp.monty.frameworks.actions.actions import QuaternionWXYZ
from vedo import Mesh, Plotter, Points, Sphere

from model_free_saloon import project

RotationLike = Union[quaternion.quaternion, ArrayLike, ScipyRotation]
QuaternionLike = Union[QuaternionWXYZ, quaternion.quaternion]


MONTY_DATASET_DIR = Path(os.environ.get("MONTY_DATA", "~/tbp/data")).expanduser()
OBJECT_DATASET_DIRS = {
    "ycb": MONTY_DATASET_DIR / "habitat" / "objects" / "ycb",
    "targets": MONTY_DATASET_DIR / "targets",
}


def as_scipy_rotation(
    obj: quaternion.quaternion | ArrayLike | ScipyRotation,
    **kwargs,
    ) -> ScipyRotation:
    """Convert a rotation description to a rotation matrix.

    Args:
        obj: The rotation to convert. This can be one of the following:
            - scipy.spatial.transform.Rotation (returned as-is)
            - quaternion.quaternion
            - 4-length array representing a quaternion. By default, it is assumed to
              be in scalar-first order. Supply `scalar_first=False` if it's in
              x, y, z, w order.
            - 3-length array representing euler angles. By default, angles are assumed
              to be in degrees. Supply `degrees=False` for radians.
              be in degrees. Supply `degrees=False` to override.
            - 3x3 rotation matrix.
            
    Returns:
        A scipy.spatial.transform.Rotation instance.
    """
        
    if isinstance(obj, ScipyRotation):
        return obj

    if isinstance(obj, quaternion.quaternion):
        return ScipyRotation.from_quat([obj.x, obj.y, obj.z, obj.w])
    
    arr = np.array(obj)

    if arr.shape == (4,):
        if kwargs.get("scalar_first", True):
            arr = [arr[1], arr[2], arr[3], arr[0]]
        return ScipyRotation.from_quat(arr)

    if arr.shape == (3,):
        axes = kwargs.get("axes", "xyz")
        degrees = kwargs.get("degrees", True)
        return ScipyRotation.from_euler(axes, arr, degrees=degrees)

    if arr.shape == (3, 3):
        return ScipyRotation.from_matrix(arr)

    raise ValueError(f"Invalid rotation description: {obj}")


def load_mesh(object_name: str, dataset: str = "ycb") -> Mesh:
        """Reads a 3D object file in glb format and returns a Vedo Mesh object.

        Args:
            obj_name: Name of the object to load.

        Returns:
            vedo.Mesh object with UV texture and transformed orientation.
        """
        dataset_dir = OBJECT_DATASET_DIRS[dataset]
        ds = HabitatDataset(dataset_dir)
        cfg = ds.load_config(object_name)
        mesh_path = cfg.attrs.render_asset
        if dataset == "ycb":
            mesh_path = mesh_path.parent / "textured.glb.orig"
        
        trimesh_scene = trimesh.load(mesh_path, file_type="glb")
        trimesh_mesh = list(trimesh_scene.geometry.values())[0]
        vertices = trimesh_mesh.vertices
        faces = trimesh_mesh.faces
        
        mesh = Mesh([vertices, faces])
        mesh.texture(
            tname=np.array(trimesh_mesh.visual.material.baseColorTexture),
            tcoords=trimesh_mesh.visual.uv,
        )

        mesh.shift(-np.mean(mesh.bounds().reshape(3, 2), axis=1))
        if dataset == "ycb":
            mesh.rotate_x(-90)
        return mesh


def mesh_rotate(mesh: Mesh, rotation: RotationLike, **kwargs) -> Mesh:
    rot = as_scipy_rotation(rotation, **kwargs)
    axis = rot.as_rotvec()
    angle = np.linalg.norm(axis)
    if np.isclose(angle, 0):
        return mesh
    axis = axis / angle
    mesh.rotate(angle, axis, rad=True)
    return mesh

def mesh_translate(mesh: Mesh, position: ArrayLike) -> Mesh:
    mesh.shift(position)
    return mesh

def mesh_set_pose(
    mesh: Mesh,
    rotation: ArrayLike | None = None,
    position: ArrayLike | None = None,
    ) -> Mesh:
    if rotation is not None:
        mesh = mesh_rotate(mesh, rotation)
    if position is not None:
        mesh = mesh_translate(mesh, position)
    return mesh

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



class InteractivePlotter:
    def __init__(self, ds: DataSource):
        self.ds = ds
        self.step = 0
        self.plotter = Plotter(shape=(1, 1))
        self.plotter.add_callback("key_press", self.on_key_press)


        self.target_mesh = load_mesh(self.ds.target["object"], dataset="ycb")
        self.target_mesh = mesh_set_pose(
            self.target_mesh,
            rotation=self.ds.target["rotation"],
            position=self.ds.target["position"],
        )
        self.cam = dict(
            pos=(0.0205607, 1.53033, 0.416332),
            focal_point=(-7.67518e-3, 1.49612, -0.0129118),
            viewup=(0, 1, 0),
            roll=0,
            distance=0.431530,
            clipping_range=(0.307518, 0.557784),
        )


        self.goals = None
        self.best_goal = None
        self.selected_goal = None

        self.view_finder_center = None
        self.patch_center = None


        self.plotter.show(
            self.target_mesh,
            viewup="y",
            camera=self.cam,
            axes=4,
            interactive=False,
        )
        self.set_step(0)


    def set_step(self, step: int):

        if step < 0 or step >= self.ds.n_steps:
            return
        
        print(f"Setting step to {step}")
        self.step = step
        
        self.update_goals()

        self.plotter.interactive = True

    
    def update_goals(self):

        if self.goals is not None:
            self.plotter.remove(self.goals)
        if self.best_goal is not None:
            self.plotter.remove(self.best_goal)
        if self.selected_goal is not None:
            self.plotter.remove(self.selected_goal)

        goal_states = self.ds.gsg[self.step]["output_goal_state"]
        if len(goal_states) == 0:
            print("No goals")
            return
        locations = extract(goal_states, "location")
        confidences = extract(goal_states, "confidence")
        self.goals = Points(locations, r=15)
        self.goals = self.goals.cmap("inferno", confidences, vmin=0, vmax=1)
        self.goals = self.goals.lighting("off")
        if len(goal_states) == 1:
            best_location = locations[0]
        else:
            inds = np.argsort(confidences)[::-1]
            locations = [locations[i] for i in inds]
            confidences = [confidences[i] for i in inds]
            if np.isclose(confidences[0], confidences[1]):
                print('Two goals with the same confidence')
            best_location = locations[0]
        self.best_goal = Sphere(best_location, r=.01, c="black", alpha=0.3)

        self.plotter.show(self.goals, self.best_goal, interactive=True)


    def on_key_press(self, event):
        if event.keypress == "b":
            self.set_step(self.step + 1)
        elif event.keypress == "v":
            self.set_step(self.step - 1)


exp_dir = project.paths.results / "ycb_dev"
ds = DataSource(exp_dir)
p = InteractivePlotter(ds)

# Debugging off-object stuff. Can ignore.
patch_on_obj = extract_centers(ds.patch["observations"]["surface"])
viewfinder_on_obj = extract_centers(ds.view_finder["observations"]["surface"])
df = pd.DataFrame(dict(patch=patch_on_obj, viewfinder=viewfinder_on_obj))
print(df)
