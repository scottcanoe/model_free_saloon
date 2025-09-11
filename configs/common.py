# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.config_utils.config_args import (
    EvalEvidenceLMLoggingConfig,
    ParallelEvidenceLMLoggingConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    InformedPolicyConfig,
    generate_action_list,
)
from tbp.monty.frameworks.environment_utils.transforms import (
    DepthTo3DLocations,
    MissingToMaxDepth,
)
from tbp.monty.frameworks.loggers.monty_handlers import (
    BasicCSVStatsHandler,
    ReproduceEpisodeHandler,
)
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.goal_state_generation import (
    EvidenceGoalStateGenerator,
)
from tbp.monty.frameworks.models.motor_policies import InformedPolicy
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.models.sensor_modules import (
    DetailedLoggingSM,
    FeatureChangeSM,
)
from tbp.monty.simulators.habitat import MultiSensorAgent
from tbp.monty.simulators.habitat.environment import (
    AgentConfig,
    HabitatEnvironment,
    ObjectConfig,
)

from model_free_saloon import project

from .selective_logging import RawObservationsFilter, SelectiveHandler

# - Path Settings
MONTY_DATA_DIR = Path(os.environ.get("MONTY_DATA", "~/tbp/data")).expanduser()
PRETRAIN_DIR = project.paths.pretrained_models
PRETRAINED_MODEL = str(Path(PRETRAIN_DIR) / "surf_agent_1lm_77obj" / "pretrained")
BIO_PRETRAINED_MODEL = str(Path(PRETRAIN_DIR) / "pretrain_bio" / "pretrained")
STANDARD_PRETRAINED_MODEL = str(Path(PRETRAIN_DIR) / "pretrain_standard" / "pretrained")
RESULTS_DIR = project.paths.results

# - Evaluation Parameters
MAX_TOTAL_STEPS = 10_000
MIN_EVAL_STEPS = 10
MAX_EVAL_STEPS = 500

# - 5 Predefined Random Rotations. From DMC.
RANDOM_ROTATIONS_5 = [
    [19, 339, 301],
    [196, 326, 225],
    [68, 100, 252],
    [256, 284, 218],
    [259, 193, 172],
]
STANDARD_ROTATIONS = get_cube_face_and_corner_views_rotations()


"""
Learning Module and Sensor Module Configs
----------------------------------------------------------------------
"""


def make_eval_lm_config(gsg_enabled: bool = True) -> dict[str, str]:
    """Create a learning module config for evaluation experiments.

    This effectively returns the default LM config used in benchmark experiments.
    Its parameters are written out completely here (it's easier to view them this way).
    Modify this function to allow for parameter overrides.

    Returns:
        A dictionary with two items:
          - "learning_module_class": The EvidenceGraphLM class.
          - "learning_module_args": A dictionary of arguments for the EvidenceGraphLM
            class.

    Raises:
        ValueError: If the agent_type is not "dist" or "surf".
    """
    return dict(
        learning_module_class=EvidenceGraphLM,
        learning_module_args=dict(
            # Specify graph matching thresholds and tolerances.
            max_match_distance=0.01,  # 1 cm
            tolerances={
                "patch": {
                    "hsv": np.array([0.1, 0.2, 0.2]),
                    "principal_curvatures_log": np.ones(2),
                }
            },
            feature_weights={
                "patch": {
                    "hsv": np.array([1.0, 0.5, 0.5]),
                }
            },
            # Update all hypotheses with evidence > 80% of max evidence.
            x_percent_threshold=20,
            evidence_threshold_config="80%",
            max_graph_size=0.3,  # 30cm
            num_model_voxels_per_dim=100,
            gsg_class=EvidenceGoalStateGenerator,
            gsg_args=dict(
                goal_tolerances=dict(
                    location=0.015,  # distance in meters
                ),  # Tolerance(s) when determining goal-state success
                elapsed_steps_factor=10,  # Factor that considers the number of elapsed
                # steps as a possible condition for initiating a hypothesis-testing goal
                # state; should be set to an integer reflecting a number of steps
                min_post_goal_success_steps=5,  # Number of necessary steps for a
                # hypothesis-driven goal-state to be considered successful
                # goal-state to be considered
                x_percent_scale_factor=0.75,  # Scale x-percent threshold to decide
                # when we should focus on pose rather than determining object ID; should
                # be bounded between 0:1.0; "mod" for modifier
                desired_object_distance=0.10,  # Distance from the
                # object to the agent that is considered "close enough" to the object
                enabled=gsg_enabled,
            ),
            hypotheses_updater_args=dict(
                # Using a smaller max_nneighbors (5 instead of 10) makes runtime faster,
                # but reduces performance a bit
                max_nneighbors=10
            ),
        ),
    )


def make_eval_patch_config() -> dict[str, str]:
    """Create a sensor module config for evaluation experiments.

    This effectively returns the default LM config used in benchmark experiments.
    Its parameters are written out completely here (it's easier to view them this way).
    Modify this function to allow for parameter overrides.

    Returns:
        A dictionary with two items:
          - "sensor_module_class": The FeatureChangeSM class.
          - "sensor_module_args": A dictionary of arguments for the SM class.
            The `sensor_module_id` item is always "patch".

    """

    return dict(
        sensor_module_class=FeatureChangeSM,
        sensor_module_args=dict(
            sensor_module_id="patch",
            features=[
                "pose_vectors",
                "pose_fully_defined",
                "on_object",
                "principal_curvatures_log",
                "hsv",
            ],
            delta_thresholds={
                "on_object": 0,
                "distance": 0.01,
            },
            surf_agent_sm=False,
            save_raw_obs=False,
        ),
    )


def make_view_finder_config(
    gsg_class: type | None = None,
    gsg_args: dict | None = None,
) -> dict[str, str]:
    """Create a sensor module config for a view-finder with an optional GSG."""
    if gsg_args is not None:
        gsg_args = copy.deepcopy(gsg_args)
    else:
        gsg_args = {}
    return dict(
        sensor_module_class=DetailedLoggingSM,
        sensor_module_args=dict(
            sensor_module_id="view_finder",
            save_raw_obs=False,
            gsg_class=gsg_class,
            gsg_args=gsg_args,
        ),
    )


"""
Motor System and Policy Configs
--------------------------------------------------------------------------------
"""


@dataclass
class MotorSystemConfig:
    motor_system_class: type
    motor_system_args: dict


def make_eval_motor_config(
    good_view_percentage: float = 0.5,
    desired_object_distance: float = 0.1,
    use_goal_state_driven_actions: bool = True,
) -> object:
    """Create a motor system config for evaluation experiments."""
    policy_class = InformedPolicy
    policy_args = InformedPolicyConfig(
        action_sampler_class=ConstantSampler,
        action_sampler_args=dict(
            actions=generate_action_list("distant_agent_no_translation"),
            rotation_degrees=5.0,
        ),
        agent_id="agent_id_0",
        file_name=None,
        good_view_percentage=good_view_percentage,
        desired_object_distance=desired_object_distance,
        use_goal_state_driven_actions=use_goal_state_driven_actions,
        switch_frequency=1.0,
        min_perc_on_obj=0.25,
    )
    motor_system_config = MotorSystemConfig(
        motor_system_class=MotorSystem,
        motor_system_args=dict(
            policy_class=policy_class,
            policy_args=policy_args,
        ),
    )
    return motor_system_config


"""
Environment, Dataset, and Dataloader Configs
--------------------------------------------------------------------------------
"""


@dataclass
class MountConfig:
    """Patch and view-finder mount config with custom defaults."""

    agent_id: str = "agent_id_0"
    sensor_ids: list[str] = field(default_factory=lambda: ["patch", "view_finder"])
    height: float = 0.0
    position: list[float] = field(default_factory=lambda: [0.0, 1.5, 0.5])
    resolutions: list[list[int]] = field(default_factory=lambda: [[64, 64], [128, 128]])
    positions: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    rotations: list[list[float]] = field(
        default_factory=lambda: [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    semantics: list[bool] = field(default_factory=lambda: [False, False])
    zooms: list[float] = field(default_factory=lambda: [10.0, 1.0])


@dataclass
class EnvInitArgs:
    agents: List[AgentConfig] = field(
        default_factory=lambda: [AgentConfig(MultiSensorAgent, MountConfig())]
    )
    objects: List[ObjectConfig] = field(default_factory=list)
    scene_id: int | None = field(default=None)
    seed: int = field(default=42)
    data_path: str = str(MONTY_DATA_DIR / "habitat" / "objects" / "ycb")


@dataclass
class DatasetArgs:
    env_init_func: Callable = field(default=HabitatEnvironment)
    env_init_args: dict = field(default_factory=lambda: EnvInitArgs())
    transform: Callable | Sequence[Callable] | None = None
    rng: Callable | None = None

    def __post_init__(self):
        agent_args = self.env_init_args.agents[0].agent_args
        self.transform = [
            MissingToMaxDepth(agent_id=agent_args.agent_id, max_depth=1),
            DepthTo3DLocations(
                agent_id=agent_args.agent_id,
                sensor_ids=agent_args.sensor_ids,
                resolutions=agent_args.resolutions,
                world_coord=True,
                zooms=agent_args.zooms,
                get_all_points=True,
                use_semantic_sensor=False,
            ),
        ]


def make_dataset_args(
    dataset: str = "ycb",
    agent_position: list[float] | None = None,
    view_finder_resolution: list[int] | None = None,
) -> DatasetArgs:
    if dataset == "ycb":
        data_path = MONTY_DATA_DIR / "habitat" / "objects" / "ycb"
    else:
        data_path = MONTY_DATA_DIR / dataset

    env_init_args = EnvInitArgs(data_path=str(data_path))
    dataset_args = DatasetArgs(env_init_args=env_init_args)

    if agent_position is not None:
        env_init_args.agents[0].agent_args.position = agent_position
    if view_finder_resolution is not None:
        env_init_args.agents[0].agent_args.resolutions[-1] = view_finder_resolution
    dataset_args.__post_init__()
    return dataset_args


def make_dataloader_args(
    names: list[str],
    positions: list[list[float]] | None = None,
    rotations: list[list[float]] | None = None,
) -> EnvironmentDataloaderPerObjectArgs:
    return EnvironmentDataloaderPerObjectArgs(
        object_names=names,
        object_init_sampler=PredefinedObjectInitializer(
            positions=positions,
            rotations=rotations,
        ),
    )


"""
Logging
--------------------------------------------------------------------------------
"""


@dataclass
class EvalLoggingConfig(ParallelEvidenceLMLoggingConfig):
    """Basic logging config with DMC-specific output directory and wandb group.

    This config also drops the reproduce episode handler which is included
    as a default handler in `ParallelEvidenceLMLoggingConfig`.
    """

    output_dir: str = str(project.paths.results)
    monty_handlers: List = field(
        default_factory=lambda: [
            BasicCSVStatsHandler,
        ]
    )

    # wandb_handlers: list = field(default_factory=list)
    wandb_group: str = "salience_showdown"
    monty_log_level: str = "BASIC"


@dataclass
class SelectiveLoggingConfig(EvalEvidenceLMLoggingConfig):
    """Logging config best used with `SelectiveEvidenceHandler`.

    Other than using a `SelectiveEvidenceHandler` by default, this config also
    has the `selective_handler_args` attribute which can be supplied to the
    `SelectiveEvidenceHandler`'s `__init__` method.
    """

    output_dir: str = str(project.paths.results)
    monty_handlers: List = field(
        default_factory=lambda: [
            BasicCSVStatsHandler,
            SelectiveHandler,
            ReproduceEpisodeHandler,
        ]
    )
    wandb_group: str = "gss"
    wandb_handlers: list = field(default_factory=list)
    monty_log_level: str = "DETAILED"
    selective_handler_args: dict = field(default_factory=dict)


def enable_telemetry(config: dict) -> None:
    """Set the config to save telemetry.

    Configures selective logger to save only SM data, and only a subset
    of its raw observations.

    Args:
        config: The config to set.
    """

    config["logging_config"] = SelectiveLoggingConfig(
        run_name=config["logging_config"].run_name,
        selective_handler_args={
            "exclude": ["LM_0"],
            "filters": [
                RawObservationsFilter(include=["rgba", "depth", "semantic_3d"])
            ],
        },
    )
    # sensor module detailed logging and telemetry
    sensor_module_configs = config["monty_config"].sensor_module_configs
    for sensor_module_config in sensor_module_configs.values():
        sm_args = sensor_module_config["sensor_module_args"]
        sm_args["save_raw_obs"] = True
        gsg_class = sm_args.get("gsg_class", None)
        if gsg_class:
            gsg_args = sm_args.get("gsg_args", {})
            gsg_args["save_telemetry"] = True
            sm_args["gsg_args"] = gsg_args

    # motor system telemetry
    motor_system_config = config["monty_config"].motor_system_config
    motor_system_config.motor_system_args["save_telemetry"] = True


"""
etc.
--------------------------------------------------------------------------------
"""


def set_view_finder_gsg(
    config: dict,
    gsg_class: type | None,
    gsg_args: dict | None = None,
) -> None:
    """Set the GSG for the view-finder sensor module."""
    if gsg_args is not None:
        gsg_args = copy.deepcopy(gsg_args)
    else:
        gsg_args = {}

    sm_config = config["monty_config"].sensor_module_configs["sensor_module_1"]
    sm_args = sm_config["sensor_module_args"]
    sm_args["gsg_class"] = gsg_class
    sm_args["gsg_args"] = gsg_args
