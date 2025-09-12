# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Configs..."""

from __future__ import annotations

import copy
import os
from dataclasses import asdict

import numpy as np
from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    MontyFeatureGraphArgs,
    MotorSystemConfigCurvatureInformedSurface,
    MotorSystemConfigNaiveScanSpiral,
    PatchAndViewMontyConfig,
    PretrainLoggingConfig,
    SurfaceAndViewMontyConfig,
    get_cube_face_and_corner_views_rotations,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    ExperimentArgs,
    PredefinedObjectInitializer,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.config_utils.policy_setup_utils import (
    make_naive_scan_policy_config,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import (
    DISTINCT_OBJECTS,
    SHUFFLED_YCB_OBJECTS,
    SIMILAR_OBJECTS,
)
from tbp.monty.frameworks.experiments import (
    MontyObjectRecognitionExperiment,
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.models.displacement_matching import DisplacementGraphLM
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.motor_policies import NaiveScanPolicy
from tbp.monty.frameworks.models.sensor_modules import (
    HabitatDistantPatchSM,
)
from tbp.monty.frameworks.models.sm_goal_state_generation import (
    OnObjectGsg,
    OnObjectGsgBio,
    OnObjectGsgMinimumBarrier,
    OnObjectGsgRobustBackground,
    OnObjectGsgSpectralResidual,
    OnObjectGsgUniform,
    OnObjectGsgIttiKoch,
)
from tbp.monty.simulators.habitat.configs import (
    FiveLMMountHabitatDatasetArgs,
    PatchViewFinderMountHabitatDatasetArgs,
    SurfaceViewFinderMontyWorldMountHabitatDatasetArgs,
    SurfaceViewFinderMountHabitatDatasetArgs,
)

from .common import (
    PRETRAINED_MODEL,
    RESULTS_DIR,
    EvalLoggingConfig,
    make_dataloader_args,
    make_dataset_args,
    make_eval_lm_config,
    make_eval_motor_config,
    make_eval_patch_config,
    make_view_finder_config,
    set_view_finder_gsg,
)

# FOR SUPERVISED PRETRAINING: 14 unique rotations that give good views of the object.
train_rotations_all = get_cube_face_and_corner_views_rotations()
MAX_TOTAL_STEPS = MAX_EVAL_STEPS = 20
MIN_EVAL_STEPS = 10

train_rotations_all = train_rotations_all[:6]

patch_config_no_fc = dict(
    sensor_module_class=HabitatDistantPatchSM,
    sensor_module_args=dict(
        sensor_module_id="patch",
        # TODO: would be nicer to just use lm.tolerances.keys() here
        # but not sure how to easily do this.
        features=[
            "pose_vectors",
            "pose_fully_defined",
            "on_object",
            "principal_curvatures_log",
            "hsv",
        ],
        save_raw_obs=False,
    ),
)


"""
--------------------------------------------------------------------------------
YCB
"""
CONFIGS = {}


pretrain_bio = dict(
    experiment_class=MontySupervisedObjectPretrainingExperiment,
    experiment_args=ExperimentArgs(
        do_eval=False,
        n_train_epochs=len(train_rotations_all),
        # model_name_or_path=PRETRAINED_MODEL,
    ),
    logging_config=PretrainLoggingConfig(
        output_dir=str(RESULTS_DIR),
        run_name="pretrain_bio",
    ),
    monty_config=PatchAndViewMontyConfig(
        monty_args=MontyArgs(num_exploratory_steps=500),
        sensor_module_configs=dict(
            sensor_module_0=make_eval_patch_config(),
            sensor_module_1=make_view_finder_config(),
        ),
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=DisplacementGraphLM,
                learning_module_args=dict(
                    k=10,
                    match_attribute="displacement",
                    tolerance=np.ones(3) * 0.0001,
                    graph_delta_thresholds=dict(
                        patch=dict(
                            distance=0.001,
                            # Only first pose vector (surface normal) is currently used
                            pose_vectors=[np.pi / 8, np.pi * 2, np.pi * 2],
                            principal_curvatures_log=[1, 1],
                            hsv=[0.1, 1, 1],
                        )
                    ),
                ),
            )
        ),
        motor_system_config=make_eval_motor_config(),
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=make_dataset_args("ycb"),
    train_dataloader_class=ED.InformedEnvironmentDataLoader,
    train_dataloader_args=make_dataloader_args(
        DISTINCT_OBJECTS,
        positions=[[0.0, 1.5, 0.0]],
        rotations=train_rotations_all,
    ),
)

set_view_finder_gsg(pretrain_bio, OnObjectGsgBio)
CONFIGS["pretrain_bio"] = pretrain_bio

pretrain_standard = copy.deepcopy(pretrain_bio)
pretrain_standard["logging_config"].run_name = "pretrain_standard"
pretrain_standard["monty_config"].sensor_module_configs["sensor_module_0"] = (
    patch_config_no_fc
)
set_view_finder_gsg(pretrain_standard, None)
pretrain_standard["monty_config"].motor_system_config=MotorSystemConfigNaiveScanSpiral(
            motor_system_args=dict(
                policy_class=NaiveScanPolicy,
                policy_args=make_naive_scan_policy_config(step_size=5),
            )
)
CONFIGS["pretrain_standard"] = pretrain_standard

########################################################
# Pretraining on Different graph_delta_th              # 
########################################################

pretrain_bio_graph_delta_th_2mm = copy.deepcopy(pretrain_bio)
pretrain_bio_graph_delta_th_2mm["logging_config"].run_name = "pretrain_bio_graph_delta_th_2mm"
pretrain_bio_graph_delta_th_2mm["monty_config"].learning_module_configs["learning_module_0"]["learning_module_args"]["graph_delta_thresholds"]["patch"]["distance"] = 0.002
CONFIGS["pretrain_bio_graph_delta_th_2mm"] = pretrain_bio_graph_delta_th_2mm

pretrain_bio_graph_delta_th_5mm = copy.deepcopy(pretrain_bio)
pretrain_bio_graph_delta_th_5mm["logging_config"].run_name = "pretrain_bio_graph_delta_th_5mm"
pretrain_bio_graph_delta_th_5mm["monty_config"].learning_module_configs["learning_module_0"]["learning_module_args"]["graph_delta_thresholds"]["patch"]["distance"] = 0.005
CONFIGS["pretrain_bio_graph_delta_th_5mm"] = pretrain_bio_graph_delta_th_5mm

pretrain_bio_graph_delta_th_10mm = copy.deepcopy(pretrain_bio)
pretrain_bio_graph_delta_th_10mm["logging_config"].run_name = "pretrain_bio_graph_delta_th_10mm"
pretrain_bio_graph_delta_th_10mm["monty_config"].learning_module_configs["learning_module_0"]["learning_module_args"]["graph_delta_thresholds"]["patch"]["distance"] = 0.01
CONFIGS["pretrain_bio_graph_delta_th_10mm"] = pretrain_bio_graph_delta_th_10mm


########################################################
# Pretraining on Minimum Barrier and Itti-Koch          # 
########################################################

pretrain_minimum_barrier = copy.deepcopy(pretrain_bio)
pretrain_minimum_barrier["logging_config"].run_name = "pretrain_minimum_barrier"
set_view_finder_gsg(pretrain_minimum_barrier, OnObjectGsgMinimumBarrier)
CONFIGS["pretrain_minimum_barrier"] = pretrain_minimum_barrier

pretrain_itti_koch = copy.deepcopy(pretrain_bio)
pretrain_itti_koch["logging_config"].run_name = "pretrain_itti_koch"
set_view_finder_gsg(pretrain_itti_koch, OnObjectGsgIttiKoch)
CONFIGS["pretrain_itti_koch"] = pretrain_itti_koch