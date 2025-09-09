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

from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    PatchAndViewMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EvalExperimentArgs,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import DISTINCT_OBJECTS
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.sm_goal_state_generation import OnObjectGsg

from .common import (
    PRETRAINED_MODEL,
    EvalLoggingConfig,
    make_dataloader_args,
    make_dataset_args,
    make_eval_lm_config,
    make_eval_motor_config,
    make_eval_patch_config,
    make_view_finder_config,
    set_view_finder_gsg,
)

MAX_TOTAL_STEPS = MAX_EVAL_STEPS = 20
MIN_EVAL_STEPS = 10


"""
--------------------------------------------------------------------------------
YCB
"""
CONFIGS = {}


baseline = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=PRETRAINED_MODEL,
        max_total_steps=1000,
        max_eval_steps=500,
        n_eval_epochs=1,
    ),
    logging_config=EvalLoggingConfig(run_name="baseline"),
    monty_config=PatchAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyArgs(min_eval_steps=MIN_EVAL_STEPS),
        sensor_module_configs=dict(
            sensor_module_0=make_eval_patch_config(),
            sensor_module_1=make_view_finder_config(),
        ),
        learning_module_configs=dict(
            learning_module_0=make_eval_lm_config(gsg_enabled=False),
        ),
        motor_system_config=make_eval_motor_config(),
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=make_dataset_args("ycb"),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=make_dataloader_args(
        DISTINCT_OBJECTS,
        [[0.0, 1.5, 0.0]],
        [[0, 0, 0]],
    ),
)
CONFIGS["baseline"] = baseline

baseline_dev = copy.deepcopy(baseline)
baseline_dev["logging_config"].run_name = "baseline_dev"
set_view_finder_gsg(baseline_dev, OnObjectGsg)
CONFIGS["baseline_dev"] = baseline_dev


ycb_std = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=PRETRAINED_MODEL,
        max_total_steps=50,
        max_eval_steps=50,
        n_eval_epochs=1,
    ),
    logging_config=EvalLoggingConfig(run_name="ycb_std"),
    monty_config=PatchAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        monty_args=MontyArgs(min_eval_steps=MIN_EVAL_STEPS),
        sensor_module_configs=dict(
            sensor_module_0=make_eval_patch_config(),
            sensor_module_1=make_view_finder_config(),
        ),
        learning_module_configs=dict(
            learning_module_0=make_eval_lm_config(gsg_enabled=False),
        ),
        motor_system_config=make_eval_motor_config(),
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=make_dataset_args("ycb"),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=make_dataloader_args(
        ["strawberry"],
        [[0.0, 1.5, 0.0]],
        [[0, 0, 0]],
    ),
)
CONFIGS["ycb_std"] = ycb_std

ycb_dev = copy.deepcopy(ycb_std)
ycb_dev["logging_config"].run_name = "ycb_dev"
set_view_finder_gsg(ycb_dev, OnObjectGsg)
CONFIGS["ycb_dev"] = ycb_dev


snapshots = copy.deepcopy(baseline)
snapshots["experiment_args"].max_total_steps = 1
snapshots["experiment_args"].max_eval_steps = 1
snapshots["logging_config"].run_name = "snapshots"
set_view_finder_gsg(baseline_dev, OnObjectGsg)
CONFIGS["snapshots"] = snapshots
