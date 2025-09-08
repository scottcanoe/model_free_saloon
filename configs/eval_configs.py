# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Configs for Figure 3: Robust Sensorimotor Inference.

This module defines the following experiments:
 - `dist_agent_1lm`
 - `dist_agent_1lm_noise_all`
 - `dist_agent_1lm_randrot_14`
 - `dist_agent_1lm_randrot_14_noise_all`
 - `dist_agent_1lm_randrot_14_noise_all_color_clamped`

 Experiments use:
 - 77 objects
 - 14 rotations
 - Goal-state-driven/hypothesis-testing policy active
 - A single LM (no voting)

NOTE: random rotation variants use the random object initializer and 14 rotations.
`dist_agent_1lm_randrot_noise` which uses the 5 predefined "random" rotations
is defined in `fig5_rapid_inference_with_voting.py`.
"""
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
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)

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

CUSTOM_GSG_CLASSES = {}
try:
    from tbp.monty.frameworks.models import sm_goal_state_generation
    if hasattr(sm_goal_state_generation, "OnObjectGsg"):
        CUSTOM_GSG_CLASSES["OnObjectGsg"] = sm_goal_state_generation.OnObjectGsg
    if hasattr(sm_goal_state_generation, "TargetFindingGsg"):
        CUSTOM_GSG_CLASSES["TargetFindingGsg"] = sm_goal_state_generation.TargetFindingGsg
except ImportError:
    pass



MAX_TOTAL_STEPS = MAX_EVAL_STEPS = 20
MIN_EVAL_STEPS = 10



CONFIGS = {}




target_std = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=PRETRAINED_MODEL,
        max_total_steps=MAX_TOTAL_STEPS,
        max_eval_steps=MAX_EVAL_STEPS,
        n_eval_epochs=1,
    ),
    logging_config=EvalLoggingConfig(run_name="target_std"),
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
    dataset_args=make_dataset_args("targets"),
    eval_dataloader_class=ED.InformedEnvironmentDataLoader,
    eval_dataloader_args=make_dataloader_args(
        ["001_red_discs"],
        [[0.0, 1.5, 0.0]],
        [[0.0, 0.0, 0.0]],
    ),
)
CONFIGS["target_std"] = target_std


if "TargetFindingGsg" in CUSTOM_GSG_CLASSES:
    target_dev = copy.deepcopy(target_std)
    target_dev["logging_config"].run_name = "target_dev"
    set_view_finder_gsg(target_dev, CUSTOM_GSG_CLASSES["TargetFindingGsg"])
    CONFIGS["target_dev"] = target_dev


"""
--------------------------------------------------------------------------------
YCB
"""

ycb_std = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=EvalExperimentArgs(
        model_name_or_path=PRETRAINED_MODEL,
        max_total_steps=MAX_TOTAL_STEPS,
        max_eval_steps=MAX_EVAL_STEPS,
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
        ["mug"],
        [[0.0, 1.5, 0.0]],
        [[0, 0, 0]],
    ),
)
CONFIGS["ycb_std"] = ycb_std

if "OnObjectGsg" in CUSTOM_GSG_CLASSES:
    ycb_dev = copy.deepcopy(ycb_std)
    ycb_dev["logging_config"].run_name = "ycb_dev"
    set_view_finder_gsg(ycb_dev, CUSTOM_GSG_CLASSES["OnObjectGsg"])
    CONFIGS["ycb_dev"] = ycb_dev


"""
--------------------------------------------------------------------------------
"""

