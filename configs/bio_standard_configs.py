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
from tbp.monty.frameworks.environments.ycb import DISTINCT_OBJECTS

from tbp.monty.frameworks.models.sm_goal_state_generation import (
    OnObjectGsg,
    OnObjectGsgBio,
    OnObjectGsgMinimumBarrier,
    OnObjectGsgSpectralResidual,
    OnObjectGsgUniform,
)

from .common import (
    BIO_PRETRAINED_MODEL,
    STANDARD_PRETRAINED_MODEL,
    make_dataloader_args,
    set_view_finder_gsg,
)
from .eval_configs import baseline
from .pretrain_configs import train_rotations_all

CONFIGS = {}

bio_baseline = copy.deepcopy(baseline)
bio_baseline["logging_config"].run_name = "bio_baseline"
bio_baseline["experiment_args"].model_name_or_path = BIO_PRETRAINED_MODEL
CONFIGS["bio_baseline"] = bio_baseline

bio_uniform_saliency = copy.deepcopy(bio_baseline)
bio_uniform_saliency["logging_config"].run_name = "bio_uniform_saliency"
set_view_finder_gsg(bio_uniform_saliency, OnObjectGsgUniform)
CONFIGS["bio_uniform_saliency"] = bio_uniform_saliency

bio_bio_saliency = copy.deepcopy(bio_baseline)
bio_bio_saliency["logging_config"].run_name = "bio_bio_saliency"
set_view_finder_gsg(bio_bio_saliency, OnObjectGsgBio)
CONFIGS["bio_bio_saliency"] = bio_bio_saliency

bio_spectral_residual = copy.deepcopy(bio_baseline)
bio_spectral_residual["logging_config"].run_name = "bio_spectral_residual"
set_view_finder_gsg(bio_spectral_residual, OnObjectGsgSpectralResidual)
CONFIGS["bio_spectral_residual"] = bio_spectral_residual

bio_minimum_barrier = copy.deepcopy(bio_baseline)
bio_minimum_barrier["logging_config"].run_name = "bio_minimum_barrier"
set_view_finder_gsg(bio_minimum_barrier, OnObjectGsgMinimumBarrier)
CONFIGS["bio_minimum_barrier"] = bio_minimum_barrier

standard_baseline = copy.deepcopy(baseline)
standard_baseline["logging_config"].run_name = "standard_baseline"
standard_baseline["experiment_args"].model_name_or_path = STANDARD_PRETRAINED_MODEL
CONFIGS["standard_baseline"] = standard_baseline

standard_uniform_saliency = copy.deepcopy(standard_baseline)
standard_uniform_saliency["logging_config"].run_name = "standard_uniform_saliency"
set_view_finder_gsg(standard_uniform_saliency, OnObjectGsgUniform)
CONFIGS["standard_uniform_saliency"] = standard_uniform_saliency

standard_bio_saliency = copy.deepcopy(standard_baseline)
standard_bio_saliency["logging_config"].run_name = "standard_bio_saliency"
set_view_finder_gsg(standard_bio_saliency, OnObjectGsgBio)
CONFIGS["standard_bio_saliency"] = standard_bio_saliency

standard_spectral_residual = copy.deepcopy(standard_baseline)
standard_spectral_residual["logging_config"].run_name = "standard_spectral_residual"
set_view_finder_gsg(standard_spectral_residual, OnObjectGsgSpectralResidual)
CONFIGS["standard_spectral_residual"] = standard_spectral_residual

standard_minimum_barrier = copy.deepcopy(standard_baseline)
standard_minimum_barrier["logging_config"].run_name = "standard_minimum_barrier"
set_view_finder_gsg(standard_minimum_barrier, OnObjectGsgMinimumBarrier)
CONFIGS["standard_minimum_barrier"] = standard_minimum_barrier

########################################################
# Testing on 6 Cube Face Rotations                     # 
########################################################

baseline_6rot = copy.deepcopy(baseline)
baseline_6rot["experiment_args"].n_eval_epochs = 6
baseline_6rot["eval_dataloader_args"] = make_dataloader_args(
    DISTINCT_OBJECTS,
    [[0.0, 1.5, 0.0]],
    train_rotations_all,
)

bio_baseline_6rot = copy.deepcopy(baseline_6rot)
bio_baseline_6rot["logging_config"].run_name = "bio_baseline_6rot"
bio_baseline_6rot["experiment_args"].model_name_or_path = BIO_PRETRAINED_MODEL
CONFIGS["bio_baseline_6rot"] = bio_baseline_6rot

bio_uniform_saliency_6rot = copy.deepcopy(bio_baseline_6rot)
bio_uniform_saliency_6rot["logging_config"].run_name = "bio_uniform_saliency_6rot"
set_view_finder_gsg(bio_uniform_saliency_6rot, OnObjectGsgUniform)
CONFIGS["bio_uniform_saliency_6rot"] = bio_uniform_saliency_6rot

bio_bio_saliency_6rot = copy.deepcopy(bio_baseline_6rot)
bio_bio_saliency_6rot["logging_config"].run_name = "bio_bio_saliency_6rot"
set_view_finder_gsg(bio_bio_saliency_6rot, OnObjectGsgBio)
CONFIGS["bio_bio_saliency_6rot"] = bio_bio_saliency_6rot

bio_spectral_residual_6rot = copy.deepcopy(bio_baseline_6rot)
bio_spectral_residual_6rot["logging_config"].run_name = "bio_spectral_residual_6rot"
set_view_finder_gsg(bio_spectral_residual_6rot, OnObjectGsgSpectralResidual)
CONFIGS["bio_spectral_residual_6rot"] = bio_spectral_residual_6rot

bio_minimum_barrier_6rot = copy.deepcopy(bio_baseline_6rot)
bio_minimum_barrier_6rot["logging_config"].run_name = "bio_minimum_barrier_6rot"
set_view_finder_gsg(bio_minimum_barrier_6rot, OnObjectGsgMinimumBarrier)
CONFIGS["bio_minimum_barrier_6rot"] = bio_minimum_barrier_6rot

standard_baseline_6rot = copy.deepcopy(baseline_6rot)
standard_baseline_6rot["logging_config"].run_name = "standard_baseline_6rot"
standard_baseline_6rot["experiment_args"].model_name_or_path = STANDARD_PRETRAINED_MODEL
CONFIGS["standard_baseline_6rot"] = standard_baseline_6rot

standard_uniform_saliency_6rot = copy.deepcopy(standard_baseline_6rot)
standard_uniform_saliency_6rot["logging_config"].run_name = "standard_uniform_saliency_6rot"
set_view_finder_gsg(standard_uniform_saliency_6rot, OnObjectGsgUniform)
CONFIGS["standard_uniform_saliency_6rot"] = standard_uniform_saliency_6rot

standard_bio_saliency_6rot = copy.deepcopy(standard_baseline_6rot)
standard_bio_saliency_6rot["logging_config"].run_name = "standard_bio_saliency_6rot"
set_view_finder_gsg(standard_bio_saliency_6rot, OnObjectGsgBio)
CONFIGS["standard_bio_saliency_6rot"] = standard_bio_saliency_6rot

standard_spectral_residual_6rot = copy.deepcopy(standard_baseline_6rot)
standard_spectral_residual_6rot["logging_config"].run_name = "standard_spectral_residual_6rot"
set_view_finder_gsg(standard_spectral_residual_6rot, OnObjectGsgSpectralResidual)
CONFIGS["standard_spectral_residual_6rot"] = standard_spectral_residual_6rot

standard_minimum_barrier_6rot = copy.deepcopy(standard_baseline_6rot)
standard_minimum_barrier_6rot["logging_config"].run_name = "standard_minimum_barrier_6rot"
set_view_finder_gsg(standard_minimum_barrier_6rot, OnObjectGsgMinimumBarrier)
CONFIGS["standard_minimum_barrier_6rot"] = standard_minimum_barrier_6rot

#####################################################
# Testing on 5 Random Rotations                     # 
#####################################################

RANDOM_ROTATIONS_5 = [
    [19, 339, 301],
    [196, 326, 225],
    [68, 100, 252],
    [256, 284, 218],
    [259, 193, 172],
] # copied from DMC


baseline_5randrot = copy.deepcopy(baseline)
baseline_5randrot["experiment_args"].n_eval_epochs = 5  
baseline_5randrot["eval_dataloader_args"] = make_dataloader_args(
    DISTINCT_OBJECTS,
    [[0.0, 1.5, 0.0]],
    RANDOM_ROTATIONS_5,
)

bio_baseline_5randrot = copy.deepcopy(baseline_5randrot)
bio_baseline_5randrot["logging_config"].run_name = "bio_baseline_5randrot"
bio_baseline_5randrot["experiment_args"].model_name_or_path = BIO_PRETRAINED_MODEL
CONFIGS["bio_baseline_5randrot"] = bio_baseline_5randrot

bio_uniform_saliency_5randrot = copy.deepcopy(bio_baseline_5randrot)
bio_uniform_saliency_5randrot["logging_config"].run_name = "bio_uniform_saliency_5randrot"
set_view_finder_gsg(bio_uniform_saliency_5randrot, OnObjectGsgUniform)
CONFIGS["bio_uniform_saliency_5randrot"] = bio_uniform_saliency_5randrot

bio_bio_saliency_5randrot = copy.deepcopy(bio_baseline_5randrot)
bio_bio_saliency_5randrot["logging_config"].run_name = "bio_bio_saliency_5randrot"
set_view_finder_gsg(bio_bio_saliency_5randrot, OnObjectGsgBio)
CONFIGS["bio_bio_saliency_5randrot"] = bio_bio_saliency_5randrot

bio_spectral_residual_5randrot = copy.deepcopy(bio_baseline_5randrot)
bio_spectral_residual_5randrot["logging_config"].run_name = "bio_spectral_residual_5randrot"
set_view_finder_gsg(bio_spectral_residual_5randrot, OnObjectGsgSpectralResidual)
CONFIGS["bio_spectral_residual_5randrot"] = bio_spectral_residual_5randrot

bio_minimum_barrier_5randrot = copy.deepcopy(bio_baseline_5randrot)
bio_minimum_barrier_5randrot["logging_config"].run_name = "bio_minimum_barrier_5randrot"
set_view_finder_gsg(bio_minimum_barrier_5randrot, OnObjectGsgMinimumBarrier)
CONFIGS["bio_minimum_barrier_5randrot"] = bio_minimum_barrier_5randrot

standard_baseline_5randrot = copy.deepcopy(baseline_5randrot)
standard_baseline_5randrot["logging_config"].run_name = "standard_baseline_5randrot"
standard_baseline_5randrot["experiment_args"].model_name_or_path = STANDARD_PRETRAINED_MODEL
CONFIGS["standard_baseline_5randrot"] = standard_baseline_5randrot

standard_uniform_saliency_5randrot = copy.deepcopy(standard_baseline_5randrot) 
standard_uniform_saliency_5randrot["logging_config"].run_name = "standard_uniform_saliency_5randrot"
set_view_finder_gsg(standard_uniform_saliency_5randrot, OnObjectGsgUniform)
CONFIGS["standard_uniform_saliency_5randrot"] = standard_uniform_saliency_5randrot

standard_bio_saliency_5randrot = copy.deepcopy(standard_baseline_5randrot)
standard_bio_saliency_5randrot["logging_config"].run_name = "standard_bio_saliency_5randrot"
set_view_finder_gsg(standard_bio_saliency_5randrot, OnObjectGsgBio)
CONFIGS["standard_bio_saliency_5randrot"] = standard_bio_saliency_5randrot

standard_spectral_residual_5randrot = copy.deepcopy(standard_baseline_5randrot)
standard_spectral_residual_5randrot["logging_config"].run_name = "standard_spectral_residual_5randrot"
set_view_finder_gsg(standard_spectral_residual_5randrot, OnObjectGsgSpectralResidual)
CONFIGS["standard_spectral_residual_5randrot"] = standard_spectral_residual_5randrot

standard_minimum_barrier_5randrot = copy.deepcopy(standard_baseline_5randrot)
standard_minimum_barrier_5randrot["logging_config"].run_name = "standard_minimum_barrier_5randrot"
set_view_finder_gsg(standard_minimum_barrier_5randrot, OnObjectGsgMinimumBarrier)
CONFIGS["standard_minimum_barrier_5randrot"] = standard_minimum_barrier_5randrot