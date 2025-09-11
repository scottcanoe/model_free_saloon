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
    set_view_finder_gsg,
)
from .eval_configs import baseline

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