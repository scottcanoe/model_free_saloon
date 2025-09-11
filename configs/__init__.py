# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from .eval_configs import CONFIGS as EVAL_CONFIGS
from .pretrain_configs import CONFIGS as PRETRAIN_CONFIGS
from .bio_standard_configs import CONFIGS as BIO_STANDARD_CONFIGS

CONFIGS = {}
CONFIGS.update(EVAL_CONFIGS)
CONFIGS.update(PRETRAIN_CONFIGS)
CONFIGS.update(BIO_STANDARD_CONFIGS)