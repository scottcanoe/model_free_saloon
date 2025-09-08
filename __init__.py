# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_NAME = "gss"
PROJECT_ROOT = Path(__file__).parent

MONTY_MODELS = Path.home() / "tbp/results/monty/pretrained_models/pretrained_ycb_v10"

@dataclass
class ProjectPaths:
    root: Path = PROJECT_ROOT
    configs: Path = PROJECT_ROOT / "configs"
    pretrained_models: Path = MONTY_MODELS
    data: Path = PROJECT_ROOT / "data"
    results: Path = PROJECT_ROOT / "results"


class Project:
    """
    A project is a collection of related experiments.
    """
    
    def __init__(self):
        self.name = PROJECT_NAME
        self.paths = OmegaConf.structured(ProjectPaths())


project = Project()



