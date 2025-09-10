# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import shutil
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).parent

paths = OmegaConf.create(
    {
        "root": ROOT,
        "configs": ROOT / "configs",
        "data": ROOT / "data",
        "results": ROOT / "results",
    }
)


def load_config(experiment_name: str) -> dict:
    from configs import CONFIGS

    return CONFIGS[experiment_name]


def run_config(
    config: dict,
    name: str | None = None,
    clean: bool = True,
) -> None:
    import os

    from tbp.monty.frameworks.run import main
    from tbp.monty.frameworks.run_env import setup_env

    setup_env()
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"

    output_dir = Path(config["logging_config"].output_dir)
    run_name = config["logging_config"].run_name

    exp_dir = output_dir / run_name
    if clean and exp_dir.exists():
        shutil.rmtree(exp_dir)

    name = name or run_name
    main(all_configs={name: config}, experiments=[name])
