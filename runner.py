# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.animation import FuncAnimation, PillowWriter

from configs.common import enable_telemetry
from data_utils import (
    DetailedJSONStatsInterface,
    extract_raw,
)
from project import load_config, run_config


def make_gifs(exp_dir: Path):
    episode = 0
    sm_id = "SM_1"

    # --- Get episode data
    exp_dir = Path(exp_dir)
    all_stats = DetailedJSONStatsInterface(exp_dir / "detailed_run_stats.json")
    stats = all_stats[episode]

    n_rows, n_cols = 2, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 6))
    infos = np.zeros((n_rows, n_cols), dtype=object)
    for i in range(n_rows):
        for j in range(n_cols):
            infos[i, j] = {}

    infos[0, 0]["sm_id"] = "SM_1"
    infos[0, 0]["kind"] = "rgba"

    infos[0, 1]["sm_id"] = "SM_1"
    infos[0, 1]["kind"] = "depth"

    infos[1, 0]["sm_id"] = "SM_0"
    infos[1, 0]["kind"] = "rgba"

    infos[1, 1]["sm_id"] = "SM_0"
    infos[1, 1]["kind"] = "depth"

    for i in range(n_rows):
        for j in range(n_cols):
            nfo = infos[i, j]
            sm_id, kind = nfo["sm_id"], nfo["kind"]

            frames = extract_raw(stats, sm_id, kind)
            if kind == "rgba":
                to_rgba = lambda arr: arr
            else:
                frames[frames >= 1.0] = np.nan
                frames[frames <= 0.0] = np.nan
                vmin = np.nanmin(frames) * 0.9
                vmax = np.nanmax(frames) * 1.1
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
                smap = plt.cm.ScalarMappable(norm=norm, cmap="inferno")
                to_rgba = lambda arr: smap.to_rgba(arr)

            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            im = ax.imshow(np.zeros_like(to_rgba(frames[0])))
            nfo["ax"] = axes[i, j]
            nfo["im"] = im
            nfo["frames"] = frames
            nfo["to_rgba"] = to_rgba
            nfo["title"] = f"{nfo['sm_id']} {nfo['kind']}"

            # Add a colorbar for depth plots
            if kind == "depth":
                cbar = fig.colorbar(smap, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("depth (m)")

    def update(frame):
        for i in range(n_rows):
            for j in range(n_cols):
                nfo = infos[i, j]
                ax, im, title = nfo["ax"], nfo["im"], nfo["title"]
                arr = nfo["frames"][frame]
                to_rgba = nfo["to_rgba"]
                rgba = to_rgba(arr)
                im.set_data(rgba)
                ax.set_title(title)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / 5)
    anim.save(exp_dir / "animation.gif", writer=PillowWriter(fps=5), dpi=200)


if __name__ == "__main__":
    experiment = "ycb_bio"

    config = load_config(experiment)
    enable_telemetry(config)
    run_config(config, clean=True)

    output_dir = Path(config["logging_config"].output_dir)
    run_name = config["logging_config"].run_name
    exp_dir = output_dir / run_name
    make_gifs(exp_dir)
