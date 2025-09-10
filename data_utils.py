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

import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import orjson
import pandas as pd

from model_free_saloon import project

RESULTS_DIR = project.paths.results


def load_eval_stats(exp: os.PathLike) -> pd.DataFrame:
    """Load `eval_stats.csv` files.

    This function has 3 main purposes:
     - Load `eval_stats.csv` given just an experiment name since this function
       is aware of result paths.
     - Convert strings of arrays into arrays. For example, some columns contain
       arrays, but they're loaded as strings (e.g., "[1.34, 232.33, 123.44]").
     - Add some useful columns to the dataframe (`"episode"`, `"epoch"`).

    Args:
        exp (os.PathLike): Name of an experiment, a directory containing
          `eval_stats.csv`, or a complete path to an `.csv` file.

    Returns:
        pd.DataFrame

    Raises:
        FileNotFoundError: If `eval_stats.csv` is not found.
    """
    path = Path(exp).expanduser()

    if path.exists():
        # Case 1: Given a path to a csv file.
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(exp)
        # Case 2: Given a path to a directory containing eval_stats.csv.
        elif (path / "eval_stats.csv").exists():
            df = pd.read_csv(path / "eval_stats.csv")
        else:
            raise FileNotFoundError(f"No eval_stats.csv found for {exp}")
    else:
        # Given a run name. Look in DMC folder.
        df = pd.read_csv(RESULTS_DIR / path / "eval_stats.csv")

    # Remove redundant first column (which just has LM IDs)
    if df.columns[0] == "Unnamed: 0":
        df = df.iloc[:, 1:]

    # Collect basic info, like number of LMs, objects, number of episodes, etc.
    n_lms = len(np.unique(df["lm_id"]))
    object_names = np.unique(df["primary_target_object"])
    n_objects = len(object_names)

    # Add 'episode' column.
    assert len(df) % n_lms == 0  # sanity check
    n_episodes = int(len(df) / n_lms)
    df["episode"] = np.repeat(np.arange(n_episodes), n_lms)

    # Add 'epoch' column.
    rows_per_epoch = n_objects * n_lms
    assert len(df) % rows_per_epoch == 0  # sanity check
    n_epochs = int(len(df) / rows_per_epoch)
    df["epoch"] = np.repeat(np.arange(n_epochs), rows_per_epoch)

    # Decode array columns.
    def maybe_decode_array_string(s: Any, dtype: type) -> Any:
        """Converts a string-represented array to a numpy array.

        Args:
            s (Any): The object to possibly decode.
            dtype (type): The dtype of the array.

        Returns:
            Any: The decoded array.

        NOTE: if the input is not a string-representation of a list/tuple/array, it is
        returned as-is.
        """
        if not isinstance(s, str):
            return s

        # Quick out for empty strings.
        if s == "":
            return ""

        # Remove the outer brackets and parentheses. If it doesn't have
        # brackets or parentheses, it's not an array, so just return it as-is.
        if (s.startswith("[") and s.endswith("]")) or (
            s.startswith("(") and s.endswith(")")
        ):
            s = s[1:-1]
        else:
            return s

        # Split the string into a list of elements.
        if "," in s:
            # list and tuples are comma-separated
            lst = [elt.strip() for elt in s.split(",")]
        else:
            # numpy arrays are space-separated
            lst = s.split()

        # arrays of strings are a special case - can return arrays with dtype 'object',
        # and we also need to strip quotes from each item.
        if np.issubdtype(dtype, np.str_):
            lst = [elt.strip("'\"") for elt in lst]
            if dtype is str:
                return np.array(lst, dtype=object)
            else:
                return np.array(lst, dtype=dtype)

        # Must replace 'None' with np.nan for float arrays.
        if np.issubdtype(dtype, np.floating):
            lst = [np.nan if elt == "None" else dtype(elt) for elt in lst]
        return np.array(lst)

    float_array_cols = [
        "primary_target_position",
        "primary_target_rotation_euler",
        "most_likely_rotation",
        "detected_location",
        "detected_rotation",
        "location_rel_body",
        "detected_path",
        "most_likely_location",
        "primary_target_rotation_quat",
    ]
    column_order = list(df.columns)
    df["result"] = df["result"].replace(np.nan, "")
    df["result"] = df["result"].apply(maybe_decode_array_string, args=[str])
    for col_name in float_array_cols:
        df[col_name] = df[col_name].apply(maybe_decode_array_string, args=[float])
    df = df[column_order]
    return df


def eval_stats_equal(
    eval_stats_1: pd.DataFrame | os.PathLike,
    eval_stats_2: pd.DataFrame | os.PathLike,
) -> bool:
    """Test if two eval stats dataframes are the same.

    Args:
        eval_stats_1 (pd.DataFrame | os.PathLike): The first dataframe, experiment
            name, or path to a csv file.
        eval_stats_2 (pd.DataFrame | os.PathLike): The second dataframe, experiment
            name, or path to a csv file.

    Returns:
        bool: True if the dataframes are the same, False otherwise.

    """
    if not isinstance(eval_stats_1, pd.DataFrame):
        eval_stats_1 = load_eval_stats(eval_stats_1)
    if not isinstance(eval_stats_2, pd.DataFrame):
        eval_stats_2 = load_eval_stats(eval_stats_2)
    diffs = diff_dataframes(eval_stats_1, eval_stats_2, ignore=["time"])
    return len(diffs) == 0


def diff_dataframes(
    df_1: pd.DataFrame | os.PathLike,
    df_2: pd.DataFrame | os.PathLike,
    ignore: Sequence[str] | None = None,
) -> list[dict]:
    """Diff two dataframes.

    Args:
        df_1: The first dataframe to compare
        df_2: The second dataframe to compare
        ignore: Optional sequence of column names to ignore in the comparison

    Returns:
        list[dict]: A list of dictionaries containing differences between the dataframes.
                   Each dict has keys:
                   - 'column': Name of column with difference
                   - 'row': Row index of difference
                   - 'a': Value from first dataframe
                   - 'b': Value from second dataframe

    Raises:
        ValueError: If the dataframes have different columns or indexes.
    """
    df_1 = df_1 if isinstance(df_1, pd.DataFrame) else load_eval_stats(df_1)
    df_2 = df_2 if isinstance(df_2, pd.DataFrame) else load_eval_stats(df_2)

    if ignore:
        ignore = [ignore] if isinstance(ignore, str) else ignore
        df_1 = df_1.drop(columns=ignore)
        df_2 = df_2.drop(columns=ignore)

    if df_1.equals(df_2):
        return []

    # Check columns and indexes are equal
    if set(df_1.columns) != set(df_2.columns):
        raise ValueError("Dataframes must have the same columns.")

    if not df_1.index.equals(df_2.index):
        raise ValueError("Dataframes must have the same index.")

    diffs = []
    for col_name in df_1.columns:
        col_a, col_b = df_1[col_name], df_2[col_name]

        if col_a.equals(col_b):
            continue

        # Check values individually
        for i in range(len(col_a)):
            val_a, val_b = col_a[i], col_b[i]
            elt_diff = {
                "column": col_name,
                "row": i,
                "a": val_a,
                "b": val_b,
            }
            if type(val_a) is not type(val_b):
                diffs.append(elt_diff)
                continue
            if isinstance(val_a, (float, np.floating)):
                if not np.isclose(val_a, val_b, equal_nan=True):
                    diffs.append(elt_diff)
            elif isinstance(val_a, np.ndarray):
                if not np.allclose(val_a, val_b, equal_nan=True):
                    diffs.append(elt_diff)
            elif val_a != val_b:
                diffs.append(elt_diff)
            else:
                continue

    return diffs


class DetailedJSONStatsInterface:
    """Convenience interface to detailed JSON stats.

    This convenience interface to detailed JSON stats files. It's primarily useful for
    efficiently iterating over episodes.

    Example:
        >>> stats = DetailedJSONStatsInterface("detailed_stats.json")
        >>> last_episode_data = stats[-1]  # Get data for the last episode.
        >>> # Iterate over all episodes. Faster than loading individual episodes
        >>> # via random access.
        >>> for i, episode_data in enumerate(stats):
        ...     # Do something with episode data.
        ...     pass
    """

    def __init__(self, exp: os.PathLike):
        path = Path(exp).expanduser()
        if not path.exists():
            path = RESULTS_DIR / path / "detailed_run_stats.json"
        self._path = path
        self._index = None  # Just used to convert possibly negative indices

    @property
    def path(self) -> Path:
        return self._path

    def read_episode(self, episode: int) -> dict:
        self._check_initialized()
        assert np.isscalar(episode)
        episode = self._index[episode]
        with open(self._path, "r") as f:
            for i, line in enumerate(f):
                if i == episode:
                    return list(orjson.loads(line).values())[0]

    def _check_initialized(self):
        if self._index is not None:
            return
        length = 0
        with open(self._path, "r") as f:
            length = sum(1 for _ in f)
        self._index = np.arange(length)

    def __iter__(self):
        with open(self._path, "r") as f:
            for line in f:
                yield list(json.loads(line).values())[0]

    def __len__(self) -> int:
        self._check_initialized()
        return len(self._index)

    def __getitem__(self, episode: int) -> dict:
        """Get the stats for a given episode.

        Args:
            episode (int): The episode number.

        Returns:
            dict: The stats for the episode.
        """
        return self.read_episode(episode)


def extract_raw(stats: dict, sm_id: str, array_name: str) -> np.ndarray:
    raw_observations = stats[sm_id]["raw_observations"]
    return np.stack([np.array(dct[array_name]) for dct in raw_observations])


def extract_gsg_telemetry(stats: dict, sm_id: str) -> np.ndarray:
    telemetry = stats[sm_id]["gsg_telemetry"]
    return telemetry


def load_actions(exp: os.PathLike, episode: int) -> list[dict]:
    exp_dir = RESULTS_DIR / exp
    path = exp_dir / "reproduce_episode_data" / f"eval_episode_{episode}_actions.jsonl"
    lines = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            lines.append(data)
    return lines


def extract_sensor_pose(
    stats: dict, sm_id: str
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    sm_properties = stats[sm_id]["sm_properties"]
    rotations = [dct["sm_rotation"] for dct in sm_properties]
    positions = [dct["sm_location"] for dct in sm_properties]
    return rotations, positions
