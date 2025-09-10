# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import copy
import fnmatch
import os
from typing import Any, Callable, Iterable

import orjson
from tbp.monty.frameworks.loggers.monty_handlers import DetailedJSONHandler
from tbp.monty.frameworks.models.buffer import BufferEncoder
from tbp.monty.frameworks.utils.logging_utils import maybe_rename_existing_file


class SelectiveHandler(DetailedJSONHandler):
    """Detailed JSON Logger that applies filters to data before saving.

    Filters can be supplied via `selective_handler_args["filters"]`. As a convenience,
    a top-level include/exclude filter can be specified by using the `include` and
    `exclude` arguments.

    Example:
    ```
    # Basic: Only save view-finder data.
    selective_handler_args = {"include": ["SM_1"]}

    # Advanced: Only save data for sensor modules, and furthermore only save
    # "rgba" and "depth" data.
    selective_handler_args = {
        "include": ["SM_0", "SM_1"],
        "filters": [RawObservationsFilter(include=["rgba", "depth"])],
    }
    ```
    """

    filters: list[Callable[[dict], dict]]

    def __init__(self, selective_handler_args: dict | None = None):
        super().__init__()
        self.args = selective_handler_args or {}
        self.filters = []

        # set up top-level include/exclude filter
        include = self.args.get("include", [])
        exclude = self.args.get("exclude", [])
        if include or exclude:
            filt = IncludeExcludeFilter(include, exclude)
            self.filters.append(filt)

        # add other filters here
        if "filters" in self.args:
            self.filters.extend(self.args["filters"])

        # specify backend options
        self.write_options = (
            orjson.OPT_SERIALIZE_NUMPY
            | orjson.OPT_NON_STR_KEYS
            | orjson.OPT_APPEND_NEWLINE
        )
        self.write_default = BufferEncoder().default

    def report_episode(
        self,
        data: dict,
        output_dir: str,
        episode: int,
        mode: str = "train",
        **kwargs,
    ):
        """Report episode data.

        Args:
            data (dict): Data to report. Contains keys "BASIC" and "DETAILED".
            output_dir (str): Directory to save the report.
            episode (int): Episode number within the epoch.
            mode (str): Either "train" or "eval".
            **kwargs: Additional keyword arguments.

        Changed name to report episode since we are currently running with
        reporting and flushing exactly once per episode.
        """
        # Initialize buffer data, a dictionary containing everything to be saved.
        episode_total = kwargs[f"{mode}_episodes_to_total"][episode]
        basic_data = data["BASIC"][f"{mode}_stats"][episode]
        detailed_data = data["DETAILED"][episode_total]
        buffer_data = copy.deepcopy(basic_data)
        buffer_data.update(detailed_data)

        # Apply filters. Filters can modify the buffer data in place, but it still
        # needs to return the modified data.
        for filt in self.filters:
            buffer_data = filt(buffer_data)

        # Finally, write the data to disk.
        self.save(episode_total, buffer_data, output_dir)

    def save(self, episode_total: int, buffer_data: dict, output_dir: str) -> None:
        """Save data to a JSON file.

        Args:
            episode_total: Cumulative episode number (not within epoch).
            buffer_data: Data to save.
            output_dir: Directory to save the data to.
        """
        save_stats_path = os.path.join(output_dir, "detailed_run_stats.json")
        maybe_rename_existing_file(save_stats_path, ".json", self.report_count)
        data = {episode_total: buffer_data}
        with open(save_stats_path, "ab") as f:
            f.write(
                orjson.dumps(
                    data,
                    default=self.write_default,
                    option=self.write_options,
                )
            )

        self.report_count += 1


class IncludeExcludeFilter:
    """Simple include/exclude filter for strings. Supports glob patterns.

    Designed to operate like the include/exclude arguments for programs like rsync.
    The rules are:
      - If 'include' is given, ONLY items that match include patterns won't be
        filtered out. If `include` is empty or None, then all items will be included.
      - If 'exclude' is given, items that match exclude patterns will be filtered out.
        If `exclude` is empty or None, then no items will be excluded.
      - `include` is evaluated first, then `exclude`.
    """

    def __init__(
        self,
        include: str | Iterable[str] = (),
        exclude: str | Iterable[str] = (),
    ):
        """Initialize the filter.

        Args:
            include: Strings/patterns to include.
            exclude: Strings/patterns to exclude.
        """
        include = [include] if isinstance(include, str) else include
        exclude = [exclude] if isinstance(exclude, str) else exclude
        self._include = set(include)
        self._exclude = set(exclude)

    def match(self, text: str) -> bool:
        """Check if text should be included.

        Returns True if included, False if excluded.
        First matching rule wins.
        """
        if self._include:
            return any(fnmatch.fnmatch(text, pattern) for pattern in self._include)
        if self._exclude:
            return not any(fnmatch.fnmatch(text, pattern) for pattern in self._exclude)
        return True

    def __call__(self, dct: dict[str, Any]) -> dict[str, Any]:
        """Filter a dict."""
        return {k: v for k, v in dct.items() if self.match(k)}


class RawObservationsFilter:
    """Filter for including/excluding sensor module raw observation data.

    This filter applies an include/exclude filter to each sensor module's
    raw observations dictionaries. For example, raw observations typically have the
    keys "rgba", "depth", and "semantic_3d", "world_coords", etc. but we may only
    want to save the "rgba" data. This filter with `include=["rgba"]` will only save
    the "rgba" data.

    """

    def __init__(
        self,
        include: str | Iterable[str] = (),
        exclude: str | Iterable[str] = (),
    ):
        self._filter = IncludeExcludeFilter(include, exclude)

    def __call__(self, buffer_data: dict[str, Any]) -> dict[str, Any]:
        """Filter a dict."""
        sm_ids = [k for k in buffer_data.keys() if k.startswith("SM_")]
        for sm_id in sm_ids:
            sm_dict = buffer_data[sm_id]
            raw_observations = sm_dict["raw_observations"]
            for i, row in enumerate(raw_observations):
                raw_observations[i] = self._filter(row)
        return buffer_data
