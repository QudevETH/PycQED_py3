"""Tools for creating combined plots from multiple measurements and experiments.

This module provides functions and classes to create grid-based visualizations
of measurement results, particularly useful for multi-qubit systems. It works
in conjunction with aggregation_plots_utils.py, which provides the underlying
utility functions for coordinate assignment, text handling, and figure saving.

Key features:
    - Plot data on customizable grids using arbitrary plotting functions
    - Automatic qubit coordinate assignment for grid layouts
    - Support for both single-qubit and two-qubit pair visualizations
    - Aggregation of calibration plots from multiple timestamps
    - Integration with QuantumExperiment results

Typical usage:
    ```python
    # Create grid plot for single-qubit data
    plot_on_qubit_grid(data_by_qubit, my_plot_function)

    # Aggregate calibration plots
    aggregator = CalibrationPlotAggregator.from_timestamps(['20230615'])
    aggregator.plot_on_qubit_grid()
    ```

Note:
    This module follows analysis_v3 design by a high degree. I.e., we defer
    execution, like plotting, as long as possible.
"""

import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib.image as mpimg
import re
import logging

import pycqed.analysis.analysis_toolbox as a_tools
import os
import fnmatch
from typing import Dict, Callable, Tuple, Any, Union, Optional, Sequence
import numpy as np
import pycqed.measurement.quantum_experiment as qe_mod
import pycqed.utilities.aggregation_plots_utils as aggr_u

logger = logging.getLogger(__name__)

# start with underscore to be 'first file shown in alphabetical order
COMBINED_PLOT_PREFIX = "_combined"


def plot_on_grid(
    data_by_index: Dict[Tuple[int, int], Any],
    plot_func: Callable,
    plot_func_kwargs: Optional[Dict] = None,
    fig_axes: Optional[Tuple[plt.Figure, np.ndarray]] = None,
    fig_kwargs: Optional[Dict] = None,
    labels: Optional[Dict[Tuple[int, int], str]] = None,
    label_as_title: bool = True,
    remove_empty_axes: bool = False,
    ax_properties: Optional[Dict] = None,
    save: bool = False,
    save_kwargs: Optional[dict] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots data on a grid using the specified plotting function.

    Args:
        data_by_index: Data mapped by grid coordinates.
        plot_func: Function to plot the experimental data on the given axis.
        plot_func_kwargs: Additional keyword arguments to pass to `plot_func`.
        fig_axes: Figure and axes to use. If None, new ones are created.
        fig_kwargs: Additional keyword arguments for figure creation.
        labels: Titles for each subplot, mapped by grid coordinates.
        label_as_title: If True, use the labels as titles for each subplot.
            If False, the labels are added as text within the subplot.
        remove_empty_axes: If True, remove any axes that do not contain data
            after plotting.
        ax_properties: Properties to apply to each axis.


    Returns:
        Tuple[plt.Figure, np.ndarray]: The figure and axes.
    """
    grid_shape, row_offset, column_offset = aggr_u._get_gridshape_and_offsets(
        list(data_by_index)
    )

    if fig_axes:
        fig, axes = fig_axes
    else:
        fig_kwargs = fig_kwargs or {}
        fig_kwargs["squeeze"] = False
        fig_kwargs.setdefault("sharex", True)
        fig_kwargs.setdefault("sharey", True)
        fig_kwargs.setdefault("figsize", (grid_shape[0] * 2.5, grid_shape[1] * 2))
        fig, axes = plt.subplots(grid_shape[0], grid_shape[1], **fig_kwargs)

    ax_properties = ax_properties or {}
    plot_func_kwargs = plot_func_kwargs or {}
    visited_axes = set()
    for (row, col), data in data_by_index.items():
        r, c = row + row_offset, col + column_offset
        plot_func(axes[r, c], data, **plot_func_kwargs)
        if labels and labels.get((row, col)):
            if label_as_title:
                axes[r, c].set_title(labels.get((row, col)))
            else:
                aggr_u.add_text(axes[r, c], labels.get((row, col)))
        axes[r, c].set(**ax_properties)
        visited_axes.add(axes[r, c])

    # remove unused axes
    for ax in axes.flatten():
        if ax not in visited_axes:
            ax.remove()
        elif remove_empty_axes and not ax.has_data():
            ax.remove()

    if save:
        sk = dict(save_kwargs) if save_kwargs else {}
        sk.setdefault("path", ".")
        sk.setdefault("fig_name", COMBINED_PLOT_PREFIX)
        sk.setdefault("extension", "png")
        aggr_u.savefig(fig, **sk)
    return fig, axes


def plot_on_qubit_grid(
    data_by_qubit: Dict[str, Any],
    plot_func: Callable,
    plot_func_kwargs: Optional[Dict] = None,
    qubit_to_coord: Optional[Callable] = None,
    fig_axes: Optional[Tuple[plt.Figure, np.ndarray]] = None,
    fig_kwargs: Optional[Dict] = None,
    qubit_labels: bool = True,
    remove_empty_axes: bool = False,
    ax_properties: Optional[Dict] = None,
    save: bool = False,
    save_kwargs: Optional[dict] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots data on a grid based on qubit coordinates.

    Args:
        data_by_qubit: Data mapped by qubit identifiers.
        plot_func: Function to plot the experimental data on the given axis.
        plot_func_kwargs: Additional keyword arguments to pass to `plot_func`.
        qubit_to_coord: Function to map qubits to grid coordinates.
        fig_axes: Figure and axes to use. If None, new ones are created.
        fig_kwargs: Additional keyword arguments for figure creation.
        qubit_labels: If True, use qubit identifiers as subplot titles.
        remove_empty_axes: If True, remove any axes that do not contain data.
        ax_properties: Properties to apply to each axis.

    Returns:
        Tuple[plt.Figure, np.ndarray]: The figure and axes used for
        plotting.
    """
    if qubit_to_coord is None:
        qubit_coordinates = aggr_u.assign_coordinates(list(data_by_qubit))
        qubit_to_coord = lambda q: qubit_coordinates[q]
    data_by_index_on_grid = {qubit_to_coord(q): d for q, d in data_by_qubit.items()}
    if qubit_labels:
        labels = {qubit_to_coord(q): q for q in data_by_qubit}
    else:
        labels = None
    return plot_on_grid(
        data_by_index=data_by_index_on_grid,
        plot_func=plot_func,
        plot_func_kwargs=plot_func_kwargs,
        fig_axes=fig_axes,
        fig_kwargs=fig_kwargs,
        labels=labels,
        ax_properties=ax_properties,
        remove_empty_axes=remove_empty_axes,
        save=save,
        save_kwargs=save_kwargs,
    )


def plot_on_pair_grid(
    data_by_pair: Dict[Tuple[str, str], Any],
    plot_func: Callable,
    plot_func_kwargs: Optional[Dict] = None,
    pair_to_coord: Optional[Callable] = None,
    fig_axes: Optional[Tuple[plt.Figure, np.ndarray]] = None,
    fig_kwargs: Optional[Dict] = None,
    pair_labels: bool = True,
    qubit_labels: bool = True,
    ax_properties: Optional[Dict] = None,
    save: bool = False,
    save_kwargs: Optional[dict] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots data on a grid based on qubit pairs and their coordinates.

    Args:
        data_by_pair: Data mapped by pairs of qubit identifiers.
        plot_func: Function to plot the data on the given axis.
        pair_to_coord: Function to map qubit pairs to grid coordinates.
        fig_axes: Figure and axes to use. If None, new ones are created.
        fig_kwargs: Additional keyword arguments for figure creation.
        labels: Titles for each subplot, mapped by qubit pairs.
        ax_properties: Properties to apply to each axis.

    Returns:
        Tuple[plt.Figure, np.ndarray]: The figure and axes used for
        plotting.
    """
    if pair_to_coord is None:
        pair_coordinates = aggr_u.assign_coordinates(list(data_by_pair))
        pair_to_coord = lambda q1, q2: pair_coordinates[q1, q2]
        # in this case we just have the minimal case of coordinates for pairs,
        # not for qubits, so, deactivate qubit labels.
        qubit_labels = False
    data_by_index_on_grid = {
        pair_to_coord(q1, q2): d for (q1, q2), d in data_by_pair.items()
    }
    plot_func_kwargs = plot_func_kwargs or {}
    _, row_offset, col_offset = aggr_u._get_gridshape_and_offsets(
        list(data_by_index_on_grid)
    )
    if qubit_labels:

        def plot_function_wrapper(ax, data, **kwargs):
            subplot_spec = ax.get_subplotspec()
            row, col = subplot_spec.rowspan.start, subplot_spec.colspan.start
            qubit_labels = kwargs.pop("qubit_labels", {})
            if (row - row_offset, col - col_offset) in qubit_labels:
                aggr_u.add_text(ax, qubit_labels[(row - row_offset, col - col_offset)])
                ax.axis("off")
            else:
                plot_func(ax, data, **kwargs)

        unique_qubits = set()
        for q1, q2 in data_by_pair:
            unique_qubits.add(q1)
            unique_qubits.add(q2)
        data_by_index_on_grid.update({pair_to_coord(q, q): q for q in unique_qubits})
        plot_func_kwargs.update(
            dict(qubit_labels={pair_to_coord(q, q): q for q in unique_qubits})
        )
        _plot_func = plot_function_wrapper
    else:
        _plot_func = plot_func
    if pair_labels:
        labels = {
            pair_to_coord(q1, q2): "_".join((q1, q2))
            for (q1, q2), d in data_by_pair.items()
        }
    else:
        labels = None
    return plot_on_grid(
        data_by_index=data_by_index_on_grid,
        plot_func=_plot_func,
        plot_func_kwargs=plot_func_kwargs,
        fig_axes=fig_axes,
        fig_kwargs=fig_kwargs,
        labels=labels,
        ax_properties=ax_properties,
        save=save,
        save_kwargs=save_kwargs,
    )


def get_qubit_grid(
    qubits: list,
    qubit_to_coord: Optional[Callable] = None,
    fig_axes: Optional[Tuple[plt.Figure, np.ndarray]] = None,
    fig_kwargs: Optional[Dict] = None,
    qubit_labels: bool = True,
    remove_empty_axes: bool = False,
    ax_properties: Optional[Dict] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots data on a grid based on qubit coordinates.

    Args:
        qubits: list of qubits or qubit names
        qubit_to_coord: Function to map qubits to grid coordinates.
        fig_axes: Figure and axes to use. If None, new ones are created.
        fig_kwargs: Additional keyword arguments for figure creation.
        qubit_labels: If True, use qubit identifiers as subplot titles.
        remove_empty_axes: If True, remove any axes that do not contain data.
        ax_properties: Properties to apply to each axis.

    Returns:
        Tuple[plt.Figure, np.ndarray]: The figure and axes used for
        plotting.
    """
    if qubit_to_coord is None:
        qubit_coordinates = aggr_u.assign_coordinates(qubits)
        qubit_to_coord = lambda q: qubit_coordinates[q]
    data_by_index_on_grid = {qubit_to_coord(q): None for q in qubits}
    if qubit_labels:
        labels = {qubit_to_coord(q): q for q in qubits}
    else:
        labels = None
    return plot_on_grid(
        data_by_index=data_by_index_on_grid,
        plot_func=lambda ax, data: None,
        fig_axes=fig_axes,
        fig_kwargs=fig_kwargs,
        labels=labels,
        ax_properties=ax_properties,
        remove_empty_axes=remove_empty_axes,
    )

# Module functions (plotting)

def fig_plot_func(ax, fig, dpi=500):
    """Plots a matplotlib figure onto a given axis.

    Args:
        ax (matplotlib.axes.Axes): The target axis to plot the figure on.
        fig (matplotlib.figure.Figure): The source figure to be plotted.
        dpi (int, optional): The resolution in dots per inch. Defaults to 500.

    Returns:
        None

    Note:
        The function converts the figure to a PNG image in memory using BytesIO,
        then displays it on the target axis with the image display turned off.
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img = mpimg.imread(buf)
    ax.imshow(img)
    ax.axis("off")

def fig_from_measurement_plot_func(
    ax, fig_info: dict, fig_name="", extension="png", ignore_missing: bool = True
):
    """Searches for a figure file matching `fig_name` in the folder associated
    with `timestamp`, then plots it onto the provided `ax`.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to plot the figure.
        fig_info (dict):
            required key: 'timestamp': The timestamp used to locate the folder.
            optional key: 'fig_name': the ax-specific (e.g. qubit specific)
                figure name to search for. For instance to use when a single
                timestamp includes the measurements of many qubits.
        fig_name (str): The name or partial name of the figure to search for.
            If there are several matches, the first one is taken.
        extension (str, optional): The file extension to search for (e.g., 'png', 'jpeg').
         Default is 'png'.
        ignore_missing (bool): whether to skip the plotting if no match is found.

    Raises:
        FileNotFoundError: If no matching figure is found in the directory.
    """
    # Get the folder path associated with the timestamp
    timestamp, fig_name = fig_info["timestamp"], fig_info.get("fig_name", fig_name)
    folder = a_tools.get_folder(timestamp)

    # Search for all files in the folder with the specified extension
    pattern = f"*.{extension}"
    matching_files = [
        f for f in os.listdir(folder) if fnmatch.fnmatch(f, pattern) and fig_name in f
    ]

    if not matching_files:
        if not ignore_missing:
            raise FileNotFoundError(
                f"No files found with the name containing '{fig_name}' and "
                f"extension '{extension}' in {folder}."
            )
        else:
            return
    # Select the first matching file (or you could implement a selection mechanism)
    file_path = os.path.join(folder, matching_files[0])

    # Load the image into a BytesIO object and display it on the provided axis
    with open(file_path, "rb") as img_file:
        img_data = BytesIO(img_file.read())
        img = mpimg.imread(img_data)
        ax.imshow(img)
        ax.axis("off")


class CalibrationPlotAggregator:
    DEFAULT_CALIBRATION_PLOT_NAMES = {
        "Rabi": "Rabi_{qbn}",
        "Ramsey": "Ramsey_{qbn}",
        "ReparkingRamsey": "reparking_{qbn}",
        "T1": "T1_{qbn}",
        "SSRO": "{qbn}_gmm_classifier_data",
        "MultiStateResonatorSpectroscopy": "s21_distance_{qbn}",
        "continuous_spec": "Source frequency distance",
    }

    @classmethod
    def from_timestamps(
        cls, timestamps: Optional[Sequence] = None, qb_names: Optional[list] = None
    ):
        """Creates a CalibrationPlotAggregator instance from a list of timestamps.
        The function scans the directories associated with each timestamp for
        qubit-related files and associates each qubit with its respective
        calibration plot.

        Args:
            timestamps: A sequence of timestamps to search for qubit plots.
            qb_names: A list of qubit names to look for. If not provided,
                      the names will be discovered from the files.

        Returns:
            CalibrationPlotAggregator: An instance of the class with the
                                        associated figure information.
        """

        if not timestamps:
            timestamps = a_tools.get_last_n_timestamps(n=1)

        fig_dict = {}
        for t in timestamps:
            if qb_names:
                qb_names_timestamp = qb_names
            else:
                qb_names_timestamp = cls.discover_qubit_names(
                    os.listdir(a_tools.get_folder(t))
                )
            for qbn in qb_names_timestamp:
                if qbn in fig_dict:
                    logger.warning(
                        f"{qbn} already in fig_dict with timestamp: "
                        f"{fig_dict[qbn]}; will be overwritten by latest "
                        f"figure with timestamp {t}"
                    )
                fig_dict[qbn] = dict(timestamp=t, fig_name="")

                # Infer the calibration type and corresponding figure name.
                for fn in os.listdir(a_tools.get_folder(t)):
                    for cal_name in cls.DEFAULT_CALIBRATION_PLOT_NAMES:
                        # find a figure that matched the calibration name which
                        # is not a combined plot (those might also have the
                        # cal name into their name)
                        if cal_name in fn and COMBINED_PLOT_PREFIX not in fn:
                            fig_dict[qbn].update(
                                fig_name=aggr_u.safe_format_str_with_keys(
                                    cls.DEFAULT_CALIBRATION_PLOT_NAMES[cal_name],
                                    qbn=qbn,
                                )
                            )
                            break

        return cls(fig_info=fig_dict)

    @classmethod
    def from_quantum_experiments(
        cls,
        quantum_experiments: Union[Sequence, "qe_mod.QuantumExperiment"],
        qb_names: Optional[list] = None,
    ):
        """Creates a CalibrationPlotAggregator instance from quantum experiments.

        Args:
            quantum_experiments: A sequence of quantum experiments or a single
                                 quantum experiment to derive timestamps from.
            qb_names: A list of qubit names to look for. If not provided,
                      the names will be discovered from the files.

        Returns:
            CalibrationPlotAggregator: An instance of the class with the
                                        associated figure information.
        """
        if isinstance(quantum_experiments, qe_mod.QuantumExperiment):
            quantum_experiments = [quantum_experiments]

        return cls.from_timestamps(
            timestamps=[qe.timestamp for qe in quantum_experiments], qb_names=qb_names
        )

    @staticmethod
    def discover_qubit_names(file_names: list[str]) -> set[str]:
        """Finds all occurrences of 'qbX' in a list of strings, where X is one or
        more digits. Each string's results are returned as a set of unique
        qubit names.

        Args:
            file_names: List of strings to search for qubit names.

        Returns:
            set: A set of qubit names found in the file names.
        """
        qubit_pattern = r"qb\d+"  # Regex to match 'qb' followed by one or more digits
        all_matches = set()

        for string in file_names:
            matches = re.findall(qubit_pattern, string)
            all_matches.update(matches)

        return all_matches

    def __init__(self, fig_info: dict):
        """
        Initializes the CalibrationPlotAggregator with figure information.

        Args:
            fig_info: A dictionary containing information about the figures,
                      keyed by qubit name.
        """
        self.fig_info = fig_info

    def plot_on_qubit_grid(
        self,
        fig_name: Optional[str] = None,
        qb_names: Optional[list] = None,
        save: bool = False,
        save_kwargs: Optional[dict] = None,
        **plot_kwargs,
    ):
        """
        Plots calibration figures on a grid based on the qubit names and
        figure information. Uses fig_from_measurement_plot_func

        Args:
            fig_name: Specific figure name to plot. If None, all figures are
                      plotted.
            qb_names: A list of qubit names to include in the plot. If None,
                      all qubits in fig_info are plotted.
            save: Whether to save the plot.
            save_kwargs: parameters passed to savefig. path, fig_name, extension, etc.
            **plot_kwargs: Additional keyword arguments passed to plot_on_qubit_grid

        Returns:
            The result of the plot_on_qubit_grid function.
        """
        if qb_names:
            fig_info = {q: i for q, i in self.fig_info.items() if q in qb_names}
        else:
            fig_info = dict(self.fig_info)
        if fig_name:
            for qbn in fig_info:
                fig_info[qbn]["fig_name"] = aggr_u.safe_format_str_with_keys(
                    fig_name, qbn=qbn
                )
        last_entry = list(fig_info.values())[-1]
        save_kwargs = save_kwargs or {}
        save_kwargs.setdefault("path", a_tools.get_folder(last_entry["timestamp"]))
        combined_fig_name = f'{COMBINED_PLOT_PREFIX}_{last_entry["fig_name"]}'
        # remove qubit names (figure-specific) from combined plot name
        for qbn in self.discover_qubit_names([combined_fig_name]):
            combined_fig_name = combined_fig_name.replace(qbn, "")
        save_kwargs.setdefault("fig_name", combined_fig_name)
        return plot_on_qubit_grid(
            fig_info,
            fig_from_measurement_plot_func,
            save=save,
            save_kwargs=save_kwargs,
            **plot_kwargs,
        )
