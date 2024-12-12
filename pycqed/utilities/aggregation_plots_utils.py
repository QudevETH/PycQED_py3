"""Utility functions for creating grid-based plots and handling coordinates.

This module provides helper functions used by aggregation_plots.py to create
grid-based visualizations. It includes utilities for coordinate assignment,
grid shape calculation, text handling, and figure saving.

Key functions:
    assign_coordinates: Maps labels to 2D grid coordinates
    _get_gridshape_and_offsets: Calculates grid dimensions and offset values
    add_text: Adds centered text labels to plot axes
    savefig: Saves figures with timestamp
    safe_format_str_with_keys: Safely formats strings with variable substitution
"""

import math
import pathlib

import numpy as np

import pycqed.analysis.analysis_toolbox as a_tools


def assign_coordinates(labels, shape=None, order="row_first"):
    """
    Assigns 2D coordinates (xi, yi) to labels in either row-first or column-first order.

    Args:
        labels (list): List of labels (e.g., qubit names, pair names) to assign coordinates to.
        shape (tuple, optional): The shape (rows, columns) of the grid. If not provided,
                                 the function will automatically infer the minimum shape
                                 required to fit all labels.
        order (str): 'row_first' or 'column_first'. Determines whether to assign
                     coordinates row-wise or column-wise.

    Returns:
        dict: A dictionary mapping labels to 2D coordinates (xi, yi).
    """
    num_labels = len(labels)

    # If shape is not provided, calculate it to fit all labels
    if shape is None:
        side_length = math.ceil(math.sqrt(num_labels))
        shape = (side_length, side_length)

    rows, cols = shape

    # Ensure the shape can accommodate all labels
    if num_labels > rows * cols:
        raise ValueError("Shape is too small to fit all labels.")

    coordinates = {}
    count = 0

    # Assign coordinates based on the chosen order
    if order == "row_first":
        for row in range(rows):
            for col in range(cols):
                if count < num_labels:
                    coordinates[labels[count]] = (row, col)
                    count += 1
    elif order == "column_first":
        for col in range(cols):
            for row in range(rows):
                if count < num_labels:
                    coordinates[labels[count]] = (row, col)
                    count += 1
    else:
        raise ValueError("Order must be 'row_first' or 'column_first'.")

    return coordinates


def _get_gridshape_and_offsets(indices: list[tuple[int, int]]):
    """
    Calculates the shape of a grid and the offsets required to adjust for
    any negative indices in a list of 2D coordinates.

    Given a list of (row, column) indices, this function determines the
    overall grid shape necessary to encompass all provided indices, as well
    as the offset values needed to translate any negative indices into a
    positive-only grid system (e.g. for plotting on a figure).

    Args:
        indices (list[tuple[int, int]]): A list of tuples where each tuple
            contains a pair of integers representing the (row, column)
            indices in a 2D grid.

    Returns:
        tuple: A tuple containing:
            - grid_shape (tuple[int, int]): The shape of the grid as
              (number of rows, number of columns).
            - row_offset (int): The amount to offset the row indices to
              ensure all are non-negative.
            - column_offset (int): The amount to offset the column indices
              to ensure all are non-negative.

    Example:
        >>> indices = [(0, 0), (-1, 2), (2, -3)]
        >>> _get_gridshape_and_offsets(indices)
        ((4, 6), 1, 3)
    """
    row_indices = [i[0] for i in indices]
    column_indices = [i[1] for i in indices]
    grid_shape = (
        max(row_indices) - min(row_indices) + 1,
        max(column_indices) - min(column_indices) + 1,
    )
    # calculate offset in case there are negative indices,
    # the index are padded by the offset
    # such that because all indices in the grid are positive
    row_offset = abs(np.minimum(0, min(row_indices)))
    column_offset = abs(np.minimum(0, min(column_indices)))
    return grid_shape, row_offset, column_offset


def add_text(ax, text, fontsize=35, alpha=0.2, **kwargs):
    """
    Adds a text label at the center of the given axis.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to place the text.
        text (str): The text to display.
        fontsize (int, optional): Font size of the text.
        alpha (float, optional): Opacity of the text.
        **kwargs: Additional keyword arguments to pass to ax.text()

    Example:
        fig, ax = plt.subplots()
        add_translucent_text(ax, "Sample Text")
        plt.show()
    """
    # Get the center of the axis in data coordinates
    x_center = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
    y_center = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2

    # Add the text at the center
    ax.text(
        x_center,
        y_center,
        text,
        fontsize=fontsize,
        alpha=alpha,
        ha="center",
        va="center",
        **kwargs,
    )


def savefig(fig, path, fig_name, bbox_inches="tight", extension="pdf", dpi=None):
    figpath = pathlib.Path(path) / (
        fig_name + f"_{a_tools.current_timestamp()}.{extension}"
    )
    fig.savefig(str(figpath), bbox_inches=bbox_inches, dpi=dpi)


def safe_format_str_with_keys(mystr: str, **kwargs) -> str:
    """Safely formats a string, ignores all keys if one key is missing."""
    try:
        return mystr.format(**kwargs)
    except KeyError:
        return mystr
