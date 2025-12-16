"""
input_validation.py

Module for dataset initialization input validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandera as pa
from pandera import Check


def _cell_contains_pathlike(x: Any, *, check_exists: bool) -> bool:
    """
    Validate that a cell contains a valid path-like object.
    Intended to be used as a lambda function in the DataFrame.applymap method.

    :param x: The cell value to check.
    :param check_exists: Whether to check if the path exists.
    :return: True if the cell contains a valid path-like object, False otherwise.
    """
    if not isinstance(x, (str, Path)):
        return False
    if not str(x).strip():
        return False
    return (not check_exists) or Path(x).exists()


def make_file_index_schema(*, check_exists: bool = False) -> pa.DataFrameSchema:
    """
    Create a pandera schema for validating the file_index DataFrame.

    :param check_exists: Whether to check if the paths in the file_index exist.
    :return: A pandera DataFrameSchema object.
    """
    
    return pa.DataFrameSchema(
        columns={},  # no fixed column names
        checks=[
            Check(
                lambda df: df.shape[1] > 0,
                error="file_index must have at least one column.",
            ),
            Check(
                lambda df: ~df.isna().to_numpy().any(),
                error="file_index may not contain NA values.",
            ),
            Check(
                lambda df: df.applymap(
                    lambda x: _cell_contains_pathlike(x, check_exists=check_exists)
                ).to_numpy().all(),
                error=(
                    "All file_index cells must be string/pathlib.Path objects"
                    + (" that exist on disk." if check_exists else ".")
                ),
            ),
        ],
        strict=False,
    )
