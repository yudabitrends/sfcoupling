"""
I/O utilities for format discovery and loading H5/CSV/TSV/NPY.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def discover_format(path: str) -> str:
    """
    Infer file format from path extension.
    Returns: "h5", "csv", "tsv", or "npy"
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    ext = Path(path).suffix.lower()
    if ext == ".h5" or ext == ".hdf5":
        return "h5"
    if ext == ".npy":
        return "npy"
    if ext == ".csv":
        return "csv"
    if ext == ".tsv" or ext == ".txt":
        return "tsv"
    raise ValueError(
        f"Cannot infer format for {path}. Supported: .h5, .hdf5, .csv, .tsv, .npy"
    )


def infer_delimiter(path: str) -> str:
    """Infer delimiter from file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".tsv" or ext == ".txt":
        return "\t"
    return ","


def load_h5_table(path: str, key: str) -> pd.DataFrame:
    """Load a DataFrame from pandas HDFStore by key."""
    try:
        df = pd.read_hdf(path, key=key)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read HDF5 {path} key={key}. "
            f"Ensure pytables is installed (pip install tables). Error: {e}"
        ) from e
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"HDF5 key {key} did not return a DataFrame, got {type(df)}")
    return df


def load_csv_table(path: str, delimiter: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV/TSV file into a DataFrame."""
    if delimiter is None:
        delimiter = infer_delimiter(path)
    try:
        return pd.read_csv(path, delimiter=delimiter, low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}") from e


def load_npy(path: str) -> np.ndarray:
    """Load a .npy file."""
    try:
        arr = np.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load NPY {path}: {e}") from e
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return arr


def get_h5_keys(path: str) -> list:
    """List available keys in an HDF5 file."""
    try:
        with pd.HDFStore(path, "r") as store:
            return list(store.keys())
    except Exception as e:
        raise RuntimeError(
            f"Failed to open HDF5 {path}. Ensure pytables is installed. Error: {e}"
        ) from e
