"""Tests for vaila/mpangles.py's stick-figure sequence frame selection."""

import numpy as np
import pandas as pd

from vaila import mpangles as mp

_LANDMARK_NAMES = mp._STICK_LANDMARK_NAMES  # noqa: SLF001 - same list the function uses
_NAME_TO_IDX = {name: i for i, name in enumerate(_LANDMARK_NAMES)}


def _landmarks_df(n: int, empty_frames: set[int]) -> pd.DataFrame:
    """``n`` frames with every landmark at (1.0, 1.0), NaN for the whole row on ``empty_frames``."""
    cols: dict[str, np.ndarray] = {"frame": np.arange(n, dtype=float)}
    for i in range(len(_LANDMARK_NAMES)):
        x = np.ones(n)
        y = np.ones(n)
        for f in empty_frames:
            x[f] = np.nan
            y[f] = np.nan
        cols[f"p{i}_x"] = x
        cols[f"p{i}_y"] = y
    return pd.DataFrame(cols)


def test_frames_with_person_skips_empty_frames():
    empty = {2, 3, 7}
    df = _landmarks_df(10, empty)
    picked = mp._frames_with_person(df, 1, _NAME_TO_IDX, num_frames=5, n=10)
    assert not (set(picked.tolist()) & empty)
    assert len(picked) == 5


def test_frames_with_person_falls_back_when_all_empty():
    df = _landmarks_df(10, set(range(10)))
    picked = mp._frames_with_person(df, 1, _NAME_TO_IDX, num_frames=4, n=10)
    assert len(picked) == 4
    assert picked.min() >= 0 and picked.max() <= 9


def test_frames_with_person_falls_back_without_core_columns():
    # Only 3 columns besides "frame": far too few to reach the hip/knee indices.
    df = pd.DataFrame({"frame": np.arange(5, dtype=float), "p0_x": 1.0, "p0_y": 1.0})
    picked = mp._frames_with_person(df, 1, _NAME_TO_IDX, num_frames=3, n=5)
    assert len(picked) == 3
    assert picked.min() >= 0 and picked.max() <= 4
