import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    """Integer row indices for one purged, expanding-window fold."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_end: str
    validation_start: str
    validation_end: str


def get_target_horizon(target_column: str) -> int:
    match = re.search(r"(\d+)dd$", target_column)
    return int(match.group(1)) if match else 0


def purge_overlapping_label_periods(
    data: pd.DataFrame,
    train_val_mask: pd.Series,
    train_mask: pd.Series,
    val_mask: pd.Series,
    target_column: str,
) -> tuple[pd.Series, pd.Series]:
    """Remove labels whose forward horizon crosses a configured boundary."""
    horizon = get_target_horizon(target_column)
    if horizon <= 0:
        return train_val_mask, val_mask

    date_values = data["Date"].astype(str)
    train_dates = sorted(date_values[train_mask].unique())
    val_dates = sorted(date_values[val_mask].unique())
    purged_dates = set(train_dates[-horizon:]) | set(val_dates[-horizon:])
    purged_train_val_mask = train_val_mask & ~date_values.isin(purged_dates)
    purged_val_mask = val_mask & ~date_values.isin(purged_dates)
    return purged_train_val_mask, purged_val_mask


def purge_development_test_boundary(
    data: pd.DataFrame,
    development_mask: pd.Series,
    target_column: str,
) -> pd.Series:
    """Purge the final development labels that look into the locked test set."""
    horizon = get_target_horizon(target_column)
    if horizon <= 0:
        return development_mask

    dates = data["Date"].astype(str)
    development_dates = sorted(dates[development_mask].unique())
    if len(development_dates) <= horizon:
        raise ValueError("Not enough development dates after the test-boundary purge")
    return development_mask & ~dates.isin(development_dates[-horizon:])


def _has_enough_classes(
    target: pd.Series,
    positive_label: str,
    minimum_per_class: int,
) -> bool:
    positive_count = int((target == positive_label).sum())
    negative_count = int(len(target) - positive_count)
    return min(positive_count, negative_count) >= minimum_per_class


def make_purged_walk_forward_splits(
    data: pd.DataFrame,
    target_column: str,
    positive_label: str,
    n_splits: int = 4,
    validation_dates: int = 120,
    min_train_dates: int = 252,
    min_class_count: int = 5,
    min_valid_splits: int = 2,
) -> list[WalkForwardFold]:
    """Create expanding time folds with a label-horizon purge before each fold.

    All rows from the same date stay in the same fold. Folds without enough
    positive and negative observations are rejected instead of silently falling
    back to accuracy.
    """
    if n_splits < 1 or validation_dates < 1 or min_train_dates < 1:
        raise ValueError("Split counts and window lengths must be positive")
    if "Date" not in data or target_column not in data:
        raise ValueError("Walk-forward splitting requires Date and target columns")

    dates = data["Date"].astype(str)
    unique_dates = np.asarray(sorted(dates.unique()))
    available_splits = (len(unique_dates) - min_train_dates) // validation_dates
    split_count = min(n_splits, max(0, int(available_splits)))
    if split_count < min_valid_splits:
        raise ValueError(
            "Insufficient history for robust walk-forward validation: "
            f"{len(unique_dates)} dates available, need at least "
            f"{min_train_dates + min_valid_splits * validation_dates}"
        )

    first_validation_position = len(unique_dates) - split_count * validation_dates
    horizon = get_target_horizon(target_column)
    folds: list[WalkForwardFold] = []

    for fold_number in range(split_count):
        validation_start_position = (
            first_validation_position + fold_number * validation_dates
        )
        validation_end_position = min(
            validation_start_position + validation_dates,
            len(unique_dates),
        )
        validation_block = unique_dates[
            validation_start_position:validation_end_position
        ]
        train_block = unique_dates[:validation_start_position]
        if horizon:
            train_block = train_block[:-horizon]
        if len(train_block) < min_train_dates or len(validation_block) == 0:
            continue

        train_mask = dates.isin(train_block)
        validation_mask = dates.isin(validation_block)
        train_target = data.loc[train_mask, target_column]
        validation_target = data.loc[validation_mask, target_column]
        if not _has_enough_classes(
            train_target, positive_label, min_class_count
        ) or not _has_enough_classes(
            validation_target, positive_label, min_class_count
        ):
            continue

        folds.append(
            WalkForwardFold(
                train_indices=np.flatnonzero(train_mask.to_numpy()),
                validation_indices=np.flatnonzero(validation_mask.to_numpy()),
                train_end=str(train_block[-1]),
                validation_start=str(validation_block[0]),
                validation_end=str(validation_block[-1]),
            )
        )

    if len(folds) < min_valid_splits:
        raise ValueError(
            "Fewer than two walk-forward folds contain enough examples of both "
            "classes; use more history or a longer validation window"
        )
    return folds
