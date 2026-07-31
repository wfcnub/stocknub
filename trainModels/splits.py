import re

import pandas as pd


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
    """
    Remove labels whose forward-looking horizon crosses a split boundary.

    Both the end of the training fold and the end of the validation fold are
    embargoed. This keeps validation outcomes out of training and test outcomes
    out of the final train+validation fit.
    """
    horizon = get_target_horizon(target_column)
    if horizon <= 0:
        return train_val_mask, val_mask

    date_values = data['Date'].astype(str)
    train_dates = sorted(date_values[train_mask].unique())
    val_dates = sorted(date_values[val_mask].unique())
    purged_dates = set(train_dates[-horizon:]) | set(val_dates[-horizon:])
    purged_train_val_mask = train_val_mask & ~date_values.isin(purged_dates)
    purged_val_mask = val_mask & ~date_values.isin(purged_dates)
    return purged_train_val_mask, purged_val_mask
