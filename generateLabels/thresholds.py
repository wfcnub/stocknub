import numpy as np


def get_purged_training_labels(
    label_values,
    rolling_window: int,
    test_length: int,
    val_length: int,
) -> np.ndarray:
    """
    Select labels from the training period only for threshold estimation.

    The tail reserved for validation and testing is excluded, and a further
    horizon-sized embargo ensures the selected training labels do not depend on
    prices from the validation period.
    """
    values = np.asarray(label_values, dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if len(valid_positions) == 0:
        return np.array([], dtype=float)

    last_valid_position = valid_positions[-1] + 1
    training_end = last_valid_position - test_length - val_length
    purged_training_end = max(0, training_end - rolling_window)
    training_values = values[:purged_training_end]
    return training_values[np.isfinite(training_values)]
