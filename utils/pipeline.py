"""
Pipeline configuration and business logic utilities.

This module contains pipeline-specific configuration functions and business logic
that don't fit into generic I/O or data fetching categories.
"""

def get_label_config(label_type: str, window: int) -> tuple:
    """
    Get configuration for a specific label type and window.

    Args:
        label_type (str): Type of label ('median_gain', 'median_loss')
        window (int): Forecast window in days

    Returns:
        tuple: (target_column, threshold_column, positive_label, negative_label)

    Raises:
        ValueError: If label_type is not recognized
    """
    if label_type == "median_gain":
        return (
            f"Median Gain {window}dd",
            f"Threshold Median Gain {window}dd",
            "High Gain",
            "Low Gain",
        )
    elif label_type == "median_loss":
        return (
            f"Median Loss {window}dd",
            f"Threshold Median Loss {window}dd",
            "High Loss",
            "Low Loss",
        )
    else:
        raise ValueError(f"Unknown label type: {label_type}")
