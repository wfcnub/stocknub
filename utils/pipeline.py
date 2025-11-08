"""
Pipeline configuration and business logic utilities.

This module contains pipeline-specific configuration functions and business logic
that don't fit into generic I/O or data fetching categories.
"""


def get_label_config(label_type: str, window: int) -> tuple:
    """
    Get configuration for a specific label type and window.

    Args:
        label_type (str): Type of label ('linear_trend', 'median_gain', 'max_loss')
        window (int): Forecast window in days

    Returns:
        tuple: (target_column, threshold_column, positive_label, negative_label)

    Raises:
        ValueError: If label_type is not recognized
    """
    if label_type == "linear_trend":
        return (
            f"Linear Trend {window}dd",
            f"Threshold Linear Trend {window}dd",
            "Up Trend",
            "Down Trend",
        )
    elif label_type == "median_gain":
        return (
            f"Median Gain {window}dd",
            f"Threshold Median Gain {window}dd",
            "High Gain",
            "Low Gain",
        )
    elif label_type == "max_loss":
        return (
            f"Max Loss {window}dd",
            f"Threshold Max Loss {window}dd",
            "Low Risk",
            "High Risk",
        )
    else:
        raise ValueError(f"Unknown label type: {label_type}")
