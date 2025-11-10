"""
Pipeline Step 3: Train Models for Stock Prediction

This script trains machine learning models for stock prediction using multiprocessing.
It reads from data/stock/02_label/*.csv and outputs:
- Trained models to data/stock/03_model/{label_type}/*.pkl
- Performance metrics to data/stock/03_model/performance/{label_type}/*.csv

Usage:
    python -m pipeline.03_train_models
    python -m pipeline.03_train_models --label_types median_gain,max_loss --windows 5,10,20
    python -m pipeline.03_train_models --workers 10
    python -m pipeline.03_train_models --tickers BBCA,BBRI,TLKM
"""

import argparse

from warnings import simplefilter
simplefilter(action="ignore")

from trainModels.main import train_models

def main():
    parser = argparse.ArgumentParser(description="Train ML models for stock prediction")

    parser.add_argument(
        "--label_types",
        type=str,
        default="median_gain,max_loss",
        help="Comma-separated label types (default: median_gain,max_loss)",
    )

    parser.add_argument(
        "--windows",
        type=str,
        default="5,10",
        help="Comma-separated rolling windows in days (default: 5,10,20)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)",
    )

    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated list of specific tickers to process (e.g., 'BBCA,BBRI,TLKM'). If not provided, all tickers will be processed.",
    )

    args = parser.parse_args()

    label_types = [lt.strip() for lt in args.label_types.split(",")]
    rolling_windows = [int(w.strip()) for w in args.windows.split(",")]
    
    valid_label_types = ["linear_trend", "median_gain", "max_loss"]
    for label_type in label_types:
        if label_type not in valid_label_types:
            print(f"Error: Invalid label type: {label_type}")
            return

    train_models(
        label_types, rolling_windows, workers=args.workers, tickers=args.tickers
    )


if __name__ == "__main__":
    main()
