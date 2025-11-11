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
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from camel_converter import to_camel
from multiprocessing import Pool, cpu_count

from warnings import simplefilter
simplefilter(action="ignore")

from trainModels.main import process_single_ticker
from trainModels.helper import _ensure_directories_exist
from prepareTechnicalIndicators.helper import get_all_technical_indicators


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

    _ensure_directories_exist(label_types)

    label_dir = Path("data/stock/02_label")
    all_label_files = sorted(label_dir.glob("*.csv"))

    if args.tickers:
        specified_tickers = [t.strip().upper() for t in args.tickers.split(",")]
        label_files = [f for f in all_label_files if f.stem in specified_tickers]

        missing_tickers = set(specified_tickers) - set([f.stem for f in label_files])
        if missing_tickers:
            print(
                f"Warning: The following tickers were not found: {', '.join(missing_tickers)}"
            )

        if not label_files:
            print(f"Error: None of the specified tickers found in {label_dir}")
            return
    else:
        label_files = all_label_files

    if args.workers is None:
        args.workers = max(1, cpu_count() - 1)

    print(f"Found {len(label_files)} stocks to process")
    if args.tickers:
        print(
            f"Processing specific tickers: {', '.join([f.stem for f in label_files])}"
        )
    print(f"Label types: {', '.join(label_types)}")
    print(f"Rolling windows: {', '.join(map(str, rolling_windows))} days")
    print(f"Workers: {args.workers}\n")

    feature_columns = get_all_technical_indicators()

    args_list = [
        (label_file, label_types, rolling_windows, feature_columns)
        for label_file in label_files
    ]

    all_failed_stocks = []
    all_metrics = {}

    with Pool(processes=args.workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_ticker, args_list),
                total=len(label_files),
                desc="Processing stocks",
            )
        )

        for failed_stocks, metrics_list in results:
            all_failed_stocks.extend(failed_stocks)

            for label_type, window, metrics_df in metrics_list:
                key = (label_type, window)
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(metrics_df)

    if all_metrics:
        print("\nSaving performance metrics...")
        for (label_type, window), metrics_dfs in all_metrics.items():
            camel_label = to_camel(label_type)
            filepath = f"data/stock/03_model/performance/{camel_label}/{window}dd.csv"

            combined_metrics = pd.concat(metrics_dfs, ignore_index=True)
            combined_metrics.to_csv(filepath, index=False)

    print(f"\n{'=' * 60}")
    print(f"Training complete! Total stocks: {len(label_files)}")

    if all_failed_stocks:
        print(f"\nFailed trainings ({len(all_failed_stocks)}):")
        for emiten, label_type, window, error in all_failed_stocks[:10]:
            print(f"  - {emiten} ({label_type} {window}dd): {error}")
        if len(all_failed_stocks) > 10:
            print(f"  ... and {len(all_failed_stocks) - 10} more")
    else:
        print("\nAll trainings successful!")

    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
