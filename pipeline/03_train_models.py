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
import pickle
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from camel_converter import to_camel
from multiprocessing import Pool, cpu_count

from modelDevelopment.main import develop_model
from technicalIndicators.helper import get_all_technical_indicators
from utils.pipeline import get_label_config

from warnings import simplefilter

simplefilter(action="ignore")


def ensure_directories_exist(label_types):
    """Ensure all required directories exist before training."""
    for label_type in label_types:
        camel_label = to_camel(label_type)
        Path(f"data/stock/03_model/{camel_label}").mkdir(parents=True, exist_ok=True)
        Path(f"data/stock/03_model/performance/{camel_label}").mkdir(
            parents=True, exist_ok=True
        )


def save_model(model, label_type, emiten, window):
    """Save a trained model to file."""
    camel_label = to_camel(label_type)
    filepath = f"data/stock/03_model/{camel_label}/{emiten}-{window}dd.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def combine_metrics(emiten, train_metrics, test_metrics, threshold):
    """Combine train and test metrics into a single DataFrame row."""
    train_df = pd.DataFrame(train_metrics)
    train_df.columns = [f"Train - {col}" for col in train_df.columns]

    test_df = pd.DataFrame(test_metrics)
    test_df.columns = [f"Test - {col}" for col in test_df.columns]

    result = pd.concat([train_df, test_df], axis=1)
    result.insert(0, "Kode", emiten)
    result["Threshold"] = threshold

    return result


def process_single_stock(args_tuple):
    """Process a single stock - train models for all label types and windows."""
    label_file, label_types, rolling_windows, feature_columns = args_tuple
    emiten = label_file.stem
    failed_stocks = []
    metrics_list = []  # Store metrics instead of saving directly

    try:
        data = pd.read_csv(label_file)

        for label_type in label_types:
            for window in rolling_windows:
                target_col, threshold_col, pos_label, neg_label = get_label_config(
                    label_type, window
                )

                if target_col not in data.columns or threshold_col not in data.columns:
                    continue

                required_cols = feature_columns + [target_col]
                clean_data = data[required_cols].dropna()

                if len(clean_data) < 100:
                    continue

                try:
                    model, train_metrics, test_metrics = develop_model(
                        data, target_col, pos_label, neg_label
                    )

                    save_model(model, label_type, emiten, window)

                    threshold_value = data[threshold_col].iloc[0]
                    metrics_df = combine_metrics(
                        emiten, train_metrics, test_metrics, threshold_value
                    )

                    # Store metrics with label_type and window info for batch saving
                    metrics_list.append((label_type, window, metrics_df))

                except Exception as e:
                    failed_stocks.append((emiten, label_type, window, str(e)))

    except Exception as e:
        failed_stocks.append((emiten, "all", "all", str(e)))

    return failed_stocks, metrics_list


def train_models(label_types, rolling_windows, workers=None, tickers=""):
    """Train models for all stocks in data/stock/02_label."""
    ensure_directories_exist(label_types)

    label_dir = Path("data/stock/02_label")
    all_label_files = sorted(label_dir.glob("*.csv"))

    # Filter by specific tickers if provided
    if tickers:
        specified_tickers = [t.strip().upper() for t in tickers.split(",")]
        label_files = [f for f in all_label_files if f.stem in specified_tickers]

        # Check if any specified tickers were not found
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

    if workers is None:
        workers = max(1, cpu_count() - 1)

    print(f"Found {len(label_files)} stocks to process")
    if tickers:
        print(
            f"Processing specific tickers: {', '.join([f.stem for f in label_files])}"
        )
    print(f"Label types: {', '.join(label_types)}")
    print(f"Rolling windows: {', '.join(map(str, rolling_windows))} days")
    print(f"Workers: {workers}\n")

    feature_columns = get_all_technical_indicators()

    # Prepare arguments for multiprocessing
    args_list = [
        (label_file, label_types, rolling_windows, feature_columns)
        for label_file in label_files
    ]

    # Process stocks in parallel
    all_failed_stocks = []
    all_metrics = {}  # Group metrics by (label_type, window)

    with Pool(processes=workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_stock, args_list),
                total=len(label_files),
                desc="Processing stocks",
            )
        )

        # Collect results
        for failed_stocks, metrics_list in results:
            all_failed_stocks.extend(failed_stocks)

            # Group metrics by label_type and window
            for label_type, window, metrics_df in metrics_list:
                key = (label_type, window)
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(metrics_df)

    # Save all metrics at once (avoids race conditions)
    if all_metrics:
        print("\nSaving performance metrics...")
        for (label_type, window), metrics_dfs in all_metrics.items():
            camel_label = to_camel(label_type)
            filepath = f"data/stock/03_model/performance/{camel_label}/{window}dd.csv"

            # Concatenate all metrics and save once
            combined_metrics = pd.concat(metrics_dfs, ignore_index=True)
            combined_metrics.to_csv(filepath, index=False)

    print(f"\n{'=' * 60}")
    print(f"Training complete! Total stocks: {len(label_files)}")

    if all_failed_stocks:
        print(f"\nFailed trainings ({len(all_failed_stocks)}):")
        for emiten, label_type, window, error in all_failed_stocks:
            print(f"  - {emiten} ({label_type} {window}dd): {error}")
    else:
        print("\nAll trainings successful!")

    print(f"{'=' * 60}")


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
        default="5,10,20",
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
