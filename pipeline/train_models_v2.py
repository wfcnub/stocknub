"""
Pipeline Step 3: Train Models for Stock Prediction

This script trains machine learning models for stock prediction using multiprocessing.
It reads from data/stock/label/*.csv and outputs:
- Trained models to data/stock/model/{label_type}/*.pkl
- Performance metrics to data/stock/model/performance/{label_type}/{window}dd/{emiten}.csv

Usage:
    # Develop models with default settings
    python -m pipeline.train_models_v2
    
    # Process specific tickers
    python -m pipeline.train_models_v2 --tickers BBCA,BBRI,TLKM
"""

import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from camel_converter import to_camel
from multiprocessing import Pool, cpu_count

from warnings import simplefilter
simplefilter(action="ignore")

from trainModelsV2.main import process_single_ticker
from trainModelsV2.helper import _ensure_directories_exist
from prepareTechnicalIndicators.helper import get_all_technical_indicators


def main():
    parser = argparse.ArgumentParser(description="Train ML models for stock prediction")

    parser.add_argument(
        "--label_types",
        type=str,
        default="median_gain",
        help="Comma-separated label types (default: median_gain,max_loss)",
    )

    parser.add_argument(
        "--windows",
        type=str,
        default="5",
        help="Comma-separated rolling windows in days (default: 5,10,20)",
    )

    parser.add_argument(
        "--labels_folder",
        type=str,
        default="data/stock/label",
        help="Folder to save labels (default: data/stock/label)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="Number of parallel workers (default: CPU count)",
    )

    parser.add_argument(
        "--industry",
        type=str,
        default="",
        help="",
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

    if args.industry:
        specified_industries = [t.strip().title() for t in args.industry.split(",")]
        
    else:
        specified_industries = pd.read_csv('data/emiten_and_industry_list.csv')['Industri'].unique().tolist()
    
    print(f"Found {len(specified_industries)} industries to process")
    if args.industry:
        print(
            f"Processing specific industries: {', '.join([f for f in specified_industries])}"
        )
    print(f"Label types: {', '.join(label_types)}")
    print(f"Rolling windows: {', '.join(map(str, rolling_windows))} days")
    print(f"Workers: {args.workers}\n")

    feature_columns = get_all_technical_indicators()

    args_list = [
        (industry, label_types, rolling_windows, feature_columns)
        for industry in specified_industries
    ]

    all_failed_industries = []
    all_metrics = {}

    with Pool(processes=args.workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_ticker, args_list),
                total=len(specified_industries),
                desc="Processing stocks",
            )
        )

        for failed_industry, metrics_list in results:
            all_failed_industries.extend(failed_industry)

            for label_type, window, metrics_df in metrics_list:
                key = (label_type, window)
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(metrics_df)

    if all_metrics:
        print("\nSaving performance metrics...")
        for (label_type, window), metrics_dfs in all_metrics.items():
            camel_label = to_camel(label_type)
            filepath = f"data/stock/model_v2/performance/{camel_label}/{window}dd.csv"

            combined_metrics = pd.concat(metrics_dfs, ignore_index=True)
            combined_metrics.to_csv(filepath, index=False)

    print(f"\n{'=' * 60}")
    print(f"Training complete! Total industries: {len(specified_industries)}")

    if all_failed_industries:
        print(f"\nFailed trainings ({len(all_failed_industries)}):")
        for industry, label_type, window, error in all_failed_industries:
            print(f"  - {industry} ({label_type} {window}dd): {error}")
    else:
        print("\nAll trainings successful!")

    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
