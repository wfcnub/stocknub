"""
Pipeline Description: Train Model for Stock Prediction

Usage:
    # Develop models with default settings
    python -m pipeline.train_models
"""

import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from camel_converter import to_camel
from multiprocessing import Pool, cpu_count

from warnings import simplefilter
simplefilter(action="ignore")

from trainModels.main import process_single_model
from trainModels.helper import _ensure_directories_exist
from prepareTechnicalIndicators.helper import get_all_technical_indicators


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Description: Train ML models for stock prediction"
    )

    parser.add_argument(
        "--label_types",
        type=str,
        default="median_gain,median_loss",
        help="Comma-separated label types (default: median_gain, median_loss)",
    )

    parser.add_argument(
        "--windows",
        type=str,
        default="5,10",
        help="Comma-separated rolling windows in days (default: 5,10)",
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
        "--model_version",
        type=int,
        default=1,
        help="The version of model to develop",
    )

    parser.add_argument(
        "--identifiers",
        type=str,
        default="",
        help="A selected identifiers for developing the model, can be an emiten or an industry",
    )

    args = parser.parse_args()

    label_types = [lt.strip() for lt in args.label_types.split(",")]
    rolling_windows = [int(w.strip()) for w in args.windows.split(",")]
    
    valid_label_types = [ "median_gain", "median_loss"]
    for label_type in label_types:
        if label_type not in valid_label_types:
            print(f"Error: Invalid label type: {label_type}")
            return

    _ensure_directories_exist(args.model_version, label_types)

    print("=" * 80)
    print(f"PIPELINE DESCRIPTION: DEVELOP MODEL V{args.model_version}")
    print("=" * 80)
    if args.identifiers:
        specified_identifiers = [t.strip() for t in args.identifiers.split(",")]

    else:
        if args.model_version == 1:
            specified_identifiers = pd.read_csv('data/selected_emiten_and_industry_list.csv') \
                                        ['Kode'] \
                                        .unique() \
                                        .tolist()
        elif args.model_version == 2:
            specified_identifiers = pd.read_csv('data/selected_emiten_and_industry_list.csv') \
                                        ['Industri'] \
                                        .unique() \
                                        .tolist()
        elif args.model_version == 3:
            specified_identifiers = ['IHSG']
    
        
    print(f"Found {len(specified_identifiers)} identifier to process")
    if args.identifiers:
        print(
            f"Processing specific identifiers: {', '.join(specified_identifiers)}"
        )
    print(f"Label types: {', '.join(label_types)}")
    print(f"Rolling windows: {', '.join(map(str, rolling_windows))} days")
    print(f"Workers: {args.workers}\n")

    feature_columns = get_all_technical_indicators()

    args_list = [
        (identifier, label_types, rolling_windows, feature_columns, args.model_version)
        for identifier in specified_identifiers
    ]

    all_failed_processes = []
    all_metrics = {}

    with Pool(processes=args.workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_model, args_list),
                total=len(specified_identifiers),
                desc="Processing single model",
            )
        )

        for failed_process, metrics_list in results:
            all_failed_processes.extend(failed_process)

            for label_type, window, metrics_df in metrics_list:
                key = (label_type, window)
                if key not in all_metrics:
                    all_metrics[key] = []

                all_metrics[key].append(metrics_df)

    if all_metrics:
        print("\nSaving performance metrics...")
        for (label_type, window), metrics_dfs in all_metrics.items():
            camel_label = to_camel(label_type)
            filepath = f"data/stock/model_v{args.model_version}/performance/{camel_label}/{window}dd.csv"

            combined_metrics = pd.concat(metrics_dfs, ignore_index=True)
            combined_metrics.to_csv(filepath, index=False)

    print(f"\n{'=' * 60}")
    print(f"Training complete! Total stocks: {len(specified_identifiers)}")

    if all_failed_processes:
        print(f"\nFailed trainings ({len(all_failed_processes)}):")
        for emiten, label_type, window, error in all_failed_processes:
            print(f"  - {emiten} ({label_type} {window}dd): {error}")
    else:
        print("\nAll trainings successful!")

    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
