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
        "--with_docker",
        dest='with_docker', 
        action='store_true',
        help="A boolean for stating whether the system uses docker. If True, than the program wouldn't us multiprocessing"
    )

    parser.set_defaults(with_docker=False)

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

    if args.model_version == 1:
        specified_identifiers = pd.read_csv(Path('data/selected_ticker_and_industry_list.csv')) \
                                    ['Ticker'] \
                                    .unique() \
                                    .tolist()

    elif args.model_version == 2:
        specified_identifiers = pd.read_csv(Path('data/selected_ticker_and_industry_list.csv')) \
                                    ['Industry'] \
                                    .unique() \
                                    .tolist()
    elif args.model_version in [3, 4]:
        specified_identifiers = ['IHSG']
    
    print(f"Found {len(specified_identifiers)} identifier to process")
    print(f"Label types: {', '.join(label_types)}")
    print(f"Rolling windows: {', '.join(map(str, rolling_windows))} days")
    
    args_list = []
    for identifier in specified_identifiers:
        for label_type in label_types:
            for window in rolling_windows:
                args_list.append((identifier, label_type, window, args.model_version))

    all_failed_processes = []
    all_metrics = {}

    if args.with_docker:
        print(f"Workers: 1\n")
        results = list(
                        tqdm(
                            map(process_single_model, args_list),
                            total=len(args_list),
                            desc="Processing single model",
                        )
                    )
        
    else:
        print(f"Workers: {args.workers}\n")
        with Pool(processes=args.workers) as pool:
            results = list(
                tqdm(
                    pool.imap(process_single_model, args_list),
                    total=len(args_list),
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
            filepath = Path(f"data/stock/model_v{args.model_version}/performance/{camel_label}/{window}dd.csv")

            combined_metrics = pd.concat(metrics_dfs, ignore_index=True)
            combined_metrics.to_csv(filepath, index=False)

    print(f"\n{'=' * 60}")
    print(f"Training complete! Total stocks: {len(specified_identifiers)}")

    if all_failed_processes:
        print(f"\nFailed trainings ({len(all_failed_processes)}):")
        for ticker, label_type, window, error in all_failed_processes:
            print(f"  - {ticker} ({label_type} {window}dd): {error}")
    else:
        print("\nAll trainings successful!")

    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
