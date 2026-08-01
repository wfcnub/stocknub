import argparse
import pandas as pd
from tqdm import tqdm
from camel_converter import to_camel
from multiprocessing import Pool, cpu_count

from utils import paths

from trainModels.main import process_single_model
from trainModels.config import CatBoostTrainingConfig
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
        default=min(4, cpu_count()),
        help="Outer parallel workers (default: min(4, CPU count))",
    )

    parser.add_argument(
        "--model_version",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="The version of model to develop",
    )

    parser.add_argument("--n_trials", type=int, default=80)
    parser.add_argument("--cv_folds", type=int, default=4)
    parser.add_argument("--validation_dates", type=int, default=120)
    parser.add_argument("--min_train_dates", type=int, default=252)
    parser.add_argument("--max_iterations", type=int, default=3000)
    parser.add_argument("--early_stopping_rounds", type=int, default=100)
    parser.add_argument("--max_features", type=int, default=160)
    parser.add_argument("--ensemble_size", type=int, default=3)
    parser.add_argument(
        "--tune_timeout_seconds",
        type=int,
        default=None,
        help="Optional wall-clock limit per Optuna study",
    )
    parser.add_argument(
        "--catboost_threads",
        type=int,
        default=None,
        help="Threads per CatBoost fit; defaults to CPU count divided by workers",
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    for name in (
        "n_trials",
        "cv_folds",
        "validation_dates",
        "min_train_dates",
        "max_iterations",
        "early_stopping_rounds",
        "max_features",
        "ensemble_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1")
    catboost_threads = args.catboost_threads or max(1, cpu_count() // args.workers)
    if args.tune_timeout_seconds is not None and args.tune_timeout_seconds < 1:
        parser.error("--tune_timeout_seconds must be at least 1")
    training_config = CatBoostTrainingConfig(
        n_trials=args.n_trials,
        n_folds=args.cv_folds,
        validation_dates=args.validation_dates,
        min_train_dates=args.min_train_dates,
        max_iterations=args.max_iterations,
        early_stopping_rounds=args.early_stopping_rounds,
        max_features=args.max_features,
        ensemble_size=args.ensemble_size,
        thread_count=catboost_threads,
        tune_timeout_seconds=args.tune_timeout_seconds,
    )

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
        specified_identifiers = pd.read_csv(paths.get_selected_ticker_and_industry_list_path()) \
                                    ['Ticker'] \
                                    .unique() \
                                    .tolist()

    elif args.model_version == 2:
        specified_identifiers = pd.read_csv(paths.get_selected_ticker_and_industry_list_path()) \
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
                args_list.append(
                    (
                        identifier,
                        label_type,
                        window,
                        args.model_version,
                        training_config,
                    )
                )

    all_failed_processes = []
    all_metrics = {}

    print(f"Workers: {args.workers}")
    if args.model_version in [1, 2, 3]:
        print(
            f"CatBoost threads/worker: {catboost_threads}; "
            f"walk-forward folds: {args.cv_folds}; trials: {args.n_trials}\n"
        )
    with Pool(processes=args.workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_model, args_list),
                    total=len(args_list),
                    desc="Processing single model",
                )
            )
    
        pool.close()
        pool.join()
    
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
            filepath = paths.get_model_performance_path(args.model_version, label_type, window)

            combined_metrics = pd.concat(metrics_dfs, ignore_index=True)
            combined_metrics.to_csv(filepath, index=False)

    print(f"\n{'=' * 60}")
    print(f"Training complete! Total stocks: {len(specified_identifiers)}")

    if all_failed_processes:
        failure_path = paths.get_model_performance_base_dir(
            args.model_version
        ) / "failures.csv"
        pd.DataFrame(
            all_failed_processes,
            columns=["Identifier", "Label Type", "Window", "Error"],
        ).to_csv(failure_path, index=False)
        print(f"\nFailed trainings ({len(all_failed_processes)}):")
        for ticker, label_type, window, error in all_failed_processes:
            print(f"  - {ticker} ({label_type} {window}dd): {error}")
    else:
        print("\nAll trainings successful!")

    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
