import pandas as pd
from tqdm import tqdm
from pathlib import Path
from camel_converter import to_camel
from multiprocessing import Pool, cpu_count
from utils.pipeline import get_label_config
from prepareTechnicalIndicators.helper import get_all_technical_indicators

from trainModels.modelling_v1 import develop_model
from trainModels.helper import _ensure_directories_exist, _save_model, _combine_metrics

def process_single_ticker(args_tuple):
    label_file, label_types, rolling_windows, feature_columns = args_tuple
    emiten = label_file.stem
    failed_stocks = []
    metrics_list = []

    try:
        data = pd.read_csv(label_file)

        for label_type in label_types:
            for window in rolling_windows:
                target_col, threshold_col, pos_label, neg_label = get_label_config(
                    label_type, window
                )

                required_cols = feature_columns + [target_col]
                clean_data = data[required_cols].dropna(subset=[target_col])

                try:
                    model, train_metrics, test_metrics = develop_model(
                        clean_data, target_col, pos_label, neg_label
                    )

                    _save_model(model, label_type, emiten, window)

                    threshold_value = data[threshold_col].iloc[0]
                    metrics_df = _combine_metrics(
                        emiten, train_metrics, test_metrics, threshold_value
                    )

                    metrics_list.append((label_type, window, metrics_df))

                except Exception as e:
                    failed_stocks.append((emiten, label_type, window, str(e)))

    except Exception as e:
        failed_stocks.append((emiten, "all", "all", str(e)))

    return failed_stocks, metrics_list


def train_models(label_types, rolling_windows, workers=None, tickers=""):
    _ensure_directories_exist(label_types)

    label_dir = Path("data/stock/02_label")
    all_label_files = sorted(label_dir.glob("*.csv"))

    if tickers:
        specified_tickers = [t.strip().upper() for t in tickers.split(",")]
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

    args_list = [
        (label_file, label_types, rolling_windows, feature_columns)
        for label_file in label_files
    ]

    all_failed_stocks = []
    all_metrics = {}

    with Pool(processes=workers) as pool:
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