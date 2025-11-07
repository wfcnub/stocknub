"""
Pipeline Step 4: Forecast Stock Performance

This script generates forecasts using trained models from step 3.
It reads models from data/stock/03_model/{label_type}/*.pkl and outputs:
- Forecast results to data/stock/04_forecast/{label_type}/{window}dd.csv

The script:
- Filters stocks by minimum test Gini performance
- Reads technical indicators from existing CSV files (step 1 output)
- Applies trained models to generate probability predictions
- Saves forecasts for the most recent date

Usage:
    # Forecast with default settings (all models with min_test_gini >= 0.3)
    python -m pipeline.04_forecast --windows 5,10,20 --label_types median_gain,max_loss --min_test_gini 0.3 --workers 10

    # Forecast without Gini filter (use all available models)
    python -m pipeline.04_forecast --windows 5,10,20 --label_types median_gain,max_loss --workers 10

    # Forecast specific tickers only
    python -m pipeline.04_forecast --tickers "ADRO,ADMR" --windows 5,10 --label_types median_gain --workers 10
"""

import argparse
import pickle
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from camel_converter import to_camel
from multiprocessing import Pool

from technicalIndicators.helper import get_all_technical_indicators

from warnings import simplefilter

simplefilter(action="ignore")


def get_label_config(label_type, window):
    """Get configuration for a specific label type and window."""
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


def ensure_directories_exist(label_types, windows):
    """Ensure all required directories exist before forecasting."""
    for label_type in label_types:
        camel_label = to_camel(label_type)
        for window in windows:
            Path(f"data/stock/04_forecast/{camel_label}").mkdir(
                parents=True, exist_ok=True
            )


def clear_forecast_files(label_types, windows):
    """Clear existing forecast files to avoid duplicates."""
    for label_type in label_types:
        camel_label = to_camel(label_type)
        for window in windows:
            filepath = f"data/stock/04_forecast/{camel_label}/{window}dd.csv"
            if Path(filepath).exists():
                Path(filepath).unlink()


def load_model_performance(label_type, window, min_test_gini=None):
    """
    Load model performance data and filter by minimum Gini.

    Args:
        label_type: Type of label (median_gain, max_loss, linear_trend)
        window: Forecast window (e.g., 5, 10, 20)
        min_test_gini: Minimum test Gini threshold (None to include all)

    Returns:
        List of emiten codes that meet the criteria
    """
    camel_label = to_camel(label_type)
    performance_path = f"data/stock/03_model/performance/{camel_label}/{window}dd.csv"

    if not Path(performance_path).exists():
        print(f"WARNING: Performance file not found: {performance_path}")
        return []

    performance_df = pd.read_csv(performance_path)

    if min_test_gini is not None:
        filtered_df = performance_df[performance_df["Test - Gini"] >= min_test_gini]
        filtered_df = filtered_df.sort_values("Test - Gini", ascending=False)
        return filtered_df["Kode"].unique().tolist()
    else:
        return performance_df["Kode"].unique().tolist()


def get_filtered_emiten_list(label_types, windows, min_test_gini=None):
    """
    Get intersection of emiten codes that meet criteria across all label types and windows.

    Args:
        label_types: List of label types
        windows: List of forecast windows
        min_test_gini: Minimum test Gini threshold

    Returns:
        List of emiten codes that have models meeting criteria for all combinations
    """
    all_emiten_sets = []

    for label_type in label_types:
        for window in windows:
            emiten_list = load_model_performance(label_type, window, min_test_gini)
            if emiten_list:
                all_emiten_sets.append(set(emiten_list))

    if not all_emiten_sets:
        return []

    # Get intersection of all sets
    common_emiten = set.intersection(*all_emiten_sets)
    return sorted(list(common_emiten))


def save_forecast(forecast_df, label_type, window):
    """Save or append forecast results to CSV."""
    camel_label = to_camel(label_type)
    filepath = f"data/stock/04_forecast/{camel_label}/{window}dd.csv"

    if Path(filepath).exists():
        forecast_df.to_csv(filepath, mode="a", index=False, header=False)
    else:
        forecast_df.to_csv(filepath, index=False)


def process_single_forecast(args_tuple):
    """
    Process forecast for a single emiten, label_type, and window combination.

    Args:
        args_tuple: Tuple containing (emiten, label_type, window, feature_columns)

    Returns:
        Tuple of (emiten, label_type, window, success, message, forecast_data_dict)
    """
    emiten, label_type, window, feature_columns = args_tuple

    try:
        # Get label configuration
        target_col, threshold_col, positive_label, negative_label = get_label_config(
            label_type, window
        )

        # Load model
        camel_label = to_camel(label_type)
        model_path = f"data/stock/03_model/{camel_label}/{emiten}-{window}dd.pkl"

        if not Path(model_path).exists():
            return (
                emiten,
                label_type,
                window,
                False,
                f"Model not found: {model_path}",
                None,
            )

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Read technical indicators data
        try:
            technical_path = f"data/stock/01_technical/{emiten}.csv"
            if not Path(technical_path).exists():
                return (
                    emiten,
                    label_type,
                    window,
                    False,
                    f"Technical data not found: {technical_path}",
                    None,
                )

            forecasting_data = pd.read_csv(technical_path)

            if forecasting_data.empty:
                return (
                    emiten,
                    label_type,
                    window,
                    False,
                    "Technical data is empty",
                    None,
                )

            # Get the most recent row (tail)
            latest_data = forecasting_data.tail(1).copy()
            latest_data["Kode"] = emiten

        except Exception as e:
            return (
                emiten,
                label_type,
                window,
                False,
                f"Failed to read data: {str(e)}",
                None,
            )

        # Check if all required features are present
        missing_features = [
            col for col in feature_columns if col not in latest_data.columns
        ]
        if missing_features:
            return (
                emiten,
                label_type,
                window,
                False,
                f"Missing features: {missing_features[:5]}...",
                None,
            )

        # Make prediction
        forecast_column_name = f"Forecast {positive_label} {window}dd"
        positive_label_index = list(model.classes_).index(positive_label)

        forecast_proba = model.predict_proba(
            latest_data[feature_columns].values.reshape(1, -1)
        )[0, positive_label_index]

        # Get the date from the latest data
        latest_date = (
            latest_data["Date"].iloc[0] if "Date" in latest_data.columns else None
        )

        # Prepare result data (don't save yet, return it)
        forecast_data = {
            "Kode": emiten,
            "Date": latest_date,
            "forecast_column": forecast_column_name,
            "forecast_value": forecast_proba,
        }

        return (
            emiten,
            label_type,
            window,
            True,
            f"Forecast: {forecast_proba:.4f}",
            forecast_data,
        )

    except Exception as e:
        return (emiten, label_type, window, False, f"Error: {str(e)}", None)


def main():
    parser = argparse.ArgumentParser(
        description="Generate stock forecasts using trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Forecast with default settings
  python -m pipeline.04_forecast --windows 5,10,20 --label_types median_gain,max_loss --min_test_gini 0.3 --workers 10
  
  # Forecast without Gini filter
  python -m pipeline.04_forecast --windows 5,10,20 --label_types median_gain,max_loss --workers 10
  
  # Forecast specific tickers
  python -m pipeline.04_forecast --tickers "BBCA,BBRI,TLKM" --windows 5,10 --label_types median_gain --workers 4
        """,
    )

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
        help="Comma-separated forecast windows in days (default: 5,10,20)",
    )

    parser.add_argument(
        "--min_test_gini",
        type=float,
        default=None,
        help="Minimum test Gini coefficient for model filtering (default: None, use all models)",
    )

    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help='Comma-separated list of specific tickers to forecast (e.g., "BBCA,BBRI,TLKM")',
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )

    args = parser.parse_args()

    # Parse label types and windows
    label_types = [lt.strip() for lt in args.label_types.split(",")]
    windows = [int(w.strip()) for w in args.windows.split(",")]

    # Validate label types
    valid_label_types = ["linear_trend", "median_gain", "max_loss"]
    for label_type in label_types:
        if label_type not in valid_label_types:
            print(f"ERROR: Invalid label type: {label_type}")
            print(f"   Valid types: {', '.join(valid_label_types)}")
            return

    # Ensure output directories exist
    ensure_directories_exist(label_types, windows)

    # Clear old forecast files to avoid duplicates
    print("\nClearing old forecast files...")
    clear_forecast_files(label_types, windows)

    # Get feature columns
    feature_columns = get_all_technical_indicators()
    print(f"Using {len(feature_columns)} technical indicators as features")

    # Determine which emiten to forecast
    if args.tickers:
        # Use specified tickers
        emiten_list = [ticker.strip() for ticker in args.tickers.split(",")]
        print(f"\nForecasting specified tickers: {', '.join(emiten_list)}")
    else:
        # Get filtered emiten based on model performance
        print("\nFinding emiten with models meeting criteria...")
        if args.min_test_gini is not None:
            print(f"   Min Test Gini: {args.min_test_gini}")
        else:
            print("   Min Test Gini: None (using all available models)")

        emiten_list = get_filtered_emiten_list(label_types, windows, args.min_test_gini)

        if not emiten_list:
            print("ERROR: No emiten found meeting the criteria")
            return

        print(f"Found {len(emiten_list)} emiten meeting criteria")

    # Prepare arguments for parallel processing
    forecast_tasks = []
    for emiten in emiten_list:
        for label_type in label_types:
            for window in windows:
                forecast_tasks.append((emiten, label_type, window, feature_columns))

    total_tasks = len(forecast_tasks)
    print(
        f"\nStarting forecasts for {len(emiten_list)} emiten × {len(label_types)} label types × {len(windows)} windows = {total_tasks} tasks"
    )
    print(f"Using {args.workers} parallel workers\n")

    # Process forecasts in parallel
    successful = 0
    failed = 0

    with Pool(processes=args.workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_forecast, forecast_tasks),
                total=total_tasks,
                desc="Generating forecasts",
            )
        )

    # Summarize results
    print("\n" + "=" * 80)
    print("FORECAST SUMMARY")
    print("=" * 80)

    # Group forecasts by label_type and window for batch saving
    forecasts_by_key = {}

    for emiten, label_type, window, success, message, forecast_data in results:
        if success and forecast_data is not None:
            successful += 1
            key = (label_type, window)
            if key not in forecasts_by_key:
                forecasts_by_key[key] = []
            forecasts_by_key[key].append(forecast_data)
        else:
            failed += 1
            if failed <= 10:  # Show first 10 failures
                print(f"FAILED: {emiten} ({label_type}, {window}dd): {message}")

    if failed > 10:
        print(f"   ... and {failed - 10} more failures")

    # Save all forecasts at once (avoids race conditions)
    if forecasts_by_key:
        print("\nSaving forecasts...")
        for (label_type, window), forecast_list in forecasts_by_key.items():
            # Convert list of dicts to DataFrame
            rows = []
            for forecast_data in forecast_list:
                row = {
                    "Kode": forecast_data["Kode"],
                    "Date": forecast_data["Date"],
                    forecast_data["forecast_column"]: forecast_data["forecast_value"],
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            save_forecast(df, label_type, window)

    print(f"\nSuccessful: {successful}/{total_tasks}")
    print(f"Failed: {failed}/{total_tasks}")

    if successful > 0:
        print(
            "\nForecasts saved to: data/stock/04_forecast/{label_type}/{window}dd.csv"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
