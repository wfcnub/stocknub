"""
Pipeline Step 4: Forecast Stock Performance

This script generates forecasts using trained models from step 3.
It reads models from data/stock/03_model/{label_type}/*.pkl and outputs:
- Forecast results to data/stock/04_forecast_stocks/{label_type}/{window}dd.csv

The script:
- Filters stocks by minimum test Gini performance
- Reads technical indicators from existing CSV files (step 1 output)
- Applies trained models to generate probability predictions
- Saves forecasts for the most recent date

Usage:
    # Forecast with default settings (all models with min_test_gini >= 0.3)
    python -m pipeline.04_forecast_stocks--windows 5,10,20 --label_types median_gain,max_loss --min_test_gini 0.3 --workers 10

    # Forecast without Gini filter (use all available models)
    python -m pipeline.04_forecast_stocks --windows 5,10,20 --label_types median_gain,max_loss --workers 10

    # Forecast specific tickers only
    python -m pipeline.04_forecast_stocks --tickers "ADRO,ADMR" --windows 5,10 --label_types median_gain --workers 10
"""

import argparse
import pandas as pd

from warnings import simplefilter
simplefilter(action="ignore")

from forecastStocks.main import forecast_stocks
from forecastStocks.helper import _save_forecast

def main():
    parser = argparse.ArgumentParser(
        description="Generate stock forecasts using trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    label_types = [lt.strip() for lt in args.label_types.split(",")]
    windows = [int(w.strip()) for w in args.windows.split(",")]

    results, successful, failed, total_tasks = forecast_stocks(label_types, windows, args.min_test_gini, args.workers, args.tickers)

    print("\n" + "=" * 80)
    print("FORECAST SUMMARY")
    print("=" * 80)

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
            if failed <= 10:
                print(f"FAILED: {emiten} ({label_type}, {window}dd): {message}")

    if failed > 10:
        print(f"   ... and {failed - 10} more failures")

    if forecasts_by_key:
        print("\nSaving forecasts...")
        for (label_type, window), forecast_list in forecasts_by_key.items():
            rows = []
            for forecast_data in forecast_list:
                row = {
                    "Kode": forecast_data["Kode"],
                    "Date": forecast_data["Date"],
                    forecast_data["forecast_column"]: forecast_data["forecast_value"],
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            _save_forecast(df, label_type, window)

    print(f"\nSuccessful: {successful}/{total_tasks}")
    print(f"Failed: {failed}/{total_tasks}")

    if successful > 0:
        print(
            "\nForecasts saved to: data/stock/04_forecast/{label_type}/{window}dd.csv"
        )

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()