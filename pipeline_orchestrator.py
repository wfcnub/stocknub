"""
Run the stocknub pipeline end-to-end or individual steps.

This script orchestrates the execution of pipeline steps in the correct order.
Each step can be run independently or as part of the full pipeline.

Usage:
    # Full run from scratch (fetches from 2021-01-01 to today)
    python pipeline_orchestrator.py --full

    # Update run (incremental update from last date to today)
    python pipeline_orchestrator.py --update-run

    # Run all steps with manual config
    python pipeline_orchestrator.py --all

    # Run specific steps
    python pipeline_orchestrator.py --steps 0 1 2
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from warnings import simplefilter

simplefilter("ignore")


PIPELINE_STEPS = {
    0: {
        "name": "Fetch Historical Data",
        "module": "pipeline.00_fetch_historical_data",
        "description": "Download OHLCV data from Yahoo Finance",
    },
    1: {
        "name": "Generate Technical Indicators",
        "module": "pipeline.01_prepare_technical_indicators",
        "description": "Calculate technical indicators from historical data",
    },
    2: {
        "name": "Generate Target Labels",
        "module": "pipeline.02_generate_labels",
        "description": "Create target labels for model training",
    },
    3: {
        "name": "Train Models",
        "module": "pipeline.03_train_models",
        "description": "Train machine learning models for stock prediction",
    },
    4: {
        "name": "Generate Forecasts",
        "module": "pipeline.04_forecast",
        "description": "Generate forecasts using trained models",
    },
}


def run_step(step_num, args):
    """Run a single pipeline step as a Python module."""
    step = PIPELINE_STEPS[step_num]

    print(f"\n{'=' * 80}")
    print(f"RUNNING STEP {step_num}: {step['name'].upper()}")
    print(f"{'=' * 80}\n")

    # Build command to run as module
    cmd = [sys.executable, "-m", step["module"]]

    # Add common arguments
    if args.workers:
        cmd.extend(["--workers", str(args.workers)])

    # Step-specific arguments
    if step_num == 0:
        # Fetch historical data arguments
        if args.update:
            cmd.extend(["--update", args.update])
        if args.start_date:
            cmd.extend(["--start_date", args.start_date])
        if args.end_date:
            cmd.extend(["--end_date", args.end_date])

    elif step_num == 1:
        # Technical indicators arguments
        if args.force:
            cmd.append("--force")

    elif step_num == 2:
        # Label generation arguments
        if args.label_types:
            cmd.extend(["--label_types", args.label_types])
        if args.windows:
            cmd.extend(["--windows", args.windows])
        if args.target_column:
            cmd.extend(["--target_column", args.target_column])
        if args.force:
            cmd.append("--force")

    elif step_num == 3:
        # Model training arguments
        if args.label_types:
            cmd.extend(["--label_types", args.label_types])
        if args.windows:
            cmd.extend(["--windows", args.windows])

    elif step_num == 4:
        # Forecast generation arguments
        if args.label_types:
            cmd.extend(["--label_types", args.label_types])
        if args.windows:
            cmd.extend(["--windows", args.windows])
        if args.min_test_gini is not None:
            cmd.extend(["--min_test_gini", str(args.min_test_gini)])

    # Execute the step
    try:
        subprocess.run(cmd, check=True)
        print(f"\nStep {step_num} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nStep {step_num} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the stocknub data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Step selection
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full run from scratch: fetch from 2021-01-01 to today, generate all indicators and labels",
    )
    parser.add_argument(
        "--update-run",
        action="store_true",
        help="Update run: incremental update from last date in historical data to today",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all pipeline steps in order (manual configuration)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        help="Specific steps to run (e.g., --steps 0 1)",
    )

    # Common arguments
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )

    # Step 0 (Fetch) specific arguments
    parser.add_argument(
        "--update",
        type=str,
        choices=["today", "yesterday"],
        help="Update mode for fetching data (Step 0 only)",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        help="Start date for fetching data in YYYY-MM-DD format (Step 0 only)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="End date for fetching data in YYYY-MM-DD format (Step 0 only)",
    )

    # Step 1 (Technical Indicators) specific arguments
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocess all data (Steps 1, 2)",
    )

    # Step 2 (Labels) specific arguments
    parser.add_argument(
        "--label_types",
        type=str,
        help="Comma-separated label types (Steps 2, 3, 4, e.g., median_gain,max_loss)",
    )
    parser.add_argument(
        "--windows",
        type=str,
        help="Comma-separated rolling windows (Steps 2, 3, 4, e.g., 5,10,20)",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        help="Target column for labels (Step 2 only, default: Close)",
    )

    # Step 4 (Forecast) specific arguments
    parser.add_argument(
        "--min_test_gini",
        type=float,
        default=None,
        help="Minimum test Gini coefficient for model filtering (Step 4 only)",
    )

    args = parser.parse_args()

    # Ensure all data directories exist
    Path("data/stock/00_historical").mkdir(parents=True, exist_ok=True)
    Path("data/stock/01_technical").mkdir(parents=True, exist_ok=True)
    Path("data/stock/02_label").mkdir(parents=True, exist_ok=True)
    Path("data/stock/03_model").mkdir(parents=True, exist_ok=True)
    Path("data/stock/04_forecast").mkdir(parents=True, exist_ok=True)

    # Determine which steps to run
    if args.full:
        # Full run mode: set defaults for complete pipeline from scratch
        steps_to_run = sorted(PIPELINE_STEPS.keys())
        args.start_date = "2021-01-01"
        args.end_date = datetime.now().strftime("%Y-%m-%d")
        args.update = None  # Don't use update mode, use start_date instead
        print("\nFULL RUN MODE: Fetching from 2021-01-01 to today, generating all data")
    elif args.update_run:
        # Update run mode: incremental update using built-in update logic
        steps_to_run = sorted(PIPELINE_STEPS.keys())
        args.update = "today"  # Use today as end date
        args.start_date = None  # Will read from last date in files
        args.end_date = None
        print("\nUPDATE RUN MODE: Incremental update from last date to today")
    elif args.all:
        steps_to_run = sorted(PIPELINE_STEPS.keys())
    elif args.steps:
        steps_to_run = sorted(args.steps)
        # Validate step numbers
        invalid_steps = [s for s in steps_to_run if s not in PIPELINE_STEPS]
        if invalid_steps:
            print(f"Invalid step numbers: {invalid_steps}")
            print(f"Available steps: {list(PIPELINE_STEPS.keys())}")
            return 1
    else:
        # Show available steps and usage
        print("Stocknub Data Pipeline")
        print("=" * 80)
        print("\nAvailable steps:")
        for num, step in PIPELINE_STEPS.items():
            print(f"  {num}: {step['name']}")
            print(f"     {step['description']}")
        print("\nUsage:")
        print("  python pipeline_orchestrator.py --all              # Run all steps")
        print(
            "  python pipeline_orchestrator.py --steps 0 1        # Run specific steps"
        )
        print("  python pipeline_orchestrator.py --help             # Show all options")
        return 0

    # Run the pipeline
    print("\n" + "=" * 80)
    print("STOCKNUB DATA PIPELINE")
    print("=" * 80)
    print(f"Steps to run: {steps_to_run}")
    print(f"Workers: {args.workers}")

    failed_steps = []
    for step_num in steps_to_run:
        success = run_step(step_num, args)
        if not success:
            failed_steps.append(step_num)
            print(f"\nStopping pipeline due to failure in step {step_num}")
            break

    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)

    if not failed_steps:
        print("All steps completed!")
        return 0
    else:
        print(f"Pipeline failed at step {failed_steps[0]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
