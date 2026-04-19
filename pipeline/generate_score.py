import argparse
import sys
import gc

from generateScore.main import process_generate_score

def main():
    parser = argparse.ArgumentParser(
        description="Generate composite scores from forecast data and model performance."
    )
    
    parser.add_argument(
        "--windows",
        dest="windows",
        type=str,
        required=True,
        help="Comma-separated list of rolling windows (e.g., '5,10').",
    )
    
    args = parser.parse_args()

    windows = [w.strip() for w in args.windows.split(",")]

    for window in windows:
        if not window.endswith('dd'):
            window = f"{window}dd"
        print(f"\nProcessing generating score for {window}...")
        process_generate_score(window)
        gc.collect()

    print(f"\nProcessing generating score for all windows completed")

if __name__ == '__main__':
    sys.exit(main())
