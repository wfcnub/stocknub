import os
import argparse

from selectTickerToProcess.main import select_ticker_to_process

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline Description: Select Ticker to Process Based on The Recent Average Valuation"
    )
    parser.add_argument(
        "--ohlcv_folder_path",
        type=str,
        default="data/stock/OHLCV",
        help="Directory path where the CSV files will be saved (default: data/stock/OHLCV)",
    )
    parser.add_argument(
        "--perc_ticker_in_industry",
        type=float,
        default=0.85,
        help="The percentile threshold of the total ticker to select from each industry",
    )
    parser.add_argument(
        "--perc_ticker",
        type=float,
        default=0.9,
        help="The percentile threshold of the total ticker to select from the entire IHSG",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PIPELINE DESCRIPTION: SELECT TICKER TO PROCESS")
    print("=" * 80)

    print('Selecting all ticker that meet the following criterias')
    print(f'    Selecting all ticker in an industry having an average valution greater then the {args.perc_ticker_in_industry} percentile')
    print(f'    Selecting all ticker having an average valution greater then the {args.perc_ticker} percentile')
    print()

    selected_ticker_df = select_ticker_to_process(args.ohlcv_folder_path, args.perc_ticker_in_industry, args.perc_ticker)

    print(f'Selected a total of {len(selected_ticker_df)} tickers, with a breakdown for each industry as follows')
    selected_industry_count = selected_ticker_df['Industry'].value_counts()
    for industry, count in zip(selected_industry_count.index, selected_industry_count.values):
        print(f' - Selected a total of {count} tickers from the {industry} sector')
    print()

    save_path = 'data/selected_ticker_and_industry_list.csv'
    selected_ticker_df.to_csv(save_path, index=False)
    print(f'Successfully saved the data to {save_path}')

    print("=" * 80)