import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

from utils import paths

from selectTickerToProcess.main import select_ticker_to_process

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline Description: Select Ticker to Process Based on Fundamental Analysis"
    )

    parser.add_argument(
        "--ohlcv_folder_path",
        type=str,
        default=str(paths.get_ohlcv_dir()),
        help="Directory path where the CSV files are stored",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PIPELINE DESCRIPTION: SELECT TOP 150 FUNDAMENTAL TICKERS")
    print("=" * 80)

    print('Evaluating and ranking tickers across the market based on fundamental upside potential.')
    print()

    all_tickers = [file.stem for file in Path(args.ohlcv_folder_path).rglob("*.csv")]
    
    print(f"Starting multiprocessing fetching for {len(all_tickers)} tickers...")
    
    results = []
    
    num_processes = min(cpu_count(), 16)
    
    with Pool(processes=num_processes) as pool:
        for res in tqdm(pool.imap_unordered(select_ticker_to_process, all_tickers), total=len(all_tickers)):
            if res is not None and not res.empty:
                results.append(res)
                
    if results:
        selected_ticker_df = pd.concat(results, ignore_index=True)
    else:
        selected_ticker_df = pd.DataFrame()

    if not selected_ticker_df.empty:
        if 'avg_value_traded' in selected_ticker_df.columns:
            q75_threshold = selected_ticker_df['avg_value_traded'].quantile(0.75)
            print(f"Applying Q75 liquidity threshold: Dropping stocks with avg daily value traded < {q75_threshold:,.0f}")
            selected_ticker_df = selected_ticker_df[selected_ticker_df['avg_value_traded'] >= q75_threshold]

        selected_ticker_df = selected_ticker_df.sort_values(by="fundamental_score", ascending=False).head(150)
        
        try:
            ticker_industry_df = pd.read_csv(paths.get_ticker_and_industry_list_path())
            selected_ticker_df = pd.merge(
                ticker_industry_df,
                selected_ticker_df,
                on='Ticker',
                how='inner'
            )
        except Exception as e:
            print(f"Warning: Could not merge with industry data: {e}")
            if 'Industry' not in selected_ticker_df.columns:
                selected_ticker_df['Industry'] = 'Unknown'

    print(f'\nSelected the top {len(selected_ticker_df)} fundamentally strongest tickers, with an industry breakdown as follows:')
    
    if not selected_ticker_df.empty and 'Industry' in selected_ticker_df.columns:
        selected_industry_count = selected_ticker_df['Industry'].value_counts()
        for industry, count in zip(selected_industry_count.index, selected_industry_count.values):
            print(f' - {count} tickers from the {industry} sector')
        print()

    save_path = paths.get_selected_ticker_and_industry_list_path()
    selected_ticker_df.to_csv(save_path, index=False)
    print(f'Successfully saved the top fundamental tickers to {save_path}')

    print("=" * 80)