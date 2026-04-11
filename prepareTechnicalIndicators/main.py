import numpy as np
import pandas as pd
from pathlib import Path

from prepareTechnicalIndicators.all_technical_indicators import generate_all_technical_indicators

def process_single_ticker(args_tuple):
    """
    Read historical data, generate technical indicators, and save to technical indicators folder.

    Args:
        args_tuple: Tuple containing (ticker, ohlcv_folder_path, foreign_flow_non_regular_folder_path, technical_folder_path)

    Returns:
        Tuple of (ticker, success, message, num_new_rows)
    """
    ticker, ohlcv_folder_path, foreign_flow_non_regular_folder_path, technical_folder_path = args_tuple

    try:
        ohlcv_path = (Path(ohlcv_folder_path) / ticker).with_suffix('.csv')
        foreign_flow_non_regular_path = (Path(foreign_flow_non_regular_folder_path) / ticker).with_suffix('.csv')
        technical_path = (Path(technical_folder_path) / ticker).with_suffix('.csv')

        if not ohlcv_path.is_file():
            return (
                ticker, 
                False, 
                f"OHLCV data not found for {ticker}", 
                0
            )

        ohlcv_df = pd.read_csv(ohlcv_path)
        if ohlcv_df.empty:
            return (
                ticker, 
                False, 
                f"OHLCV data is empty for {ticker}", 
                0
            )

        if not foreign_flow_non_regular_path.is_file():
            return (
                ticker, 
                False, 
                f"Foreign flow and non-regular data not found for {ticker}", 
                0
            )

        foreign_flow_non_regular_df = pd.read_csv(foreign_flow_non_regular_path)
        if foreign_flow_non_regular_df.empty:
            return (
                ticker, 
                False, 
                f"Foreign flow and non-regular data is empty for {ticker}", 
                0
            )
        foreign_flow_non_regular_df = pd.merge(ohlcv_df, foreign_flow_non_regular_df, on='Date', how='left')

        close_variance = np.var(ohlcv_df["Close"].values)
        if close_variance < 1e-10:
            return (
                ticker,
                False,
                f"{ticker} - No price variation (likely suspended/delisted, variance={close_variance:.2e})",
                0,
            )

        ohlcv_df["Date"] = pd.to_datetime(ohlcv_df["Date"]).dt.strftime("%Y-%m-%d")
        foreign_flow_non_regular_df["Date"] = pd.to_datetime(foreign_flow_non_regular_df["Date"]).dt.strftime("%Y-%m-%d")

        technical_df = generate_all_technical_indicators(ohlcv_df, foreign_flow_non_regular_df)
        technical_df.reset_index(inplace=True)
        technical_df["Date"] = pd.to_datetime(technical_df["Date"]).dt.strftime("%Y-%m-%d")
        
        technical_df.to_csv(technical_path, index=False)
        num_rows = len(technical_df)
        return (
            ticker,
            True,
            f"Generated {num_rows} rows of technical indicators for {ticker}",
            num_rows,
        )

    except Exception as e:
        return (
            ticker, 
            False, 
            f"Error processing {ticker}: {str(e)}", 
            0
        )