import numpy as np
import pandas as pd
from stock_indicators import indicators

def calculate_ehler_fisher_transform(prepared_data):
    result = indicators.get_fisher_transform(prepared_data)
    result_df = pd.DataFrame({
        'Date': [val.date for val in result],
        'Fisher Transform': [val.fisher for val in result],
        'Fisher Transform Trigger': [val.trigger for val in result]
    })

    result_df['Fisher Up trend'] = (result_df['Fisher Transform'] >= 2).astype(int)
    result_df['Fisher Down Trend'] = (result_df['Fisher Transform'] <= -2).astype(int)
    result_df['Fisher Spread'] = (
        result_df['Fisher Transform'] - result_df['Fisher Transform Trigger']
    )
    result_df['Fisher Change 5D'] = result_df['Fisher Transform'].diff(5)

    previous_position = result_df['Fisher Transform'].values >= result_df['Fisher Transform Trigger'].values
    result_df['Fisher Reversal'] = [np.nan] + (previous_position[:-1] != previous_position[1:]).astype(int).tolist()

    return result_df.set_index('Date')
