import math
import numpy as np
import pandas as pd

def _calculate_foreign_moving_average(data, window):
    return data['Net Foreign'].rolling(window=10) \
                .mean() \
                .apply(lambda val: np.nan if math.isnan(val) else val < 1)

def calculate_foreign_moving_average(data):
    data.fillna(0, inplace=True) # DELETE LATER
    data['Net Foreign'] = data['Foreign Buy'] - data['Foreign Sell']

    data['MA10 Net Foreign'] = _calculate_foreign_moving_average(data, 10)
    data['MA30 Net Foreign'] = _calculate_foreign_moving_average(data, 30)
    data['MA60 Net Foreign'] = _calculate_foreign_moving_average(data, 60)

    result_df = data[['MA10 Net Foreign', 'MA30 Net Foreign', 'MA60 Net Foreign']]

    return result_df

def _calculate_foreign_accumulation(data, window):
    return data['Net Foreign'].rolling(window=10) \
                .sum() \
                .apply(lambda val: np.nan if math.isnan(val) else val < 1)

def calculate_foreign_accumulation(data):
    data.fillna(0, inplace=True) # DELETE LATER
    data['Net Foreign'] = data['Foreign Buy'] - data['Foreign Sell']

    data['ACUM10 Net Foreign'] = _calculate_foreign_accumulation(data, 10)
    data['ACUM30 Net Foreign'] = _calculate_foreign_accumulation(data, 30)
    data['ACUM60 Net Foreign'] = _calculate_foreign_accumulation(data, 60)

    result_df = data[['ACUM10 Net Foreign', 'ACUM30 Net Foreign', 'ACUM60 Net Foreign']]

    return result_df    