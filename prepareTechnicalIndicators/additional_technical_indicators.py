import math
import numpy as np
import pandas as pd

def _calculate_additional_technical_indicators(data, n_window):
    data['Net Foreign'] = data['Foreign Buy'] - data['Foreign Sell']
    data['Net Domestic'] = data['Volume'] - data['Foreign Buy']
    data['Average Price'] = data[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    data['Domestic Accumulation'] = data['Average Price'] * data['Net Domestic']
    data['Foreign Accumulation'] = data['Average Price'] * data['Net Foreign']

    data[f'Foreign {n_window} Days Accumulation'] = data['Foreign Accumulation'].rolling(window=n_window).sum()
    data[f'Domestic {n_window} Days Accumulation'] = data['Domestic Accumulation'].rolling(window=n_window).sum()
    
    data[f'Foreign {n_window} Days Volume'] = data['Net Foreign'].rolling(window=n_window).sum()
    data[f'Domestic {n_window} Days Volume'] = data['Net Domestic'].rolling(window=n_window).sum()
    
    data[f'Foreign {n_window} Days Average Price'] = data[f'Foreign {n_window} Days Accumulation'] / data[f'Foreign {n_window} Days Volume']
    data[f'Domestic {n_window} Days Average Price'] = data[f'Domestic {n_window} Days Accumulation'] / data[f'Domestic {n_window} Days Volume']
    
    data[f'Foreign to Domestic {n_window} Days Ownership'] = data[f'Foreign {n_window} Days Volume'] / (data[f'Foreign {n_window} Days Volume'] + data[f'Domestic {n_window} Days Volume'])
    data[f'Foreign {n_window} Days Average Price Over Current Price'] = (data[f'Foreign {n_window} Days Average Price'] - data['Average Price']) / data['Average Price']
    data[f'Domestic {n_window} Days Average Price Over Current Price'] = (data[f'Domestic {n_window} Days Average Price'] - data['Average Price']) / data['Average Price']
    
    returned_values = [
        data[f'Foreign to Domestic {n_window} Days Ownership'].values,
        data[f'Foreign {n_window} Days Average Price Over Current Price'].values,
        data[f'Domestic {n_window} Days Average Price Over Current Price'].values
    ]
    return returned_values

def calculate_additional_technical_indicators(data):
    for window in [15, 60, 90]:
        returned_values = _calculate_additional_technical_indicators(data.copy(), window)
        data[f'Foreign to Domestic {window} Days Ownership'] = returned_values[0]
        data[f'Foreign {window} Days Average Price Over Current Price'] = returned_values[1]
        data[f'Domestic {window} Days Average Price Over Current Price'] = returned_values[2]
        
    result_df = data[data.columns[-9:]]

    return result_df