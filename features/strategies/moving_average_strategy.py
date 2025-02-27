import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from movavgstrat.py
def moving_average_strategy(df, window=15, average_type='simple', mode='trend Following'):
    # Compute moving average
    if average_type == 'simple':
        df['moving_avg'] = df['Close'].rolling(window=window).mean()
    elif average_type == 'exponential':
        df['moving_avg'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    # Create buy and sell signal columns
    df['moving_average_buy_signal'] = np.where(df['Close'] > df['moving_avg'], 1, 0) if mode == 'trend Following' else np.where(df['Close'] < df['moving_avg'], 1, 0)
    df['moving_average_sell_signal'] = np.where(df['Close'] < df['moving_avg'], 1, 0) if mode == 'trend Following' else np.where(df['Close'] > df['moving_avg'], 1, 0)
    
    return df[['moving_average_buy_signal', 'moving_average_sell_signal']]
