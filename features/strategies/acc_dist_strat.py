import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from accumulationdistributionstrat.py
def acc_dist_strat(df, length = 4, factor = 0.75, vol_ratio= 1 , vol_avg_length = 4, mode='high-low'):
    
    signals = pd.DataFrame(index=df.index)
    
    if mode == 'high-low':
        df['range'] = df['High'] - df['Low']
    else:
        average_true_range = volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=length)
        df['range'] = average_true_range.average_true_range()
        
    df['last_range'] = df['range'].shift(length)
    df['range_ratio'] = df['range'] / df['last_range']
    
    df['vol_av'] = df['Volume'].rolling(window=vol_avg_length).mean()
    df['vol_last_av'] = df['vol_av'].shift(vol_avg_length)
    df['vol_ratio_av'] = df['vol_av'] / df['vol_last_av']
    
    df['higher_low'] = df['Low'] > df['Low'].shift().rolling(window=12).min()
    df['break_high'] = df['Close'] > df['High'].shift().rolling(window=12).max()
    df['fall_below'] = df['Close'] < df['Low'].shift().rolling(window=12).min()
    
    # Buy signals
    signals['acc_dist_buy_signal'] = np.where(
        (df['range_ratio'] < factor) &
        df['higher_low'] &
        df['break_high'] &
        (df['vol_ratio_av'] > vol_ratio),
        1, 0)
    
    # Sell signals
    signals['acc_dist_sell_signal'] = np.where(
        df['fall_below'],
        1, 0)
    return signals
