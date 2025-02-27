import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from gapmomentumsystem.py
def gap_momentum_signals(data_df, length=14, signal_length=9, full_range=False):
    signals = pd.DataFrame(index=data_df.index)
    # calculate the gap prices
    gaps = data_df['Open'] - data_df['Close'].shift()
    # calculate the signal line 
    signals['gap_avg'] = gaps.rolling(window=length).mean() 
    # Calculate the signal based on the gap average
    signals['gap_momentum_buy_signal'] = 0
    signals['gap_momentum_sell_signal'] = 0
    signals.loc[(signals['gap_avg'] > signals['gap_avg'].shift()), 'gap_momentum_buy_signal'] = 1
    signals.loc[(signals['gap_avg'] < signals['gap_avg'].shift()), 'gap_momentum_sell_signal'] = 1
    signals.drop(['gap_avg'], axis=1, inplace=True)
    return signals
