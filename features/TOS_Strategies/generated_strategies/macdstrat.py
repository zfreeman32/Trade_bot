import pandas as pd
import numpy as np
from ta import trend
def macd_signals(df, fast_length=12, slow_length=26, macd_length=9):
    # Create a signals DataFrame
    signals = pd.DataFrame(index=df.index)

    # Create MACD indicator
    macd = trend.MACD(df['Close'], window_slow=slow_length, window_fast=fast_length, window_sign=macd_length)

    # Generate MACD line and signal line
    signals['MACD_line'] = macd.macd()
    signals['MACD_signal'] = macd.macd_signal()
    
    # Create a column for the macd strategy signal
    signals['macd_strat'] = 0.0

    # Create signals: When the MACD line crosses the signal line upward, buy the stock
    signals['macd_strat'][(signals['MACD_line'] > signals['MACD_signal']) & (signals['MACD_line'].shift(1) < signals['MACD_signal'].shift(1))] = 1.0
    
    # When the MACD line crosses the signal line downward, sell the stock
    signals['macd_strat'][(signals['MACD_line'] < signals['MACD_signal']) & (signals['MACD_line'].shift(1) > signals['MACD_signal'].shift(1))] = -1.0
    signals.drop(['MACD_line','MACD_signal'], axis=1, inplace=True)
    return signals

