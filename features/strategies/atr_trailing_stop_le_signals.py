import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from atrtrailingstople.py
def atr_trailing_stop_le_signals(stock_df, atr_period=14, atr_factor=3):
    close = stock_df['Close']
    high = stock_df['High']
    low = stock_df['Low']
    
    atr = volatility.AverageTrueRange(high, low, close, window=atr_period).average_true_range()
    atr_trailing_stop = close - atr_factor * atr

    signals = pd.DataFrame(index=stock_df.index)
    signals['atr_trailing_stop'] = atr_trailing_stop
    signals['atr_trailing_stop_le_buy_signal'] = np.where(
        (close.shift(1) <= signals['atr_trailing_stop'].shift(1)) &  # Previous close was below ATR
        (close > signals['atr_trailing_stop']),  # Current close crosses above ATR
        1,  # Signal triggered
        0   # No signal
    )

    signals.drop(['atr_trailing_stop'], axis=1, inplace=True)

    return signals
