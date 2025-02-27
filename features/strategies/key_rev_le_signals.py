import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from keyrevle.py
def key_rev_le_signals(stock_df, length=5):
    """
    Generate KeyRevLE strategy signals.
    :param stock_df: OHLCV dataset.
    :param length: The number of preceding bars whose Low prices are compared to the current Low.
    :return: DataFrame with 'key_rev_le_signal' column.
    """
    signals = pd.DataFrame(index=stock_df.index)
    signals['key_rev_buy_signal'] = 0

    for i in range(length, len(stock_df) - 1):  # Avoid index error at the end
        if stock_df['Low'].iloc[i] < stock_df['Low'].iloc[i-length:i].min() and \
           stock_df['Close'].iloc[i] > stock_df['Close'].iloc[i-1]:
            signals.loc[signals.index[i+1], 'key_rev_buy_signal'] = 1  # Fix using `.loc[]`
            
    return signals
