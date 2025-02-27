import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from threebarinsidebarle.py
def three_bar_inside_bar_le(ohlcv_df):
    signals = pd.DataFrame(index=ohlcv_df.index)
    # Define the conditions for the three bar inside bar pattern
    condition1 = ohlcv_df['Close'].shift(2) < ohlcv_df['Close'].shift(1)
    condition2 = ohlcv_df['High'].shift(1) > ohlcv_df['High']
    condition3 = ohlcv_df['Low'].shift(1) < ohlcv_df['Low']
    condition4 = ohlcv_df['Close'].shift(1) < ohlcv_df['Close']
    
    # Aggregate all conditions
    conditions = condition1 & condition2 & condition3 & condition4

    # Generate the long entry signals
    signals['three_bar_inside_bar_buy_signal'] = np.where(conditions.shift(-1), 0, 0)

    return signals
