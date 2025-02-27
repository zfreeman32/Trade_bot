import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# STC
def stc_signals(stock_df, window_slow=50, window_fast=23, cycle=10, smooth1=3, smooth2=3):
    """
    Computes Schaff Trend Cycle (STC) signals.

    Returns:
    A DataFrame with 'stc_signal' and 'stc_direction'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate STC
    stc_indicator = trend.STCIndicator(stock_df['Close'], window_slow, window_fast, cycle, smooth1, smooth2)
    signals['STC'] = stc_indicator.stc()

    # Determine overbought/oversold conditions
    signals['stc_overbought_signal'] = 0
    signals['stc_oversold_signal'] = 0
    signals.loc[signals['STC'] > 75, 'stc_overbought_signal'] = 1
    signals.loc[signals['STC'] < 25, 'stc_oversold_signal'] = 1

    return signals
