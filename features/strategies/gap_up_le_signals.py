import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from gapuple.py
def gap_up_le_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['gap_up_le_buy_signal'] = 0
    
    # Identify rows where the current Low is higher than the previous High
    gap_up = (stock_df['Low'] > stock_df['High'].shift(1))
    
    # Shift the signal to the next row
    shifted_gap_up = gap_up.shift(-1).fillna(False).astype(bool)  # Fill NaN with False to avoid indexer errors
    
    # Generate Long Entry signal for the next bar
    signals.loc[shifted_gap_up, 'gap_up_le_buy_signal'] = 1
    
    return signals
