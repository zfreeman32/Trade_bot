import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from consbarsuple.py
def cons_bars_up_le_signals(stock_df, consec_bars_up=4, price='Close'):
    # Initialize the signal DataFrame with the same index as stock_df
    signals = pd.DataFrame(index=stock_df.index)
    signals['Signal'] = 0

    # Create a boolean mask indicating where price bars are consecutively increasing
    mask = stock_df[price].diff() > 0
    
    # Calculate the rolling sum of increasing bars
    rolling_sum = mask.rolling(window=consec_bars_up).sum()
    
    # Identify where the count of increasing bars exceeds the threshold
    signals.loc[rolling_sum >= consec_bars_up, 'Signal'] = 1

    # Take the difference of signals to capture the transition points
    signals['Signal'] = signals['Signal'].diff().fillna(0)

    # Initialize column to ensure integer type
    signals['cons_bars_up_le_buy_signal'] = 0  

    # Assign buy signal only at valid transition points
    signals.loc[signals['Signal'] > 0, 'cons_bars_up_le_buy_signal'] = 1

    # Drop intermediate column
    signals.drop(columns=['Signal'], inplace=True)

    # Ensure the output is integer type (avoids float 1.0 and NaN)
    signals['cons_bars_up_le_buy_signal'] = signals['cons_bars_up_le_buy_signal'].astype(int)

    return signals
