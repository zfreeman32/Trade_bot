
import pandas as pd
import numpy as np
from ta import momentum

# Schaff Trend Cycle (STC) Strategy
def stc_signals(stock_df, cycle_length=10, long_cycle=23, short_cycle=50):
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate STC
    stc = momentum.STC(stock_df['Close'], window_slow=long_cycle, window_fast=short_cycle, cycle_length=cycle_length)
    signals['STC'] = stc.stc()
    
    # Initialize signal column
    signals['stc_signal'] = 'neutral'
    
    # Generate long and short signals based on STC
    signals.loc[(signals['STC'] < 25) & (signals['STC'].shift(1) >= 25), 'stc_signal'] = 'long'  # Buy signal when STC crosses above 25
    signals.loc[(signals['STC'] > 75) & (signals['STC'].shift(1) <= 75), 'stc_signal'] = 'short'  # Sell signal when STC crosses below 75
    
    # Dropping the STC column if not needed
    signals.drop(['STC'], axis=1, inplace=True)
    
    return signals
