
import pandas as pd
import numpy as np

# Range Expansion Index (REI) Strategy
def rei_signals(stock_df, lookback=14, overbought=60, oversold=-60):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the True High and True Low
    high = stock_df['High'].rolling(window=lookback).max()
    low = stock_df['Low'].rolling(window=lookback).min()
    
    # Calculate the Range Expansion Index (REI)
    rei = ((stock_df['Close'] - low) - (high - stock_df['Close'])) / (high - low) * 100
    signals['REI'] = rei
    
    # Generate signals
    signals['rei_signal'] = 'neutral'
    signals.loc[(signals['REI'] > overbought) & (signals['REI'].shift(1) <= overbought), 'rei_signal'] = 'short'
    signals.loc[(signals['REI'] < oversold) & (signals['REI'].shift(1) >= oversold), 'rei_signal'] = 'long'
    
    return signals
