
import pandas as pd
import numpy as np

# Breakout Trading Strategy
def breakout_signals(stock_df, breakout_threshold=0.05):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate resistance and support levels
    stock_df['Resistance'] = stock_df['High'].rolling(window=20).max()
    stock_df['Support'] = stock_df['Low'].rolling(window=20).min()

    # Initialize signals
    signals['signal'] = 'neutral'
    
    # Generate signals based on breakout criteria
    signals.loc[stock_df['Close'] > stock_df['Resistance'] * (1 + breakout_threshold), 'signal'] = 'long'
    signals.loc[stock_df['Close'] < stock_df['Support'] * (1 - breakout_threshold), 'signal'] = 'short'
    
    # Forward fill signals to maintain position until criteria changes
    signals['signal'].fillna(method='ffill', inplace=True)
    
    return signals
