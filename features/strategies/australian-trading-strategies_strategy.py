
import pandas as pd
import numpy as np

# Australian Market Trading Strategy
def australian_market_signals(stock_df, window=20):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate daily returns
    stock_df['returns'] = stock_df['Close'].pct_change()
    
    # Calculate moving average
    stock_df['moving_avg'] = stock_df['Close'].rolling(window=window).mean()
    
    # Generate signals
    signals['signal'] = 'neutral'
    signals.loc[(stock_df['Close'] > stock_df['moving_avg']), 'signal'] = 'long'
    signals.loc[(stock_df['Close'] < stock_df['moving_avg']), 'signal'] = 'short'
    
    return signals
