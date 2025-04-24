
import pandas as pd
import numpy as np

# Membership-Based Strategy
def membership_based_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Simple moving average for membership strategy
    signals['SMA_10'] = stock_df['Close'].rolling(window=10).mean()
    signals['SMA_30'] = stock_df['Close'].rolling(window=30).mean()
    
    # Generating signals based on moving averages
    signals['signal'] = 'neutral'
    signals.loc[(signals['SMA_10'] > signals['SMA_30']), 'signal'] = 'long'
    signals.loc[(signals['SMA_10'] < signals['SMA_30']), 'signal'] = 'short'
    
    return signals[['signal']]
