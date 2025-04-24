
import pandas as pd
import numpy as np

# Cotton Trading Strategy
def cotton_trading_signals(cotton_df):
    signals = pd.DataFrame(index=cotton_df.index)
    
    # Calculate the 10-day moving average
    signals['SMA10'] = cotton_df['Close'].rolling(window=10).mean()
    
    # Calculate the 30-day moving average
    signals['SMA30'] = cotton_df['Close'].rolling(window=30).mean()
    
    # Generate trading signals
    signals['signal'] = 0
    signals.loc[signals['SMA10'] > signals['SMA30'], 'signal'] = 1  # Long signal
    signals.loc[signals['SMA10'] < signals['SMA30'], 'signal'] = -1  # Short signal

    # Create 'long', 'short', 'neutral' labels
    signals['trading_signal'] = 'neutral'
    signals.loc[signals['signal'] == 1, 'trading_signal'] = 'long'
    signals.loc[signals['signal'] == -1, 'trading_signal'] = 'short'
    
    return signals[['trading_signal']]
