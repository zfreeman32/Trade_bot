
import pandas as pd
import numpy as np

# Membership Strategy Trading Signals
def membership_strategy_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['membership_signal'] = 'neutral'
    
    # Example conditions for generating trading signals
    # This could be based on simple moving averages (SMA) or any other criteria
    short_window = 10
    long_window = 30

    # Calculate moving averages
    signals['short_mavg'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = stock_df['Close'].rolling(window=long_window).mean()

    # Generate signals based on moving average crossovers
    signals.loc[(signals['short_mavg'] > signals['long_mavg']), 'membership_signal'] = 'long'
    signals.loc[(signals['short_mavg'] < signals['long_mavg']), 'membership_signal'] = 'short'
    
    # Drop the moving averages columns to clean up DataFrame
    signals.drop(['short_mavg', 'long_mavg'], axis=1, inplace=True)
    
    return signals
