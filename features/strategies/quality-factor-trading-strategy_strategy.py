
import pandas as pd
import numpy as np

# Membership Plans Trading Strategy
def membership_plans_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['membership_strategy'] = 'neutral'
    
    # Calculate moving averages for Gold and Platinum membership strategies
    signals['short_mavg'] = stock_df['Close'].rolling(window=10).mean()  # Short-term moving average
    signals['long_mavg'] = stock_df['Close'].rolling(window=30).mean()   # Long-term moving average
    
    # Generate signals based on moving averages
    signals.loc[(signals['short_mavg'] > signals['long_mavg']), 'membership_strategy'] = 'long'
    signals.loc[(signals['short_mavg'] < signals['long_mavg']), 'membership_strategy'] = 'short'
    
    # Drop the moving averages from the signals DataFrame
    signals.drop(['short_mavg', 'long_mavg'], axis=1, inplace=True)
    return signals
