
import pandas as pd
import numpy as np

# Membership Plans Trading Strategy
def membership_plan_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Simulating trading logic based on hypothetical strategy parameters
    # Assume we have a specific condition to buy or sell based on last month's close prices
    signals['last_close'] = stock_df['Close'].shift(1)
    signals['current_close'] = stock_df['Close']
    
    # Create long signals when current close is higher than last close
    signals['signal'] = 'neutral'
    signals.loc[signals['current_close'] > signals['last_close'], 'signal'] = 'long'
    
    # Create short signals when current close is lower than last close
    signals.loc[signals['current_close'] < signals['last_close'], 'signal'] = 'short'
    
    return signals[['signal']]
