
import pandas as pd
import numpy as np

# Membership Plans Trading Strategy
def membership_plans_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate monthly returns
    stock_df['Monthly Return'] = stock_df['Close'].pct_change()
    
    # Define the conditions for 'long', 'short', and 'neutral' signals based on hypothetical anomalies
    signals['signal'] = 'neutral'
    signals.loc[(stock_df['Monthly Return'] > 0) & (stock_df['Monthly Return'].shift(1) <= 0), 'signal'] = 'long'
    signals.loc[(stock_df['Monthly Return'] < 0) & (stock_df['Monthly Return'].shift(1) >= 0), 'signal'] = 'short'
    
    return signals[['signal']]
