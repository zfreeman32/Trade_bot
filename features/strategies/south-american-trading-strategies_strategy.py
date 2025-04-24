
import pandas as pd
import numpy as np

# Membership Plans Strategy
def membership_plan_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Generating a dummy strategy based on the concept of monthly strategies being published
    # For simplicity, we'll assume 'Close' prices of a stock are utilized
    
    signals['returns'] = stock_df['Close'].pct_change()
    
    # Create a signal column based on simple return thresholds
    signals['signal'] = 'neutral'
    
    # Assuming a simplified condition for 'long' and 'short' signals is based on returns
    signals.loc[signals['returns'] > 0.02, 'signal'] = 'long'    # Buy if returns > 2%
    signals.loc[signals['returns'] < -0.02, 'signal'] = 'short'  # Sell if returns < -2%
    
    return signals[['signal']]
