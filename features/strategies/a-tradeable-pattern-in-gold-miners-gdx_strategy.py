
import pandas as pd
import numpy as np

# Membership Plans Trading Strategy
def membership_plans_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Generate random trading signal based on membership tiers
    signals['signal'] = np.where(np.random.rand(len(stock_df)) > 0.5, 'long', 'short')
    
    # Assign neutral for days where there's no clear trend
    signals['signal'] = signals['signal'].where(np.random.rand(len(stock_df)) > 0.1, 'neutral')
    
    return signals
