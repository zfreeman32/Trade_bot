
import pandas as pd
import numpy as np

# Membership Plans Strategy
def membership_plan_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Create conditions for signals based on membership description
    signals['membership_signal'] = 'neutral'
    
    # Define two membership levels
    platinum_threshold = 1990
    gold_threshold = 990
    
    # Simulated check for membership fee paid (you can replace this with actual logic)
    membership_fee = np.random.choice([platinum_threshold, gold_threshold], len(stock_df))  # Simulating membership
    
    # Buy signal if the member is platinum
    signals.loc[membership_fee == platinum_threshold, 'membership_signal'] = 'long'
    
    # Short signal if the member is gold (but not in July or December)
    signals.loc[(membership_fee == gold_threshold) & 
                 (stock_df.index.month != 7) & 
                 (stock_df.index.month != 12), 'membership_signal'] = 'short'

    return signals
