
import pandas as pd
import numpy as np

# Quantified Strategies Membership Trading Signals
def membership_strategy_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Placeholder for membership strategy rules, for example: 
    # - Check monthly returns 
    # - Evaluate market patterns
    
    # Example implementation of a hypothetical entry and exit strategy 
    # Replace this logic with the actual quantified rules as per the strategy description 
    stock_df['Monthly_Return'] = stock_df['Close'].pct_change(periods=30)  # Calculate monthly return
    stock_df['Signal'] = np.where(stock_df['Monthly_Return'] > 0.05, 'long', 
                                   np.where(stock_df['Monthly_Return'] < -0.05, 'short', 'neutral'))
    
    signals['membership_signal'] = stock_df['Signal']
    return signals
