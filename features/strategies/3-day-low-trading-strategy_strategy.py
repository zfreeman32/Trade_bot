
import pandas as pd
import numpy as np

# Membership Plans Trading Strategy
def membership_plans_signals(stock_df, threshold=0.05):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate monthly returns
    stock_df['Monthly_Return'] = stock_df['Close'].pct_change(periods=21)  # Approximating 21 trading days in a month
    
    # Define thresholds for signals
    signals['signal'] = 'neutral'
    
    # Generate Buy signal if monthly return exceeds the threshold
    signals.loc[stock_df['Monthly_Return'] > threshold, 'signal'] = 'long'
    
    # Generate Sell signal if monthly return is below the negative threshold
    signals.loc[stock_df['Monthly_Return'] < -threshold, 'signal'] = 'short'
    
    return signals
