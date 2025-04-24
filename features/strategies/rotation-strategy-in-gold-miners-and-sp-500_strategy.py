
import pandas as pd
import numpy as np

# Membership Plans Trading Strategy
def membership_plans_signals(stock_df, entry_threshold=0.02, exit_threshold=0.01):
    """
    Generates trading signals based on the Membership Plans trading strategy.
    
    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock data with 'Close' prices.
    entry_threshold (float): The threshold for entering a long position.
    exit_threshold (float): The threshold for exiting a long position.
    
    Returns:
    pd.DataFrame: DataFrame with trading signals ('long', 'short', 'neutral').
    """
    
    signals = pd.DataFrame(index=stock_df.index)
    signals['Close'] = stock_df['Close']
    
    # Calculate daily returns
    signals['Returns'] = signals['Close'].pct_change()
    
    # Initialize signals
    signals['signal'] = 'neutral'
    
    # Define conditions for long and short signals
    signals.loc[signals['Returns'] > entry_threshold, 'signal'] = 'long'
    signals.loc[signals['Returns'] < -exit_threshold, 'signal'] = 'short'
    
    # Forward fill signals to maintain signals until the next change
    signals['signal'] = signals['signal'].replace({'neutral': np.nan}).ffill()
    
    return signals[['signal']]
