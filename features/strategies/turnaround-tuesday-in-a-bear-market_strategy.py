
import pandas as pd
import numpy as np
from ta import momentum, volume

# Monthly Trading Strategy Based on Quantitative Edge
def membership_strategy_signals(stock_df):
    """
    Generates trading signals based on a quantifiable membership strategy.
    
    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock price data with a 'Close' column.
    
    Returns:
    pd.DataFrame: DataFrame with trading signals ('long', 'short', 'neutral').
    """
    
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate monthly returns
    stock_df['Monthly_Return'] = stock_df['Close'].pct_change(periods=21)  # Assuming 21 trading days in a month
    signals['Monthly_Return'] = stock_df['Monthly_Return']
    
    # Define conditions for trading signals
    signals['membership_signal'] = 'neutral'
    signals.loc[signals['Monthly_Return'] > 0, 'membership_signal'] = 'long'
    signals.loc[signals['Monthly_Return'] < 0, 'membership_signal'] = 'short'
    
    return signals[['membership_signal']]
