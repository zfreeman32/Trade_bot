
import pandas as pd
import numpy as np

# Financial Services Sector Trading Strategy
def financial_services_signals(stock_df, short_window=20, long_window=50):
    """
    Generates trading signals for the Financial Services sector based on simple moving averages.
    
    Parameters:
    stock_df (pd.DataFrame): DataFrame containing at least 'Close' prices of the stock.
    short_window (int): The short moving average window.
    long_window (int): The long moving average window.
    
    Returns:
    pd.DataFrame: DataFrame with columns 'short_ma', 'long_ma', and 'signal' indicating 'long', 'short', or 'neutral'.
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate moving averages
    signals['short_ma'] = stock_df['Close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_ma'] = stock_df['Close'].rolling(window=long_window, min_periods=1).mean()
    
    # Create signals
    signals['signal'] = 'neutral'
    signals.loc[(signals['short_ma'] > signals['long_ma']), 'signal'] = 'long'
    signals.loc[(signals['short_ma'] < signals['long_ma']), 'signal'] = 'short'
    
    return signals[['short_ma', 'long_ma', 'signal']]
