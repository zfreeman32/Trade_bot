
import pandas as pd
import numpy as np

# S&P 500 Interest Rate Strategy
def sp500_interest_rate_signals(sp500_df, interest_rate_df, sma_window=200):
    """
    Generate trading signals based on the S&P 500's performance relative to interest rates.
    
    Parameters:
    sp500_df (DataFrame): DataFrame containing S&P 500 price data with a 'Close' column.
    interest_rate_df (DataFrame): DataFrame containing interest rate data with a 'Yield' column.
    sma_window (int): Window size for calculating the simple moving average.
    
    Returns:
    DataFrame: DataFrame containing trading signals ('long', 'short', 'neutral').
    """
    
    # Calculate the SMA of the S&P 500
    sp500_df['SMA'] = sp500_df['Close'].rolling(window=sma_window).mean()
    
    # Create signals DataFrame
    signals = pd.DataFrame(index=sp500_df.index)
    
    # Prepare to generate signals
    signals['signal'] = 'neutral'
    
    # Generate long signal when SMA is above interest rate yield
    signals.loc[(sp500_df['SMA'] > interest_rate_df['Yield']), 'signal'] = 'long'
    
    # Generate short signal when SMA is below interest rate yield
    signals.loc[(sp500_df['SMA'] < interest_rate_df['Yield']), 'signal'] = 'short'
    
    return signals
