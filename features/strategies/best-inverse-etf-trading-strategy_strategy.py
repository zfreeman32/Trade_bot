
import pandas as pd
import numpy as np

# Inverse ETF Trading Strategy
def inverse_etf_signals(etf_df, moving_average_window=20):
    """
    Generate trading signals for an Inverse ETF Trading Strategy.

    Parameters:
    etf_df (DataFrame): DataFrame containing the historical price data of the inverse ETF with 'Close' prices.
    moving_average_window (int): The window size for the moving average to determine market trends.

    Returns:
    DataFrame: A DataFrame with trading signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=etf_df.index)
    
    # Calculate moving average
    signals['Moving_Average'] = etf_df['Close'].rolling(window=moving_average_window).mean()
    
    # Determine signals based on price action relative to the moving average
    signals['Signal'] = 'neutral'
    signals.loc[(etf_df['Close'] > signals['Moving_Average']), 'Signal'] = 'short'  # Expecting a decline
    signals.loc[(etf_df['Close'] < signals['Moving_Average']), 'Signal'] = 'long'   # Expecting a further decline
    
    return signals[['Signal']]
