
import pandas as pd
import numpy as np

# Williams %R Trading Strategy
def williams_r_signals(stock_df, lookback_period=5, overbought=-20, oversold=-80):
    """
    Generates trading signals based on the Williams %R indicator.
    
    Parameters:
        stock_df (pd.DataFrame): DataFrame containing stock price data with a 'Close' column.
        lookback_period (int): The lookback period for calculating Williams %R.
        overbought (float): The threshold for the overbought condition.
        oversold (float): The threshold for the oversold condition.
    
    Returns:
        pd.DataFrame: DataFrame with 'williams_r' and 'signal' columns.
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the highest high and lowest low over the lookback period
    highest_high = stock_df['Close'].rolling(window=lookback_period).max()
    lowest_low = stock_df['Close'].rolling(window=lookback_period).min()
    
    # Calculate Williams %R
    signals['williams_r'] = (highest_high - stock_df['Close']) / (highest_high - lowest_low) * -100
    
    # Generate signals based on Williams %R
    signals['signal'] = 'neutral'
    signals.loc[(signals['williams_r'] < overbought), 'signal'] = 'short'  # overbought territory
    signals.loc[(signals['williams_r'] > oversold), 'signal'] = 'long'    # oversold territory
    
    return signals[['williams_r', 'signal']]
