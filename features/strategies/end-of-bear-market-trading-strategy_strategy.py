
import pandas as pd
import numpy as np

# Membership Plan Trading Strategy
def membership_plan_signals(stock_df, period=14):
    """
    Generates trading signals based on monthly trading strategies from membership plans.
    
    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock prices with DateTime index and a 'Close' column.
    period (int): The number of periods for calculating moving averages.

    Returns:
    pd.DataFrame: DataFrame containing trading signals ('long', 'short', 'neutral') based on the strategy.
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate moving averages
    signals['MA_Fast'] = stock_df['Close'].rolling(window=period).mean()
    signals['MA_Slow'] = stock_df['Close'].rolling(window=period * 2).mean()
    
    # Initialize the signal column
    signals['membership_signal'] = 'neutral'
    
    # Generate signals based on moving averages
    signals.loc[(signals['MA_Fast'] > signals['MA_Slow']), 'membership_signal'] = 'long'
    signals.loc[(signals['MA_Fast'] < signals['MA_Slow']), 'membership_signal'] = 'short'
    
    return signals[['membership_signal']]
