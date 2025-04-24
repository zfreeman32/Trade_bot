
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

# Seasonal Strategy for Quantified Trading
def seasonal_trading_signals(stock_df, month_to_trade=5):
    """
    Generate trading signals based on a seasonal strategy that favors trading in May.
    
    Parameters:
    stock_df (pd.DataFrame): Dataframe containing stock price data with a datetime index.
    month_to_trade (int): The month in which to focus trading signals (May is 5).
    
    Returns:
    pd.DataFrame: DataFrame containing trading signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=stock_df.index)
    signals['signal'] = 'neutral'
    
    # Create a Month column to identify the trading month
    signals['month'] = signals.index.month
    
    # Generate 'long' signals for May
    signals.loc[signals['month'] == month_to_trade, 'signal'] = 'long'
    
    # Generate 'short' signals for other months
    signals.loc[signals['month'] != month_to_trade, 'signal'] = 'short'
    
    return signals[['signal']].drop('month', axis=1)
