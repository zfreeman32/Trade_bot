
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

# Membership Plans Trading Strategy
def membership_trading_signals(stock_df, low_window=20, high_window=50):
    """
    Generates trading signals based on the membership plans trading strategy.
    
    Parameters:
    stock_df (DataFrame): DataFrame containing stock price data with 'Close' column.
    low_window (int): Short moving average window size.
    high_window (int): Long moving average window size.
    
    Returns:
    DataFrame: Signals indicating 'long', 'short', or 'neutral' positions.
    """
    
    signals = pd.DataFrame(index=stock_df.index)
    signals['short_ma'] = stock_df['Close'].rolling(window=low_window).mean()
    signals['long_ma'] = stock_df['Close'].rolling(window=high_window).mean()
    
    # Initialize the signal column
    signals['signal'] = 'neutral'
    
    # Create long and short signals based on moving averages
    signals.loc[(signals['short_ma'] > signals['long_ma']) & (signals['short_ma'].shift(1) <= signals['long_ma'].shift(1)), 'signal'] = 'long'
    signals.loc[(signals['short_ma'] < signals['long_ma']) & (signals['short_ma'].shift(1) >= signals['long_ma'].shift(1)), 'signal'] = 'short'
    
    # Drop the moving average columns for clarity
    signals.drop(['short_ma', 'long_ma'], axis=1, inplace=True)
    
    return signals
