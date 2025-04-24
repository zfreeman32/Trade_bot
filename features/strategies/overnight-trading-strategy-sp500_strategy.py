
import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange

# Overnight Trading Strategy for S&P 500
def overnight_trading_signals(stock_df):
    """
    Generates trading signals for an overnight trading strategy on the S&P 500.
    
    This strategy involves entering a long position at market close if specific criteria are met, 
    and exiting at market open the next day.

    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock price data with a 'Close' column.
    
    Returns:
    pd.DataFrame: DataFrame with trading signals.
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate Average True Range (ATR) over a 25-day window
    atr = AverageTrueRange(high=stock_df['High'], low=stock_df['Low'], close=stock_df['Close'], window=25)
    stock_df['ATR'] = atr.average_true_range()

    # Calculate the last 10 days' high
    stock_df['10_day_high'] = stock_df['High'].rolling(window=10).max()

    # Calculate the last 10 days' low
    stock_df['10_day_low'] = stock_df['Low'].rolling(window=10).min()

    # Calculate IBS (Internal Bar Strength)
    stock_df['IBS'] = (stock_df['Close'] - stock_df['10_day_low']) / (stock_df['10_day_high'] - stock_df['10_day_low'])
    
    # Generate signals
    signals['signal'] = 'neutral'
    signals.loc[(stock_df['Close'] < (stock_df['10_day_high'] - stock_df['ATR'])) & 
                 (stock_df['IBS'] < 0.5), 'signal'] = 'long'
    
    # Note: Exit logic would be handled separately as per the strategy's requirement.
    
    return signals
