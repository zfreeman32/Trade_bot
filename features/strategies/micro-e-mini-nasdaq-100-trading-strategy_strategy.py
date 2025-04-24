
import pandas as pd
import numpy as np
from ta import volatility

# Micro E-mini Nasdaq 100 Trading Strategy
def micro_e_mini_nasdaq_signals(stock_df, lookback_period=20, atr_multiplier=1.5):
    """
    Generate trading signals for the Micro E-mini Nasdaq 100 trading strategy.

    Parameters:
    stock_df (pd.DataFrame): DataFrame containing historical stock data with 'Close' prices.
    lookback_period (int): Number of periods to calculate average true range (ATR).
    atr_multiplier (float): Multiplier for ATR to set entry and exit levels.

    Returns:
    pd.DataFrame: DataFrame with trading signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate Average True Range (ATR)
    stock_df['high_low'] = stock_df['High'] - stock_df['Low']
    stock_df['high_close'] = np.abs(stock_df['High'] - stock_df['Close'].shift())
    stock_df['low_close'] = np.abs(stock_df['Low'] - stock_df['Close'].shift())
    stock_df['tr'] = stock_df[['high_low', 'high_close', 'low_close']].max(axis=1)
    stock_df['atr'] = stock_df['tr'].rolling(window=lookback_period).mean()
    
    # Calculate entry and exit levels
    stock_df['long_entry'] = stock_df['Close'] + (stock_df['atr'] * atr_multiplier)
    stock_df['short_entry'] = stock_df['Close'] - (stock_df['atr'] * atr_multiplier)

    # Generate signals
    signals['position'] = 'neutral'
    signals.loc[stock_df['Close'] > stock_df['long_entry'].shift(1), 'position'] = 'long'
    signals.loc[stock_df['Close'] < stock_df['short_entry'].shift(1), 'position'] = 'short'

    return signals
