
import pandas as pd
import numpy as np

# Leveraged ETF Trading Strategy
def leveraged_etf_signals(stock_df, short_window=5, long_window=20):
    """
    Generates trading signals for a leveraged ETF trading strategy.
    
    Parameters:
    stock_df (pd.DataFrame): DataFrame containing 'Close' price for the ETF.
    short_window (int): The short window period for moving average.
    long_window (int): The long window period for moving average.

    Returns:
    pd.DataFrame: DataFrame with trading signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate short and long moving averages
    signals['short_mavg'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = stock_df['Close'].rolling(window=long_window).mean()
    
    # Generate signals
    signals['signal'] = 0  # Default to neutral
    signals['signal'][short_window:] = np.where(
        signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0
    )
    signals['position'] = signals['signal'].diff()

    # Assign trading signals based on position changes
    signals['trading_signal'] = 'neutral'
    signals.loc[signals['position'] == 1, 'trading_signal'] = 'long'
    signals.loc[signals['position'] == -1, 'trading_signal'] = 'short'
    
    return signals[['trading_signal']]
