
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

# Moving Average Crossover Strategy
def moving_average_crossover_signals(stock_df, short_window=10, long_window=50):
    """
    Generates trading signals based on a moving average crossover strategy.

    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock price data with a 'Close' column.
    short_window (int): The window size for the short moving average.
    long_window (int): The window size for the long moving average.

    Returns:
    pd.DataFrame: DataFrame containing signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=stock_df.index)
    signals['short_mavg'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = stock_df['Close'].rolling(window=long_window).mean()
    
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(
        signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0
    )
    
    signals['positions'] = signals['signal'].diff()
    signals['trade_signal'] = 'neutral'
    signals.loc[signals['positions'] == 1, 'trade_signal'] = 'long'
    signals.loc[signals['positions'] == -1, 'trade_signal'] = 'short'
    
    return signals[['trade_signal']]
