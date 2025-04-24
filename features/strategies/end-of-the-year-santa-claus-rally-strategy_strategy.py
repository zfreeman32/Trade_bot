
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility

# Trend Following Moving Average Crossover Strategy
def moving_average_crossover_signals(stock_df, short_window=10, long_window=30):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate moving averages
    signals['short_mavg'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = stock_df['Close'].rolling(window=long_window).mean()
    
    # Generate signals
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0)
    
    # Create a column to hold the actual signals ('long', 'short', 'neutral')
    signals['position'] = signals['signal'].diff()
    
    signals['market_signal'] = 'neutral'
    signals.loc[signals['position'] == 1, 'market_signal'] = 'long'
    signals.loc[signals['position'] == -1, 'market_signal'] = 'short'
    
    # Keep only relevant columns
    signals = signals[['market_signal']]
    
    return signals
