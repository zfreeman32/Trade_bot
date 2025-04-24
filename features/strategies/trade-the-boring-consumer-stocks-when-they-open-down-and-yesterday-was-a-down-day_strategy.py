
import pandas as pd
import numpy as np
from ta import momentum, trend

# Moving Average Crossover Strategy
def moving_average_crossover_signals(stock_df, short_window=50, long_window=200):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate short and long moving averages
    signals['short_mavg'] = stock_df['Close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = stock_df['Close'].rolling(window=long_window, min_periods=1).mean()
    
    # Create signals
    signals['signal'] = 0  # Default to neutral
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0)
    
    # Identify when to buy or sell
    signals['positions'] = signals['signal'].diff()
    signals['trade_signal'] = 'neutral'
    signals.loc[signals['positions'] == 1, 'trade_signal'] = 'long'   # Buy signal
    signals.loc[signals['positions'] == -1, 'trade_signal'] = 'short'  # Sell signal
    
    # Drop extra columns
    signals.drop(['short_mavg', 'long_mavg', 'signal', 'positions'], axis=1, inplace=True)
    
    return signals
