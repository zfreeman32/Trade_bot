
import pandas as pd
import numpy as np
from ta import trend

# Moving Average Crossover Strategy
def moving_average_crossover_signals(stock_df, short_window=50, long_window=200):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate short and long moving averages
    signals['short_mavg'] = stock_df['Close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = stock_df['Close'].rolling(window=long_window, min_periods=1).mean()
    
    # Initialize signals
    signals['signal'] = 0
    signals.loc[signals['short_mavg'] > signals['long_mavg'], 'signal'] = 1  # Buy Signal
    signals.loc[signals['short_mavg'] < signals['long_mavg'], 'signal'] = -1  # Sell Signal
    
    # Create a new column for trading signals
    signals['trade_signal'] = 'neutral'
    signals.loc[signals['signal'] == 1, 'trade_signal'] = 'long'
    signals.loc[signals['signal'] == -1, 'trade_signal'] = 'short'
    
    # Drop moving averages and original signal column
    signals.drop(['short_mavg', 'long_mavg', 'signal'], axis=1, inplace=True)
    
    return signals
