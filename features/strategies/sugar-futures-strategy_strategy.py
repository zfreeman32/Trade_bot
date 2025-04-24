
import pandas as pd
import numpy as np
from ta import trend

# Moving Average Crossover Strategy
def moving_average_crossover_signals(stock_df, short_window=50, long_window=200):
    signals = pd.DataFrame(index=stock_df.index)
    signals['short_mavg'] = stock_df['Close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = stock_df['Close'].rolling(window=long_window, min_periods=1).mean()
    
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0)   
    signals['positions'] = signals['signal'].diff()
    
    signals['trading_signal'] = 'neutral'
    signals.loc[signals['positions'] == 1, 'trading_signal'] = 'long'
    signals.loc[signals['positions'] == -1, 'trading_signal'] = 'short'

    return signals[['trading_signal']]
