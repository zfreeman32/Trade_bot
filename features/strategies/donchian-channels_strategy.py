
import pandas as pd
import numpy as np
from ta import trend

# Moving Average Crossover Strategy
def moving_average_crossover_signals(stock_df, short_window=50, long_window=200):
    signals = pd.DataFrame(index=stock_df.index)
    signals['short_mavg'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = stock_df['Close'].rolling(window=long_window).mean()
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0)
    signals['trading_signal'] = signals['signal'].diff()
    
    # Create signals
    signals['strategy_signal'] = 'neutral'
    signals.loc[signals['trading_signal'] == 1, 'strategy_signal'] = 'long'
    signals.loc[signals['trading_signal'] == -1, 'strategy_signal'] = 'short'
    
    return signals[['strategy_signal']]
