
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility

# Moving Average Crossover Strategy
def moving_average_crossover_signals(stock_df, short_window=50, long_window=200):
    signals = pd.DataFrame(index=stock_df.index)
    signals['short_m avg'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['long_m avg'] = stock_df['Close'].rolling(window=long_window).mean()
    
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['short_m avg'][short_window:] > signals['long_m avg'][short_window:], 1, 0)
    
    # Generate trading orders
    signals['positions'] = signals['signal'].diff()
    signals['trade_signal'] = 'neutral'
    
    signals.loc[signals['positions'] == 1, 'trade_signal'] = 'long'
    signals.loc[signals['positions'] == -1, 'trade_signal'] = 'short'
    
    return signals[['trade_signal']]
