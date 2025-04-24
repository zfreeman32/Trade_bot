
import pandas as pd
import numpy as np
from ta import trend

# Moving Average Crossover Strategy
def moving_average_crossover_signals(stock_df, short_window=50, long_window=200):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Short_MA'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['Long_MA'] = stock_df['Close'].rolling(window=long_window).mean()
    
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['Short_MA'][short_window:] > signals['Long_MA'][short_window:], 1, 0)   
    signals['positions'] = signals['signal'].diff()
    
    # Generate trading signals
    signals['trade_signal'] = 'neutral'
    signals.loc[signals['positions'] == 1, 'trade_signal'] = 'long'   # Buy signal
    signals.loc[signals['positions'] == -1, 'trade_signal'] = 'short'  # Sell signal
    
    return signals[['trade_signal']]
