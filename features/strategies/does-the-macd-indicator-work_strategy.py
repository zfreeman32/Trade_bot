
import pandas as pd
import numpy as np
from ta import momentum, trend

# Simple Moving Average Crossover Strategy
def sma_crossover_signals(stock_df, short_window=50, long_window=200):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Short_SMA'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['Long_SMA'] = stock_df['Close'].rolling(window=long_window).mean()
    
    signals['sma_signal'] = 'neutral'
    signals.loc[(signals['Short_SMA'] > signals['Long_SMA']), 'sma_signal'] = 'long'
    signals.loc[(signals['Short_SMA'] < signals['Long_SMA']), 'sma_signal'] = 'short'
    
    return signals[['sma_signal']]
