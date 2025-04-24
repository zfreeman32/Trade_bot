
import pandas as pd
import numpy as np
from ta import momentum, trend

# Example: Simple Moving Average Crossover Strategy
def sma_crossover_signals(stock_df, short_window=50, long_window=200):
    signals = pd.DataFrame(index=stock_df.index)
    signals['SMA_Short'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['SMA_Long'] = stock_df['Close'].rolling(window=long_window).mean()
    
    signals['sma_signal'] = 'neutral'
    signals.loc[(signals['SMA_Short'] > signals['SMA_Long']), 'sma_signal'] = 'long'
    signals.loc[(signals['SMA_Short'] < signals['SMA_Long']), 'sma_signal'] = 'short'
    
    return signals[['sma_signal']]
