
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

# Moving Average Crossover Strategy
def moving_average_crossover_signals(stock_df, short_window=50, long_window=200):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Short_MA'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['Long_MA'] = stock_df['Close'].rolling(window=long_window).mean()
    
    signals['Signal'] = 0
    signals['Signal'][short_window:] = np.where(signals['Short_MA'][short_window:] > signals['Long_MA'][short_window:], 1, 0)
    signals['Position'] = signals['Signal'].diff()
    
    signals['trade_signal'] = 'neutral'
    signals.loc[signals['Position'] == 1, 'trade_signal'] = 'long'
    signals.loc[signals['Position'] == -1, 'trade_signal'] = 'short'

    signals.drop(['Short_MA', 'Long_MA', 'Signal', 'Position'], axis=1, inplace=True)
    return signals
