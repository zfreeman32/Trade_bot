
import pandas as pd
import numpy as np

# Moving Average Crossover Strategy
def moving_average_crossover_signals(stock_df, short_window=50, long_window=200):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Short_MA'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['Long_MA'] = stock_df['Close'].rolling(window=long_window).mean()
    signals['Signal'] = 'neutral'
    signals.loc[(signals['Short_MA'] > signals['Long_MA']), 'Signal'] = 'long'
    signals.loc[(signals['Short_MA'] < signals['Long_MA']), 'Signal'] = 'short'
    return signals[['Signal']]
