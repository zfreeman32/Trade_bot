import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from insidebarse.py
# InsideBarSE Strategy
def inside_bar_se_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)

    # Identify the inside bars and close price is lower than open
    signals['inside_bar'] = np.where((stock_df['High'] < stock_df['High'].shift(1)) &
                                     (stock_df['Low'] > stock_df['Low'].shift(1)) &
                                     (stock_df['Close'] < stock_df['Open']), 1, 0)

    # Generate signals
    signals['inside_bar_sell_signal'] = 0
    signals.loc[(signals['inside_bar'].shift(1) == 1), 'inside_bar_sell_signal'] = 1
    signals.drop(['inside_bar'], axis=1, inplace=True)
    return signals
