import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# GapDownSE Strategy
def gap_down_se_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['high'] = stock_df["High"]
    signals['prev_low'] = stock_df["Low"].shift(1)
    signals['gap_down_se_sell_signals'] = np.where(signals['high'] < signals['prev_low'], 1, 0)
    signals.drop(['high', 'prev_low'], axis=1, inplace=True)
    return signals
