import pandas as pd
import numpy as np

# GapDownSE Strategy
def gap_down_se_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['high'] = stock_df["High"]
    signals['prev_low'] = stock_df["Low"].shift(1)
    signals['gap_down_se_signals'] = np.where(signals['high'] < signals['prev_low'], 'short', 'neutral')
    signals.drop(['high', 'prev_low'], axis=1, inplace=True)
    return signals

