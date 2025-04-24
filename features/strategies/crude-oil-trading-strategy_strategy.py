
import pandas as pd
import numpy as np
from ta import momentum

# Trading Strategy Function
def tsi_signal_strategy(stock_df, window_slow=25, window_fast=13):
    signals = pd.DataFrame(index=stock_df.index)
    tsi = momentum.TSIIndicator(stock_df['Close'], window_slow, window_fast)
    signals['TSI'] = tsi.tsi()
    signals['tsi_signal'] = 'neutral'
    signals.loc[(signals['TSI'] > 0) & (signals['TSI'].shift(1) <= 0), 'tsi_signal'] = 'long'
    signals.loc[(signals['TSI'] < 0) & (signals['TSI'].shift(1) >= 0), 'tsi_signal'] = 'short'
    signals.drop(['TSI'], axis=1, inplace=True)
    return signals
