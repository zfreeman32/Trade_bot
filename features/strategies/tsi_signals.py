import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

def tsi_signals(stock_df, window_slow=25, window_fast=13):
    """
    Computes True Strength Index (TSI) crossover signals.

    Returns:
    A DataFrame with 'tsi_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate TSI
    tsi = momentum.TSIIndicator(stock_df['Close'], window_slow, window_fast)
    signals['TSI'] = tsi.tsi()

    # Generate signals based on TSI zero-crossing
    signals['tsi_buy_signal'] = 0
    signals['tsi_sell_signal'] = 0
    signals.loc[(signals['TSI'] > 0) & (signals['TSI'].shift(1) <= 0), 'tsi_buy_signal'] = 1
    signals.loc[(signals['TSI'] < 0) & (signals['TSI'].shift(1) >= 0), 'tsi_sell_signal'] = 1

    # Drop temporary column
    signals.drop(['TSI'], axis=1, inplace=True)

    return signals

#%% 
# Williams %R Overbought/Oversold Strategy
