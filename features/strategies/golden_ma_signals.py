import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

def golden_ma_signals(stock_df, short_period=50, long_period=200):
    """
    Computes SMA crossover trend and trading signals.

    Returns:
    A DataFrame with 'ma_direction' and 'ma_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute short and long SMAs
    short_sma = trend.SMAIndicator(stock_df['Close'], short_period).sma_indicator()
    long_sma = trend.SMAIndicator(stock_df['Close'], long_period).sma_indicator()

    # Generate crossover signals
    signals['ma_buy_signal'] = 0
    signals['ma_sell_signal'] = 0
    signals.loc[(short_sma > long_sma) & (short_sma.shift(1) <= long_sma.shift(1)), 'ma_buy_signal'] = 1
    signals.loc[(short_sma <= long_sma) & (short_sma.shift(1) > long_sma.shift(1)), 'ma_sell_signal'] = 1

    return signals

#%% 
# 5-8-13 SMA Strategy
