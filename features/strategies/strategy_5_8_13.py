import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

def strategy_5_8_13(stock_df):
    """
    Computes 5-8-13 SMA crossover trend direction.

    Returns:
    A DataFrame with '5_8_13_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute short SMAs
    sma5 = trend.SMAIndicator(stock_df['Close'], 5).sma_indicator()
    sma8 = trend.SMAIndicator(stock_df['Close'], 8).sma_indicator()
    sma13 = trend.SMAIndicator(stock_df['Close'], 13).sma_indicator()

    # Determine market direction
    signals['5_8_13_buy_signal'] = 0
    signals['5_8_13_sell_signal'] = 0
    signals.loc[(sma5 > sma8) & (sma8 > sma13), '5_8_13_buy_signal'] = 1
    signals.loc[(sma5 < sma8) & (sma8 < sma13), '5_8_13_sell_signal'] = 1

    return signals

#%% 
# 5-8-13 WMA Strategy
