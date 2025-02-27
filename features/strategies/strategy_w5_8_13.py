import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

def strategy_w5_8_13(stock_df):
    """
    Computes 5-8-13 WMA crossover trend direction.

    Returns:
    A DataFrame with 'w5_8_13_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute weighted moving averages
    wma5 = trend.WMAIndicator(stock_df['Close'], 5).wma()
    wma8 = trend.WMAIndicator(stock_df['Close'], 8).wma()
    wma13 = trend.WMAIndicator(stock_df['Close'], 13).wma()

    # Determine market direction
    signals['w5_8_13_buy_signal'] = 0
    signals['w5_8_13_sell_signal'] = 0
    signals.loc[(wma5 > wma8) & (wma8 > wma13), 'w5_8_13_buy_signal'] = 1
    signals.loc[(wma5 < wma8) & (wma8 < wma13), 'w5_8_13_sell_signal'] = 1

    return signals

#%% 
# Keltner Channel Strategy
