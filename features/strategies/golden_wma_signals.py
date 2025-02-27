import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Golden Cross WMA
def golden_wma_signals(stock_df, short_period=50, long_period=200):
    """
    Computes Weighted Moving Average (WMA) crossover signals.

    Returns:
    A DataFrame with 'wma_direction' and 'wma_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute short and long WMAs
    short_wma = trend.WMAIndicator(stock_df['Close'], short_period).wma()
    long_wma = trend.WMAIndicator(stock_df['Close'], long_period).wma()


    # Generate crossover signals
    signals['wma_buy_signal'] = 0
    signals['wma_sell_signal'] = 0
    signals.loc[
        (short_wma > long_wma) & 
        (short_wma.shift(1) <= long_wma.shift(1)), 'wma_buy_signal'
    ] = 1
    
    signals.loc[
        (short_wma <= long_wma) & 
        (short_wma.shift(1) > long_wma.shift(1)), 'wma_sell_signal'
    ] = 1

    return signals